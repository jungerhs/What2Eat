"""
多轮对话意图识别模块

用途：
    在多轮对话第 2 轮及以后，替代原有的 LR → BERT → LLM 管线。
    使用小参数 LLM（Ollama 本地部署）快速识别意图 + 输出置信度，
    置信度不足时由大参数 LLM（API 调用）兜底。

流程：
    1. 小 LLM (Ollama) → {"label": ..., "confidence": 0~1}
    2. 若 confidence < threshold → 大 LLM (API) 兜底
    3. 若都失败 → unknown

输出格式与 QueryClassifier.predict() 兼容，可直接替换。
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# 意图标签（与 classfier/inference/predictor.py 一致）
LABELS = ("general", "detail", "multi-hop", "recommend")

# ===================== Prompt 模板 =====================

SMALL_LLM_SYSTEM = """你是一个中文烹饪意图分类器。根据用户当前问题和历史提问判断意图类别。

类别定义：
- general: 一般查询 —— "怎么做红烧肉"、"介绍番茄炒蛋"
- detail: 细节/数值 —— "放多少盐"、"炖多久"、"第几步放酱油"
- multi-hop: 多实体/图遍历 —— "鸡肉配什么蔬菜"
- recommend: 推荐 —— "推荐几道菜"、"有什么好吃的"、"来点下饭菜"

【要求】
- 只输出 JSON：{"label": "<类别>", "confidence": <0.0~1.0>}
- confidence 表示你对分类的把握程度
- 禁止任何 think / 思考 / 解释过程
- 如果与烹饪完全无关，输出 {"label": "unknown", "confidence": 1.0}

【示例】
用户问题：怎么做红烧肉
输出： {"label": "general", "confidence": 0.8}

用户问题：红烧肉炖多久
输出： {"label": "detail", "confidence": 0.8}

用户问题：推荐几道下饭菜
输出： {"label": "recommend", "confidence": 0.8}

用户问题：鸡肉搭配什么蔬菜
输出： {"label": "multi-hop", "confidence": 0.8}

用户问题：今天天气怎么样
输出： {"label": "unknown", "confidence": 1.0}

"""


SMALL_LLM_USER = """
用户当前问题：{query}


输出 JSON："""

LARGE_LLM_SYSTEM = """你是一个中文烹饪意图分类器。判断用户问题的意图类别。
**严格输出 JSON object**（不得包含 markdown 代码块、不得包含任何额外说明文字）。

类别定义：
- general: 一般查询 —— "怎么做红烧肉"、"介绍番茄炒蛋"
- detail: 细节/数值 —— "放多少盐"、"炖多久"、"第几步放酱油"
- multi-hop: 多实体/图遍历 —— "鸡肉配什么蔬菜"
- recommend: 推荐 —— "推荐几道菜"、"有什么好吃的"、"来点下饭菜"

如果问题与烹饪完全无关，输出：
{"label": "unknown", "confidence": 1.0}

【示例】
用户输入：怎么做红烧肉
输出： {"label": "general", "confidence": 0.8}

用户输入：红烧肉炖多久
输出： {"label": "detail", "confidence": 0.8}

用户输入：推荐几道下饭菜
输出： {"label": "recommend", "confidence": 0.8}

用户输入：鸡肉搭配什么蔬菜
输出： {"label": "multi-hop", "confidence": 0.8}

用户输入：今天天气怎么样
输出： {"label": "unknown", "confidence": 1.0}

只输出 JSON，不要任何其它内容。"""


# ===================== JSON 提取工具 =====================

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中提取 JSON（兼容 markdown 围栏、思考噪音等）。"""
    if not text:
        return None
    text = text.strip()
    # 剥 markdown 代码围栏
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 提取首个 {...} 块
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # 修复常见错误：单引号、尾逗号
    fixed = text
    fixed = re.sub(r"'\s*([\w\u4e00-\u9fa5]+)\s*'\s*:", r'"\1":', fixed)
    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


# ===================== 主类 =====================

class MultiTurnIntentClassifier:
    """
    多轮对话意图识别器。

    第 1 轮仍用原有 LR→BERT→LLM 管线；
    第 2 轮起使用本类：小 LLM (Ollama) → 大 LLM (API) 兜底。

    构造参数均取自环境变量（带合理默认值），无需手动传参。
    """

    def __init__(
        self,
        small_llm_base_url: Optional[str] = None,
        small_llm_model: Optional[str] = None,
        small_llm_api_key: Optional[str] = None,
        large_llm_base_url: Optional[str] = None,
        large_llm_model: Optional[str] = None,
        large_llm_api_key: Optional[str] = None,
        confidence_threshold: float = 0.65,
        disable_large_llm: bool = False,
    ):
        """
        Args:
            disable_large_llm: True 时禁用大模型兜底，用于单独测试小 LLM。
        """
        # ── 小 LLM（Ollama 本地部署） ──
        self.small_llm_base_url = (
            small_llm_base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        )
        self.small_llm_model = (
            small_llm_model or os.getenv("OLLAMA_MODEL", "gemma4:e4b")
        )
        self.small_llm_api_key = (
            small_llm_api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        )
        self.small_llm = OpenAI(
            api_key=self.small_llm_api_key,
            base_url=self.small_llm_base_url,
        )

        # ── 大 LLM（API 调用，兜底用） ──
        self.large_llm_base_url = (
            large_llm_base_url
            or os.getenv("MOONSHOT_BASE_URL")
            or os.getenv("BASE_URL")
            or "https://api.moonshot.cn/v1"
        )
        self.large_llm_model = (
            large_llm_model
            or os.getenv("MOONSHOT_MODEL")
            or os.getenv("LLM_MODEL")
            or "kimi-k2-0711-preview"
        )
        large_key = (
            large_llm_api_key
            or os.getenv("MOONSHOT_API_KEY")
            or os.getenv("API_KEY")
        )
        if disable_large_llm:
            self.large_llm = None
            logger.info("大 LLM 兜底已禁用（disable_large_llm=True）")
        elif large_key:
            self.large_llm = OpenAI(
                api_key=large_key,
                base_url=self.large_llm_base_url,
            )
        else:
            logger.warning("大 LLM API Key 未配置，大模型兜底不可用")
            self.large_llm = None

        self.confidence_threshold = confidence_threshold

        logger.info(
            f"MultiTurnIntentClassifier: 小LLM={self.small_llm_model} "
            f"({self.small_llm_base_url}), "
            f"大LLM={self.large_llm_model} "
            f"({'已配置' if self.large_llm else '未配置'}), "
            f"置信度阈值={confidence_threshold}"
        )

    # ── 公开接口 ──────────────────────────────────────────

    def predict(self, text: str) -> Dict[str, Any]:
        """
        执行多轮意图识别。

        Args:
            text: 用户问题（已改写后的 self-contained 查询）。

        Returns:
            与 QueryClassifier.predict() 兼容的 dict:
            {
                "text": str,           # 原始问题
                "label": str,          # general/detail/multi-hop/recommend/unknown
                "label_id": int,       # -1 表示 unknown
                "confidence": float,
                "probs": dict,         # 各标签概率（此处为占位）
                "top2": list,          # 前 2 候选
                "is_unknown": bool,
                "stage": str,          # "small_llm" / "large_llm_fallback" / "both_fail"
            }
        """
        # Step 1: 小 LLM
        result = self._call_small_llm(text)
        if result is not None:
            label = result.get("label")
            conf = result.get("confidence", 0.0)
            if label in LABELS and conf >= self.confidence_threshold:
                return self._format(text, label, conf, "small_llm")
            # unknown 且高信度 → 直接接受
            if label == "unknown" and conf >= self.confidence_threshold:
                return self._format(text, "unknown", conf, "small_llm")

        # Step 2: 大 LLM 兜底
        if self.large_llm is not None:
            fallback = self._call_large_llm(text)
            if fallback is not None:
                label = fallback.get("label")
                conf = fallback.get("confidence", 0.9)
                if label in LABELS or label == "unknown":
                    return self._format(
                        text, label, conf, "large_llm_fallback"
                    )

        # Step 3: 都失败 → 保守返回 unknown
        logger.warning(f"多轮意图识别全部失败，返回 unknown: text={text[:60]}")
        return self._format(text, "unknown", 0.0, "both_fail")

    # ── 内部调用 ──────────────────────────────────────────

    def _call_small_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """调用 Ollama 小 LLM。返回 {"label": str, "confidence": float} 或 None。"""
        try:
            kwargs: Dict[str, Any] = dict(
                model=self.small_llm_model,
                messages=[
                    {"role": "system", "content": SMALL_LLM_SYSTEM},
                    {"role": "user", "content": SMALL_LLM_USER.format(query=text)},
                ],
                temperature=0.0
               
            )
            # Ollama 兼容层支持 extra_body 关闭 thinking
            if "localhost" in self.small_llm_base_url:
                kwargs["extra_body"] = {
                    "think": False,
                    "keep_alive": "30m",
                }
            resp = self.small_llm.chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            obj = _extract_json(content)
            if obj is not None:
                label = obj.get("label")
                conf = float(obj.get("confidence", 0.0))
                if label.lower() in LABELS or label.lower() == "unknown":
                    return {"label": label, "confidence": conf}
            logger.warning(
                f"小 LLM JSON 解析失败: content={content[:120]!r}"
            )
        except Exception as e:
            logger.warning(f"小 LLM 调用失败: {e}")
        return None

    def _call_large_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """调用大 LLM（API）兜底。返回 {"label": str, "confidence": float} 或 None。

        使用 response_format={"type": "json_object"} 强制结构化输出，避免
        模型在 content 里加 ```json ``` 代码块包裹或额外解释文本。
        """
        try:
            resp = self.large_llm.chat.completions.create(
                model=self.large_llm_model,
                messages=[
                    {"role": "system", "content": LARGE_LLM_SYSTEM},
                    {"role": "user", "content": f"用户问题：{text}\n\n输出 JSON："},
                ],
                temperature=0.0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            obj = _extract_json(content)
            if obj is not None:
                label = obj.get("label")
                conf = float(obj.get("confidence", 0.9))
                if label in LABELS or label == "unknown":
                    return {"label": label, "confidence": conf}
            logger.warning(
                f"大 LLM JSON 解析失败: content={content[:120]!r}"
            )
        except Exception as e:
            logger.warning(f"大 LLM 调用失败: {e}")
        return None

    # ── 格式化输出（与 QueryClassifier 兼容） ─────────────

    def _format(
        self, text: str, label: str, confidence: float, stage: str
    ) -> Dict[str, Any]:
        is_unknown = label == "unknown"
        return {
            "text": text,
            "label": label,
            "label_id": -1 if is_unknown else LABELS.index(label),
            "confidence": confidence,
            "probs": {l: 0.0 for l in LABELS},
            "top2": [{"label": label, "prob": confidence}],
            "is_unknown": is_unknown,
            "stage": stage,
        }
