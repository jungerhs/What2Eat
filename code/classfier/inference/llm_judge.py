"""
LLM 兜底模块 (Stage 1.5 / Stage 2.5)：
  - is_cooking(query) -> {"is_cooking": bool, "confidence": 0~1, "raw": str}
  - classify_4(query) -> {"label": str, "confidence": 0~1, "raw": str}

调用统计：每次调用会累加 self.call_count / self.token_used（粗估），供 eval_offline 报告用。
"""
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classfier.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LABELS, LOG_LEVEL, CLASSFIER_ROOT,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm_judge")

GUIDELINES_PATH = CLASSFIER_ROOT / "dataset" / "label_guidelines.md"

IS_COOKING_SYSTEM = """你是中文 query 分类器。判断一条 query 是否与"烹饪 / 做菜 / 菜谱"相关。
仅输出一个 JSON：{"is_cooking": true|false, "confidence": 0~1}
不要任何其它字符。"""

IS_COOKING_USER = """# 判定规则
is_cooking = true 当 query 涉及：
  - 烹饪方法（怎么做、红烧/清蒸/水煮/糖醋 等）
  - 菜谱/食材/调味/火候/时间/比例
  - 菜系/菜品（川菜/红烧肉/汤/粥 等）
  - 厨房工具（锅/刀/烤箱/微波炉 等）
  - 营养/卡路里/食疗
  - 餐厅/点餐/外卖

is_cooking = false 当 query 是：股票/天气/学习/工作/游戏/医疗/旅游/法律/编程 等不相关主题。

# 例子
- "红烧肉要炖多久" → is_cooking=true
- "量子计算是什么" → is_cooking=false
- "推荐几道川菜" → is_cooking=true
- "王者荣耀怎么上分" → is_cooking=false
- "微波炉可以做哪些菜" → is_cooking=true
- "Python 怎么读取 JSON" → is_cooking=false
- "鸡蛋和番茄能一起吃吗" → is_cooking=true（食材搭配）

# Query
{q}

# 输出 JSON
"""

CLASSIFY_4_SYSTEM = """你是中文 query 分类质检员。把 query 分到 4 类之一：general / detail / multi-hop / recommend。
仅输出一个 JSON：{"label": "general|detail|multi-hop|recommend", "confidence": 0~1}
不要任何其它字符。"""

CLASSIFY_4_USER = """# 类别定义
- general: 单实体 + 宽泛动词（怎么做/是什么/介绍）
- detail: 数字疑问（多少/多久/比例/火候/温度/第几步）
- multi-hop: 多实体/过滤/比较（X 和 Y/适合 X 的/区别）
- recommend: 推荐意图（推荐/介绍几道/有什么/来点）

# Query
{q}

# 输出 JSON
"""


def _strip_codeblock(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    return m.group(1).strip() if m else text


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出 (含噪声/think/解释) 提取 JSON 对象。
    适用于小模型 (gemma4:e4b) 输出可能含中文解释/换行/重复 JSON 等情况。"""
    if not text:
        return None
    cleaned = _strip_codeblock(text)
    # 1) 直接解析
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # 2) 提取首个 {...} 块 (greedy 截取到最后一个 })
    m = re.search(r"\{[\s\S]+\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 3) 修复常见错误: 单引号、True/False 大小写、尾部逗号
    fixed = cleaned
    fixed = re.sub(r"'\s*([\w\u4e00-\u9fa5]+)\s*'\s*:", r'"\1":', fixed)
    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return json.loads(fixed)
    except Exception:
        pass
    # 4) 提取最后一个 {...} (有时 LLM 在前面输出多个示例)
    matches = re.findall(r"\{[^{}]*\}", cleaned)
    for cand in reversed(matches):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


@dataclass
class LLMJudge:
    """带调用统计的 LLM 兜底器"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    call_count: int = 0
    token_used: int = 0  # 粗估：每次 request 字符数 + response 字符数
    fail_count: int = 0

    def __post_init__(self):
        self.api_key = self.api_key or LLM_API_KEY
        self.base_url = self.base_url or LLM_BASE_URL
        self.model = self.model or LLM_MODEL
        if not self.api_key:
            raise ValueError("未配置 API_KEY / MOONSHOT_API_KEY")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call_json(self, system: str, user: str, max_tokens: int = 200,
                   max_retry: int = 5) -> Optional[Dict[str, Any]]:
        # Ollama 小模型 (gemma4 等) 默认开启 thinking，
        # 在 system prompt 末尾加强制指令以关闭思考 (兜底)
        if self.base_url and "localhost:11434" in self.base_url:
            system = system + "\n\n【重要】直接输出 JSON，禁止任何 think / 思考 / 解释 / 推理过程。"
        self.call_count += 1
        last_err = None
        for attempt in range(1, max_retry + 1):
            try:
                # Ollama OpenAI 兼容层支持 extra_body={"think": False} 关思考
                create_kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                if self.base_url and "localhost:11434" in self.base_url:
                    create_kwargs["extra_body"] = {
                        "think": False,
                        "keep_alive": "30m",  # 让 Ollama 模型常驻 30 分钟，避免每次重载 9.6GB
                    }
                resp = self.client.chat.completions.create(**create_kwargs)
                content = (resp.choices[0].message.content or "").strip()
                self.token_used += len(user) + len(content)  # 粗估
                obj = _extract_json(content)
                if obj is not None:
                    return obj
                last_err = f"json_parse_fail content[:80]={content[:80]!r}"
                logger.warning(f"LLM 兜底 JSON 解析失败 attempt={attempt} {last_err}")
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:120]}"
                logger.warning(f"LLM 兜底失败 attempt={attempt} err={last_err}")
            time.sleep(0.5 * attempt)
        self.fail_count += 1
        if last_err:
            logger.error(f"LLM 兜底彻底失败 (5 次): {last_err}")
        return None

    def is_cooking(self, query: str) -> Dict[str, Any]:
        obj = self._call_json(IS_COOKING_SYSTEM, IS_COOKING_USER.format(q=query))
        if obj is None:
            return {"is_cooking": True, "confidence": 0.0,
                    "raw": "fail", "stage": "llm_fallback_fail"}
        is_c = bool(obj.get("is_cooking", True))
        conf = float(obj.get("confidence", 0.5))
        return {"is_cooking": is_c, "confidence": conf,
                "raw": json.dumps(obj, ensure_ascii=False), "stage": "llm_judge"}

    def classify_4(self, query: str) -> Dict[str, Any]:
        obj = self._call_json(CLASSIFY_4_SYSTEM, CLASSIFY_4_USER.format(q=query),
                              max_tokens=100)
        if obj is None:
            return {"label": None, "confidence": 0.0,
                    "raw": "fail", "stage": "llm_fallback_fail"}
        label = obj.get("label")
        if label not in LABELS:
            return {"label": None, "confidence": 0.0,
                    "raw": json.dumps(obj, ensure_ascii=False),
                    "stage": "llm_invalid_label"}
        conf = float(obj.get("confidence", 0.5))
        return {"label": label, "confidence": conf,
                "raw": json.dumps(obj, ensure_ascii=False), "stage": "llm_classify"}


# ===================== CLI =====================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--task", choices=["is_cooking", "classify_4"], default="is_cooking")
    args = ap.parse_args()
    if not args.text:
        ap.print_help()
        return
    j = LLMJudge()
    if args.task == "is_cooking":
        print(json.dumps(j.is_cooking(args.text), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(j.classify_4(args.text), ensure_ascii=False, indent=2))
    print(f"# stats: call={j.call_count} fail={j.fail_count} ~token={j.token_used}")


if __name__ == "__main__":
    main()
