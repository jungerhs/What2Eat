"""
使用 OpenAI 兼容 SDK 调用 Kimi (Moonshot) 批量合成 4 类 query。

输入：dataset/label_guidelines.md（标签契约）
输出：data/synthetic/llm_<class>.jsonl  (4 个文件)
字段：{text, label, source}

使用：
    export API_KEY=sk-...
    python -m classfier.dataset.generate_by_llm           # 全量
    python -m classfier.dataset.generate_by_llm --n 500  # 每类 500 条（测试用）
"""
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI

# 允许从 classfier/ 直接 import config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_PER_CLASS, LLM_BATCH_SIZE, LLM_CONCURRENCY, LLM_TEMPERATURE,
    SYNTHETIC_DIR, LOG_LEVEL, LABELS,
    CLASSFIER_ROOT,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_by_llm")

GUIDELINES_PATH = CLASSFIER_ROOT / "dataset" / "label_guidelines.md"

# ===================== Few-shot 样例（10 条/类） =====================
FEWSHOT: Dict[str, List[Dict[str, str]]] = {
    "general": [
        {"text": "红烧肉怎么做", "label": "general"},
        {"text": "介绍一下麻婆豆腐", "label": "general"},
        {"text": "川菜是什么菜系", "label": "general"},
        {"text": "糖醋口味是怎么回事", "label": "general"},
        {"text": "可乐鸡翅是哪里的名菜", "label": "general"},
        {"text": "水煮牛肉怎么做的", "label": "general"},
        {"text": "粤菜有什么特点", "label": "general"},
        {"text": "东坡肉的历史是什么", "label": "general"},
        {"text": "意大利面的种类有哪些", "label": "general"},
        {"text": "杨枝甘露是甜品还是饮料", "label": "general"},
    ],
    "detail": [
        {"text": "红烧肉要炖多久", "label": "detail"},
        {"text": "冰糖和酱油的比例是多少", "label": "detail"},
        {"text": "第 3 步先放姜还是先放蒜", "label": "detail"},
        {"text": "麻婆豆腐要小火还是大火", "label": "detail"},
        {"text": "没有料酒可以用什么替代", "label": "detail"},
        {"text": "戚风蛋糕要烤几分钟", "label": "detail"},
        {"text": "水煮牛肉要多少克花椒", "label": "detail"},
        {"text": "泡打粉什么时候加", "label": "detail"},
        {"text": "红烧肉放多少盐合适", "label": "detail"},
        {"text": "糖醋汁加热多久能变稠", "label": "detail"},
    ],
    "multi-hop": [
        {"text": "用鸡蛋和番茄可以做什么菜", "label": "multi-hop"},
        {"text": "川菜中适合冬天的汤", "label": "multi-hop"},
        {"text": "红烧肉和糖醋排骨的区别", "label": "multi-hop"},
        {"text": "减脂期吃什么水产", "label": "multi-hop"},
        {"text": "不含花生的川菜有哪些", "label": "multi-hop"},
        {"text": "早餐适合的汤类有哪些", "label": "multi-hop"},
        {"text": "用牛奶和鸡蛋能做什么甜品", "label": "multi-hop"},
        {"text": "新手能做的早餐有哪些", "label": "multi-hop"},
        {"text": "鸡胸肉和西兰花能一起做什么", "label": "multi-hop"},
        {"text": "夏天吃的凉拌菜怎么做", "label": "multi-hop"},
    ],
    "recommend": [
        {"text": "推荐一道川菜", "label": "recommend"},
        {"text": "有什么简单易做的早餐", "label": "recommend"},
        {"text": "介绍几道下饭菜", "label": "recommend"},
        {"text": "夏天适合吃什么凉菜", "label": "recommend"},
        {"text": "来个简单的家常菜", "label": "recommend"},
        {"text": "有哪些适合新手的菜", "label": "recommend"},
        {"text": "推荐几道减脂餐", "label": "recommend"},
        {"text": "求几道下饭的荤菜", "label": "recommend"},
        {"text": "夜宵有什么推荐", "label": "recommend"},
        {"text": "有什么适合带饭的菜", "label": "recommend"},
    ],
}

# ===================== 提示词 =====================

SYSTEM_PROMPT = """你是烹饪领域的 query 数据合成专家。你的任务是根据"标签规范"批量生成中文 query。

严格规则：
1. 严格只输出 JSON 数组，不要任何解释、Markdown 代码块、前后缀。
2. 每条 query 长度 5~40 字，贴近真实用户口吻（口语化）。
3. 不要重复表达方式（避免句式单一）。
4. 所有 query 必须属于指定的 label，不允许混入其他类别。
5. 烹饪场景必须真实可信，不要凭空捏造不存在的菜名/菜系。
"""


def build_user_prompt(label: str, n: int, guidelines_text: str) -> str:
    fewshot = FEWSHOT[label]
    fs_text = "\n".join(
        f'  - {{"text": "{x["text"]}", "label": "{x["label"]}"}}'
        for x in fewshot
    )
    return f"""# 标签契约（节选）
{guidelines_text}

# 任务
请为 **{label}** 类别合成 **{n} 条** 中文 query。

# Few-shot 示例（仅参考风格，不要照抄）
{fs_text}

# 输出格式（严格遵守）
仅输出 JSON 数组，每个元素格式：
{{"text": "...", "label": "{label}"}}

只输出 JSON，不要其它任何字符。"""


# ===================== LLM 调用 =====================

def get_client() -> OpenAI:
    if not LLM_API_KEY:
        raise ValueError("未配置 API_KEY / MOONSHOT_API_KEY 环境变量")
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _strip_codeblock(text: str) -> str:
    """去掉 ```json ... ``` 包装"""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_array(text: str) -> Optional[List[Dict]]:
    """尝试解析 LLM 输出为 JSON 数组；失败返回 None"""
    text = _strip_codeblock(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 退化：尝试找首个 [ 到末个 ]
        s = text.find("[")
        e = text.rfind("]")
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(data, list):
        return None
    return data


def call_one_batch(client: OpenAI, label: str, n: int,
                   guidelines_text: str, max_retry: int = 3) -> List[Dict]:
    """单次 LLM 调用，返回 (含 {text,label,source} 的) 列表"""
    prompt = build_user_prompt(label, n, guidelines_text)
    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=min(4096, n * 60),
            )
            content = resp.choices[0].message.content or ""
            data = _parse_json_array(content)
            if data is None:
                logger.warning(f"[{label}] 第 {attempt} 次 JSON 解析失败: {content[:200]!r}")
                continue
            out = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                text = (item.get("text") or "").strip()
                if not (5 <= len(text) <= 60):
                    continue
                if item.get("label") not in LABELS:
                    item["label"] = label
                out.append({"text": text,
                            "label": item.get("label", label),
                            "source": "llm"})
            return out
        except Exception as e:
            logger.warning(f"[{label}] 第 {attempt} 次调用异常: {e}")
            time.sleep(2 ** attempt)
    return []


def gen_for_label(client: OpenAI, label: str, target: int,
                  guidelines_text: str) -> List[Dict]:
    """并发跑多批直到达到 target"""
    collected: List[Dict] = []
    seen: set = set()
    rounds = 0
    while len(collected) < target and rounds < 30:
        rounds += 1
        need = min(LLM_BATCH_SIZE, target - len(collected))
        # 并发 4~8 批
        batch_jobs = max(1, min(LLM_CONCURRENCY, (target - len(collected) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE))
        with ThreadPoolExecutor(max_workers=batch_jobs) as ex:
            futs = [ex.submit(call_one_batch, client, label, LLM_BATCH_SIZE, guidelines_text)
                    for _ in range(batch_jobs)]
            for fut in as_completed(futs):
                items = fut.result()
                for it in items:
                    t = it["text"]
                    if t in seen:
                        continue
                    seen.add(t)
                    collected.append(it)
                    if len(collected) >= target:
                        break
        logger.info(f"[{label}] 第 {rounds} 轮，累计 {len(collected)}/{target}")
    return collected[:target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=LLM_PER_CLASS,
                        help=f"每类目标条数，默认 {LLM_PER_CLASS}")
    parser.add_argument("--label", type=str, choices=LABELS + ["all"], default="all",
                        help="只跑某一类")
    args = parser.parse_args()

    guidelines_text = GUIDELINES_PATH.read_text(encoding="utf-8")
    # 截断 contract，避免提示词过长
    guidelines_text = guidelines_text[:3000]

    client = get_client()
    logger.info(f"LLM 模型: {LLM_MODEL}, base_url: {LLM_BASE_URL}")

    targets = LABELS if args.label == "all" else [args.label]
    for label in targets:
        out = gen_for_label(client, label, args.n, guidelines_text)
        out_path = SYNTHETIC_DIR / f"llm_{label}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        logger.info(f"[{label}] 落盘 {len(out)} 条 → {out_path}")


if __name__ == "__main__":
    main()
