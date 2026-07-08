"""
对 LLM 合成数据 + 模板数据进行二次打标，
不一致的样本写入 disputed.jsonl 供人工/规则兜底。

输入：data/synthetic/llm_*.jsonl + data/synthetic/template_*.jsonl
输出：data/processed/double_check_report.json
      data/processed/disputed.jsonl
      (并把一致样本按 source 写回 data/processed/agreed.jsonl)
"""
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_CONCURRENCY, DISPUTE_RATIO, LABEL_TO_ID,
    DOUBLE_CHECK_AGREEMENT_THRESHOLD, LABELS,
    SYNTHETIC_DIR, PROCESSED_DIR, LOG_LEVEL, CLASSFIER_ROOT,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("double_check")

GUIDELINES_PATH = CLASSFIER_ROOT / "dataset" / "label_guidelines.md"

SYSTEM_PROMPT = """你是 query 分类质检员。给定一条中文 query 和 label_guidelines，
判断它属于 4 类之一：general / detail / multi-hop / recommend。

只输出一个 JSON，不要任何其它字符、Markdown 包装、解释或前缀。

示例：
输入 query: "红烧肉要炖多久"
输出: {"label": "detail", "confidence": 0.95}

输入 query: "推荐几道川菜"
输出: {"label": "recommend", "confidence": 0.92}

输入 query: "用鸡蛋和番茄可以做什么菜"
输出: {"label": "multi-hop", "confidence": 0.9}

输入 query: "川菜是什么菜系"
输出: {"label": "general", "confidence": 0.88}
"""


def get_client() -> OpenAI:
    if not LLM_API_KEY:
        raise ValueError("未配置 API_KEY / MOONSHOT_API_KEY")
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def call_one(client: OpenAI, text: str, guidelines: str) -> Optional[str]:
    prompt = f"""# Label Guidelines (节选)
{guidelines[:2500]}

# Query
{text}

# 输出（严格 JSON）
{{"label": "general|detail|multi-hop|recommend", "confidence": 0~1}}"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            finish = resp.choices[0].finish_reason
            content = re.sub(r"^```(?:json)?|```$", "", content).strip()
            if not content:
                logger.warning(
                    f"质检空响应 attempt={attempt} finish_reason={finish} "
                    f"query={text[:30]!r}"
                )
                time.sleep(1.5 * (attempt + 1))
                continue
            obj = json.loads(content)
            label = obj.get("label")
            if label in LABELS:
                return label
            logger.warning(
                f"质检非法 label attempt={attempt} label={label!r} content={content[:120]!r}"
            )
        except Exception as e:
            snippet = repr(content[:120]) if (content or "").strip() else "n/a"
            logger.warning(f"质检失败 attempt={attempt} err={e} content={snippet}")
            time.sleep(1.5 * (attempt + 1))
    return None


def _iter_synthetic() -> List[Dict]:
    items = []
    for p in sorted(SYNTHETIC_DIR.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def main(sample_ratio: float = DISPUTE_RATIO):
    guidelines = GUIDELINES_PATH.read_text(encoding="utf-8")
    items = _iter_synthetic()
    if not items:
        logger.warning("无合成/模板数据，请先跑 generate_*.py")
        return
    logger.info(f"载入 {len(items)} 条合成/模板样本")

    # 抽样
    sample_size = max(50, int(len(items) * sample_ratio))
    sample = random.sample(items, k=min(sample_size, len(items)))

    client = get_client()
    results: List[Tuple[Dict, Optional[str]]] = []

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as ex:
        futs = {ex.submit(call_one, client, x["text"], guidelines): x for x in sample}
        for fut in as_completed(futs):
            x = futs[fut]
            try:
                new_label = fut.result()
            except Exception as e:
                logger.warning(f"质检异常: {e}")
                new_label = None
            results.append((x, new_label))

    # 统计
    agree, disagree, fail = 0, 0, 0
    disputed, agreed = [], []
    for x, new_label in results:
        if new_label is None:
            fail += 1
            continue
        if new_label == x["label"]:
            agree += 1
            agreed.append(x)
        else:
            disagree += 1
            disputed.append({**x, "original_label": x["label"], "recheck_label": new_label})

    valid = agree + disagree
    agreement = agree / valid if valid else 0.0
    logger.info(f"一致 {agree} / 不一致 {disagree} / 失败 {fail} / 一致率 {agreement:.2%}")

    # 落盘
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (PROCESSED_DIR / "agreed.jsonl").open("w", encoding="utf-8") as f:
        for x in agreed:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with (PROCESSED_DIR / "disputed.jsonl").open("w", encoding="utf-8") as f:
        for x in disputed:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    report = {
        "total_sample": len(sample),
        "agree": agree,
        "disagree": disagree,
        "fail": fail,
        "agreement": round(agreement, 4),
        "threshold": DOUBLE_CHECK_AGREEMENT_THRESHOLD,
        "pass": agreement >= DOUBLE_CHECK_AGREEMENT_THRESHOLD,
        "disputed_count": len(disputed),
    }
    (PROCESSED_DIR / "double_check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"报告: {PROCESSED_DIR / 'double_check_report.json'}")


if __name__ == "__main__":
    main()
