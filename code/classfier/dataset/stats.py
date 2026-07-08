"""
统计 processed/ 下数据集，生成 reports/dataset_stats.md。

统计项：
  - train/dev/test 总数与每类分布
  - text 长度分布 (min/p25/median/p75/max)
  - 来源分布 (template / llm / agreed)
  - OOD 数量
  - 类别不平衡比例
"""
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, LABELS, CLASSFIER_ROOT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stats")


REPORTS_DIR = CLASSFIER_ROOT / "reports"


def _percentile(xs: List[int], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _load(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _stats_block(name: str, items: List[Dict]) -> str:
    if not items:
        return f"### {name}\n- 数据为空\n\n"
    n = len(items)
    dist = Counter(x.get("label", "?") for x in items)
    src = Counter(x.get("source", "?") for x in items)
    lens = [len(x.get("text", "")) for x in items]
    return (
        f"### {name}\n"
        f"- 总数: **{n}**\n"
        f"- 类别分布: " + " | ".join(f"`{k}`={v}" for k, v in sorted(dist.items())) + "\n"
        f"- 来源分布: " + " | ".join(f"`{k}`={v}" for k, v in sorted(src.items())) + "\n"
        f"- 长度: min={min(lens)} p25={_percentile(lens, 0.25):.0f} "
        f"median={_percentile(lens, 0.5):.0f} p75={_percentile(lens, 0.75):.0f} max={max(lens)}\n\n"
    )


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines: List[str] = ["# 数据集统计报告\n"]

    for split in ["train", "dev", "test"]:
        lines.append(_stats_block(split, _load(PROCESSED_DIR / f"{split}.jsonl")))

    lines.append(_stats_block("all", _load(PROCESSED_DIR / "all.jsonl")))
    lines.append(_stats_block("out_of_domain", _load(PROCESSED_DIR / "out_of_domain.jsonl")))

    # 类别平衡性
    train = _load(PROCESSED_DIR / "train.jsonl")
    if train:
        dist = Counter(x["label"] for x in train)
        n_min = min(dist.values())
        n_max = max(dist.values())
        ratio = n_max / max(1, n_min)
        lines.append("### 类别平衡性\n")
        lines.append(f"- 最少类: **{min(dist, key=dist.get)}** = {n_min}")
        lines.append(f"- 最多类: **{max(dist, key=dist.get)}** = {n_max}")
        lines.append(f"- 不平衡比: **{ratio:.2f}** ({'健康' if ratio < 1.5 else '需考虑加权或过采样'})\n")

    out = REPORTS_DIR / "dataset_stats.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"报告: {out}")


if __name__ == "__main__":
    main()
