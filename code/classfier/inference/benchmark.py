"""
推理延迟 / 吞吐基准测试。

用法：
    python -m classfier.inference.benchmark
    python -m classfier.inference.benchmark --model models/best --n 500 --batch 1
"""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classfier.config import CLASSFIER_ROOT
from classfier.training.utils import load_jsonl
from classfier.inference.predictor import QueryClassifier

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("classfier.inference.benchmark")


SAMPLE_QUERIES = [
    "红烧肉要炖多久", "推荐几道川菜", "用鸡蛋和番茄可以做什么菜", "川菜是什么菜系",
    "麻婆豆腐要小火还是大火", "有什么简单易做的早餐", "适合减脂的汤", "红烧肉怎么做",
    "冰糖和酱油的比例", "介绍几道下饭菜", "夏天吃什么凉菜", "新手适合做什么",
    "桂花糕要用什么粉", "牛肉怎么腌制", "川菜里适合冬天的汤", "求几道快手菜",
]


def _build_corpus(n: int, seed: int = 42) -> List[str]:
    random.seed(seed)
    return [random.choice(SAMPLE_QUERIES) for _ in range(n)]


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CLASSFIER_ROOT / "models" / "best"))
    ap.add_argument("--n", type=int, default=500, help="总 query 数")
    ap.add_argument("--batch", type=int, default=1, help="batch_size；1=模拟单条场景")
    ap.add_argument("--warmup", type=int, default=10, help="预热条数（不计入统计）")
    ap.add_argument("--out", default=str(CLASSFIER_ROOT / "reports" / "benchmark.json"))
    args = ap.parse_args()

    clf = QueryClassifier(args.model)
    corpus = _build_corpus(args.n)

    # 预热
    if args.warmup:
        _ = clf.predict_batch(corpus[:args.warmup], top_k=1, batch_size=args.batch)

    # 测延迟 (batch=1 时 = 单条延迟)
    latencies: List[float] = []
    if args.batch == 1:
        for t in corpus[args.warmup:]:
            s = time.perf_counter()
            _ = clf.predict(t, top_k=1)
            latencies.append((time.perf_counter() - s) * 1000)
    else:
        # 按 batch 切分测
        for i in range(args.warmup, args.n, args.batch):
            chunk = corpus[i:i + args.batch]
            s = time.perf_counter()
            _ = clf.predict_batch(chunk, top_k=1, batch_size=args.batch)
            latencies.append((time.perf_counter() - s) * 1000 / len(chunk))

    measured = latencies
    p50 = _percentile(measured, 0.5)
    p95 = _percentile(measured, 0.95)
    p99 = _percentile(measured, 0.99)
    qps = 1000.0 / (np.mean(measured) + 1e-9) if measured else 0.0
    result = {
        "model": args.model,
        "n": args.n,
        "warmup": args.warmup,
        "batch": args.batch,
        "device": clf.device,
        "latency_ms": {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "mean": round(float(np.mean(measured)), 3),
            "min": round(float(np.min(measured)), 3),
            "max": round(float(np.max(measured)), 3),
        },
        "throughput_qps": round(float(qps), 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"batch={args.batch}  p50={p50:.2f}ms  p95={p95:.2f}ms  "
                f"p99={p99:.2f}ms  qps≈{qps:.1f}")
    logger.info(f"结果 → {out_path}")


if __name__ == "__main__":
    main()
