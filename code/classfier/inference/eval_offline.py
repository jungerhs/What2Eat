"""
端到端离线评估（4 阶段 pipeline）：
  - in-domain (test.jsonl): 评估 4 分类 macro-F1 等指标
  - OOD (out_of_domain.jsonl): 统计兜住率 + stage 分布
  - LLM 调用统计 (call_count / token_used / fail_count)

用法：
    python -m classfier.inference.eval_offline
    python -m classfier.inference.eval_offline --no-lr          # 不挂 LR
    python -m classfier.inference.eval_offline --no-llm        # 不挂 LLM 兜底
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classfier.config import (
    CLASSFIER_ROOT, BERT_THRESHOLD, LR_THRESHOLD_LOW, LR_THRESHOLD_HIGH,
)
from classfier.training.utils import (
    full_eval_report, load_jsonl, load_label_map, write_eval_report,
)
from classfier.inference.predictor import QueryClassifier

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("classfier.inference.eval_offline")


def _stage_distribution(preds: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter()
    for p in preds:
        c[p.get("stage", "?")] += 1
    return dict(c)


def _eval_ood(clf: QueryClassifier, ood_path: Path, id2label: Dict[int, str]) -> Dict[str, Any]:
    items = load_jsonl(ood_path)
    if not items:
        return {"n": 0}
    texts = [it["text"] for it in items]
    preds = clf.predict_batch(texts, top_k=1)

    is_unknown_count = sum(1 for p in preds if p["is_unknown"])
    return {
        "n": len(preds),
        "caught_count": is_unknown_count,
        "caught_rate": round(is_unknown_count / len(preds), 4) if preds else 0.0,
        "stage_distribution": _stage_distribution(preds),
        "pred_label_dist": dict(Counter(p["label"] for p in preds)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CLASSFIER_ROOT / "models" / "best"))
    ap.add_argument("--data", default=str(CLASSFIER_ROOT / "data" / "processed" / "test.jsonl"))
    ap.add_argument("--ood", default=str(CLASSFIER_ROOT / "data" / "processed" / "out_of_domain.jsonl"),
                    help="主 OOD 评估集（手工 20 条）")
    ap.add_argument("--ood-extra", default=str(CLASSFIER_ROOT / "data" / "lr_filter" / "ood_synth.jsonl"),
                    help="额外 OOD 评估集（LLM 合成 200 条）；文件不存在则跳过")
    ap.add_argument("--label-map", default=str(CLASSFIER_ROOT / "data" / "processed" / "label_map.json"))
    ap.add_argument("--out-dir", default=str(CLASSFIER_ROOT / "reports"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bert-threshold", type=float, default=BERT_THRESHOLD)
    ap.add_argument("--bert-device", default=None, help="BERT 推理 device (cpu/cuda)，不传则自动检测；跑 ollama 评估时建议传 cpu 避免 GPU 冲突")
    ap.add_argument("--lr-threshold-low", type=float, default=LR_THRESHOLD_LOW)
    ap.add_argument("--lr-threshold-high", type=float, default=LR_THRESHOLD_HIGH)
    ap.add_argument("--lr-model", default=str(CLASSFIER_ROOT / "models" / "lr_filter" / "lr.pkl"))
    ap.add_argument("--lr-tfidf", default=str(CLASSFIER_ROOT / "models" / "lr_filter" / "tfidf.pkl"))
    ap.add_argument("--no-lr", action="store_true", help="不挂 LR")
    ap.add_argument("--no-llm", action="store_true", help="关掉 LLM 兜底")
    args = ap.parse_args()

    label2id = load_label_map(Path(args.label_map))
    id2label = {v: k for k, v in label2id.items()}

    clf = QueryClassifier(
        args.model,
        lr_model_path=None if args.no_lr else args.lr_model,
        lr_tfidf_path=None if args.no_lr else args.lr_tfidf,
        bert_threshold=args.bert_threshold,
        lr_threshold_low=args.lr_threshold_low,
        lr_threshold_high=args.lr_threshold_high,
        use_llm_fallback=(not args.no_llm),
        bert_device=args.bert_device,
    )
    logger.info(f"LR={'on' if clf.lr else 'off'}, "
                f"LLM_fallback={'on' if clf.use_llm_fallback else 'off'}, "
                f"bert_th={args.bert_threshold}, lr_low={args.lr_threshold_low}, "
                f"lr_high={args.lr_threshold_high}")

    # ============ in-domain (test.jsonl) ============
    items = load_jsonl(Path(args.data))
    texts = [it["text"] for it in items]
    preds = clf.predict_batch(texts, top_k=2, batch_size=args.batch_size)

    y_true = np.array([label2id[it["label"]] for it in items])
    y_pred = np.array([p["label_id"] if p["label_id"] >= 0 else -1 for p in preds])
    # llm_fallback (label_id=-2) 暂时用 0 占位
    y_pred = np.where(y_pred < 0, 0, y_pred)

    valid_mask = np.array([p["label_id"] >= 0 for p in preds])
    y_proba = np.array([
        [p["probs"][lbl] for lbl in id2label.values()] for p in preds
    ])

    report = full_eval_report(y_true, y_pred, y_proba, id2label, top_k=2)
    report["in_domain_stage_distribution"] = _stage_distribution(preds)
    report["in_domain_llm_fallback_count"] = sum(
        1 for p in preds if p.get("stage") == "llm_fallback"
    )

    run_id = args.run_id or "pipeline"
    paths = write_eval_report(report, Path(args.out_dir), f"{run_id}_test")
    logger.info(f"test  accuracy={report['accuracy']:.4f}  macro_f1={report['macro_f1']:.4f}")
    logger.info(f"test stage 分布: {report['in_domain_stage_distribution']}")
    logger.info(f"test 报告 → {paths['json']}\n          {paths['md']}")

    # ============ OOD ============
    ood_report: Dict[str, Any] = {}
    if Path(args.ood).exists():
        ood_report = _eval_ood(clf, Path(args.ood), id2label)
        ood_path = Path(args.out_dir) / f"ood_report_{run_id}.json"
        ood_path.write_text(json.dumps(ood_report, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        logger.info(f"OOD  n={ood_report['n']}  caught_rate={ood_report['caught_rate']}  "
                    f"stage={ood_report['stage_distribution']}")
        logger.info(f"OOD 报告 → {ood_path}")

    # ============ OOD (额外集：LLM 合成 N 条) ============
    ood_extra_report: Dict[str, Any] = {}
    if args.ood_extra and Path(args.ood_extra).exists():
        ood_extra_report = _eval_ood(clf, Path(args.ood_extra), id2label)
        ood_extra_path = Path(args.out_dir) / f"ood_report_{run_id}_extra.json"
        ood_extra_path.write_text(json.dumps(ood_extra_report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        logger.info(f"OOD(extra)  n={ood_extra_report['n']}  "
                    f"caught_rate={ood_extra_report['caught_rate']}  "
                    f"stage={ood_extra_report['stage_distribution']}")
        logger.info(f"OOD(extra) 报告 → {ood_extra_path}")
    else:
        logger.info(f"OOD(extra) 跳过（{args.ood_extra} 不存在）")

    # ============ LLM 统计 ============
    llm_stats = clf.llm_stats
    logger.info(f"LLM  calls={llm_stats['call_count']}  fail={llm_stats['fail_count']}  "
                f"~token={llm_stats['token_used']}")

    # ============ 总报告 ============
    summary = {
        "run_id": run_id,
        "model": args.model,
        "lr": "on" if clf.lr else "off",
        "llm_fallback": "on" if clf.use_llm_fallback else "off",
        "thresholds": {
            "bert": args.bert_threshold,
            "lr_low": args.lr_threshold_low,
            "lr_high": args.lr_threshold_high,
        },
        "in_domain": {
            "n": len(items),
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "stage_dist": report["in_domain_stage_distribution"],
            "llm_fallback_count": report["in_domain_llm_fallback_count"],
        },
        "ood": ood_report,
        "ood_extra": ood_extra_report,
        "llm_stats": llm_stats,
    }
    summary_path = Path(args.out_dir) / f"pipeline_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"汇总 → {summary_path}")


if __name__ == "__main__":
    main()
