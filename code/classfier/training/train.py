"""
训练入口 (HF Trainer + 3 seeds)。

用法：
    python -m classfier.training.train
    python -m classfier.training.train --epochs 3 --seeds 42
    python -m classfier.training.train --config training/config.yaml
"""
import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classfier.config import CLASSFIER_ROOT, PROCESSED_DIR, LOG_LEVEL
from classfier.training.utils import (
    Collator, JsonlDataset, compute_classification_metrics,
    device_info, load_jsonl, load_label_map, set_seed, write_eval_report,
    full_eval_report,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("classfier.training.train")


# ===================== 工具 =====================

def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _has_gpu() -> bool:
    return torch.cuda.is_available()


def _resolve_fp16(cfg: Any) -> bool:
    """auto: 有 GPU 开, 否则关"""
    if isinstance(cfg, bool):
        return cfg and _has_gpu()
    if isinstance(cfg, str) and cfg.lower() == "auto":
        return _has_gpu()
    return False


# ===================== 训练单个 seed =====================

def train_one_seed(
    seed: int,
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    """返回该 seed 的关键指标"""
    set_seed(seed)
    tcfg = cfg["training"]
    ecfg = cfg["eval"]

    train_items = load_jsonl(paths["train"])
    dev_items = load_jsonl(paths["dev"])
    logger.info(f"[seed={seed}] train={len(train_items)} dev={len(dev_items)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_ds = JsonlDataset(train_items, tokenizer, label2id, cfg["max_length"])
    dev_ds = JsonlDataset(dev_items, tokenizer, label2id, cfg["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=cfg["num_labels"],
        id2label=id2label,
        label2id=label2id,
    )

    run_dir = paths["models_dir"] / f"seed_{seed}"
    best_dir = run_dir / cfg.get("best_model_subdir", "best")
    best_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=ecfg["batch_size"],
        learning_rate=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        warmup_ratio=tcfg["warmup_ratio"],
        gradient_accumulation_steps=tcfg.get("grad_accum_steps", 1),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        label_smoothing_factor=tcfg.get("label_smoothing", 0.0),
        fp16=_resolve_fp16(tcfg.get("fp16", "auto")),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=tcfg["early_stopping"]["metric"],
        greater_is_better=tcfg["early_stopping"].get("greater_is_better", True),
        logging_steps=20,
        report_to=[],
        seed=seed,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=Collator(tokenizer),
        compute_metrics=lambda p: compute_classification_metrics(
            np.argmax(p.predictions, axis=-1), p.label_ids, id2label
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=tcfg["early_stopping"]["patience"])
        ],
    )

    logger.info(f"[seed={seed}] start training")
    train_result = trainer.train()
    logger.info(f"[seed={seed}] train done. metrics={train_result.metrics}")

    # 拷贝 best 模型
    # Trainer 已经在 best_dir 存了 best checkpoint (load_best_model_at_end)
    # 找最近一个 checkpoint 目录
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        src = ckpts[-1]
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, best_dir / f.name)
    else:
        # 没产生 checkpoint（不可能但兜底），把当前模型保存到 best_dir
        trainer.save_model(str(best_dir))

    tokenizer.save_pretrained(str(best_dir))
    (best_dir / "label_map.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 训练 summary
    (run_dir / "train_summary.json").write_text(
        json.dumps(train_result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 在 dev 上重 eval 一次（best 模型）
    best_model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
    best_model.eval()
    if _has_gpu():
        best_model.cuda()
    best_tokenizer = AutoTokenizer.from_pretrained(str(best_dir))
    best_collator = Collator(best_tokenizer)
    best_trainer = Trainer(model=best_model, args=TrainingArguments(
        output_dir=str(best_dir), report_to=[], per_device_eval_batch_size=ecfg["batch_size"],
        disable_tqdm=True,
    ), data_collator=best_collator, eval_dataset=dev_ds)
    pred = best_trainer.predict(dev_ds)
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = np.array([label2id[x["label"]] for x in dev_items])
    y_proba = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
    dev_report = full_eval_report(y_true, y_pred, y_proba, id2label, top_k=ecfg.get("top_k", 2))
    paths_out = write_eval_report(dev_report, paths["reports_dir"], f"seed_{seed}_dev")
    logger.info(f"[seed={seed}] dev report → {paths_out['json']}, {paths_out['md']}")

    return {
        "seed": seed,
        "dev_accuracy": dev_report["accuracy"],
        "dev_macro_f1": dev_report["macro_f1"],
        "train_metrics": train_result.metrics,
        "best_dir": str(best_dir),
    }


# ===================== 入口 =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CLASSFIER_ROOT / "training" / "config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="覆盖 config 里的 seeds；空列表 = 不跑任何 seed")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练 epoch")
    ap.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr

    pcfg = cfg["paths"]
    paths = {
        "processed_dir": CLASSFIER_ROOT / pcfg["processed_dir"],
        "train": CLASSFIER_ROOT / pcfg["processed_dir"] / pcfg["train_file"],
        "dev": CLASSFIER_ROOT / pcfg["processed_dir"] / pcfg["dev_file"],
        "test": CLASSFIER_ROOT / pcfg["processed_dir"] / pcfg["test_file"],
        "label_map_path": CLASSFIER_ROOT / pcfg["processed_dir"] / pcfg["label_map"],
        "models_dir": CLASSFIER_ROOT / pcfg["models_dir"],
        "reports_dir": CLASSFIER_ROOT / pcfg["reports_dir"],
    }
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    label2id = load_label_map(paths["label_map_path"])
    id2label = {v: k for k, v in label2id.items()}
    logger.info(f"label2id={label2id}")
    logger.info(f"device={device_info()}")

    seeds = args.seeds if args.seeds is not None else cfg["training"].get("seeds", [42])
    if not seeds:
        logger.warning("seeds 为空，nothing to do")
        return

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        try:
            r = train_one_seed(seed, cfg, paths, label2id, id2label)
            results.append(r)
        except Exception as e:
            logger.exception(f"[seed={seed}] 训练失败: {e}")

    if not results:
        logger.error("所有 seed 都失败")
        return

    # 汇总
    accs = [r["dev_accuracy"] for r in results]
    f1s = [r["dev_macro_f1"] for r in results]
    best = max(results, key=lambda r: r["dev_macro_f1"])
    summary = {
        "n_seeds": len(results),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "best_seed": best["seed"],
        "best_dir": best["best_dir"],
        "per_seed": results,
    }
    summary_path = paths["reports_dir"] / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"训练汇总 → {summary_path}")
    logger.info(f"macro-F1 mean={summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}, "
                f"best_seed={summary['best_seed']}")

    # 把 best seed 的模型软链/复制到 models/best
    final_best = paths["models_dir"] / "best"
    if final_best.exists():
        if final_best.is_symlink():
            final_best.unlink()
        else:
            shutil.rmtree(final_best)
    try:
        # Windows 没 symlink，用 junction
        os.link(best["best_dir"], final_best)
    except OSError:
        shutil.copytree(best["best_dir"], final_best)
    logger.info(f"final best → {final_best}")


if __name__ == "__main__":
    main()
