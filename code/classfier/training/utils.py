"""
训练 / 评估共用工具：
  - JsonlDataset：torch Dataset
  - build_collator：动态 padding
  - load_label_map / load_jsonl
  - compute_metrics：给 HF Trainer 用
  - write_eval_report：JSON + Markdown
"""
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger("classfier.training.utils")


# ===================== IO =====================

def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_label_map(path: Path) -> Dict[str, int]:
    m = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(m, dict):
        raise ValueError(f"label_map 格式错误: {path}")
    return m


# ===================== Dataset =====================

class JsonlDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer: PreTrainedTokenizerBase,
                 label2id: Dict[str, int], max_length: int):
        self.items = items
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        enc = self.tokenizer(
            it["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        label = self.label2id[it["label"]]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label,
        }


@dataclass
class Collator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attn, labels = [], [], []
        for x in batch:
            n = len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * (max_len - n))
            attn.append(x["attention_mask"] + [0] * (max_len - n))
            labels.append(x["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ===================== 评估指标 =====================

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_classification_metrics(preds: np.ndarray, labels: np.ndarray,
                                   id2label: Dict[int, str]) -> Dict[str, float]:
    """给 HF Trainer 用的轻量指标 (Trainer 内部需要小 dict)。"""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


def full_eval_report(y_true: Sequence, y_pred: Sequence, y_proba: Optional[np.ndarray],
                     id2label: Dict[int, str], top_k: int = 2) -> Dict[str, Any]:
    """完整评估报告 (evaluate.py / eval_offline.py 共享)。"""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support,
        classification_report, confusion_matrix, top_k_accuracy_score,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(id2label.keys())
    target_names = [id2label[i] for i in labels]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        target_names[i]: {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]), "support": int(support[i]),
        } for i in range(len(labels))
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    top_k_acc = None
    if y_proba is not None and y_proba.shape[1] > 1:
        try:
            top_k_acc = float(top_k_accuracy_score(y_true, y_proba, k=min(top_k, y_proba.shape[1]),
                                                    labels=labels))
        except Exception:
            top_k_acc = None
    cls_report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        f"top_{top_k}_accuracy": top_k_acc,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": target_names,
            "matrix": cm,
        },
        "classification_report": cls_report,
        "n_samples": int(len(y_true)),
    }


def write_eval_report(report: Dict[str, Any], out_dir: Path, run_id: str) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_report_{run_id}.json"
    md_path = out_dir / f"eval_report_{run_id}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report, run_id), encoding="utf-8")
    return {"json": json_path, "md": md_path}


def _render_markdown(report: Dict[str, Any], run_id: str) -> str:
    lines = [f"# 评估报告 — {run_id}", ""]
    lines.append(f"- 样本数: **{report['n_samples']}**")
    lines.append(f"- accuracy: **{report['accuracy']:.4f}**")
    lines.append(f"- macro-F1: **{report['macro_f1']:.4f}**")
    lines.append(f"- micro-F1: **{report['micro_f1']:.4f}**")
    lines.append(f"- weighted-F1: **{report['weighted_f1']:.4f}**")
    if report.get("top_2_accuracy") is not None:
        lines.append(f"- top-2 accuracy: **{report['top_2_accuracy']:.4f}**")
    lines.append("")
    lines.append("## Per-class")
    lines.append("| class | precision | recall | f1 | support |")
    lines.append("|---|---|---|---|---|")
    for k, v in report["per_class"].items():
        lines.append(f"| {k} | {v['precision']:.4f} | {v['recall']:.4f} | "
                     f"{v['f1']:.4f} | {v['support']} |")
    lines.append("")
    lines.append("## Confusion Matrix (rows=true, cols=pred)")
    cm = report["confusion_matrix"]
    labels = cm["labels"]
    matrix = cm["matrix"]
    lines.append("| true \\ pred | " + " | ".join(labels) + " |")
    lines.append("|---|" + "|".join(["---"] * len(labels)) + "|")
    for i, row in enumerate(matrix):
        lines.append(f"| **{labels[i]}** | " + " | ".join(str(x) for x in row) + " |")
    lines.append("")
    lines.append("## Classification Report")
    lines.append("```")
    lines.append(report["classification_report"])
    lines.append("```")
    return "\n".join(lines)


# ===================== Misc =====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_info() -> Dict[str, Any]:
    info = {"cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
    return info
