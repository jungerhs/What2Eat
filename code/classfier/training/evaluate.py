"""
在 test.jsonl 上评估已训练模型，输出 JSON + Markdown 报告。

用法：
    python -m classfier.training.evaluate
    python -m classfier.training.evaluate --model models/seed_42/best
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classfier.config import CLASSFIER_ROOT, PROCESSED_DIR
from classfier.training.utils import (
    Collator, JsonlDataset, device_info, full_eval_report,
    load_jsonl, load_label_map, write_eval_report,
)

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("classfier.training.evaluate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CLASSFIER_ROOT / "models" / "best"),
                    help="模型目录，含 pytorch_model.bin / tokenizer")
    ap.add_argument("--data", default=str(CLASSFIER_ROOT / "data" / "processed" / "test.jsonl"))
    ap.add_argument("--label-map", default=str(CLASSFIER_ROOT / "data" / "processed" / "label_map.json"))
    ap.add_argument("--out-dir", default=str(CLASSFIER_ROOT / "reports"))
    ap.add_argument("--run-id", default=None, help="报告文件名后缀；默认从 model 目录名推断")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    model_dir = Path(args.model)
    data_path = Path(args.data)
    label_map_path = Path(args.label_map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label2id = load_label_map(label_map_path)
    id2label = {v: k for k, v in label2id.items()}

    items = load_jsonl(data_path)
    logger.info(f"载入 {len(items)} 条 test 样本；device={device_info()}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    ds = JsonlDataset(items, tokenizer, label2id, args.max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir),
            report_to=[],
            per_device_eval_batch_size=args.batch_size,
            disable_tqdm=True,
        ),
        data_collator=Collator(tokenizer),
    )
    pred = trainer.predict(ds)
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = np.array([label2id[x["label"]] for x in items])
    y_proba = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()

    run_id = args.run_id or f"eval_{model_dir.parent.name}_{model_dir.name}"
    report = full_eval_report(y_true, y_pred, y_proba, id2label, top_k=2)
    paths = write_eval_report(report, out_dir, run_id)
    logger.info(f"accuracy={report['accuracy']:.4f}  macro_f1={report['macro_f1']:.4f}")
    logger.info(f"报告 → {paths['json']}\n        {paths['md']}")


if __name__ == "__main__":
    main()
