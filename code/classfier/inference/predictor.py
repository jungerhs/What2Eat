"""
QueryClassifier 4 阶段推理：

  Query
    ↓
  [Stage 1] LR (cooking/ood)
    ├─ p < lr_threshold_low        → unknown (lr_reject)
    ├─ low ≤ p < lr_threshold_high → [Stage 1.5] LLM is_cooking
    │     ├─ false → unknown (llm_reject_cooking)
    │     └─ true  → 进入 Stage 2
    └─ p ≥ lr_threshold_high       → 进入 Stage 2
  [Stage 2] BERT 4 分类
    ├─ conf ≥ bert_threshold       → 输出
    └─ conf < bert_threshold       → [Stage 2.5] LLM classify_4
          └─ 输出 (llm_fallback)

用法：
    from classfier.inference.predictor import QueryClassifier
    clf = QueryClassifier(
        "models/best",
        lr_model_path="models/lr_filter/lr.pkl",
        lr_tfidf_path="models/lr_filter/tfidf.pkl",
    )
    clf.predict("红烧肉要炖多久")
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classfier.config import (
    CLASSFIER_ROOT,
    LR_THRESHOLD_LOW, LR_THRESHOLD_HIGH, BERT_THRESHOLD, USE_LLM_FALLBACK,
)
from classfier.inference.lr_filter import LRFilter
from classfier.inference.llm_judge import LLMJudge

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("classfier.inference.predictor")


# ===================== BERT 子模块 =====================

class _BertClassifier:
    """纯 BERT 4 分类子模块，与旧版兼容"""
    def __init__(self, model_dir: str, max_length: int = 64, device: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        id2label = getattr(self.model.config, "id2label", None)
        if id2label and all(str(k).isdigit() for k in id2label.keys()):
            self.id2label = {int(k): v for k, v in id2label.items()}
        else:
            mp = self.model_dir / "label_map.json"
            label2id = json.loads(mp.read_text(encoding="utf-8"))
            self.id2label = {v: k for k, v in label2id.items()}

    @torch.inference_mode()
    def _proba_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**enc).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)


# ===================== 主类 =====================

class QueryClassifier:
    def __init__(
        self,
        model_dir: str,
        lr_model_path: Optional[str] = None,
        lr_tfidf_path: Optional[str] = None,
        bert_threshold: float = BERT_THRESHOLD,
        lr_threshold_low: float = LR_THRESHOLD_LOW,
        lr_threshold_high: float = LR_THRESHOLD_HIGH,
        use_llm_fallback: bool = USE_LLM_FALLBACK,
        max_length: int = 64,
        bert_device: Optional[str] = None,
    ):
        self.bert = _BertClassifier(model_dir, max_length=max_length, device=bert_device)
        self.id2label = self.bert.id2label
        self.bert_threshold = bert_threshold
        self.lr_threshold_low = lr_threshold_low
        self.lr_threshold_high = lr_threshold_high
        self.use_llm_fallback = use_llm_fallback

        # LR (可选)
        self.lr: Optional[LRFilter] = None
        if lr_model_path and lr_tfidf_path and Path(lr_model_path).exists():
            self.lr = LRFilter(lr_model_path, lr_tfidf_path)
            logger.info(f"LR 已加载: {lr_model_path}")
        else:
            logger.info("LR 未加载（lr_model_path 为空或文件不存在）")

        # LLM judge (可选，但 use_llm_fallback=True 时强制加载)
        self.llm: Optional[LLMJudge] = None
        if use_llm_fallback:
            try:
                self.llm = LLMJudge()
                logger.info("LLM 兜底已加载")
            except Exception as e:
                logger.warning(f"LLM 兜底加载失败: {e}, use_llm_fallback 自动降级为 False")
                self.use_llm_fallback = False

    # ---------- 辅助 ----------

    def _stage_unknown(self, text: str, stage: str, p: float, **extra) -> Dict[str, Any]:
        return {
            "text": text,
            "label": "unknown",
            "label_id": -1,
            "confidence": p,
            "probs": {lbl: 0.0 for lbl in self.id2label.values()},
            "top2": [],
            "is_unknown": True,
            "stage": stage,
            **extra,
        }

    def _format_bert(self, text: str, proba: np.ndarray, top_k: int) -> Dict[str, Any]:
        idx_sorted = np.argsort(proba)[::-1][:top_k]
        topk = [{"label": self.id2label[int(i)], "prob": float(proba[i])} for i in idx_sorted]
        best = int(idx_sorted[0])
        return {
            "text": text,
            "label": self.id2label[best],
            "label_id": best,
            "confidence": float(proba[best]),
            "probs": {self.id2label[i]: float(proba[i]) for i in range(len(proba))},
            "top2": topk,
            "is_unknown": False,
            "stage": "bert",
        }

    def _stage_bert(self, text: str, top_k: int) -> Dict[str, Any]:
        """Stage 2: BERT 4 分类 + 可选 LLM 4 类兜底。

        注意：Stage 2/2.5 假设 query 已是烹饪 4 类之一（OOD 兜住由 Stage 1/1.5 负责）。
        若 query 是 OOD 但 LR 缺失，越界会直接被 BERT 4 分类，本函数不再做 is_cooking 判别。
        """
        proba = self.bert._proba_batch([text])[0]
        result = self._format_bert(text, proba, top_k)
        if result["confidence"] < self.bert_threshold and self.use_llm_fallback:
            llm = self.llm.classify_4(text)
            if llm.get("label"):
                result.update({
                    "label": llm["label"],
                    "label_id": -2,
                    "confidence": llm["confidence"],
                    "is_unknown": False,
                    "stage": "llm_fallback",
                    "llm_raw": llm.get("raw"),
                })
        return result

    # ---------- 单条 ----------

    def predict(self, text: str, top_k: int = 2) -> Dict[str, Any]:
        # Stage 1: LR
        if self.lr is not None:
            p = float(self.lr.cooking_proba([text])[0])
            if p ==0.0:
                llm_r = self.llm.is_cooking(text)
                if not llm_r["is_cooking"]:
                    return self._stage_unknown(text, "llm_reject_cooking", p,
                                                lr_proba=p, llm_conf=llm_r["confidence"])
            if p < self.lr_threshold_low :
                return self._stage_unknown(text, "lr_reject", p, lr_proba=p)
            if p < self.lr_threshold_high and self.use_llm_fallback :
                llm_r = self.llm.is_cooking(text)
                if not llm_r["is_cooking"]:
                    return self._stage_unknown(text, "llm_reject_cooking", p,
                                                lr_proba=p, llm_conf=llm_r["confidence"])
        # Stage 2: BERT (可能再 Stage 2.5)
        return self._stage_bert(text, top_k)

    # ---------- 批量 ----------

    def predict_batch(self, texts: List[str], top_k: int = 2,
                      batch_size: int = 32) -> List[Dict[str, Any]]:
        if not texts:
            return []
        results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
        # Stage 1: LR 批量判
        if self.lr is not None:
            lr_probs = self.lr.cooking_proba(texts)
        else:
            lr_probs = np.ones(len(texts))  # 不开 LR 时全部直通

        # 哪些要进入 Stage 2 (BERT)
        need_bert_idx: List[int] = []
        for i, p in enumerate(lr_probs):
            if p >= self.lr_threshold_low:
                if p < self.lr_threshold_high and self.use_llm_fallback:
                    # 中间地带：先调 LLM 判烹饪
                    llm_r = self.llm.is_cooking(texts[i])
                    if not llm_r["is_cooking"]:
                        results[i] = self._stage_unknown(
                            texts[i], "llm_reject_cooking", top_k,
                            lr_proba=float(p), llm_conf=llm_r["confidence"])
                        continue
                need_bert_idx.append(i)
            else:
                # LR 极不自信
                results[i] = self._stage_unknown(texts[i], "lr_reject", top_k,
                                                  lr_proba=float(p))

        # Stage 2: 批量 BERT
        if need_bert_idx:
            bert_texts = [texts[i] for i in need_bert_idx]
            proba_all = self.bert._proba_batch(bert_texts, batch_size=batch_size)
            for j, idx in enumerate(need_bert_idx):
                result = self._format_bert(texts[idx], proba_all[j], top_k)
                if result["confidence"] < self.bert_threshold and self.use_llm_fallback:
                    llm = self.llm.classify_4(texts[idx])
                    if llm.get("label"):
                        result.update({
                            "label": llm["label"],
                            "label_id": -2,
                            "confidence": llm["confidence"],
                            "is_unknown": False,
                            "stage": "llm_fallback",
                            "llm_raw": llm.get("raw"),
                        })
                results[idx] = result
        return results

    # ---------- LLM 统计 ----------

    @property
    def llm_stats(self) -> Dict[str, int]:
        if self.llm is None:
            return {"call_count": 0, "token_used": 0, "fail_count": 0}
        return {
            "call_count": self.llm.call_count,
            "token_used": self.llm.token_used,
            "fail_count": self.llm.fail_count,
        }


# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CLASSFIER_ROOT / "models" / "best"))
    ap.add_argument("--lr-model", default=str(CLASSFIER_ROOT / "models" / "lr_filter" / "lr.pkl"))
    ap.add_argument("--lr-tfidf", default=str(CLASSFIER_ROOT / "models" / "lr_filter" / "tfidf.pkl"))
    ap.add_argument("--text", default=None, help="单条 query")
    ap.add_argument("--file", default=None, help="jsonl 批量推理")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--bert-threshold", type=float, default=BERT_THRESHOLD)
    ap.add_argument("--lr-threshold-low", type=float, default=LR_THRESHOLD_LOW)
    ap.add_argument("--lr-threshold-high", type=float, default=LR_THRESHOLD_HIGH)
    ap.add_argument("--no-llm", action="store_true", help="关掉 LLM 兜底")
    args = ap.parse_args()

    clf = QueryClassifier(
        args.model,
        lr_model_path=args.lr_model,
        lr_tfidf_path=args.lr_tfidf,
        bert_threshold=args.bert_threshold,
        lr_threshold_low=args.lr_threshold_low,
        lr_threshold_high=args.lr_threshold_high,
        use_llm_fallback=(not args.no_llm),
    )
    logger.info(f"LR={'on' if clf.lr else 'off'}, "
                f"LLM_fallback={'on' if clf.use_llm_fallback else 'off'}")

    if args.text:
        r = clf.predict(args.text, top_k=args.top_k)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    if args.file:
        items = []
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if args.limit:
            items = items[:args.limit]
        texts = [it[args.text_field] for it in items if args.text_field in it]
        preds = clf.predict_batch(texts, top_k=args.top_k)
        for it, p in zip(items, preds):
            print(json.dumps({**it, "pred": p}, ensure_ascii=False))
        if args.out:
            with Path(args.out).open("w", encoding="utf-8") as f:
                for it, p in zip(items, preds):
                    f.write(json.dumps({**it, "pred": p}, ensure_ascii=False) + "\n")
        logger.info(f"LLM stats: {clf.llm_stats}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
