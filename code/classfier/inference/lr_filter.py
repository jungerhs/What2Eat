"""
LR 二分类器（烹饪 / 非烹饪）：
  - 训练 + 序列化 (TF-IDF + LogisticRegression)
  - 推理：predict_proba / predict
  - 报告：precision/recall/F1 + top 重要词

数据准备：自动从 data/processed/{train,dev,test}.jsonl (cooking) +
                 data/lr_filter/ood_synth.jsonl (ood) 重新 8:1:1 切分。

用法：
    # 训练 (需要先跑 generate_ood.py 生成 ood_synth.jsonl)
    python -m classfier.inference.lr_filter train

    # 推理
    from classfier.inference.lr_filter import LRFilter
    clf = LRFilter("models/lr_filter/lr.pkl", "models/lr_filter/tfidf.pkl")
    proba = clf.predict_proba(["红烧肉怎么做", "今天天气"])
"""
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import joblib
import numpy as np

# jieba 兜底
try:
    import jieba
    jieba.setLogLevel(logging.ERROR)  # 抑制 jieba 初始化日志
except ImportError:
    jieba = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classfier.config import (
    PROCESSED_DIR, LR_DATA_DIR, LR_MODEL_DIR, CLASSFIER_ROOT,
    LR_TFIDF_MAX_FEATURES, LR_TFIDF_NGRAM, LR_TFIDF_MIN_DF,
    LR_C, LR_MAX_ITER, LOG_LEVEL, LR_FILTER_DIR,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lr_filter")


# ===================== 工具 =====================

def _tokenize(text: str) -> str:
    if jieba is not None:
        return " ".join(jieba.cut(text))
    # 兜底：逐字切 + 2-gram
    out = list(text) + [text[i:i + 2] for i in range(len(text) - 1)]
    return " ".join(out)


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ===================== 训练 =====================

def _prepare_dataset(seed: int = 42) -> Dict[str, List[Dict]]:
    """从 cooking (data/processed) + ood (data/lr_filter/ood_synth) 重新切分 8:1:1"""
    # 1) 烹饪 query (4 类都算 cooking)
    cooking: List[Dict] = []
    for split in ("train", "dev", "test"):
        for it in _load_jsonl(PROCESSED_DIR / f"{split}.jsonl"):
            cooking.append({**it, "binary_label": "cooking"})

    # 2) OOD
    ood_items = _load_jsonl(LR_FILTER_DIR / "ood_synth.jsonl")
    ood = [{**it, "binary_label": "ood"} for it in ood_items]

    logger.info(f"cooking={len(cooking)}, ood={len(ood)}")
    if not ood:
        raise RuntimeError("data/lr_filter/ood_synth.jsonl 不存在或为空，请先跑 generate_ood.py")

    # 3) 混合 + 切分
    all_items = cooking + ood
    random.Random(seed).shuffle(all_items)
    n = len(all_items)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    splits = {
        "train": all_items[:n_train],
        "dev":   all_items[n_train:n_train + n_dev],
        "test":  all_items[n_train + n_dev:],
    }
    for k, lst in splits.items():
        out = LR_DATA_DIR / f"{k}.jsonl"
        out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lst) + "\n",
                       encoding="utf-8")
        dist = {lbl: sum(1 for x in lst if x["binary_label"] == lbl) for lbl in ("cooking", "ood")}
        logger.info(f"lr_filter/{k}.jsonl  n={len(lst)}  分布={dist}")
    return splits


def _train_eval(splits: Dict[str, List[Dict]]) -> Dict[str, Any]:
    def texts_labels(items):
        X_text = [_tokenize(it["text"]) for it in items]
        y = np.array([1 if it["binary_label"] == "cooking" else 0 for it in items])
        return X_text, y

    Xtr_text, ytr = texts_labels(splits["train"])
    Xde_text, yde = texts_labels(splits["dev"])
    Xte_text, yte = texts_labels(splits["test"])

    logger.info(f"训练 cooking={(ytr==1).sum()} ood={(ytr==0).sum()}")
    # ===== 双通道 TF-IDF (解决 OOV) =====
    # 通道 1: word-level n-gram (1,3) + jieba 分词 → 抓短语模式
    # 通道 2: char-level n-gram (2,5) + char_wb        → 解决 OOV (新菜名/生僻词)
    word_vec = TfidfVectorizer(
        max_features=LR_TFIDF_MAX_FEATURES,
        ngram_range=LR_TFIDF_NGRAM,  # (1, 3)
        min_df=LR_TFIDF_MIN_DF,
        analyzer="word",
        tokenizer=_tokenize,  # jieba 分词
        token_pattern=None,    # 用自定义 tokenizer 时禁用默认 regex
    )
    char_vec = TfidfVectorizer(
        max_features=30000,
        ngram_range=(2, 5),
        min_df=2,
        analyzer="char_wb",
        sublinear_tf=True,
    )
    tfidf = FeatureUnion([("word", word_vec), ("char", char_vec)])
    tfidf.fit(Xtr_text)
    Xtr = tfidf.transform(Xtr_text)
    Xde = tfidf.transform(Xde_text)
    Xte = tfidf.transform(Xte_text)
    logger.info(f"TF-IDF 维度 (word+char 拼接): {Xtr.shape[1]}")

    clf = LogisticRegression(
        C=LR_C, max_iter=LR_MAX_ITER,
        class_weight="balanced", solver="liblinear",
    )
    clf.fit(Xtr, ytr)

    def eval_split(X, y, name):
        proba = clf.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, zero_division=0)
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
        rep = classification_report(
            y, pred, labels=[0, 1],
            target_names=["ood", "cooking"], zero_division=0,
        )
        logger.info(f"[{name}] acc={acc:.4f}  f1={f1:.4f}  "
                    f"P[ood/cooking]={p[0]:.3f}/{p[1]:.3f}  R={r[0]:.3f}/{r[1]:.3f}")
        return {"acc": acc, "f1": f1, "p_ood": p[0], "r_ood": r[0],
                "p_cooking": p[1], "r_cooking": r[1], "report": rep,
                "proba": proba, "pred": pred, "y": y}

    res = {
        "dev": eval_split(Xde, yde, "dev"),
        "test": eval_split(Xte, yte, "test"),
    }

    # top 重要词 (从 word 通道取, char 通道高维难读)
    feat_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]
    # 只统计 word 通道的 top 词 (char 通道高维难读)
    word_feats_out = word_vec.get_feature_names_out()
    n_word_feats = word_feats_out.shape[0]
    coefs_word = coefs[:n_word_feats]
    feat_word = np.array(word_feats_out)
    top_cooking = sorted(
        [(feat_word[i], float(coefs_word[i])) for i in np.argsort(coefs_word)[-30:][::-1]],
        key=lambda x: -x[1],
    )
    top_ood = sorted(
        [(feat_word[i], float(coefs_word[i])) for i in np.argsort(coefs_word)[:30]],
        key=lambda x: x[1],
    )
    res["top_cooking_words"] = top_cooking
    res["top_ood_words"] = top_ood

    # 落盘模型
    out_dir = CLASSFIER_ROOT / "models" / "lr_filter"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "lr.pkl")
    joblib.dump(tfidf, out_dir / "tfidf.pkl")
    logger.info(f"模型 → {out_dir / 'lr.pkl'} + tfidf.pkl")

    return res


# ===================== 推理封装 =====================

@dataclass
class LRFilter:
    model_path: str
    tfidf_path: str

    def __post_init__(self):
        self.clf = joblib.load(self.model_path)
        self.tfidf = joblib.load(self.tfidf_path)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        segs = [_tokenize(t) for t in texts]
        X = self.tfidf.transform(segs)
        return self.clf.predict_proba(X)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(texts)[:, 1]
        return (p >= threshold).astype(int)

    def cooking_proba(self, texts: List[str]) -> np.ndarray:
        return self.predict_proba(texts)[:, 1]


# ===================== CLI =====================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"], default="train", nargs="?")
    ap.add_argument("--model", default=str(CLASSFIER_ROOT / "models" / "lr_filter" / "lr.pkl"))
    ap.add_argument("--tfidf", default=str(CLASSFIER_ROOT / "models" / "lr_filter" / "tfidf.pkl"))
    ap.add_argument("--report", default=str(CLASSFIER_ROOT / "reports" / "lr_filter_report.md"))
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()

    if args.cmd == "train":
        splits = _prepare_dataset()
        res = _train_eval(splits)
        # 写报告
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# LR 二分类器评估", ""]
        for split in ("dev", "test"):
            r = res[split]
            lines += [f"## {split}", f"- accuracy: **{r['acc']:.4f}**",
                      f"- F1 (cooking): **{r['f1']:.4f}**",
                      f"- P/R ood: **{r['p_ood']:.4f}** / **{r['r_ood']:.4f}**",
                      f"- P/R cooking: **{r['p_cooking']:.4f}** / **{r['r_cooking']:.4f}**",
                      "```", r["report"], "```", ""]
        lines += ["## Top 30 cooking 强信号词", "",
                  "| 词 | 权重 |", "|---|---|"]
        for w, c in res["top_cooking_words"]:
            lines.append(f"| {w} | {c:.3f} |")
        lines += ["", "## Top 30 ood 强信号词", "",
                  "| 词 | 权重 |", "|---|---|"]
        for w, c in res["top_ood_words"]:
            lines.append(f"| {w} | {c:.3f} |")
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"训练报告 → {out}")
    elif args.cmd == "predict":
        clf = LRFilter(args.model, args.tfidf)
        if args.text:
            p = clf.cooking_proba([args.text])[0]
            print(json.dumps({"text": args.text, "cooking_proba": float(p),
                              "label": "cooking" if p >= 0.5 else "ood"},
                             ensure_ascii=False, indent=2))
        elif args.file:
            items = _load_jsonl(Path(args.file))
            probs = clf.cooking_proba([it["text"] for it in items])
            for it, p in zip(items, probs):
                print(json.dumps({**it, "cooking_proba": float(p),
                                  "pred": "cooking" if p >= 0.5 else "ood"},
                                 ensure_ascii=False))


if __name__ == "__main__":
    main()
