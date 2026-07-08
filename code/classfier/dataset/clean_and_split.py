"""
合并所有数据源 (template_*.jsonl + llm_*.jsonl + agreed.jsonl)，
清洗、SimHash 去重、长度过滤、OOD 拼接、8:1:1 分层切分。

输出：
  data/processed/all.jsonl
  data/processed/train.jsonl
  data/processed/dev.jsonl
  data/processed/test.jsonl
  data/processed/label_map.json
  data/processed/out_of_domain.jsonl
"""
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Iterable

try:
    from simhash import Simhash as _ExtSimhash
    _HAS_SIMHASH = True
except ImportError:  # 第三方 simhash 未装时使用内置实现
    _HAS_SIMHASH = False


def _builtin_simhash(text: str) -> int:
    """轻量 SimHash: 字符 2-gram + md5 → 64 位指纹。
    足够用于 ~10w 量级内的去重，不依赖第三方包。"""
    import hashlib
    tokens = [text[i:i + 2] for i in range(len(text) - 1)] or [text]
    vec = [0] * 64
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(64):
            vec[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, v in enumerate(vec):
        if v > 0:
            out |= (1 << i)
    return out


def _simhash(text: str) -> int:
    if _HAS_SIMHASH:
        from simhash import Simhash
        return Simhash([text[i:i + 2] for i in range(len(text) - 1)] or [text]).value
    return _builtin_simhash(text)
from classfier.config import (
    SYNTHETIC_DIR, PROCESSED_DIR, LABEL_TO_ID, LABELS,
    MIN_LEN, MAX_LEN, SIMHASH_HAMMING_THRESHOLD,
    TRAIN_RATIO, DEV_RATIO, TEST_RATIO, LOG_LEVEL,
    QM_RATIO,  # cooking 训练样本尾部补问号比例
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_and_split")

random.seed(42)

# ===================== 来源合并 =====================

def _iter_source_files() -> Iterable[Path]:
    for p in sorted(SYNTHETIC_DIR.glob("template_*.jsonl")):
        yield p
    for p in sorted(SYNTHETIC_DIR.glob("llm_*.jsonl")):
        yield p
    p = PROCESSED_DIR / "agreed.jsonl"
    if p.exists():
        yield p


def load_all() -> List[Dict]:
    items = []
    for p in _iter_source_files():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj.setdefault("source", p.stem)
            items.append(obj)
    return items


# ===================== 清洗 =====================

FORBIDDEN_RE = re.compile(
    r"(\bhttps?://|色情|赌博|毒品|暴力|恐怖袭击|邪教|反动)", re.IGNORECASE
)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # 去尾部标点（保留尾部问号，避免 LR 学到 "问号 ⇒ OOD" 的虚假偏置）
    text = re.sub(r"[，。！、,!]+$", "", text)
    return text


def _passes_filter(text: str) -> bool:
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return False
    if FORBIDDEN_RE.search(text):
        return False
    # 必须含至少一个中文字符
    if not re.search(r"[\u4e00-\u9fa5]", text):
        return False
    return True


def _simhash(text: str) -> int:  # noqa: F811
    return _builtin_simhash(text)


def dedup(items: List[Dict]) -> List[Dict]:
    seen_hashes = []
    out = []
    for it in items:
        h = _simhash(it["text"])
        is_dup = False
        for sh in seen_hashes:
            # 汉明距离
            xor = h ^ sh
            dist = bin(xor).count("1")
            if dist <= SIMHASH_HAMMING_THRESHOLD:
                is_dup = True
                break
        if is_dup:
            continue
        seen_hashes.append(h)
        out.append(it)
    return out


# ===================== OOD 数据 =====================

OOD_QUERIES = [
    "今天天气怎么样",
    "上海到北京多少公里",
    "推荐一部好看的电影",
    "感冒了吃什么药",
    "王者荣耀怎么上分",
    "Python 怎么读取 JSON",
    "北京有什么好玩的地方",
    "如何学习英语",
    "明天会下雨吗",
    "我该买什么基金",
    "马斯克是谁",
    "怎么做自我介绍",
    "杭州西湖在哪里",
    "什么是量子计算",
    "为什么天空是蓝色的",
    "怎么写简历",
    "求一个笑话",
    "适合发朋友圈的句子",
    "股票怎么入门",
    "猫和狗哪个好养",
]


# ===================== 数据增强：尾部补问号 =====================

def _maybe_add_question_mark(items: List[Dict], ratio: float) -> List[Dict]:
    """对 cooking 4 类样本按 `ratio` 概率在尾部补 "？"，使 cooking 侧问号分布
    与真实用户输入对齐，避免 LR 把 "？" 学成 OOD 强偏置。
    - OOD / 已在尾部带问号的样本不动
    - 用 fixed seed 以保证可复现"""
    if ratio <= 0:
        return items
    rng = random.Random(42)
    out = []
    n_added = 0
    for it in items:
        if it.get("label") == "ood":
            out.append(it)
            continue
        text = it["text"]
        if text.endswith("?") or text.endswith("？"):
            out.append(it)
            continue
        if rng.random() < ratio:
            out.append({**it, "text": text + "？"})
            n_added += 1
        else:
            out.append(it)
    logger.info(f"尾部补问号: 共 {n_added} 条 (cooking 比例={ratio*100:.0f}%)")
    return out


# ===================== 切分 =====================

def stratified_split(items: List[Dict]) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = {lbl: [] for lbl in LABELS}
    for it in items:
        buckets[it["label"]].append(it)
    split = {"train": [], "dev": [], "test": []}
    for lbl, lst in buckets.items():
        random.shuffle(lst)
        n = len(lst)
        n_train = int(n * TRAIN_RATIO)
        n_dev = int(n * DEV_RATIO)
        split["train"].extend(lst[:n_train])
        split["dev"].extend(lst[n_train:n_train + n_dev])
        split["test"].extend(lst[n_train + n_dev:])
    for k in split:
        random.shuffle(split[k])
    return split


# ===================== 入口 =====================

def main():
    raw = load_all()
    logger.info(f"原始载入 {len(raw)} 条")

    # 1) 清洗文本
    cleaned = []
    for it in raw:
        text = _clean_text(it.get("text", ""))
        if not _passes_filter(text):
            continue
        if it.get("label") not in LABELS:
            continue
        cleaned.append({**it, "text": text})
    logger.info(f"过滤后 {len(cleaned)} 条")

    # 2) 去重
    deduped = dedup(cleaned)
    logger.info(f"SimHash 去重后 {len(deduped)} 条")

    # 2.5) 数据增强：cooking 样本按 QM_RATIO 概率尾部补 "？"
    augmented = _maybe_add_question_mark(deduped, QM_RATIO)

    # 3) 统计
    cnt = Counter(x["label"] for x in deduped)
    logger.info(f"分类分布: {dict(cnt)}")

    # 4) 落盘 all.jsonl
    (PROCESSED_DIR / "all.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in augmented) + "\n",
        encoding="utf-8",
    )

    # 5) 切分
    splits = stratified_split(augmented)
    for k, lst in splits.items():
        out = PROCESSED_DIR / f"{k}.jsonl"
        out.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in lst) + "\n",
            encoding="utf-8",
        )
        dist = Counter(x["label"] for x in lst)
        logger.info(f"{k:5s} {len(lst):5d} 条  分布={dict(dist)}")

    # 6) label_map
    (PROCESSED_DIR / "label_map.json").write_text(
        json.dumps(LABEL_TO_ID, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 7) OOD
    ood = [{"text": t, "label": "ood", "source": "manual"} for t in OOD_QUERIES]
    (PROCESSED_DIR / "out_of_domain.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in ood) + "\n",
        encoding="utf-8",
    )
    logger.info(f"OOD 落盘 {len(ood)} 条")

    logger.info("完成。")


if __name__ == "__main__":
    main()
