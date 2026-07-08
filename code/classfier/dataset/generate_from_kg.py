"""
基于 parse_dishes.py 输出的 dishes_parsed.jsonl，
按 4 类模板批量生成 query（每道菜每个类别生成若干条）。

模板策略：
  general    : 菜名 + 概念性动词
  detail     : 菜名 + 食材/步骤 + 数字疑问
  multi-hop  : 多实体组合（两菜 / 菜+食材 / 类+场景 / 食材搭配）
  recommend  : 类别 + 场景 / 关键词推荐

输出：data/synthetic/template_*.jsonl  (4 个文件，每个一类)
字段：{text, label, source, dish_refs?, source_dish?}
"""
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Iterable

from classfier.config import (
    RAW_DIR, SYNTHETIC_DIR, SCENES, LOG_LEVEL, TEMPLATE_PER_DISH
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_from_kg")

random.seed(42)


# ===================== 模板 =====================

GENERAL_TEMPLATES = [
    "{dish}怎么做",
    "介绍一下{dish}",
    "{dish}是哪里的菜",
    "教我做{dish}",
    "讲讲{dish}",
    "{dish}是啥",
    "什么是{dish}",
    "{category}里有什么经典菜",
    "{category}有什么特点",
]

DETAIL_TEMPLATES_FROM_INGREDIENT = [
    "{dish}中{ingredient}放多少",
    "{dish}要加多少{ingredient}",
    "做{dish}时{ingredient}的比例是多少",
    "{dish}里的{ingredient}能用别的替代吗",
    "{dish}可以不放{ingredient}吗",
    "没有{ingredient}怎么做{dish}",
]

DETAIL_TEMPLATES_FROM_STEP = [
    "{dish}要{step}",
    "做{dish}{step}",
    "{dish}{step}才好吃吗",
    "{dish}为什么需要{step}",
    "第几步要{step}",
]

MULTIHOP_TEMPLATES = [
    # 多菜组合
    "{dish_a}和{dish_b}的区别",
    "{dish_a}和{dish_b}哪个好吃",
    "{dish_a}和{dish_b}哪个适合{s}",
    # 菜 + 食材
    "用{ing}可以做哪些{category}",
    "有{ing}的{category}有哪些",
    "用{ing}和{ing2}能做什么菜",
    # 类别 + 场景
    "{s}适合吃什么{category}",
    "{s}有哪些{category}",
    "减脂期适合吃哪些{category}",
    "新手适合做什么{category}",
    "不辣的{category}有哪些",
]

RECOMMEND_TEMPLATES = [
    "推荐一道{category}",
    "介绍几道{category}",
    "有什么{s}的{category}",
    "推荐一道{s}吃的{category}",
    "来一道{s}的{category}",
    "适合{s}的{category}有哪些推荐",
    "求推荐好吃的{category}",
    "求几道简单的{category}",
]


# ===================== 生成器 =====================

def _load_dishes() -> List[Dict]:
    src = RAW_DIR / "dishes_parsed.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"请先运行 parse_dishes.py: {src}")
    return [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]


def _format(template: str, **kwargs) -> str:
    """安全 format，缺字段返回空串（让外层丢弃）"""
    try:
        return template.format(**kwargs)
    except KeyError:
        return ""


def gen_general(dishes: List[Dict]) -> Iterable[Dict]:
    for d in dishes:
        dish = d["dish"]
        category = d["dir_category"]
        for _ in range(TEMPLATE_PER_DISH):
            t = random.choice(GENERAL_TEMPLATES)
            text = _format(t, dish=dish, category=category)
            if text and 5 <= len(text) <= 60:
                yield {"text": text, "label": "general",
                       "source": "template", "source_dish": dish}


def gen_detail(dishes: List[Dict]) -> Iterable[Dict]:
    for d in dishes:
        dish = d["dish"]
        ingredients = (d.get("main_ingredients") or []) + (d.get("aux_ingredients") or [])
        steps = d.get("steps_with_number") or []
        for _ in range(TEMPLATE_PER_DISH):
            # 优先用 step，其次用 ingredient
            if steps and random.random() < 0.5:
                t = random.choice(DETAIL_TEMPLATES_FROM_STEP)
                text = _format(t, dish=dish, step=random.choice(steps))
            elif ingredients:
                t = random.choice(DETAIL_TEMPLATES_FROM_INGREDIENT)
                text = _format(t, dish=dish, ingredient=random.choice(ingredients))
            else:
                continue
            if text and 5 <= len(text) <= 60:
                yield {"text": text, "label": "detail",
                       "source": "template", "source_dish": dish}


def gen_multihop(dishes: List[Dict]) -> Iterable[Dict]:
    """构造多实体组合。多用「category + 场景/食材」模板，少用「菜+菜」避免和 recommend 撞。"""
    categories = list({d["dir_category"] for d in dishes if d["dir_category"] != "未分类"})
    all_ingredients: List[str] = []
    for d in dishes:
        all_ingredients.extend(d.get("main_ingredients") or [])
    all_ingredients = [i for i in all_ingredients if i and len(i) <= 6]

    for _ in range(len(dishes) * TEMPLATE_PER_DISH):
        t = random.choice(MULTIHOP_TEMPLATES)
        s = random.choice(SCENES)
        category = random.choice(categories)
        if "{dish_a}" in t and "{dish_b}" in t:
            a, b = random.sample(dishes, 2)
            text = _format(t, dish_a=a["dish"], dish_b=b["dish"], s=s, category=category)
        elif "{ing2}" in t:
            if len(all_ingredients) < 2:
                continue
            ing1, ing2 = random.sample(all_ingredients, 2)
            text = _format(t, ing=ing1, ing2=ing2, s=s, category=category)
        elif "{ing}" in t:
            ing = random.choice(all_ingredients) if all_ingredients else ""
            text = _format(t, ing=ing, s=s, category=category)
        else:
            text = _format(t, s=s, category=category)
        if text and 5 <= len(text) <= 60:
            yield {"text": text, "label": "multi-hop",
                   "source": "template"}


def gen_recommend(dishes: List[Dict]) -> Iterable[Dict]:
    categories = list({d["dir_category"] for d in dishes if d["dir_category"] != "未分类"})
    for _ in range(len(dishes) * TEMPLATE_PER_DISH):
        t = random.choice(RECOMMEND_TEMPLATES)
        s = random.choice(SCENES)
        category = random.choice(categories)
        text = _format(t, s=s, category=category)
        if text and 5 <= len(text) <= 60:
            yield {"text": text, "label": "recommend",
                   "source": "template"}


GENERATORS = {
    "general": gen_general,
    "detail": gen_detail,
    "multi-hop": gen_multihop,
    "recommend": gen_recommend,
}


def main():
    dishes = _load_dishes()
    logger.info(f"载入 {len(dishes)} 道菜谱")

    summary = {}
    for label, gen in GENERATORS.items():
        out = list(gen(dishes))
        # 类内去重
        seen, uniq = set(), []
        for x in out:
            if x["text"] not in seen:
                seen.add(x["text"])
                uniq.append(x)
        out_path = SYNTHETIC_DIR / f"template_{label}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for x in uniq:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        summary[label] = len(uniq)
        logger.info(f"{label:10s} 模板生成 {len(uniq)} 条 → {out_path.name}")

    logger.info(f"汇总: {summary}")


if __name__ == "__main__":
    main()
