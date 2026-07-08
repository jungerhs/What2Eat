"""
解析 data/C9/dishes/ 下的菜谱 markdown，提取结构化字段。
扫描规则：递归遍历，跳过 SKIP_DIRS；每个 .md 视为一道菜（菜名 = 文件名去后缀）。

输出：data/raw/dishes_parsed.jsonl
每行一条菜谱：
{
  "md_path": "data/C9/dishes/meat_dish/红烧肉/简易红烧肉.md",
  "dish": "简易红烧肉",
  "dir_category": "荤菜",
  "main_ingredients": ["大肉", "鸡蛋", "豆皮"],
  "aux_ingredients": ["生姜", "冰糖", "生抽", ...],
  "ingredient_amounts": [
     {"name": "猪五花肉", "amount": "约 3~4 斤"},
     ...
  ],
  "steps_with_number": [
     "煮 15 分钟去掉血腥",
     "炖煮 40 分钟",
     ...
  ],
  "difficulty": 3,
  "calories": 2775
}
"""
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Iterable, Optional

from classfier.config import (
    DISHES_DIR, RAW_DIR, SKIP_DIRS, DIR_TO_CATEGORY, LOG_LEVEL
)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("parse_dishes")


# ===================== 数据结构 =====================

@dataclass
class DishRecord:
    md_path: str
    dish: str
    dir_category: str
    main_ingredients: List[str] = field(default_factory=list)
    aux_ingredients: List[str] = field(default_factory=list)
    ingredient_amounts: List[Dict[str, str]] = field(default_factory=list)
    steps_with_number: List[str] = field(default_factory=list)
    difficulty: Optional[int] = None
    calories: Optional[int] = None


# ===================== 解析逻辑 =====================

RE_TITLE = re.compile(r"^#\s+(.+?)(?:的做法)?$", re.MULTILINE)
RE_DIFFICULTY = re.compile(r"预估烹饪难度[：:]\s*(★+)")
RE_CALORIES = re.compile(r"预估卡路里[：:]\s*(\d+)")
RE_MAIN_LINE = re.compile(r"主料[：:]\s*`([^`]+)`")
RE_AUX_LINE = re.compile(r"辅料[：:]\s*`([^`]+)`")
RE_INGREDIENT_NAME = re.compile(r"[`「]?([\u4e00-\u9fa5A-Za-z0-9]+)[`]?")
RE_NUMERIC_STEP = re.compile(r"(\d+\s*(?:分钟|克|ml|毫升|片|块|个|勺|大勺|小勺|厘米|cm|°C|度|把|颗|根|瓣))", re.IGNORECASE)
RE_PURE_NUMBER = re.compile(r"\d+")

# 工具/非食材的关键词（行首为 - 时过滤）
NON_INGREDIENT_KEYWORDS = {
    "工具", "可选", "家庭", "小陶瓷碗", "陶瓷碗", "铁勺子", "量杯",
    "厨房秤", "厨房", "微波炉", "烤箱", "不粘锅", "电饭煲", "蒸锅",
    "刀", "砧板", "锅", "勺", "碗", "叉子", "牙签", "筷子", "盆",
    "盘子", "玻璃碗", "密封盒", "密封容器", "可密封容器", "案板",
    "保鲜膜", "烘焙纸", "锡纸", "烤盘", "模具", "打蛋器", "电动打蛋器",
    "搅拌机", "料理机", "研磨器", "削皮刀", "刨丝器", "滤网", "筛子",
    "厨房纸", "吸油纸", "油纸", "模具", "花边",
}


def _split_ingredient_line(line: str) -> List[str]:
    """把 '`大肉`、`鸡蛋`（可选）、`豆皮`（可选）' 切成 ['大肉', '鸡蛋', '豆皮']"""
    parts = re.split(r"[、，,；;]", line)
    result = []
    for p in parts:
        # 提取 `` 内的内容
        m = re.search(r"[\`'\"\u300c\u300d]([^\u3001\u3002\u300a\u300b\`\'\"]+)[\`\'\"\u300c\u300d]", p)
        if m:
            name = m.group(1).strip()
        else:
            # 兜底：去掉括号和空白
            name = re.sub(r"[（(][^）)]*[）)]", "", p).strip()
        if name and len(name) <= 12:
            result.append(name)
    return result


def _parse_ingredients_from_list(text: str) -> List[str]:
    """从 '## 必备原料和工具' 段的 - 列表行提取食材（兜底）。"""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        item = line.lstrip("-").strip()
        # 过滤工具/可选
        if any(kw in item for kw in NON_INGREDIENT_KEYWORDS):
            continue
        # 过滤明显是工具的（极短且是单字名词）
        if len(item) <= 1:
            continue
        # 去掉 (xxx) / （xxx） 注释
        item = re.sub(r"[（(][^）)]*[）)]", "", item).strip()
        # 名字里不能含空格（食材一般是 1~6 字）
        if " " in item and len(item) > 6:
            continue
        # 去掉数字开头的（如 "500g 巴沙鱼"）
        item = re.sub(r"^\d+(?:\.\d+)?\s*(?:g|ml|克|毫升|kg|斤|个|片|块|根|把|颗|瓣|勺|杯|把|份|条|粒|朵|头|块|只|条|袋|包|盒|瓶|罐|支|根|束|份|份|份)?\s*", "", item).strip()
        if 1 <= len(item) <= 8 and re.search(r"[\u4e00-\u9fa5]", item):
            out.append(item)
    # 去重保序
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _parse_amounts(text: str) -> List[Dict[str, str]]:
    """从 ## 计算 段提取 '猪五花肉：约 3~4 斤' 这种行"""
    amounts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "：" not in line and ":" not in line:
            continue
        # 拆 name : amount
        sep = "：" if "：" in line else ":"
        left, right = line.split(sep, 1)
        name = left.strip().lstrip("-").strip()
        amount = right.strip()
        if not name or not amount:
            continue
        if not RE_PURE_NUMBER.search(amount) and not any(c in amount for c in "适量少许"):
            continue
        amounts.append({"name": name, "amount": amount})
    return amounts


def _parse_steps_with_number(text: str) -> List[str]:
    """从 ## 操作 段提取含数字量/时间/温度的步骤短句"""
    steps = []
    # 拆行 + 拆编号
    raw_lines = re.split(r"\n", text)
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # 去掉编号前缀 "1. " "2、"
        line = re.sub(r"^\d+[\.\u3001]\s*", "", line)
        # 提取含数字量的小句
        if RE_NUMERIC_STEP.search(line):
            # 截取含数字的最短子句
            m = RE_NUMERIC_STEP.search(line)
            if m:
                # 找最近的标点范围
                start = max(0, m.start() - 6)
                end = min(len(line), m.end() + 4)
                snippet = line[start:end].strip(" ，,。.；;`")
                if 4 <= len(snippet) <= 30:
                    steps.append(snippet)
    # 去重
    seen = set()
    out = []
    for s in steps:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:10]  # 每道菜最多保留 10 步


def _parse_dish_name(md_path: Path, content: str) -> str:
    """从标题或文件名获取菜名"""
    m = RE_TITLE.search(content)
    if m:
        title = m.group(1).strip()
        # 去掉 "的做法"
        title = re.sub(r"(的做法|怎么做)$", "", title)
        return title
    return md_path.stem


def _parse_category(md_path: Path) -> str:
    """从父目录名映射中文分类"""
    parent_name = md_path.parent.name
    if parent_name in DIR_TO_CATEGORY:
        return DIR_TO_CATEGORY[parent_name]
    # 单文件情况：菜直接放在子分类下，取祖父目录
    if md_path.parent.parent.name in DIR_TO_CATEGORY:
        return DIR_TO_CATEGORY[md_path.parent.parent.name]
    return "未分类"


def parse_one(md_path: Path) -> Optional[DishRecord]:
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"读取失败 {md_path}: {e}")
        return None

    rec = DishRecord(
        md_path=str(md_path),
        dish=_parse_dish_name(md_path, content),
        dir_category=_parse_category(md_path),
    )

    # 难度 / 卡路里
    m = RE_DIFFICULTY.search(content)
    if m:
        rec.difficulty = len(m.group(1))
    m = RE_CALORIES.search(content)
    if m:
        rec.calories = int(m.group(1))

    # 主料 / 辅料（内联格式）；若没匹配，兜底从必备原料列表行提取
    m = RE_MAIN_LINE.search(content)
    if m:
        rec.main_ingredients = _split_ingredient_line(m.group(1))
    m = RE_AUX_LINE.search(content)
    if m:
        rec.aux_ingredients = _split_ingredient_line(m.group(1))
    if not rec.main_ingredients and not rec.aux_ingredients:
        ing_match = re.search(r"##\s*必备原料和工具\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
        if ing_match:
            items = _parse_ingredients_from_list(ing_match.group(1))
            rec.main_ingredients = items[:5]
            rec.aux_ingredients = items[5:10]

    # ## 计算 段
    calc_match = re.search(r"##\s*计算\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
    if calc_match:
        rec.ingredient_amounts = _parse_amounts(calc_match.group(1))

    # ## 操作 段
    op_match = re.search(r"##\s*操作\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
    if op_match:
        rec.steps_with_number = _parse_steps_with_number(op_match.group(1))

    # 过滤：完全没食材也没用量也没步骤视为解析失败
    if (not rec.main_ingredients and not rec.aux_ingredients
            and not rec.ingredient_amounts and not rec.steps_with_number):
        return None

    return rec


def iter_md_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.md"):
        # 跳过指定目录
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def main():
    out_path = RAW_DIR / "dishes_parsed.jsonl"
    count_total, count_ok, count_fail = 0, 0, 0
    category_counter: Dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for md_path in iter_md_files(DISHES_DIR):
            count_total += 1
            rec = parse_one(md_path)
            if rec is None:
                count_fail += 1
                continue
            count_ok += 1
            category_counter[rec.dir_category] = category_counter.get(rec.dir_category, 0) + 1
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    logger.info(f"扫描 {count_total} 个 .md，成功 {count_ok}，失败 {count_fail}")
    logger.info(f"分类分布: {category_counter}")
    logger.info(f"输出: {out_path}")


if __name__ == "__main__":
    main()
