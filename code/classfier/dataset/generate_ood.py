"""
用 LLM 批量合成 OOD 负样本（不是烹饪问题的 query），
用于训练 LR 二分类器（烹饪 / 非烹饪）。

输出：data/lr_filter/ood_synth.jsonl
字段：{text, label, source}
"""
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classfier.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LR_TARGET_NEG, LLM_BATCH_SIZE, LLM_CONCURRENCY, LLM_TEMPERATURE,
    LR_DATA_DIR, LOG_LEVEL, CLASSFIER_ROOT,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_ood")

GUIDELINES_PATH = CLASSFIER_ROOT / "dataset" / "label_guidelines.md"

# 主题清单：让 LLM 在这些主题上发散生成 OOD query
OOD_THEMES = [
    "科技 (AI/编程/手机/电脑/互联网)",
    "金融 (股票/基金/理财/汇率)",
    "医疗 (药品/症状/医院/养生)",
    "教育 (学习方法/考试/选课/留学)",
    "娱乐 (游戏/电影/音乐/追星)",
    "天气 (预报/穿衣/出行)",
    "旅游 (景点/交通/攻略)",
    "职场 (简历/面试/升职/跳槽)",
    "汽车 (买车/保养/驾驶/路况)",
    "宠物 (猫/狗/鸟/鱼)",
    "运动 (健身/跑步/球类)",
    "情感 (恋爱/婚姻/家庭)",
    "法律 (合同/纠纷/权益)",
    "时事 (新闻/政治/国际)",
    "生活 (水电/租房/搬家/快递)",
    "数码 (相机/耳机/平板/智能家居)",
    "文化 (历史/哲学/文学/艺术)",
    "自然 (天文/地理/动物/植物)",
    "社交 (聊天/朋友/聚会/礼仪)",
    "购物 (优惠/比价/海淘/退货)",
]


SYSTEM_PROMPT = """你是中文 query 数据合成专家。任务：合成**与烹饪无关**的中文 query。
要求：
1. 严格只输出 JSON 数组，不要任何解释、Markdown 包装、前后缀。
2. 每条 query 长度 5~40 字，贴近真实用户口吻。
3. 主题多样化，覆盖：科技、金融、医疗、教育、娱乐、天气、旅游、职场、宠物、运动、情感等。
4. 绝对不要包含任何与"菜/做菜/食材/调味/煮/炒/烤/红烧/川菜/汤/饭"相关的内容。
5. 句式要自然口语化，不要模板化。
"""


def build_user_prompt(theme: str, n: int) -> str:
    return f"""# 任务
请围绕主题 **{theme}** 合成 **{n} 条**中文 query。
query 应当是真实用户会问的问题（如"今天天气怎么样"、"股票怎么入门"）。

严格禁止：出现"菜/做菜/食材/调味/红烧/川菜/汤/粥/烤/炒/煮"等烹饪相关词。

# 输出（严格 JSON 数组）
[{{"text": "...", "label": "ood", "theme": "{theme}"}}]

只输出 JSON。"""


def get_client() -> OpenAI:
    if not LLM_API_KEY:
        raise ValueError("未配置 API_KEY / MOONSHOT_API_KEY")
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _strip_codeblock(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_array(text: str) -> Optional[List[Dict]]:
    text = _strip_codeblock(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        s = text.find("[")
        e = text.rfind("]")
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(data, list):
        return None
    return data


# 烹饪相关词（合成后过滤，防止 LLM 跑偏）
COOKING_LEAK_RE = re.compile(
    r"(做菜|菜谱|食材|调味|红烧|川菜|鲁菜|粤菜|湘菜|清蒸|水煮|油焖|糖醋|"
    r"爆炒|小炒|煎|炸|烤|炖|焖|煮|蒸|煸|卤|凉拌|主料|辅料|食谱|烹饪|"
    r"汤|粥|米饭|面条|饺子|包子|馒头|寿司|沙拉|三明治|蛋糕|面包|饼干|"
    r"红烧肉|麻婆豆腐|宫保鸡丁|鱼香肉丝|西红柿炒蛋|回锅肉|东坡肉|梅菜扣肉|"
    r"红烧排骨|糖醋里脊|红烧鱼|酸辣土豆丝|地三鲜|蚝油生菜|手撕包菜|"
    r"酸辣汤|西红柿鸡蛋|炒饭|炒面|蛋炒饭|扬州炒饭|蛋花汤|"
    r"番茄|鸡蛋|豆腐|土豆|胡萝卜|白菜|青椒|肉|鸡|鱼|虾|蟹|猪|牛|羊|"
    r"盐|糖|酱油|醋|料酒|花椒|八角|香叶|姜|葱|蒜|辣椒|胡椒|味精|蚝油)",
    re.IGNORECASE,
)


def call_one_batch(client: OpenAI, theme: str, n: int, max_retry: int = 3) -> List[Dict]:
    prompt = build_user_prompt(theme, n)
    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=min(4096, n * 60),
            )
            content = resp.choices[0].message.content or ""
            data = _parse_json_array(content)
            if data is None:
                logger.warning(f"[{theme}] 第 {attempt} 次 JSON 解析失败: {content[:200]!r}")
                continue
            out = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                text = (item.get("text") or "").strip()
                if not (5 <= len(text) <= 60):
                    continue
                # 过滤烹饪泄漏
                if COOKING_LEAK_RE.search(text):
                    continue
                out.append({"text": text, "label": "ood", "source": "llm", "theme": theme})
            return out
        except Exception as e:
            logger.warning(f"[{theme}] 第 {attempt} 次调用异常: {e}")
            time.sleep(2 ** attempt)
    return []


def main():
    LR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LR_DATA_DIR / "ood_synth.jsonl"

    client = get_client()
    logger.info(f"LLM 模型: {LLM_MODEL}, base_url: {LLM_BASE_URL}")
    logger.info(f"目标: {LR_TARGET_NEG} 条 OOD；主题数: {len(OOD_THEMES)}")

    # 按主题分摊：每个主题生成 n_per_theme 条
    n_per_theme = max(10, LR_TARGET_NEG // len(OOD_THEMES) + 1)

    collected: List[Dict] = []
    seen: set = set()

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as ex:
        # 同时提交所有主题
        futs = {ex.submit(call_one_batch, client, t, n_per_theme): t for t in OOD_THEMES}
        for fut in as_completed(futs):
            theme = futs[fut]
            items = fut.result()
            added = 0
            for it in items:
                if it["text"] in seen:
                    continue
                seen.add(it["text"])
                collected.append(it)
                added += 1
            logger.info(f"主题 [{theme}] 新增 {added} / 本批 {len(items)}，累计 {len(collected)}")
            if len(collected) >= LR_TARGET_NEG * 1.2:
                break

    # 截断到目标
    random.seed(42)
    random.shuffle(collected)
    final = collected[:LR_TARGET_NEG]

    with out_path.open("w", encoding="utf-8") as f:
        for x in final:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    logger.info(f"OOD 落盘 {len(final)} 条 → {out_path}")


if __name__ == "__main__":
    main()
