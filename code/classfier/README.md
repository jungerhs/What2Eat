# classfier — 烹饪 query 4 分类数据集构建

把 `data/C9/dishes/` 下的菜谱结构化 + Kimi 合成 + 模板 + 二次打标，最终产出
`train/dev/test.jsonl` 用于后续 `chinese-roberta-wwm-ext` 全量微调。

## 4 类

`general` | `detail` | `multi-hop` | `recommend` — 详细定义见
[`dataset/label_guidelines.md`](dataset/label_guidelines.md)。

## 目录

```
classfier/
├── config.py                 # 路径/标签/目标量/超参
├── requirements.txt
├── dataset/
│   ├── label_guidelines.md   # 4 类契约（必读）
│   ├── parse_dishes.py       # 解析 .md → data/raw/dishes_parsed.jsonl
│   ├── generate_from_kg.py   # 4 类模板生成 → data/synthetic/template_*.jsonl
│   ├── generate_by_llm.py    # Kimi 批量合成   → data/synthetic/llm_*.jsonl
│   ├── double_check.py       # 二次打标 + 争议集 → data/processed/{agreed,disputed}.jsonl
│   ├── clean_and_split.py    # 清洗 + 去重 + 8:1:1 切分
│   └── stats.py              # 统计报告
├── data/
│   ├── raw/                  # 菜谱结构化
│   ├── synthetic/            # 模板 + LLM 合成
│   └── processed/            # 清洗/切分最终产物
├── reports/
│   └── dataset_stats.md      # 自动生成
└── run_all.py                # 一键流程
```

## 环境

```powershell
# 0) 安装依赖
pip install -r code/classfier/requirements.txt

# 1) Kimi (Moonshot) 兼容 OpenAI SDK 的环境变量
$env:API_KEY   = "sk-..."          # 也可写 MOONSHOT_API_KEY
$env:BASE_URL  = "https://api.moonshot.cn/v1"
$env:LLM_MODEL = "kimi-k2-0711-preview"
```

## 跑法

### 一键全流程（含 LLM 合成）

```powershell
cd f:\cook项目\cook
python code/classfier/run_all.py
```

### 一键流程（跳过 LLM，只跑模板）

适合网络不可用 / token 紧张时验证流程：

```powershell
python code/classfier/run_all.py --skip-llm
```

### 分步跑

```powershell
# 1. 解析菜谱
python -m classfier.dataset.parse_dishes

# 2. 4 类模板生成
python -m classfier.dataset.generate_from_kg

# 3. Kimi 合成（每类 2000 条，约 4 轮 × 50 条/批，并发 8）
python -m classfier.dataset.generate_by_llm --n 2000

# 4. 二次打标
python -m classfier.dataset.double_check

# 5. 清洗 + 切分
python -m classfier.dataset.clean_and_split

# 6. 统计
python -m classfier.dataset.stats
```

## 输出

```
data/raw/dishes_parsed.jsonl              # 菜谱结构化
data/synthetic/template_{label}.jsonl     # 4 个文件，模板生成
data/synthetic/llm_{label}.jsonl          # 4 个文件，LLM 合成
data/processed/agreed.jsonl               # 二次打标一致样本
data/processed/disputed.jsonl             # 二次打标争议样本（待人工/规则兜底）
data/processed/all.jsonl                  # 合并去重后
data/processed/train.jsonl                # 8
data/processed/dev.jsonl                  # 1
data/processed/test.jsonl                 # 1
data/processed/label_map.json             # {"general":0, "detail":1, ...}
data/processed/out_of_domain.jsonl        # 鲁棒性测试
reports/dataset_stats.md
```

## 数据格式

每条 jsonl：

```json
{"text": "红烧肉要炖多久", "label": "detail", "source": "llm"}
```

`source ∈ {template, llm, agreed, manual}`，训练时建议只取 `template / llm / agreed`。

## 质量门槛

`double_check_report.json` 的 `agreement` 字段 ≥ **0.85** 才算合格；
不达标时优先检查 `disputed.jsonl` 中最多被改判的类别，补充 few-shot。

## 下一步

数据集就绪后，参见后续 `training/` 目录训练 `chinese-roberta-wwm-ext`。
