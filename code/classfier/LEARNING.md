# 意图识别 Pipeline 学习文档

> 本文档面向想从 0 理解 / 复现 / 改进本仓库意图识别 pipeline 的开发者。
> 阅读完应该能：(1) 画出整个 pipeline 数据流图；(2) 解释每个阶段为什么存在；(3) 独立跑一次训练 + 评估；(4) 知道在哪儿调优。

---

## 0. 阅读路径

按顺序读，1~2 小时可读完：

1. **第 1~2 节**：项目背景 + 架构总览（30 min，建立心智模型）
2. **第 3 节**：数据流（30 min，理解数据怎么来）
3. **第 4 节**：各阶段详解（45 min，每阶段独立小节）
4. **第 5 节**：评估方法（15 min）
5. **第 6~7 节**：代码导读 + 复现命令（20 min）
6. **第 8~10 节**：性能数据 + 故障排查 + 进阶（按需）

---

## 1. 项目背景

### 1.1 C9（尝尝咸淡）是什么

C9 是一个**中文烹饪知识图谱 RAG 系统**：

```
用户问"红烧肉要炖多久"  →  RAG 回答
                              ↑
        Neo4j (知识图谱) + Milvus (向量) 联合检索
```

- 数据源：`HowToCook-master` 仓库的 350 个 Markdown 菜谱
- 上层应用：用户在交互式 CLI 里问烹饪问题，系统给出回答

### 1.2 为什么需要"意图识别"

RAG 系统**不能对所有 query 都用同一种检索策略**。根据 query 意图，应该走不同路径：

| 用户问 | 意图 | 应该走的路径 |
|---|---|---|
| "红烧肉要炖多久" | **detail**（细节问）| 知识图谱精确查 1 个节点 → 走 deepseek LLM 生成 |
| "用鸡蛋和番茄可以做什么菜" | **multi-hop**（多实体）| 知识图谱多跳遍历 / 子图抽取 |
| "推荐几道川菜" | **recommend**（推荐）| 向量检索 + 排序 |
| "川菜是什么菜系" | **general**（概念）| 走 LLM 通用回答 |
| "量子计算是什么" | **非烹饪 OOD** | 直接拒识 / 转 LLM 通用对话 |

**意图识别的作用**：把 query 路由到合适的下游处理模块，避免 LLM 在错误数据源上做错误推理。

### 1.3 4 类意图定义

详见 [`dataset/label_guidelines.md`](dataset/label_guidelines.md)，核心判别规则：

| 类别 | 强信号词 | 例子 |
|---|---|---|
| **general** | "是什么"、"怎么做"、"介绍一下" | "川菜是什么菜系" |
| **detail** | "多少/多久/几勺/比例/温度/火候" | "红烧肉要炖多久" |
| **multi-hop** | ≥2 个实体 / 含过滤条件（场景词）| "适合夏天的川菜" |
| **recommend** | "推荐/介绍几道/有什么 + 场景" | "推荐几道川菜" |

**反例边界**（容易混）：
- "推荐川菜" → recommend（不是 multi-hop）
- "鸡蛋和番茄可以做什么菜" → multi-hop（不是 general）
- "鸡蛋价格" → **OOD**（非烹饪）

---

## 2. 架构总览

### 2.1 4 阶段 Pipeline

```
Query (text)
    ↓
[Stage 1]  LR (TF-IDF + LogisticRegression, cooking/ood 二分类)
           probability p_cooking
    ↓
  p < 0.5  →  直接 unknown（lr_reject）
  0.5 ≤ p < 0.6
    ↓
[Stage 1.5]  LLM (is_cooking?) ← DeepSeek-chat
              ↑ 兜底"伪烹饪"难样本
    ↓
  LLM: false → unknown
  LLM: true / p ≥ 0.6
    ↓
[Stage 2]  BERT (chinese-roberta-wwm-ext, 4 分类)
           max softmax = confidence
    ↓
  conf ≥ 0.6  →  输出 4 类预测
  conf < 0.6
    ↓
[Stage 2.5]  LLM (4 类分类) ← DeepSeek-chat
              ↑ 兜底"4 类难判"样本
    ↓
  输出最终 4 类 / unknown
```

### 2.2 各阶段职责

| 阶段 | 模型 | 触发 | 目标 | 延迟 |
|---|---|---|---|---|
| **Stage 1** | TF-IDF + LR | 全部 query | OOD 兜住（成本最低）| < 1ms |
| **Stage 1.5** | LLM is_cooking | p∈[0.5, 0.6) | LR 不自信的中间地带 | ~500ms |
| **Stage 2** | BERT 4 分类 | LR 判烹饪 + conf≥0.6 | 4 类细分类 | ~6ms |
| **Stage 2.5** | LLM 4 分类 | BERT conf<0.6 | BERT 不自信的难样本 | ~800ms |

### 2.3 关键设计决策

1. **为什么先 LR 后 BERT？**
   - LR < 1ms 拒掉 87.5% OOD query，BERT 不需要看这些
   - LR 是"廉价看门人"，BERT 是"高准确率专家"

2. **为什么中间地带 [0.5, 0.6) 调 LLM？**
   - LR 模糊 → LLM 二次确认
   - 比"LR 全过 0.5 即拒"多 4% OOD 兜住率

3. **为什么 Stage 2.5 不判 is_cooking？**
   - 边界清晰：**OOD 必须由 Stage 1/1.5 拦截**
   - 越权会破坏架构（Stage 2 假设 query 已是烹饪）

4. **为什么用 label_smoothing=0.1？**
   - 4 类 softmax max prob 不会都 ≥ 0.9
   - smoothing 让输出更平滑，配合阈值 0.6 工作

5. **为什么 label_smoothing 让 conf 偏低？**
   - 训练时目标不是 [1,0,0,0] 而是 [0.925, 0.025, ...]
   - 模型学出来 max prob 通常 0.6~0.9 → 阈值 0.6 是合理门槛

---

## 3. 数据流：从菜谱到 query 分类

### 3.1 数据源

| 来源 | 路径 | 数量 | 用途 |
|---|---|---|---|
| 菜谱 Markdown | `data/C9/dishes/{category}/{dish}/{dish}.md` | 350+ | 模板生成原料 |
| OOD 模板 | 手工 + Kimi 合成 | 200+ | LR / 评估 OOD |

**关键过滤**：`HowToCook-master/` 子目录**跳过**（用 `SKIP_DIRS`）。

### 3.2 数据构建 5 步流程

```
Markdown 菜谱 (350 .md)
    ↓  parse_dishes.py
dishes_parsed.jsonl (结构化: 菜名/类别/食材/步骤)
    ↓  generate_from_kg.py
template_{label}.jsonl (4 个文件, ~1083 条)
    ↓  generate_by_llm.py (Kimi / DeepSeek)
llm_{label}.jsonl (4 个文件, 2000/类)
    ↓  double_check.py (LLM 二次打标 + 争议集)
agreed.jsonl + disputed.jsonl
    ↓  clean_and_split.py (SimHash 去重 + 8:1:1 切分)
train.jsonl (865) / dev.jsonl (106) / test.jsonl (112)
    ↓  + OOD out_of_domain.jsonl (20 条)
    ↓  stats.py
data/processed/dataset_stats.md
```

### 3.3 模板生成 4 类套路

详见 [`dataset/generate_from_kg.py`](dataset/generate_from_kg.py)：

| 类别 | 模板骨架 | 占位 |
|---|---|---|
| general | "{category}是什么菜系" / "怎么做{category}中的{dish}" | 菜名、类别 |
| detail | "{dish}中{ingredient}放多少" / "第{step_idx}步要{action}多久" | 食材、步骤 |
| multi-hop | "用{ing1}和{ing2}可以做什么" / "适合{scene}的{category}" | 食材、场景 |
| recommend | "推荐一道{category}里的{scene}菜" | 类别、场景 |

### 3.4 LLM 合成 prompt 设计

详见 [`dataset/generate_by_llm.py`](dataset/generate_by_llm.py)。每条 prompt 包含：
- 4 类意图说明 + 边界规则
- 10 条 few-shot 正反例
- 严格 JSON 输出约束
- 长度 8~40 字约束
- 主题多样化（避免模板化）

### 3.5 二次打标质量门

LLM 合成 2000/类 → 不能全信。`double_check.py` 随机抽 5% 让 LLM 重新打标，**一致率 ≥ 0.85** 才算合格。一致样本入训练集，争议样本留人工。

---

## 4. 各阶段详解

### 4.1 Stage 1: LR 二分类器

**目标**：判 query 是否与"烹饪/做菜/菜谱"相关。

**模型**：TF-IDF + Logistic Regression（sklearn）。

**为什么用 LR？**
- 训练快（10s），推理 < 1ms
- 二分类任务，LR 足够
- 输出概率（0~1），便于设置阈值

**训练数据**（`models/lr_filter/lr.pkl`）：
- in-domain: 4 类 query（~10000 条）
- OOD: 200 条 LLM 合成 + 20 条手工

**特征**：TF-IDF `(1, 2)` ngram，`max_features=20000`。

**输出**：`p_cooking` ∈ [0, 1]。

**关键代码**：[`inference/lr_filter.py`](inference/lr_filter.py)：
```python
def cooking_proba(self, texts: List[str]) -> np.ndarray:
    X = self.tfidf.transform(texts)
    return self.clf.predict_proba(X)[:, 1]  # P(cooking)
```

### 4.2 Stage 1.5: LLM is_cooking

**触发**：`p_cooking ∈ [0.5, 0.6)` 中间地带。

**为什么需要**：LR 给中间值的 query 是"伪烹饪"难样本（如"鸡蛋价格"），LR 拿不准，调 LLM 二次确认。

**Prompt 设计**（[`inference/llm_judge.py`](inference/llm_judge.py) `IS_COOKING_SYSTEM`）：
```
你是中文 query 分类器。判断一条 query 是否与"烹饪 / 做菜 / 菜谱"相关。
仅输出一个 JSON：{"is_cooking": true|false, "confidence": 0~1}
```

**判定规则**（写在 user prompt 里）：
- is_cooking=true：烹饪方法 / 菜谱 / 食材 / 菜系 / 厨房工具 / 营养
- is_cooking=false：股票 / 天气 / 学习 / 工作 / 游戏 / 医疗 / 等

**输出处理**：
- LLM false → unknown（拒绝）
- LLM true → 进入 Stage 2

### 4.3 Stage 2: BERT 4 分类

**模型**：`hfl/chinese-roberta-wwm-ext`（中文 RoBERTa base，102M 参数）。

**为什么选这个**：
- 中文 NLP 经典基座，社区验证
- 全词掩码 (WWM) 比字符级 MLM 在中文任务上更好
- base 规模匹配 8932 训练样本（再大容易过拟合）

**训练超参**（`training/config.yaml`）：
```yaml
lr: 2.0e-5
batch_size: 32
epochs: 5
warmup_ratio: 0.1
weight_decay: 0.01
label_smoothing: 0.1
fp16: auto
max_length: 64
seeds: [42, 123, 1024]
```

**为什么 3 seed？**
- 取最佳 + 算 mean±std
- 减小随机性影响，评估更稳

**输出**：
- logits → softmax → 4 类概率
- `max_prob` = `confidence`
- `argmax` = 预测类别

**关键代码**：[`inference/predictor.py`](inference/predictor.py) `_stage_bert`：
```python
outputs = self.bert(**inputs)
probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
max_idx = int(probs.argmax())
max_prob = float(probs[max_idx])
label = self.id2label[max_idx]
```

### 4.4 Stage 2.5: LLM 4 分类兜底

**触发**：`BERT max_prob < 0.6`。

**为什么需要**：BERT 不自信的 4 类难样本，LLM 二次判。

**Prompt 设计**（[`inference/llm_judge.py`](inference/llm_judge.py) `CLASSIFY_4_SYSTEM`）：
```
你是 query 分类质检员。给定一条中文 query 和 label_guidelines，
判断它属于 4 类之一：general / detail / multi-hop / recommend。
只输出一个 JSON：{"label": "...", "confidence": 0~1}
```

**包含 4 条 few-shot 示例**（detail/recommend/multi-hop/general 各 1 条）。

**重要约束**：**Stage 2.5 不判 is_cooking**。OOD 兜住完全由 Stage 1/1.5 负责。

---

## 5. 端到端评估

### 5.1 评估指标

| 维度 | 指标 | 目的 |
|---|---|---|
| **in-domain 性能** | accuracy / macro-F1 / per-class F1 / 混淆矩阵 | 衡量 4 类分类质量 |
| **OOD 兜住率** | `caught_rate` = 拒识数 / 总 OOD | 衡量非烹饪拒识能力 |
| **stage 分布** | 各 stage 触发计数 | 看 LLM 调用模式 |
| **LLM 成本** | 调用次数 / token / 失败率 | 上线经济性 |
| **延迟** | p50 / p95 / p99 | 线上响应时间 |

### 5.2 E1 实验方法论

**问题**：如何知道 LR 真实分辨力？如何选最佳 LR 阈值？

**E1.1 分布**：[`experiments/sweep_lr.py dist`](experiments/sweep_lr.py)
- 加载 in-domain + OOD 大集（合并 ~1096 样本）
- 跑 LR predict_proba
- 输出 AUC / KS / 重叠度 / 分位数
- **目标**：验证 LR 有没有"分辨力"

**E1.2 PR/ROC**：[`experiments/sweep_lr.py pr`](experiments/sweep_lr.py)
- 扫阈值 [0.05, 0.95]
- 每阈值算 P/R/F1
- **F1 最大点** = LR 理论最优阈值

**E1.3 误伤分析**：[`experiments/lr_analysis.py`](experiments/lr_analysis.py)
- 跑 LR 在 in-domain 全量
- per-class 误伤率
- 误伤样本 top 30
- **目标**：理解 LR 错在哪里

### 5.3 阈值选择（关键经验）

**不要直接用 PR 曲线 F1 最大点**作为最终阈值。要考虑：
1. **业务目标**：in-domain 准确率 vs OOD 兜住率的 trade-off
2. **架构约束**：LR 拒 OOD，BERT 4 分类，LMR 调用成本
3. **经验法则**：
   - in-domain 优先：选 P 高的阈值（误伤少）
   - OOD 兜住优先：选 R 高的阈值（漏判少）

**本项目最终阈值**：
- `lr_low=0.5`（P=0.94, R=0.88 in-distribution）
- `lr_high=0.6`（中间地带 LLM is_cooking）
- `bert=0.6`（4 类判别）

---

## 6. 代码导读

### 6.1 目录结构

```
code/classfier/
├── config.py                  # 阈值 + 路径 + 标签常量 (一切的中心)
├── requirements.txt
├── dataset/                   # 数据集构建
│   ├── label_guidelines.md    # 4 类意图契约 (LLM 必读)
│   ├── parse_dishes.py        # .md → 结构化 jsonl
│   ├── generate_from_kg.py    # 模板生成 (4 类)
│   ├── generate_by_llm.py     # LLM 合成 (OpenAI SDK)
│   ├── double_check.py        # 二次打标
│   ├── clean_and_split.py     # 清洗 + SimHash + 8:1:1
│   └── stats.py
├── training/                  # BERT 训练
│   ├── config.yaml            # 超参
│   ├── train.py               # HF Trainer + 3 seed
│   └── evaluate.py
├── inference/                 # 推理 + 评估
│   ├── lr_filter.py           # TF-IDF + LR 训练/加载
│   ├── llm_judge.py           # LLM is_cooking + 4 类
│   ├── predictor.py           # 4 阶段 pipeline 集成
│   ├── eval_offline.py        # 端到端评估
│   └── benchmark.py           # 延迟
├── experiments/               # 实验代码
│   ├── sweep_lr.py            # E1.1 + E1.2
│   └── lr_analysis.py         # E1.3
├── data/                      # 数据 (gitignore)
│   ├── processed/             # train/dev/test/all.jsonl
│   ├── synthetic/             # 模板 + LLM 合成
│   └── lr_filter/             # LR 训练数据 + OOD
└── models/                    # 模型权重 (gitignore)
    ├── best/                  # 3 seed 中最佳
    ├── seed_42/ ...
    └── lr_filter/             # LR 模型 + TF-IDF
```

### 6.2 关键文件阅读顺序

1. **入口**：[`classfier/config.py`](classfier/config.py) - 路径 / 阈值 / 标签
2. **架构**：[`classfier/inference/predictor.py`](classfier/inference/predictor.py) `QueryClassifier` 类
3. **LR**：[`classfier/inference/lr_filter.py`](classfier/inference/lr_filter.py) `LRFilter` 类
4. **LLM**：[`classfier/inference/llm_judge.py`](classfier/inference/llm_judge.py) `LLMJudge` 类
5. **BERT 训练**：[`classfier/training/train.py`](classfier/training/train.py)
6. **评估**：[`classfier/inference/eval_offline.py`](classfier/inference/eval_offline.py)

### 6.3 执行入口

| 任务 | 命令 |
|---|---|
| 数据全流程 | `python -m classfier.dataset.run_all` |
| 训练 LR | `python -m classfier.inference.lr_filter train` |
| 训练 BERT 3 seeds | `python -m classfier.training.train` |
| 端到端评估 | `python -m classfier.inference.eval_offline --run-id final` |
| 单条预测 | `python -m classfier.inference.predictor --text "..."` |
| 延迟测试 | `python -m classfier.inference.benchmark --n 500` |
| LR 分布分析 | `python -m classfier.experiments.sweep_lr dist --ood-large <abs> --in-domain <abs>` |

---

## 7. 复现指南

### 7.1 环境准备

```bash
# Python 环境 (假设 conda)
conda create -n cook-rag-1 python=3.10
conda activate cook-rag-1

# 安装依赖
pip install -r code/classfier/requirements.txt
pip install -r code/C9/requirements.txt  # 上层 RAG
```

`.env` 配置（`code/.env`）：
```ini
API_KEY=sk-your-deepseek-key
base_url=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

启动 Neo4j + Milvus（用 docker-compose，详见 CLAUDE.md）。

### 7.2 数据构建

```powershell
cd "f:\cook项目\cook\code"
# 全流程 (含 LLM 合成，约 30 min, 需 API key)
python -B -m classfier.run_all

# 或只跑模板 + 切分 (10 min, 不需 API key)
python -B -m classfier.run_all --skip-llm
```

**关键产物**：
- `data/processed/train.jsonl` (~800 条)
- `data/lr_filter/ood_synth.jsonl` (200 条)

### 7.3 模型训练

```powershell
# 1) 训练 LR (10s)
python -B -m classfier.inference.lr_filter train

# 2) 训练 BERT 3 seeds (21 min)
python -B -m classfier.training.train
```

**关键产物**：
- `models/lr_filter/lr.pkl` + `tfidf.pkl`
- `models/seed_42/best/`, `seed_123/best/`, `seed_1024/best/`
- `models/best/` (3 seed 中 macro-F1 最佳)
- `reports/train_summary.json`

### 7.4 端到端评估

```powershell
# 跑 final 评估 (用新默认阈值 0.5/0.6/0.6)
python -B -m classfier.inference.eval_offline --run-id final

# 跑 baseline 对比 (--no-llm / --no-lr)
python -B -m classfier.inference.eval_offline --run-id baseline_no_llm --no-llm
python -B -m classfier.inference.eval_offline --run-id baseline_no_lr --no-lr
```

**关键报告**：
- `reports/pipeline_summary_final.json` (端到端汇总)
- `reports/eval_report_final_test.md` (in-domain 详细)
- `reports/ood_report_final.json` (OOD 小集)
- `reports/ood_report_final_extra.json` (OOD 大集)

### 7.5 单条推理

```powershell
# 跑 in-domain query
python -B -m classfier.inference.predictor --text "红烧肉要炖多久"

# 跑 OOD query
python -B -m classfier.inference.predictor --text "量子计算是什么"

# 跑多分类 query
python -B -m classfier.inference.predictor --text "推荐几道川菜"
```

**期望输出**：
```json
{
  "label": "detail",
  "confidence": 0.92,
  "top2": ["detail", "general"],
  "stage": "bert",
  "lr_proba": 0.95
}
```

---

## 8. 性能数据（来自实际跑过的实验）

### 8.1 BERT 训练 (3 seeds)

| Seed | dev_accuracy | dev_macro_f1 | 训练耗时 (5 epochs) |
|---|---|---|---|
| 42 | 0.9888 | 0.9888 | 84s |
| 123 | 0.9865 | 0.9866 | 83s |
| 1024 | 0.9877 | 0.9877 | 85s |
| **mean ± std** | **0.9877 ± 0.0009** | **0.9877 ± 0.0009** | — |

### 8.2 端到端 final_v2 (0.5/0.6/0.6)

| 维度 | 值 |
|---|---|
| in-domain accuracy | 0.9788 |
| in-domain macro-F1 | **0.9790** |
| OOD(20) caught | 14/20 (70%) |
| **OOD(200) caught** | **176/200 (88%)** |
| LLM calls | 36 |
| LLM tokens | 12,947 |
| LLM fails | 0 |

### 8.3 LR 单独 (大集 1096 样本)

| 指标 | 值 |
|---|---|
| AUC | 0.9737 |
| AP | 0.9369 |
| KS | 0.8633 |
| F1 最大点 | 0.9044 @ threshold=0.50 |

**E1.1 / E1.2 完整数据**：[`reports/experiments/lr_distribution.md`](reports/experiments/lr_distribution.md) + [`reports/experiments/lr_pr_curve.md`](reports/experiments/lr_pr_curve.md)

### 8.4 推理延迟 (BERT, GPU, batch=1)

| 指标 | 值 |
|---|---|
| p50 | 6.2ms |
| p95 | 9.5ms |
| p99 | 10.9ms |
| qps | 149 |

---

## 9. 故障排查

### 9.1 LLM 全部 401 失败

**症状**：`LLM stats: call_count=36, fail_count=36, token_used=0`

**根因**：
- PowerShell session 残留 `$env:API_KEY`（错的 key）
- `dotenv.load_dotenv()` 默认**不覆盖**已有 env var

**修复**：
```powershell
cd "f:\cook项目\cook\code"
Remove-Item env:API_KEY -ErrorAction SilentlyContinue
Remove-Item env:BASE_URL -ErrorAction SilentlyContinue
Remove-Item env:LLM_MODEL -ErrorAction SilentlyContinue
# 重新跑，.env 里的真 key 生效
python -B -m classfier.inference.eval_offline --run-id final
```

### 9.2 GPU 显存不够

**症状**：`RuntimeError: CUDA out of memory`

**修复**：`training/config.yaml`：
```yaml
training:
  batch_size: 16   # 32 → 16 (4GB 显存也够)
  grad_accum_steps: 2  # 等效 batch 32
```

### 9.3 LR 误伤 in-domain 偏高

**症状**：`lr_reject` 数 > 30 in-domain

**根因**：
- 训练 OOD 数据主题不够多样
- 训练数据中"什么是 XX"类 general query 偏少

**修复**：
1. 加"伪烹饪"OOD 训练样本（`鸡蛋价格/菜刀品牌/如何挑选食材`）
2. 加 general 类训练数据多样性
3. 调高 `lr_low`（如 0.5 → 0.6）减少误伤

### 9.4 BERT 4 类边界混

**症状**：混淆矩阵 `detail` ↔ `multi-hop` 互串 > 5%

**修复**：
1. 跑 `dataset/label_guidelines.md` 重新审 4 类边界
2. 加大 `dataset/generate_by_llm.py` 合成 4 类 hard example
3. 调高 `classfier.config.LR_THRESHOLD_LOW` 减少 4 类难样本触发

---

## 10. 进阶方向

按 ROI 排序：

### 高 ROI (推荐)
1. **数据增强**：用 LLM 把现有 query 改写 1 份（保留 label），训练数据 8932 → 17864，预期 +0.2~0.5pp
2. **多 seed 模型集成**：3 seed 模型投票，预期 +0.2~0.4pp
3. **E2 grid 扫描**：5×5 threshold 网格找 Pareto 前沿

### 中等 ROI
4. **R-Drop 正则**：解决 label_smoothing 副作用
5. **chinese-macbert-base**：用 MacBERT 替代 WWM-RoBERTa
6. **多 seed 平均评估**：3 seed 平均而非取最佳

### 低 ROI（边际收益小）
7. **chinese-roberta-wwm-ext-large**：3 倍参数，预期 +0.3~0.5pp 但 3x 慢
8. **扩训练数据到 20000+**：LLM 合成 4 倍量
9. **蒸馏大模型**：Erlangshen-1.3B 蒸馏

### 不建议
- **多任务学习**（4 类 + 实体识别）：复杂度高、收益小
- **半监督 pseudo-label**：需要大量未标数据

---

## 附录 A: 关键决策历史

| 决策 | 之前 | 之后 | 触发 |
|---|---|---|---|
| LR 阈值 | 0.2/0.4 | **0.5/0.6** | E1 大集评估发现误伤 19.6% → 1.3% |
| LLM 兜底 prompt | 无示例 | 4 条 few-shot | double_check 失败率下降 |
| Stage 2.5 职责 | 越权判 is_cooking | **只判 4 类** | 架构边界清晰化 |
| OOD 训练集 | 200 条 | 500+ 可选 | 提升 LR 真实分辨力 |

## 附录 B: 关键术语

- **macro-F1**：4 类 F1 的算术平均（不分权），反映"小类"性能
- **AP (Average Precision)**：PR 曲线下面积，反映"精确率-召回率"权衡
- **AUC**：ROC 曲线下面积，反映分类器对随机正负样本的分辨力
- **KS (Kolmogorov-Smirnov)**：两分布最大累积差，反映分离度
- **label smoothing**：训练时把 [1,0,0,0] 改成 [0.925, 0.025, ...]，正则化
- **WWM (Whole Word Masking)**：中文按词掩码而非字掩码，语义更连贯
- **Pareto 前沿**：多目标优化中，无法在不牺牲一个目标下改进另一个的最优解集

---

**最后更新**：2026-06-08
**项目仓库**：[code/classfier/](file:///f:/cook项目/cook/code/classfier/)
**作者**：本项目由 C9 团队开发，本文档由 Claude 编写供团队学习使用
