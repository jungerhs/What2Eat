# What2Eat 

> 智能烹饪问答助手 —— 基于 Graph RAG 与多轮对话的中文烹饪知识系统

（**What2Eat**）是一个面向烹饪领域的智能问答系统，名字取自厨房里"尝一口咸淡"的动作。它把大语言模型与烹饪知识深度结合，让用户用自然语言自由提问（"红烧肉要炖多久" / "减脂期有什么低卡素菜" / "麻婆豆腐和回锅肉共用了哪些调料"），系统能理解意图、精准检索、并生成高质量回答。



## 🎬 Demo



https://github.com/user-attachments/assets/06a83897-610d-4180-b72b-3b5cacabb401





---

## ✨ 核心特性

### 1. 三层意图识别（LR → BERT → LLM 级联）

自研级联分类器把用户查询分成 4 类（`general` / `detail` / `multi-hop` / `recommend`）：

- **第一层 LR**：TF-IDF + LogisticRegression，<1 ms 拦截非烹饪领域问题
- **第二层 BERT**：微调 `chinese-roberta-wwm-ext`，烹饪类精准四分类
- **第三层 LLM 兜底**：前两层置信度不足时再调用大模型

90%+ 查询在前两层就完成分类，避免昂贵 LLM 调用。训练这套分类器的数据合成 pipeline 详见 `code/classfier/`：Neo4j 抽实体/关系模板 → Kimi 批量合成 → 二次打标 → 8 : 1 : 1 切分，产出 8000+ 条高质量样本。

### 2. Graph RAG：知识图谱 + 向量检索双引擎

构建了一个 Neo4j 烹饪知识图谱（200+ 道菜谱、1000+ 节点、3000+ 关系），按意图自动路由：

| 意图 | 策略 | 后端 |
|---|---|---|
| `general` / `detail` | 传统混合检索 | BM25 + Milvus 向量 |
| `multi-hop`         | Graph RAG     | Text2Cypher + 多跳遍历 |
| `recommend`         | 检索 + 画像加权 | 用户偏好 / 忌口 / 烹饪水平 |

Graph RAG 擅长回答关系推理类问题（"川菜里哪些荤菜用到豆瓣酱"），这类问题纯向量检索很难命中。

### 3. ReAct Agent

针对 `general` / `recommend` 三类查询实现了一个ReAct 引擎：

- **细粒度检索工具**：Milvus 向量 + BM25 关键词
- **粗粒度检索工具**：知识图谱子图扩展（提供"上下文感知"的关联菜谱）

Agent 自主决定调哪个工具、是否多次调用（`max_iterations=4`），工具观察结果回传 LLM 后综合最终答案。

### 4. 多轮对话与用户画像

完整的对话系统：

- **Redis**：短期会话历史（滑动窗口 + LLM 摘要压缩，TTL 24h）
- **PostgreSQL**：长期会话归档 + 用户画像（菜系偏好 / 忌口 / 烹饪水平）
- **Query Rewriting**：自动把"它"、"这个菜"等模糊指代改写成 self-contained 查询
- **多轮意图识别**：第 2 轮起走 Ollama 本地小模型（`gemma4:e4b`），置信度不足时回退到 API 大模型

### 5. 食谱智能导入（增量，无须重建索引）

支持上传 Markdown 食谱自动入库：

1. Kimi AI Agent 解析 → 提取食材 / 步骤 / 分类等结构化信息
2. Cypher `MERGE` 直连 Neo4j（幂等）
3. 增量更新 Milvus 向量索引 + BM25 索引（**不重建整个知识库**）
4. AI 返回空时自动降级到正则规则解析（双保险）


```



## 🧰 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.10+ |
| Web 框架 | FastAPI + Uvicorn（同时提供 REST + SSE 流式） |
| 知识图谱 | Neo4j 5.x（Cypher 查询、Text2Cypher） |
| 向量库 | Milvus 2.5（`BGE-small-zh-v1.5`，**512 维**） |
| 缓存 & 会话 | Redis 7（检索缓存、24h 会话历史） |
| 长期存储 | PostgreSQL（会话归档 + 用户画像，可选） |
| 鉴权 | SQLite（`storage/auth.db`） |
| LLM | DeepSeek / Kimi（OpenAI 兼容协议） |
| 文本分类 | scikit-learn LR + Transformers BERT 微调 |
| 嵌入 | `BAAI/bge-small-zh-v1.5` (sentence-transformers) |
| 容器化 | Docker Compose（Milvus + etcd + MinIO） |
| 前端 | 原生 HTML / CSS / JS（无构建步骤），位于 `code/C9/landing-page/` |

---

## 🚀 快速开始

### 环境要求

- **Python 3.10+**
- **Docker Desktop** + Docker Compose v2
- 推荐使用 [conda](https://docs.conda.io) 隔离环境：`conda create -n cook-rag-1 python=3.10`
- ≥ 8 GB 可用内存（Milvus standalone 较重）

### 1. 克隆与安装

```bash
git clone https://github.com/jungerhs/What2Eat.git
cd What2Eat

conda create -n cook-rag-1 python=3.10 -y
conda activate cook-rag-1

pip install -r code/C9/requirements.txt
```

### 2. 启动基础设施（Milvus 栈）

```bash
docker compose -f code/C9/docker-compose.yml up -d
docker compose -f code/C9/docker-compose.yml ps
```

需要起 etcd / MinIO / Milvus standalone 三个容器。

### 3. 写 `.env`

```bash
cp code/C9/.env.example code/C9/.env
# 编辑 .env，至少填 API_KEY（LLM 密钥）
```

主要变量：

| 变量 | 必填 | 说明 |
|---|---|---|
| `API_KEY` 或 `MOONSHOT_API_KEY` | ✅ | LLM 密钥（OpenAI 兼容协议） |
| `NEO4J_URI` / `NEO4J_PASSWORD`  | ✅ | 默认 `bolt://localhost:7687` / `all-in-rag` |
| `MILVUS_HOST` / `MILVUS_PORT`  | ✅ | 默认 `localhost:19530` |
| `LLM_MODEL`                    | ✅ | 默认 `deepseek-v4-flash` |
| `EMBEDDING_MODEL`              | ✅ | 默认 `BAAI/bge-small-zh-v1.5` |
| `base_url`                     | ✅ | 默认 DeepSeek，可换 Moonshot / 自部署 Ollama |
| `POSTGRES_DSN`                 | ❌ | 留空则只走 Redis 24h |
| `OLLAMA_*`                     | ❌ | 多轮意图识别走本地小模型 |

> ⚠️ 修改 `EMBEDDING_MODEL` 会改变向量维度。`bge-small-zh-v1.5` 是 **512 维**，换模型需同步修改 `config.GraphRAGConfig.milvus_dimension` 并重建 Milvus collection。

### 4. 启动服务

```bash
# 方式 A：Web API（REST + SSE + landing page）
python code/C9/api_server.py
# → http://127.0.0.1:8000 浏览器直接打开就是前端

# 方式 B：CLI 多轮交互
python code/C9/main.py
# 同一进程内 session_id 共享历史，跨进程会话通过 PostgreSQL 持久化
```

### 5.（可选）准备分类器训练数据

```bash
pip install -r code/classfier/requirements.txt

# 仅模板生成（无需联网）
python code/classfier/run_all.py --skip-llm

# 完整流程（含 LLM 合成 + 二次打标）
export API_KEY="sk-..."
export BASE_URL="https://api.moonshot.cn/v1"
export LLM_MODEL="kimi-k2-0711-preview"
python code/classfier/run_all.py
```

---

## 📁 项目结构

```
What2Eat/
├── code/
│   ├── C9/                     # 主应用（FastAPI + CLI）
│   │   ├── api_server.py       #   REST + SSE 流式入口
│   │   ├── main.py             #   AdvancedGraphRAGSystem + CLI
│   │   ├── config.py           #   GraphRAGConfig (Neo4j/Milvus/Redis/LLM)
│   │   ├── recipe_import.py    #   Markdown → Cypher 增量导入
│   │   ├── docker-compose.yml
│   │   ├── rag_modules/        #   检索 / Agent / Orchestrator
│   │   ├── rag_modules/tools/  #   ReAct Agent 检索工具
│   │   ├── image_retrieval/    #   CLIP 图文检索子模块
│   │   ├── storage/            #   auth_store + orchestrator_db
│   │   ├── tests/              #   诊断脚本
│   │   └── landing-page/       #   index.html / auth.html / app.html
│   └── classfier/              # 意图分类器训练数据 pipeline
├── data/
│   ├── C9/
│   │   ├── dishes/             # 200+ Markdown 食谱
│   │   └── images/             # 342 张 dish 图片（扁平）
│   └── HowToCook-master/       # 上游食谱源仓库
├── CLAUDE.md                   # 给 Claude Code 看的开发指南
└── README.md                   # 你正在看的这份
```

---

## 📊 数据规模

| 维度 | 数量 |
|---|---|
| 菜谱 | 200+ 道（10 类：水产 / 荤菜 / 素菜 / 主食 / 汤类 / 甜品 / 饮品 / 早餐 / 调料 / 半成品） |
| 知识图谱节点 | 1000+ |
| 知识图谱关系 | 3000+ |
| 分类器训练样本 | 8000+（template + Kimi 合成 + 二次打标） |
| 菜谱图片 | 342 张（来自 [king-jingxiang/HowToCook](https://github.com/king-jingxiang/HowToCook)） |
| Python 模块 | 50+ |

---

## 🧪 开发与诊断

```bash
# CLI 多轮的语法/结构检查
python test_cli_multi_turn.py

# 本地 Ollama 联通测试
python _ollama_smoke.py

# 缓存 / 流式 / 多轮意图 诊断（需要先跑起 api_server.py）
python code/C9/tests/test_cache_5x.py
python code/C9/tests/test_cache_isolation.py
python code/C9/tests/test_streaming.py
python code/C9/tests/test_multi_turn_intent.py
```

> 这些是连接到已起 server 上的**黑盒诊断脚本**，不是单元测试。

---

## 🙏 致谢

- 食谱 Markdown 数据：[Anduin2017/HowToCook](https://github.com/Anduin2017/HowToCook)
- 食谱图片（`data/C9/images/`）：[king-jingxiang/HowToCook](https://github.com/king-jingxiang/HowToCook)

---

## 📜 License

本仓库暂未指定开源许可证。如需使用、复用或二次分发，请先与作者沟通。
