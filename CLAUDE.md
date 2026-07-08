# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project: C9（尝尝咸淡）— Graph RAG Cooking Assistant

A Chinese cooking Q&A system built on **Graph RAG** (Neo4j + Milvus + vector search + LLM). The system understands free-form cooking queries, routes them through a 3-tier intent classifier, retrieves from either a traditional hybrid index (BM25 + Milvus) or a knowledge graph (Cypher multi-hop), and generates answers via a hand-written ReAct agent. Also handles multi-turn conversation, user profiles, and recipe import. Detailed scope/claims live in `项目介绍.md` — treat that as the authoritative project overview (do not invent additional scale numbers beyond what it documents).

---

## Repository Layout

```
F:\cook项目\cook\
├── code/
│   ├── C9/                  # Main application (FastAPI + CLI)
│   │   ├── api_server.py       # FastAPI REST + SSE streaming
│   │   ├── main.py             # AdvancedGraphRAGSystem + CLI entry
│   │   ├── config.py           # GraphRAGConfig dataclass (Neo4j/Milvus/Redis/LLM)
│   │   ├── logging_setup.py    # setup_logging() + flow markers (imports across files)
│   │   ├── recipe_import.py    # Markdown → Cypher (incremental to KG + indexes)
│   │   ├── docker-compose.yml  # Milvus stack (etcd + MinIO + standalone)
│   │   ├── rag_modules/        # All retrieval/RAG logic (see below)
│   │   ├── rag_modules/tools/  # FineGrainedSearchTool, CoarseGrainedSearchTool
│   │   ├── agents/recipe_parser/   # Kimi-based recipe parser
│   │   ├── image_retrieval/        # CLIP-based image indexing (separate sub-system)
│   │   ├── storage/                # auth_store.py, orchestrator_db.py (PostgreSQL), schema.sql
│   │   ├── tests/                  # Smoke & diagnostic scripts (see Tests)
│   │   └── landing-page/           # index.html (landing), auth.html, app.html (chat)
│   └── classfier/            # Training data pipeline for the BERT intent classifier
│       ├── run_all.py            # One-shot dataset pipeline entry
│       ├── config.py             # Labels, target counts, paths
│       ├── dataset/              # parse_dishes / generate_from_kg / generate_by_llm / double_check / clean_and_split / stats
│       └── data/{raw,synthetic,processed}/
├── data/
│   ├── C9/dishes/           # 200+ recipes (Markdown) organized by category
│   │   ├── aquatic/ breakfast/ condiment/ dessert/ drink/
│   │   ├── meat_dish/ semi-finished/ soup/ staple_food/ vegetable_dish/
│   └── HowToCook-master/    # Upstream recipe source
└── para.py                  # Misc LangGraph/agent experiment script (NOT production)
```

---

## Key Architecture

### Request flow (FastAPI → answer)

```
HTTP /api/chat
   │
   ▼
[ChatRequest] → AdvancedGraphRAGSystem.process_query_question
   │
   ├─ MultiTurnIntentClassifier  (Phase 6, ≥ turn 2)
   │     └─ small LLM (Ollama) → big LLM (API) fallback
   │
   ├─ Intent classify (LR → BERT → LLM cascade)        intent_classifier
   │
   ├─ ConversationOrchestrator                         session_id in Redis (+ PG)
   │     ├─ query rewriting (resolves "它" / "这个菜")
   │     └─ sliding window + LLM summary compression
   │
   ├─ Route by intent:
   │     • general / detail  → ReActAgent (hand-written, OpenAI tool_calls)
   │                            ├─ FineGrainedSearchTool   (Milvus vec + BM25)
   │                            └─ CoarseGrainedSearchTool (KG subgraph expansion)
   │     • multi-hop         → GraphRAGRetrieval (Text2Cypher + multi-hop traversal)
   │     • recommend         → hybrid + user profile weighting
   │
   ├─ GenerationIntegrationModule  →  LLM (DeepSeek / Kimi, OpenAI-compatible)
   │
   └─ orchestrator.record_turn  (Redis double-write, async PG write)
```

### Module map (`code/C9/rag_modules/`)

| File | Role |
|---|---|
| `intelligent_query_router.py` | Decision: traditional hybrid vs graph RAG vs combined (LLM-judged strategy + `QueryAnalysis` dataclass) |
| `hybrid_retrieval.py` | BM25 + Milvus vector fusion (the "traditional" engine) |
| `graph_rag_retrieval.py` | Text2Cypher generation + Cypher execution + multi-hop traversal over Neo4j |
| `graph_data_preparation.py` | Builds the KG from recipes (chunks/entities/relations) |
| `milvus_index_construction.py` | Constructs/updates Milvus indexes (supports incremental upsert) |
| `react_agent.py` | Hand-written ReAct loop, `max_iterations=4`, tool observations back into messages. Disables `thinking` mode via `extra_body={"thinking":{"type":"disabled"}}` |
| `conversation_orchestrator.py` | Session history, query rewriting, sliding-window summary |
| `multi_turn_intent.py` | Small-LLM-first intent classifier for turn ≥ 2; outputs `{label, confidence}` JSON |
| `generation_integration.py` | Prompt assembly (system + history + context + user profile) |
| `graph_indexing.py` | Older KG index builder |
| `tools/fine_grained_search.py` | Vector + keyword retrieval tool exposed to the agent |
| `tools/coarse_grained_search.py` | KG neighborhood expansion tool exposed to the agent |

### Storage topology

- **Neo4j** (`bolt://localhost:7687`, db `neo4j`, pwd `all-in-rag`) — knowledge graph of recipes/ingredients/steps/categories/cuisines.
- **Milvus** (`localhost:19530`, collection `cooking_knowledge`, **dim=512** because embeddings use `BAAI/bge-small-zh-v1.5`).
- **Redis** (`localhost:6379/0`) — retrieval cache + 24h session history (TTL controlled by `conversation_orchestrator.ttl`).
- **PostgreSQL** — long-term session archive + user profiles (Phase 4). Used via `OrchestratorDB` with **lazy-load**: Redis miss falls back to PG; writes are **async double-write** (`ThreadPoolExecutor`).
- **MinIO + etcd** — Milvus dependencies (started by `docker-compose.yml`).

### Intent classes (classifier contract)

`general` | `detail` | `multi-hop` | `recommend` | (`unknown` for non-cooking)
The training pipeline and the multi-turn LLM classifier MUST agree on these labels — definitions live in `code/classfier/dataset/label_guidelines.md`.

---

## Common Commands

> All commands assume `conda activate cook-rag-1` (env name preserved in memory). Run from `F:\cook项目\cook`.

### Backend service

```powershell
# 1) Milvus stack (etcd + MinIO + standalone)
cd "F:\cook项目\cook\code\C9"
docker compose up -d

# 2) Start FastAPI (server with SSE + auth + static landing page)
python api_server.py
# Default: http://127.0.0.1:8000  (landing-page/ served as static, /api/* for backend)

# 3) CLI multi-turn interactive session (shares one CLI session_id)
python main.py
```

### Tests / diagnostics (`code/C9/tests/`)

Most "tests" are black-box diagnostic scripts run against a live server. They expect `http://127.0.0.1:8000` and a logged-in user.

```powershell
# Example: cache hit-rate × 5  (requires server up + logged-in token)
python tests/test_cache_5x.py
python tests/test_cache_isolation.py
python tests/test_cache_recommend.py
python tests/test_streaming.py        # verifies SSE token-by-token emission
python tests/test_multi_turn_intent.py
```

Standalone smoke tests (no server needed):

```powershell
python _ollama_smoke.py                 # ollama-judge provider check
python test_cli_multi_turn.py           # syntax + structural checks of CLI multi-turn code
```

### Classifier dataset pipeline (`code/classfier/`)

```powershell
pip install -r code/classfier/requirements.txt
$env:API_KEY="sk-..."; $env:BASE_URL="https://api.moonshot.cn/v1"; $env:LLM_MODEL="kimi-k2-0711-preview"

# Full pipeline (template + LLM synthesize + double-check + split)
python code/classfier/run_all.py

# Skip LLM synthesis (template only, network-free smoke)
python code/classfier/run_all.py --skip-llm

# Step-by-step (each as a module)
python -m classfier.dataset.parse_dishes
python -m classfier.dataset.generate_from_kg
python -m classfier.dataset.generate_by_llm --n 2000
python -m classfier.dataset.double_check
python -m classfier.dataset.clean_and_split
python -m classfier.dataset.stats
```

Outputs land in `code/classfier/data/{raw,synthetic,processed}/` and `reports/dataset_stats.md`. `double_check_report.json` `agreement` must be **≥ 0.85** before proceeding to training.

### Recipe import (incremental to KG + indexes)

```powershell
# Single .md file uploaded via the web app OR:
python -c "from code.C9.recipe_import import import_recipe; import_recipe('path/to/recipe.md')"
# AI parse → Cypher → Neo4j upsert, then Milvus + BM25 incremental update (no full rebuild).
```

---

## Conventions & Gotchas

- **.env**: required keys (`code/C9/.env`) — `API_KEY`/`MOONSHOT_API_KEY`, `NEO4J_*`, `MILVUS_*`, `LLM_MODEL`, `base_url`, optional `POSTGRES_DSN`, optional `OLLAMA_*` for local intent LLM. **Note**: the same project uses two env-variable conventions for the LLM key — `API_KEY` (script path) **and** `MOONSHOT_API_KEY` (some pipelines). `api_server.py` reads via `python-dotenv`; classifier pipeline reads `os.getenv('API_KEY' or 'MOONSHOT_API_KEY')`.
- **Embedding dimension is 512** for `bge-small-zh-v1.5`. If you change the embedding model, `milvus_dimension` in `config.GraphRAGConfig` AND the Milvus collection schema must change together, or drop & recreate the `cooking_knowledge` collection.
- **Disable thinking mode for ReAct** — `react_agent.py` sets `extra_body={"thinking":{"type":"disabled"}}` because some providers (e.g. deepseek-v4-flash) require this to avoid 400 errors on follow-up turns. Don't remove this without re-testing tool-calling.
- **Text2Cypher model must be non-reasoning** — `config.t2c_llm_model` defaults to `deepseek-chat`. Reasoning models will burn `max_tokens` on `reasoning_content` and return empty `content`.
- **GBK stdout on Windows**: `api_server.py` wraps `sys.stdout`/`stderr` in UTF-8 `TextIOWrapper` because of Unicode checkmarks/emojis in logs. If you add a new entry-point with non-ASCII output on Windows, do the same.
- **Session IDs**: CLI uses a process-stable `cli-<8hex>` so consecutive turns in one process share history. API clients pass `session_id` explicitly.
- **User IDs**: Persisted to `~/.cook-rag/user_id` so CLI sessions share profiles across restarts. API clients pass `user_id` explicitly.
- **PostgreSQL is optional** — if `POSTGRES_DSN` is empty/unreachable, the system silently degrades to Redis-only 24h storage. DB writes are **async and best-effort**; never block request flow on PG.
- **AI parse + regex fallback** — `recipe_import.py` uses Kimi first; if the AI returns empty data it falls back to regex parsing. Don't remove the regex fallback.

---

## Where to look when …

| Symptom | Inspect |
|---|---|
| Wrong intent routing | `intelligent_query_router.py`, `multi_turn_intent.py`, classifier data in `code/classfier/data/processed/` |
| Hallucinated ingredients/multi-hop answers wrong | Neo4j data freshness; `graph_data_preparation.py`; `graph_rag_retrieval.py` Cypher templates |
| SSE not streaming | `tests/test_streaming.py` + `api_server.py` `StreamingResponse`; check `react_agent.py` stream path |
| Cache hit/miss weird | `tests/test_cache_*.py`; Redis key prefix `c9:session:*` |
| New recipe not retrievable | Did `recipe_import.py` run end-to-end? Check Cypher commit + Milvus upsert logs |
| Multi-turn loss of context | `conversation_orchestrator.py` (Redis TTL, summary trigger) |
| Login fails | `storage/auth_store.py` + `storage/auth.db` (SQLite); auth endpoints in `api_server.py` |

---

## Remember

- **Memory**: conda env is `cook-rag-1` (persisted in `MEMORY.md`).
- **Project facts** to preserve (verified from `项目介绍.md`): 200+ recipes, 10 categories, 1000+ KG nodes / 3000+ relations, 8000+ classifier samples, 50+ Python modules. Do NOT inflate these numbers.
- **`para.py` at the repo root is throwaway LangGraph experimentation**, not part of C9. Leave it alone unless asked.
- **`code/C9/agents/recipe_parser/`** is the older Kimi-based parser pre-`recipe_import.py`; referenced by an earlier README. The current import path is `code/C9/recipe_import.py`.
- **No `.cursorrules`, no copilot instructions, no `CLAUDE.md` history to migrate** (the deleted `CLAUDE.md` was a stub).
