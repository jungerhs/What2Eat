# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

C9 (尝尝咸淡) is a Chinese cooking knowledge graph-based RAG (Retrieval-Augmented Generation) system. It parses markdown recipes into a Neo4j knowledge graph, then uses a dual-engine retrieval system (vector search via Milvus + graph traversal via Neo4j) with an LLM to answer cooking questions.

## Repository Structure

```
cook/
├── code/C9/                          # Application code
│   ├── agent(代码系ai生成)/           # AI recipe parser (generated code)
│   │   ├── recipe_ai_agent.py        # Core: Kimi-based recipe → KG parser
│   │   ├── run_ai_agent.py           # CLI entry point for recipe parsing
│   │   ├── batch_manager.py          # Batch processing & incremental updates
│   │   ├── amount_normalizer.py      # Ingredient amount normalization
│   │   └── config.json               # API keys, category mappings, output config
│   ├── rag_modules/                  # Graph RAG retrieval system
│   │   ├── graph_data_preparation.py # Neo4j → LangChain Document conversion
│   │   ├── milvus_index_construction.py # Milvus vector indexing (HNSW/COSINE)
│   │   ├── graph_indexing.py         # Entity/relation key-value index for graph search
│   │   ├── hybrid_retrieval.py       # Dual-level retrieval (entity + topic) + BM25
│   │   ├── graph_rag_retrieval.py    # Graph-native retrieval: multi-hop, subgraph, paths
│   │   ├── intelligent_query_router.py # Routes queries to optimal search strategy
│   │   └── generation_integration.py # LLM answer generation (Moonshot/Kimi API)
│   ├── main.py                       # AdvancedGraphRAGSystem orchestrator + interactive CLI
│   ├── config.py                     # GraphRAGConfig dataclass (Neo4j, Milvus, model settings)
│   ├── docker-compose.yml            # Milvus + etcd + MinIO
│   └── .env.example                  # MOONSHOT_API_KEY, Neo4j, Milvus env vars
└── data/C9/                          # Data & database infrastructure
    ├── docker-compose.yml            # Neo4j 5.18 + APOC + auto-import init service
    └── cypher/
        ├── nodes.csv                 # Knowledge graph nodes
        ├── relationships.csv         # Knowledge graph relationships
        └── neo4j_import.cypher       # Cypher import script (constraints, indexes, LOAD CSV)
```

## Infrastructure & Prerequisites

The system requires three services running:

| Service | Port | Purpose |
|---------|------|---------|
| Neo4j 5.18 | 7474 (HTTP), 7687 (Bolt) | Graph database for recipe knowledge graph |
| Milvus 2.5 | 19530, 9091 | Vector database for embedding search |
| etcd + MinIO | — | Milvus dependencies (auto-started with Milvus compose) |

**Start databases:**
```bash
# Start Neo4j (from data/C9/)
cd data/C9 && docker-compose up -d

# Start Milvus + etcd + MinIO (from code/C9/)
cd code/C9 && docker-compose up -d
```

**API key required:** Set `MOONSHOT_API_KEY` environment variable (used by both the recipe parser and RAG system). Copy `.env.example` to `.env` and fill in your key.

## Common Commands

### Recipe Knowledge Graph Construction

```bash
cd code/C9

# Install dependencies
pip install -r requirements.txt
# Install AI agent dependencies (separate venv recommended)
pip install -r "agent(代码系ai生成)/requirements.txt"

# Test single recipe parsing
python "agent(代码系ai生成)/run_ai_agent.py" test

# Full batch processing of a recipe directory (e.g., HowToCook)
python "agent(代码系ai生成)/run_ai_agent.py" /path/to/HowToCook-master

# Incremental processing (only changed files)
python "agent(代码系ai生成)/run_ai_agent.py" /path/to/HowToCook-master --incremental

# Batch manager utilities
python "agent(代码系ai生成)/batch_manager.py" status
python "agent(代码系ai生成)/batch_manager.py" merge
```

### RAG System

```bash
cd code/C9

# Install RAG deps
pip install -r requirements.txt

# Start the interactive cooking assistant
python main.py
```

Interactive commands within the RAG assistant: `stats`, `rebuild`, `quit`.

## Architecture

### Data Flow

```
Markdown Recipes ──→ KimiRecipeAgent ──→ nodes.csv + relationships.csv ──→ Neo4j
                                                                              │
                                                                              ▼
User Question ──→ IntelligentQueryRouter ──→ HybridRetrieval (BM25 + Milvus vectors)
                   │                         │
                   │                         ├──→ GraphRAGRetrieval (multi-hop/subgraph)
                   │                         │
                   ▼                         ▼
              QueryAnalysis          Retrieved Documents
                                             │
                                             ▼
                                   GenerationIntegration (Moonshot LLM)
                                             │
                                             ▼
                                        Answer
```

### Three Retrieval Strategies

The `IntelligentQueryRouter` analyzes each query and picks one of three strategies:

1. **hybrid_traditional** — For simple lookups. Uses `HybridRetrievalModule` which does dual-level retrieval (entity keywords + topic keywords) combined with Milvus vector search, merged via round-robin.
2. **graph_rag** — For complex relationship queries. Uses `GraphRAGRetrieval` which performs Cypher-based multi-hop traversal, subgraph extraction, and path finding on Neo4j.
3. **combined** — Splits results between both engines and interleaves them.

### Key Design Decisions

- **Embedding model:** `BAAI/bge-small-zh-v1.5` (512-dimensional, Chinese-optimized)
- **LLM:** Kimi (Moonshot API) via OpenAI-compatible client (`api.moonshot.cn/v1`)
- **Vector index:** Milvus with HNSW index, COSINE similarity
- **Graph database:** Neo4j 5.18, accessed via Bolt driver (no APOC dependency in retrieval code)
- **Round-robin merging:** Used throughout instead of weighted fusion — fair alternation between result sources
- **Query routing:** LLM-based analysis of query complexity and relationship intensity, with rule-based fallback
- **Entity encoding:** Recipe nodes use hierarchical IDs (e.g., `710000000` for 素菜 category, `200000000+` for instance nodes)

### Neo4j Graph Schema

**Node types:** `Recipe`, `Ingredient`, `CookingStep`, `Category`
**Relationships:** `REQUIRES` (Recipe→Ingredient), `BELONGS_TO_CATEGORY` (Recipe→Category), `CONTAINS_STEP` (Recipe→CookingStep)
