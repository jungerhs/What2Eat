"""
细粒度检索工具（Phase 2）

召回与查询最相关的**菜谱分块（chunk）**，分块粒度对应一道菜的某个章节
（描述/食材/步骤/标签）。适合回答需要精确定位某道菜具体细节的问题。

召回通道：
    1. Milvus 向量检索（语义相似度）
    2. BM25 关键词检索（精确关键词匹配）
    3. RRF（Reciprocal Rank Fusion）合并去重

依赖：
    - milvus_module:  MilvusIndexConstructionModule（提供 similarity_search）
    - bm25_retriever: langchain BM25Retriever 实例（由 HybridRetrievalModule 初始化）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FineGrainedSearchTool:
    """细粒度菜谱检索工具：返回分块（chunk），不聚合成完整菜谱。"""

    name = "fine_grained_search"
    description = (
        "细粒度菜谱检索工具。返回与查询最相关的**菜谱分块（chunk）**，"
        "每个分块是一道菜的某一段内容（描述/所需食材/制作步骤/标签）。\n"
        "适合回答需要精确定位某道菜**具体细节**的问题，例如：\n"
        "  - 「宫保鸡丁第几步放花生」\n"
        "  - 「红烧肉的糖色怎么炒」\n"
        "  - 「麻婆豆腐需要哪些调料」\n"
        "如果不确定该用哪种粒度，优先调用此工具（细粒度返回的信息更聚焦）。"
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "检索关键词或子问题。建议使用完整、可独立理解的子问题，"
                    "避免使用代词（『它』『那菜』等），因为多轮指代会由上层处理。"
                ),
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        milvus_module,
        bm25_retriever,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self.milvus_module = milvus_module
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    # ── 工具入口 ────────────────────────────────────────────────────────
    def __call__(self, query: str) -> str:
        """执行检索并返回文本观察（给 LLM 看）。"""
        chunks = self._search(query)
        if not chunks:
            return "（未找到相关分块）"
        return self._format_observations(chunks)

    # ── 检索逻辑 ────────────────────────────────────────────────────────
    def _search(self, query: str) -> List[Document]:
        """Milvus + BM25 → RRF 合并 → top_k。"""
        vec_hits: List[Document] = []
        try:
            raw_vec = self.milvus_module.similarity_search(query, k=self.top_k * 2)
            vec_hits = [d for d in (self._hit_to_doc(h) for h in raw_vec) if d is not None]
        except Exception as e:
            logger.warning(f"[FineGrained] 向量检索失败: {e}")

        bm25_hits: List[Document] = []
        if self.bm25_retriever is not None:
            try:
                # BM25Retriever.invoke() 返回 List[Document]
                bm25_hits = list(self.bm25_retriever.invoke(query, k=self.top_k * 2))
            except Exception as e:
                logger.warning(f"[FineGrained] BM25 检索失败: {e}")

        merged = self._rrf_merge(vec_hits, bm25_hits)
        return merged[: self.top_k]

    def _hit_to_doc(self, hit) -> Optional[Document]:
        """把 Milvus 返回的 dict 包装成 LangChain Document。"""
        if isinstance(hit, Document):
            return hit
        if not isinstance(hit, dict):
            return None
        try:
            content = hit.get("text", "")
            metadata = dict(hit.get("metadata") or {})
            metadata["vector_score"] = hit.get("score", 0.0)
            metadata["hit_source"] = "milvus"
            return Document(page_content=content, metadata=metadata)
        except Exception:
            return None

    def _rrf_merge(
        self,
        vec_hits: List[Document],
        bm25_hits: List[Document],
    ) -> List[Document]:
        """Reciprocal Rank Fusion：k 常数抑制高分文档的绝对优势。"""
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(vec_hits):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_hits):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map.setdefault(key, doc)  # 已被向量召回的不替换（保留 metadata）

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for key, _ in ranked:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = scores[key]
        return [doc_map[k] for k, _ in ranked]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """RRF 用的去重 key，优先用 chunk_id，没有就用 content 前 80 字。"""
        cid = doc.metadata.get("chunk_id")
        if cid:
            return f"id::{cid}"
        return f"c::{hash(doc.page_content[:80])}"

    # ── 文本观察 ────────────────────────────────────────────────────────
    def _format_observations(self, chunks: List[Document]) -> str:
        """把分块列表格式化成 LLM 易读的文本。"""
        lines: List[str] = []
        for i, c in enumerate(chunks, 1):
            recipe = c.metadata.get("recipe_name") or "未知菜品"
            chunk_id = c.metadata.get("chunk_id", "?")
            section = c.metadata.get("section_title", "")
            section_tag = f" · {section}" if section else ""
            rrf = c.metadata.get("rrf_score", 0.0)
            content = c.page_content.strip()
            lines.append(
                f"--- [分块 {i}] {recipe}{section_tag} "
                f"(chunk_id={chunk_id}, rrf={rrf:.4f}) ---\n{content}"
            )
        return "\n\n".join(lines)
