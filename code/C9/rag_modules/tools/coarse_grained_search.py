"""
粗粒度检索工具（Phase 2）

召回与查询最相关的**完整菜谱**。适合回答需要了解一道菜或多道菜整体信息的问题。

实现：
    1. 调细粒度工具拿到 top-K chunks（已经过向量+BM25 合并）
    2. 按 chunk.metadata.parent_id 去重，每个菜谱只保留一个 anchor chunk
    3. 用 ``data_module.parent_id_to_doc`` 映射直接查到**原始菜谱 Document** 返回
       （不做 chunks 拼接 —— 原始 Document 才是规范版本）
    4. 返回 top-K 个完整菜谱
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from .fine_grained_search import FineGrainedSearchTool

logger = logging.getLogger(__name__)


class CoarseGrainedSearchTool:
    """粗粒度菜谱检索工具：返回完整菜谱（按 parent_id 聚合分块）。"""

    name = "coarse_grained_search"
    description = (
        "粗粒度菜谱检索工具。"
        "返回每道菜的**完整信息**"
        "（菜品描述 + 所需食材 + 制作步骤 + 标签）。\n"
        "适合回答需要了解一道菜或多道菜**整体情况**的问题，例如：\n"
        "  - 「川菜有什么特色」\n"
        "  - 「宫保鸡丁是哪里的菜」\n"
        "  - 「帮我介绍几道简单的家常菜」\n"
        "如果用户希望看到完整菜谱（描述+食材+步骤），而不是某个具体细节，优先用此工具。"
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "检索关键词或子问题。建议用完整、可独立理解的子问题。"
                ),
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        fine_tool: FineGrainedSearchTool,
        data_module,
        top_k: int = 3,
    ) -> None:
        self.fine_tool = fine_tool
        self.data_module = data_module
        self.top_k = top_k

    # ── 工具入口 ────────────────────────────────────────────────────────
    def __call__(self, query: str) -> str:
        recipes = self._search(query)
        if not recipes:
            return "（未找到相关菜谱）"
        return self._format_observations(recipes)

    # ── 检索逻辑 ────────────────────────────────────────────────────────
    def _search(self, query: str) -> List[Document]:
        # 1) 调细粒度（拿结构化结果，RRF 排名按顺序）
        try:
            anchor_chunks = self.fine_tool._search(query)
        except Exception as e:
            logger.warning(f"[CoarseGrained] 调用细粒度失败: {e}")
            return []

        if not anchor_chunks:
            return []

        # 2) 按 parent_id 聚合，记录每个菜谱在细粒度中**排名最靠前**的 chunk
        #    enumerate(rank) 显式记录"chunk 在细粒度结果里的位置"，
        #    即「对应 chunk 在细粒度检索的排名」。
        parent_anchors: Dict[str, tuple[int, Document]] = {}
        for rank, chunk in enumerate(anchor_chunks):
            parent_id = chunk.metadata.get("parent_id") or chunk.metadata.get("node_id")
            if not parent_id:
                continue
            if parent_id not in parent_anchors:
                parent_anchors[parent_id] = (rank, chunk)

        # 3) 显式按细粒度排名升序排序（rank 越小越靠前）
        sorted_parents = sorted(parent_anchors.items(), key=lambda x: x[1][0])

        # 4) 查父文档，取 top-K
        result: List[Document] = []
        for parent_id, (_rank, anchor) in sorted_parents[: self.top_k]:
            full = self._fetch_parent_doc(parent_id, anchor)
            if full is not None:
                result.append(full)

        return result

    def _fetch_parent_doc(
        self, parent_id: str, anchor: Document
    ) -> Document:
        """通过 ``data_module.parent_id_to_doc`` 查原始父文档，找不到回退到 anchor。"""
        parent_map = getattr(self.data_module, "parent_id_to_doc", None)
        if parent_map and parent_id in parent_map:
            return parent_map[parent_id]
        # 回退：如果数据模块没建索引，就用 anchor（不应该发生，仅作兜底）
        logger.warning(
            f"[CoarseGrained] parent_id={parent_id} 不在 parent_id_to_doc 中，"
            f"回退使用 anchor chunk（可能不完整）"
        )
        return anchor

    # ── 文本观察 ────────────────────────────────────────────────────────
    def _format_observations(self, docs: List[Document]) -> str:
        """把父文档列表格式化成 LLM 易读的文本。

        每个 doc 是 ``build_recipe_documents`` 拼出的原始菜谱（未分块）。
        """
        lines: List[str] = []
        for i, d in enumerate(docs, 1):
            recipe = d.metadata.get("recipe_name") or "未知菜品"
            node_id = d.metadata.get("node_id", "?")
            content = d.page_content.strip()
            lines.append(
                f"=== [菜谱 {i}] {recipe} (node_id={node_id}) ===\n{content}"
            )
        return "\n\n".join(lines)
