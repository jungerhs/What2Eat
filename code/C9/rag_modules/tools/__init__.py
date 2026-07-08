"""
检索工具集合 - Phase 2 Agent 化使用。

每个工具需要实现：
    - name:           str           工具名（英文，函数调用 key）
    - description:    str           工具描述（给 LLM 看，决定何时调用）
    - parameters:     dict          OpenAI tool schema（parameters 部分）
    - __call__(self, **kwargs) -> str  执行并返回文本观察（给 LLM 看）
"""

from .fine_grained_search import FineGrainedSearchTool
from .coarse_grained_search import CoarseGrainedSearchTool

__all__ = ["FineGrainedSearchTool", "CoarseGrainedSearchTool"]
