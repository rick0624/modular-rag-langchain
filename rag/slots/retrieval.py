"""槽位:Retrieval —— 查詢 → 相關切片。契約:query → list[Document]。"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pydantic import Field

from rag.errors import ConfigError
from rag.interfaces import RetrieveFn
from rag.registry import BaseParams, BuildContext, register, validate_params


class _VectorParams(BaseParams):
    top_k: int = Field(default=5, gt=0, description="取回筆數")


@register("retrieval", "vector")
def build_vector(params: dict[str, Any], ctx: BuildContext) -> RetrieveFn:
    """向量相似度檢索(對 indexing 槽位的 vector store 查詢)。

    分數寫入 ``metadata["score"]``:越大越相關、結果降冪,
    只在同一次結果內可比。
    """
    p = validate_params("retrieval", "vector", _VectorParams, params)
    store = ctx.store
    if store is None:
        raise ConfigError(
            "retrieval 方法 'vector' 需要 indexing 槽位提供 vector store"
            "(內部建構順序錯誤)"
        )

    def retrieve(query: str) -> list[Document]:
        pairs = store.similarity_search_with_score(query, k=p.top_k)
        documents: list[Document] = []
        for document, score in pairs:
            document.metadata["score"] = float(score)
            documents.append(document)
        documents.sort(key=lambda d: d.metadata["score"], reverse=True)
        return documents

    return retrieve
