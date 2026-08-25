"""槽位:Fusion —— 多子查詢結果合併。契約:results → documents。

一律執行:query_transformation 只輸出 1 筆子查詢時,合併等於去重 + 截斷。
config 省略 fusion 區塊時,框架以 ``merge`` 的預設參數建立本槽位。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pydantic import Field

from rag.interfaces import FuseFn
from rag.registry import BaseParams, BuildContext, register, validate_params


class _MergeParams(BaseParams):
    top_k: int = Field(default=5, gt=0, description="合併後保留的筆數上限")


@register("fusion", "merge")
def build_merge(params: dict[str, Any], ctx: BuildContext) -> FuseFn:
    """串接各子查詢結果,依 chunk_id 去重(保留最高分),分數降冪取 top_k。

    注意:分數只在同一次檢索內嚴格可比,跨子查詢合併是 MVP 的近似做法;
    需要名次法(RRF)等更穩健的融合時,以 ``method: custom`` 替換本槽位。
    """
    p = validate_params("fusion", "merge", _MergeParams, params)

    def fuse(results: list[list[Document]]) -> list[Document]:
        best: dict[Any, Document] = {}
        for documents in results:
            for document in documents:
                key = document.id or document.metadata.get("chunk_id") or id(document)
                score = document.metadata.get("score", 0.0)
                current = best.get(key)
                if current is None or score > current.metadata.get("score", 0.0):
                    best[key] = document
        merged = sorted(
            best.values(), key=lambda d: d.metadata.get("score", 0.0), reverse=True
        )
        return merged[: p.top_k]

    return fuse
