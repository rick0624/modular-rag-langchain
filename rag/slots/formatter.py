"""槽位:Formatter —— 對外格式(選填終端支線)。

契約:(query, documents, answer) → payload(型別自由 —— 終端槽位特權;
走 HTTP 時須可 JSON 序列化)。結果進 query() 輸出的 ``output`` 鍵,
canonical 的 answer / documents 等鍵照舊。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pydantic import Field

from rag.interfaces import FormatFn
from rag.registry import BaseParams, BuildContext, register, validate_params


class _SimpleJsonParams(BaseParams):
    include_content: bool = Field(
        default=True, description="false = 引用只留識別資訊,不含切片內容"
    )


@register("formatter", "simple_json")
def build_simple_json(params: dict[str, Any], ctx: BuildContext) -> FormatFn:
    """通用 JSON 形狀:answer + 引用清單(chunk_id / doc_id / page / score)。"""
    p = validate_params("formatter", "simple_json", _SimpleJsonParams, params)

    def format_payload(query: str, documents: list[Document], answer: str | None) -> Any:
        references = []
        for document in documents:
            reference: dict[str, Any] = {
                "chunk_id": document.metadata.get("chunk_id"),
                "doc_id": document.metadata.get("doc_id"),
                "page": document.metadata.get("page"),
                "score": document.metadata.get("score"),
            }
            if p.include_content:
                reference["content"] = document.page_content
            references.append(reference)
        return {"query": query, "answer": answer, "references": references}

    return format_payload
