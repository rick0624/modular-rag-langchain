"""槽位:Parsing —— 來源轉純文字 Document。契約:sources → list[Document]。"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pydantic import Field

from rag.errors import ComponentError
from rag.interfaces import ParseFn, Source
from rag.registry import BaseParams, BuildContext, register, validate_params


class _TextParams(BaseParams):
    encoding: str = Field(default="utf-8", description="文字檔編碼")


@register("parsing", "text")
def build_text(params: dict[str, Any], ctx: BuildContext) -> ParseFn:
    """純文字 parser(txt / md):一份來源 → 一個 Document。

    PDF 等其他格式請以 ``method: custom`` 掛自訂 parser
    (契約相同:sources → documents)。
    """
    p = validate_params("parsing", "text", _TextParams, params)

    def parse(sources: list[Source]) -> list[Document]:
        documents: list[Document] = []
        for source in sources:
            if source.text is not None:
                content = source.text
            elif source.path is not None:
                try:
                    content = source.path.read_text(encoding=p.encoding)
                except (OSError, UnicodeDecodeError) as exc:
                    raise ComponentError(
                        f"讀取來源 '{source.doc_id}' 失敗({source.path}):{exc}"
                    ) from exc
            else:
                raise ComponentError(
                    f"來源 '{source.doc_id}' 既沒有 path 也沒有 text,無法解析"
                )
            documents.append(
                Document(
                    page_content=content,
                    metadata={"doc_id": source.doc_id, "page": 1, **source.meta},
                )
            )
        return documents

    return parse
