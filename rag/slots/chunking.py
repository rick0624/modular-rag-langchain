"""槽位:Chunking —— 文件切片。契約:documents → list[Document]。"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field

from rag.errors import ConfigError
from rag.interfaces import ChunkFn
from rag.registry import BaseParams, BuildContext, register, validate_params


class _RecursiveParams(BaseParams):
    chunk_size: int = Field(default=300, gt=0, description="每片字元數上限")
    chunk_overlap: int = Field(default=50, ge=0, description="相鄰切片重疊字元數")
    separators: list[str] | None = Field(
        default=None,
        description="切分邊界優先序;None = 內建(段落 → 行 → 中文句號 → 空白 → 字元)",
    )


# 內建分隔符加入中文句號:CJK 文字沒有空白,LangChain 預設會直接退到
# 逐字元切,切在句中的機率高。
_DEFAULT_SEPARATORS = ["\n\n", "\n", "。", " ", ""]


@register("chunking", "recursive")
def build_recursive(params: dict[str, Any], ctx: BuildContext) -> ChunkFn:
    """遞迴字元切分(RecursiveCharacterTextSplitter);metadata 逐塊複製。"""
    p = validate_params("chunking", "recursive", _RecursiveParams, params)
    if p.chunk_overlap >= p.chunk_size:
        raise ConfigError(
            f"模組 'chunking' 方法 'recursive' 的 chunk_overlap({p.chunk_overlap})"
            f"必須小於 chunk_size({p.chunk_size})"
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=p.chunk_size,
        chunk_overlap=p.chunk_overlap,
        separators=p.separators if p.separators is not None else _DEFAULT_SEPARATORS,
    )

    def chunk(documents: list[Document]) -> list[Document]:
        return splitter.split_documents(documents)

    return chunk
