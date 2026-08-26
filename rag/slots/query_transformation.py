"""槽位:Query Transformation —— 查詢改寫 / 拆解。契約:queries → queries。

輸出 1 筆 = 傳統單查詢流程;輸出 N 筆 = 逐子查詢檢索後由 fusion 合併。
可方法鏈(method 清單依序執行)。
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from rag.interfaces import TransformFn
from rag.llm import build_text_llm, parse_lines
from rag.registry import BaseParams, BuildContext, register, validate_params

logger = logging.getLogger(__name__)


class _PassthroughParams(BaseParams):
    """passthrough 不接受任何參數。"""


@register("query_transformation", "passthrough")
def build_passthrough(params: dict[str, Any], ctx: BuildContext) -> TransformFn:
    """原樣通過(不改寫、不拆解)。"""
    validate_params("query_transformation", "passthrough", _PassthroughParams, params)

    def transform(queries: list[str]) -> list[str]:
        return list(queries)

    return transform


def _dedup(queries: list[str]) -> list[str]:
    seen: list[str] = []
    for query in queries:
        if query not in seen:
            seen.append(query)
    return seen


_MULTI_HYDE_PROMPT = """\
你是檢索輔助系統。請針對以下問題,想像 {num_documents} 篇「可能包含答案」的
文件段落,從不同角度撰寫(Multi-HyDE)。每篇一行,不要編號、不要其他說明。

問題:{query}"""


class _MultiHydeParams(BaseParams):
    num_documents: int = Field(default=3, gt=0, description="假設文件篇數")
    keep_original: bool = Field(
        default=True, description="原查詢也保留一路(與假設文件各自檢索後融合)"
    )
    prompt: str | None = Field(
        default=None, description="自訂 prompt(需含 {query};None = 內建)"
    )
    llm: dict[str, Any] = Field(description="LLM 連線(見 rag/llm.py 的 llm 區塊說明)")


@register("query_transformation", "llm_multi_hyde")
def build_llm_multi_hyde(params: dict[str, Any], ctx: BuildContext) -> TransformFn:
    """Multi-HyDE:LLM 生成 k 篇多角度假設文件,各自當一路查詢去檢索。"""
    p = validate_params("query_transformation", "llm_multi_hyde", _MultiHydeParams, params)
    complete = build_text_llm("query_transformation", "llm_multi_hyde", p.llm)
    template = p.prompt or _MULTI_HYDE_PROMPT

    def transform(queries: list[str]) -> list[str]:
        expanded: list[str] = list(queries) if p.keep_original else []
        for query in queries:
            prompt = template.replace(
                "{num_documents}", str(p.num_documents)
            ).replace("{query}", query)
            documents = parse_lines(complete(prompt), cap=p.num_documents)
            if not documents:
                logger.warning("llm_multi_hyde:LLM 沒有產出假設文件,該路回退原查詢")
            expanded.extend(documents)
        return _dedup(expanded) or list(queries)

    return transform


_PREQRAG_CLASSIFY_PROMPT = """\
判斷以下問題是「單一主題」還是「複合問題」(包含多個需要分別查證的子問題)。
只回答 single 或 multi,不要其他文字。

問題:{query}"""

_PREQRAG_REWRITE_PROMPT = """\
請把以下問題改寫成 {num_rewrites} 種不同說法(語意相同、用詞不同,利於檢索)。
每行一個,不要編號、不要其他說明。

問題:{query}"""

_PREQRAG_DECOMPOSE_PROMPT = """\
請把以下複合問題拆解成最多 {max_subqueries} 個可獨立檢索的子問題。
每行一個,不要編號、不要其他說明。

問題:{query}"""


class _PreqragParams(BaseParams):
    num_rewrites: int = Field(default=2, gt=0, description="single 分支的改寫條數")
    max_subqueries: int = Field(default=4, gt=0, description="multi 分支的子查詢數上限")
    include_original: bool = Field(default=True, description="原查詢也保留一路")
    classify_prompt: str | None = Field(
        default=None, description="自訂分類 prompt(需含 {query};None = 內建)"
    )
    rewrite_prompt: str | None = Field(
        default=None, description="自訂改寫 prompt(需含 {query};None = 內建)"
    )
    decompose_prompt: str | None = Field(
        default=None, description="自訂拆解 prompt(需含 {query};None = 內建)"
    )
    llm: dict[str, Any] = Field(description="LLM 連線(見 rag/llm.py 的 llm 區塊說明)")


@register("query_transformation", "preqrag")
def build_preqrag(params: dict[str, Any], ctx: BuildContext) -> TransformFn:
    """PreQRAG:先分類 single / multi,單一主題改寫、複合問題拆解。"""
    p = validate_params("query_transformation", "preqrag", _PreqragParams, params)
    complete = build_text_llm("query_transformation", "preqrag", p.llm)
    classify_template = p.classify_prompt or _PREQRAG_CLASSIFY_PROMPT
    rewrite_template = p.rewrite_prompt or _PREQRAG_REWRITE_PROMPT
    decompose_template = p.decompose_prompt or _PREQRAG_DECOMPOSE_PROMPT

    def transform(queries: list[str]) -> list[str]:
        expanded: list[str] = []
        for query in queries:
            category = complete(classify_template.replace("{query}", query)).strip().lower()
            is_multi = "multi" in category
            logger.debug("preqrag:%r 分類為 %s", query, "multi" if is_multi else "single")
            if p.include_original:
                expanded.append(query)
            if is_multi:
                prompt = decompose_template.replace(
                    "{max_subqueries}", str(p.max_subqueries)
                ).replace("{query}", query)
                expanded.extend(parse_lines(complete(prompt), cap=p.max_subqueries))
            else:
                prompt = rewrite_template.replace(
                    "{num_rewrites}", str(p.num_rewrites)
                ).replace("{query}", query)
                expanded.extend(parse_lines(complete(prompt), cap=p.num_rewrites))
        return _dedup(expanded) or list(queries)

    return transform
