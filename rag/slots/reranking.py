"""槽位:Reranking —— 檢索結果重排。契約:(query, documents) → documents。

只能重排 / 過濾 / 改分,不得改動切片內容。可方法鏈。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.documents import Document
from pydantic import Field

from rag.errors import APIResponseFormatError, RagError
from rag.interfaces import RerankFn
from rag.llm import build_text_llm
from rag.registry import BaseParams, BuildContext, register, validate_params
from rag.slots.api_utils import locate_list, post_json

logger = logging.getLogger(__name__)


class _NoneParams(BaseParams):
    """none 不接受任何參數。"""


@register("reranking", "none")
def build_none(params: dict[str, Any], ctx: BuildContext) -> RerankFn:
    """不重排(保留檢索順序)。"""
    validate_params("reranking", "none", _NoneParams, params)

    def rerank(query: str, documents: list[Document]) -> list[Document]:
        return list(documents)

    return rerank


_INSERTRANK_PROMPT = """\
請依「與問題的相關程度」重新排序以下候選段落。每個段落附有{score_label}
(僅供參考,InsertRank:分數與內容一併判斷)。只輸出排序後的段落編號,
以逗號分隔(例如:2,1,3),不要其他文字。

問題:{query}

{documents}"""


class _InsertRankParams(BaseParams):
    top_k: int = Field(default=5, gt=0, description="重排後保留筆數")
    score_label: str = Field(
        default="檢索分數",
        description="prompt 中分數的名稱,依上游據實描述(vector 檢索為相似度)",
    )
    prompt: str | None = Field(
        default=None,
        description="自訂 prompt(需含 {query} 與 {documents};None = 內建)",
    )
    llm: dict[str, Any] = Field(description="LLM 連線(見 rag/llm.py 的 llm 區塊說明)")


def _parse_ranking(reply: str, total: int) -> list[int] | None:
    """把 LLM 回覆解析成 1-based 名次序;完全解析不出來回傳 None。"""
    order: list[int] = []
    for match in re.findall(r"\d+", reply):
        number = int(match)
        if 1 <= number <= total and number not in order:
            order.append(number)
    return order or None


@register("reranking", "insertrank")
def build_insertrank(params: dict[str, Any], ctx: BuildContext) -> RerankFn:
    """InsertRank:候選附檢索分數的 LLM listwise 重排。

    解析不出 LLM 的排序回覆時 fail-soft:保留原檢索順序前 top_k 筆並記
    WARNING;LLM 回覆中缺漏的候選補在後面(依原順序)。
    """
    p = validate_params("reranking", "insertrank", _InsertRankParams, params)
    complete = build_text_llm("reranking", "insertrank", p.llm)
    template = p.prompt or _INSERTRANK_PROMPT

    def rerank(query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        lines = [
            f"[{position}] ({p.score_label}={document.metadata.get('score', 0.0):.4f}) "
            f"{document.page_content}"
            for position, document in enumerate(documents, start=1)
        ]
        prompt = (
            template.replace("{score_label}", p.score_label)
            .replace("{query}", query)
            .replace("{documents}", "\n".join(lines))
        )
        reply = complete(prompt)
        order = _parse_ranking(reply, len(documents))
        if order is None:
            logger.warning(
                "insertrank:無法解析 LLM 排序回覆(%r),保留原檢索順序", reply[:80]
            )
            return list(documents[: p.top_k])
        chosen = set(order)
        reranked = [documents[position - 1] for position in order]
        reranked.extend(
            document
            for position, document in enumerate(documents, start=1)
            if position not in chosen
        )
        return reranked[: p.top_k]

    return rerank


class _ApiParams(BaseParams):
    endpoint: str = Field(description="rerank API 的完整 URL")
    headers: dict[str, str] = Field(default_factory=dict, description="額外 HTTP 標頭(認證放這裡)")
    model: str | None = Field(default=None, description="None = 請求不帶 model 欄位")
    top_k: int = Field(default=5, gt=0, description="重排後保留筆數")
    timeout: float = Field(default=30.0, gt=0, description="逾時秒數")
    query_field: str = Field(default="question", description="請求中放查詢的欄位名")
    documents_field: str = Field(default="documents", description="請求中放候選文字清單的欄位名")
    model_field: str = Field(default="model", description="請求中放模型的欄位名")
    results_field: str | None = Field(
        default="returnData",
        description="回應中重排結果清單的欄位(a.b 巢狀路徑;回應本身是清單時設 null)",
    )
    index_field: str = Field(default="index", description="結果元素中候選序號的欄位名")
    score_field: str = Field(default="score", description="結果元素中分數的欄位名")
    index_base: int = Field(default=0, description="回應 index 的起算值(0 或 1;設錯會整體位移)")
    higher_is_better: bool = Field(
        default=True, description="false = API 回傳的是距離(越小越相關)"
    )
    raise_on_failure: bool = Field(
        default=False,
        description="false = API 掛掉時保留原檢索順序並記 WARNING(fail-soft);"
        "初次接線建議設 true 讓錯誤直接炸出來",
    )


@register("reranking", "api")
def build_api(params: dict[str, Any], ctx: BuildContext) -> RerankFn:
    """通用 HTTP rerank API:一次把 query 與全部候選送出,依回傳分數重排。"""
    p = validate_params("reranking", "api", _ApiParams, params)

    def rerank(query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        try:
            return _rerank(p, query, documents)
        except RagError as exc:
            if p.raise_on_failure:
                raise
            # fail-soft:重排失敗仍有檢索順序可用,不讓查詢整條掛掉。
            logger.warning(
                "fail-soft:rerank API 失敗(%s: %s),保留原檢索順序前 %d 筆",
                type(exc).__name__, exc, p.top_k,
            )
            return list(documents[: p.top_k])

    return rerank


def _rerank(p: _ApiParams, query: str, documents: list[Document]) -> list[Document]:
    body: dict[str, Any] = {
        p.query_field: query,
        p.documents_field: [doc.page_content for doc in documents],
    }
    if p.model is not None:
        body[p.model_field] = p.model
    data = post_json(p.endpoint, body, headers=p.headers, timeout=p.timeout)
    results = locate_list(
        data,
        p.results_field,
        api_label="rerank API ",
        setting_name="results_field",
        what="重排結果清單",
    )
    scored = _parse_results(p, results, len(documents))
    # API 不保證已排序,一律自己排;兩段穩定排序 = 同分時回到送出順序。
    scored.sort(key=lambda pair: pair[0])
    scored.sort(key=lambda pair: pair[1], reverse=p.higher_is_better)
    reranked: list[Document] = []
    for position, score in scored[: p.top_k]:
        document = documents[position]
        document.metadata["score"] = score
        reranked.append(document)
    return reranked


def _parse_results(
    p: _ApiParams, results: list[Any], total: int
) -> list[tuple[int, float]]:
    """把回應清單轉成 ``(送出清單中的位置, 分數)``,並擋掉無效項目。"""
    pairs: list[tuple[int, float]] = []
    seen: set[int] = set()
    out_of_range: list[int] = []
    for order, item in enumerate(results):
        if not isinstance(item, dict):
            raise APIResponseFormatError(
                f"rerank API 結果清單的第 {order} 個元素必須是物件,"
                f"實際得到:{type(item).__name__}"
            )
        for field in (p.index_field, p.score_field):
            if field not in item:
                raise APIResponseFormatError(
                    f"rerank API 結果的第 {order} 個元素缺少 '{field}' 欄位;"
                    f"實際的欄位:{sorted(item.keys())}。"
                    "請用 index_field / score_field 對應你的 API 回應"
                )
        raw_index = item[p.index_field]
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            raise APIResponseFormatError(
                f"rerank API 結果的第 {order} 個元素的 '{p.index_field}' 必須是整數,"
                f"實際得到:{raw_index!r}"
            )
        try:
            score = float(item[p.score_field])
        except (TypeError, ValueError) as exc:
            raise APIResponseFormatError(
                f"rerank API 結果的第 {order} 個元素的 '{p.score_field}' 不是數字:"
                f"{item[p.score_field]!r}"
            ) from exc
        position = raw_index - p.index_base
        if not 0 <= position < total:
            out_of_range.append(raw_index)
            continue
        if position in seen:
            logger.warning("rerank API 回傳重複的 index %d,已忽略後者", raw_index)
            continue
        seen.add(position)
        pairs.append((position, score))
    if out_of_range and not pairs:
        raise APIResponseFormatError(
            f"rerank API 回傳的 index 全數越界(送出 {total} 筆,收到 "
            f"{out_of_range[:5]});index_base 目前是 {p.index_base},"
            f"若你的 API 是 {1 - p.index_base} 起算請改設 index_base: {1 - p.index_base}"
        )
    if out_of_range:
        logger.warning(
            "rerank API 回傳 %d 個越界的 index(送出 %d 筆:%s),已忽略",
            len(out_of_range), total, out_of_range[:5],
        )
    return pairs
