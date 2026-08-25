"""槽位:Routing —— 查詢分類(選填支線)。契約:query → route dict。

吃**原始查詢**(不經 transform);結果附加於 query() 輸出的 ``routing``
鍵,不影響檢索行為。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from rag.interfaces import RouteFn
from rag.registry import BaseParams, BuildContext, register, validate_params


class _KeywordMatchParams(BaseParams):
    routes: dict[str, list[str]] = Field(description="類別 → 關鍵字清單")
    default_category: str = Field(default="general", description="無命中時的類別")


@register("routing", "keyword_match")
def build_keyword_match(params: dict[str, Any], ctx: BuildContext) -> RouteFn:
    """關鍵字規則分類:回傳第一個命中的類別與命中的關鍵字。"""
    p = validate_params("routing", "keyword_match", _KeywordMatchParams, params)

    def route(query: str) -> dict:
        for category, keywords in p.routes.items():
            matched = [keyword for keyword in keywords if keyword in query]
            if matched:
                return {"category": category, "matched_keywords": matched}
        return {"category": p.default_category, "matched_keywords": []}

    return route
