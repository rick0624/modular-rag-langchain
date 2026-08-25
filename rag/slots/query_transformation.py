"""槽位:Query Transformation —— 查詢改寫 / 拆解。契約:queries → queries。

輸出 1 筆 = 傳統單查詢流程;輸出 N 筆 = 逐子查詢檢索後由 fusion 合併。
可方法鏈(method 清單依序執行)。
"""

from __future__ import annotations

from typing import Any

from rag.interfaces import TransformFn
from rag.registry import BaseParams, BuildContext, register, validate_params


class _PassthroughParams(BaseParams):
    """passthrough 不接受任何參數。"""


@register("query_transformation", "passthrough")
def build_passthrough(params: dict[str, Any], ctx: BuildContext) -> TransformFn:
    """原樣通過(不改寫、不拆解)。"""
    validate_params("query_transformation", "passthrough", _PassthroughParams, params)

    def transform(queries: list[str]) -> list[str]:
        return list(queries)

    return transform
