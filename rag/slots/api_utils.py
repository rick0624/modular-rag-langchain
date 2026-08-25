"""embedding: api 與 reranking: api 共用的 HTTP/JSON 請求路徑。

外部 API 的請求 / 回應欄位各不相同,以參數對應形狀,不必為每種服務寫
新程式;錯誤一律翻成框架例外,訊息帶 endpoint 與回應內容前 200 字。
"""

from __future__ import annotations

from typing import Any

import httpx

from rag.errors import APICallError, APIResponseFormatError


def post_json(
    endpoint: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    """POST 一個 JSON 請求並回傳解析後的回應。

    Raises:
        APICallError: timeout、連線失敗或非 2xx 狀態碼。
        APIResponseFormatError: 回應不是合法的 JSON。
    """
    try:
        response = httpx.post(endpoint, json=body, headers=headers, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise APICallError(f"呼叫 API 逾時({timeout} 秒):POST {endpoint}") from exc
    except httpx.HTTPError as exc:
        raise APICallError(f"呼叫 API 失敗:POST {endpoint}({exc})") from exc
    if not 200 <= response.status_code < 300:
        preview = response.text[:200]
        raise APICallError(
            f"API 回應非 2xx 狀態碼 {response.status_code}:POST {endpoint}"
            f"(回應內容前 200 字:{preview!r})"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise APIResponseFormatError(
            f"API 回應不是合法的 JSON:POST {endpoint}"
        ) from exc


def locate_list(
    data: Any,
    path: str | None,
    *,
    api_label: str,
    setting_name: str,
    what: str,
) -> list[Any]:
    """依 ``a.b`` 路徑從回應中取出一個清單。

    找不到時,錯誤訊息會列出該層實際存在的欄位 —— 對照你的 API 回應
    調整設定,不必猜。

    Args:
        path: 巢狀路徑;``None`` 表示回應本身就是清單。

    Raises:
        APIResponseFormatError: 路徑不存在,或該處的值不是 list。
    """
    current: Any = data
    if path is not None:
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                hint = (
                    f"該層實際的欄位:{sorted(current.keys())}"
                    if isinstance(current, dict)
                    else f"該層實際型別:{type(current).__name__}"
                )
                raise APIResponseFormatError(
                    f"{api_label}回應中找不到 '{path}'(在 '{part}' 處中斷);{hint}。"
                    f"請把 {setting_name} 設成你的 API 回應中{what}所在的欄位"
                    "(支援 a.b 巢狀路徑;回應本身就是清單時設為 null)"
                )
            current = current[part]
    if not isinstance(current, list):
        location = (
            f"'{path}' 的值" if path is not None else f"{setting_name} 為 null 時,回應本身"
        )
        raise APIResponseFormatError(
            f"{location}必須是 list,實際得到:{type(current).__name__}"
        )
    return current
