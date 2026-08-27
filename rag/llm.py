"""方法內部用 LLM 的共用底座(prompt 文字 → 回覆文字)。

query_transformation 的 llm_multi_hyde / preqrag、reranking 的 insertrank
等方法以 ``llm:`` 參數區塊指定連線;generation 槽位獨立設定,不互相
沿用(config 要共用同一組連線時,用 YAML anchor 去重)::

    llm:
      provider: openai_compatible   # 或 mock(離線測試)
      model: gpt-5-mini
      base_url: ${OPENAI_BASE_URL}  # 選填;vLLM / Ollama / 公司閘道
      api_key: ${OPENAI_API_KEY}
      # replies: ["..."]            # provider: mock 用,依序循環
"""

from __future__ import annotations

import re
from typing import Any, Callable, Literal

from pydantic import Field

from rag.errors import APIResponseFormatError, ConfigError
from rag.registry import BaseParams, validate_params
from rag.slots.api_utils import post_json


def message_text(message: Any) -> str:
    """取出訊息的純文字;相容 text 為 property(core 1.x)或方法的版本。"""
    text = getattr(message, "text", None)
    if callable(text):
        text = text()
    if text is None:
        text = str(getattr(message, "content", message))
    return text


class _LlmParams(BaseParams):
    provider: Literal["mock", "openai_compatible", "gateway"] = Field(
        description="mock = 離線固定回覆;openai_compatible = OpenAI SDK"
        "(請求必帶 model;OpenAI / vLLM / Ollama);gateway = 手寫"
        "OpenAI 式 HTTP client(model 選填,None = 請求完全不帶 model "
        "欄位 —— 給不吃 model 的公司閘道)"
    )
    replies: list[str] | None = Field(
        default=None, description="provider: mock 的固定回覆(依序循環)"
    )
    model: str | None = Field(
        default=None,
        description="openai_compatible 必填;gateway 選填(None = 不帶此欄位)",
    )
    base_url: str | None = Field(
        default=None,
        description="openai_compatible:None = 官方 OpenAI;gateway:必填",
    )
    completions_path: str = Field(
        default="/chat/completions", description="gateway 的補全端點路徑"
    )
    api_key: str | None = Field(
        default=None,
        description="openai_compatible:None = 用 OPENAI_API_KEY 環境變數;"
        "gateway:None = 不帶 Authorization 標頭",
    )
    temperature: float | None = Field(default=None, description="None = 不帶此欄位")
    max_tokens: int | None = Field(default=None, description="None = 不帶此欄位")
    timeout: float | None = Field(
        default=None, description="None = client 預設(gateway 為 60 秒)"
    )
    headers: dict[str, str] = Field(default_factory=dict, description="額外 HTTP 標頭")


def _gateway_chat(p: _LlmParams) -> Callable[[list[dict[str, str]]], str]:
    """手寫 OpenAI 式 chat client:messages dict 清單 → 回覆文字。

    與 openai_compatible 的差別:``model`` 為 None 時**請求完全不帶
    model 欄位**(OpenAI SDK 做不到)。共用 api_utils 的錯誤翻譯。
    """
    endpoint = p.base_url.rstrip("/") + p.completions_path
    headers = dict(p.headers)
    if p.api_key is not None:
        headers.setdefault("Authorization", f"Bearer {p.api_key}")

    def chat(messages: list[dict[str, str]]) -> str:
        body: dict[str, Any] = {"messages": messages}
        if p.model is not None:
            body["model"] = p.model
        if p.temperature is not None:
            body["temperature"] = p.temperature
        if p.max_tokens is not None:
            body["max_tokens"] = p.max_tokens
        data = post_json(
            endpoint, body, headers=headers, timeout=p.timeout or 60.0
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            shape = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
            raise APIResponseFormatError(
                f"gateway 回應缺少 choices[0].message.content;"
                f"回應頂層實際的欄位:{shape}"
            ) from exc
        return str(content)

    return chat


def build_text_llm(slot: str, method: str, raw: Any) -> Callable[[str], str]:
    """依 ``llm:`` 參數區塊建立 ``complete(prompt) -> 回覆文字``。

    Raises:
        ConfigError: 缺少 llm 區塊、參數不合法,或 openai_compatible 缺 model。
    """
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(
            f"模組 '{slot}' 方法 '{method}' 需要 llm 參數區塊,例如 "
            "llm: {provider: openai_compatible, model: gpt-5-mini} "
            "或離線測試用 llm: {provider: mock, replies: ['...']}"
        )
    p = validate_params(slot, f"{method}.llm", _LlmParams, raw)

    if p.provider == "mock":
        replies = p.replies or ["(mock LLM 回覆)"]
        state = {"count": 0}

        def complete(prompt: str) -> str:
            reply = replies[state["count"] % len(replies)]
            state["count"] += 1
            return reply

        return complete

    if p.provider == "gateway":
        if p.base_url is None:
            raise ConfigError(
                f"模組 '{slot}' 方法 '{method}' 的 llm 為 gateway 時必須提供 "
                "base_url(公司閘道的 /v1 位址)"
            )
        chat = _gateway_chat(p)

        def complete_via_gateway(prompt: str) -> str:
            return chat([{"role": "user", "content": prompt}])

        return complete_via_gateway

    if p.model is None:
        raise ConfigError(
            f"模組 '{slot}' 方法 '{method}' 的 llm 為 openai_compatible 時"
            "必須提供 model;API 不吃 model 欄位的公司閘道請改用 "
            "provider: gateway(model 選填,None = 請求不帶 model 欄位)"
        )
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {"model": p.model}
    if p.base_url is not None:
        kwargs["base_url"] = p.base_url
    if p.api_key is not None:
        kwargs["api_key"] = p.api_key
    if p.temperature is not None:
        kwargs["temperature"] = p.temperature
    if p.max_tokens is not None:
        kwargs["max_tokens"] = p.max_tokens
    if p.timeout is not None:
        kwargs["timeout"] = p.timeout
    if p.headers:
        kwargs["default_headers"] = p.headers
    llm = ChatOpenAI(**kwargs)

    def complete(prompt: str) -> str:
        return message_text(llm.invoke([HumanMessage(prompt)]))

    return complete


_LISTING_PREFIX = re.compile(r"^(?:[-*•]|\d+[.、)])\s*")


def parse_lines(text: str, cap: int | None = None) -> list[str]:
    """把 LLM 的逐行輸出整理成清單:去空行、去編號/項目符號、去重。"""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _LISTING_PREFIX.sub("", raw_line.strip()).strip()
        if line and line not in lines:
            lines.append(line)
    return lines[:cap] if cap is not None else lines
