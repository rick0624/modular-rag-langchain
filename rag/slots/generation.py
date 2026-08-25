"""槽位:Generation —— LLM 答案生成。契約:messages → 答案文字。

prompt 由框架組好(:mod:`rag.prompts`)才送進來;query() 回傳的
``prompt`` 即實際送 LLM 的內容(可稽核)。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import Field

from rag.interfaces import GenerateFn
from rag.registry import BaseParams, BuildContext, register, validate_params


def message_text(message: Any) -> str:
    """取出訊息的純文字;相容 text 為 property(core 1.x)或方法的版本。"""
    text = getattr(message, "text", None)
    if callable(text):
        text = text()
    if text is None:
        text = str(getattr(message, "content", message))
    return text


class _MockParams(BaseParams):
    replies: list[str] | None = Field(
        default=None, description="固定回覆清單(依序循環);None = 內建預設回覆"
    )


@register("generation", "mock")
def build_mock(params: dict[str, Any], ctx: BuildContext) -> GenerateFn:
    """離線假答案(開發測試用):依序循環回覆固定內容。"""
    p = validate_params("generation", "mock", _MockParams, params)
    replies = p.replies or ["(mock 回覆)已依據檢索到的內容組成答案;接上真實 LLM 請改用 generation: openai_compatible。"]
    state = {"count": 0}

    def generate(messages: list[BaseMessage]) -> str:
        reply = replies[state["count"] % len(replies)]
        state["count"] += 1
        return reply

    return generate


class _OpenAIParams(BaseParams):
    model: str = Field(description="模型名稱")
    base_url: str | None = Field(
        default=None,
        description="None = 官方 OpenAI;vLLM / Ollama / 公司閘道填其 /v1 位址",
    )
    api_key: str | None = Field(
        default=None, description="None = 使用 OPENAI_API_KEY 環境變數"
    )
    temperature: float | None = Field(
        default=None,
        description="None = 不帶此欄位(注意:gpt-5 系列不接受非預設 temperature)",
    )
    max_tokens: int | None = Field(default=None, description="None = 不帶此欄位")
    timeout: float | None = Field(default=None, description="None = client 預設")
    headers: dict[str, str] = Field(default_factory=dict, description="額外 HTTP 標頭")


@register("generation", "openai_compatible")
def build_openai_compatible(params: dict[str, Any], ctx: BuildContext) -> GenerateFn:
    """OpenAI 相容 chat API(langchain-openai 的 ChatOpenAI)。"""
    p = validate_params("generation", "openai_compatible", _OpenAIParams, params)
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

    def generate(messages: list[BaseMessage]) -> str:
        return message_text(llm.invoke(messages))

    return generate
