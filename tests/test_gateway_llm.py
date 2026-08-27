"""provider: gateway 與 generation: gateway_openai_compatible(離線假 httpx)。"""

from __future__ import annotations

from typing import Any

import pytest

from rag import build_runtime, ingest, query
from rag.errors import APIResponseFormatError, ConfigError
from rag.llm import build_text_llm
from rag.slots import api_utils


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = str(payload)

    def json(self) -> Any:
        return self._payload


@pytest.fixture()
def fake_post(monkeypatch):
    calls: list[dict[str, Any]] = []
    state = {"payload": {"choices": [{"message": {"content": "閘道回覆"}}]}}

    def _post(endpoint, json=None, headers=None, timeout=None):
        calls.append({"endpoint": endpoint, "json": json, "headers": headers})
        return _FakeResponse(state["payload"])

    monkeypatch.setattr(api_utils.httpx, "post", _post)

    def _configure(payload: Any) -> list[dict[str, Any]]:
        state["payload"] = payload
        return calls

    return _configure


def test_gateway_omits_model_field_by_default(fake_post):
    calls = fake_post({"choices": [{"message": {"content": "回覆"}}]})
    complete = build_text_llm(
        "query_transformation", "llm_multi_hyde",
        {"provider": "gateway", "base_url": "https://llm.test/v1", "api_key": "tok"},
    )
    assert complete("問題") == "回覆"
    request = calls[0]
    assert request["endpoint"] == "https://llm.test/v1/chat/completions"
    assert "model" not in request["json"]  # ← 重點:請求完全不帶 model 欄位
    assert request["json"]["messages"] == [{"role": "user", "content": "問題"}]
    assert request["headers"]["Authorization"] == "Bearer tok"


def test_gateway_includes_model_when_set(fake_post):
    calls = fake_post({"choices": [{"message": {"content": "回覆"}}]})
    complete = build_text_llm(
        "reranking", "insertrank",
        {"provider": "gateway", "base_url": "https://llm.test/v1", "model": "m1",
         "temperature": 0.2},
    )
    complete("問題")
    assert calls[0]["json"]["model"] == "m1"
    assert calls[0]["json"]["temperature"] == 0.2


def test_gateway_requires_base_url():
    with pytest.raises(ConfigError, match="base_url"):
        build_text_llm("query_transformation", "llm_multi_hyde", {"provider": "gateway"})


def test_gateway_bad_response_lists_actual_keys(fake_post):
    fake_post({"result": "不是 OpenAI 形狀"})
    complete = build_text_llm(
        "query_transformation", "llm_multi_hyde",
        {"provider": "gateway", "base_url": "https://llm.test/v1"},
    )
    with pytest.raises(APIResponseFormatError, match="result"):
        complete("問題")


def test_multi_hyde_with_gateway_end_to_end(make_config, fake_post):
    fake_post({"choices": [{"message": {"content": "假設一\n假設二"}}]})
    runtime = build_runtime(
        make_config(
            **{
                "inference.query_transformation": {
                    "method": "llm_multi_hyde",
                    "params": {
                        "llm": {"provider": "gateway", "base_url": "https://llm.test/v1"}
                    },
                }
            }
        )
    )
    assert runtime.transform(["原問題"]) == ["原問題", "假設一", "假設二"]


def test_generation_gateway_converts_message_roles(make_config, fake_post):
    calls = fake_post({"choices": [{"message": {"content": "答案"}}]})
    runtime = build_runtime(
        make_config(
            **{
                "inference.generation": {
                    "method": "gateway_openai_compatible",
                    "params": {"base_url": "https://llm.test/v1"},
                },
                "inference.prompt": {"system": "只根據內容回答。"},
            }
        )
    )
    ingest(runtime)
    result = query(runtime, "VPN 位址?")
    assert result["answer"] == "答案"
    messages = calls[-1]["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "model" not in calls[-1]["json"]
