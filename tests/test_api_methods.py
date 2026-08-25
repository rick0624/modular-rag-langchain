"""embedding: api 與 reranking: api 的欄位對映(以假 httpx 回應測,不碰網路)。"""

from __future__ import annotations

from typing import Any

import pytest

from rag.errors import APIResponseFormatError
from rag.registry import BuildContext
from rag.slots import api_utils
from rag.slots.embedding import build_api as build_api_embedding
from rag.slots.reranking import build_api as build_api_rerank
from langchain_core.documents import Document


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> Any:
        return self._payload


@pytest.fixture()
def fake_post(monkeypatch):
    """把 api_utils 的 httpx.post 換成假實作;回傳可設定回應與觀察請求。"""
    calls: list[dict[str, Any]] = []
    state = {"payload": None}

    def _post(endpoint, json=None, headers=None, timeout=None):
        calls.append({"endpoint": endpoint, "json": json, "headers": headers})
        return _FakeResponse(state["payload"])

    monkeypatch.setattr(api_utils.httpx, "post", _post)

    def _configure(payload: Any) -> list[dict[str, Any]]:
        state["payload"] = payload
        return calls

    return _configure


def _ctx() -> BuildContext:
    return BuildContext(config=None)


def test_api_embedding_openai_shape(fake_post):
    calls = fake_post({"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]})
    embeddings = build_api_embedding(
        {"endpoint": "https://api.test/embed", "model": "m1"}, _ctx()
    )
    vectors = embeddings.embed_documents(["甲", "乙"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert calls[0]["json"] == {"input": ["甲", "乙"], "model": "m1"}

    fake_post({"data": [{"embedding": [1, 0]}]})
    assert embeddings.embed_query("甲") == [1.0, 0.0]


def test_api_embedding_custom_shape_and_batching(fake_post):
    calls = fake_post({"result": {"vectors": [[0.5, 0.5]]}})
    embeddings = build_api_embedding(
        {
            "endpoint": "https://api.test/embed",
            "texts_field": "texts",
            "embeddings_field": "result.vectors",
            "item_field": None,
            "batch_size": 1,
        },
        _ctx(),
    )
    vectors = embeddings.embed_documents(["甲", "乙"])
    assert vectors == [[0.5, 0.5], [0.5, 0.5]]
    assert len(calls) == 2  # batch_size=1 → 兩批
    assert calls[0]["json"] == {"texts": ["甲"]}


def test_api_embedding_bad_field_names_actual_keys(fake_post):
    fake_post({"vectors": [[1.0]]})
    embeddings = build_api_embedding({"endpoint": "https://api.test/embed"}, _ctx())
    with pytest.raises(APIResponseFormatError, match="vectors"):
        embeddings.embed_query("甲")


def _docs(n: int) -> list[Document]:
    return [
        Document(page_content=f"內容{i}", metadata={"chunk_id": f"d::chunk_{i}"})
        for i in range(n)
    ]


def test_api_rerank_reorders_and_scores(fake_post):
    calls = fake_post(
        {"returnData": [{"index": 2, "score": 0.9}, {"index": 0, "score": 0.5}]}
    )
    rerank = build_api_rerank(
        {"endpoint": "https://api.test/rerank", "top_k": 2}, _ctx()
    )
    result = rerank("問題", _docs(3))
    assert [doc.metadata["chunk_id"] for doc in result] == ["d::chunk_2", "d::chunk_0"]
    assert [doc.metadata["score"] for doc in result] == [0.9, 0.5]
    assert calls[0]["json"] == {"question": "問題", "documents": ["內容0", "內容1", "內容2"]}


def test_api_rerank_index_base_one(fake_post):
    fake_post({"returnData": [{"index": 1, "score": 1.0}]})
    rerank = build_api_rerank(
        {"endpoint": "https://api.test/rerank", "index_base": 1}, _ctx()
    )
    result = rerank("問題", _docs(2))
    assert result[0].metadata["chunk_id"] == "d::chunk_0"


def test_api_rerank_all_out_of_range_hints_index_base(fake_post):
    fake_post({"returnData": [{"index": 3, "score": 1.0}]})
    rerank = build_api_rerank(
        {"endpoint": "https://api.test/rerank", "raise_on_failure": True}, _ctx()
    )
    with pytest.raises(APIResponseFormatError, match="index_base"):
        rerank("問題", _docs(2))


def test_api_rerank_fail_soft_keeps_retrieval_order(fake_post):
    fake_post({"unexpected": []})  # results_field 找不到 → 預設 fail-soft
    rerank = build_api_rerank(
        {"endpoint": "https://api.test/rerank", "top_k": 2}, _ctx()
    )
    result = rerank("問題", _docs(3))
    assert [doc.metadata["chunk_id"] for doc in result] == ["d::chunk_0", "d::chunk_1"]
