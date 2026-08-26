"""elasticsearch 的 layout: flat(扁平文件;離線以假 client 測讀寫形狀)。"""

from __future__ import annotations

import pytest

from langchain_core.documents import Document

from rag.errors import ComponentError
from rag.slots.embedding import HashedNgramEmbeddings
from rag.slots.indexing import FlatElasticsearchStore


class _FakeIndices:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = existing or set()
        self.created: list[dict] = []

    def exists(self, index: str) -> bool:
        return index in self.existing

    def create(self, index: str, **kwargs) -> None:
        self.created.append({"index": index, **kwargs})
        self.existing.add(index)


class _FakeClient:
    def __init__(self) -> None:
        self.indices = _FakeIndices()
        self.bulk_calls: list[dict] = []
        self.search_response = {"hits": {"hits": []}}

    def bulk(self, operations, refresh=None):
        self.bulk_calls.append({"operations": operations, "refresh": refresh})
        return {"errors": False, "items": []}

    def search(self, **kwargs):
        self.last_search = kwargs
        return self.search_response


def _store(client: _FakeClient) -> FlatElasticsearchStore:
    return FlatElasticsearchStore(
        client=client,
        index="idx",
        embeddings=HashedNgramEmbeddings(dim=8),
        query_field="text",
        vector_query_field="vector",
    )


def _chunk(doc_id: str, content: str, **meta) -> Document:
    metadata = {"doc_id": doc_id, "seq": 0, "page": 1, "chunk_id": f"{doc_id}::chunk_0", **meta}
    return Document(id=metadata["chunk_id"], page_content=content, metadata=metadata)


def test_add_documents_writes_flat_source_and_upsert_ids():
    client = _FakeClient()
    store = _store(client)
    chunk = _chunk("a.txt", "內文一", summary="摘要")

    ids = store.add_documents([chunk], ids=[chunk.id])

    assert ids == ["a.txt::chunk_0"]
    operations = client.bulk_calls[0]["operations"]
    assert operations[0] == {"index": {"_index": "idx", "_id": "a.txt::chunk_0"}}
    source = operations[1]
    # 全頂層:內文 / 向量 / 框架欄位 / 自訂欄位,沒有巢狀 metadata
    assert source["text"] == "內文一"
    assert len(source["vector"]) == 8
    assert source["doc_id"] == "a.txt" and source["chunk_id"] == "a.txt::chunk_0"
    assert source["summary"] == "摘要"
    assert "metadata" not in source
    # 索引不存在 → 以預設 mapping 自動建立(dims 取自實際向量)
    created = client.indices.created[0]
    assert created["mappings"]["properties"]["vector"] == {
        "type": "dense_vector", "dims": 8,
    }


def test_add_documents_metadata_clash_with_text_field_raises():
    store = _store(_FakeClient())
    chunk = _chunk("a.txt", "內文", text="撞名")
    with pytest.raises(ComponentError, match="text"):
        store.add_documents([chunk], ids=[chunk.id])


def test_similarity_search_rebuilds_documents_from_flat_source():
    client = _FakeClient()
    client.indices.existing.add("idx")
    client.search_response = {
        "hits": {
            "hits": [
                {
                    "_id": "a.txt::chunk_0",
                    "_score": 1.5,
                    "_source": {
                        "text": "內文一",
                        "vector": [0.0] * 8,
                        "doc_id": "a.txt",
                        "seq": 0,
                        "page": 1,
                        "chunk_id": "a.txt::chunk_0",
                        "summary": "摘要",
                    },
                }
            ]
        }
    }
    store = _store(client)

    results = store.similarity_search_with_score("查詢", k=3)

    document, score = results[0]
    assert score == 1.5
    assert document.id == "a.txt::chunk_0"
    assert document.page_content == "內文一"
    # 頂層欄位裝回 metadata;內文與主向量不重複進 metadata
    assert document.metadata["chunk_id"] == "a.txt::chunk_0"
    assert document.metadata["summary"] == "摘要"
    assert "text" not in document.metadata and "vector" not in document.metadata
    # kNN 查詢參數
    assert client.last_search["knn"]["field"] == "vector"
    assert client.last_search["knn"]["k"] == 3


def test_flat_layout_builder_wiring(make_config, monkeypatch):
    """layout: flat 走 FlatElasticsearchStore,且通過槽位 duck-type 驗證。"""
    import rag.slots.indexing as indexing_module
    from rag import build_runtime

    fake = _FakeClient()
    monkeypatch.setattr(indexing_module, "_build_client", lambda p: fake)
    runtime = build_runtime(
        make_config(
            **{
                "ingestion.indexing": {
                    "method": "elasticsearch",
                    "params": {"es_url": "http://localhost:9200", "layout": "flat"},
                }
            }
        )
    )
    assert isinstance(runtime.store, FlatElasticsearchStore)
