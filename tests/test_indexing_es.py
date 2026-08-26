"""elasticsearch indexing 的 custom_mapping / settings(離線:假 client,不連 ES)。"""

from __future__ import annotations

import pytest

from rag import build_runtime
from rag.errors import ConfigError
from rag.slots.indexing import _ensure_index


class _FakeIndices:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.created: list[dict] = []

    def exists(self, index: str) -> bool:
        return index in self.existing

    def create(self, index: str, **kwargs) -> None:
        self.created.append({"index": index, **kwargs})
        self.existing.add(index)


class _FakeClient:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.indices = _FakeIndices(existing or set())


def test_ensure_index_creates_with_mapping_and_settings():
    client = _FakeClient()
    mappings = {"properties": {"text": {"type": "text"}, "vector": {"type": "dense_vector", "dims": 8}}}
    settings = {"analysis": {"analyzer": {}}}

    assert _ensure_index(client, "idx", mappings, settings) is True
    assert client.indices.created == [
        {"index": "idx", "mappings": mappings, "settings": settings}
    ]
    # 已存在 → 原樣沿用,不再 create
    assert _ensure_index(client, "idx", mappings, settings) is False
    assert len(client.indices.created) == 1


def test_ensure_index_without_settings_omits_key():
    client = _FakeClient()
    _ensure_index(client, "idx", {"properties": {}}, None)
    assert "settings" not in client.indices.created[0]


def test_client_options_only_carries_set_keys():
    from rag.slots.indexing import _ElasticsearchParams, _client_options

    defaults = _ElasticsearchParams(es_url="http://localhost:9200")
    assert _client_options(defaults) == {}

    tuned = _ElasticsearchParams(
        es_url="http://localhost:9200",
        request_timeout=60,
        retry_on_timeout=True,
        max_retries=5,
    )
    assert _client_options(tuned) == {
        "request_timeout": 60,
        "retry_on_timeout": True,
        "max_retries": 5,
    }


def test_elasticsearchstore_param_names_compatible():
    """守住我們傳給 ElasticsearchStore 的參數名(升版改名時在此炸出來)。"""
    import inspect

    from langchain_elasticsearch import ElasticsearchStore

    accepted = inspect.signature(ElasticsearchStore.__init__).parameters
    for name in (
        "index_name", "embedding", "client", "es_url", "es_user",
        "es_password", "es_api_key", "es_params", "query_field",
        "vector_query_field",
    ):
        assert name in accepted, f"ElasticsearchStore 已無參數 {name}"


def _es_config(make_config, **extra):
    return make_config(
        **{
            "ingestion.indexing": {
                "method": "elasticsearch",
                "params": {"es_url": "http://localhost:9200", **extra},
            }
        }
    )


def test_settings_requires_custom_mapping(make_config):
    with pytest.raises(ConfigError, match="settings 必須搭配 custom_mapping"):
        build_runtime(_es_config(make_config, settings={"analysis": {}}))


def test_custom_mapping_must_declare_content_and_vector_fields(make_config):
    with pytest.raises(ConfigError, match="vector"):
        build_runtime(
            _es_config(make_config, custom_mapping={"properties": {"text": {"type": "text"}}})
        )


def test_custom_mapping_field_names_follow_query_field_overrides(make_config):
    # 欄位名改用 query_field / vector_query_field 宣告時,檢查跟著走
    with pytest.raises(ConfigError, match="content"):
        build_runtime(
            _es_config(
                make_config,
                query_field="content",
                custom_mapping={"properties": {"vector": {"type": "dense_vector"}}},
            )
        )
