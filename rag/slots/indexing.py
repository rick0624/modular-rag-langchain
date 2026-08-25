"""槽位:Indexing —— vector store。

本槽位是「產物是物件不是函式」的兩個例外之一:產物為 LangChain
``VectorStore`` 物件(消費 ``ctx.embeddings``);寫入在 ``ingest()``
中以 ``add_documents(ids=chunk_id)`` 進行 → 重跑 ingest 即 upsert。
"""

from __future__ import annotations

from typing import Any

from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from pydantic import Field

from rag.errors import ConfigError
from rag.registry import BaseParams, BuildContext, register, validate_params


class _InMemoryParams(BaseParams):
    """in_memory 不接受任何參數。"""


@register("indexing", "in_memory")
def build_in_memory(params: dict[str, Any], ctx: BuildContext) -> VectorStore:
    """行程內記憶體索引(開發測試用;service 重啟即消失)。"""
    validate_params("indexing", "in_memory", _InMemoryParams, params)
    if ctx.embeddings is None:
        raise ConfigError("indexing 需要 embedding 槽位先建立(內部建構順序錯誤)")
    return InMemoryVectorStore(embedding=ctx.embeddings)


class _ElasticsearchParams(BaseParams):
    es_url: str = Field(description="Elasticsearch 位址,如 http://localhost:9200")
    index: str = Field(
        default="modular-rag",
        description="索引名稱。dense_vector 的維度建立後不可變,"
        "換 embedding 模型/維度時請換索引名(或先刪除重建)",
    )
    username: str | None = Field(default=None, description="basic auth,需與 password 成對")
    password: str | None = Field(default=None)
    api_key: str | None = Field(default=None, description="API key 認證(與帳密擇一)")


@register("indexing", "elasticsearch")
def build_elasticsearch(params: dict[str, Any], ctx: BuildContext) -> VectorStore:
    """Elasticsearch 索引(langchain-elasticsearch;向量檢索,索引持久化)。"""
    p = validate_params("indexing", "elasticsearch", _ElasticsearchParams, params)
    if ctx.embeddings is None:
        raise ConfigError("indexing 需要 embedding 槽位先建立(內部建構順序錯誤)")
    if (p.username is None) != (p.password is None):
        raise ConfigError(
            "indexing 方法 'elasticsearch' 的 username 與 password 必須成對提供"
        )
    from langchain_elasticsearch import ElasticsearchStore

    return ElasticsearchStore(
        index_name=p.index,
        embedding=ctx.embeddings,
        es_url=p.es_url,
        es_user=p.username,
        es_password=p.password,
        es_api_key=p.api_key,
    )
