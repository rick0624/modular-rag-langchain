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
    query_field: str = Field(
        default="text", description="索引中放切片內文的欄位名(LangChain 慣例預設 text)"
    )
    vector_query_field: str = Field(
        default="vector",
        description="索引中放向量的欄位名(dense_vector;LangChain 慣例預設 vector)",
    )
    custom_mapping: dict[str, Any] | None = Field(
        default=None,
        description="自訂 index mapping(完整覆蓋,須自含 query_field 與 "
        "vector_query_field 兩個欄位的宣告);僅在索引不存在、由框架建立時生效",
    )
    settings: dict[str, Any] | None = Field(
        default=None,
        description="index settings(analyzer、index.default_pipeline 等);"
        "必須搭配 custom_mapping",
    )


def _ensure_index(
    client: Any,
    index: str,
    mappings: dict[str, Any],
    settings: dict[str, Any] | None,
) -> bool:
    """索引不存在時以自訂 mapping / settings 建立;已存在則沿用(不比對)。

    Returns:
        True = 本次建立了索引;False = 索引已存在。
    """
    if client.indices.exists(index=index):
        return False
    kwargs: dict[str, Any] = {"mappings": mappings}
    if settings is not None:
        kwargs["settings"] = settings
    client.indices.create(index=index, **kwargs)
    return True


def _build_client(p: "_ElasticsearchParams") -> Any:
    from elasticsearch import Elasticsearch

    kwargs: dict[str, Any] = {}
    if p.api_key is not None:
        kwargs["api_key"] = p.api_key
    if p.username is not None:
        kwargs["basic_auth"] = (p.username, p.password)
    return Elasticsearch(p.es_url, **kwargs)


@register("indexing", "elasticsearch")
def build_elasticsearch(params: dict[str, Any], ctx: BuildContext) -> VectorStore:
    """Elasticsearch 索引(langchain-elasticsearch;向量檢索,索引持久化)。

    ``custom_mapping``(可加 ``settings``)提供時,建構期會連線 ES:
    索引不存在就以其建立,已存在則原樣沿用 —— 改 mapping 請換索引名或
    先刪除重建。
    """
    p = validate_params("indexing", "elasticsearch", _ElasticsearchParams, params)
    if ctx.embeddings is None:
        raise ConfigError("indexing 需要 embedding 槽位先建立(內部建構順序錯誤)")
    if (p.username is None) != (p.password is None):
        raise ConfigError(
            "indexing 方法 'elasticsearch' 的 username 與 password 必須成對提供"
        )
    if p.settings is not None and p.custom_mapping is None:
        raise ConfigError(
            "indexing 方法 'elasticsearch' 的 settings 必須搭配 custom_mapping"
            "(否則框架無從得知內文與向量欄位的宣告);"
            "只想自訂 mapping 時可單獨設定 custom_mapping"
        )
    from langchain_elasticsearch import ElasticsearchStore

    store_kwargs: dict[str, Any] = {
        "index_name": p.index,
        "embedding": ctx.embeddings,
        "query_field": p.query_field,
        "vector_query_field": p.vector_query_field,
    }
    if p.custom_mapping is not None:
        for field in (p.query_field, p.vector_query_field):
            if field not in p.custom_mapping.get("properties", {}):
                raise ConfigError(
                    f"custom_mapping 的 properties 缺少欄位 '{field}';"
                    "mapping 是完整覆蓋,須自含內文欄位"
                    f"(query_field: {p.query_field})與向量欄位"
                    f"(vector_query_field: {p.vector_query_field},"
                    "type: dense_vector,dims = embedding 維度)"
                )
        client = _build_client(p)
        _ensure_index(client, p.index, p.custom_mapping, p.settings)
        return ElasticsearchStore(es_connection=client, **store_kwargs)
    return ElasticsearchStore(
        es_url=p.es_url,
        es_user=p.username,
        es_password=p.password,
        es_api_key=p.api_key,
        **store_kwargs,
    )
