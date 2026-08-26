"""槽位:Indexing —— vector store。

本槽位是「產物是物件不是函式」的兩個例外之一:產物為 LangChain
``VectorStore`` 物件(消費 ``ctx.embeddings``);寫入在 ``ingest()``
中以 ``add_documents(ids=chunk_id)`` 進行 → 重跑 ingest 即 upsert。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from pydantic import Field

from rag.errors import ComponentError, ConfigError
from rag.registry import BaseParams, BuildContext, register, validate_params

logger = logging.getLogger(__name__)


class _CommonIndexingParams(BaseParams):
    """所有 indexing 方法共通的參數(由框架的 ingest 流程消費)。"""

    fields: dict[str, str] | None = Field(
        default=None,
        description="自訂欄位白名單 + 改名 {索引欄位名: meta 欄位名};"
        "None = 自訂 meta 欄位全帶。設定後未列出的自訂欄位不寫入索引;"
        "框架欄位(doc_id/seq/page/chunk_id)與 extra_vectors 向量欄位永遠保留",
    )


class _InMemoryParams(_CommonIndexingParams):
    """in_memory 只接受共通參數。"""


@register("indexing", "in_memory")
def build_in_memory(params: dict[str, Any], ctx: BuildContext) -> VectorStore:
    """行程內記憶體索引(開發測試用;service 重啟即消失)。"""
    validate_params("indexing", "in_memory", _InMemoryParams, params)
    if ctx.embeddings is None:
        raise ConfigError("indexing 需要 embedding 槽位先建立(內部建構順序錯誤)")
    return InMemoryVectorStore(embedding=ctx.embeddings)


class _ElasticsearchParams(_CommonIndexingParams):
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
    layout: Literal["nested", "flat"] = Field(
        default="nested",
        description="文件 layout:nested = langchain-elasticsearch 慣例"
        "(自訂欄位在巢狀 metadata.* 內);flat = Haystack 式扁平文件"
        "(所有欄位在頂層,方便外部系統直接讀,由框架自行讀寫)",
    )
    request_timeout: float | None = Field(
        default=None,
        description="單一請求逾時秒數;None = client 預設(10 秒)。整批 bulk "
        "寫入超過時 writer 會拋 Connection timed out",
    )
    retry_on_timeout: bool | None = Field(
        default=None,
        description="逾時是否自動重試;None = client 預設(false)。"
        "線路間歇不穩時開啟(連線層錯誤 client 本來就會重試,本參數只擴及逾時)",
    )
    max_retries: int | None = Field(
        default=None,
        description="單次請求的最大重試次數;None = client 預設(3)",
    )


def _client_options(p: "_ElasticsearchParams") -> dict[str, Any]:
    """組出 ES client 的連線韌性選項(只帶有設定的鍵,其餘交給 client 預設)。"""
    options: dict[str, Any] = {}
    if p.request_timeout is not None:
        options["request_timeout"] = p.request_timeout
    if p.retry_on_timeout is not None:
        options["retry_on_timeout"] = p.retry_on_timeout
    if p.max_retries is not None:
        options["max_retries"] = p.max_retries
    return options


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


class FlatElasticsearchStore:
    """Haystack 式扁平 layout 的 ES store(由框架自行讀寫,不經 langchain)。

    文件形狀:``{query_field: 內文, vector_query_field: 主向量,
    doc_id, seq, page, chunk_id, ...自訂欄位}`` —— 全部頂層欄位,
    外部系統可直接讀,與舊 Haystack 版索引 layout 對齊。

    索引不存在時於首次寫入自動建立(內文 text + 向量 dense_vector,
    dims 取自實際向量;要釘 analyzer / 欄位型別請用 custom_mapping)。
    """

    def __init__(
        self,
        *,
        client: Any,
        index: str,
        embeddings: Any,
        query_field: str = "text",
        vector_query_field: str = "vector",
    ) -> None:
        self.client = client
        self.index = index
        self.embeddings = embeddings
        self.query_field = query_field
        self.vector_query_field = vector_query_field

    def _ensure_default_index(self, dims: int) -> None:
        if self.client.indices.exists(index=self.index):
            return
        self.client.indices.create(
            index=self.index,
            mappings={
                "properties": {
                    self.query_field: {"type": "text"},
                    self.vector_query_field: {"type": "dense_vector", "dims": dims},
                }
            },
        )
        logger.info("已建立扁平 layout 索引 '%s'(dims=%d)", self.index, dims)

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        if not documents:
            return []
        doc_ids = ids or [doc.id for doc in documents]
        vectors = self.embeddings.embed_documents(
            [doc.page_content for doc in documents]
        )
        self._ensure_default_index(dims=len(vectors[0]))
        operations: list[dict[str, Any]] = []
        for doc_id, document, vector in zip(doc_ids, documents, vectors):
            clash = {self.query_field, self.vector_query_field} & set(document.metadata)
            if clash:
                raise ComponentError(
                    f"切片 '{doc_id}' 的 metadata 欄位 {sorted(clash)} 與扁平 "
                    f"layout 的內文/向量欄位名衝突;請改欄位名,或以 "
                    "query_field / vector_query_field 錯開"
                )
            operations.append({"index": {"_index": self.index, "_id": doc_id}})
            operations.append(
                {
                    self.query_field: document.page_content,
                    self.vector_query_field: vector,
                    **document.metadata,
                }
            )
        response = self.client.bulk(operations=operations, refresh="wait_for")
        if response.get("errors"):
            failed = [
                item["index"].get("error")
                for item in response.get("items", [])
                if item.get("index", {}).get("error")
            ]
            raise ComponentError(
                f"扁平 layout bulk 寫入失敗 {len(failed)} 筆;第一筆錯誤:{failed[:1]}"
            )
        return list(doc_ids)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        query_vector = self.embeddings.embed_query(query)
        response = self.client.search(
            index=self.index,
            knn={
                "field": self.vector_query_field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": max(50, k * 5),
            },
            size=k,
        )
        results: list[tuple[Document, float]] = []
        for hit in response["hits"]["hits"]:
            source = dict(hit["_source"])
            content = source.pop(self.query_field, "")
            source.pop(self.vector_query_field, None)  # 主向量不進 metadata
            results.append(
                (
                    Document(id=hit["_id"], page_content=content, metadata=source),
                    float(hit["_score"]),
                )
            )
        return results


def _build_client(p: "_ElasticsearchParams") -> Any:
    from elasticsearch import Elasticsearch

    kwargs: dict[str, Any] = _client_options(p)
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
    if p.custom_mapping is not None:
        declared = sorted(p.custom_mapping.get("properties", {}))
        for field in (p.query_field, p.vector_query_field):
            if field not in p.custom_mapping.get("properties", {}):
                raise ConfigError(
                    f"custom_mapping 的 properties 缺少欄位 '{field}'"
                    f"(實際宣告的欄位:{declared or '(無)'})。mapping 是完整"
                    "覆蓋,須自含內文欄位"
                    f"(query_field: {p.query_field})與向量欄位"
                    f"(vector_query_field: {p.vector_query_field},"
                    "type: dense_vector,dims = embedding 維度)。"
                    "若 mapping 沿用其他欄位名(如舊 Haystack 版的 content / "
                    "embedding),請同步設定 query_field / vector_query_field "
                    "指向它們,讀寫與本檢查都會跟著走"
                )

    if p.layout == "flat":
        client = _build_client(p)
        if p.custom_mapping is not None:
            _ensure_index(client, p.index, p.custom_mapping, p.settings)
        return FlatElasticsearchStore(
            client=client,
            index=p.index,
            embeddings=ctx.embeddings,
            query_field=p.query_field,
            vector_query_field=p.vector_query_field,
        )

    from langchain_elasticsearch import ElasticsearchStore

    store_kwargs: dict[str, Any] = {
        "index_name": p.index,
        "embedding": ctx.embeddings,
        "query_field": p.query_field,
        "vector_query_field": p.vector_query_field,
    }
    if p.custom_mapping is not None:
        client = _build_client(p)
        _ensure_index(client, p.index, p.custom_mapping, p.settings)
        return ElasticsearchStore(client=client, **store_kwargs)
    return ElasticsearchStore(
        es_url=p.es_url,
        es_user=p.username,
        es_password=p.password,
        es_api_key=p.api_key,
        es_params=_client_options(p) or None,
        **store_kwargs,
    )
