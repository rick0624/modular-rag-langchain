"""核心流程:config → Runtime → ingest() / query() / evaluate()。

沒有 pipeline graph engine —— 兩條流程都是普通的循序函式呼叫,
trace 是逐步附加的 list of dicts:

- ``ingest``:import → parsing → chunking → 蓋章 meta → 寫入 store。
- ``query``:routing?(支線)→ query_transformation(1 → N 子查詢)→
  逐子查詢 retrieval → reranking → fusion → formatter? → 組 prompt →
  generation。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.documents import Document

from rag.config import MethodConfig, RAGConfig
from rag.errors import ComponentError, ConfigError
from rag.interfaces import META_KEYS

logger = logging.getLogger(__name__)
from rag.interfaces import (
    ChunkFn,
    EvalCase,
    EvaluateFn,
    FormatFn,
    FuseFn,
    GenerateFn,
    ImportFn,
    ParseFn,
    RerankFn,
    RetrieveFn,
    RouteFn,
    TransformFn,
)
from rag.prompts import build_messages
from rag.registry import BuildContext, build_slot
from rag.slots.evaluation import load_default_cases


class SourceFieldEmbeddings:
    """embedding 的包裝:ingest 時把「拿去 embed 的文字」換成指定 meta 欄位。

    vector store 在 ``add_documents`` 內部會以 ``page_content`` 呼叫
    ``embed_documents`` —— 本包裝維護一個「替換文字」佇列:ingest 先以
    :meth:`feed` 依切片順序放入 ``source_field`` 欄位的文字,store 來
    要向量時就改 embed 佇列裡的文字(依序消化,子批次也正確)。
    佇列為空時原樣委派;``embed_query``(查詢端)永遠原樣委派 ——
    同向量空間不變。
    """

    def __init__(self, inner: Any, source_field: str) -> None:
        self.inner = inner
        self.source_field = source_field
        self._queue: deque[str] = deque()

    def feed(self, texts: list[str]) -> None:
        self._queue.extend(texts)

    def drain(self) -> int:
        """清空佇列並回傳殘留數(正常 ingest 後應為 0)。"""
        leftover = len(self._queue)
        self._queue.clear()
        return leftover

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._queue:
            return self.inner.embed_documents(texts)
        if len(self._queue) < len(texts):
            raise ComponentError(
                f"source_field 替換佇列只剩 {len(self._queue)} 筆,但 store 要求 "
                f"embed {len(texts)} 筆 —— 自訂 store 的 add_documents 呼叫"
                "embed_documents 的次數或順序與傳入的切片不一致"
            )
        replaced = [self._queue.popleft() for _ in texts]
        return self.inner.embed_documents(replaced)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


@dataclass
class Runtime:
    """一份 config 建構出的全部槽位產物(service 的常駐狀態)。"""

    config: RAGConfig
    importer: ImportFn
    parser: ParseFn
    chunker: ChunkFn
    embeddings: Any
    store: Any
    transform: TransformFn
    retrieve: RetrieveFn
    rerank: RerankFn
    fuse: FuseFn
    generate: GenerateFn
    route: RouteFn | None
    format: FormatFn | None
    evaluator: EvaluateFn | None
    source_field: str | None = None
    extra_vectors: dict[str, str] = field(default_factory=dict)
    index_fields: dict[str, str] = field(default_factory=dict)


def build_runtime(config: RAGConfig, *, store: Any = None) -> Runtime:
    """依 config 建構全部槽位;任何方法或參數錯誤都在這裡直接報錯。

    Args:
        store: 注入既有的 vector store(跳過 indexing 槽位建構)。
            供實驗掃描等場景讓 ingestion 相同的多組 config 共用同一個
            索引 —— 注入的 store 必須由**相同的 ingestion 設定**建出
            (同 embedding 設定才保證同向量空間)。
    """
    import rag.slots  # noqa: F401  # import 即完成內建方法註冊

    ctx = BuildContext(config=config)
    ingestion = config.ingestion
    inference = config.inference

    importer = build_slot("import", ingestion.import_, ctx)
    parser = build_slot("parsing", ingestion.parsing, ctx)
    chunker = build_slot("chunking", ingestion.chunking, ctx)
    ctx.embeddings = build_slot("embedding", ingestion.embedding, ctx)
    source_field, extra_vectors = _embedding_field_options(ingestion.embedding)
    if source_field:
        # store 持有的是包裝後的物件 → indexing / 查詢端都走同一份。
        ctx.embeddings = SourceFieldEmbeddings(ctx.embeddings, source_field)
    ctx.store = store if store is not None else build_slot(
        "indexing", ingestion.indexing, ctx
    )

    transform = build_slot(
        "query_transformation", inference.query_transformation, ctx
    )
    retrieve = build_slot("retrieval", inference.retrieval, ctx)
    rerank = build_slot("reranking", inference.reranking, ctx)
    fuse = build_slot(
        "fusion", inference.fusion or MethodConfig(method="merge"), ctx
    )
    generate = build_slot("generation", inference.generation, ctx)
    route = build_slot("routing", inference.routing, ctx) if inference.routing else None
    formatter = (
        build_slot("formatter", inference.formatter, ctx) if inference.formatter else None
    )
    evaluator = (
        build_slot("evaluation", config.evaluation, ctx) if config.evaluation else None
    )

    return Runtime(
        config=config,
        importer=importer,
        parser=parser,
        chunker=chunker,
        embeddings=ctx.embeddings,
        store=ctx.store,
        transform=transform,
        retrieve=retrieve,
        rerank=rerank,
        fuse=fuse,
        generate=generate,
        route=route,
        format=formatter,
        evaluator=evaluator,
        source_field=source_field,
        extra_vectors=extra_vectors,
        index_fields=_indexing_field_options(ingestion.indexing, extra_vectors),
    )


def _embedding_field_options(cfg: MethodConfig) -> tuple[str | None, dict[str, str]]:
    """讀出 embedding 的 source_field / extra_vectors 並驗證欄位名。

    這兩個參數由**框架的 ingest 流程**消費(embedding 產物看不到
    metadata),所有 embedding 方法(含 custom)都支援。
    """
    params = cfg.params_for()
    source_field = params.get("source_field")
    extra_vectors = params.get("extra_vectors") or {}
    if source_field is not None and (
        not isinstance(source_field, str) or not source_field
    ):
        raise ConfigError(
            "embedding 的 source_field 必須是非空字串(chunking 生成的 "
            f"metadata 欄位名),實際得到:{source_field!r}"
        )
    if not isinstance(extra_vectors, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and key and value
        for key, value in extra_vectors.items()
    ):
        raise ConfigError(
            "embedding 的 extra_vectors 必須是 {向量欄位名: 來源欄位} 的"
            f"字串對映,實際得到:{extra_vectors!r}"
        )
    reserved = set(META_KEYS) | {"score"}
    bad = sorted(reserved & set(extra_vectors))
    if bad:
        raise ConfigError(
            f"extra_vectors 的向量欄位名不可使用框架保留名 {bad};"
            f"保留名:{sorted(reserved)}"
        )
    return source_field, dict(extra_vectors)


def _stamp_chunks(chunks: list[Document]) -> list[Document]:
    """為切片蓋上框架保證的 metadata 鍵(見 docs/interfaces.md)。

    - ``doc_id``:來源文件識別碼(必須由 parsing / 自訂 chunker 保留)。
    - ``seq``:文件內切片序號(0 起)。
    - ``page``:來源頁碼(1 起;無分頁概念時為 1)。
    - ``chunk_id``:``"{doc_id}::chunk_{seq}"``,同時設為 ``Document.id``
      → 同輸入必同 id,重跑 ingest 即 upsert 不倍增。
    """
    seq_by_doc: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata.get("doc_id")
        if not doc_id:
            raise ComponentError(
                "切片缺少 metadata['doc_id'];parsing 產出的 Document 必須帶 "
                "doc_id,且自訂 chunker 切片時必須逐塊保留 metadata"
            )
        seq = seq_by_doc.get(doc_id, 0)
        seq_by_doc[doc_id] = seq + 1
        chunk.metadata["seq"] = seq
        chunk.metadata.setdefault("page", 1)
        chunk.metadata["chunk_id"] = f"{doc_id}::chunk_{seq}"
        chunk.id = chunk.metadata["chunk_id"]
    return chunks


def _preview(text: str, limit: int = 120) -> str:
    """log 用的單行預覽(截斷 + 摺疊換行)。"""
    flat = str(text).replace("\n", "\\n")
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _doc_scores(documents: list[Document], limit: int = 5) -> list[tuple[str, Any]]:
    """log 用:前 N 筆的 (chunk_id, score)。"""
    return [
        (
            doc.metadata.get("chunk_id", doc.id or "?"),
            round(doc.metadata["score"], 4) if "score" in doc.metadata else None,
        )
        for doc in documents[:limit]
    ]


def _traced(
    trace: list[dict[str, Any]], slot: str, cfg: MethodConfig | None, fn: Callable[[], Any]
) -> Any:
    """執行一步並附加 trace 紀錄(slot / method / count / elapsed_ms)。

    同時打兩條 DEBUG log(開始 / 完成);步驟丟例外時記 ERROR 標明
    是哪個槽位、哪個方法炸掉,再原樣往上拋。
    """
    method = "+".join(cfg.methods()) if cfg else None
    logger.debug("[%s] 開始(method=%s)", slot, method)
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        logger.error(
            "[%s] 失敗(method=%s):%s: %s", slot, method, type(exc).__name__, exc
        )
        raise
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    entry: dict[str, Any] = {"slot": slot, "method": method, "elapsed_ms": elapsed_ms}
    if isinstance(result, list):
        entry["count"] = len(result)
        logger.debug("[%s] 完成:%d 筆,%.1f ms", slot, len(result), elapsed_ms)
    else:
        logger.debug("[%s] 完成:%.1f ms", slot, elapsed_ms)
    trace.append(entry)
    return result


def ingest(runtime: Runtime) -> dict[str, Any]:
    """執行一次全量 ingestion(切片 id 確定 → 重跑即 upsert)。"""
    ingestion = runtime.config.ingestion
    trace: list[dict[str, Any]] = []
    start = time.perf_counter()

    logger.info("ingest 開始")
    sources = _traced(trace, "import", ingestion.import_, runtime.importer)
    logger.debug(
        "[import] 來源 %d 筆:%s%s",
        len(sources),
        [source.doc_id for source in sources[:10]],
        "…" if len(sources) > 10 else "",
    )
    documents = _traced(
        trace, "parsing", ingestion.parsing, lambda: runtime.parser(sources)
    )
    logger.debug(
        "[parsing] %d 份文件,總字數 %d",
        len(documents),
        sum(len(doc.page_content) for doc in documents),
    )
    chunks = _traced(
        trace, "chunking", ingestion.chunking, lambda: runtime.chunker(documents)
    )
    chunks = _stamp_chunks(chunks)
    if logger.isEnabledFor(logging.DEBUG):
        per_doc: dict[str, int] = {}
        for chunk in chunks:
            per_doc[chunk.metadata["doc_id"]] = per_doc.get(chunk.metadata["doc_id"], 0) + 1
        logger.debug("[chunking] %d 個切片;每文件切片數:%s", len(chunks), per_doc)
    if runtime.extra_vectors:
        logger.debug("[embedding] extra_vectors:%s", runtime.extra_vectors)
        _add_extra_vectors(runtime, chunks)
    if runtime.source_field:
        logger.debug("[embedding] source_field=%s:主向量改用該欄位", runtime.source_field)
        runtime.embeddings.feed(_field_texts(chunks, runtime.source_field, "source_field"))
    if runtime.index_fields:
        # 必須在 extra_vectors 與 source_field 取值「之後」:白名單丟掉的
        # 欄位仍可作為向量來源。
        _apply_index_fields(runtime, chunks)
    try:
        _traced(
            trace,
            "indexing",
            ingestion.indexing,
            lambda: runtime.store.add_documents(
                chunks, ids=[chunk.id for chunk in chunks]
            ),
        )
    finally:
        if runtime.source_field:
            leftover = runtime.embeddings.drain()
            if leftover:
                logger.warning(
                    "source_field 替換佇列殘留 %d 筆(store 實際 embed 的切片數"
                    "少於傳入數),已清空", leftover,
                )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "ingest 完成:寫入 %d 個切片(%d 份來源),%.0f ms",
        len(chunks), len(sources), elapsed_ms,
    )
    return {
        "documents_written": len(chunks),
        "files": [source.doc_id for source in sources],
        "elapsed_ms": elapsed_ms,
        "steps": trace,
    }


def _indexing_field_options(
    cfg: MethodConfig, extra_vectors: dict[str, str]
) -> dict[str, str]:
    """讀出 indexing 的 fields(白名單 + 改名)並驗證欄位名。

    ``fields: {索引欄位名: meta 欄位名}``:設定後只有列出的自訂欄位會寫入
    索引(改用左邊的名字);框架欄位(doc_id/seq/page/chunk_id)與
    extra_vectors 的向量欄位永遠保留。由框架的 ingest 流程消費,
    所有 indexing 方法(含 custom)都支援。
    """
    fields = cfg.params_for().get("fields") or {}
    if not isinstance(fields, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and key and value
        for key, value in fields.items()
    ):
        raise ConfigError(
            "indexing 的 fields 必須是 {索引欄位名: meta 欄位名} 的字串對映,"
            f"實際得到:{fields!r}"
        )
    reserved = set(META_KEYS) | {"score"}
    bad = sorted(reserved & set(fields))
    if bad:
        raise ConfigError(
            f"indexing 的 fields 目標欄位名不可使用框架保留名 {bad}"
            f"(框架欄位永遠自動保留,不需列入);保留名:{sorted(reserved)}"
        )
    clash = sorted(set(extra_vectors) & set(fields))
    if clash:
        raise ConfigError(
            f"indexing 的 fields 目標欄位名 {clash} 與 extra_vectors 的向量"
            "欄位名衝突;向量欄位自動保留,不需列入 fields"
        )
    return dict(fields)


def _apply_index_fields(runtime: Runtime, chunks: list[Document]) -> None:
    """依 fields 白名單重組切片 metadata(寫入索引前的最後一步)。

    重組後:框架欄位 + extra_vectors 向量欄位 + fields 列出的欄位
    (改名);未列出的自訂欄位不寫入。來源欄位缺漏時跳過該鍵(自訂
    欄位可能非每個切片都有),不視為錯誤。
    """
    keep = set(META_KEYS) | set(runtime.extra_vectors)
    for chunk in chunks:
        original = chunk.metadata
        rebuilt = {key: original[key] for key in keep if key in original}
        for target, source in runtime.index_fields.items():
            if source in original:
                rebuilt[target] = original[source]
        chunk.metadata = rebuilt
    logger.debug(
        "[indexing] fields 白名單生效:%s(框架欄位與向量欄位自動保留)",
        runtime.index_fields,
    )


def _field_texts(chunks: list[Document], field_name: str, purpose: str) -> list[str]:
    """依切片順序取出指定 metadata 欄位的文字;缺欄位直接報錯。"""
    texts: list[str] = []
    for chunk in chunks:
        value = chunk.metadata.get(field_name)
        if not isinstance(value, str) or not value:
            raise ComponentError(
                f"切片 '{chunk.id}' 缺少 embedding 來源欄位 '{field_name}'"
                f"({purpose});請確認 chunking(或 custom chunker)有為每個"
                "切片生成該 metadata 欄位"
            )
        texts.append(value)
    return texts


def _add_extra_vectors(runtime: Runtime, chunks: list[Document]) -> None:
    """為每組 extra_vectors 算好向量,寫進切片 metadata(隨 metadata 落入索引)。

    內建檢索只用主向量;額外向量供 custom retrieval 使用。ES 上要對
    額外向量做 kNN,須以 custom_mapping 宣告 ``metadata.<欄位>`` 為
    dense_vector,否則落入 dynamic mapping 變普通 float 陣列。
    """
    for vector_field, source in runtime.extra_vectors.items():
        texts = _field_texts(chunks, source, f"extra_vectors.{vector_field}")
        vectors = runtime.embeddings.embed_documents(texts)  # 佇列空 → 原樣委派
        for chunk, vector in zip(chunks, vectors):
            chunk.metadata[vector_field] = vector


def query(runtime: Runtime, text: str) -> dict[str, Any]:
    """執行一次查詢,回傳 canonical 結果(answer / prompt / documents / …)。"""
    if not text or not text.strip():
        raise ComponentError("查詢內容不可為空")
    inference = runtime.config.inference
    trace: list[dict[str, Any]] = []
    start = time.perf_counter()
    logger.info("query 開始:%s", _preview(text))

    routing = (
        _traced(trace, "routing", inference.routing, lambda: runtime.route(text))
        if runtime.route
        else None
    )
    if routing is not None:
        logger.debug("[routing] 結果:%s", routing)

    subqueries = _traced(
        trace,
        "query_transformation",
        inference.query_transformation,
        lambda: runtime.transform([text]),
    )
    if not subqueries:
        subqueries = [text]
    logger.debug(
        "[query_transformation] %d 條子查詢:%s",
        len(subqueries),
        [_preview(subquery, 60) for subquery in subqueries],
    )

    results: list[list[Document]] = []
    for subquery in subqueries:
        retrieved = _traced(
            trace, "retrieval", inference.retrieval, lambda: runtime.retrieve(subquery)
        )
        logger.debug(
            "[retrieval] 子查詢 %r → %d 筆,前段:%s",
            _preview(subquery, 40), len(retrieved), _doc_scores(retrieved),
        )
        reranked = _traced(
            trace,
            "reranking",
            inference.reranking,
            lambda: runtime.rerank(subquery, retrieved),
        )
        logger.debug(
            "[reranking] 重排後 %d 筆,前段:%s", len(reranked), _doc_scores(reranked)
        )
        results.append(reranked)

    documents = _traced(
        trace,
        "fusion",
        inference.fusion or MethodConfig(method="merge"),  # 省略時實際生效的是 merge
        lambda: runtime.fuse(results),
    )
    logger.debug(
        "[fusion] 融合後 %d 筆:%s", len(documents), _doc_scores(documents, limit=10)
    )

    prompt_cfg = inference.prompt
    messages, prompt = build_messages(
        text,
        documents,
        template=prompt_cfg.template if prompt_cfg else None,
        system=prompt_cfg.system if prompt_cfg else None,
    )
    logger.debug("[prompt] 實際送 LLM 的內容(%d 字):\n%s", len(prompt), prompt)
    answer = _traced(
        trace, "generation", inference.generation, lambda: runtime.generate(messages)
    )
    logger.debug("[generation] 答案(%d 字):%s", len(answer), _preview(answer))

    output = (
        _traced(
            trace,
            "formatter",
            inference.formatter,
            lambda: runtime.format(text, documents, answer),
        )
        if runtime.format
        else None
    )
    if output is not None:
        logger.debug("[formatter] payload 型別:%s", type(output).__name__)

    logger.info(
        "query 完成:%d 筆引用,answer %d 字,%.0f ms",
        len(documents), len(answer), (time.perf_counter() - start) * 1000,
    )
    return {
        "answer": answer,
        "prompt": prompt,
        "documents": documents,
        "subqueries": subqueries,
        "routing": routing,
        "output": output,
        "trace": trace,
    }


def evaluate(runtime: Runtime, cases: list[EvalCase] | None = None) -> dict[str, Any]:
    """逐題執行 query() 並以 evaluation 槽位計算指標。

    Args:
        cases: 測試案例;None 時使用 config 的 evaluation 參數
            (行內 cases 或 dataset_path)。

    Raises:
        ConfigError: 配置沒有 evaluation 區塊,又沒有提供 cases。
    """
    evaluator = runtime.evaluator
    if evaluator is None:
        if cases is None:
            raise ConfigError(
                "配置沒有 evaluation 區塊,也沒有提供測試案例;"
                "請在 config 加上 evaluation 或在請求中提供 cases"
            )
        # 有行內案例時,用預設的 retrieval_metrics 評估。
        evaluator = build_slot(
            "evaluation",
            MethodConfig(method="retrieval_metrics"),
            BuildContext(config=runtime.config),
        )
    if cases is None:
        cases = load_default_cases(runtime.config.evaluation)
    results = [query(runtime, case.query) for case in cases]
    return evaluator(cases, results)
