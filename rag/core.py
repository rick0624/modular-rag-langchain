"""核心流程:config → Runtime → ingest() / query() / evaluate()。

沒有 pipeline graph engine —— 兩條流程都是普通的循序函式呼叫,
trace 是逐步附加的 list of dicts:

- ``ingest``:import → parsing → chunking → 蓋章 meta → 寫入 store。
- ``query``:routing?(支線)→ query_transformation(1 → N 子查詢)→
  逐子查詢 retrieval → reranking → fusion → formatter? → 組 prompt →
  generation。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document

from rag.config import MethodConfig, RAGConfig
from rag.errors import ComponentError, ConfigError
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


def build_runtime(config: RAGConfig) -> Runtime:
    """依 config 建構全部槽位;任何方法或參數錯誤都在這裡直接報錯。"""
    import rag.slots  # noqa: F401  # import 即完成內建方法註冊

    ctx = BuildContext(config=config)
    ingestion = config.ingestion
    inference = config.inference

    importer = build_slot("import", ingestion.import_, ctx)
    parser = build_slot("parsing", ingestion.parsing, ctx)
    chunker = build_slot("chunking", ingestion.chunking, ctx)
    ctx.embeddings = build_slot("embedding", ingestion.embedding, ctx)
    ctx.store = build_slot("indexing", ingestion.indexing, ctx)

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
    )


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


def _traced(
    trace: list[dict[str, Any]], slot: str, cfg: MethodConfig | None, fn: Callable[[], Any]
) -> Any:
    """執行一步並附加 trace 紀錄(slot / method / count / elapsed_ms)。"""
    start = time.perf_counter()
    result = fn()
    entry: dict[str, Any] = {
        "slot": slot,
        "method": "+".join(cfg.methods()) if cfg else None,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
    }
    if isinstance(result, list):
        entry["count"] = len(result)
    trace.append(entry)
    return result


def ingest(runtime: Runtime) -> dict[str, Any]:
    """執行一次全量 ingestion(切片 id 確定 → 重跑即 upsert)。"""
    ingestion = runtime.config.ingestion
    trace: list[dict[str, Any]] = []
    start = time.perf_counter()

    sources = _traced(trace, "import", ingestion.import_, runtime.importer)
    documents = _traced(
        trace, "parsing", ingestion.parsing, lambda: runtime.parser(sources)
    )
    chunks = _traced(
        trace, "chunking", ingestion.chunking, lambda: runtime.chunker(documents)
    )
    chunks = _stamp_chunks(chunks)
    _traced(
        trace,
        "indexing",
        ingestion.indexing,
        lambda: runtime.store.add_documents(chunks, ids=[chunk.id for chunk in chunks]),
    )

    return {
        "documents_written": len(chunks),
        "files": [source.doc_id for source in sources],
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
        "steps": trace,
    }


def query(runtime: Runtime, text: str) -> dict[str, Any]:
    """執行一次查詢,回傳 canonical 結果(answer / prompt / documents / …)。"""
    if not text or not text.strip():
        raise ComponentError("查詢內容不可為空")
    inference = runtime.config.inference
    trace: list[dict[str, Any]] = []

    routing = (
        _traced(trace, "routing", inference.routing, lambda: runtime.route(text))
        if runtime.route
        else None
    )

    subqueries = _traced(
        trace,
        "query_transformation",
        inference.query_transformation,
        lambda: runtime.transform([text]),
    )
    if not subqueries:
        subqueries = [text]

    results: list[list[Document]] = []
    for subquery in subqueries:
        retrieved = _traced(
            trace, "retrieval", inference.retrieval, lambda: runtime.retrieve(subquery)
        )
        reranked = _traced(
            trace,
            "reranking",
            inference.reranking,
            lambda: runtime.rerank(subquery, retrieved),
        )
        results.append(reranked)

    documents = _traced(
        trace, "fusion", inference.fusion, lambda: runtime.fuse(results)
    )

    prompt_cfg = inference.prompt
    messages, prompt = build_messages(
        text,
        documents,
        template=prompt_cfg.template if prompt_cfg else None,
        system=prompt_cfg.system if prompt_cfg else None,
    )
    answer = _traced(
        trace, "generation", inference.generation, lambda: runtime.generate(messages)
    )

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
