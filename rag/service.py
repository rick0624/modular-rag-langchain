"""FastAPI service:單一 app 同時提供 ingestion 與 inference。

啟動::

    python -m rag.service --config configs/default.yaml
    # 或 RAG_CONFIG=configs/default.yaml python -m rag.service

Endpoints:GET /health、POST /ingest、POST /query、POST /evaluate。

並行模型:單一 ``threading.Lock`` 包住所有 pipeline 執行與 runtime
置換,uvicorn 請維持單一 worker(in_memory 索引在行程內,多 worker
會各有一份互不相通的索引)。

啟動時的自動 ingest 由環境變數 ``RAG_INGEST_ON_STARTUP`` 控制:
``auto``(預設)= indexing 為 in_memory 時 ingest(揮發性索引空著沒有
意義),elasticsearch 等持久索引則跳過;``always`` / ``never`` 強制。
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from rag import core
from rag.config import load_config
from rag.errors import ConfigError, RagError
from rag.interfaces import EvalCase

logger = logging.getLogger(__name__)

_INGEST_MODES = ("auto", "always", "never")


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="查詢文字")


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[EvalCase] | None = Field(
        default=None, description="行內測試案例;省略時使用 config 的 evaluation 設定"
    )


def _document_payload(document: Any) -> dict[str, Any]:
    """把 Document 轉成回應用的 dict(框架保證的 meta 鍵 + 內容)。"""
    return {
        "chunk_id": document.metadata.get("chunk_id"),
        "doc_id": document.metadata.get("doc_id"),
        "page": document.metadata.get("page"),
        "score": document.metadata.get("score"),
        "content": document.page_content,
    }


def _should_ingest_on_startup(runtime: core.Runtime) -> bool:
    mode = os.environ.get("RAG_INGEST_ON_STARTUP", "auto")
    if mode not in _INGEST_MODES:
        raise ConfigError(
            f"RAG_INGEST_ON_STARTUP 必須是 {'/'.join(_INGEST_MODES)} 之一,"
            f"實際得到:{mode!r}"
        )
    if mode == "always":
        return True
    if mode == "never":
        return False
    return runtime.config.ingestion.indexing.methods()[0] == "in_memory"


def create_app(config_path: str) -> FastAPI:
    """載入 config、建構 Runtime,回傳 FastAPI app(建構失敗即 fail-fast)。"""
    runtime = core.build_runtime(load_config(config_path))

    app = FastAPI(
        title="modular-rag-langchain",
        description="模組化 RAG service:config 驅動、方法可替換(見 configs/)",
    )
    app.state.config_path = config_path
    app.state.runtime = runtime
    app.state.lock = threading.Lock()

    if _should_ingest_on_startup(runtime):
        result = core.ingest(runtime)
        logger.info(
            "啟動時已 ingest:%d 個切片(來源:%s)",
            result["documents_written"], result["files"],
        )

    @app.exception_handler(RagError)
    async def _rag_error_handler(request: Request, exc: RagError) -> JSONResponse:
        logger.warning("請求失敗(%s):%s", type(exc).__name__, exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        rt: core.Runtime = app.state.runtime
        return {
            "status": "ok",
            "config_path": app.state.config_path,
            "indexing_method": rt.config.ingestion.indexing.methods()[0],
        }

    @app.post("/ingest")
    def run_ingest() -> dict[str, Any]:
        """重讀 config → 全量重建 runtime 與索引 → 置換服務狀態。

        upsert 語意:切片 id 由內容位置決定,重跑不會倍增;但已刪除的
        來源檔在**持久索引(elasticsearch)**中會留下舊切片 —— MVP 不做
        淘汰,需要時請換索引名或先清索引。
        """
        with app.state.lock:
            new_runtime = core.build_runtime(load_config(app.state.config_path))
            result = core.ingest(new_runtime)
            app.state.runtime = new_runtime
            return result

    @app.post("/query")
    def run_query(request: QueryRequest) -> dict[str, Any]:
        with app.state.lock:
            result = core.query(app.state.runtime, request.query)
        result["documents"] = [
            _document_payload(document) for document in result["documents"]
        ]
        return result

    @app.post("/evaluate")
    def run_evaluate(request: EvaluateRequest | None = None) -> dict[str, Any]:
        cases = request.cases if request else None
        with app.state.lock:
            return core.evaluate(app.state.runtime, cases)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="modular-rag-langchain service")
    parser.add_argument(
        "--config",
        default=os.environ.get("RAG_CONFIG", "configs/default.yaml"),
        help="YAML 設定檔路徑(預設:$RAG_CONFIG 或 configs/default.yaml)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    import uvicorn

    # workers 固定為 1:索引與 lock 都在行程內(見模組 docstring)。
    uvicorn.run(create_app(args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
