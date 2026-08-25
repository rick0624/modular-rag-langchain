"""modular-rag-langchain:模組化 RAG 框架(LangChain 版 MVP)。

公開 API(也可不經 service 直接在 Python 中使用)::

    from rag import load_config, build_runtime, ingest, query, evaluate

    runtime = build_runtime(load_config("configs/default.yaml"))
    ingest(runtime)
    result = query(runtime, "VPN 伺服器位址是多少?")
"""

from rag.config import RAGConfig, load_config, parse_config
from rag.core import Runtime, build_runtime, evaluate, ingest, query
from rag.errors import ComponentError, ConfigError, RagError, UnknownMethodError
from rag.interfaces import EvalCase, Source
from rag.registry import BuildContext, register

__all__ = [
    "BuildContext",
    "ComponentError",
    "ConfigError",
    "EvalCase",
    "RAGConfig",
    "RagError",
    "Runtime",
    "Source",
    "UnknownMethodError",
    "build_runtime",
    "evaluate",
    "ingest",
    "load_config",
    "parse_config",
    "query",
    "register",
]
