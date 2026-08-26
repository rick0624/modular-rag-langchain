"""pipeline 的 debug log:每一步在 DEBUG 等級都有紀錄。"""

from __future__ import annotations

import logging

from rag import build_runtime, ingest, query


def test_debug_logs_cover_each_step(make_config, caplog):
    runtime = build_runtime(make_config())
    with caplog.at_level(logging.DEBUG, logger="rag.core"):
        ingest(runtime)
        query(runtime, "VPN 伺服器位址?")

    for marker in [
        "ingest 開始",
        "[import]",
        "[parsing]",
        "[chunking]",
        "ingest 完成",
        "query 開始",
        "[query_transformation]",
        "[retrieval]",
        "[reranking]",
        "[fusion]",
        "[prompt]",
        "[generation]",
        "query 完成",
    ]:
        assert marker in caplog.text, f"缺少 log 標記:{marker}"


def test_step_failure_logged_with_slot_name(make_config, caplog):
    runtime = build_runtime(
        make_config(
            **{
                "ingestion.import": {
                    "method": "local_files",
                    "params": {"input_dir": "./no-such-dir"},
                }
            }
        )
    )
    with caplog.at_level(logging.ERROR, logger="rag.core"):
        try:
            ingest(runtime)
        except Exception:
            pass
    assert "[import] 失敗" in caplog.text