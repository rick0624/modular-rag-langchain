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


def test_setup_logging_writes_debug_file(tmp_path, make_config):
    from rag.logging_setup import setup_logging

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    log_path = tmp_path / "logs" / "rag.log"  # 父目錄不存在 → 自動建立
    try:
        setup_logging(console_level="warning", log_file=log_path)
        runtime = build_runtime(make_config())
        ingest(runtime)
        query(runtime, "VPN 伺服器位址?")
        for handler in logging.getLogger().handlers:
            handler.flush()
        text = log_path.read_text(encoding="utf-8")
        # terminal 等級是 warning,但檔案收到 DEBUG 全量
        assert "DEBUG rag.core: [import]" in text
        assert "[prompt]" in text
        assert "query 完成" in text
    finally:
        for handler in root.handlers[:]:
            if handler not in saved_handlers:
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_default_log_file_has_timestamp(tmp_path):
    import re

    from rag.logging_setup import default_log_file

    path = default_log_file(tmp_path / "logs")
    assert path.parent == tmp_path / "logs"
    assert re.fullmatch(r"rag-\d{8}-\d{6}\.log", path.name), path.name


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