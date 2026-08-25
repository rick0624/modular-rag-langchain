"""ingestion 流程:meta 蓋章、doc_id 穩定、upsert 冪等、方法鏈限制。"""

from __future__ import annotations

import pytest

from rag import build_runtime, ingest
from rag.errors import ConfigError, UnknownMethodError
from rag.interfaces import META_KEYS


def test_ingest_stamps_meta_and_upserts(make_config):
    runtime = build_runtime(make_config())
    result = ingest(runtime)

    assert result["documents_written"] > 3  # chunk_size=60,長文會切成多片
    assert result["files"] == ["leave.md", "sub/expense.txt", "vpn.txt"]
    assert [step["slot"] for step in result["steps"]] == [
        "import", "parsing", "chunking", "indexing",
    ]

    chunks = list(runtime.store.store.values())  # InMemoryVectorStore 內部 dict
    for chunk in chunks:
        for key in META_KEYS:
            assert key in chunk["metadata"], f"切片缺少 meta 鍵 {key}"
    ids = [chunk["id"] for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert any(chunk["metadata"]["chunk_id"] == "vpn.txt::chunk_0" for chunk in chunks)
    # seq 從 0 起算且逐文件遞增
    leave_seqs = sorted(
        chunk["metadata"]["seq"]
        for chunk in chunks
        if chunk["metadata"]["doc_id"] == "leave.md"
    )
    assert leave_seqs == list(range(len(leave_seqs)))

    # 重跑 ingest:切片 id 確定 → upsert,不倍增
    again = ingest(runtime)
    assert again["documents_written"] == result["documents_written"]
    assert len(runtime.store.store) == len(chunks)


def test_unknown_method_lists_available(make_config):
    with pytest.raises(UnknownMethodError, match="local_files"):
        build_runtime(make_config(**{"ingestion.import": {"method": "nope"}}))


def test_bad_params_lists_accepted(make_config):
    with pytest.raises(ConfigError, match="可接受的參數"):
        build_runtime(
            make_config(**{"ingestion.chunking": {"method": "recursive", "params": {"typo": 1}}})
        )


def test_chunking_overlap_must_be_smaller(make_config):
    with pytest.raises(ConfigError, match="chunk_overlap"):
        build_runtime(
            make_config(
                **{
                    "ingestion.chunking": {
                        "method": "recursive",
                        "params": {"chunk_size": 50, "chunk_overlap": 50},
                    }
                }
            )
        )


def test_non_chainable_slot_rejects_method_list(make_config):
    with pytest.raises(ConfigError, match="方法鏈"):
        build_runtime(make_config(**{"ingestion.chunking": {"method": ["recursive", "recursive"]}}))


def test_missing_input_dir_raises_at_run(make_config):
    runtime = build_runtime(
        make_config(**{"ingestion.import": {"method": "local_files", "params": {"input_dir": "./no-such-dir"}}})
    )
    with pytest.raises(Exception, match="找不到匯入資料夾"):
        ingest(runtime)
