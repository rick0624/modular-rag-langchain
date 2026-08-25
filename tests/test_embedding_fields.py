"""embedding 的 source_field / extra_vectors(框架層參數,全離線)。"""

from __future__ import annotations

import pytest

from rag import build_runtime, ingest, query
from rag.errors import ComponentError, ConfigError
from rag.slots.embedding import HashedNgramEmbeddings


@pytest.fixture()
def field_chunker(tmp_path):
    """custom chunker:每份文件一個切片,並生成 summary / title 欄位。"""
    path = tmp_path / "field_chunker.py"
    path.write_text(
        "def build(params, ctx):\n"
        "    def chunk(documents):\n"
        "        for doc in documents:\n"
        "            doc.metadata['summary'] = doc.page_content[:5]\n"
        "            doc.metadata['title'] = doc.metadata['doc_id']\n"
        "        return documents\n"
        "    return chunk\n",
        encoding="utf-8",
    )
    return {"method": "custom", "params": {"file": str(path)}}


def test_source_field_embeds_chosen_field(make_config, field_chunker):
    runtime = build_runtime(
        make_config(
            **{
                "ingestion.chunking": field_chunker,
                "ingestion.embedding": {
                    "method": "mock",
                    "params": {"dim": 128, "source_field": "summary"},
                },
            }
        )
    )
    ingest(runtime)

    row = runtime.store.store["vpn.txt::chunk_0"]  # InMemoryVectorStore 內部 dict
    reference = HashedNgramEmbeddings(128)
    # 主向量來自 summary 欄位,不是切片內文
    assert row["vector"] == reference.embed_query(row["metadata"]["summary"])
    assert row["vector"] != reference.embed_query(row["text"])
    # 查詢端不受影響,prompt 仍給切片內文
    result = query(runtime, "VPN 伺服器位址?")
    assert result["documents"]
    assert result["documents"][0].page_content in result["prompt"]


def test_source_field_missing_raises(make_config, field_chunker):
    runtime = build_runtime(
        make_config(
            **{
                "ingestion.chunking": field_chunker,
                "ingestion.embedding": {
                    "method": "mock",
                    "params": {"source_field": "nope"},
                },
            }
        )
    )
    with pytest.raises(ComponentError, match="nope"):
        ingest(runtime)


def test_extra_vectors_written_to_metadata(make_config, field_chunker):
    runtime = build_runtime(
        make_config(
            **{
                "ingestion.chunking": field_chunker,
                "ingestion.embedding": {
                    "method": "mock",
                    "params": {"dim": 64, "extra_vectors": {"title_vector": "title"}},
                },
            }
        )
    )
    ingest(runtime)

    row = runtime.store.store["vpn.txt::chunk_0"]
    reference = HashedNgramEmbeddings(64)
    # 額外向量寫進 metadata,主向量仍來自切片內文
    assert row["metadata"]["title_vector"] == reference.embed_query(
        row["metadata"]["title"]
    )
    assert row["vector"] == reference.embed_query(row["text"])


def test_source_field_and_extra_vectors_together(make_config, field_chunker):
    runtime = build_runtime(
        make_config(
            **{
                "ingestion.chunking": field_chunker,
                "ingestion.embedding": {
                    "method": "mock",
                    "params": {
                        "dim": 64,
                        "source_field": "summary",
                        "extra_vectors": {"title_vector": "title"},
                    },
                },
            }
        )
    )
    ingest(runtime)
    row = runtime.store.store["leave.md::chunk_0"]
    reference = HashedNgramEmbeddings(64)
    assert row["vector"] == reference.embed_query(row["metadata"]["summary"])
    assert row["metadata"]["title_vector"] == reference.embed_query("leave.md")


def test_extra_vectors_reserved_name_rejected(make_config, field_chunker):
    with pytest.raises(ConfigError, match="保留名"):
        build_runtime(
            make_config(
                **{
                    "ingestion.chunking": field_chunker,
                    "ingestion.embedding": {
                        "method": "mock",
                        "params": {"extra_vectors": {"chunk_id": "title"}},
                    },
                }
            )
        )


def test_source_field_must_be_nonempty_string(make_config):
    with pytest.raises(ConfigError, match="source_field"):
        build_runtime(
            make_config(
                **{
                    "ingestion.embedding": {
                        "method": "mock",
                        "params": {"source_field": ""},
                    }
                }
            )
        )