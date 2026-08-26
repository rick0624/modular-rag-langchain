"""LLM 類方法(llm_multi_hyde / preqrag / insertrank),以 mock LLM 全離線測。"""

from __future__ import annotations

import pytest

from rag import build_runtime, ingest, query
from rag.errors import ConfigError
from rag.llm import parse_lines


def test_parse_lines_strips_listing_and_dedups():
    text = "1. 子問題一\n- 子問題二\n\n• 子問題三\n子問題一\n"
    assert parse_lines(text) == ["子問題一", "子問題二", "子問題三"]
    assert parse_lines(text, cap=2) == ["子問題一", "子問題二"]


def _transform(make_config, method: str, params: dict):
    runtime = build_runtime(
        make_config(
            **{"inference.query_transformation": {"method": method, "params": params}}
        )
    )
    return runtime.transform


def test_multi_hyde_expands_with_original(make_config):
    transform = _transform(
        make_config,
        "llm_multi_hyde",
        {
            "num_documents": 2,
            "llm": {"provider": "mock", "replies": ["假設一\n假設二\n假設三"]},
        },
    )
    assert transform(["原問題"]) == ["原問題", "假設一", "假設二"]  # cap=2


def test_multi_hyde_without_original_falls_back_when_empty(make_config):
    transform = _transform(
        make_config,
        "llm_multi_hyde",
        {"keep_original": False, "llm": {"provider": "mock", "replies": ["   \n  "]}},
    )
    assert transform(["原問題"]) == ["原問題"]  # LLM 沒產出 → 回退原查詢


def test_preqrag_multi_branch_decomposes(make_config):
    transform = _transform(
        make_config,
        "preqrag",
        {
            "max_subqueries": 2,
            "llm": {"provider": "mock", "replies": ["multi", "子一\n子二\n子三"]},
        },
    )
    assert transform(["複合問題"]) == ["複合問題", "子一", "子二"]


def test_preqrag_single_branch_rewrites_without_original(make_config):
    transform = _transform(
        make_config,
        "preqrag",
        {
            "include_original": False,
            "num_rewrites": 2,
            "llm": {"provider": "mock", "replies": ["single", "改寫一\n改寫二"]},
        },
    )
    assert transform(["單一問題"]) == ["改寫一", "改寫二"]


def test_llm_block_required(make_config):
    with pytest.raises(ConfigError, match="llm"):
        build_runtime(
            make_config(
                **{"inference.query_transformation": {"method": "preqrag"}}
            )
        )


def test_insertrank_reorders_and_appends_missing(make_config):
    runtime = build_runtime(
        make_config(
            **{
                "inference.reranking": {
                    "method": "insertrank",
                    "params": {
                        "top_k": 3,
                        "llm": {"provider": "mock", "replies": ["3, 1"]},
                    },
                }
            }
        )
    )
    ingest(runtime)
    documents = runtime.retrieve("VPN 伺服器位址?")
    assert len(documents) >= 3
    reranked = runtime.rerank("VPN 伺服器位址?", documents)
    # LLM 給 3,1 → 第 3、第 1,缺漏的候選依原順序補在後面,取 top_k=3
    assert [d.metadata["chunk_id"] for d in reranked] == [
        documents[2].metadata["chunk_id"],
        documents[0].metadata["chunk_id"],
        documents[1].metadata["chunk_id"],
    ]


def test_insertrank_fail_soft_on_unparseable_reply(make_config):
    runtime = build_runtime(
        make_config(
            **{
                "inference.reranking": {
                    "method": "insertrank",
                    "params": {
                        "top_k": 2,
                        "llm": {"provider": "mock", "replies": ["看起來都很相關喔"]},
                    },
                }
            }
        )
    )
    ingest(runtime)
    documents = runtime.retrieve("VPN 伺服器位址?")
    reranked = runtime.rerank("VPN 位址?", documents)
    # 回覆不含合法編號 → 保留原順序前 top_k
    assert [d.metadata["chunk_id"] for d in reranked] == [
        d.metadata["chunk_id"] for d in documents[:2]
    ]


def test_preqrag_end_to_end_multi_subquery_fusion(make_config):
    """preqrag(multi)→ 多子查詢逐路檢索 → fusion 融合,全流程串通。"""
    runtime = build_runtime(
        make_config(
            **{
                "inference.query_transformation": {
                    "method": "preqrag",
                    "params": {
                        "llm": {
                            "provider": "mock",
                            "replies": ["multi", "VPN 伺服器位址\n病假 診斷證明"],
                        }
                    },
                }
            }
        )
    )
    ingest(runtime)
    result = query(runtime, "VPN 跟請假規定?")
    assert len(result["subqueries"]) == 3  # 原查詢 + 兩條子查詢
    doc_ids = {doc.metadata["doc_id"] for doc in result["documents"]}
    assert {"vpn.txt", "leave.md"} <= doc_ids
