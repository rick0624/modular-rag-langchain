"""evaluation:hit_rate / MRR 數值正確性、資料集載入錯誤。"""

from __future__ import annotations

import pytest

from rag import build_runtime, evaluate, ingest
from rag.errors import ComponentError, ConfigError
from rag.interfaces import EvalCase
from rag.slots.evaluation import load_cases


def test_metrics_values(make_config):
    runtime = build_runtime(
        make_config(
            evaluation={
                "method": "retrieval_metrics",
                "params": {
                    "cases": [
                        {"query": "VPN 伺服器位址?", "relevant_doc_ids": ["vpn.txt"]},
                        {"query": "病假診斷證明?", "relevant_doc_ids": ["leave.md"]},
                        {"query": "完全無關的量子力學問題", "relevant_doc_ids": ["no-such-doc"]},
                    ]
                },
            }
        )
    )
    ingest(runtime)
    result = evaluate(runtime)

    metrics = result["metrics"]
    assert metrics["num_cases"] == 3
    # 前兩題命中(詞袋向量下相關文件排第一),第三題必不命中
    assert metrics["hit_rate"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx(2 / 3)
    assert [row["hit"] for row in result["per_case"]] == [True, True, False]
    # 同文件多切片只算一次(doc_id 去重)
    first = result["per_case"][0]["retrieved_doc_ids"]
    assert len(first) == len(set(first))


def test_inline_cases_without_config_block(make_config):
    runtime = build_runtime(make_config())
    ingest(runtime)
    result = evaluate(
        runtime, [EvalCase(query="VPN 位址?", relevant_doc_ids=["vpn.txt"])]
    )
    assert result["metrics"]["hit_rate"] == 1.0


def test_no_cases_anywhere_raises(make_config):
    runtime = build_runtime(make_config())
    with pytest.raises(ConfigError, match="evaluation"):
        evaluate(runtime)


def test_load_cases_reports_line_number(tmp_path):
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        '{"query": "q", "relevant_doc_ids": ["a"]}\n{broken json\n', encoding="utf-8"
    )
    with pytest.raises(ComponentError, match="第 2 行"):
        load_cases(dataset)


def test_load_cases_missing_file(tmp_path):
    with pytest.raises(ComponentError, match="找不到"):
        load_cases(tmp_path / "nope.jsonl")
