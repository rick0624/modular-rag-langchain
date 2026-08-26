"""experiment.py:組合生成純函式 + run_experiments(全離線)。"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

import rag.core
from experiment import make_variants, option_name, run_experiments
from rag import build_runtime, ingest, query
from tests.conftest import BASE_CONFIG


# ─── 組合生成 ───


_BASE = {"ingestion": {"chunking": {"method": "recursive"}},
         "inference": {"reranking": {"method": "none"}}}


def test_one_at_a_time_baseline_plus_each_option():
    variants = make_variants(
        copy.deepcopy(_BASE), "one_at_a_time",
        {"inference.reranking": ["none", "api"], "ingestion.chunking": ["recursive"]},
    )
    labels = [label for label, _, _ in variants]
    assert labels == ["baseline", "reranking=none", "reranking=api", "chunking=recursive"]


def test_product_crosses_dimensions():
    variants = make_variants(
        copy.deepcopy(_BASE), "product",
        {"inference.reranking": ["none", "api"], "ingestion.chunking": ["recursive"]},
    )
    assert len(variants) == 2  # 2 × 1
    assert variants[0][0] == "reranking=none + chunking=recursive"


def test_dict_option_replaces_whole_slot():
    variants = make_variants(
        copy.deepcopy(_BASE), "one_at_a_time",
        {"inference.reranking": [{"method": "custom", "params": {"file": "./x.py"}}]},
    )
    _, overrides, cfg = variants[1]
    assert cfg["inference"]["reranking"] == {"method": "custom", "params": {"file": "./x.py"}}
    assert overrides == {"inference.reranking": {"method": "custom", "params": {"file": "./x.py"}}}


def test_bundle_expands_multiple_slots_with_label():
    bundle = {
        "_label": "stack-a",
        "ingestion.chunking": {"method": "recursive", "params": {"chunk_size": 100}},
        "inference.reranking": "api",
    }
    variants = make_variants(copy.deepcopy(_BASE), "one_at_a_time", {"stack": [bundle]})
    label, overrides, cfg = variants[1]
    assert label == "stack=stack-a"
    assert cfg["ingestion"]["chunking"]["params"]["chunk_size"] == 100
    assert cfg["inference"]["reranking"]["method"] == "api"
    assert "_label" not in overrides


def test_overlapping_dimensions_rejected():
    with pytest.raises(ValueError, match="都覆蓋槽位"):
        make_variants(
            copy.deepcopy(_BASE), "one_at_a_time",
            {
                "inference.reranking": ["none"],
                "stack": [{"inference.reranking": "api"}],
            },
        )


def test_option_name_shapes():
    assert option_name("api") == "api"
    assert option_name(["none", "custom"]) == "none+custom"
    assert option_name({"method": "custom", "params": {}}) == "custom"
    # 純量參數進 label,同方法不同參數的變體分得出來
    assert option_name({"method": "vector", "params": {"top_k": 3}}) == "vector(top_k=3)"


# ─── run_experiments(離線端到端)───


@pytest.fixture()
def base_config_file(corpus_dir: Path, tmp_path: Path) -> str:
    data = copy.deepcopy(BASE_CONFIG)
    data["ingestion"]["import"]["params"]["input_dir"] = str(corpus_dir)
    data["evaluation"] = {
        "method": "retrieval_metrics",
        "params": {"cases": [
            {"query": "VPN 伺服器位址?", "relevant_doc_ids": ["vpn.txt"]},
        ]},
    }
    path = tmp_path / "exp_base.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_run_experiments_sweeps_and_shares_store(base_config_file, monkeypatch):
    ingest_calls = {"count": 0}
    real_ingest = rag.core.ingest

    def counting_ingest(runtime):
        ingest_calls["count"] += 1
        return real_ingest(runtime)

    # experiment.py 是 from rag import ingest → patch experiment 模組內的名字
    import experiment as experiment_module

    monkeypatch.setattr(experiment_module, "ingest", counting_ingest)

    records = run_experiments(
        base_config=base_config_file,
        mode="one_at_a_time",
        slot_options={
            "inference.reranking": [
                "none",
                {"method": "insertrank",
                 "params": {"llm": {"provider": "mock", "replies": ["2,1"]}}},
            ],
        },
        queries=["VPN 伺服器位址?"],
    )

    assert [record["error"] for record in records] == [None, None, None]
    assert len(records) == 3  # baseline + none + insertrank
    # ingestion 全部相同 → 只 ingest 一次,索引共用
    assert ingest_calls["count"] == 1
    for record in records:
        assert record["results"][0][1]["documents"]
        assert record["metrics"]["hit_rate"] == 1.0


def test_run_experiments_bad_variant_does_not_break_batch(base_config_file):
    records = run_experiments(
        base_config=base_config_file,
        mode="one_at_a_time",
        slot_options={"inference.reranking": ["no_such_method"]},
        queries=["VPN 位址?"],
    )
    baseline, bad = records
    assert baseline["error"] is None
    assert "no_such_method" in bad["error"]


def test_build_runtime_with_injected_store(make_config):
    first = build_runtime(make_config())
    ingest(first)
    # 注入既有 store:不重建索引、不重 ingest,檢索直接可用
    second = build_runtime(make_config(**{"inference.reranking": {"method": "none"}}),
                           store=first.store)
    assert second.store is first.store
    result = query(second, "VPN 伺服器位址?")
    assert result["documents"]
