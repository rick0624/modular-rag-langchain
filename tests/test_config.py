"""config 載入:env 展開、method_params 優先序、方法鏈限制、extra=forbid。"""

from __future__ import annotations

import pytest

from rag.config import MethodConfig, expand_env_vars, parse_config
from rag.errors import ConfigError


def test_params_for_method_params_wins():
    cfg = MethodConfig(
        method="a",
        params={"flat": 1},
        method_params={"a": {"scoped": 2}, "b": {"other": 3}},
    )
    assert cfg.params_for() == {"scoped": 2}
    assert cfg.params_for("b") == {"other": 3}


def test_params_for_falls_back_to_flat_params():
    cfg = MethodConfig(method="a", params={"flat": 1}, method_params={"b": {"x": 2}})
    assert cfg.params_for() == {"flat": 1}


def test_null_method_params_block_is_empty():
    cfg = MethodConfig(method="a", method_params={"a": None})
    assert cfg.params_for() == {}


def test_chain_with_flat_params_rejected():
    with pytest.raises(Exception, match="method_params"):
        MethodConfig(method=["a", "b"], params={"x": 1})


def test_expand_env_vars(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret")
    data = {"key": "Bearer ${MY_TOKEN}", "nested": ["${MY_TOKEN}"], "escaped": "$${LITERAL}"}
    expanded = expand_env_vars(data)
    assert expanded == {
        "key": "Bearer secret",
        "nested": ["secret"],
        "escaped": "${LITERAL}",
    }


def test_expand_env_vars_missing_raises(monkeypatch):
    monkeypatch.delenv("NOPE_NOT_SET", raising=False)
    with pytest.raises(ConfigError, match="NOPE_NOT_SET"):
        expand_env_vars({"key": "${NOPE_NOT_SET}"})


def test_extra_keys_rejected():
    with pytest.raises(ConfigError, match="驗證失敗"):
        parse_config(
            {
                "ingestion": {
                    "import": {"method": "x"},
                    "parsing": {"method": "x"},
                    "chunking": {"method": "x"},
                    "embedding": {"method": "x"},
                    "indexing": {"method": "x"},
                    "typo_slot": {"method": "x"},
                },
                "inference": {
                    "query_transformation": {"method": "x"},
                    "retrieval": {"method": "x"},
                    "reranking": {"method": "x"},
                    "generation": {"method": "x"},
                },
            }
        )


def test_missing_stage_rejected():
    with pytest.raises(ConfigError, match="驗證失敗"):
        parse_config({"ingestion": None, "inference": None})
