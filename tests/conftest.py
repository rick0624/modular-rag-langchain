"""測試共用 fixture。鐵則:所有測試離線可跑,不碰網路、不下載模型。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from rag.config import parse_config


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    """三份小語料(含子資料夾與非收錄副檔名)。"""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "vpn.txt").write_text(
        "公司 VPN 伺服器位址為 vpn.example.com,連接埠 443。"
        "驗證使用公司帳號加動態密碼。連不上時請改用 TCP 模式。",
        encoding="utf-8",
    )
    (raw / "leave.md").write_text(
        "# 請假規定\n\n特休依年資計算:滿一年七天、滿兩年十天。\n\n"
        "病假連續三天以上需檢附診斷證明。請假單請於休假前一個工作天送出。",
        encoding="utf-8",
    )
    sub = raw / "sub"
    sub.mkdir()
    (sub / "expense.txt").write_text(
        "報帳款項於每月五日與二十日兩批撥付,單據需於三十天內送出。",
        encoding="utf-8",
    )
    (raw / "ignore.bin").write_bytes(b"\x00\x01")
    return raw


BASE_CONFIG: dict[str, Any] = {
    "ingestion": {
        "import": {"method": "local_files", "params": {"input_dir": "PLACEHOLDER"}},
        "parsing": {"method": "text"},
        "chunking": {"method": "recursive", "params": {"chunk_size": 60, "chunk_overlap": 10}},
        "embedding": {"method": "mock", "params": {"dim": 128}},
        "indexing": {"method": "in_memory"},
    },
    "inference": {
        "query_transformation": {"method": "passthrough"},
        "retrieval": {"method": "vector", "params": {"top_k": 5}},
        "reranking": {"method": "none"},
        "generation": {"method": "mock"},
    },
}


@pytest.fixture()
def make_config(corpus_dir: Path):
    """回傳以 BASE_CONFIG 為底、套用覆寫後的 RAGConfig。

    覆寫以「點路徑 → 值」表示,例如::

        make_config(**{"inference.reranking": {"method": "custom", "params": {...}}})
    """

    def _make(**overrides: Any):
        data = copy.deepcopy(BASE_CONFIG)
        data["ingestion"]["import"]["params"]["input_dir"] = str(corpus_dir)
        for dotted, value in overrides.items():
            node = data
            *parents, leaf = dotted.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        return parse_config(data, source="<test>")

    return _make
