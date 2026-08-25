"""FastAPI service 端到端(TestClient,全離線,in_memory 索引)。"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from rag.service import create_app
from tests.conftest import BASE_CONFIG


@pytest.fixture()
def config_file(corpus_dir: Path, tmp_path: Path) -> Path:
    data = copy.deepcopy(BASE_CONFIG)
    data["ingestion"]["import"]["params"]["input_dir"] = str(corpus_dir)
    data["evaluation"] = {
        "method": "retrieval_metrics",
        "params": {
            "cases": [{"query": "VPN 伺服器位址?", "relevant_doc_ids": ["vpn.txt"]}]
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture()
def client(config_file: Path) -> TestClient:
    # in_memory 索引 → RAG_INGEST_ON_STARTUP 預設 auto 會在啟動時 ingest
    return TestClient(create_app(str(config_file)))


def test_health(client: TestClient, config_file: Path):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "config_path": str(config_file),
        "indexing_method": "in_memory",
    }


def test_query_after_startup_ingest(client: TestClient):
    response = client.post("/query", json={"query": "VPN 伺服器位址與連接埠?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    documents = body["documents"]
    assert documents[0]["doc_id"] == "vpn.txt"
    assert set(documents[0]) == {"chunk_id", "doc_id", "page", "score", "content"}
    scores = [doc["score"] for doc in documents]
    assert scores == sorted(scores, reverse=True)
    assert body["trace"]


def test_ingest_idempotent(client: TestClient):
    first = client.post("/ingest").json()
    second = client.post("/ingest").json()
    assert first["documents_written"] == second["documents_written"]
    assert first["files"] == second["files"]


def test_evaluate_config_cases_and_inline(client: TestClient):
    from_config = client.post("/evaluate", json={}).json()
    assert from_config["metrics"]["hit_rate"] == 1.0

    inline = client.post(
        "/evaluate",
        json={"cases": [{"query": "病假診斷證明?", "relevant_doc_ids": ["leave.md"]}]},
    ).json()
    assert inline["metrics"]["num_cases"] == 1
    assert inline["metrics"]["hit_rate"] == 1.0


def test_rag_error_becomes_400(client: TestClient):
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 400
    assert "查詢內容不可為空" in response.json()["detail"]


def test_bad_config_fails_fast(tmp_path: Path, corpus_dir: Path):
    data = copy.deepcopy(BASE_CONFIG)
    data["ingestion"]["import"]["params"]["input_dir"] = str(corpus_dir)
    data["inference"]["retrieval"] = {"method": "nope"}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(Exception, match="nope"):
        create_app(str(path))
