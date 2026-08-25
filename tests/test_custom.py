"""custom module 載入:file/function 慣例、簽名驗證的 zh-TW 錯誤、範例可跑。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag import build_runtime, ingest, query
from rag.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples" / "custom_modules"


def test_custom_requires_file(make_config):
    with pytest.raises(ConfigError, match="file"):
        build_runtime(make_config(**{"inference.reranking": {"method": "custom"}}))


def test_custom_missing_file(make_config):
    with pytest.raises(ConfigError, match="找不到檔案"):
        build_runtime(
            make_config(
                **{"inference.reranking": {"method": "custom", "params": {"file": "./nope.py"}}}
            )
        )


def test_custom_missing_function_lists_available(make_config, tmp_path):
    module = tmp_path / "mod.py"
    module.write_text("def other(params, ctx):\n    pass\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="other"):
        build_runtime(
            make_config(
                **{"inference.reranking": {"method": "custom", "params": {"file": str(module)}}}
            )
        )


def test_custom_wrong_signature_names_expected_params(make_config, tmp_path):
    module = tmp_path / "bad_sig.py"
    module.write_text(
        "def build(params, ctx):\n"
        "    def rerank(docs):\n"
        "        return docs\n"
        "    return rerank\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"\(query, documents\)"):
        build_runtime(
            make_config(
                **{"inference.reranking": {"method": "custom", "params": {"file": str(module)}}}
            )
        )


def test_custom_builder_must_take_params_and_ctx(make_config, tmp_path):
    module = tmp_path / "one_arg.py"
    module.write_text("def build(params):\n    return None\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\(params, ctx\)"):
        build_runtime(
            make_config(
                **{"inference.reranking": {"method": "custom", "params": {"file": str(module)}}}
            )
        )


def test_custom_params_passthrough_and_function_name(make_config, tmp_path):
    module = tmp_path / "named.py"
    module.write_text(
        "def make(params, ctx):\n"
        "    assert params == {'keep': 1}\n"
        "    assert ctx.store is not None\n"
        "    def rerank(query, documents):\n"
        "        return documents[: params['keep']]\n"
        "    return rerank\n",
        encoding="utf-8",
    )
    runtime = build_runtime(
        make_config(
            **{
                "inference.reranking": {
                    "method": "custom",
                    "params": {"file": str(module), "function": "make", "keep": 1},
                }
            }
        )
    )
    ingest(runtime)
    assert len(query(runtime, "VPN 位址?")["documents"]) == 1


def test_example_reranker_runs(make_config, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    runtime = build_runtime(
        make_config(
            **{
                "inference.reranking": {
                    "method": "custom",
                    "params": {
                        "file": str(EXAMPLES / "my_reranker.py"),
                        "top_k": 2,
                    },
                }
            }
        )
    )
    ingest(runtime)
    result = query(runtime, "VPN 伺服器位址是多少?")
    assert len(result["documents"]) <= 2
    assert result["documents"][0].metadata["doc_id"] == "vpn.txt"


def test_example_importer_runs(make_config):
    runtime = build_runtime(
        make_config(
            **{
                "ingestion.import": {
                    "method": "custom",
                    "params": {"file": str(EXAMPLES / "my_importer.py")},
                }
            }
        )
    )
    result = ingest(runtime)
    assert result["files"] == ["notice-2026-001", "notice-2026-002"]
    # Source.meta 複製進切片 metadata
    chunk = next(iter(runtime.store.store.values()))
    assert chunk["metadata"]["department"] in {"總務部", "人資部"}
