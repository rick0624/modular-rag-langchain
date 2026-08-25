"""inference 流程:檢索排序、prompt 可稽核、多子查詢融合、支線槽位。"""

from __future__ import annotations

from rag import build_runtime, ingest, query


def _ready_runtime(config):
    runtime = build_runtime(config)
    ingest(runtime)
    return runtime


def test_query_end_to_end(make_config):
    runtime = _ready_runtime(make_config())
    result = query(runtime, "VPN 伺服器位址與連接埠是多少?")

    assert result["answer"]
    assert result["subqueries"] == ["VPN 伺服器位址與連接埠是多少?"]
    documents = result["documents"]
    assert documents and documents[0].metadata["doc_id"] == "vpn.txt"
    scores = [doc.metadata["score"] for doc in documents]
    assert scores == sorted(scores, reverse=True)
    # prompt 可稽核:切片帶 [chunk_id] 前綴
    assert f"[{documents[0].metadata['chunk_id']}]" in result["prompt"]
    assert "VPN 伺服器位址與連接埠是多少?" in result["prompt"]
    assert result["routing"] is None
    assert result["output"] is None


def test_multi_subquery_goes_through_fusion(make_config, tmp_path):
    """自訂 transform 拆成多子查詢 → 逐子查詢檢索 → merge 融合去重。"""
    transform_file = tmp_path / "split_transform.py"
    transform_file.write_text(
        "def build(params, ctx):\n"
        "    def transform(queries):\n"
        "        return ['VPN 伺服器位址', '請假 診斷證明']\n"
        "    return transform\n",
        encoding="utf-8",
    )
    runtime = _ready_runtime(
        make_config(
            **{
                "inference.query_transformation": {
                    "method": "custom",
                    "params": {"file": str(transform_file)},
                },
                "inference.fusion": {"method": "merge", "params": {"top_k": 4}},
            }
        )
    )
    result = query(runtime, "VPN 跟請假規定?")

    assert len(result["subqueries"]) == 2
    doc_ids = {doc.metadata["doc_id"] for doc in result["documents"]}
    assert {"vpn.txt", "leave.md"} <= doc_ids  # 兩路子查詢的結果都進了融合
    chunk_ids = [doc.metadata["chunk_id"] for doc in result["documents"]]
    assert len(chunk_ids) == len(set(chunk_ids))  # 融合有去重
    assert len(result["documents"]) <= 4
    retrieval_steps = [s for s in result["trace"] if s["slot"] == "retrieval"]
    assert len(retrieval_steps) == 2  # 逐子查詢各檢索一次


def test_routing_and_formatter_side_branches(make_config):
    runtime = _ready_runtime(
        make_config(
            **{
                "inference.routing": {
                    "method": "keyword_match",
                    "params": {"routes": {"資訊": ["VPN"], "人資": ["請假"]}},
                },
                "inference.formatter": {
                    "method": "simple_json",
                    "params": {"include_content": False},
                },
            }
        )
    )
    result = query(runtime, "VPN 連不上怎麼辦?")

    assert result["routing"] == {"category": "資訊", "matched_keywords": ["VPN"]}
    payload = result["output"]
    assert payload["query"] == "VPN 連不上怎麼辦?"
    assert payload["answer"] == result["answer"]
    assert payload["references"]
    assert "content" not in payload["references"][0]


def test_reranking_chain_runs_in_order(make_config, tmp_path):
    """方法鏈 [none, custom]:custom 在後,top_k 生效。"""
    rerank_file = tmp_path / "head_reranker.py"
    rerank_file.write_text(
        "def build(params, ctx):\n"
        "    def rerank(query, documents):\n"
        "        return documents[: params.get('keep', 1)]\n"
        "    return rerank\n",
        encoding="utf-8",
    )
    runtime = _ready_runtime(
        make_config(
            **{
                "inference.reranking": {
                    "method": ["none", "custom"],
                    "method_params": {
                        "none": {},
                        "custom": {"file": str(rerank_file), "keep": 2},
                    },
                }
            }
        )
    )
    result = query(runtime, "VPN 伺服器位址?")
    # fusion 預設 top_k=5,但 rerank 鏈已把每路候選截到 2
    assert len(result["documents"]) <= 2


def test_mock_generation_cycles_replies(make_config):
    runtime = _ready_runtime(
        make_config(
            **{
                "inference.generation": {
                    "method": "mock",
                    "params": {"replies": ["答案一", "答案二"]},
                }
            }
        )
    )
    assert query(runtime, "問題 1")["answer"] == "答案一"
    assert query(runtime, "問題 2")["answer"] == "答案二"
    assert query(runtime, "問題 3")["answer"] == "答案一"


def test_prompt_template_override(make_config):
    runtime = _ready_runtime(
        make_config(
            **{
                "inference.prompt": {
                    "template": "內容:{context}\n\n請回答:{query}",
                    "system": "只根據內容回答。",
                }
            }
        )
    )
    result = query(runtime, "VPN 位址?")
    assert result["prompt"].startswith("內容:[")
    assert result["prompt"].endswith("請回答:VPN 位址?")
