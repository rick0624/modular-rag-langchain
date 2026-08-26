"""不起 service 的試跑腳本:直接用 Python API 跑一輪 ingest → query → evaluate。

用法(請在 repo 根目錄執行,config 內的相對路徑以工作目錄解析)::

    python try_rag.py
    python try_rag.py --config configs/default.yaml --query "特休假滿兩年有幾天?"
"""

from __future__ import annotations

import argparse
import json
import logging

from rag import build_runtime, evaluate, ingest, load_config, query


def main() -> None:
    parser = argparse.ArgumentParser(description="modular-rag-langchain 試跑")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 設定檔路徑")
    parser.add_argument("--query", default="VPN 伺服器位址與連接埠是多少?", help="查詢文字")
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning"],
        help="log 等級(預設 warning 保持輸出乾淨;debug 印出每步細節)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    runtime = build_runtime(load_config(args.config))  # 建構期即驗證 config

    print("=== ingest ===")
    result = ingest(runtime)
    print(f"寫入 {result['documents_written']} 個切片,來源:{result['files']}")

    print("\n=== query ===")
    answer = query(runtime, args.query)
    print(f"查詢:{args.query}")
    print(f"答案:{answer['answer']}")
    print("\n檢索結果(降冪):")
    for doc in answer["documents"]:
        print(f"  [{doc.metadata['chunk_id']}] score={doc.metadata['score']:.4f}")
    print("\ntrace(每步的方法與耗時):")
    for step in answer["trace"]:
        print(f"  {step}")
    print("\nprompt(實際送 LLM 的內容):")
    print(answer["prompt"])

    if runtime.evaluator is not None:
        print("\n=== evaluate ===")
        metrics = evaluate(runtime)["metrics"]
        print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
