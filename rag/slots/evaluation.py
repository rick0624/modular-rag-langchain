"""槽位:Evaluation —— 檢索指標(hit rate / MRR)。

契約:(cases, results) → metrics dict;``results`` 是各題 ``query()``
的完整輸出。資料集格式(JSONL,每行一筆)::

    {"query": "...", "relevant_doc_ids": ["doc_id", ...], "reference_answer": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from rag.config import MethodConfig
from rag.errors import ComponentError, ConfigError
from rag.interfaces import EvalCase, EvaluateFn
from rag.registry import BaseParams, BuildContext, register, validate_params


def load_cases(path: str | Path) -> list[EvalCase]:
    """載入 JSONL 測試集;格式錯誤時指出檔案與 1-based 行號。

    Raises:
        ComponentError: 檔案不存在、任一行不是合法 JSON / 欄位不符,
            或資料集為空。
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise ComponentError(f"找不到評估資料集:{dataset_path}")
    cases: list[EvalCase] = []
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            cases.append(EvalCase.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ComponentError(
                f"評估資料集 '{dataset_path}' 第 {line_number} 行格式不正確:{exc}"
            ) from exc
    if not cases:
        raise ComponentError(f"評估資料集 '{dataset_path}' 沒有任何測試案例")
    return cases


def load_default_cases(cfg: MethodConfig) -> list[EvalCase]:
    """依 evaluation 配置載入預設測試案例(行內 cases 優先於 dataset_path)。"""
    p = validate_params(
        "evaluation", cfg.methods()[0], _RetrievalMetricsParams,
        cfg.params_for(),
    )
    if p.cases:
        return [EvalCase.model_validate(case) for case in p.cases]
    if p.dataset_path is None:
        raise ConfigError(
            "evaluation 方法 'retrieval_metrics' 需要 dataset_path(JSONL 檔)"
            "或行內 cases 其中之一"
        )
    return load_cases(p.dataset_path)


def _retrieved_doc_ids(result: dict[str, Any]) -> list[str]:
    """依名次序取出去重後的 doc_id(同文件多切片只算一次)。"""
    seen: list[str] = []
    for document in result["documents"]:
        doc_id = document.metadata.get("doc_id") or document.id
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


class _RetrievalMetricsParams(BaseParams):
    dataset_path: str | None = Field(default=None, description="JSONL 測試集路徑")
    cases: list[dict[str, Any]] | None = Field(
        default=None, description="行內測試案例(優先於 dataset_path)"
    )


@register("evaluation", "retrieval_metrics")
def build_retrieval_metrics(params: dict[str, Any], ctx: BuildContext) -> EvaluateFn:
    """基本檢索指標:hit rate 與 MRR(以 doc_id 計)。

    - ``hit``:融合後結果中任一相關文件出現即為命中。
    - ``reciprocal_rank``:第一個相關文件名次的倒數(未出現為 0)。
    """
    validate_params("evaluation", "retrieval_metrics", _RetrievalMetricsParams, params)

    def evaluate(cases: list[EvalCase], results: list[dict]) -> dict:
        per_case: list[dict[str, Any]] = []
        for case, result in zip(cases, results):
            retrieved = _retrieved_doc_ids(result)
            relevant = set(case.relevant_doc_ids)
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break
            per_case.append(
                {
                    "query": case.query,
                    "relevant_doc_ids": case.relevant_doc_ids,
                    "retrieved_doc_ids": retrieved,
                    "hit": reciprocal_rank > 0,
                    "reciprocal_rank": reciprocal_rank,
                    "answer": result.get("answer"),
                }
            )
        if not per_case:
            raise ComponentError("evaluation 沒有任何測試案例可評")
        num_cases = len(per_case)
        return {
            "metrics": {
                "hit_rate": sum(1 for row in per_case if row["hit"]) / num_cases,
                "mrr": sum(row["reciprocal_rank"] for row in per_case) / num_cases,
                "num_cases": num_cases,
            },
            "per_case": per_case,
        }

    return evaluate
