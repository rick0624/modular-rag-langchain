"""自訂 evaluation 範例:計算 recall(找回的相關文件數 / 標註的相關文件數)。

契約:evaluate(cases: list[EvalCase], results: list[dict]) -> dict
(results 是各題 query() 的完整輸出)

掛法::

    evaluation:
      method: custom
      params:
        file: ./examples/custom_modules/my_evaluator.py
"""

from rag.interfaces import EvalCase


def build(params, ctx):
    def evaluate(cases: list[EvalCase], results: list[dict]) -> dict:
        # TODO(替換點):換成你要的指標。
        per_case = []
        for case, result in zip(cases, results):
            retrieved = {document.metadata["doc_id"] for document in result["documents"]}
            found = len(retrieved & set(case.relevant_doc_ids))
            per_case.append({"query": case.query, "recall": found / len(case.relevant_doc_ids)})
        recall = sum(row["recall"] for row in per_case) / len(per_case)
        return {"metrics": {"recall": recall, "num_cases": len(per_case)}, "per_case": per_case}

    return evaluate
