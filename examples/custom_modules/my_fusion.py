"""自訂 fusion 範例:名次分數法(RRF)合併多子查詢結果。

契約:fuse(results: list[list[Document]]) -> list[Document]
(results 是各子查詢的結果清單;單一子查詢時也會執行)

掛法::

    fusion:
      method: custom
      params:
        file: ./examples/custom_modules/my_fusion.py
        top_k: 5
"""

from langchain_core.documents import Document


def build(params, ctx):
    top_k = params.get("top_k", 5)

    def fuse(results: list[list[Document]]) -> list[Document]:
        # TODO(替換點):換成你的融合邏輯。這裡示範 RRF:
        # 在某一路結果中排第 rank 名,得 1/(60+rank) 分;同一切片跨路加總。
        scores = {}  # chunk_id → 累積分數
        docs = {}    # chunk_id → Document
        for documents in results:
            for rank, document in enumerate(documents):
                key = document.metadata["chunk_id"]
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
                docs[key] = document
        merged = sorted(docs.values(), key=lambda d: scores[d.metadata["chunk_id"]], reverse=True)
        for document in merged:
            document.metadata["score"] = scores[document.metadata["chunk_id"]]
        return merged[:top_k]

    return fuse
