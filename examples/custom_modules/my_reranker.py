"""自訂 reranking 範例:查詢與切片的字元重疊率重排(離線可跑)。

契約:rerank(query: str, documents: list[Document]) -> list[Document]
(只能重排 / 過濾 / 改 metadata["score"],不得改切片內容)

掛法::

    reranking:
      method: custom
      params:
        file: ./examples/custom_modules/my_reranker.py
        top_k: 3          # file / function 以外的鍵透傳為 params
"""

from langchain_core.documents import Document


def build(params, ctx):
    top_k = params.get("top_k", 5)

    def rerank(query: str, documents: list[Document]) -> list[Document]:
        # TODO(替換點):換成你的重排邏輯(呼叫內部服務、規則引擎…)。
        def overlap(document: Document) -> float:
            a, b = set(query), set(document.page_content)
            return len(a & b) / len(a | b) if b else 0.0

        reranked = sorted(documents, key=overlap, reverse=True)
        for document in reranked:
            document.metadata["score"] = overlap(document)
        return reranked[:top_k]

    return rerank
