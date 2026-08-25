"""自訂 retrieval 範例:查本地索引(ctx.store);要接外部檢索 API 就改打 API。

契約:retrieve(query: str) -> list[Document]
(metadata["score"] 越大越相關,結果降冪)

掛法::

    retrieval:
      method: custom
      params:
        file: ./examples/custom_modules/my_retriever.py
        top_k: 5
"""

from langchain_core.documents import Document


def build(params, ctx):
    top_k = params.get("top_k", 5)
    store = ctx.store  # indexing 槽位建好的 vector store

    def retrieve(query: str) -> list[Document]:
        # TODO(替換點):要接外部檢索 API 時,把這段換成 HTTP 呼叫,再把
        # 回應映射進 Document(內文 → page_content、分數 → metadata["score"])。
        documents = []
        for document, score in store.similarity_search_with_score(query, k=top_k):
            document.metadata["score"] = float(score)
            documents.append(document)
        return documents

    return retrieve
