"""自訂 indexing 範例:自己管「寫入 + 查詢」的最小 store(離線可跑)。

契約:回傳帶 add_documents(documents, ids=...) 與
similarity_search_with_score(query, k=...) 兩個方法的物件;
向量請用 ctx.embeddings 算(同向量空間)。

掛法::

    indexing:
      method: custom
      params:
        file: ./examples/custom_modules/my_indexing.py
"""

from langchain_core.documents import Document


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def build(params, ctx):
    embeddings = ctx.embeddings  # embedding 槽位建好的物件

    class MyStore:
        def __init__(self):
            self.rows = {}  # id → (Document, 向量)

        def add_documents(self, documents: list[Document], ids=None):
            # TODO(替換點):換成寫入你的索引服務(自研 DB、其他向量庫…)。
            ids = ids or [document.id for document in documents]
            vectors = embeddings.embed_documents(
                [document.page_content for document in documents]
            )
            for id_, document, vector in zip(ids, documents, vectors):
                self.rows[id_] = (document, vector)  # 同 id 覆寫 = upsert
            return ids

        def similarity_search_with_score(self, query: str, k: int = 4):
            # TODO(替換點):換成查詢你的索引服務;回傳 [(Document, 分數), ...]。
            query_vector = embeddings.embed_query(query)
            scored = [
                (document, _cosine(query_vector, vector))
                for document, vector in self.rows.values()
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            return scored[:k]

    return MyStore()
