"""自訂 embedding 範例:接自家向量服務的骨架(離線示範用字元頻率向量)。

契約:回傳帶 embed_documents(texts) 與 embed_query(text) 兩個方法的物件
(同一物件同時服務建索引與查詢端 → 同向量空間)

掛法::

    embedding:
      method: custom
      params:
        file: ./examples/custom_modules/my_embeddings.py
        dim: 64
        # source_field / extra_vectors 也可以在這裡設(由框架消費,
        # 會一併透傳進 params,builder 忽略即可)
"""


def build(params, ctx):
    dim = params.get("dim", 64)

    def embed_one(text: str) -> list[float]:
        # TODO(替換點):換成呼叫你的向量服務(一次一筆或整批都行)。
        vector = [0.0] * dim
        for ch in text:
            vector[ord(ch) % dim] += 1.0
        return vector

    class MyEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [embed_one(text) for text in texts]  # 建索引端(整批)

        def embed_query(self, text: str) -> list[float]:
            return embed_one(text)                      # 查詢端(同一模型!)

    return MyEmbeddings()
