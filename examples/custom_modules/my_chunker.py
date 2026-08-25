"""自訂 chunking 範例:按中文句號切片。

契約:chunk(documents: list[Document]) -> list[Document]
(metadata 逐塊複製,doc_id 必須保留;seq / chunk_id 由框架事後蓋章)

掛法::

    chunking:
      method: custom
      params:
        file: ./examples/custom_modules/my_chunker.py
"""

from langchain_core.documents import Document


def build(params, ctx):
    def chunk(documents: list[Document]) -> list[Document]:
        chunks = []
        for document in documents:
            # TODO(替換點):換成你的切塊規則。
            for sentence in document.page_content.split("。"):
                if sentence.strip():
                    chunks.append(
                        Document(
                            page_content=sentence.strip() + "。",
                            metadata=dict(document.metadata),  # 逐塊複製
                        )
                    )
        return chunks

    return chunk
