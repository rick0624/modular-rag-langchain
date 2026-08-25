"""自訂 parsing 範例:把來源轉成純文字 Document(示範簡單清理)。

契約:parse(sources: list[Source]) -> list[Document]
(metadata 必帶 doc_id 與 page;PDF / HTML / OCR 都從這裡接)

掛法::

    parsing:
      method: custom
      params:
        file: ./examples/custom_modules/my_parser.py
"""

from langchain_core.documents import Document

from rag.interfaces import Source


def build(params, ctx):
    def parse(sources: list[Source]) -> list[Document]:
        documents = []
        for source in sources:
            # TODO(替換點):換成你的解析邏輯(PDF、HTML、OCR…)。
            if source.text is not None:
                text = source.text
            else:
                text = source.path.read_text(encoding="utf-8")
            text = " ".join(text.split())  # 示範:壓掉多餘空白與換行
            documents.append(
                Document(
                    page_content=text,
                    metadata={"doc_id": source.doc_id, "page": 1, **source.meta},
                )
            )
        return documents

    return parse
