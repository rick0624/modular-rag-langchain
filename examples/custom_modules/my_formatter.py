"""自訂 formatter 範例:把結果組成你們系統的信封格式(終端支線)。

契約:format_payload(query: str, documents: list[Document], answer: str | None) -> dict
(結果進 query() 輸出的 output 鍵;走 HTTP 須可 JSON 序列化)

掛法::

    formatter:
      method: custom
      params:
        file: ./examples/custom_modules/my_formatter.py
"""

from langchain_core.documents import Document


def build(params, ctx):
    def format_payload(query: str, documents: list[Document], answer: str | None) -> dict:
        # TODO(替換點):換成你們對外系統要的欄位。
        return {
            "code": 200,
            "data": {
                "question": query,
                "answer": answer,
                "sources": [document.metadata["chunk_id"] for document in documents],
            },
        }

    return format_payload
