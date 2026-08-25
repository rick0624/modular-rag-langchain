"""自訂 routing 範例:規則式查詢分類(支線;不影響檢索)。

契約:route(query: str) -> dict
(吃原始查詢;結果附加於 query() 輸出的 routing 鍵)

掛法::

    routing:
      method: custom
      params:
        file: ./examples/custom_modules/my_router.py
"""


def build(params, ctx):
    def route(query: str) -> dict:
        # TODO(替換點):換成你的分類模型 / 規則引擎。
        category = "問題" if ("?" in query or "?" in query) else "陳述"
        return {"category": category, "length": len(query)}

    return route
