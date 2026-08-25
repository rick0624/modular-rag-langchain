"""自訂 query_transformation 範例:同義詞擴充(1 → N 條子查詢)。

契約:transform(queries: list[str]) -> list[str]
(輸出多筆時,框架會逐子查詢檢索,再交給 fusion 合併)

掛法::

    query_transformation:
      method: custom
      params:
        file: ./examples/custom_modules/my_query_transform.py
        synonyms: {"VPN": "虛擬私人網路"}
"""


def build(params, ctx):
    synonyms = params.get("synonyms", {})

    def transform(queries: list[str]) -> list[str]:
        # TODO(替換點):換成你的改寫 / 拆解邏輯(呼叫 LLM 也可以)。
        expanded = list(queries)
        for query in queries:
            for word, alias in synonyms.items():
                if word in query:
                    expanded.append(query.replace(word, alias))
        return expanded

    return transform
