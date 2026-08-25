"""自訂 reranking 範例:查詢與切片的字元重疊率重排(離線可跑)。

掛法(config)::

    reranking:
      method: custom
      params:
        file: ./examples/custom_modules/my_reranker.py
        # function: build          # 預設 build
        top_k: 3                   # file / function 以外的鍵透傳為 params

慣例:builder 函式 ``build(params, ctx)`` 回傳符合槽位契約的函式
(reranking 的契約:``(query, documents) → documents``,只能重排 /
過濾 / 改分,不得改動切片內容)。``ctx`` 帶 config / embeddings / store,
需要時可取用。
"""

from __future__ import annotations


def build(params, ctx):
    top_k = params.get("top_k", 5)

    def rerank(query, documents):
        # TODO(替換點):換成你的重排邏輯(呼叫內部服務、規則引擎…)。
        query_chars = set(query)

        def overlap(document) -> float:
            content_chars = set(document.page_content)
            if not content_chars:
                return 0.0
            return len(query_chars & content_chars) / len(query_chars | content_chars)

        reranked = sorted(documents, key=overlap, reverse=True)
        for document in reranked:
            document.metadata["score"] = overlap(document)
        return reranked[:top_k]

    return rerank
