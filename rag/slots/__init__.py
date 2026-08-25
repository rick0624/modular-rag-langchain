"""方法實作(import 本套件即完成全部註冊)。

每個子模組對應一個槽位;新增內建方法 = 在對應子模組寫一個
``@register(slot, name)`` 的 builder 函式,框架其他部分零改動。
"""

from rag.slots import (  # noqa: F401
    chunking,
    embedding,
    evaluation,
    formatter,
    fusion,
    generation,
    importing,
    indexing,
    parsing,
    query_transformation,
    reranking,
    retrieval,
    routing,
)
