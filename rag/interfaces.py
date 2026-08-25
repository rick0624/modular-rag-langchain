"""模組間的介面契約(對應舊 repo 的 docs/interfaces.md 與 SLOT_CONTRACTS)。

模組之間傳遞的是 LangChain 的 :class:`~langchain_core.documents.Document`
(``page_content`` / ``metadata`` / ``id``)與
:class:`~langchain_core.messages.BaseMessage`,不自訂資料物件;僅有兩個
輕量 dataclass:

- :class:`Source`:import → parsing 之間的「一份來源文件」。
- :class:`EvalCase`:評估資料集的一筆。

每個槽位的實作(方法)都由 builder ``build(params, ctx) -> 產物`` 建立;
產物是**一個函式**,簽名定義於本檔的型別別名與 :data:`EXPECTED_PARAMS`
(建構期以 ``inspect.signature`` 驗證參數名)。例外:``embedding`` 產出
LangChain ``Embeddings`` 物件、``indexing`` 產出 ``VectorStore`` 物件
(vector store 必須持有 embeddings,查詢端才保證同向量空間)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


@dataclass
class Source:
    """import 槽位輸出的一份來源文件。

    ``path`` 與 ``text`` 至少提供一個:本地檔案給 ``path``;
    來源本身就是文字(API 回應等)時直接給 ``text``。
    ``meta`` 的鍵值會複製進 parsing 產出的 Document metadata。
    """

    doc_id: str
    path: Path | None = None
    text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class EvalCase(BaseModel):
    """評估資料集的一筆:查詢 + 相關文件 doc_id(+ 選填參考答案)。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    relevant_doc_ids: list[str]
    reference_answer: str | None = None


# --- 各槽位產物的函式簽名 ---------------------------------------------------

ImportFn = Callable[[], list[Source]]
"""import:() → 來源清單(來源資訊寫在 params,不吃輸入)。"""

ParseFn = Callable[[list[Source]], list[Document]]
"""parsing:sources → Document 清單(內容變純文字;metadata 帶 doc_id/page)。"""

ChunkFn = Callable[[list[Document]], list[Document]]
"""chunking:documents → 切片清單(metadata 逐塊複製;doc_id 必須保留)。"""

TransformFn = Callable[[list[str]], list[str]]
"""query_transformation:queries → queries(1 筆 = 不拆解;可方法鏈)。"""

RetrieveFn = Callable[[str], list[Document]]
"""retrieval:query → 相關切片(metadata["score"] 越大越相關,降冪)。"""

RerankFn = Callable[[str, list[Document]], list[Document]]
"""reranking:(query, documents) → documents(只能重排/過濾/改分,不得改內容;可方法鏈)。"""

FuseFn = Callable[[list[list[Document]]], list[Document]]
"""fusion:各子查詢的結果清單 → 合併後的 documents(一律執行;單一子查詢時等於截斷)。"""

GenerateFn = Callable[[list[BaseMessage]], str]
"""generation:messages(prompt 由框架組好)→ 答案文字。"""

RouteFn = Callable[[str], dict]
"""routing:原始查詢 → route dict(支線;不影響檢索,附加於輸出)。"""

FormatFn = Callable[[str, list[Document], str | None], Any]
"""formatter:(query, documents, answer) → 對外 payload(終端槽位,型別自由)。"""

EvaluateFn = Callable[[list[EvalCase], list[dict]], dict]
"""evaluation:(測試案例, 各題 query() 完整輸出) → metrics dict。"""


# 建構期簽名驗證用:各槽位產物函式的參數名(順序也要一致)。
EXPECTED_PARAMS: dict[str, tuple[str, ...]] = {
    "import": (),
    "parsing": ("sources",),
    "chunking": ("documents",),
    "query_transformation": ("queries",),
    "retrieval": ("query",),
    "reranking": ("query", "documents"),
    "fusion": ("results",),
    "generation": ("messages",),
    "routing": ("query",),
    "formatter": ("query", "documents", "answer"),
    "evaluation": ("cases", "results"),
}

# 輸入輸出同型別、支援方法鏈(method 清單)的槽位。
CHAINABLE_SLOTS = frozenset({"query_transformation", "reranking"})

# 切片(經框架蓋章後)保證帶有的 metadata 鍵。
META_KEYS = ("doc_id", "seq", "page", "chunk_id")
