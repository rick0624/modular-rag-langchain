"""答案生成的 prompt 組裝(切片帶 ``[chunk_id]`` 前綴,引用可回溯)。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

DEFAULT_TEMPLATE = """\
請根據以下內容回答問題;內容不足以回答時,請直接說明無法回答,不要編造。

{context}

問題:{query}
回答:"""


def build_messages(
    query: str,
    documents: list[Document],
    *,
    template: str | None = None,
    system: str | None = None,
) -> tuple[list[BaseMessage], str]:
    """組出送給 generation 槽位的 messages,並回傳實際 prompt 文字。

    模板佔位符:``{context}``(各切片以 ``[chunk_id]`` 前綴串接)與
    ``{query}``。以字串替換而非 str.format,模板中其他大括號(JSON 範例
    等)不需逸出。
    """
    context = "\n\n".join(
        f"[{doc.metadata.get('chunk_id', doc.id or '?')}] {doc.page_content}"
        for doc in documents
    )
    prompt = (template or DEFAULT_TEMPLATE).replace("{context}", context).replace(
        "{query}", query
    )
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(system))
    messages.append(HumanMessage(prompt))
    return messages, prompt
