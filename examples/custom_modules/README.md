# 自訂模組範本

**用法:複製對應槽位的範本,改掉 `TODO(替換點)`,再用 config 掛上。**
每份範本都離線可跑(測試有覆蓋),函式簽名上的型別就是該槽位要求的
input / output 格式 —— 範本即契約(完整定義見
[docs/interfaces.md](../../docs/interfaces.md) 與 `rag/interfaces.py`)。

```yaml
  某槽位:
    method: custom
    params:
      file: ./examples/custom_modules/my_xxx.py   # .py 路徑(相對執行目錄)
      # function: build      # 選填,預設 build
      # 其餘鍵原樣透傳給 build 的 params
```

## 範本一覽(function 槽位)

| 槽位 | 範本 | 槽位函式簽名 |
|---|---|---|
| import | `my_importer.py` | `() -> list[Source]` |
| parsing | `my_parser.py` | `(sources: list[Source]) -> list[Document]` |
| chunking | `my_chunker.py` | `(documents: list[Document]) -> list[Document]` |
| query_transformation | `my_query_transform.py` | `(queries: list[str]) -> list[str]` |
| retrieval | `my_retriever.py` | `(query: str) -> list[Document]` |
| reranking | `my_reranker.py` | `(query: str, documents: list[Document]) -> list[Document]` |
| fusion | `my_fusion.py` | `(results: list[list[Document]]) -> list[Document]` |
| generation | `my_generator.py` | `(messages: list[BaseMessage]) -> str` |
| routing | `my_router.py` | `(query: str) -> dict` |
| formatter | `my_formatter.py` | `(query: str, documents: list[Document], answer: str \| None) -> Any` |
| evaluation | `my_evaluator.py` | `(cases: list[EvalCase], results: list[dict]) -> dict` |

參數名必須跟表上一致 —— 建構期會用 `inspect.signature` 驗證,寫錯會在
啟動時直接報錯並指明期望的參數名。`build(params, ctx)` 的 `ctx` 帶
`config` / `embeddings` / `store`,需要時取用(`my_retriever.py` 有示範)。

## 兩個「物件槽位」:embedding 與 indexing

這兩個槽位的產物不是函式,而是 LangChain 介面的物件(沒有範本):

- **embedding**:`build(params, ctx)` 回傳實作 `embed_documents(texts)` 與
  `embed_query(text)` 的物件(LangChain `Embeddings` 介面)。
- **indexing**:回傳實作 `add_documents(docs, ids=...)` 與
  `similarity_search_with_score(query, k=...)` 的物件(LangChain
  `VectorStore` 介面),建立時請使用 `ctx.embeddings`。

一般情況用內建的 `api` embedding(欄位對映)與 `in_memory` /
`elasticsearch` indexing 就夠;真的要換底層時再自訂。
