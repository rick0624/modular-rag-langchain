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

這兩個槽位一樣支援 custom,差別是產物**不是單一函式,而是帶兩個方法的
物件**(duck-type 檢查,不必繼承任何 LangChain class):

| 槽位 | 範本 | 產物需要的方法 |
|---|---|---|
| embedding | `my_embeddings.py` | `embed_documents(texts) -> list[list[float]]`、`embed_query(text) -> list[float]`(同一物件服務建索引與查詢端 → 同向量空間) |
| indexing | `my_indexing.py` | `add_documents(documents, ids=...)`、`similarity_search_with_score(query, k=...) -> [(Document, 分數), ...]`(向量用 `ctx.embeddings` 算) |

一般情況用內建的 `api` embedding(欄位對映)與 `in_memory` /
`elasticsearch` indexing 就夠;要接自研索引、非 LangChain 生態的
向量庫時再自訂。`source_field` / `extra_vectors` 是框架層參數,
所有 embedding 方法(含 custom)都支援,不用在自訂模組裡處理。
