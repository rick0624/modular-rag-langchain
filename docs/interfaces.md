# 介面契約

模組之間傳遞的是 LangChain 的 `Document`(`page_content` / `metadata` /
`id`)與 `BaseMessage`,不自訂資料物件;僅有兩個輕量 dataclass:
`Source`(import → parsing 之間的一份來源)與 `EvalCase`(評估資料一筆)。
契約 = **每個模組的輸入輸出形狀** + **Document 的 metadata 鍵**;兩者不變,
方法怎麼換都不影響其他模組。可用的方法選項與參數見
[methods.md](methods.md)。

每個方法都是一個 builder 函式 `build(params, ctx) -> 產物`,產物是
**一個函式**(簽名見下表;`method: custom` 的產物在建構期以
`inspect.signature` 驗證,不符直接報錯並指明期望簽名)。例外:
`embedding` 產出 LangChain `Embeddings` 物件、`indexing` 產出
`VectorStore` 物件 —— vector store 必須持有 embeddings,查詢端才是
結構性的同向量空間保證。

## 1. 模組 Input / Output 一覽

| 模組 | 產物簽名 | 說明 |
|---|---|---|
| 1 Import | `() -> list[Source]` | 來源資訊寫在 params;每筆 `Source` 必帶穩定的 `doc_id`,本地檔給 `path`、文字給 `text`,`meta` 會複製進切片 |
| 2 Parsing | `(sources) -> list[Document]` | 內容變純文字;metadata 帶 `doc_id` / `page`(無分頁概念時為 1) |
| 3 Chunking | `(documents) -> list[Document]` | 切片;metadata 逐塊複製,`doc_id` 必須保留 |
| 4 Embedding | `Embeddings` 物件 | 建索引與查詢共用同一物件(同向量空間);換方法或模型後必須重建索引。框架層參數(所有方法皆支援):`source_field` 指定主向量的來源 metadata 欄位(prompt 仍給切片內文);`extra_vectors` {向量欄位: 來源欄位} 產出額外向量寫進 metadata,供 custom retrieval 使用 |
| 5 Indexing | `VectorStore` 物件 | 寫入由框架以 `add_documents(ids=chunk_id)` 進行 → 重跑 ingest 即 upsert |
| 6 Query Transformation | `(queries) -> list[str]` | 1 筆 = 不拆解(傳統流程);N 筆 = 逐子查詢檢索後由 fusion 合併;可方法鏈 |
| 7 Retrieval | `(query) -> list[Document]` | `metadata["score"]` 越大越相關、結果降冪 |
| 8 Reranking | `(query, documents) -> list[Document]` | 只能重排 / 過濾 / 改分,不得改內容;可方法鏈 |
| 9 Generation | `(messages) -> str` | prompt 由框架組好(`inference.prompt` 設定模板)才送進來 |
| 10 Evaluation | `(cases, results) -> dict` | `results` 為各題 `query()` 完整輸出;回傳 `{metrics, per_case}` |
| Fusion | `(results) -> list[Document]` | `results` = 各子查詢的結果清單;**一律執行**,單一子查詢時等於去重 + 截斷 |
| Routing(選填支線) | `(query) -> dict` | 吃原始查詢(不經 transform);不影響檢索,附加於輸出的 `routing` 鍵 |
| Formatter(選填終端支線) | `(query, documents, answer) -> Any` | 進輸出的 `output` 鍵;型別自由(終端槽位特權),走 HTTP 須可序列化 |

custom module(`method: custom`)掛進槽位時,builder 回傳的函式參數名
必須符合上表(定義於 `rag/interfaces.py` 的 `EXPECTED_PARAMS`)——
建構期驗證,不符直接報錯並指明缺什麼。

## 2. Document metadata 鍵

切片(經框架蓋章後)保證帶有:

| metadata 鍵 | 語意 |
|---|---|
| `doc_id` | 來源文件識別碼(local_files = 檔案相對 `input_dir` 的路徑);跨執行穩定,upsert / 評估靠它 |
| `seq` | 文件內切片序號(0 起) |
| `page` | 來源頁碼(1 起;非分頁來源為 1) |
| `chunk_id` | `"{doc_id}::chunk_{seq}"`;**同時是 `Document.id`**(→ 寫入即 upsert;代價:切片內容此後不得改動) |

檢索之後另帶 `score`(越大越相關)。自訂 chunker 可額外生成自訂
metadata 欄位,但不可使用框架保留名(`doc_id` / `seq` / `page` /
`chunk_id` / `score`);embedding 的 `source_field` / `extra_vectors`
可引用這些自訂欄位(`extra_vectors` 的向量欄位名同樣不可用保留名)。

## 3. 不變量(修改程式時不可破壞)

- **分數**:越大越相關、結果降冪;只在同一次結果內可比(不可跨方法比較)。
- **識別碼確定性**:同輸入必同 `doc_id` / `chunk_id`。
- **同向量空間**:查詢端與建索引共用 `ingestion.embedding` 建出的同一個
  `Embeddings` 物件;換 embedding 方法或模型後必須重建索引。
- **prompt 可稽核**:`query()` 回傳的 `prompt` 即實際送 LLM 的內容,
  切片帶 `[chunk_id]` 前綴,引用可回溯。
- **邊界映射責任**:外部系統的欄位在 custom 模組內部轉成 canonical 型別
  (內文 → `page_content`、分數 → `metadata["score"]`、其餘進 `metadata`);
  模組之間永遠只流 canonical 型別。

## 4. 建構期檢查

不合法組合在建 Runtime 時直接報錯(不會跑到一半才炸):

1. **方法存在**:`method` 不在註冊表 → 列出該槽位所有可用方法。
2. **參數 schema**:各方法的參數以 pydantic(`extra="forbid"`)驗證,
   多打 / 打錯欄位直接報錯並列出可接受的參數。
3. **產物簽名**:function 槽位比對函式參數名;embedding / indexing
   槽位 duck-type 檢查(`embed_query` / `add_documents` …)。
4. **方法鏈限制**:只有 `query_transformation` 與 `reranking`
   (輸入輸出同型)接受 `method` 清單。

## 5. 新需求怎麼接

1. 同一模組的另一種做法 → 通用的寫進 `rag/slots/`(一個
   `@register(slot, name)` builder);特定系統的寫 custom module
   (`file` + `function`,零框架改動)。九成需求在這裡。
2. 資料要多帶資訊 → 加 metadata 鍵(只加不改)。
3. 流程真的要多一步 → 改 `rag/core.py` 的 `ingest()` / `query()`。
