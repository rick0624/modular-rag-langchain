# 方法型錄

每個模組(槽位)可用的方法與參數的完整清單。與本文件互補的兩份:
[configs/default.yaml](../configs/default.yaml) 是**可跑的示範**(同樣
的參數以註解展示),[interfaces.md](interfaces.md) 是**模組間的契約**
(輸入輸出形狀)。參數以 pydantic 驗證(`extra="forbid"`):多打或
打錯欄位會在啟動時直接報錯,並列出可接受的參數。

## 共通機制

### `custom`:掛自訂實作(所有槽位皆可用)

| 參數 | 預設 | 說明 |
|---|---|---|
| `file` | **必填** | 自訂 .py 檔路徑(相對執行目錄) |
| `function` | `build` | 檔案內的 builder 函式名,簽名 `build(params, ctx)` |
| (其餘鍵) | — | 原樣透傳給 builder 的 `params` |

範本見 [examples/custom_modules/](../examples/custom_modules/),
契約見 [interfaces.md](interfaces.md)。

### `llm:` 區塊:方法內部用的 LLM 連線

`llm_multi_hyde`、`preqrag`、`insertrank` 都需要這個區塊;多個方法
共用同一組連線時用 YAML anchor(`&llm` / `*llm`)去重。

| 參數 | 預設 | 說明 |
|---|---|---|
| `provider` | **必填** | `mock`(離線固定回覆)/ `openai_compatible`(OpenAI SDK,請求必帶 model)/ `gateway`(手寫 OpenAI 式 client,model 可不帶) |
| `replies` | None | mock 用:固定回覆清單,依序循環 |
| `model` | None | openai_compatible **必填**;gateway 選填(None = 請求完全不帶 model 欄位) |
| `base_url` | None | openai_compatible:None = 官方 OpenAI;gateway:**必填** |
| `completions_path` | `/chat/completions` | gateway 的補全端點路徑 |
| `api_key` | None | openai_compatible:None = 用 `OPENAI_API_KEY` 環境變數;gateway:None = 不帶 Authorization |
| `temperature` | None | None = 不帶此欄位 |
| `max_tokens` | None | None = 不帶此欄位 |
| `timeout` | None | None = client 預設(gateway 為 60 秒) |
| `headers` | `{}` | 額外 HTTP 標頭 |

## Ingestion

### 1. Import

**`local_files`** — 掃描本地資料夾;`doc_id` = 檔案相對 `input_dir` 的路徑。

| 參數 | 預設 | 說明 |
|---|---|---|
| `input_dir` | **必填** | 要匯入的資料夾 |
| `extensions` | `[".txt", ".md"]` | 收錄的副檔名(小寫比對) |
| `recursive` | true | 是否含子資料夾 |

### 2. Parsing

**`text`** — 純文字檔(txt / md);PDF、OCR 等其他格式以 `custom` 接。

| 參數 | 預設 | 說明 |
|---|---|---|
| `encoding` | `utf-8` | 文字檔編碼 |

### 3. Chunking

**`recursive`** — 遞迴字元切分(段落 → 行 → 中文句號 → 空白 → 字元)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `chunk_size` | 300 | 每片字元數上限 |
| `chunk_overlap` | 50 | 相鄰切片重疊字元數(需 < chunk_size) |
| `separators` | 內建 | 切分邊界優先序;預設 `["\n\n", "\n", "。", " ", ""]` |

### 4. Embedding

**框架層參數**(所有方法含 `custom` 皆支援,由 ingest 流程消費):

| 參數 | 預設 | 說明 |
|---|---|---|
| `source_field` | None | 主向量的來源欄位:None = 切片內文;指定 chunking 生成的 metadata 欄位時,拿該欄位做 embedding、prompt 仍給切片內文(解耦);切片缺該欄位時 ingest 直接報錯 |
| `extra_vectors` | None | 額外向量 `{向量欄位名: 來源欄位}`:用同一個模型對額外欄位各出一組向量,寫進切片 metadata;內建檢索只用主向量,額外向量供 custom retrieval 用。欄位名不可用框架保留名 |

**`mock`** — 離線確定性詞袋向量(開發測試用)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `dim` | 256 | 向量維度 |

**`api`** — 通用 HTTP embedding API(回應形狀以參數對映,預設 OpenAI 式)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `endpoint` | **必填** | API 完整 URL |
| `headers` | `{}` | 額外 HTTP 標頭(認證放這裡) |
| `model` | None | None = 請求不帶 model 欄位 |
| `batch_size` | 16 | 每批文字數 |
| `timeout` | 30.0 | 逾時秒數 |
| `texts_field` | `input` | 請求中放文字清單的欄位名 |
| `model_field` | `model` | 請求中放模型的欄位名 |
| `embeddings_field` | `data` | 回應中向量清單的欄位(支援 `a.b` 巢狀路徑;回應本身是清單時設 null) |
| `item_field` | `embedding` | 清單元素是物件時向量所在欄位;元素直接是向量時設 null |

### 5. Indexing

**框架層參數**(所有方法含 `custom` 皆支援):

| 參數 | 預設 | 說明 |
|---|---|---|
| `fields` | None | 自訂欄位白名單 + 改名 `{索引欄位名: meta 欄位名}`;None = 自訂欄位全帶。設定後未列出的自訂欄位不寫入;框架欄位與 extra_vectors 向量欄位永遠保留。改對映請重建索引 |

**`in_memory`** — 行程內索引(無額外參數;重啟即消失)。

**`elasticsearch`**

| 參數 | 預設 | 說明 |
|---|---|---|
| `es_url` | **必填** | ES 位址,如 `http://localhost:9200` |
| `index` | `modular-rag` | 索引名稱;dense_vector 維度建立後不可變,換 embedding 模型/維度請換索引名 |
| `username` / `password` | None | basic auth,需成對(與 api_key 擇一) |
| `api_key` | None | API key 認證 |
| `layout` | `nested` | `nested` = 自訂欄位在巢狀 `metadata.*`;`flat` = 扁平文件(所有欄位在頂層,外部系統可直接讀,由框架自行讀寫)。換 layout 請換索引名重建 |
| `query_field` | `text` | 索引中放切片內文的欄位名 |
| `vector_query_field` | `vector` | 索引中放向量的欄位名(dense_vector) |
| `custom_mapping` | None | 自訂 index mapping(完整覆蓋,須自含上兩個欄位的宣告);僅在索引不存在、由框架建立時生效 |
| `settings` | None | index settings(analyzer、`index.default_pipeline` 等);必須搭配 custom_mapping |
| `request_timeout` | None | 單一請求逾時秒數;None = client 預設 10 秒 |
| `retry_on_timeout` | None | 逾時自動重試;None = client 預設 false |
| `max_retries` | None | 單次請求最大重試次數;None = client 預設 3 |

## Inference

### 6. Query Transformation(可方法鏈)

**`passthrough`** — 原樣通過(無參數)。

**`llm_multi_hyde`** — LLM 生成 k 篇多角度假設文件,各自當一路查詢檢索後融合。

| 參數 | 預設 | 說明 |
|---|---|---|
| `num_documents` | 3 | 假設文件篇數 |
| `keep_original` | true | 原查詢也保留一路 |
| `prompt` | 內建 | 自訂 prompt,需含 `{query}` |
| `llm` | **必填** | LLM 連線(見上方 llm 區塊) |

**`preqrag`** — 先分類 single / multi:單一主題改寫、複合問題拆解。

| 參數 | 預設 | 說明 |
|---|---|---|
| `num_rewrites` | 2 | single 分支的改寫條數 |
| `max_subqueries` | 4 | multi 分支的子查詢數上限 |
| `include_original` | true | 原查詢也保留一路 |
| `classify_prompt` / `rewrite_prompt` / `decompose_prompt` | 內建 | 自訂 prompt,需含 `{query}` |
| `llm` | **必填** | LLM 連線 |

### 7. Retrieval

**`vector`** — 向量相似度檢索(查 indexing 建的 store)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `top_k` | 5 | 取回筆數 |

### 8. Reranking(可方法鏈)

**`none`** — 不重排(無參數)。

**`api`** — 通用 HTTP rerank API(請求/回應形狀以參數對映)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `endpoint` | **必填** | API 完整 URL |
| `headers` | `{}` | 額外 HTTP 標頭 |
| `model` | None | None = 請求不帶 model 欄位 |
| `top_k` | 5 | 重排後保留筆數 |
| `timeout` | 30.0 | 逾時秒數 |
| `query_field` | `question` | 請求中放查詢的欄位名 |
| `documents_field` | `documents` | 請求中放候選文字清單的欄位名 |
| `model_field` | `model` | 請求中放模型的欄位名 |
| `results_field` | `returnData` | 回應中結果清單的欄位(`a.b` 巢狀路徑;回應本身是清單時設 null) |
| `index_field` | `index` | 結果元素中候選序號的欄位名 |
| `score_field` | `score` | 結果元素中分數的欄位名 |
| `index_base` | 0 | 回應 index 的起算值(1 起算的 API 設 1,設錯會整體位移) |
| `higher_is_better` | true | false = API 回傳的是距離(越小越相關) |
| `raise_on_failure` | false | false = API 掛掉時保留原檢索順序並記 WARNING(fail-soft);初次接線建議設 true |

**`insertrank`** — 候選附檢索分數的 LLM listwise 重排;LLM 回覆解析失敗時保留原順序。

| 參數 | 預設 | 說明 |
|---|---|---|
| `top_k` | 5 | 重排後保留筆數 |
| `score_label` | `檢索分數` | prompt 中分數的名稱,依上游據實描述 |
| `prompt` | 內建 | 自訂 prompt,需含 `{query}` 與 `{documents}` |
| `llm` | **必填** | LLM 連線 |

### 9. Generation

**`mock`** — 離線假答案。

| 參數 | 預設 | 說明 |
|---|---|---|
| `replies` | 內建一句 | 固定回覆清單,依序循環 |

**`openai_compatible`** — OpenAI SDK(OpenAI / vLLM / Ollama;請求必帶 model)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `model` | **必填** | 模型名稱 |
| `base_url` | None | None = 官方 OpenAI;vLLM / Ollama 填其 /v1 位址 |
| `api_key` | None | None = 用 `OPENAI_API_KEY` 環境變數 |
| `temperature` | None | None = 不帶此欄位 |
| `max_tokens` | None | None = 不帶此欄位 |
| `timeout` | None | None = client 預設 |
| `headers` | `{}` | 額外 HTTP 標頭 |

**`gateway_openai_compatible`** — OpenAI 式的公司閘道(手寫 client,model 可不帶)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `base_url` | **必填** | 閘道的 /v1 位址 |
| `completions_path` | `/chat/completions` | 補全端點路徑 |
| `model` | None | None = 請求完全不帶 model 欄位 |
| `api_key` | None | None = 不帶 Authorization 標頭 |
| `temperature` / `max_tokens` | None | None = 不帶此欄位 |
| `timeout` | None | None = 60 秒 |
| `headers` | `{}` | 額外 HTTP 標頭 |

### 10. Evaluation

**`retrieval_metrics`** — hit_rate / MRR(以 doc_id 計,同文件多切片算一次)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `dataset_path` | None | JSONL 測試集路徑,每行 `{"query", "relevant_doc_ids", "reference_answer"?}` |
| `cases` | None | 行內測試案例(優先於 dataset_path);兩者至少給一個 |

## 選填模組

### Fusion

**`merge`** — 各子查詢結果串接、依 chunk_id 去重(保最高分)、分數降冪。
省略整個 `fusion:` 區塊時,以本方法的預設參數執行。

| 參數 | 預設 | 說明 |
|---|---|---|
| `top_k` | 5 | 合併後保留的筆數上限 |

### Routing

**`keyword_match`** — 關鍵字規則分類(結果進輸出的 `routing` 鍵,不影響檢索)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `routes` | **必填** | `{類別: [關鍵字, ...]}` |
| `default_category` | `general` | 無命中時的類別 |

### Formatter

**`simple_json`** — 通用 JSON 形狀(結果進輸出的 `output` 鍵)。

| 參數 | 預設 | 說明 |
|---|---|---|
| `include_content` | true | false = 引用只留識別資訊,不含切片內容 |

## 非方法的相關設定

- **`inference.prompt`**(選填):答案生成的 prompt —— `template`
  (需含 `{query}` 與 `{context}`;None = 內建繁中模板)與 `system`
  (None = 不帶 system 訊息)。
- **方法鏈**:`query_transformation` 與 `reranking` 的 `method` 可寫
  清單依序執行(如 `method: [api, custom]`);其他槽位不支援。
- 模組間的輸入輸出契約、metadata 鍵與不變量,見
  [interfaces.md](interfaces.md)。
