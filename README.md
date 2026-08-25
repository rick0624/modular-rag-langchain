# modular-rag-langchain

模組化 RAG 框架的 **LangChain 版 MVP**:整條 pipeline 由單一 YAML config
控制,13 個模組(槽位)各自可獨立替換實作 —— 換方法只改 config 的
`method` 一行;自訂實作用**一個 function** 就能掛進來,不必寫 class、
不必改框架。以 **ingestion / inference service**(FastAPI)形式執行。

本專案是 [modular-rag-final](https://github.com/rick0624/modular-rag-final)
(Haystack 版)的簡化遷移:模組邊界與介面契約承襲原設計,執行引擎改為
LangChain 元件 + 純 Python 循序流程(捨棄清單見文末)。

```
Ingestion:import → parsing → chunking → embedding → indexing
                                              │(同一個 Embeddings 物件)
Inference:query ─ routing?(支線)           ▼
           └→ query_transformation → [逐子查詢:retrieval → reranking]
              → fusion → formatter?(支線)→ prompt 組裝 → generation
```

## 模組與內建方法

| 模組 | 內建方法 | custom |
|---|---|---|
| import | `local_files` | ✓ |
| parsing | `text`(txt/md) | ✓(PDF / OCR 由此接) |
| chunking | `recursive` | ✓ |
| embedding | `mock`、`api`(HTTP API,欄位對映) | ✓ |
| indexing | `in_memory`、`elasticsearch` | ✓ |
| query_transformation | `passthrough`(可方法鏈) | ✓ |
| retrieval | `vector` | ✓ |
| reranking | `none`、`api`(HTTP API,欄位對映;可方法鏈) | ✓ |
| generation | `mock`、`openai_compatible`(OpenAI / vLLM / Ollama / 閘道) | ✓ |
| evaluation | `retrieval_metrics`(hit_rate / MRR) | ✓ |
| fusion | `merge`(去重 + 分數排序) | ✓ |
| routing(選填) | `keyword_match` | ✓ |
| formatter(選填) | `simple_json` | ✓ |

各方法的完整參數見 `configs/default.yaml`(同時是「方法型錄」:所有方法
的參數區塊並存,只有被選中的會生效與驗證);模組間的輸入輸出契約見
[docs/interfaces.md](docs/interfaces.md)。

## 快速開始(完全離線)

預設配置不需網路、金鑰或模型下載(mock embedding + in_memory 索引 +
mock 生成):

```bash
pip install -r requirements.txt
python -m rag.service --config configs/default.yaml
```

啟動時會自動 ingest `data/raw/` 的樣本文件(in_memory 索引;持久索引
見下方環境變數說明),然後:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query": "VPN 伺服器位址與連接埠是多少?"}'
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{}'
curl -X POST localhost:8000/evaluate -H 'content-type: application/json' -d '{}'
```

測試(全離線):

```bash
python -m pytest
```

### Service 端點

| 端點 | 說明 |
|---|---|
| `GET /health` | 服務狀態、config 路徑、索引方法 |
| `POST /ingest` | 重讀 config → 全量重建索引(切片 id 確定 → 重跑即 upsert 不倍增)|
| `POST /query` | `{"query": "..."}` → `{answer, prompt, documents, subqueries, routing, output, trace}` |
| `POST /evaluate` | 跑評估;`{"cases": [...]}` 行內給題,省略則用 config 的 evaluation 設定 |

錯誤一律回 400 + 繁中訊息(指出收到什麼、期望什麼、該調哪個欄位)。

環境變數:`RAG_CONFIG`(config 路徑,`--config` 優先)、
`RAG_INGEST_ON_STARTUP`(`auto` 預設 = in_memory 才自動 ingest /
`always` / `never`)。並行模型:單 worker + 全域 lock(in_memory 索引在
行程內,多 worker 會各有一份互不相通的索引)。

## 換方法 = 改一行 config

```yaml
  reranking:
    method: none          # ← 換成 api 或 custom 就完成替換
    method_params:
      none: {}
      api: { endpoint: ..., top_k: 5 }   # 各方法的參數並存,互不干擾
```

`params`(扁平)與 `method_params`(分方法區塊)兩種寫法並存時,
`method_params` 中有當前方法的區塊者優先。機密以 `${ENV_VAR}` 注入
(自動載入 `.env`,見 `.env.example`);`query_transformation` 與
`reranking` 支援方法鏈:`method: [api, custom]` 依序執行。

線上組合(API embedding + Elasticsearch + API rerank + OpenAI 相容生成)
見 `configs/online.yaml`:

```bash
cp .env.example .env   # 填入實際的 API 位址與金鑰
python -m rag.service --config configs/online.yaml
curl -X POST localhost:8000/ingest -d '{}'   # 首次啟動後建索引
```

注意:ES 的 `dense_vector` 維度建立後不可變,換 embedding 模型 / 維度時
請換 `index` 名稱;`embedding: api` 與 `reranking: api` 的請求 / 回應
欄位名都可用參數對映(預設 OpenAI 式),形狀不同時對照錯誤訊息中列出的
實際欄位調整 `*_field` 參數即可。

## 自訂模組(custom):一個 function 就能掛

任何槽位都可以 `method: custom` 掛自己的 .py 檔,**不必改框架、不必寫
class**。慣例:檔案裡提供一個 builder 函式(預設名 `build`),吃
`(params, ctx)`、回傳符合槽位契約的函式:

```python
# my_reranker.py
def build(params, ctx):
    top_k = params.get("top_k", 5)

    def rerank(query, documents):        # 參數名須符合槽位契約
        ...                              # 你的邏輯(呼叫內部服務、規則引擎…)
        return documents[:top_k]

    return rerank
```

```yaml
  reranking:
    method: custom
    params:
      file: ./my_reranker.py    # .py 路徑(相對執行目錄)
      # function: build         # 預設 build
      top_k: 3                  # 其餘鍵原樣透傳給 build 的 params
```

`ctx` 帶 `config` / `embeddings` / `store`,需要時可取用(例如自訂
retrieval 直接查 `ctx.store`)。函式簽名不符契約時,**建構期**就會報錯
並指明期望的參數名。可跑的完整範例見 `examples/custom_modules/`
(自訂 importer 與 reranker,離線可跑,測試有覆蓋)。

各槽位契約(函式參數名 → 回傳值)整理於
[docs/interfaces.md](docs/interfaces.md);新增「內建」方法則是在
`rag/slots/` 對應檔案加一個 `@register(slot, name)` 的 builder(參數
schema 用 pydantic,`extra="forbid"`),框架其他部分零改動。

## 專案結構

```
rag/
├── config.py        # YAML schema(method/params/method_params)、${ENV} 展開、.env
├── interfaces.py    # 13 槽位契約:Source/EvalCase、函式簽名、EXPECTED_PARAMS
├── registry.py      # 註冊表、@register、build_slot(方法鏈、custom 載入、簽名驗證)
├── core.py          # build_runtime / ingest / query / evaluate(循序流程 + trace)
├── prompts.py       # prompt 組裝(切片帶 [chunk_id] 前綴,可稽核)
├── service.py       # FastAPI:/health /ingest /query /evaluate
└── slots/           # 各槽位的內建方法(一檔一槽位)
configs/             # default.yaml(離線型錄)、online.yaml(ES + API)
examples/custom_modules/   # 自訂模組範例(function-based)
data/                # 樣本語料與評估集
docs/interfaces.md   # 介面契約
tests/               # 全離線測試(含 service 端到端)
```

## 相對 Haystack 版刻意捨棄的機制

MVP 求簡,以下原版機制**刻意**不保留(需要時再加回):
pipeline graph engine(改為循序函式)、建構期語意相容檢查
(content_type / pages / 索引能力宣告)、ingestion 指紋與增量 ingest
(以確定 chunk_id 的全量 upsert 取代;持久索引中已刪來源會留舊切片)、
parsing 方法鏈與 PDF 解析(custom 可接)、LLM 查詢改寫 / 拆解 / 重排
的內建方法(llm_rewrite / decompose / HyDE / insertrank;自訂 transform
即可接回,多子查詢 + fusion 的通道仍在)、sentence-transformers 與
cross-encoder 本地模型、in-memory BM25 / hybrid、實驗掃描腳本。
