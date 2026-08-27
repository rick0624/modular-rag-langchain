# modular-rag-langchain

模組化 RAG(Retrieval-Augmented Generation,檢索增強生成)框架,
[modular-rag-final](https://github.com/rick0624/modular-rag-final)
(Haystack 版)的 LangChain 簡化版。

- RAG 的流程:先從知識庫**檢索**相關內容,再把內容連同問題交給 LLM **生成**回答。
- 本專案的核心是**十三個模組**,每個模組有多種可換的方法。
  換方法只要改 YAML 設定檔的一行,不用改程式。
- 自訂實作用**一個 function** 就能掛進來,不必寫 class、不必改框架。
- 底層用 [LangChain](https://python.langchain.com/) 當零件庫
  (Document、切分器、vector store、LLM 客戶端);流程本身是純 Python
  循序函式,沒有 pipeline graph 引擎,從上往下讀就是整條管線。
- 以 **FastAPI service** 形式執行(ingestion / inference 都是 HTTP 端點)。
- 完全離線也能跑:內建 mock embedding 與 mock LLM,不需金鑰、不需網路。

```
Ingestion:  Import → Parsing → Chunking → Embedding → Indexing        (建索引)
Inference:  查詢 → Query Transformation → 檢索 → 重排 → 融合 → 生成    (回答問題)
Evaluation: 測試集逐題查詢 → hit rate / MRR                            (評估品質)
```

## Quick Start

```bash
pip install -r requirements.txt
python -m rag.service --config configs/default.yaml
```

啟動後(預設配置全離線,會自動 ingest `data/raw/` 的樣本文件):

```bash
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query": "VPN 伺服器位址與連接埠是多少?"}'
```

不起 service、直接跑一輪 建索引 → 查詢 → 評估(請在 repo 根目錄執行):

```bash
python try_rag.py
```

跑測試(也全部離線):

```bash
python -m pytest
```

## 十三個模組

每個模組(在設定檔中叫「槽位」)負責流程中的一件事。
`custom` 是共通選項:掛上自己寫的 function(見[新增自訂方法](#新增自訂方法))。

**Ingestion(建索引)**

| 模組 | 做什麼 | 可用方法 |
|---|---|---|
| 1 Import | 找出要匯入的文件 | `local_files` / `custom` |
| 2 Parsing | 把文件轉成純文字 | `text` / `custom`(PDF、OCR 由此接) |
| 3 Chunking | 把長文切成小段(切片) | `recursive` / `custom` |
| 4 Embedding | 把切片轉成向量 | `mock` / `api` / `custom` |
| 5 Indexing | 把切片與向量存進索引 | `in_memory` / `elasticsearch` / `custom` |

**Inference(回答問題)**

| 模組 | 做什麼 | 可用方法 |
|---|---|---|
| 6 Query Transformation | 改寫查詢讓檢索更準 | `passthrough` / `llm_multi_hyde` / `preqrag` / `custom` |
| 7 Retrieval | 從索引找出相關切片 | `vector` / `custom` |
| 8 Reranking | 把檢索結果重新排序 | `none` / `api` / `insertrank` / `custom` |
| 9 Generation | 用 LLM 生成回答 | `mock` / `openai_compatible` / `gateway_openai_compatible` / `custom` |
| 10 Evaluation | 算檢索品質指標 | `retrieval_metrics` / `custom` |

**選填模組**(設定檔省略 = 用預設或不做):

| 模組 | 做什麼 | 可用方法 |
|---|---|---|
| fusion | 多個子查詢的結果融合(省略 = merge 預設參數) | `merge` / `custom` |
| routing | 查詢分類(結果附在輸出上,不影響檢索) | `keyword_match` / `custom` |
| formatter | 把最終結果組成對外格式 | `simple_json` / `custom` |

每個方法的參數與預設值,直接看 [configs/default.yaml](configs/default.yaml)
(它同時是「方法型錄」:所有方法的參數區塊並存展示,每個參數都有註解);
模組之間的輸入輸出契約見 [docs/interfaces.md](docs/interfaces.md)。

## 專案結構

```
configs/
  default.yaml            預設設定檔,也是「方法型錄」:所有方法與參數都展示在裡面
  online.yaml             線上組合示範(API embedding + ES + API rerank + LLM)
rag/                      框架本體
  config.py               設定檔載入、驗證、${ENV_VAR} 展開
  interfaces.py           模組契約:各槽位的函式簽名、Source / EvalCase
  registry.py             方法註冊表、custom 載入與簽名驗證
  core.py                 核心流程:build_runtime / ingest / query / evaluate
  prompts.py              prompt 組裝(切片帶 [chunk_id] 前綴,可稽核)
  llm.py                  LLM 連線共用底座(llm: 參數區塊)
  service.py              FastAPI service
  logging_setup.py        log 檔設定
  slots/                  各模組的內建方法(一檔一槽位)
examples/custom_modules/  自訂模組範本(13 個槽位各一份,複製來改)
data/                     範例語料與評估集(qa.jsonl)
try_rag.py                不起 service 的試跑腳本
experiment.py             管線組合實驗(批次比較不同方法組合)
tests/                    測試(全部離線,不碰網路)
docs/interfaces.md        模組契約的完整說明
.env.example              金鑰範本(複製成 .env 填入)
```

需要 Python 3.10+;`requirements.txt` 內含 `-e .`,會把 `rag` 套件裝進
環境,自己的腳本才能 `import rag`。

## 執行、測試與實驗

Service 端點:

| 端點 | 說明 |
|---|---|
| `GET /health` | 服務狀態 |
| `POST /ingest` | 重讀設定檔、全量重建索引(切片 id 固定,重跑即 upsert 不倍增) |
| `POST /query` | `{"query": "..."}` → answer、documents、實際送 LLM 的 prompt、每步 trace |
| `POST /evaluate` | 跑評估(hit_rate / MRR);省略 body 用設定檔的 evaluation 區塊 |

- 錯誤一律回 400 + 繁中訊息(指出收到什麼、期望什麼、該調哪個欄位)。
- 環境變數:`RAG_CONFIG`(設定檔路徑)、`RAG_INGEST_ON_STARTUP`、
  `RAG_LOG_LEVEL` / `RAG_LOG_FILE`。
- **每次執行都會寫一個帶時間戳的 log 檔**(`logs/rag-*.log`),檔案收
  DEBUG 全量(每步耗時、檢索分數、完整 prompt);除錯就
  `tail -f logs/rag-*.log`,terminal 保持乾淨。

```bash
python -m pytest              # 全部測試(離線)
python experiment.py          # 批次比較方法組合
```

`experiment.py`:改頂部「實驗定義」區塊,批次生成多組 config 逐組跑
(基線 + 每次換一個槽位,或全交叉);ingestion 相同的組合共用索引、
不重複 embed 語料;每組合自動算 hit_rate / MRR 印成總表。
詳見腳本開頭的說明。

## 設定檔怎麼用

一個設定檔描述一整條 pipeline。每個模組固定這個形狀:

```yaml
  reranking:
    method: none                # 用哪個方法:改這一行就換方法
    method_params:              # 各方法的參數,分區並存
      none: {}
      api:                      # 沒被選中的區塊放著不影響
        endpoint: https://rerank.example.com/v1/rerank
        top_k: 5
```

只有一個方法時也可以寫扁平的 `params:`。要點:

- **方法鏈**:query_transformation、reranking 可以把 `method` 寫成清單
  依序執行,例如 `method: [api, custom]`。
- **金鑰注入**:設定值可寫 `${ENV_VAR}`,載入時從環境變數(或 `.env`)
  展開,金鑰不進版控。
- **錯誤提早報**:方法名打錯、參數打錯、custom 函式簽名不符,都在
  service 啟動時就報錯,訊息會指出位置與可用的選項。

完整的方法與參數示範都在 [configs/default.yaml](configs/default.yaml),
建議直接打開看。

## 接上公司環境

1. 設定檔選用 `configs/online.yaml`(API embedding + Elasticsearch +
   API rerank + LLM 已配置好形狀)
2. 填金鑰:`cp .env.example .env`,填入 API 位址、金鑰與 ES 連線資訊
3. 啟動:`python -m rag.service --config configs/online.yaml`
4. 建索引:`curl -X POST localhost:8000/ingest -d '{}'`(ES 是持久索引,
   啟動時不會自動 ingest)

ES 的三個注意事項:

- `dense_vector` 維度建立後不可變:換 embedding 模型/維度時請換
  `index` 名稱重建。
- 外部系統要直接讀索引欄位時,設 `layout: flat`(Haystack 式扁平文件,
  搭配 `fields` 白名單);預設 layout 的自訂欄位在巢狀 `metadata.*`。
- API 形狀對不上時:embedding / rerank 用 `*_field` 參數對映欄位名;
  LLM 閘道不吃 model 欄位時用 `provider: gateway` /
  `generation: gateway_openai_compatible`。

## 新增自訂方法

公司特有的邏輯(自家的檢索 API、切塊規則…)不用改框架,
寫一個 function 掛上去就好:

**1. 從範本開始改**:[examples/custom_modules/](examples/custom_modules/)
有 13 個槽位各一份可跑的範本,`TODO(替換點)` 標明要換入真實邏輯的位置,
槽位函式的型別註記就是該槽位要求的輸入輸出格式:

```python
# my_reranker.py
def build(params, ctx):
    top_k = params.get("top_k", 5)

    def rerank(query: str, documents: list[Document]) -> list[Document]:
        ...                    # 你的邏輯(呼叫內部服務、規則引擎…)
        return documents[:top_k]

    return rerank
```

**2. 在設定檔掛進模組**:

```yaml
  reranking:
    method: custom
    params:
      file: ./my_reranker.py    # 路徑相對「執行目錄」
      top_k: 3                  # file / function 以外的鍵透傳給 build
```

**3. 跑起來驗證**:啟動 service 或 `python try_rag.py`,看 trace。

`ctx` 帶 `config` / `embeddings` / `store`,需要時可取用(例如自訂
retrieval 直接查 `ctx.store`)。簽名寫錯的話**啟動時**就會報錯並指明
期望的參數名;完整契約表見 [docs/interfaces.md](docs/interfaces.md)。

要把方法做成「內建」(所有專案共用)也很簡單:在 `rag/slots/` 對應
檔案加一個 `@register(槽位, 方法名)` 的 builder 函式即可,框架其他
部分零改動。embedding 的進階參數(`source_field` 挑欄位做向量、
`extra_vectors` 多向量)見 default.yaml 型錄的註解。

## 與 Haystack 版的差異

- **執行引擎**:Haystack 的 component + pipeline graph → LangChain
  零件 + 純 Python 循序函式(`rag/core.py` 從上往下讀就是整條流程)。
- **資料物件**:Haystack `Document` → LangChain `Document`
  (`content`→`page_content`、`meta`→`metadata`);框架保證的
  metadata 鍵(doc_id / seq / page / chunk_id)與舊版相同。
- **自訂模組**:Haystack `@component` class → 一個 builder function。
- **沒有增量 ingest 與 ingestion 指紋**:`/ingest` 一律全量重建,
  靠固定的 chunk_id upsert 保持冪等。
- 未搬過來的方法(sentence-transformers、cross-encoder、BM25/hybrid、
  glossary / jargon / llm_rewrite / llm_decompose、PDF 解析等)都可以
  用 custom 接回,或之後加進內建型錄。
