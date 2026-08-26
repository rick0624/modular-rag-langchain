#!/usr/bin/env python
"""管線組合實驗:程式化生成 config、逐組合執行查詢與評估、比較結果。

用法:改下面「實驗定義」區塊,然後(在 repo 根目錄):

    python experiment.py

或在你自己的腳本裡::

    from experiment import run_experiments
    records = run_experiments()
    for rec in records:
        ...  # rec["results"] 是 [(query, query() 的完整輸出), ...]

每筆 record 都帶出處:label(人讀的組合名)、overrides(這個組合改了
哪些槽位)、config(完整 config dict,可存檔重現)、metrics(config 有
evaluation 區塊且 EVALUATE=True 時的 hit_rate / MRR)。

行為要點:
- **ingestion 區塊相同的組合共用同一個索引**(只重建 inference 端),
  掃 inference 槽位時不會重複 embed 語料。
- 一個壞組合不中斷整批(record 帶 error)。
- 注意(ES 實驗):不同 ingestion 變體(換 embedding 模型 / chunking /
  layout / mapping)必須用不同 index 名,否則會互寫同一個索引。
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from rag import build_runtime, evaluate, ingest, parse_config, query

# ─────────────────────────── 實驗定義(改這裡)───────────────────────────

BASE_CONFIG = "configs/default.yaml"

# "one_at_a_time":基線 + 每次只動一個槽位(每槽位每方法各跑一次)。
# "product":SLOT_OPTIONS 全交叉乘積。小網格 = 把 SLOT_OPTIONS 收窄到
#            2-3 個有交互作用的槽位後用 product,不需要第三種模式。
MODE = "one_at_a_time"

# 槽位 → 要比較的選項清單。選項的四種寫法:
#   字串   → 只覆蓋 method,參數沿用 base config 的 method_params 型錄
#   list  → 方法鏈(如 ["none", "custom"];限 query_transformation / reranking)
#   dict  → 整個槽位配置直接替換(要自訂參數、或 fusion / routing 這類
#           基底沒啟用的槽位時用),如:
#           {"method": "custom", "params": {"file": "./my_reranker.py"}}
#   bundle → key 含「.」的 dict:一次覆蓋多個槽位,綁定一起換(例如
#           embedding 模型與 ES index 名必須一致換)。外層 key 只是維度名
#           (自取,不必是槽位路徑);"_label" 選填,當這個選項的名字:
#           "embedding_stack": [
#               {"_label": "openai-small",
#                "ingestion.embedding": {"method": "api", "params": {...}},
#                "ingestion.indexing": {"method": "elasticsearch", "params": {...}}},
#           ]
#           同一個槽位只能屬於一個維度(重疊時直接報錯)。
SLOT_OPTIONS: dict[str, list[Any]] = {
    "inference.reranking": [
        "none",
        {"method": "custom",
         "params": {"file": "./examples/custom_modules/my_reranker.py", "top_k": 3}},
    ],
    "inference.retrieval": [
        {"method": "vector", "params": {"top_k": 3}},
        {"method": "vector", "params": {"top_k": 8}},
    ],
}

QUERIES = [
    "VPN 伺服器位址與連接埠是多少?",
    "特休假滿兩年有幾天?",
]

# True = base config 有 evaluation 區塊時,每組合以其測試集算 hit_rate / MRR。
EVALUATE = True

# ─────────────────────────── 組合生成 ───────────────────────────


def apply_option(cfg: dict, dotted_slot: str, option: Any) -> None:
    """把一個選項套進 config dict(就地修改)。"""
    section, slot = dotted_slot.split(".")
    if isinstance(option, dict):
        cfg[section][slot] = copy.deepcopy(option)
    else:  # 字串或方法鏈:只換 method,method_params 型錄原樣保留
        cfg[section].setdefault(slot, {})["method"] = option


def is_bundle(option: Any) -> bool:
    """bundle 選項 = key 含「.」的 dict(一次覆蓋多個槽位)。

    不會與「整個槽位配置」的 dict 寫法混淆:後者的 key 是 method /
    params 等欄位名,不含「.」。
    """
    return isinstance(option, dict) and any("." in key for key in option)


def resolve_overrides(dim: str, option: Any) -> dict[str, Any]:
    """把一個選項展成 {dotted_slot: 選項} 的覆蓋表。"""
    if is_bundle(option):
        return {key: value for key, value in option.items() if key != "_label"}
    if "." not in dim:
        raise ValueError(
            f"維度 {dim!r} 不是槽位路徑(缺少「.」),它的選項必須是 bundle"
            f"(key 含「.」的 dict),但得到:{option!r}"
        )
    return {dim: option}


def check_dimensions(slot_options: dict[str, list[Any]]) -> None:
    """同一個槽位只能屬於一個維度,否則後套用的會安靜地蓋掉先套用的。"""
    owner: dict[str, str] = {}  # dotted_slot → 首見的維度名
    for dim, options in slot_options.items():
        for option in options:
            for slot in resolve_overrides(dim, option):
                if owner.setdefault(slot, dim) != dim:
                    raise ValueError(
                        f"維度 {owner[slot]!r} 與 {dim!r} 都覆蓋槽位 {slot!r},"
                        "請把重疊的槽位合併進同一個維度(bundle)"
                    )


def dim_name(dim: str) -> str:
    """維度的顯示名:槽位路徑取槽位名,bundle 維度用原名。"""
    return dim.split(".")[1] if "." in dim else dim


def option_name(option: Any) -> str:
    """選項的簡短名稱(組 label 用)。"""
    if is_bundle(option):
        if option.get("_label"):
            return str(option["_label"])
        return "+".join(  # 沒給 _label 時的後備名:各槽位的選項名串起來
            f"{dim_name(slot)}:{option_name(opt)}"
            for slot, opt in option.items()
            if slot != "_label"
        )
    if isinstance(option, dict):
        method = str(option.get("method", "custom"))
        # 帶上純量參數,同方法不同參數的變體才分得出來(長值截尾)
        scalars = {
            key: value
            for key, value in (option.get("params") or {}).items()
            if isinstance(value, (str, int, float, bool))
        }
        if scalars:
            def _short(value: Any) -> str:
                text = str(value)
                return text if len(text) <= 14 else "…" + text[-12:]

            joined = ",".join(f"{k}={_short(v)}" for k, v in scalars.items())
            return f"{method}({joined})"
        return method
    if isinstance(option, list):
        return "+".join(map(str, option))
    return str(option)


def make_variants(
    base: dict, mode: str, slot_options: dict[str, list[Any]]
) -> list[tuple[str, dict, dict]]:
    """回傳 [(label, overrides, config_dict), ...]。overrides 已展平為
    {dotted_slot: 選項}(bundle 展開、_label 移除),可直接重現該組合。"""
    check_dimensions(slot_options)
    variants: list[tuple[str, dict, dict]] = []
    if mode == "one_at_a_time":
        variants.append(("baseline", {}, copy.deepcopy(base)))
        for dim, options in slot_options.items():
            for option in options:
                cfg = copy.deepcopy(base)
                overrides = resolve_overrides(dim, option)
                for dotted, opt in overrides.items():
                    apply_option(cfg, dotted, opt)
                variants.append(
                    (f"{dim_name(dim)}={option_name(option)}", overrides, cfg)
                )
    elif mode == "product":
        dims = list(slot_options)
        for combo in itertools.product(*(slot_options[d] for d in dims)):
            cfg = copy.deepcopy(base)
            overrides: dict[str, Any] = {}
            for dim, option in zip(dims, combo):
                overrides.update(resolve_overrides(dim, option))
            for dotted, opt in overrides.items():
                apply_option(cfg, dotted, opt)
            label = " + ".join(
                f"{dim_name(d)}={option_name(o)}" for d, o in zip(dims, combo)
            )
            variants.append((label, overrides, cfg))
    else:
        raise ValueError(f"未知的 MODE:{mode!r}(可用:one_at_a_time / product)")
    return variants


# ─────────────────────────── 執行 ───────────────────────────


def run_experiments(
    base_config: str | None = None,
    mode: str | None = None,
    slot_options: dict[str, list[Any]] | None = None,
    queries: list[str] | None = None,
    evaluate_variants: bool | None = None,
) -> list[dict]:
    """逐組合建 Runtime、跑查詢(與評估);ingestion 相同的組合共用索引。

    參數省略時使用模組頂部的實驗定義;可從自己的腳本 import 並覆寫。
    """
    base_config = base_config if base_config is not None else BASE_CONFIG
    mode = mode if mode is not None else MODE
    slot_options = slot_options if slot_options is not None else SLOT_OPTIONS
    queries = queries if queries is not None else QUERIES
    evaluate_variants = evaluate_variants if evaluate_variants is not None else EVALUATE

    load_dotenv()  # 金鑰由 .env 注入(已存在的環境變數優先);展開交給 parse_config
    base = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    records: list[dict] = []
    stores: dict[str, Any] = {}  # ingestion 區塊 JSON → 已建好的索引

    for label, overrides, cfg_dict in make_variants(base, mode, slot_options):
        record: dict[str, Any] = {
            "label": label, "overrides": overrides, "config": cfg_dict,
            "results": [], "metrics": None, "error": None,
        }
        records.append(record)
        try:
            # 共用 key 用展開前的內容:機密(${ENV})不進 key,語意同舊版指紋
            key = json.dumps(cfg_dict["ingestion"], sort_keys=True)
            config = parse_config(cfg_dict, source=label)
            if key in stores:  # 索引沿用,只重建 inference 端
                runtime = build_runtime(config, store=stores[key])
            else:
                runtime = build_runtime(config)
                ingest(runtime)
                stores[key] = runtime.store
            record["results"] = [(q, query(runtime, q)) for q in queries]
            if evaluate_variants and config.evaluation is not None:
                record["metrics"] = evaluate(runtime)["metrics"]
        except Exception as exc:  # 一個壞組合不中斷整批
            record["error"] = f"{type(exc).__name__}: {exc}"
    return records


def _format_record(record: dict) -> str:
    if record["error"]:
        return f"[FAIL] {record['label']}: {record['error']}"
    docs_per_query = [len(result["documents"]) for _, result in record["results"]]
    line = f"[OK]   {record['label']}  各題檢回 {docs_per_query}"
    if record["metrics"]:
        line += (
            f"  hit_rate={record['metrics']['hit_rate']:.3f}"
            f"  mrr={record['metrics']['mrr']:.3f}"
        )
    return line


if __name__ == "__main__":
    from rag.logging_setup import default_log_file, setup_logging

    log_file = default_log_file()
    setup_logging("warning", log_file)  # terminal 只留總表;細節在 log 檔
    print(f"log 檔:{log_file}(DEBUG 全量)")
    for rec in run_experiments():
        print(_format_record(rec))
