"""方法註冊表與槽位建構:``method`` 名稱 → builder 函式。

替換一個模組的實作 = 換 config 的 ``method`` 那一行。每個方法都是一個
**builder 函式** ``build(params, ctx) -> 產物``:

- 內建方法以 :func:`register` 裝飾器登記在 :data:`REGISTRY`;
- ``method: custom`` 從 ``params.file`` 指定的 .py 檔載入同慣例的
  builder 函式(函式名由 ``params.function`` 指定,預設 ``build``),
  零框架改動即可掛入自訂實作。

產物在建構期驗證:function 槽位比對函式參數名
(:data:`rag.interfaces.EXPECTED_PARAMS`);embedding / indexing 槽位
duck-type 檢查必要方法。不符直接報錯並指明期望的簽名。
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from rag.config import MethodConfig, RAGConfig
from rag.errors import ConfigError, UnknownMethodError
from rag.interfaces import CHAINABLE_SLOTS, EXPECTED_PARAMS


@dataclass
class BuildContext:
    """建構期共享的執行環境(跨槽位依賴由此傳遞)。

    - ``embeddings``:embedding 槽位的產物;indexing 建 store 時消費它,
      查詢端經 store 使用同一物件 embed → 同向量空間是結構性保證。
    - ``store``:indexing 槽位的產物;retrieval 由此檢索。
    """

    config: RAGConfig
    embeddings: Any = None
    store: Any = None


Builder = Callable[[dict[str, Any], BuildContext], Any]

# 註冊表:REGISTRY[槽位][方法名稱] -> builder。
REGISTRY: dict[str, dict[str, Builder]] = {}


def register(slot: str, name: str) -> Callable[[Builder], Builder]:
    """把 builder 函式登記為 ``slot`` 槽位的 ``name`` 方法。"""

    def decorator(builder: Builder) -> Builder:
        REGISTRY.setdefault(slot, {})[name] = builder
        return builder

    return decorator


class BaseParams(BaseModel):
    """所有方法參數 schema 的基底:多打欄位直接報錯,不靜默忽略。"""

    model_config = ConfigDict(extra="forbid")


def validate_params(
    slot: str, method: str, params_cls: type[BaseParams], raw: dict[str, Any]
) -> Any:
    """以方法的 Params schema 驗證參數,錯誤時列出可接受的參數。"""
    try:
        return params_cls(**raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        accepted = ", ".join(sorted(params_cls.model_fields.keys())) or "(不接受任何參數)"
        raise ConfigError(
            f"模組 '{slot}' 方法 '{method}' 的參數不合法:{details}。"
            f"可接受的參數:{accepted}"
        ) from exc


def build_slot(slot: str, cfg: MethodConfig, ctx: BuildContext) -> Any:
    """依 config 建立一個槽位的產物(方法鏈時依序組合)。

    Raises:
        ConfigError: 不支援鏈的槽位收到方法清單、custom 載入失敗,
            或產物不符合槽位契約。
        UnknownMethodError: 指定的方法未註冊(訊息列出可用方法)。
    """
    methods = cfg.methods()
    if len(methods) > 1 and slot not in CHAINABLE_SLOTS:
        raise ConfigError(
            f"模組 '{slot}' 的輸入輸出型別不同,不支援方法鏈;"
            f"收到 method 清單 {methods},請指定單一方法"
        )
    products = [
        _build_one(slot, method, cfg.params_for(method), ctx) for method in methods
    ]
    if len(products) == 1:
        return products[0]
    return _compose_chain(slot, products)


def _build_one(slot: str, method: str, params: dict[str, Any], ctx: BuildContext) -> Any:
    if method == "custom":
        product = _build_custom(slot, params, ctx)
    else:
        table = REGISTRY.get(slot, {})
        if method not in table:
            raise UnknownMethodError(slot, method, [*table, "custom"])
        product = table[method](params, ctx)
    _validate_product(slot, method, product)
    return product


def _compose_chain(slot: str, products: list[Any]) -> Any:
    """把方法鏈的多個產物組成一個同簽名的函式(依序執行)。"""
    if slot == "query_transformation":

        def transform(queries: list[str]) -> list[str]:
            for fn in products:
                queries = fn(queries)
            return queries

        return transform

    def rerank(query: str, documents: list) -> list:
        for fn in products:
            documents = fn(query, documents)
        return documents

    return rerank


# --- custom module 載入 -----------------------------------------------------


class _CustomParams(BaseParams):
    """``method: custom`` 的框架層參數;其餘鍵透傳給自訂 builder。"""

    model_config = ConfigDict(extra="allow")

    file: str
    function: str = "build"


def _build_custom(slot: str, params: dict[str, Any], ctx: BuildContext) -> Any:
    """載入自訂 .py 檔中的 builder 函式並呼叫。

    慣例:``file`` 指定 .py 檔路徑(相對執行目錄),``function`` 指定
    頂層 builder 函式名(預設 ``build``,簽名同內建方法
    ``build(params, ctx)``);``file`` / ``function`` 以外的參數原樣
    透傳給 builder。
    """
    p = validate_params(slot, "custom", _CustomParams, params)
    passthrough = {
        key: value for key, value in params.items() if key not in ("file", "function")
    }
    path = Path(p.file)
    if not path.is_file():
        raise ConfigError(
            f"模組 '{slot}' 的 custom 方法找不到檔案:{path}"
            "(file 路徑相對於執行目錄)"
        )
    module = _exec_module(path)
    builder = getattr(module, p.function, None)
    if builder is None:
        available = sorted(
            name
            for name, value in vars(module).items()
            if callable(value) and not name.startswith("_")
        )
        raise ConfigError(
            f"自訂模組 '{path}' 沒有名為 '{p.function}' 的函式;"
            f"檔案內可用的函式:{available or '(無)'}。"
            "請以 function 參數指定 builder 函式名(預設 build)"
        )
    if not callable(builder):
        raise ConfigError(
            f"自訂模組 '{path}' 的 '{p.function}' 不是函式"
            f"(實際型別:{type(builder).__name__})"
        )
    try:
        inspect.signature(builder).bind("params", "ctx")
    except TypeError as exc:
        raise ConfigError(
            f"自訂模組 '{path}' 的 builder '{p.function}' 必須接受兩個參數 "
            f"(params, ctx),實際簽名:{inspect.signature(builder)}"
        ) from exc
    return builder(passthrough, ctx)


def _exec_module(path: Path) -> Any:
    """以檔案路徑載入模組;模組名帶路徑雜湊,同名檔案不互撞。"""
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"_rag_custom_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"無法載入自訂模組:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"自訂模組 '{path}' 載入失敗:{exc}") from exc
    return module


# --- 產物驗證 ---------------------------------------------------------------

_DUCK_TYPE_SLOTS: dict[str, tuple[str, ...]] = {
    # 兩個「產物是物件不是函式」的槽位:duck-type 檢查必要方法。
    "embedding": ("embed_documents", "embed_query"),
    "indexing": ("add_documents", "similarity_search_with_score"),
}


def _validate_product(slot: str, method: str, product: Any) -> None:
    if slot in _DUCK_TYPE_SLOTS:
        missing = [
            attr for attr in _DUCK_TYPE_SLOTS[slot] if not callable(getattr(product, attr, None))
        ]
        if missing:
            expected = " / ".join(_DUCK_TYPE_SLOTS[slot])
            kind = "Embeddings" if slot == "embedding" else "VectorStore"
            raise ConfigError(
                f"模組 '{slot}' 方法 '{method}' 的產物必須是 LangChain "
                f"{kind} 介面的物件(需要方法:{expected}),"
                f"缺少:{', '.join(missing)}"
            )
        return
    expected = EXPECTED_PARAMS[slot]
    if not callable(product):
        raise ConfigError(
            f"模組 '{slot}' 方法 '{method}' 的 builder 必須回傳函式,"
            f"實際回傳:{type(product).__name__}"
        )
    actual = tuple(inspect.signature(product).parameters)
    if actual != expected:
        expected_sig = ", ".join(expected) or "(無參數)"
        actual_sig = ", ".join(actual) or "(無參數)"
        raise ConfigError(
            f"模組 '{slot}' 方法 '{method}' 回傳的函式參數應為 "
            f"({expected_sig}),實際為 ({actual_sig});"
            "請調整函式簽名以符合槽位契約(見 docs/interfaces.md)"
        )
