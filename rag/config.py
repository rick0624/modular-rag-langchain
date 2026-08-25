"""YAML 配置的 schema 定義與載入(含 ``${ENV_VAR}`` 展開與 .env 載入)。

整條 pipeline 由單一 YAML config 控制:每個槽位指定 ``method`` 與該
方法專屬的 ``params``。這裡只驗證「結構層」(每個槽位都要有 method /
params);``params`` 內容的驗證交給各方法的 builder(:mod:`rag.slots`),
因此新增方法時完全不需改動本檔案。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rag.errors import ConfigError

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ESCAPE_SENTINEL = "\x00ESCAPED_DOLLAR_BRACE\x00"


def _expand_env_in_str(text: str, source: str) -> str:
    """展開單一字串中的 ``${ENV_VAR}`` 佔位符。"""
    escaped = text.replace("$${", _ESCAPE_SENTINEL)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = os.environ.get(name)
        if value is None:
            raise ConfigError(
                f"配置 '{source}' 引用了未設定的環境變數 '{name}'"
                f"(於字串 {text!r})。請先設定該環境變數;"
                "若要輸出字面值 ${...},請寫成 $${...}"
            )
        return value

    return _ENV_VAR_PATTERN.sub(_replace, escaped).replace(_ESCAPE_SENTINEL, "${")


def expand_env_vars(value: Any, source: str = "<dict>") -> Any:
    """遞迴展開設定值中所有字串的 ``${ENV_VAR}`` 佔位符。

    機密(API key 等)因此不必寫死在 YAML 中,改由環境變數注入,
    設定檔可以安心進版控。``$${...}`` 會輸出字面值 ``${...}``。

    Raises:
        ConfigError: 引用的環境變數未設定。
    """
    if isinstance(value, str):
        return _expand_env_in_str(value, source)
    if isinstance(value, dict):
        return {key: expand_env_vars(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item, source) for item in value]
    return value


class MethodConfig(BaseModel):
    """單一槽位的配置:方法名稱 + 參數。

    ``method`` 也可以寫成**清單**(方法鏈):輸入輸出同型別的槽位
    (query_transformation、reranking)會依序執行清單中的方法。
    其他槽位不支援清單,建構期會報錯。

    參數有兩種寫法,可以並用:

    1. ``params``:扁平寫法,直接放「當前 method」的參數。
    2. ``method_params``:以方法名稱分區;切換 ``method`` 時只取用對應
       區塊,其餘區塊不驗證、不干擾 —— 這讓 default.yaml 可以當「方法
       型錄」用。

    有效參數的決定規則(:meth:`params_for`):``method_params`` 中存在
    當前方法的區塊時,採用該區塊並忽略 ``params``;否則使用 ``params``。
    方法鏈(清單長度 > 1)時 ``params`` 有「屬於哪個方法」的歧義,
    必須為空,一律改用 ``method_params`` 分區。
    """

    model_config = ConfigDict(extra="forbid")

    method: str | list[str] = Field(
        description="方法名稱;同型槽位可寫清單(方法鏈,依序執行)"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="當前 method 的參數(扁平寫法;方法鏈時必須為空)",
    )
    method_params: dict[str, dict[str, Any] | None] = Field(
        default_factory=dict,
        description="各方法專屬參數,以方法名稱分區;切換 method 時互不干擾",
    )

    @field_validator("method_params", mode="after")
    @classmethod
    def _null_block_is_empty(
        cls, value: dict[str, dict[str, Any] | None]
    ) -> dict[str, dict[str, Any]]:
        # 參數全被註解掉的區塊在 YAML 中是 null 而非 {},不該因此驗證失敗。
        return {key: (block or {}) for key, block in value.items()}

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list):
            if not value:
                raise ValueError("method 清單不可為空")
            if not all(isinstance(item, str) and item for item in value):
                raise ValueError("method 清單的元素必須是非空字串")
        return value

    @model_validator(mode="after")
    def _reject_ambiguous_params(self) -> "MethodConfig":
        if isinstance(self.method, list) and len(self.method) > 1 and self.params:
            raise ValueError(
                "method 為方法鏈(多個方法)時,params 無法判斷屬於哪個方法;"
                "請改用 method_params 以方法名稱分區"
            )
        return self

    def methods(self) -> list[str]:
        """回傳方法名稱清單(單一方法回傳長度 1 的清單)。"""
        return list(self.method) if isinstance(self.method, list) else [self.method]

    def params_for(self, method: str | None = None) -> dict[str, Any]:
        """回傳指定方法(預設為第一個 ``method``)的有效參數。"""
        target = method or self.methods()[0]
        if target in self.method_params:
            return dict(self.method_params[target])
        return dict(self.params)


class PromptConfig(BaseModel):
    """答案生成的 prompt 設定(選填;省略時使用內建 zh-TW 模板)。"""

    model_config = ConfigDict(extra="forbid")

    template: str | None = Field(
        default=None,
        description="prompt 模板,需含 {query} 與 {context} 佔位符;None = 內建模板",
    )
    system: str | None = Field(default=None, description="system 訊息;None = 不帶")


class IngestionConfig(BaseModel):
    """Ingestion 階段(import → parsing → chunking → embedding → indexing)的配置。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    import_: MethodConfig = Field(alias="import", description="槽位:Import")
    parsing: MethodConfig = Field(description="槽位:Parsing")
    chunking: MethodConfig = Field(description="槽位:Chunking")
    embedding: MethodConfig = Field(
        description="槽位:Embedding(查詢端向量同源派生,保證同向量空間)"
    )
    indexing: MethodConfig = Field(description="槽位:Indexing(vector store)")


class InferenceConfig(BaseModel):
    """Inference 階段(query_transformation → retrieval → reranking →
    fusion → generation)的配置。"""

    model_config = ConfigDict(extra="forbid")

    query_transformation: MethodConfig = Field(
        description="槽位:Query Transformation(可方法鏈)"
    )
    retrieval: MethodConfig = Field(description="槽位:Retrieval")
    reranking: MethodConfig = Field(description="槽位:Reranking(可方法鏈)")
    generation: MethodConfig = Field(description="槽位:Generation")
    fusion: MethodConfig | None = Field(
        default=None,
        description="槽位:Fusion(選填;省略時使用內建 merge 預設參數)",
    )
    routing: MethodConfig | None = Field(
        default=None,
        description="槽位:Routing(選填支線;判斷查詢類別,不影響檢索,"
        "結果附加於輸出的 routing key)",
    )
    formatter: MethodConfig | None = Field(
        default=None,
        description="槽位:Formatter(選填終端支線;把最終結果組成對外格式,"
        "結果進輸出的 output key;省略時 output 為 None)",
    )
    prompt: PromptConfig | None = Field(
        default=None, description="答案生成的 prompt 模板設定(選填)"
    )


class RAGConfig(BaseModel):
    """整條 pipeline 的頂層配置(ingestion 與 inference 皆必填)。"""

    model_config = ConfigDict(extra="forbid")

    ingestion: IngestionConfig = Field(description="Ingestion 槽位配置")
    inference: InferenceConfig = Field(description="Inference 槽位配置")
    evaluation: MethodConfig | None = Field(
        default=None, description="Evaluation 配置(可省略)"
    )


def _format_validation_error(exc: ValidationError, source: str) -> str:
    """把 pydantic 的 ValidationError 轉成人類可讀的多行訊息。"""
    lines = [f"配置 '{source}' 驗證失敗,共 {exc.error_count()} 個錯誤:"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        raw_input = repr(err.get("input"))
        if len(raw_input) > 80:
            raw_input = raw_input[:77] + "..."
        lines.append(f"  - {loc}: {err['msg']}(輸入值:{raw_input})")
    return "\n".join(lines)


def parse_config(data: Any, source: str = "<dict>") -> RAGConfig:
    """把已解析的 dict 驗證成 :class:`RAGConfig`。

    驗證前會先以 :func:`expand_env_vars` 展開字串值中的 ``${ENV_VAR}``。

    Raises:
        ConfigError: 結構不是 mapping、環境變數未設定,或欄位缺漏 / 型別錯誤。
    """
    if not isinstance(data, dict):
        raise ConfigError(
            f"配置 '{source}' 的頂層必須是 mapping(YAML 物件),"
            f"實際得到:{type(data).__name__}"
        )
    data = expand_env_vars(data, source)
    try:
        return RAGConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source)) from exc


def load_config(
    path: str | Path, dotenv_path: str | Path | None = ".env"
) -> RAGConfig:
    """從 YAML 檔載入並驗證整條 pipeline 的配置。

    載入前會先讀取 ``dotenv_path`` 指定的 .env 檔(預設為目前工作目錄下
    的 ``.env``,不存在時安靜跳過;已存在的環境變數不會被覆蓋),再展開
    config 中的 ``${ENV_VAR}``。

    Raises:
        ConfigError: 檔案不存在、YAML 語法錯誤,或 schema 驗證失敗。
    """
    if dotenv_path is not None:
        load_dotenv(dotenv_path)
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"找不到設定檔:{config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"設定檔 '{config_path}' 不是合法的 YAML:{exc}") from exc
    return parse_config(data, source=str(config_path))
