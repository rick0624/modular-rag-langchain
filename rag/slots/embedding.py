"""槽位:Embedding —— 文字 → 向量。

本槽位是「產物是物件不是函式」的兩個例外之一:產物為 LangChain
``Embeddings`` 物件,由 indexing 槽位的 vector store 持有 —— 建索引與
查詢共用同一物件,同向量空間是結構性保證(換方法或模型後必須重建索引)。
"""

from __future__ import annotations

import zlib
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import Field

from rag.errors import APIResponseFormatError
from rag.registry import BaseParams, BuildContext, register, validate_params
from rag.slots.api_utils import locate_list, post_json


class HashedNgramEmbeddings(Embeddings):
    """離線確定性詞袋向量:字元 bigram 雜湊 + L2 正規化(CJK 友善)。

    不是語意向量 —— 但共用字詞的文字餘弦相似度高,離線 demo 與測試的
    檢索結果因此「看起來合理」(相對於隨機假向量檢索等同亂數)。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        import numpy as np

        vector = np.zeros(self.dim, dtype=np.float64)
        normalized = text.lower()
        grams = (
            [normalized[i : i + 2] for i in range(len(normalized) - 1)]
            if len(normalized) > 1
            else [normalized]
        )
        for gram in grams:
            # zlib.crc32 跨行程確定性(內建 hash() 有隨機鹽,不可用)
            vector[zlib.crc32(gram.encode("utf-8")) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class _CommonEmbeddingParams(BaseParams):
    """所有 embedding 方法共通的參數(由框架的 ingest 流程消費)。"""

    source_field: str | None = Field(
        default=None,
        description="主向量的來源欄位:None = 切片內文(page_content);"
        "指定 chunking 生成的 metadata 欄位名時,拿該欄位做 embedding、"
        "prompt 仍給切片內文(解耦)。查詢端不受影響(同一模型)",
    )
    extra_vectors: dict[str, str] | None = Field(
        default=None,
        description="額外向量 {向量欄位名: 來源欄位}:用同一個模型對額外"
        "欄位各出一組向量,寫進切片 metadata 隨之落入索引;內建檢索只用"
        "主向量,額外向量供 custom retrieval 使用。欄位名不可用框架保留名",
    )


class _MockParams(_CommonEmbeddingParams):
    dim: int = Field(default=256, gt=0, description="向量維度")


@register("embedding", "mock")
def build_mock(params: dict[str, Any], ctx: BuildContext) -> Embeddings:
    """離線確定性偽向量(開發測試用,不需網路與金鑰)。"""
    p = validate_params("embedding", "mock", _MockParams, params)
    return HashedNgramEmbeddings(dim=p.dim)


class _ApiParams(_CommonEmbeddingParams):
    endpoint: str = Field(description="embedding API 的完整 URL")
    headers: dict[str, str] = Field(default_factory=dict, description="額外 HTTP 標頭(認證放這裡)")
    model: str | None = Field(default=None, description="None = 請求不帶 model 欄位")
    batch_size: int = Field(default=16, gt=0, description="每批文字數")
    timeout: float = Field(default=30.0, gt=0, description="逾時秒數")
    texts_field: str = Field(default="input", description="請求中放文字清單的欄位名")
    model_field: str = Field(default="model", description="請求中放模型的欄位名")
    embeddings_field: str | None = Field(
        default="data",
        description="回應中向量清單的欄位(a.b 巢狀路徑;回應本身是清單時設 null)",
    )
    item_field: str | None = Field(
        default="embedding",
        description="清單元素是物件時,向量所在欄位;元素直接是向量時設 null",
    )


class ApiEmbeddings(Embeddings):
    """通用 HTTP embedding API:回應形狀以參數對映(預設 OpenAI 式)。"""

    def __init__(self, p: _ApiParams) -> None:
        self.p = p

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body: dict[str, Any] = {self.p.texts_field: texts}
        if self.p.model is not None:
            body[self.p.model_field] = self.p.model
        data = post_json(
            self.p.endpoint, body, headers=self.p.headers, timeout=self.p.timeout
        )
        items = locate_list(
            data,
            self.p.embeddings_field,
            api_label="embedding API ",
            setting_name="embeddings_field",
            what="向量清單",
        )
        if len(items) != len(texts):
            raise APIResponseFormatError(
                f"embedding API 回傳的向量數({len(items)})與輸入文字數({len(texts)})不符"
            )
        return [self._to_vector(item, position) for position, item in enumerate(items)]

    def _to_vector(self, item: Any, position: int) -> list[float]:
        value: Any = item
        if isinstance(item, dict):
            if self.p.item_field is None:
                raise APIResponseFormatError(
                    f"向量清單的第 {position} 個元素是物件(欄位:{sorted(item.keys())}),"
                    "請設定 item_field 指出向量所在的欄位(OpenAI 式回應設為 embedding)"
                )
            if self.p.item_field not in item:
                raise APIResponseFormatError(
                    f"向量清單的第 {position} 個元素缺少 '{self.p.item_field}' 欄位;"
                    f"實際的欄位:{sorted(item.keys())}"
                )
            value = item[self.p.item_field]
        if not isinstance(value, list):
            raise APIResponseFormatError(
                f"第 {position} 個向量必須是數字 list,實際得到:{type(value).__name__}"
            )
        try:
            return [float(element) for element in value]
        except (TypeError, ValueError) as exc:
            raise APIResponseFormatError(f"第 {position} 個向量含有非數字元素") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.p.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.p.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


@register("embedding", "api")
def build_api(params: dict[str, Any], ctx: BuildContext) -> Embeddings:
    """公司 / 第三方 HTTP embedding API(形狀對映見各參數說明)。"""
    p = validate_params("embedding", "api", _ApiParams, params)
    return ApiEmbeddings(p)
