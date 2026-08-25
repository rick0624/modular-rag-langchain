"""框架層級的例外定義。

訊息守則(承襲舊 repo):指出收到什麼、期望什麼、該調整哪個 config
欄位,並在可能時列出可用的替代選項。
"""

from __future__ import annotations

from collections.abc import Sequence


class RagError(Exception):
    """框架內所有自訂例外的共同基底。"""


class ConfigError(RagError):
    """config 載入或驗證失敗(欄位缺漏、型別錯誤、參數不合法等)。"""


class UnknownMethodError(ConfigError):
    """config 指定的 method 不在該槽位的註冊表中;訊息列出可用方法。"""

    def __init__(self, slot: str, method: str, available: Sequence[str]) -> None:
        self.slot = slot
        self.method = method
        self.available = sorted(available)
        listed = ", ".join(repr(name) for name in self.available) or "(目前沒有任何可用的方法)"
        super().__init__(
            f"模組 '{slot}' 沒有名為 '{method}' 的方法。可用的方法:{listed}"
        )


class ComponentError(RagError):
    """模組在執行期間發生的錯誤(檔案不存在、資料格式不符等)。"""


class APICallError(ComponentError):
    """呼叫外部 API 失敗(timeout、連線錯誤、非 2xx 狀態碼)。"""


class APIResponseFormatError(APICallError):
    """外部 API 回傳格式不符預期(非 JSON、缺少欄位、型別不對)。"""
