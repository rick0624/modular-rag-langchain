"""log 設定:console + 選配檔案(rotating)。

分工:console 用指定等級(保持乾淨),檔案**固定收 DEBUG 全量** ——
出問題時細節已經在檔案裡,不必調高等級重跑一次。
"""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def default_log_file(log_dir: str | Path = "logs") -> Path:
    """預設 log 檔路徑:每次執行一個帶 timestamp 的檔名。

    例如 ``logs/rag-20260826-091530.log`` —— 不同次執行各自一個檔案,
    不會覆寫或混寫;舊檔不自動清理,占空間時自行刪 ``logs/``。
    """
    return Path(log_dir) / f"rag-{datetime.now():%Y%m%d-%H%M%S}.log"

# 檔案收 DEBUG 時,壓低吵雜的第三方套件(它們的 DEBUG 是連線細節)。
_NOISY_LIBS = ("httpx", "httpcore", "urllib3", "elastic_transport", "openai")


def setup_logging(
    console_level: str = "info",
    log_file: str | Path | None = None,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """設定全域 logging(重複呼叫會重設 handler,不會疊加)。

    Args:
        console_level: terminal 輸出等級(debug / info / warning / error)。
        log_file: log 檔路徑;None = 不寫檔。父目錄不存在時自動建立;
            單檔超過 ``max_bytes`` 自動輪替,保留 ``backup_count`` 份。
    """
    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level.upper()))
    console.setFormatter(formatter)
    handlers: list[logging.Handler] = [console]

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # root 開到兩個 handler 需要的最低等級,由各 handler 自行過濾。
    root_level = logging.DEBUG if log_file is not None else console.level
    logging.basicConfig(level=root_level, handlers=handlers, force=True)

    if log_file is not None:
        for name in _NOISY_LIBS:
            logging.getLogger(name).setLevel(logging.INFO)
