"""槽位:Import —— 列出來源文件。契約:() → list[Source]。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from rag.errors import ComponentError
from rag.interfaces import ImportFn, Source
from rag.registry import BaseParams, BuildContext, register, validate_params


class _LocalFilesParams(BaseParams):
    input_dir: str = Field(description="要匯入的資料夾")
    extensions: list[str] = Field(
        default=[".txt", ".md"], description="收錄的副檔名(小寫比對)"
    )
    recursive: bool = Field(default=True, description="是否含子資料夾")


@register("import", "local_files")
def build_local_files(params: dict[str, Any], ctx: BuildContext) -> ImportFn:
    """本地資料夾 importer:doc_id = 檔案相對 input_dir 的路徑(跨執行穩定)。"""
    p = validate_params("import", "local_files", _LocalFilesParams, params)
    extensions = {ext.lower() for ext in p.extensions}

    def import_() -> list[Source]:
        base = Path(p.input_dir)
        if not base.is_dir():
            raise ComponentError(
                f"找不到匯入資料夾:{base}(import 方法 'local_files' 的 input_dir)"
            )
        walker = base.rglob("*") if p.recursive else base.glob("*")
        return [
            Source(doc_id=path.relative_to(base).as_posix(), path=path)
            for path in sorted(walker)
            if path.is_file() and path.suffix.lower() in extensions
        ]

    return import_
