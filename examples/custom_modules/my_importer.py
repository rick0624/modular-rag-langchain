"""自訂 import 範例:來源不是本地檔案,而是程式內(或外部系統)的文字。

契約:import_() -> list[Source]
(每筆必帶穩定的 doc_id;本地檔給 path、文字內容直接給 text)

掛法::

    import:
      method: custom
      params:
        file: ./examples/custom_modules/my_importer.py
"""

from rag.interfaces import Source


def build(params, ctx):
    def import_() -> list[Source]:
        # TODO(替換點):改成呼叫你的文件管理系統 / API,每份文件一筆 Source
        # (外部欄位放進 meta,之後會複製進切片 metadata)。
        return [
            Source(
                doc_id="notice-2026-001",
                text="十月一日起,辦公室門禁改用新版員工證,舊卡月底停用。",
                meta={"department": "總務部"},
            ),
            Source(
                doc_id="notice-2026-002",
                text="年度健康檢查於十一月舉行,請於月底前完成線上預約。",
                meta={"department": "人資部"},
            ),
        ]

    return import_
