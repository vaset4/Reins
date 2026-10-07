"""【Reins】【存储版本】文件原件格式与派生索引启动检查。

作者：xxx
时间：2026-09-29 20:00:00
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from runtime.persistence import SPACE_ID_FILE, FORMAT_VERSION, RuntimeStore

SCHEMA_META_FILE = SPACE_ID_FILE
CURRENT_SCHEMA = {"files": FORMAT_VERSION}
SchemaStatus = Literal["initialized", "current"]


@dataclass(frozen=True, slots=True)
class SchemaCheckResult:
    """返回空间身份原件位置与初始化状态。"""

    meta_file: Path
    status: SchemaStatus


class UnsupportedSchemaError(RuntimeError):
    """旧格式、未知格式和缺失原件均需要显式处理。"""

    def __init__(self, code: str, reason: str) -> None:
        """建立稳定启动错误；参数：类别与原因；返回：无。"""
        self.code, self.reason = code, reason
        super().__init__(
            f"SCHEMA_UNSUPPORTED[{code}]: data_root=<runtime-data>; reason={reason}"
        )


def ensure_current_schema(reins_data_dir: Path | str) -> SchemaCheckResult:
    """初始化文件空间并核对原件与索引；参数：数据根；返回：版本状态，不迁移旧数据。"""
    database = RuntimeStore(reins_data_dir)
    marker = database.data_root / SPACE_ID_FILE
    existed = marker.exists()
    try:
        database.ensure_space()
        database.require_current_format()
        database.rebuild_index()
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise UnsupportedSchemaError("DATABASE_UNSUPPORTED", str(exc)) from exc
    return SchemaCheckResult(marker, "current" if existed else "initialized")


__all__ = [
    "CURRENT_SCHEMA",
    "SCHEMA_META_FILE",
    "SchemaCheckResult",
    "UnsupportedSchemaError",
    "ensure_current_schema",
]
