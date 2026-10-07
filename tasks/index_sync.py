"""【领域查询】【索引重建】从文件原件重建统一的任务、计划和产物投影。

作者：xxx
时间：2026-09-30 16:00:00
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from runtime.persistence import RuntimeStore


def connect_index(data_root: Path | str) -> sqlite3.Connection:
    """打开已同步的派生查询连接；传参：数据根；返回：由调用方关闭的连接。"""
    return RuntimeStore(data_root).open_index_connection()


def sync_tasks_table(data_root: Path | str) -> int:
    """从原件重建目标检索视图；传参：数据根；返回：已保存目标数量。"""
    return rebuild_full_index(data_root)["tasks"]


def rebuild_full_index(data_root: Path | str) -> dict[str, int]:
    """重建统一派生索引，保留文件原件；传参：数据根；返回：各领域的记录数。"""
    store = RuntimeStore(data_root)
    store.rebuild_index(force=True)
    kinds = {"tasks": "task", "schedules": "schedule", "artifacts": "artifact_records"}
    with store.index_connection() as connection:
        return {
            name: int(
                connection.execute(
                    "SELECT COUNT(*) FROM records WHERE kind=?", (kind,)
                ).fetchone()[0]
            )
            for name, kind in kinds.items()
        }
