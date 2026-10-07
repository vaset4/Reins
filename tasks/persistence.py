"""任务元数据的独占写入与原子发布边界。

作者：xxx
时间：2026-09-13 20:00:00
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

TASK_WRITE_LOCK_FILE = ".task_write.lock"
_MAINTENANCE_LOCK_FILE = ".message_owner.lock"


class TaskRevisionConflict(ValueError):
    """目标正在被其他写者修改，调用方需要重新读取当前状态。"""


def validate_task_id(task_id: str) -> None:
    """校验外部目标身份；传参：目标编号；返回：无，非法编号抛错。"""
    if (
        not task_id
        or task_id in {".", "..", "_inbox"}
        or any(char in task_id for char in "/\\:")
    ):
        raise ValueError("task id must be a single storage name")


def task_record_path(data_root: Path, task_id: str) -> Path:
    """返回目标工作材料中的显式导出位置；传参：根与身份；返回：导出路径。"""
    validate_task_id(task_id)
    return data_root / "workspace" / task_id / "task.yaml"


@contextmanager
def task_write_lock(data_root: Path, *, maintenance: bool = False) -> Iterator[None]:
    """串行提交任务元数据，建任务和收件箱转正也使用同一写入边界。

    传参：data_root 为数据根；maintenance 表示调用方已持有维护锁
    返回：独占窗口；锁冲突明确失败，进程异常遗留的锁需停机核查后移除
    """
    from runtime.persistence import RuntimeStore

    if not maintenance and (data_root / _MAINTENANCE_LOCK_FILE).exists():
        raise TaskRevisionConflict("task metadata is locked for maintenance")
    with RuntimeStore(data_root).transaction():
        yield


def write_task_yaml(path: Path, text: str) -> None:
    """原子发布任务记录，IO 失败不留下半份状态或完成依据。

    传参：path/text 为目标与完整 YAML；返回：无，写入错误直接暴露
    """
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
