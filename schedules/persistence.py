"""本机持久工作的原子发布和随进程退出释放的认领锁。

作者：xxx
时间：2026-09-14 19:16:10
"""

from __future__ import annotations

import errno
import json
import msvcrt
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from tools.file_persistence import file_edit_lock

LOCK_BYTE_COUNT = 1
SCHEDULE_DISPATCH_LOCK = Path("runtime") / "locks" / "schedule-dispatch.lock"


def record_path(root: Path, identity: str, *, suffix: str = ".json") -> Path:
    """限制外部身份只能定位本存储的一条记录；传参：根目录、身份和扩展名；返回：安全路径。"""
    if (
        not identity
        or identity in {".", ".."}
        or any(char in identity for char in "/\\:")
    ):
        raise ValueError("record identity must be a single storage name")
    path = root / f"{identity}{suffix}"
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("record path escapes its storage root")
    return path


@contextmanager
def claim_file(path: Path, *, blocking: bool = False) -> Iterator[bool]:
    """认领本机工作，短统计更新可等待系统锁；传参：锁文件与等待方式；返回：是否获得独占权。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            msvcrt.locking(handle.fileno(), mode, LOCK_BYTE_COUNT)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, LOCK_BYTE_COUNT)


def atomic_text(path: Path, content: str) -> None:
    """先同步完整临时文件再原子发布；传参：目标路径和正文；返回：无，写入失败直接暴露。"""
    atomic_bytes(path, content.encode("utf-8"))


def atomic_bytes(path: Path, content: bytes) -> None:
    """按原始字节原子发布，适用于不可改换编码的原件；传参：路径与字节；返回：无。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # 【本机持久化】【状态发布】Windows 读句柄未释放时不能替换，轮询与发布共用短时互斥
        with file_edit_lock(path, wait=True):
            temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def immutable_bytes(path: Path, content: bytes) -> None:
    """在调用者持锁时保存不可变原件，重投只接受相同字节；传参：路径和原件；返回：无。"""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(
                f"immutable record already has different content: {path.name}"
            )
        return
    atomic_bytes(path, content)


def write_record(path: Path, payload: Mapping[str, object]) -> None:
    """发布完整JSON记录；传参：路径和结构化内容；返回：无。"""
    atomic_text(path, json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n")


def read_record(path: Path) -> dict[str, object]:
    """读取已存在记录并拒绝损坏结构；传参：路径；返回：JSON对象。"""
    with file_edit_lock(path, wait=True):
        text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"record is not an object: {path.name}")
    return payload
