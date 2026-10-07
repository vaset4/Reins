"""【文件恢复】【写入准入】Windows 进程持有的物理目录共享与独占窗口。

作者：xxx
时间：2026-09-30 20:00:00
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from ctypes import wintypes
from pathlib import Path

from tools.file_persistence import FileEditConflict
from tools.file_resources import file_identity

_GENERIC_READ_WRITE = 0xC0000000
_SHARE_ALL = 7
_OPEN_ALWAYS = 4
_NORMAL_ATTRIBUTE = 0x80
_LOCK_FAIL_IMMEDIATELY = 1
_LOCK_EXCLUSIVE = 2
_ERROR_LOCK_VIOLATION = 33
_WAIT_SECONDS = 30
_POLL_SECONDS = 0.05
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)


class _Overlapped(ctypes.Structure):
    """同步字节范围锁需要的 Windows 偏移结构。"""

    _fields_ = [
        ("internal", ctypes.c_size_t),
        ("internal_high", ctypes.c_size_t),
        ("offset", wintypes.DWORD),
        ("offset_high", wintypes.DWORD),
        ("event", wintypes.HANDLE),
    ]


_KERNEL.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_KERNEL.CreateFileW.restype = wintypes.HANDLE
_KERNEL.LockFileEx.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(_Overlapped),
]
_KERNEL.LockFileEx.restype = wintypes.BOOL
_KERNEL.UnlockFileEx.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(_Overlapped),
]
_KERNEL.UnlockFileEx.restype = wintypes.BOOL
_KERNEL.CloseHandle.argtypes = [wintypes.HANDLE]
_KERNEL.CloseHandle.restype = wintypes.BOOL


class WorkspaceBusyError(FileEditConflict):
    """重叠物理范围仍有活跃写者；路径说明被占用的真实范围。"""

    def __init__(self, path: Path, owner: str) -> None:
        """建立明确占用回执；参数：冲突资源和申请者；返回：异常。"""
        self.path, self.requester = path, owner
        super().__init__(
            f"workspace write scope is busy: {path}; occupying process is not identified; request was not executed"
        )


@contextmanager
def workspace_write_window(
    path: Path, *, subtree: bool = False, wait: bool = False, owner: str = ""
) -> Iterator[None]:
    """准入真实文件写入；参数：物理目标、是否目录范围、等待和操作身份；返回：系统持有窗口。

    祖先共享、目标独占使父子覆盖互斥且不同文件可并行；不持有数据空间提交锁
    """
    with workspace_write_windows(((path, subtree),), wait=wait, owner=owner):
        yield


@contextmanager
def workspace_write_windows(
    scopes: Iterable[tuple[Path, bool]], *, wait: bool = False, owner: str = ""
) -> Iterator[None]:
    """统一取得一次执行涉及的物理范围；参数：路径/子树标记、等待、身份；返回：去重后的持锁窗口。"""
    directory = Path(os.environ["LOCALAPPDATA"]) / "Reins" / "workspace-locks"
    directory.mkdir(parents=True, exist_ok=True)
    resources: dict[str, tuple[Path, bool]] = {}
    for path, subtree in scopes:
        resolved = Path(path).resolve()
        for resource, exclusive in [
            *((ancestor, False) for ancestor in resolved.parents),
            (resolved, True),
        ]:
            key = "path:" + os.path.normcase(str(resource))
            resources[key] = (
                resource,
                exclusive or resources.get(key, (resource, False))[1],
            )
        identity = None if subtree else file_identity(resolved)
        if identity is not None:
            resources["file:" + identity] = (resolved, True)
    # 1. 【文件恢复】【跨目录准入】祖先锁合并为最强模式，避免重叠执行范围与自己冲突
    with ExitStack() as locks:
        for key, (resource, exclusive) in sorted(resources.items()):
            locks.enter_context(
                _range_lock(
                    directory,
                    key,
                    resource,
                    exclusive=exclusive,
                    wait=wait,
                    owner=owner,
                )
            )
        yield


@contextmanager
def _range_lock(
    directory: Path, key: str, path: Path, *, exclusive: bool, wait: bool, owner: str
) -> Iterator[None]:
    """取得一个共享或独占字节锁；参数：全用户固定目录、资源和模式；返回：退出或进程结束释放。"""
    lock_path = directory / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".lock")
    handle = _KERNEL.CreateFileW(
        str(lock_path),
        _GENERIC_READ_WRITE,
        _SHARE_ALL,
        None,
        _OPEN_ALWAYS,
        _NORMAL_ATTRIBUTE,
        None,
    )
    if handle == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    overlap = _Overlapped()
    flags = _LOCK_FAIL_IMMEDIATELY | (_LOCK_EXCLUSIVE if exclusive else 0)
    deadline = time.monotonic() + _WAIT_SECONDS
    try:
        while not _KERNEL.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(overlap)):
            error = ctypes.get_last_error()
            if error != _ERROR_LOCK_VIOLATION:
                raise ctypes.WinError(error)
            if not wait or time.monotonic() >= deadline:
                raise WorkspaceBusyError(path, owner)
            time.sleep(_POLL_SECONDS)
        try:
            yield
        finally:
            if not _KERNEL.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(overlap)):
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _KERNEL.CloseHandle(handle)
