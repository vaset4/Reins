"""Windows 文件修改的资源互斥、版本核对与完整发布。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
import msvcrt
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from ctypes import wintypes
from pathlib import Path
from tempfile import NamedTemporaryFile
from tools.file_resources import file_identity

_WAIT_ACQUIRED = 0
_WAIT_ABANDONED = 0x80
_WAIT_TIMEOUT = 0x102
_LOCK_WAIT_SECONDS = 30
_LOCK_POLL_MILLISECONDS = 250
_MOVE_WRITE_THROUGH = 0x8
_ERROR_ALREADY_EXISTS = 183
_ERROR_FILE_EXISTS = 80
_KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)
_KERNEL.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
_KERNEL.CreateMutexW.restype = wintypes.HANDLE
_KERNEL.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_KERNEL.WaitForSingleObject.restype = wintypes.DWORD
_KERNEL.ReleaseMutex.argtypes = [wintypes.HANDLE]
_KERNEL.ReleaseMutex.restype = wintypes.BOOL
_KERNEL.CloseHandle.argtypes = [wintypes.HANDLE]
_KERNEL.CloseHandle.restype = wintypes.BOOL
_KERNEL.ReplaceFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.LPVOID,
]
_KERNEL.ReplaceFileW.restype = wintypes.BOOL
_KERNEL.MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
_KERNEL.MoveFileExW.restype = wintypes.BOOL
_KERNEL.GetFileInformationByHandleEx.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
]
_KERNEL.GetFileInformationByHandleEx.restype = wintypes.BOOL


class _FileBasicInfo(ctypes.Structure):
    """Windows文件实际变更时间，区别于可恢复的mtime和创建时间。"""

    _fields_ = [
        ("creation", ctypes.c_longlong),
        ("access", ctypes.c_longlong),
        ("write", ctypes.c_longlong),
        ("change", ctypes.c_longlong),
        ("attributes", wintypes.DWORD),
    ]


def file_change_time(descriptor: int) -> int:
    """查询实际文件变更代次；参数：已打开Python描述符；返回：Windows ChangeTime。"""
    info = _FileBasicInfo()
    handle = msvcrt.get_osfhandle(descriptor)
    if not _KERNEL.GetFileInformationByHandleEx(
        handle, 0, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(info.change)


def file_revision(path: Path) -> tuple[int, int]:
    """读取原件长度与真实变更时间；参数：文件路径；返回：恢复mtime仍会变化的版本。"""
    with path.open("rb") as handle:
        return os.fstat(handle.fileno()).st_size, file_change_time(handle.fileno())


class FileEditConflict(ValueError):
    """内容已变或正在被另一写者修改，需要基于当前文件重新决定。"""

    def __init__(self, message: str, *, backup_path: Path | None = None) -> None:
        """保存冲突和受保护的旧字节位置；参数：原因及备份；返回：可定位异常。"""
        self.backup_path = backup_path
        super().__init__(message)


def content_sha256(content: bytes) -> str:
    """计算原始字节版本；传参：文件内容；返回：SHA256十六进制摘要。"""
    return hashlib.sha256(content).hexdigest()


@contextmanager
def file_edit_lock(path: Path, *, wait: bool = False) -> Iterator[None]:
    """按规范化文件路径串行化本机Reins写者，进程退出后由系统释放锁。

    传参：path 为实际文件；wait 仅供短时事实追加等待前一写者；返回：独占修改窗口
    """
    identities = {"path:" + os.path.normcase(str(path.resolve()))}
    actual = file_identity(path)
    if actual is not None:
        identities.add("file:" + actual)
    with ExitStack() as locks:
        for identity in sorted(identities):
            locks.enter_context(_resource_mutex(identity, wait=wait))
        yield


@contextmanager
def _resource_mutex(resource: str, *, wait: bool) -> Iterator[None]:
    """锁定路径或卷内文件编号，进程结束自动释放；传参：资源名与等待策略；返回：独占窗口。"""
    identity = content_sha256(resource.encode("utf-8"))
    handle = _KERNEL.CreateMutexW(None, False, f"Local\\ReinsFileEdit-{identity}")
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        status = _KERNEL.WaitForSingleObject(
            handle, _LOCK_POLL_MILLISECONDS if wait else 0
        )
        while wait and status == _WAIT_TIMEOUT and time.monotonic() < deadline:
            status = _KERNEL.WaitForSingleObject(handle, _LOCK_POLL_MILLISECONDS)
        if status == _WAIT_TIMEOUT:
            raise FileEditConflict(
                "file has another active writer; read its current version before retrying"
            )
        if status not in {_WAIT_ACQUIRED, _WAIT_ABANDONED}:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not _KERNEL.ReleaseMutex(handle):
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _KERNEL.CloseHandle(handle)


def read_file_bytes(path: Path) -> bytes | None:
    """读取当前完整字节；传参：文件路径；返回：内容，不存在时为None，其他IO错误直接暴露。"""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def publish_prepared_file(path: Path, temporary: Path) -> None:
    """耐久发布已经同步的完整新文件；参数：目标及同卷暂存文件；返回：无，绝不覆盖已有目标。"""
    if not _KERNEL.MoveFileExW(str(temporary), str(path), _MOVE_WRITE_THROUGH):
        error = ctypes.get_last_error()
        if error in {_ERROR_ALREADY_EXISTS, _ERROR_FILE_EXISTS}:
            raise FileEditConflict("file was created by another writer")
        raise ctypes.WinError(error)


def replace_prepared_file(path: Path, temporary: Path, *, backup_path: Path) -> None:
    """替换同卷完整暂存并保留实际原件；参数：目标、已同步暂存、备份位置；返回：无，调用方核对前后态。"""
    if backup_path.exists():
        raise FileEditConflict(
            "replacement backup path already exists", backup_path=backup_path
        )
    os.chmod(temporary, path.stat().st_mode)
    # 1. 【文件恢复】【发布原件】ReplaceFile 保存当前 ACL 与实际被替换字节，不重新读取大文件
    if not _KERNEL.ReplaceFileW(
        str(path), str(temporary), str(backup_path), 0, None, None
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def publish_file(
    path: Path,
    original: bytes | None,
    updated: bytes,
    *,
    backup_path: Path | None = None,
) -> None:
    """复核读取版本后发布完整文件，不把临时文件当作成功结果。

    传参：path为目标；original为依据；updated为新字节；backup_path保留实际被替换内容
    返回：无，冲突或IO失败抛错；备份冲突不自动回写当前文件
    """
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        # 【文件】【提交修改】受控写者持有同一互斥；外部编辑器的检查后写入竞争不视为已解决
        if read_file_bytes(path) != original:
            raise FileEditConflict(
                "file changed during edit; no replacement was published"
            )
        if original is None:
            # 【文件】【创建文件】仅在目标仍不存在时发布，避免并发新建文件被覆盖
            publish_prepared_file(path, temporary)
        else:
            if backup_path is not None and backup_path.exists():
                raise FileEditConflict(
                    "replacement backup path already exists", backup_path=backup_path
                )
            os.chmod(temporary, path.stat().st_mode)
            # 【文件】【发布内容】ReplaceFile保留原文件的ACL与附加信息，避免替换时继承更宽权限
            backup_name = None if backup_path is None else str(backup_path)
            if not _KERNEL.ReplaceFileW(
                str(path), str(temporary), backup_name, 0, None, None
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if backup_path is not None and backup_path.read_bytes() != original:
                raise FileEditConflict(
                    "file changed at publication; actual replaced content retained in backup",
                    backup_path=backup_path,
                )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
