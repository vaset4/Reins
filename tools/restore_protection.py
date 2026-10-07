"""【文件恢复】【敏感原件】DPAPI 密文与移动前已私有化的 Windows 暂存。

作者：xxx
时间：2026-09-30 20:00:00
"""

from __future__ import annotations

import base64
import ctypes
import importlib
import json
import os
import msvcrt
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from runtime.file_content import ContentFiles, confined_path
from runtime.file_records import (
    ContentReference,
    SourceCorruptionError,
    digest_bytes,
    json_bytes,
)

PROTECTED_MEDIA_TYPE = "application/vnd.reins.dpapi"
_CRYPT_UI_FORBIDDEN = 1
_DACL_SECURITY_INFORMATION = 4
_PROTECTED_DACL_INFORMATION = 0x80000000
_UNPROTECTED_DACL_INFORMATION = 0x20000000
_SE_DACL_PROTECTED = 0x1000
_SECURITY_ACCESS = 0x00020000 | 0x00040000
_DELETE_ACCESS = 0x00010000
_GENERIC_READ = 0x80000000
_FILE_RENAME_INFO_CLASS = 3
_SHARE_ALL = 7
_OPEN_EXISTING = 3
_BACKUP_SEMANTICS = 0x02000000
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)
_CRYPT = ctypes.WinDLL("crypt32", use_last_error=True)
# 1. 【文件恢复】【Windows初始化】先由pywin32引导正确DLL，避免后续通知模块载入旧pywintypes
importlib.import_module("pywintypes")
_SECURITY = importlib.import_module("win32security")


class _DataBlob(ctypes.Structure):
    """DPAPI 的二进制参数，不经字符串或日志中转明文。"""

    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


class _SecurityAttributes(ctypes.Structure):
    """在创建瞬间应用私有 DACL，避免先创建再收紧的窗口。"""

    _fields_ = [
        ("length", wintypes.DWORD),
        ("descriptor", wintypes.LPVOID),
        ("inherit", wintypes.BOOL),
    ]


_CRYPT.CryptProtectData.argtypes = [
    ctypes.POINTER(_DataBlob),
    wintypes.LPCWSTR,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(_DataBlob),
]
_CRYPT.CryptProtectData.restype = wintypes.BOOL
_CRYPT.CryptUnprotectData.argtypes = [
    ctypes.POINTER(_DataBlob),
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(_DataBlob),
]
_CRYPT.CryptUnprotectData.restype = wintypes.BOOL
_KERNEL.LocalFree.argtypes = [wintypes.HLOCAL]
_KERNEL.LocalFree.restype = wintypes.HLOCAL
_KERNEL.CreateDirectoryW.argtypes = [
    wintypes.LPCWSTR,
    ctypes.POINTER(_SecurityAttributes),
]
_KERNEL.CreateDirectoryW.restype = wintypes.BOOL
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
_KERNEL.CloseHandle.argtypes = [wintypes.HANDLE]
_KERNEL.CloseHandle.restype = wintypes.BOOL
_KERNEL.SetFileInformationByHandle.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
]
_KERNEL.SetFileInformationByHandle.restype = wintypes.BOOL


class _FileRenameInfo(ctypes.Structure):
    """按打开的真实文件对象重命名，不重新解析可被替换的原路径。"""

    _fields_ = [
        ("replace", wintypes.BOOL),
        ("root", wintypes.HANDLE),
        ("length", wintypes.DWORD),
        ("name", wintypes.WCHAR * 1),
    ]


@contextmanager
def stable_file_read(path: Path) -> Iterator[BinaryIO]:
    """固定捕获对象并禁止并发写入或删除；参数：真实文件；返回：同一只读流，冲突明确失败。"""
    handle = _KERNEL.CreateFileW(
        str(path), _GENERIC_READ, 1, None, _OPEN_EXISTING, 0, None
    )
    if handle == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except BaseException:
        _KERNEL.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, "rb") as stream:
        yield stream


def _dpapi(content: bytes, *, decrypt: bool) -> bytes:
    """执行当前 Windows 账户保护；参数：原文或密文、方向；返回：结果，失败不返回替代字节。"""
    buffer = (ctypes.c_ubyte * len(content)).from_buffer_copy(content)
    source, target = _DataBlob(len(content), buffer), _DataBlob()
    function = _CRYPT.CryptUnprotectData if decrypt else _CRYPT.CryptProtectData
    if not function(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        _CRYPT_UI_FORBIDDEN,
        ctypes.byref(target),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        _KERNEL.LocalFree(target.data)


def protect_bytes(content: bytes) -> bytes:
    """加密完整敏感字节；参数：明文；返回：账户绑定的 DPAPI 密文。"""
    return _dpapi(content, decrypt=False)


def unprotect_bytes(content: bytes) -> bytes:
    """解密并校验完整密文；参数：DPAPI 字节；返回：原文，篡改或账户不符明确失败。"""
    return _dpapi(content, decrypt=True)


def _private_descriptor() -> Any:
    """生成当前账户和系统私有权限；参数：无；返回：可在创建时使用的安全描述符。"""
    token = _SECURITY.OpenProcessToken(-1, _SECURITY.TOKEN_QUERY)
    try:
        sid = _SECURITY.GetTokenInformation(token, _SECURITY.TokenUser)[0]
        identity = _SECURITY.ConvertSidToStringSid(sid)
    finally:
        token.Close()
    return _SECURITY.ConvertStringSecurityDescriptorToSecurityDescriptor(
        f"D:P(A;OICI;FA;;;{identity})(A;OICI;FA;;;SY)", 1
    )


def _descriptor_text(descriptor: Any) -> str:
    """编码文件权限；参数：Windows 描述符；返回：仅含 DACL 的 SDDL。"""
    return str(
        _SECURITY.ConvertSecurityDescriptorToStringSecurityDescriptor(
            descriptor, 1, _DACL_SECURITY_INFORMATION
        )
    )


def read_security(path: Path) -> str:
    """读取当前文件权限证据；参数：文件或目录；返回：完整 DACL，不读取正文。"""
    return _descriptor_text(
        _SECURITY.GetFileSecurity(str(path), _DACL_SECURITY_INFORMATION)
    )


def apply_security(path: Path, descriptor: str) -> None:
    """应用已核验权限；参数：目标与此前保存的 DACL；返回：无，不自行扩大或解除限制。"""
    with security_handle(path) as handle:
        set_handle_security(handle, descriptor)


@contextmanager
def security_handle(path: Path, *, rename: bool = False) -> Iterator[int]:
    """固定权限变更的真实文件对象；参数：路径；返回：支持重命名的句柄，退出关闭。"""
    access = _SECURITY_ACCESS | (_DELETE_ACCESS if rename else 0)
    handle = _KERNEL.CreateFileW(
        str(path), access, _SHARE_ALL, None, _OPEN_EXISTING, _BACKUP_SEMANTICS, None
    )
    if handle == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield int(handle)
    finally:
        _KERNEL.CloseHandle(handle)


def handle_security(handle: int) -> str:
    """读取固定对象的权限；参数：已打开句柄；返回：DACL，路径替换不改变读取对象。"""
    return _descriptor_text(
        _SECURITY.GetSecurityInfo(
            handle, _SECURITY.SE_FILE_OBJECT, _DACL_SECURITY_INFORMATION
        )
    )


def set_handle_security(handle: int, descriptor: str) -> None:
    """设置固定对象的权限；参数：句柄与保存的 DACL；返回：无，不能误修改同名新文件。"""
    parsed = _SECURITY.ConvertStringSecurityDescriptorToSecurityDescriptor(
        descriptor, 1
    )
    control = parsed.GetSecurityDescriptorControl()[0]
    flags = (
        _PROTECTED_DACL_INFORMATION
        if control & _SE_DACL_PROTECTED
        else _UNPROTECTED_DACL_INFORMATION
    )
    information = ctypes.c_long(_DACL_SECURITY_INFORMATION | flags).value
    _SECURITY.SetSecurityInfo(
        handle,
        _SECURITY.SE_FILE_OBJECT,
        information,
        None,
        None,
        parsed.GetSecurityDescriptorDacl(),
        None,
    )


def private_handle_security(handle: int) -> None:
    """在原件移动前收紧权限；参数：真实原件句柄；返回：无，后续备份继承已私有的权限。"""
    set_handle_security(handle, _descriptor_text(_private_descriptor()))


def move_handle_file(handle: int, destination: Path) -> None:
    """将固定原件移至未占用同卷位置；参数：含DELETE权限句柄及目标；返回：无，绝不覆盖竞争文件。"""
    name = str(destination.absolute()).encode("utf-16-le")
    length = _FileRenameInfo.name.offset + len(name)
    buffer = ctypes.create_string_buffer(
        max(length + ctypes.sizeof(wintypes.WCHAR), ctypes.sizeof(_FileRenameInfo))
    )
    info = _FileRenameInfo.from_buffer(buffer)
    info.replace, info.root, info.length = False, None, len(name)
    ctypes.memmove(
        ctypes.addressof(buffer) + _FileRenameInfo.name.offset, name, len(name)
    )
    if not _KERNEL.SetFileInformationByHandle(
        handle, _FILE_RENAME_INFO_CLASS, buffer, ctypes.sizeof(buffer)
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def protected_replace_prepared_file(
    path: Path, temporary: Path, *, backup_path: Path
) -> str:
    """安全移动敏感原件再发布；参数：原路径、私有暂存和已登记备份；返回：实际原ACL，失败保留真实进度。"""
    from tools.file_persistence import publish_prepared_file

    verify_private_acl(temporary)
    verify_private_acl(backup_path.parent)
    with security_handle(path, rename=True) as original:
        security = handle_security(original)
        private_handle_security(original)
        try:
            move_handle_file(original, backup_path)
        except BaseException:
            set_handle_security(original, security)
            raise
        # 1. 【文件恢复】【敏感发布】新路径只在仍空缺时接纳；中断保留已登记且私有的实际原件
        with security_handle(temporary) as replacement:
            publish_prepared_file(path, temporary)
            set_handle_security(replacement, security)
    return security


def create_private_staging(parent: Path) -> Path:
    """在目标卷创建私有暂存目录；参数：已存在父目录；返回：需登记进恢复意图的精确目录。"""
    path = Path(parent).resolve() / (".reins-restore-" + uuid4().hex)
    descriptor = ctypes.create_string_buffer(bytes(_private_descriptor()))
    attributes = _SecurityAttributes(
        ctypes.sizeof(_SecurityAttributes),
        ctypes.cast(descriptor, wintypes.LPVOID),
        False,
    )
    if not _KERNEL.CreateDirectoryW(str(path), ctypes.byref(attributes)):
        raise ctypes.WinError(ctypes.get_last_error())
    return path


def verify_private_acl(path: Path) -> None:
    """验证暂存没有其他可读取账户；参数：私有路径；返回：无，权限被放宽明确失败。"""
    expected = _private_descriptor().GetSecurityDescriptorDacl()
    allowed = {
        _SECURITY.ConvertSidToStringSid(expected.GetAce(index)[2])
        for index in range(expected.GetAceCount())
    }
    descriptor = _SECURITY.GetFileSecurity(str(path), _DACL_SECURITY_INFORMATION)
    dacl = descriptor.GetSecurityDescriptorDacl()
    if dacl is None:
        raise PermissionError("restore staging has no private DACL")
    for index in range(dacl.GetAceCount()):
        ace = dacl.GetAce(index)
        if ace[0][0] == _SECURITY.ACCESS_DENIED_ACE_TYPE:
            continue
        if (
            ace[0][0] != _SECURITY.ACCESS_ALLOWED_ACE_TYPE
            or _SECURITY.ConvertSidToStringSid(ace[2]) not in allowed
        ):
            raise PermissionError("restore staging grants access to another account")


def write_private_file(path: Path, content: bytes) -> None:
    """在私有目录创建同步文件；参数：未占用路径及完整字节；返回：无，拒绝覆盖和宽权限暂存。"""
    verify_private_acl(path.parent)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    with security_handle(path) as security:
        private_handle_security(security)
    verify_private_acl(path)


class ProtectedContentStore:
    """仅持久化 DPAPI 密文，明文摘要和权限材料留在同一个加密封装内。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定密文原件空间；参数：数据根；返回：不读取秘密的服务。"""
        self.data_root = Path(data_root).resolve()

    def freeze(
        self,
        content: bytes,
        *,
        workspace_id: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContentReference:
        """发布不可变密文；参数：原文、工作区、私有元信息；返回：只有密文摘要的共享内容引用。"""
        from runtime.persistence import RuntimeStore
        from tools.file_persistence import publish_prepared_file

        store = RuntimeStore(self.data_root)
        store.require_current_format()
        owner = (
            self.data_root / "global"
            if workspace_id is None
            else store.workspace_directory(workspace_id)
        )
        directory = owner / "protected"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "content": base64.b64encode(content).decode("ascii"),
            "sha256": digest_bytes(content),
            "metadata": dict(metadata or {}),
        }
        encrypted = protect_bytes(json_bytes(payload))
        digest = digest_bytes(encrypted)
        path = directory / (digest + ".dpapi")
        staging = create_private_staging(directory)
        temporary = staging / "encrypted.tmp"
        try:
            write_private_file(temporary, encrypted)
            publish_prepared_file(path, temporary)
        finally:
            temporary.unlink(missing_ok=True)
            staging.rmdir()
        return ContentReference(
            path.relative_to(self.data_root).as_posix(),
            digest,
            len(encrypted),
            PROTECTED_MEDIA_TYPE,
        )

    def _unpack(self, reference: ContentReference) -> dict[str, Any]:
        """读取并验证受保护封装；参数：密文引用；返回：内部原文和元信息，错误不附秘密。"""
        if (
            reference.media_type != PROTECTED_MEDIA_TYPE
            or "protected" not in Path(reference.path).parts
        ):
            raise SourceCorruptionError("invalid protected content reference")
        verify_private_acl(confined_path(self.data_root, reference.path))
        encrypted = ContentFiles(self.data_root).read(reference)
        try:
            value = json.loads(unprotect_bytes(encrypted))
            content = base64.b64decode(value["content"], validate=True)
            if digest_bytes(content) != value["sha256"] or not isinstance(
                value["metadata"], dict
            ):
                raise SourceCorruptionError("protected content integrity failed")
        except (ValueError, KeyError, TypeError) as exc:
            raise SourceCorruptionError("invalid protected content envelope") from exc
        return {"content": content, "metadata": value["metadata"]}

    def read(self, reference: ContentReference) -> bytes:
        """恢复原始秘密字节；参数：密文引用；返回：仅供宿主写入使用的准确原文。"""
        content: bytes = self._unpack(reference)["content"]
        return content

    def metadata(self, reference: ContentReference) -> dict[str, Any]:
        """读取私有权限与版本证据；参数：密文引用；返回：宿主使用的私有元信息。"""
        metadata: dict[str, Any] = self._unpack(reference)["metadata"]
        return metadata
