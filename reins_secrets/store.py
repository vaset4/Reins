from __future__ import annotations

import getpass
import importlib
import logging
import sys
from collections.abc import Callable
from ctypes import c_int32
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_WINDOWS_ACL_ERROR = "secrets vault requires Windows NTFS ACL checks"
_READ_MASK = 0x80000000 | 0x10000000 | 0x00020000 | 0x0089
_PRIVATE_ACCESS_MASK = 0xC0000000


class SecretsACLError(RuntimeError):
    pass


class SecretNotFoundError(RuntimeError):
    pass


ActionHandler = Callable[..., Any]


class SecretsVault:
    def __init__(self, env_path: Path | None = None) -> None:
        self.env_path = env_path or Path.home() / ".reins" / ".env"
        self._action_handlers: dict[str, ActionHandler] = {
            "sign_request": self._sign_request,
            "sign_oauth": self._not_implemented,
            "encrypt_payload": self._not_implemented,
        }
        self._ensure_file()
        self._check_acl(self.env_path)

    def set(self, name: str, value: str) -> None:
        values = self._read_values()
        values[name] = value
        self._write_values(values)

    def get(self, name: str) -> str | None:
        return self._read_values().get(name)

    def use(self, name: str, action: str, **kwargs: object) -> Any:
        secret_value = self.get(name)
        if secret_value is None:
            raise SecretNotFoundError(f"secret not found: {name}")
        handler = self._action_handlers.get(action)
        if handler is None:
            raise NotImplementedError(action)
        return handler(secret_value, **kwargs)

    def list_names(self) -> list[str]:
        return list(self._read_values())

    def delete(self, name: str) -> bool:
        values = self._read_values()
        if name not in values:
            return False
        del values[name]
        self._write_values(values)
        return True

    def _ensure_file(self) -> None:
        if not self.env_path.exists():
            self.env_path.parent.mkdir(parents=True, exist_ok=True)
            self.env_path.write_text("", encoding="utf-8")
            self._set_private_acl(self.env_path)

    def _check_acl(self, path: Path) -> None:
        if sys.platform != "win32":
            raise SecretsACLError(_WINDOWS_ACL_ERROR)
        win32security = _load_win32security()
        allowed = {
            _sid_text(win32security, sid) for sid in _allowed_sids(win32security)
        }
        info = win32security.DACL_SECURITY_INFORMATION
        descriptor = win32security.GetFileSecurity(str(path), info)
        dacl = descriptor.GetSecurityDescriptorDacl()
        if dacl is None:
            logger.error("secrets ACL check failed: missing DACL")
            raise SecretsACLError("secrets ACL missing DACL")
        for index in range(dacl.GetAceCount()):
            ace = dacl.GetAce(index)
            ace_type = int(ace[0][0])
            access_mask = int(ace[1])
            sid = _sid_text(win32security, ace[2])
            allowed_ace_type = int(getattr(win32security, "ACCESS_ALLOWED_ACE_TYPE", 0))
            if (
                ace_type == allowed_ace_type
                and access_mask & _READ_MASK
                and sid not in allowed
            ):
                logger.error("secrets ACL check failed: unexpected readable SID")
                raise SecretsACLError("secrets ACL allows another SID")

    def _set_private_acl(self, path: Path) -> None:
        """为凭据文件设置用户和系统读写权限；传参：文件路径；返回：无。"""
        if sys.platform != "win32":
            raise SecretsACLError(_WINDOWS_ACL_ERROR)
        win32security = _load_win32security()
        sids = _allowed_sids(win32security)
        current_user_sid, system_sid = sids[0], sids[1]
        dacl = win32security.ACL()
        # 【凭据库】【设置权限】Windows ACL 接口接收有符号32位值，读写权限位保持不变
        for sid in (current_user_sid, system_sid):
            dacl.AddAccessAllowedAce(
                win32security.ACL_REVISION, c_int32(_PRIVATE_ACCESS_MASK).value, sid
            )
        descriptor = win32security.SECURITY_DESCRIPTOR()
        descriptor.SetSecurityDescriptorDacl(1, dacl, 0)
        win32security.SetFileSecurity(
            str(path), win32security.DACL_SECURITY_INFORMATION, descriptor
        )

    def _read_values(self) -> dict[str, str]:
        values: dict[str, str] = {}
        lines = (
            self.env_path.read_text(encoding="utf-8").splitlines()
            if self.env_path.exists()
            else []
        )
        for line in lines:
            if line and not line.lstrip().startswith("#") and "=" in line:
                name, value = line.split("=", maxsplit=1)
                stripped_name = name.strip()
                if stripped_name:
                    values[stripped_name] = value.strip()
        return values

    def _write_values(self, values: dict[str, str]) -> None:
        self.env_path.write_text(
            "".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8"
        )

    def _sign_request(
        self,
        secret: str,
        *,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        signed_headers = dict(headers or {})
        signed_headers["Authorization"] = f"Bearer {secret}"
        return {"url": url, "signed_headers": signed_headers}

    def _not_implemented(self, _secret: str, **_kwargs: object) -> None:
        raise NotImplementedError


def _allowed_sids(win32security: Any) -> tuple[Any, ...]:
    current_user_sid, _domain, _account_type = win32security.LookupAccountName(
        None, getpass.getuser()
    )
    system_sid = win32security.CreateWellKnownSid(win32security.WinLocalSystemSid, None)
    admins_sid = win32security.CreateWellKnownSid(
        win32security.WinBuiltinAdministratorsSid, None
    )
    return current_user_sid, system_sid, admins_sid


def _load_win32security() -> Any:
    """先加载配套Windows类型库再使用ACL组件；传参：无；返回：权限模块。"""
    try:
        # 【凭据库】【依赖加载】先由pywin32定位配套DLL，避免ACL组件锁定旧DLL后使MCP加载失败
        importlib.import_module("pywintypes")
        import win32security
    except ImportError as exc:
        raise SecretsACLError("pywin32 win32security is required on Windows") from exc
    return win32security


def _sid_text(win32security: Any, sid: object) -> str:
    converter = getattr(win32security, "ConvertSidToStringSid", None)
    return str(converter(sid) if converter else sid)
