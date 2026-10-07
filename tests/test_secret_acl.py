from __future__ import annotations

import getpass
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reins_secrets.store import SecretsACLError, SecretsVault


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows ACL and SDK DLL initialization"
)
@pytest.mark.parametrize("existing_file", [False, True])
def test_vault_initialization_keeps_mcp_sdk_loadable_in_fresh_process(
    tmp_path: Path, existing_file: bool
) -> None:
    """凭据库创建和读取后均能加载MCP；传参：隔离目录及是否已有私有文件；返回：无。"""
    if existing_file:
        (tmp_path / ".env").write_text("", encoding="utf-8")
        subprocess.run(
            [
                "icacls",
                str(tmp_path / ".env"),
                "/inheritance:r",
                "/grant:r",
                f"{getpass.getuser()}:(R,W)",
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
    script = """
import sys
from pathlib import Path
from reins_secrets.store import SecretsVault
vault = SecretsVault(env_path=Path(sys.argv[1]))
assert vault.list_names() == []
from mcp.client.streamable_http import streamable_http_client
import win32api
assert callable(streamable_http_client)
assert win32api.GetCurrentProcessId() > 0
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script, str(tmp_path / ".env")],
        capture_output=True,
        encoding="utf-8",
        timeout=20,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr


def test_acl_allows_current_user_and_system(tmp_path: Path, monkeypatch) -> None:
    _install_win32security(monkeypatch, [("CURRENT_USER", 0x80000000), ("SYSTEM", 1)])

    vault = SecretsVault(env_path=tmp_path / ".env")

    assert vault.list_names() == []


def test_acl_rejects_everyone_read(tmp_path: Path, monkeypatch) -> None:
    _install_win32security(
        monkeypatch,
        [("CURRENT_USER", 0x80000000), ("EVERYONE", 0x80000000)],
    )

    with pytest.raises(SecretsACLError):
        SecretsVault(env_path=tmp_path / ".env")


def test_acl_ignores_non_readable_other_sid(tmp_path: Path, monkeypatch) -> None:
    _install_win32security(
        monkeypatch,
        [("CURRENT_USER", 0x80000000), ("EVERYONE", 0x40000000)],
    )

    vault = SecretsVault(env_path=tmp_path / ".env")

    assert vault.list_names() == []


def test_first_create_sets_private_acl(tmp_path: Path, monkeypatch) -> None:
    module = _install_win32security(
        monkeypatch,
        [("CURRENT_USER", 0x80000000), ("SYSTEM", 1)],
    )
    env_path = tmp_path / ".reins" / ".env"

    SecretsVault(env_path=env_path)

    assert env_path.exists()
    assert module.set_calls == [str(env_path)]
    assert module.last_dacl.added_sids == ["CURRENT_USER", "SYSTEM"]
    assert [mask & 0xFFFFFFFF for mask in module.last_dacl.added_masks] == [
        0xC0000000,
        0xC0000000,
    ]


class _FakeAcl:
    def __init__(self, aces: list[tuple[str, int]] | None = None) -> None:
        self.aces = aces or []
        self.added_sids: list[str] = []
        self.added_masks: list[int] = []

    def GetAceCount(self) -> int:
        return len(self.aces)

    def GetAce(self, index: int) -> tuple[tuple[int, int], int, str]:
        sid, access_mask = self.aces[index]
        return (0, 0), access_mask, sid

    def AddAccessAllowedAce(self, _revision: int, _access_mask: int, sid: str) -> None:
        self.added_sids.append(sid)
        self.added_masks.append(_access_mask)


class _FakeSecurityDescriptor:
    def __init__(self, dacl: _FakeAcl | None = None) -> None:
        self.dacl = dacl or _FakeAcl()

    def GetSecurityDescriptorDacl(self) -> _FakeAcl:
        return self.dacl

    def SetSecurityDescriptorDacl(
        self, _present: int, dacl: _FakeAcl, _defaulted: int
    ) -> None:
        self.dacl = dacl


def _install_win32security(monkeypatch, aces: list[tuple[str, int]]) -> SimpleNamespace:
    descriptor = _FakeSecurityDescriptor(_FakeAcl(aces))

    module = SimpleNamespace(
        ACL=lambda: _FakeAcl(),
        ACL_REVISION=2,
        ACCESS_ALLOWED_ACE_TYPE=0,
        DACL_SECURITY_INFORMATION=4,
        SECURITY_DESCRIPTOR=lambda: _FakeSecurityDescriptor(),
        WinLocalSystemSid=22,
        WinBuiltinAdministratorsSid=26,
        CreateWellKnownSid=lambda sid_type, _domain: (
            "SYSTEM" if sid_type == 22 else "BUILTIN\\Administrators"
        ),
        GetFileSecurity=lambda _path, _info: descriptor,
        LookupAccountName=lambda _system, _name: ("CURRENT_USER", "domain", "type"),
        SetFileSecurity=lambda path, _info, sd: _record_set(module, path, sd),
        set_calls=[],
        last_dacl=None,
    )
    monkeypatch.setitem(sys.modules, "win32security", module)
    return module


def _record_set(
    module: SimpleNamespace, path: str, sd: _FakeSecurityDescriptor
) -> None:
    module.set_calls.append(path)
    module.last_dacl = sd.dacl
