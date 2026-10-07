from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from secrets import (
    SystemRandom,
    choice,
    compare_digest,
    randbelow,
    randbits,
    token_bytes,
    token_hex,
    token_urlsafe,
)
from reins_secrets.store import SecretsVault


def test_store_round_trip_and_env_format(tmp_path: Path, monkeypatch) -> None:
    _install_acl(monkeypatch)
    env_path = tmp_path / ".reins" / ".env"
    vault = SecretsVault(env_path=env_path)

    vault.set("API_TOKEN", "first")
    vault.set("OTHER", "two=parts")
    vault.set("API_TOKEN", "second")

    assert env_path.read_text(encoding="utf-8") == "API_TOKEN=second\nOTHER=two=parts\n"
    assert vault.get("API_TOKEN") == "second"
    assert vault.get("MISSING") is None
    assert vault.list_names() == ["API_TOKEN", "OTHER"]
    assert vault.delete("API_TOKEN") is True
    assert vault.delete("API_TOKEN") is False
    assert vault.list_names() == ["OTHER"]


def test_store_reads_manually_spaced_env_lines(tmp_path: Path, monkeypatch) -> None:
    _install_acl(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n# manual edit\nAPI_TOKEN = raw-token \nOTHER=two=parts\n",
        encoding="utf-8",
    )
    vault = SecretsVault(env_path=env_path)

    assert vault.get("API_TOKEN") == "raw-token"
    assert vault.get("OTHER") == "two=parts"
    assert vault.list_names() == ["API_TOKEN", "OTHER"]


def test_use_calls_action_handler_without_returning_get_value(
    tmp_path: Path, monkeypatch
) -> None:
    _install_acl(monkeypatch)
    vault = SecretsVault(env_path=tmp_path / ".env")
    vault.set("API_TOKEN", "raw-token")

    result = vault.use(
        "API_TOKEN",
        "sign_request",
        url="https://example.test",
        headers={"Accept": "application/json"},
    )

    assert result == {
        "url": "https://example.test",
        "signed_headers": {
            "Accept": "application/json",
            "Authorization": "Bearer raw-token",
        },
    }


def test_unknown_action_raises(tmp_path: Path, monkeypatch) -> None:
    _install_acl(monkeypatch)
    vault = SecretsVault(env_path=tmp_path / ".env")
    vault.set("API_TOKEN", "raw-token")

    with pytest.raises(NotImplementedError):
        vault.use("API_TOKEN", "sign_oauth")


def test_stdlib_secrets_api_remains_available() -> None:
    assert len(token_bytes(4)) == 4
    assert len(token_hex(4)) == 8
    assert isinstance(token_urlsafe(4), str)
    assert choice(["a", "b"]) in {"a", "b"}
    assert 0 <= randbelow(10) < 10
    assert 0 <= randbits(8) < 256
    assert compare_digest("same", "same")
    assert isinstance(SystemRandom(), SystemRandom)


class _FakeAcl:
    def __init__(self) -> None:
        self.added: list[tuple[int, str]] = []

    def GetAceCount(self) -> int:
        return 2

    def GetAce(self, index: int) -> tuple[tuple[int, int], int, str]:
        sid = "CURRENT_USER" if index == 0 else "SYSTEM"
        return (0, 0), 0x80000000, sid

    def AddAccessAllowedAce(self, _revision: int, access_mask: int, sid: str) -> None:
        self.added.append((access_mask, sid))


class _FakeSecurityDescriptor:
    def __init__(self) -> None:
        self.dacl = _FakeAcl()

    def GetSecurityDescriptorDacl(self) -> _FakeAcl:
        return self.dacl

    def SetSecurityDescriptorDacl(
        self, _present: int, dacl: _FakeAcl, _defaulted: int
    ) -> None:
        self.dacl = dacl


def _install_acl(monkeypatch) -> SimpleNamespace:
    descriptor = _FakeSecurityDescriptor()
    module = SimpleNamespace(
        ACL=_FakeAcl,
        ACL_REVISION=2,
        ACCESS_ALLOWED_ACE_TYPE=0,
        DACL_SECURITY_INFORMATION=4,
        SECURITY_DESCRIPTOR=_FakeSecurityDescriptor,
        WinBuiltinAdministratorsSid=32,
        WinLocalSystemSid=22,
        CreateWellKnownSid=lambda sid_type, _domain: (
            "ADMINISTRATORS" if sid_type == 32 else "SYSTEM"
        ),
        GetFileSecurity=lambda _path, _info: descriptor,
        LookupAccountName=lambda _system, _name: ("CURRENT_USER", "domain", "type"),
        SetFileSecurity=lambda _path, _info, sd: setattr(module, "last_sd", sd),
        last_sd=None,
    )
    monkeypatch.setitem(sys.modules, "win32security", module)
    return module
