from __future__ import annotations

import tools.secret_tool as secret_tool
from app import cli
from runtime.lease import from_trigger
from runtime.watchdog import WatchdogDecision
from tools import builtin_tools
from tools.tool_registry import ToolRegistry


def test_secret_use_returns_summary_without_raw_token(monkeypatch) -> None:
    monkeypatch.setattr(secret_tool, "SecretsVault", lambda: _Vault())

    result = secret_tool.secret_use(
        "API_TOKEN",
        url="https://example.test",
        headers={"Accept": "application/json"},
    )

    assert result["status"] == "ok"
    assert "action_result_summary" in result
    assert "raw-token" not in str(result)
    assert "Authorization" not in str(result)


def test_secret_list_names_returns_only_keys(monkeypatch) -> None:
    monkeypatch.setattr(secret_tool, "SecretsVault", lambda: _Vault())

    assert secret_tool.secret_list_names() == ["API_TOKEN"]


def test_model_tool_does_not_expose_get_or_set() -> None:
    assert not hasattr(secret_tool, "secret_get")
    assert not hasattr(secret_tool, "secret_set")


def test_secret_tools_register_as_safe_conditional() -> None:
    registry = ToolRegistry()

    secret_tool.register_tools(registry)

    use_definition = registry.get("secret_use")
    list_definition = registry.get("secret_list_names")
    assert use_definition is not None
    assert list_definition is not None
    assert use_definition.risk_level == "safe"
    assert use_definition.idempotent == "conditional"
    assert list_definition.risk_level == "safe"
    assert list_definition.idempotent == "conditional"


def test_builtin_registry_includes_secret_tools() -> None:
    registry = ToolRegistry()

    builtin_tools.register_tools(registry)

    assert registry.get("secret_use") is not None
    assert registry.get("secret_list_names") is not None


def test_secret_use_executor_passes_action_kwargs(monkeypatch) -> None:
    monkeypatch.setattr(secret_tool, "SecretsVault", lambda: _Vault())
    registry = ToolRegistry()

    secret_tool.register_tools(registry)

    result = registry.execute_tool(
        "secret_use",
        {
            "name": "API_TOKEN",
            "url": "https://example.test",
            "headers": {"Accept": "application/json"},
        },
        from_trigger("user", task_id="task"),
        watchdog=_Watchdog(),
    )

    assert result == {
        "status": "ok",
        "action_result_summary": "secret action completed",
    }


def test_cli_secret_set_uses_getpass_without_echo(monkeypatch, capsys) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(cli.sys, "argv", ["reins", "secret", "set", "API_TOKEN"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "raw-token")
    monkeypatch.setattr(cli, "SecretsVault", lambda: _CliVault(calls))

    assert cli.main() == 0

    captured = capsys.readouterr()
    assert calls == [("API_TOKEN", "raw-token")]
    assert "raw-token" not in captured.out


def test_cli_secret_list_prints_names_only(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.sys, "argv", ["reins", "secret", "list"])
    monkeypatch.setattr(cli, "SecretsVault", lambda: _ListCliVault())

    assert cli.main() == 0

    captured = capsys.readouterr()
    assert "API_TOKEN" in captured.out
    assert "raw-token" not in captured.out


def test_cli_secret_delete_reports_status(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.sys, "argv", ["reins", "secret", "delete", "API_TOKEN"])
    monkeypatch.setattr(cli, "SecretsVault", lambda: _DeleteCliVault())

    assert cli.main() == 0

    captured = capsys.readouterr()
    assert "secret deleted: API_TOKEN" in captured.out


class _Vault:
    def use(self, name: str, action: str, **kwargs) -> dict[str, object]:
        assert name == "API_TOKEN"
        assert action == "sign_request"
        return {
            "url": kwargs["url"],
            "signed_headers": {
                **dict(kwargs["headers"]),
                "Authorization": "Bearer raw-token",
            },
        }

    def list_names(self) -> list[str]:
        return ["API_TOKEN"]


class _CliVault:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls

    def set(self, name: str, value: str) -> None:
        self.calls.append((name, value))


class _ListCliVault:
    def list_names(self) -> list[str]:
        return ["API_TOKEN"]


class _DeleteCliVault:
    def delete(self, name: str) -> bool:
        return name == "API_TOKEN"


class _Watchdog:
    data_root = None
    tool_timeout_seconds = 30.0

    def run_tool_with_timeout(self, operation, *, cancellation=None, on_late=None):
        return operation()

    def reserve_tool_step(self, *, operation_id: str = "") -> WatchdogDecision:
        """记录一次工具派发；传参：操作身份；返回：放行决定，步数不再拦派发。"""
        return WatchdogDecision(False)

    def record_tool_failure(self, _tool: str, _args: dict[str, object]) -> None:
        return None
