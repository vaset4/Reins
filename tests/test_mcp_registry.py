from __future__ import annotations

import json
import sys
from contextlib import closing
from pathlib import Path

import pytest

from approval import ApprovalDecision
from runtime.lease import Lease, from_trigger
from runtime.watchdog import WatchdogDecision
from tools.types import ToolError, ToolErrorCategory
from tools.tool_registry import (
    MCPToolRiskError,
    TARGET_SCOPE_PATH,
    ToolRegistry,
    ToolRisk,
)


def test_mcp_yaml_registers_allowed_tools_and_resolves_secrets(tmp_path: Path) -> None:
    from tools.mcp_client.registry import MCPRegistry

    config = tmp_path / "mcp.yaml"
    config.write_text(
        """
servers:
  github:
    command: ["python", "server.py"]
    transport: stdio
    env:
      TOKEN: ${secret:GITHUB_TOKEN}
    tools:
      read_file:
        description: Read file.
        inputSchema:
          type: object
          properties:
            path: {type: string}
          required: [path]
""",
        encoding="utf-8",
    )
    registry = ToolRegistry()
    mcp = MCPRegistry(
        _lease(config, ["github"]), registry, vault=_Vault({"GITHUB_TOKEN": "token"})
    )

    mcp.register_allowed_tools()

    definition = registry.get("mcp_github_read_file")
    assert definition is not None
    assert definition.risk is ToolRisk.CONFIRM
    assert definition.target_scope_rule == TARGET_SCOPE_PATH
    assert mcp.configs["github"].env["TOKEN"] == "token"


def test_mcp_risk_override_to_deny_and_safe_lock(tmp_path: Path) -> None:
    from tools.mcp_client.registry import MCPRegistry

    config = _config_with_tool(tmp_path, "deny")
    registry = ToolRegistry()
    MCPRegistry(_lease(config, ["echo"]), registry).register_allowed_tools()
    assert registry.get("mcp_echo_echo").risk is ToolRisk.DENY  # type: ignore[union-attr]

    config = _config_with_tool(tmp_path, "safe")
    with pytest.raises(MCPToolRiskError):
        MCPRegistry(_lease(config, ["echo"]), ToolRegistry()).register_allowed_tools()


def test_disallowed_server_and_invalid_schema_are_skipped(tmp_path: Path) -> None:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        """
servers:
  echo:
    command: ["python", "server.py"]
    tools:
      broken:
        parameters: "bad"
""",
        encoding="utf-8",
    )

    registry = _attach(config, [])
    assert registry.list_tool_names() == []

    _attach(config, ["echo"], registry)
    assert registry.list_tool_names() == []


def test_mcp_path_like_parameter_still_uses_path_security(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        """
servers:
  echo:
    command: ["python", "server.py"]
    tools:
      read_file:
        description: Read file.
        inputSchema:
          type: object
          properties:
            filePath: {type: string}
          required: [filePath]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    registry = _attach(config, ["echo"])

    result = registry.execute_tool(
        "mcp_echo_read_file",
        {"filePath": ".env"},
        _lease(config, ["echo"]),
        watchdog=_Watchdog(),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "path_security_deny"


def test_mcp_schema_format_path_parameter_uses_path_security(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        """
servers:
  echo:
    command: ["python", "server.py"]
    tools:
      read_file:
        description: Read file.
        inputSchema:
          type: object
          properties:
            target:
              type: string
              format: path
          required: [target]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    registry = _attach(config, ["echo"])

    result = registry.execute_tool(
        "mcp_echo_read_file",
        {"target": ".env"},
        _lease(config, ["echo"]),
        watchdog=_Watchdog(),
    )

    definition = registry.get("mcp_echo_read_file")
    assert definition is not None
    assert definition.target_scope_rule == TARGET_SCOPE_PATH
    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "path_security_deny"


def test_mcp_atypical_write_verb_uses_write_path_security(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    workspace = project / ".reins" / "workspace"
    workspace.mkdir(parents=True)
    source_file = project / "source.txt"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("source", encoding="utf-8")
    config = tmp_path / "mcp.yaml"
    config.write_text(
        """
servers:
  echo:
    command: ["python", "server.py"]
    tools:
      put_file:
        description: Put file.
        inputSchema:
          type: object
          properties:
            path: {type: string}
          required: [path]
""",
        encoding="utf-8",
    )
    approval_calls: list[str] = []
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda req: approval_calls.append(req.tool) or ApprovalDecision.DENY,
    )
    lease = _lease_with_fs(config, ["echo"], project, workspace)
    from tools.mcp_client.registry import attach_mcp_registry

    registry = ToolRegistry()
    attach_mcp_registry(lease, registry)

    result = registry.execute_tool(
        "mcp_echo_put_file",
        {"path": str(source_file)},
        lease,
        watchdog=_Watchdog(tmp_path / "data"),
    )

    definition = registry.get("mcp_echo_put_file")
    assert definition is not None
    assert definition.readonly is False
    assert approval_calls == ["mcp_echo_put_file"]
    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "approval_denied"


def test_execute_mcp_tool_discovers_and_lazy_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "mcp.yaml"
    fixture = Path("tests/fixtures/standard_mcp_server.py").resolve()
    command = json.dumps([sys.executable, str(fixture)])
    config.write_text(
        f"""
servers:
  echo:
    command: {command}
    transport: stdio
    env: {{}}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    # 真实服务器进程的所有权已经收回到目录上，用完必须关目录，否则进程留着不放
    with closing(_attach(config, ["echo"])) as registry:
        result = registry.execute_tool(
            "mcp_echo_echo",
            {"text": "hello"},
            _lease(config, ["echo"]),
            watchdog=_Watchdog(tmp_path / "data"),
        )

    assert result["content"]["structuredContent"]["text"] == "hello"


def test_mcp_without_a_usable_config_reports_the_reason(tmp_path: Path) -> None:
    # 配了允许的服务器但配置读不到时，失败原因要落到来源状态上，
    # 让宿主和模型能解释「为什么这个连接没出现」，而不是只看到目录空着
    registry = _attach(tmp_path / "missing-mcp.yaml", ["echo"])

    status = registry.source_status()["mcp"]

    assert registry.list_tool_names() == []
    assert "mcp_not_configured" in status["errors"]["echo"]


def test_mcp_disallowed_server_exposes_nothing(tmp_path: Path) -> None:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        """
servers:
  echo:
    command: ["python", "server.py"]
    tools:
      echo:
        description: Echo.
        parameters: {}
""",
        encoding="utf-8",
    )
    registry = _attach(config, [])

    assert registry.list_tool_names() == []
    assert registry.source_status()["mcp"]["servers"] == []


def _attach(
    config: Path,
    allow_servers: list[str],
    registry: ToolRegistry | None = None,
) -> ToolRegistry:
    """按当前授权接通 MCP 目录；传参：配置、允许的服务器和已有目录；返回：刷新后的目录。"""
    from tools.mcp_client.registry import attach_mcp_registry

    registry = registry if registry is not None else ToolRegistry()
    attach_mcp_registry(_lease(config, allow_servers), registry)
    return registry


def _lease(config: Path, allow_servers: list[str]) -> Lease:
    caps = from_trigger("user", task_id="task").capabilities
    caps["mcp"] = {
        "enabled": True,
        "allow_servers": allow_servers,
        "config_path": str(config),
    }
    return from_trigger("user", task_id="task", capabilities=caps)


def _lease_with_fs(
    config: Path,
    allow_servers: list[str],
    project: Path,
    workspace: Path,
) -> Lease:
    caps = from_trigger("user", task_id="task").capabilities
    caps["fs"] = {
        "project_root": str(project),
        "read": [str(project)],
        "write": [str(workspace)],
    }
    caps["mcp"] = {
        "enabled": True,
        "allow_servers": allow_servers,
        "config_path": str(config),
    }
    return from_trigger("user", task_id="task", capabilities=caps)


def _config_with_tool(tmp_path: Path, risk: str) -> Path:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        f"""
servers:
  echo:
    command: ["python", "server.py"]
    tool_risk_overrides:
      echo: {risk}
    tools:
      echo:
        description: Echo.
        parameters: {{}}
""",
        encoding="utf-8",
    )
    return config


class _Vault:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get(self, name: str) -> str | None:
        return self.values.get(name)


class _Watchdog:
    tool_timeout_seconds = 30.0

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root

    def run_tool_with_timeout(
        self, operation, *, cancellation=None, on_late=None
    ) -> object:
        return operation()  # type: ignore[operator]

    def reserve_tool_step(self, *, operation_id: str = "") -> WatchdogDecision:
        """记录一次工具派发；传参：操作身份；返回：放行决定，步数不再拦派发。"""
        return WatchdogDecision(False)

    def record_tool_failure(self, _tool: str, _args: dict[str, object]) -> None:
        pass
