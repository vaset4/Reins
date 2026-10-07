from __future__ import annotations

import json
import sys
from contextlib import closing
from pathlib import Path

import pytest

from approval import ApprovalDecision
from app.run_task import run_task
from llm.tool_selection import select_tools
from llm.types import LLMPlan
from runtime.lease import Lease, from_trigger
from runtime.watchdog import Watchdog
from tools.tool_registry import ToolRegistry
from tools.builtin_tools import build_tool_registry


def test_stable_mcp_scenario_uses_standard_service_and_real_exit_evidence(tmp_path):
    """旧场景迁移后仍验证真实发现、拒绝与进程退出；传参：隔离证据根；返回：无。"""
    from scripts.stable_ux_oracles import evaluate
    from scripts.stable_ux_scenarios import run_scenario

    evidence = run_scenario(
        "scenario_6",
        evidence_root=tmp_path,
        source_root=Path(__file__).resolve().parents[1],
    )
    result = evaluate("scenario_6", evidence)
    assert result.status == "pass", result


def test_run_task_discovers_mcp_before_first_catalog_and_retains_injected_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.run_task as run_task_module

    project_root = tmp_path / "project"
    project_root.mkdir()
    config = _declared_echo_config(tmp_path)
    lease = _lease(config, ["echo"])
    llm = _CatalogCaptureLLM()
    monkeypatch.setattr(run_task_module, "_default_chat_lease", lambda **_kwargs: lease)
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )

    data_root = project_root / ".reins" / "data"
    with closing(
        build_tool_registry(repo_root=project_root, data_root=data_root)
    ) as registry:
        response = run_task(
            task="inspect allowed mcp tools",
            project_root=project_root,
            llm_client=llm,
            tool_registry=registry,
        )
        assert response.status == "done"
        assert "capabilities" in llm.visible_tool_names
        assert "mcp_echo_echo" not in llm.visible_tool_names
        assert "mcp_echo_echo" in llm.discoverable_tool_names
        result = registry.execute_tool(
            "mcp_echo_echo",
            {"text": "hello"},
            lease,
            watchdog=Watchdog(lease, data_root=data_root),
        )
        assert result["content"]["structuredContent"]["text"] == "hello"


def _lease(config: Path, allow_servers: list[str]) -> Lease:
    caps = from_trigger("user", task_id="task").capabilities
    caps["mcp"] = {
        "enabled": True,
        "allow_servers": allow_servers,
        "config_path": str(config),
    }
    return from_trigger("user", task_id="task", capabilities=caps)


def _declared_echo_config(tmp_path: Path) -> Path:
    fixture = Path("tests/fixtures/standard_mcp_server.py").resolve()
    command = json.dumps([sys.executable, str(fixture)])
    config = tmp_path / "mcp.yaml"
    config.write_text(
        f"""
servers:
  echo:
    command: {command}
    transport: stdio
""",
        encoding="utf-8",
    )
    return config


class _CatalogCaptureLLM:
    def __init__(self) -> None:
        self.visible_tool_names: set[str] = set()
        self.discoverable_tool_names: set[str] = set()
        self.registry: ToolRegistry | None = None

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task
        if not isinstance(context, dict):
            raise AssertionError("model context is required")
        registry = context.get("tool_registry")
        lease = context.get("capability_lease")
        if not isinstance(registry, ToolRegistry):
            raise AssertionError("per-run tool registry is required")
        self.registry = registry
        selection = select_tools(registry, lease=lease)
        self.visible_tool_names = {item.name for item in selection.selected_definitions}
        self.discoverable_tool_names = {
            item.name
            for item in select_tools(
                registry, lease=lease, include_deferred=True
            ).selected_definitions
        }
        return LLMPlan(final_output="catalog captured")

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: object,
        context: object | None = None,
    ) -> LLMPlan:
        del task, run_tools_result, context
        raise AssertionError("unexpected tool continuation")
