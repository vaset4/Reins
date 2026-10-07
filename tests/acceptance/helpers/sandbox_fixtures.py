from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from approval import ApprovalDecision, ApprovalRequest
from runtime.lease import Lease, from_trigger
from tasks.store import TaskStore
from tools.tool_registry import (
    IDEMPOTENT_CONDITIONAL,
    IDEMPOTENT_YES,
    TARGET_SCOPE_PATH,
    TOOL_RISK_CONFIRM,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_FILE,
    ToolDefinition,
    ToolRegistry,
)


@dataclass(frozen=True, slots=True)
class SandboxProject:
    home: Path
    data_root: Path
    project_root: Path
    task_id: str

    @property
    def workspace_scratch(self) -> Path:
        return self.project_root / ".reins" / "workspace" / self.task_id / "scratch"

    @property
    def source_file(self) -> Path:
        return self.project_root / "app" / "cli.py"


class ApprovalRecorder:
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.calls: list[ApprovalRequest] = []

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.calls.append(request)
        return self.decision


def create_sandbox_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_id: str = "task-1",
) -> SandboxProject:
    home = tmp_path / "home"
    data_root = home / ".reins" / "data"
    project_root = tmp_path / "project"
    workspace_scratch = project_root / ".reins" / "workspace" / task_id / "scratch"
    workspace_scratch.mkdir(parents=True)
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "cli.py").write_text("print('before')\n", encoding="utf-8")
    (home / ".ssh").mkdir(parents=True)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("REINS_HOME", str(home / ".reins"))

    TaskStore(data_root).create_task("DoD 3 sandbox acceptance", task_id=task_id)
    return SandboxProject(
        home=home,
        data_root=data_root,
        project_root=project_root,
        task_id=task_id,
    )


def lease_for(
    sandbox: SandboxProject,
    *,
    trigger: str = "user",
    max_steps: int = 30,
    max_tokens: int = 200000,
    required_permanent_grants: list[dict[str, object]] | None = None,
) -> Lease:
    capabilities: dict[str, object] = {
        "fs": {
            "project_root": str(sandbox.project_root),
            "read": [
                str(sandbox.data_root),
                str(sandbox.project_root / ".reins" / "workspace"),
                str(sandbox.project_root),
            ],
            "write": [
                str(sandbox.project_root / ".reins" / "workspace"),
                str(sandbox.data_root),
            ],
            "deny_read": ["%USERPROFILE%\\.ssh\\", "*.pem", "*.key", ".env"],
        },
        "terminal": {"enabled": True, "allow_commands": []},
        "browser": {"enabled": True, "profile": "default"},
        "mouse_keyboard": {"enabled": False},
        "network": {"enabled": True, "deny_domains": []},
        "background_run": {"enabled": False},
        "mcp": {"enabled": True, "allow_servers": []},
    }
    if required_permanent_grants is not None:
        capabilities["schedule"] = {
            "required_permanent_grants": required_permanent_grants
        }
    return from_trigger(
        trigger,
        task_id=sandbox.task_id,
        capabilities=capabilities,
        max_steps=max_steps,
        max_tokens=max_tokens,
    )


def sandbox_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="file_write",
            description="acceptance file write",
            parameters={
                "path": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
            },
            toolset=TOOLSET_FILE,
            risk_level=TOOL_RISK_CONFIRM,
            readonly=False,
            target_scope_rule=TARGET_SCOPE_PATH,
            source=TOOL_SOURCE_BUILTIN,
            idempotent=IDEMPOTENT_CONDITIONAL,
            executor=_write_file,
        )
    )
    registry.register(
        ToolDefinition(
            name="file_read",
            description="acceptance file read",
            parameters={"path": {"type": "string", "required": True}},
            toolset=TOOLSET_FILE,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_PATH,
            source=TOOL_SOURCE_BUILTIN,
            idempotent=IDEMPOTENT_YES,
            executor=_read_file,
        )
    )
    return registry


def _write_file(args: dict[str, object]) -> str:
    path = Path(str(args["path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(args["content"]), encoding="utf-8")
    return "wrote"


def _read_file(args: dict[str, object]) -> str:
    return Path(str(args["path"])).read_text(encoding="utf-8")
