from __future__ import annotations

from pathlib import Path

from artifacts.store import ArtifactStore
from runtime.lease import from_trigger
from tasks.index_sync import connect_index
from tools.code_execution_tool import code_execute, executor
from tools.types import ToolError, ToolErrorCategory


def _workspace_lease(project: Path, task_id: str = "task-1"):
    workspace = project / ".reins" / "workspace"
    (workspace / task_id).mkdir(parents=True, exist_ok=True)
    return from_trigger(
        "user",
        task_id=task_id,
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(project), str(workspace)],
                "write": [str(workspace)],
            },
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {"enabled": True, "profile": "default", "deny_domains": []},
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
            "code_execution": {"enabled": True},
        },
    )


def test_code_execute_returns_stdout(tmp_path: Path) -> None:
    result = code_execute('print("hi")', tmp_path)

    assert result["stdout"] == "hi\n"
    assert result["stderr"] == ""
    assert result["exit_code"] == 0


def test_code_execute_returns_exception_exit(tmp_path: Path) -> None:
    result = code_execute('raise RuntimeError("bad")', tmp_path)

    assert result["exit_code"] != 0
    assert "RuntimeError" in str(result["stderr"])


def test_code_execute_timeout_flags_timed_out(tmp_path: Path) -> None:
    lease = _workspace_lease(tmp_path / "project")

    result = executor(
        {
            "code": "import time; time.sleep(30)",
            "__lease__": lease,
            "__timeout_seconds__": 1,
        }
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.TIMEOUT


def test_code_execute_large_stdout_uses_artifact(tmp_path: Path) -> None:
    connect_index(tmp_path).close()
    project = tmp_path / "project"
    lease = _workspace_lease(project, task_id="task-1")

    result = executor(
        {
            "code": 'print("x" * 5000)',
            "__lease__": lease,
            "__data_root__": tmp_path,
            "__task_id__": "task-1",
        }
    )

    assert result["stdout"] == ""  # type: ignore[index]
    ref = result["stdout_artifact"]  # type: ignore[index]
    assert ref["artifact_id"].startswith("art-")
    artifact_path = ArtifactStore(tmp_path).read_path(ref["artifact_id"])
    assert artifact_path.read_text(encoding="utf-8") == "x" * 5000 + "\n"


def test_executor_runs_via_exec_channel_in_workspace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    lease = _workspace_lease(project)

    result = executor(
        {
            "code": "import os; print(os.getcwd())",
            "__lease__": lease,
        }
    )

    assert isinstance(result, dict)
    assert result["exit_code"] == 0
    reported = Path(str(result["stdout"]).strip()).resolve()
    expected = (project / ".reins" / "workspace" / "task-1").resolve()
    assert reported == expected


def test_executor_denies_when_code_execution_capability_disabled(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    lease = _workspace_lease(project)
    object.__setattr__(
        lease,
        "capabilities",
        {**lease.capabilities, "code_execution": {"enabled": False}},
    )

    result = executor({"code": "print('x')", "__lease__": lease})

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "code_execution_disabled"


def test_executor_denies_when_lease_missing() -> None:
    # fail-closed：无 lease 不放行（受控通道不在无能力快照下跑代码）。
    result = executor({"code": "print('x')"})

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
