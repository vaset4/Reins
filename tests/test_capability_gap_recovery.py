"""Behavior tests for 06-09-capability-gap-recovery.

Scope (per prd.md): the failed session proved the gap is *recovery strategy*,
not missing capability. PDF parsing and file_read/list/grep already exist.
These tests pin three concrete gaps:

- G2: a `find_path` tool that locates a real path from a bare filename
  when the model does not know the directory.
- G1: a `read_artifact` misuse on an ordinary workspace filename produces a
  recovery observation that points at the usable tools (file_read/inspect).
- G3: a recoverable tool failure carries a structured contract
  (category / recovery_possible / suggested_next_tools), not just a "re-plan"
  sentence.

The model-facing `find_path` contract uses `query` for a bare filename / glob /
fragment and `path` for the search root. The low-level inspection tests retain
`action="find_file"` and `target_path`, which remain valid internal inputs. The
registry path returns the executor dict ({content, summary, meta}), matching
the other readonly file tools.
"""

from __future__ import annotations

from pathlib import Path

from runtime.agent_loop import AgentLoop
from runtime.lease import from_trigger
from runtime.types import (
    ReadOnlyInspectionRequest,
    RunContext,
    RunToolsResult,
    Trigger,
)
from tasks.store import TaskStore


# --- G2: find_file primitive ------------------------------------------------


def test_find_file_locates_path_from_bare_filename(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "2605.pdf").write_bytes(b"%PDF-1.4 stub")
    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=50,
        max_chars=4000,
        max_matches=50,
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(action="find_file", target_path=".", query="2605.pdf")
    )

    assert result.status == "ok"
    assert "inputs/2605.pdf" in result.output or "inputs\\2605.pdf" in result.output
    assert result.meta["total_count"] == 1


def test_find_file_supports_glob_pattern(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.pdf").write_bytes(b"x")
    (tmp_path / "docs" / "b.pdf").write_bytes(b"y")
    (tmp_path / "docs" / "c.txt").write_text("z", encoding="utf-8")
    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=50,
        max_chars=4000,
        max_matches=50,
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(action="find_file", target_path=".", query="*.pdf")
    )

    assert result.status == "ok"
    assert "a.pdf" in result.output
    assert "b.pdf" in result.output
    assert "c.txt" not in result.output
    assert result.meta["total_count"] == 2


def test_find_file_reports_no_match_explicitly(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=50,
        max_chars=4000,
        max_matches=50,
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(
            action="find_file", target_path=".", query="missing.pdf"
        )
    )

    # No match is a real, honest empty result, not an error/clarification.
    assert result.status == "ok"
    assert result.meta["total_count"] == 0
    assert "NO_MATCH" in result.output


def test_find_file_rejects_nonexistent_directory(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=50,
        max_chars=4000,
        max_matches=50,
    )

    # A missing search dir is an honest rejection, not a silent fall back to a
    # whole-repo scan (which would hide that the model's directory was wrong).
    result = executor.execute(
        ReadOnlyInspectionRequest(
            action="find_file", target_path="no_such_dir", query="x.pdf"
        )
    )

    assert result.status == "rejected"
    assert "path not found: no_such_dir" in result.output
    assert result.meta["error_category"] == "not_found"


def test_find_path_is_registered_and_model_visible() -> None:
    """模型能看到当前路径查找工具且其保持只读；参数：无；返回：无。"""
    from tools.builtin_tools import build_tool_registry

    registry = build_tool_registry(repo_root=".")
    model_visible = registry.list_tool_names(model_visible_only=True)

    assert "find_path" in model_visible
    definition = registry.get("find_path")
    assert definition is not None
    assert definition.readonly is True


def test_find_path_executes_through_registry(tmp_path: Path) -> None:
    """经正式注册表查出实际文件路径；参数：隔离工作区；返回：无。"""
    from tools.builtin_tools import build_tool_registry

    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "2605.pdf").write_bytes(b"%PDF stub")
    registry = build_tool_registry(repo_root=tmp_path)
    lease = from_trigger(
        "user",
        capabilities={
            "fs": {
                "project_root": str(tmp_path),
                "read": [str(tmp_path)],
                "write": [],
            }
        },
    )

    result = registry.execute_tool("find_path", {"query": "2605.pdf"}, lease)

    # The registry path returns the executor dict, like the other file tools.
    assert isinstance(result, dict)
    assert "2605.pdf" in str(result["content"])
    assert result["meta"]["total_count"] == 1


# --- G1: read_artifact misuse points at usable tools ------------------------


def test_read_artifact_misuse_observation_points_at_file_tools(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    store = TaskStore(data_root)
    record = store.create_task("goal")
    loop = AgentLoop(data_root)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": "goal"},
        capability_lease=from_trigger("user", task_id=record.task_id),
    )
    result = RunToolsResult.error_result(
        action="read_artifact",
        tool_name="read_artifact",
        error="invalid_input: 2605.pdf",
    )

    observation = loop._handle_recoverable_tool_error(context, result)

    assert observation.meta["replan_required"] is True
    suggested = observation.meta["suggested_next_tools"]
    assert "file_read" in suggested
    assert "find_path" in suggested
    # The actionable alternative must be visible in the text the model reads.
    assert "file_read" in observation.output


# --- G3: structured recovery contract ---------------------------------------


def test_recoverable_observation_carries_structured_contract(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    store = TaskStore(data_root)
    record = store.create_task("goal")
    loop = AgentLoop(data_root)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": "goal"},
        capability_lease=from_trigger("user", task_id=record.task_id),
    )
    result = RunToolsResult.error_result(
        action="read_artifact",
        tool_name="read_artifact",
        error="invalid_input: 2605.pdf",
    )

    observation = loop._handle_recoverable_tool_error(context, result)

    assert observation.meta["category"] == "wrong_tool"
    assert observation.meta["recovery_possible"] is True
    assert isinstance(observation.meta["suggested_next_tools"], list)


def test_non_file_tool_invalid_input_is_not_labelled_wrong_tool(
    tmp_path: Path,
) -> None:
    """A wrong *parameter* on the *right* tool (terminal missing command,
    web_fetch bad url) must NOT be mislabelled as a wrong-tool/file misuse.
    Otherwise the recovery contract pushes the model toward file tools for a
    failure that has nothing to do with files - the mirror image of the bug
    this task removes."""
    data_root = tmp_path / "data"
    store = TaskStore(data_root)
    record = store.create_task("goal")
    loop = AgentLoop(data_root)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": "goal"},
        capability_lease=from_trigger("user", task_id=record.task_id),
    )

    for tool_name, error in (
        ("terminal_tool", "invalid_input: missing command"),
        ("web_fetch", "invalid_input: invalid_url"),
    ):
        result = RunToolsResult.error_result(
            action=tool_name,
            tool_name=tool_name,
            error=error,
        )

        observation = loop._handle_recoverable_tool_error(context, result)

        # Still a recoverable re-plan observation, but NOT a wrong-tool file
        # misuse: no wrong_tool category, no file-tool suggestions in either
        # the structured meta or the model-facing text.
        assert "category" not in observation.meta
        assert "suggested_next_tools" not in observation.meta
        assert "find_file" not in observation.output
        assert "find_path" not in observation.output
        assert "file_read" not in observation.output
        assert observation.meta["replan_required"] is True
