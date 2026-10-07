"""Goal focus 连续性与模型动作能力合同测试。"""

from __future__ import annotations
from scripts.testing.llm import from_test_sequence

from dataclasses import asdict
from pathlib import Path

import pytest

from app.run_task import run_task
from llm.model_request import ModelActionCapability, compose_model_request
from runtime.agent_loop import AgentLoop
from runtime.checkpoint import Checkpoint, checkpoint_to_ledger_state, save_checkpoint
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_facts import RunFactStore
from runtime.session_state import SessionStateStore
from runtime.types import RunContext, TerminalFocusPolicy
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import (
    IDEMPOTENT_YES,
    TARGET_SCOPE_PATH,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_FILE,
    ToolDefinition,
    ToolRegistry,
)
from triggers.resume import make_run_context as make_resume_context


def _project(tmp_path: Path) -> tuple[Path, Path]:
    """创建隔离项目与数据目录。

    传参：tmp_path 为 pytest 临时目录
    返回：项目根目录与 data root
    """
    project_root = tmp_path / "project"
    data_root = project_root / ".reins" / "data"
    (project_root / "tools").mkdir(parents=True)
    data_root.mkdir(parents=True)
    return project_root, data_root


def _safe_tool() -> ToolDefinition:
    """创建 capability 矩阵使用的可见安全工具。

    传参：无
    返回：最小 ToolDefinition
    """
    return ToolDefinition(
        name="file_read",
        description="Read a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        toolset=TOOLSET_FILE,
        risk_level=TOOL_RISK_SAFE,
        readonly=True,
        target_scope_rule=TARGET_SCOPE_PATH,
        source=TOOL_SOURCE_BUILTIN,
        idempotent=IDEMPOTENT_YES,
        executor=lambda _args: "ok",
    )


def test_one_shot_goal_new_commits_durable_focus_without_moving_run_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 one-shot goal new 同步 context/session/response，但不搬迁 run facts。

    传参：tmp_path 为隔离目录；monkeypatch 用于捕获生产 RunContext
    返回：无
    """
    project_root, data_root = _project(tmp_path)
    registry = build_tool_registry(repo_root=project_root, data_root=data_root)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"goal","arguments":{"action":"new","goal_body":"durable target"}}',
            '{"type":"final","content":"done"}',
        ],
        protocol_mode="text_json",
    )
    captured: list[RunContext] = []
    original_run = AgentLoop.run

    def capture_run(loop: AgentLoop, context: RunContext):
        """捕获 run_task 交给 AgentLoop 的可变上下文。

        传参：loop 为生产 AgentLoop；context 为本轮上下文
        返回：生产 AgentLoop.run 的终态
        """
        captured.append(context)
        return original_run(loop, context)

    monkeypatch.setattr(AgentLoop, "run", capture_run)

    response = run_task(
        "start here",
        project_root,
        data_root=data_root,
        llm_client=client,
        tool_registry=registry,
    )

    context = captured[0]
    storage_task_id = context.storage_task_id
    state = SessionStateStore(data_root).load(context.session_id)
    facts = RunFactStore(data_root).read_run(context.run_id)
    assert context.focus_task_id != storage_task_id
    assert response.task_id == context.focus_task_id
    assert state is not None
    assert state.focus_task_id == context.focus_task_id
    assert context.terminal_focus_policy is TerminalFocusPolicy.PRESERVE
    assert {str(row["task_id"]) for row in facts if row.get("task_id") is not None} == {
        storage_task_id
    }


def test_terminal_focus_policy_round_trips_through_checkpoint_resume(
    tmp_path: Path,
) -> None:
    """验证 PRESERVE 经过 Ledger checkpoint 与 resume 后不丢失。

    传参：tmp_path 为隔离 data root
    返回：无
    """
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id="task-storage",
            segment_id="segment-source",
            state="done",
            session_id="session-policy",
            run_id="run-source",
            focus_task_id="task-focus",
            terminal_focus_policy=TerminalFocusPolicy.PRESERVE,
            reason="done",
        )
    )
    LedgerWriter(
        LedgerStore(tmp_path), source="tests.goal_contract"
    ).record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        checkpoint.reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )

    resumed = make_resume_context("task-storage", data_root=tmp_path)

    assert resumed.focus_task_id == "task-focus"
    assert resumed.terminal_focus_policy is TerminalFocusPolicy.PRESERVE


def test_tui_bridge_projects_durable_goal_focus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证正式 TUI 桥接一轮结束后只投影 durable session focus。

    传参：tmp_path 为隔离项目目录，monkeypatch 替换网络传输
    返回：无
    """
    project_root, data_root = _project(tmp_path)
    store = TaskStore(data_root)
    initial = store.create_task("initial TUI goal")
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"goal","arguments":{"action":"new","goal_body":"TUI durable target"}}',
            '{"type":"final","content":"done"}',
        ],
        protocol_mode="text_json",
    )

    from tests.frontends.tui.test_stream_display_contracts import connected_tui_bridge

    registry = build_tool_registry(repo_root=project_root, data_root=data_root)
    try:
        with connected_tui_bridge(
            project_root,
            data_root,
            client,
            registry,
            monkeypatch,
            session_id="goal-tui",
            task_id=initial.task_id,
        ) as (bridge, background, notices):
            bridge.submit("/help")
            assert any(
                "Available commands" in str(data) and "/status" in str(data)
                for kind, data in notices
                if kind == "output"
            )
            bridge.submit("open a TUI goal")
            assert background.runtime.wait_idle(10)
            bridge.require_host().refresh_history()
            assert any(
                row.get("text") == "done"
                for row in background.snapshot(history=True)["history"]
            )
            session = SessionStateStore(data_root).load(bridge.state.session_id)
            assert session is not None
            assert bridge.state.current_task_id == session.focus_task_id
            assert bridge.state.current_task_id != initial.task_id
    finally:
        store.close()


@pytest.mark.parametrize(
    ("has_tools", "expected"),
    [(True, {"final", "run_tools"}), (False, {"final"})],
)
def test_model_action_capability_matrix(
    has_tools: bool,
    expected: set[str],
) -> None:
    """原生动作共享工具能力入口；传参：可见工具/预期动作；返回：无。"""
    capability = ModelActionCapability.for_turn(
        has_selected_tools=has_tools,
    )

    assert capability.allowed_actions == frozenset(expected)
    assert capability.to_mapping() == {"allowed_actions": sorted(expected)}


def test_composed_prompt_and_runtime_context_share_pending_capability() -> None:
    """验证 prompt 与 runtime 都读取 ComposedRequest 内同一 capability 映射。

    传参：无
    返回：无
    """
    registry = ToolRegistry()
    registry.register(_safe_tool())
    bundle = compose_model_request(
        task="resume pending",
        stage="plan",
        protocol_mode="text_json",
        model_context={
            "resume_choice_pending_evidence": {
                "tool_name": "file_write",
                "call_id": "call-pending",
            }
        },
        registry=registry,
        context_window=30000,
    )

    capability = bundle.prompt_context["model_action_capability"]
    prompt = "\n".join(part.text for part in bundle.request.instructions)
    assert capability == {"allowed_actions": ["final", "run_tools"]}
    assert '{"type":"resume_choice"' not in prompt
    assert '{"type":"final"' in prompt
    assert '{"type":"goal_op"' not in prompt
    assert '{"type":"run_tools"' in prompt
    assert asdict(bundle.tool_selection.selected[0])["name"] == "file_read"
