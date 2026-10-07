"""验证旧预算/分级提醒退役，并与基于真实证据的进展判断区分。

步数和token只计量；进展提醒由三维证据决定，可独立关闭。
作者：xxx
"""

from __future__ import annotations
from scripts.testing.llm import from_test_sequence

from pathlib import Path

import pytest

from runtime.agent_loop import AgentLoop, State
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.stream_events import (
    LifecycleChanged,
    SegmentPaused,
    ToolExecutionStarted,
)
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    tools_dir = project / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "alpha.py").write_text("print('hi')\n", encoding="utf-8")
    (project / ".reins" / "data").mkdir(parents=True, exist_ok=True)
    return project


def _lease(project: Path, task_id: str, *, max_steps: int = 30):
    data = project / ".reins" / "data"
    workspace = project / ".reins" / "workspace"
    return from_trigger(
        "user",
        task_id=task_id,
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(project), str(data), str(workspace)],
                "write": [str(data), str(workspace)],
                "deny_read": ["*.pem", "*.key", ".env"],
            },
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {"enabled": True, "profile": "default", "headless": True},
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
        max_steps=max_steps,
        max_tokens=200000,
    )


def _build_loop(
    project: Path, client, *, max_steps: int = 30, message: str = "test goal"
):
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task(message)
    lease = _lease(project, record.task_id, max_steps=max_steps)
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(data_root, llm_client=client, tool_registry=registry)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": message},
        capability_lease=lease,
        segment_id=f"user-{record.task_id}",
    )
    return loop, context


def test_no_progress_repeats_not_paused(tmp_path: Path) -> None:
    # 【进展】【提醒机会】基线之后三次重复产生提醒，模型仍可在下一请求完成回复
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"model decided to stop"}',
        ],
        protocol_mode="text_json",
    )
    loop, context = _build_loop(project, client)

    events = list(loop.run_stream(context))

    pause_events = [e for e in events if isinstance(e, SegmentPaused)]
    lifecycles = [e for e in events if isinstance(e, LifecycleChanged)]
    assert not pause_events
    assert loop.state is State.DONE
    assert lifecycles[-1].lifecycle == "done"
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    # 【进展】【运行事实】实际请求和运行轨迹使用同一重复次数
    no_progress_rows = [
        row for row in facts if row.get("event") == "progress:no_progress"
    ]
    assert no_progress_rows

    assert any(
        row["evidence"]["decision"] == "suspected" and row["evidence"]["repeats"] == 3
        for row in no_progress_rows
    )
    # 【进展】【阈值】只到提醒阈值，尚未满足暂停条件
    assert not any(row.get("event") == "progress:paused" for row in facts)
    assert "model decided to stop" in loop.last_output


def test_readonly_loop_runs_past_max_steps_without_pausing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 【预算】【只计量】关闭独立的进展检测，隔离验证旧步数上限不会拦截工具
    monkeypatch.setenv("REINS_DISABLE_PROGRESS_GUARD", "1")
    max_steps = 5
    project = _project(tmp_path)
    client = from_test_sequence(
        ['{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}'] * 20,
        protocol_mode="text_json",
    )
    loop, context = _build_loop(project, client, max_steps=max_steps)

    events = list(loop.run_stream(context))

    started = [e for e in events if isinstance(e, ToolExecutionStarted)]
    pause_events = [e for e in events if isinstance(e, SegmentPaused)]
    #  超出 max_steps 的工具调用照常执行，且不产生任何暂停
    assert len(started) == 20
    assert not pause_events


def test_max_steps_does_not_stop_the_loop(tmp_path: Path) -> None:
    # max_steps=1 不再截断运行，脚本里的两次工具调用都会派发
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
        ],
        protocol_mode="text_json",
    )
    loop, context = _build_loop(project, client, max_steps=1)

    events = list(loop.run_stream(context))

    started = [e for e in events if isinstance(e, ToolExecutionStarted)]
    pause_events = [e for e in events if isinstance(e, SegmentPaused)]
    assert len(started) == 2
    assert not pause_events


def test_progress_guard_switch_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    # 开关关闭提醒和暂停，实际观察仍写入运行事实
    from runtime.progress import ProgressGuard

    monkeypatch.delenv("REINS_DISABLE_PROGRESS_GUARD", raising=False)
    assert ProgressGuard({}).enabled is True

    monkeypatch.setenv("REINS_DISABLE_PROGRESS_GUARD", "1")
    assert ProgressGuard({}).enabled is False


def test_convergence_reminder_suppressed_when_guard_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REINS_DISABLE_PROGRESS_GUARD", "1")
    loop = AgentLoop(tmp_path / "data")
    assert loop.progress.model_notice() == ""


def test_no_progress_evidence_survives_guard_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 【进展】【关闭检测】保留实际观察，停止主动提醒和暂停
    monkeypatch.setenv("REINS_DISABLE_PROGRESS_GUARD", "1")
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"ok"}',
        ],
        protocol_mode="text_json",
    )
    loop, context = _build_loop(project, client)

    events = list(loop.run_stream(context))

    pause_events = [e for e in events if isinstance(e, SegmentPaused)]
    assert not pause_events
    assert loop.state is State.DONE
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    assert any(row.get("event") == "progress:observed" for row in facts)
    assert not any(row.get("event") == "progress:no_progress" for row in facts)
