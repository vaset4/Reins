from __future__ import annotations
from scripts.testing.llm import from_test_stub

import json
from pathlib import Path

from app.repl.slash_commands import (
    ReplState,
    SlashCommandContext,
    create_default_registry,
)
from app.run_task import inspect_resume
from context.engine import build_context_sections_for_tests
from frontends.tui.data.task_query import TaskQueryService
from runtime.agent_loop import AgentLoop, State
from runtime.ledger import LedgerStore
from runtime.lease import Lease
from runtime.session_state import SessionState, SessionStateStore
from runtime.types import Trigger
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from triggers.user import make_run_context as make_user_context


def test_plain_session_run_updates_session_state_without_formal_task(
    tmp_path: Path,
) -> None:
    context = make_user_context("plain chat", data_root=tmp_path, formal_task=False)

    assert (
        AgentLoop(tmp_path, llm_client=from_test_stub("会话答复")).run(context)
        is State.DONE
    )

    state = SessionStateStore(tmp_path).load(context.session_id)
    assert state is not None
    assert state.last_run_id == context.run_id
    assert state.last_run_status == "done"
    assert state.focus_task_id is None
    assert state.compatibility_task_id == context.compatibility_task_id
    assert state.writeback_targets["session"] is True
    assert "compatibility_task" not in state.writeback_targets
    assert state.writeback_targets["task"] is False

    formal_tasks = [
        record for record in TaskStore(tmp_path).list_tasks() if not record.is_inbox
    ]
    assert formal_tasks == []


def test_formal_task_run_updates_session_state_and_task_writeback(
    tmp_path: Path,
) -> None:
    context = make_user_context("pytest work", data_root=tmp_path, formal_task=True)

    assert (
        AgentLoop(tmp_path, llm_client=from_test_stub("会话答复")).run(context)
        is State.DONE
    )

    state = SessionStateStore(tmp_path).load(context.session_id)
    assert state is not None
    assert state.last_run_id == context.run_id
    assert state.last_run_status == "done"
    assert state.focus_task_id == context.task_id
    assert state.writeback_targets["task"] is True
    assert TaskStore(tmp_path).require_task(str(context.task_id)).status == "active"


def test_task_clear_writes_session_focus_unset(tmp_path: Path) -> None:
    project = tmp_path / "project"
    data_root = project / ".reins" / "data"
    project.mkdir()
    store = TaskStore(data_root)
    record = store.create_task("focused")
    state = ReplState(
        current_task_id=record.task_id, compatibility_task_id="chat-compat"
    )
    ctx = SlashCommandContext(
        repl_state=state,
        store=store,
        registry=build_tool_registry(repo_root=project, data_root=data_root),
        llm_client=None,
        project_root=project,
        data_root=data_root,
        prompt_fn=lambda _prompt: "",
    )

    result = create_default_registry().dispatch("/task clear", ctx)

    session_state = SessionStateStore(data_root).load(state.session_id)
    assert result is not None
    assert state.current_task_id is None
    assert session_state is not None
    assert session_state.focus_task_id is None
    assert session_state.compatibility_task_id == "chat-compat"
    focus_events = [
        event
        for event in LedgerStore(data_root).read_session_events(state.session_id)
        if event.event == "task.focus_changed"
    ]
    assert focus_events[-1].payload["previous_task_id"] == record.task_id
    assert focus_events[-1].payload["next_task_id"] is None


def test_context_minimal_identity_reads_session_state(tmp_path: Path) -> None:
    context = make_user_context("plain chat", data_root=tmp_path, formal_task=False)
    AgentLoop(tmp_path, llm_client=from_test_stub("会话答复")).run(context)

    sections = build_context_sections_for_tests(
        None,
        Trigger.USER,
        {"message": "new topic"},
        lease=Lease(),
        session_id=context.session_id,
        run_id="run-new",
        data_root=tmp_path,
    )

    assert [section.name for section in sections] == [
        "identity",
        "trigger_payload",
        "tools_output",
    ]
    identity = json.loads(sections[0].content)
    assert identity["session_state"]["last_run_id"] == context.run_id
    assert identity["context_decision"]["reason"] == "no_focus_task"


def test_tui_and_resume_read_session_state_summary(tmp_path: Path) -> None:
    project = tmp_path / "project"
    data_root = project / ".reins" / "data"
    project.mkdir()
    context = make_user_context(
        "plain chat",
        data_root=data_root,
        formal_task=False,
        session_id="session-e2e",
    )
    assert (
        AgentLoop(data_root, llm_client=from_test_stub("会话答复")).run(context)
        is State.DONE
    )

    tui = TaskQueryService(data_root=data_root)
    overview = tui.read_session_overview()
    lines = tui.render_session_overview_lines()
    resumed = inspect_resume(
        checkpoint_id="session-e2e", project_root=project, data_root=data_root
    )

    assert overview.sessions[0].session_id == "session-e2e"
    assert any("session-e2e" in line for line in lines)
    assert "session_summary:" in resumed.output
    assert context.run_id in resumed.output


def test_session_state_round_trips_toolsets_and_prompt_hash(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path)
    state = store.load("session-missing") or SessionState(session_id="session-missing")
    state.toolsets_enabled = ["file"]
    state.toolsets_disabled = ["terminal"]
    state.toolsets_updated_at = "2026-05-27T00:00:00Z"
    state.stable_prompt_hash = "sha256:abc"
    state.stable_prompt_updated_at = "2026-05-27T00:00:01Z"

    store.save(state)
    loaded = store.load("session-missing")
    with store.database.snapshot() as source:
        raw = source.get("session_state", state.session_id)

    assert loaded is not None
    assert loaded.toolsets_enabled == ["file"]
    assert loaded.toolsets_disabled == ["terminal"]
    assert loaded.stable_prompt_hash == "sha256:abc"
    assert raw is not None and "stable_prompt" not in raw


def test_session_state_load_rejects_corrupt_json(tmp_path: Path) -> None:
    import pytest

    store = SessionStateStore(tmp_path)
    path = store.save(SessionState("session-corrupt"))
    path.write_bytes(b"broken\n")
    with pytest.raises(ValueError):
        store.load("session-corrupt")
