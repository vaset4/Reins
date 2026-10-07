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
from runtime.lease import Lease
from runtime.session_state import SessionStateStore
from runtime.types import Trigger
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from triggers.user import make_run_context as make_user_context


def test_dod_7_plain_session_writeback_stays_out_of_formal_tasks(
    tmp_path: Path,
) -> None:
    context = make_user_context(
        "quick exploratory chat",
        data_root=tmp_path,
        formal_task=False,
        session_id="session-dod7-plain",
    )

    assert (
        AgentLoop(tmp_path, llm_client=from_test_stub("会话答复")).run(context)
        is State.DONE
    )

    state = SessionStateStore(tmp_path).load("session-dod7-plain")
    assert state is not None
    assert state.last_run_id == context.run_id
    assert state.last_run_status == "done"
    assert state.focus_task_id is None
    assert state.compatibility_task_id == context.compatibility_task_id
    assert state.writeback_targets == {
        "session": True,
        "run_facts": True,
        "task": False,
    }

    formal_tasks = [
        record for record in TaskStore(tmp_path).list_tasks() if not record.is_inbox
    ]
    assert formal_tasks == []


def test_dod_7_formal_task_writeback_and_task_clear_keep_context_minimal(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    data_root = project / ".reins" / "data"
    project.mkdir()

    formal = make_user_context(
        "ship a formal task",
        data_root=data_root,
        formal_task=True,
        session_id="session-dod7-formal",
    )
    assert (
        AgentLoop(data_root, llm_client=from_test_stub("会话答复")).run(formal)
        is State.DONE
    )
    assert formal.task_id is not None

    store = TaskStore(data_root)
    assert store.require_task(formal.task_id).status == "active"
    state = SessionStateStore(data_root).load("session-dod7-formal")
    assert state is not None
    assert state.writeback_targets["task"] is True
    assert state.focus_task_id == formal.task_id

    repl_state = ReplState(
        session_id="session-dod7-formal",
        current_task_id=formal.task_id,
        compatibility_task_id="chat-compat",
    )
    ctx = SlashCommandContext(
        repl_state=repl_state,
        store=store,
        registry=build_tool_registry(repo_root=project, data_root=data_root),
        llm_client=None,
        project_root=project,
        data_root=data_root,
        prompt_fn=lambda _prompt: "",
    )
    result = create_default_registry().dispatch("/task clear", ctx)
    assert result is not None
    assert repl_state.current_task_id is None

    sections = build_context_sections_for_tests(
        None,
        Trigger.USER,
        {"message": "new unrelated topic"},
        lease=Lease(),
        session_id="session-dod7-formal",
        run_id="run-dod7-after-clear",
        focus_task_id=repl_state.current_task_id,
        data_root=data_root,
    )
    assert [section.name for section in sections] == [
        "identity",
        "trigger_payload",
        "tools_output",
    ]
    identity = json.loads(sections[0].content)
    assert identity["context_decision"]["include_task_context"] is False
    assert identity["session_state"]["focus_task_id"] is None


def test_dod_7_resume_tui_and_context_share_session_state(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    data_root = project / ".reins" / "data"
    project.mkdir()
    context = make_user_context(
        "pause-ready chat",
        data_root=data_root,
        formal_task=False,
        session_id="session-dod7-resume",
    )

    assert (
        AgentLoop(data_root, llm_client=from_test_stub("会话答复")).run(context)
        is State.DONE
    )

    resumed = inspect_resume(
        "session-dod7-resume", project_root=project, data_root=data_root
    )
    overview = TaskQueryService(data_root=data_root).read_session_overview()
    sections = build_context_sections_for_tests(
        None,
        Trigger.USER,
        {"message": "continue"},
        lease=Lease(),
        session_id="session-dod7-resume",
        run_id="run-dod7-context",
        data_root=data_root,
    )
    identity = json.loads(sections[0].content)

    assert "source: session" in resumed.output
    assert "session_summary:" in resumed.output
    assert context.run_id in resumed.output
    assert overview.sessions[0].session_id == "session-dod7-resume"
    assert overview.sessions[0].last_run_id == context.run_id
    assert identity["session_state"]["last_run_id"] == context.run_id
    assert (
        identity["session_state"]["compatibility_task_id"]
        == context.compatibility_task_id
    )
