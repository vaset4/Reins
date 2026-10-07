from __future__ import annotations

from pathlib import Path

import pytest

from frontends.tui.data.task_query import TaskQueryService
from memory.store import MemoryStore
from runtime.checkpoint import Checkpoint, checkpoint_to_ledger_state, save_checkpoint
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore
from skills.store import SkillStore, build_skill_markdown
from tasks.store import TaskStore
from runtime.persistence import RuntimeStore, SourceSnapshot
from tools.todo_tool import add_todo, update_todo


def test_list_tasks_page_queries_index_db_with_paging(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    for idx in range(35):
        task_id = f"2026-05-06-{idx:02d}"
        store.create_task(f"goal {idx}", task_id=task_id)
    service = TaskQueryService(data_root=tmp_path)

    page1 = service.list_tasks_page(page=1, page_size=30)
    page2 = service.list_tasks_page(page=2, page_size=30)

    assert len(page1.items) == 30
    assert page1.has_more is True
    assert len(page2.items) == 5
    assert page2.has_more is False
    assert page1.warnings == []


def test_list_tasks_page_hides_inbox_compatibility_tasks(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    store.create_task("plain chat", task_id="chat-compat", is_inbox=True)
    store.create_task("formal task", task_id="formal-task")
    service = TaskQueryService(data_root=tmp_path)

    page = service.list_tasks_page(page=1, page_size=30)

    assert [item.task_id for item in page.items] == ["formal-task"]


def test_list_tasks_page_rebuilds_when_index_is_missing(tmp_path: Path) -> None:
    """丢失派生索引后从原件重建正常分页；参数：隔离根；返回：无。"""
    store = TaskStore(tmp_path)
    store.create_task("alpha", task_id="2026-05-06-01")
    store.create_task("beta", task_id="2026-05-06-02")
    service = TaskQueryService(data_root=tmp_path)
    (tmp_path / "index.sqlite").unlink()

    page = service.list_tasks_page(page=1, page_size=30)

    assert len(page.items) == 2
    assert page.warnings == []


def test_task_page_failure_is_explicit_and_never_an_empty_success(
    tmp_path, monkeypatch
):
    """索引不可用时明确失败，不能偷偷切换读取策略；参数：隔离根与故障器；返回：无。"""
    import sqlite3

    TaskStore(tmp_path).create_task("visible goal", task_id="goal")
    service = TaskQueryService(tmp_path)

    def unavailable(_store):
        """模拟索引读取设施故障；传参：源服务；返回：不返回。"""
        raise sqlite3.OperationalError("index unavailable")

    monkeypatch.setattr(RuntimeStore, "open_index_connection", unavailable)
    with pytest.raises(sqlite3.OperationalError, match="index unavailable"):
        service.list_tasks_page(page=1)


def test_task_page_loads_only_selected_originals(tmp_path, monkeypatch):
    """目录分页只展开当前页的目标正文；参数：隔离根与读取计数器；返回：无。"""
    store = TaskStore(tmp_path)
    for index in range(8):
        store.create_task(f"goal-{index}:" + "长正文" * 300, task_id=f"goal-{index}")
    service = TaskQueryService(tmp_path)
    read = SourceSnapshot.get
    selected = []

    def capture(source, kind, identity):
        """记录当前页展开的真实目标；传参：快照和身份；返回：原件。"""
        if kind == "task":
            selected.append(identity)
        return read(source, kind, identity)

    monkeypatch.setattr(SourceSnapshot, "get", capture)
    page = service.list_tasks_page(page=2, page_size=3)
    assert len(page.items) == 3 and page.has_more
    assert selected == [item.task_id for item in page.items]
    assert all(item.goal.endswith("长正文" * 300) for item in page.items)


def test_read_task_detail_returns_todo_lease_watchdog(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-06-detail"
    session_id = "session-detail"
    run_id = "run-detail"
    store = TaskStore(tmp_path)
    store.create_task("detail", task_id=task_id)
    add_todo(task_id, "pending", data_root=tmp_path)
    done = add_todo(task_id, "done", data_root=tmp_path)
    ongoing = add_todo(task_id, "keep going", data_root=tmp_path)
    update_todo(task_id, done.idx, "done", data_root=tmp_path)
    update_todo(task_id, ongoing.idx, "in_progress", data_root=tmp_path)
    facts = RunFactStore(tmp_path)
    facts.append(
        {
            "event": "run:start",
            "session_id": session_id,
            "run_id": run_id,
            "task_id": task_id,
            "ts": "2026-05-06T00:00:01Z",
            "trigger": "resume",
            "lease_summary": {
                "trigger": "resume",
                "max_steps": 30,
                "max_tokens": 200000,
                "expires_at": "lease_segment_end",
                "capabilities": {"terminal": True},
            },
        }
    )
    facts.append(
        {
            "event": "tool:response",
            "session_id": session_id,
            "run_id": run_id,
            "task_id": task_id,
            "ts": "2026-05-06T00:00:02Z",
            "tool": {
                "name": "file_read",
                "status": "error",
                "error_category": "not_found",
            },
        }
    )
    facts.append(
        {
            "event": "tool:response",
            "session_id": session_id,
            "run_id": run_id,
            "task_id": task_id,
            "ts": "2026-05-06T00:00:03Z",
            "tool": {
                "name": "file_read",
                "status": "error",
                "error_category": "not_found",
            },
        }
    )
    facts.append(
        {
            "event": "llm:response",
            "session_id": session_id,
            "run_id": run_id,
            "task_id": task_id,
            "ts": "2026-05-06T00:00:04Z",
            "summary": {
                "error": "model_error",
                "has_final": False,
                "has_run_tools": False,
            },
        }
    )
    facts.append(
        {
            "event": "llm:response",
            "session_id": session_id,
            "run_id": run_id,
            "task_id": task_id,
            "ts": "2026-05-06T00:00:05Z",
            "summary": {
                "error": "model_error",
                "has_final": False,
                "has_run_tools": False,
            },
        }
    )
    facts.append(
        {
            "event": "run:lifecycle",
            "session_id": session_id,
            "run_id": run_id,
            "task_id": task_id,
            "ts": "2026-05-06T00:00:06Z",
            "lifecycle": "paused",
            "reason": "lease step limit reached",
        }
    )

    service = TaskQueryService(data_root=tmp_path)
    detail = service.read_task_detail(task_id)

    assert detail.todo == ["- [ ] pending", "in_progress: keep going"]
    assert detail.lease.trigger == "resume"
    assert detail.watchdog.tool_failures == 2
    assert detail.watchdog.llm_failures == 2
    assert detail.watchdog.paused is True
    assert any("lifecycle | paused" in row for row in detail.trajectory_tail)


def test_memory_and_skill_lists_are_exposed(tmp_path: Path) -> None:
    service = TaskQueryService(data_root=tmp_path)
    memory_id = MemoryStore(tmp_path).create_memory("rule", "always test", ["qa"])
    SkillStore(tmp_path).create_skill(
        "skill-1",
        build_skill_markdown(
            name="Skill 1",
            body="body",
            trigger_keywords=["Skill one description."],
        ),
        meta={},
    )

    memories = service.list_active_memories()
    skills = service.list_active_skills()

    assert any(memory_id in row for row in memories)
    assert any("skill-1" in row for row in skills)


def test_session_overview_reads_recent_runs_and_recovery_points(tmp_path: Path) -> None:
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id="chat-compat",
            session_id="session-tui",
            run_id="run-tui",
            compatibility_task_id="chat-compat",
            segment_id="user-1",
            state="PAUSED",
            reason="paused",
        )
    )
    RunFactStore(tmp_path).append_checkpoint_ref(
        session_id="session-tui",
        run_id="run-tui",
        task_id=None,
        focus_task_id=None,
        compatibility_task_id="chat-compat",
        segment_id="user-1",
        checkpoint=checkpoint,
    )
    LedgerWriter(
        LedgerStore(tmp_path), source="tests.tui_data_query"
    ).record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        checkpoint.reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )

    service = TaskQueryService(data_root=tmp_path)
    overview = service.read_session_overview()
    lines = service.render_session_overview_lines()

    assert overview.recent_runs[0].run_id == "run-tui"
    assert overview.recovery_points[0].checkpoint_id == checkpoint.checkpoint_id
    assert any("run-tui" in line for line in lines)
    assert any(checkpoint.checkpoint_id in line for line in lines)


def test_read_run_detail_summarizes_evidence_and_expands_raw(tmp_path: Path) -> None:
    """由当前证据引用展开实际原件，折叠视图仍保留概要；参数：隔离目录；返回：无。"""
    session_id = "session-detail"
    run_id = "run-detail"
    evidence = RunEvidenceStore(tmp_path)
    request_path = evidence.write_record(
        session_id=session_id,
        run_id=run_id,
        kind="model_request",
        source_id="request-1",
        payload={
            "protocol_mode": "native_tool_calls",
            "render_text_to_model": "system\nuser",
            "prompt_context": {
                "stage": "plan",
                "context_summary": "intent: inspect\nprogress: running",
                "system_reminder": False,
            },
            "request": {
                "messages": [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "hello"},
                ]
            },
        },
    )
    response_path = evidence.write_record(
        session_id=session_id,
        run_id=run_id,
        kind="model_response",
        source_id="response-1",
        payload={"response": {"ok": True, "text": "hi", "tool_calls": []}},
    )
    parsed_path = evidence.write_record(
        session_id=session_id,
        run_id=run_id,
        kind="parsed_plan",
        source_id="plan-1",
        payload={"success": True, "has_final": True, "has_run_tools": False},
    )
    facts = RunFactStore(tmp_path)
    # 1. 【运行详情】【原件关联】事件和三种证据共享会话与运行身份，概要与展开都从这些真实记录读取
    for fact in (
        {"event": "run:start", "ts": "2026-05-23T00:00:00Z"},
        {"event": "context:built", "summary": {"tool_history_count": 2}},
        {
            "event": "llm:response",
            "summary": {
                "observation": {
                    "stage": "plan",
                    "provider": "openai",
                    "model": "gpt-test",
                    "elapsed_ms": 12,
                    "attempt_count": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                },
                "evidence": {
                    "model_request": request_path,
                    "model_response": response_path,
                    "parsed_plan": parsed_path,
                },
            },
        },
    ):
        facts.append({"session_id": session_id, "run_id": run_id, **fact})
    service = TaskQueryService(data_root=tmp_path)

    collapsed = service.read_run_detail(session_id, run_id)
    expanded = service.read_run_detail(session_id, run_id, expand_raw=True)

    assert collapsed.context_summary == [
        "context_events: 1",
        "last_tool_history_count: 2",
        "stage: plan",
        "system_reminder: False",
        "render_text_chars: 11",
        "context_summary: intent: inspect",
        "context_summary: progress: running",
    ]
    assert "messages: 2" in collapsed.model_input_summary
    assert "roles: system=1, user=1" in collapsed.model_input_summary
    assert "response_ok: True" in collapsed.model_output_summary
    assert "has_final: True" in collapsed.model_output_summary
    assert any("provider: openai" in line for line in collapsed.call_summary)
    assert collapsed.sections[0].raw_preview == []
    assert any("messages" in line for line in expanded.sections[0].raw_preview)
