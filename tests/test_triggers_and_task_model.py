from __future__ import annotations

from pathlib import Path

import pytest

from runtime.checkpoint import Checkpoint, checkpoint_to_ledger_state, save_checkpoint
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.workspaces import WorkspaceStore
from schedules.store import ScheduleStore
from runtime.types import Trigger
from tasks.store import TaskStore
from tools.todo_tool import add_todo, list_todos, update_todo
from triggers.cron import make_run_context as make_cron_context
from triggers.resume import make_run_context as make_resume_context
from triggers.user import make_run_context as make_user_context


def _record_checkpoint(data_root: Path, checkpoint: Checkpoint) -> None:
    saved = save_checkpoint(checkpoint)
    RunFactStore(data_root).append_checkpoint_ref(
        session_id=saved.session_id or "session-test",
        run_id=saved.run_id or "run-test",
        task_id=saved.task_id if saved.compatibility_task_id is None else None,
        focus_task_id=saved.focus_task_id,
        compatibility_task_id=saved.compatibility_task_id,
        segment_id=saved.segment_id,
        checkpoint=saved,
    )
    LedgerWriter(
        LedgerStore(data_root), source="tests.triggers"
    ).record_checkpoint_saved(
        saved.checkpoint_id,
        checkpoint_to_ledger_state(saved),
        saved.reason,
        task_id=saved.task_id,
        session_id=saved.session_id,
        run_id=saved.run_id,
    )


def test_user_trigger_creates_regular_and_inbox_tasks(tmp_path: Path) -> None:
    regular = make_user_context("build task", data_root=tmp_path)
    inbox = make_user_context("quick chat", data_root=tmp_path, inbox=True)

    assert regular.trigger is Trigger.USER
    assert regular.task_id is not None
    # user turn 落唯一消息 owner，旧 conversation.jsonl 不再被写
    assert SessionMessageStore(tmp_path).materialize(regular.session_id).messages
    assert not (tmp_path / "tasks" / regular.task_id / "conversation.jsonl").exists()
    assert inbox.task_id is None
    assert inbox.compatibility_task_id is not None
    assert TaskStore(tmp_path).require_task(inbox.compatibility_task_id).is_inbox
    assert inbox.payload["inbox_compatibility"] is True


def test_user_trigger_plain_message_has_session_run_without_formal_task(
    tmp_path: Path,
) -> None:
    context = make_user_context("ordinary chat", data_root=tmp_path, formal_task=False)

    assert context.task_id is None
    assert context.focus_task_id is None
    assert context.compatibility_task_id is not None
    assert context.storage_task_id == context.compatibility_task_id
    assert context.session_id.startswith("session-")
    assert context.run_id.startswith("run-")


def test_task_store_text_files_and_promote_inbox(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    task = store.create_task("quick", task_id="2026-05-04-10", is_inbox=True)
    store.append_journal(task.task_id, "started")
    store.update_summary(task.task_id, "summary")

    promoted = store.promote_inbox_to_task(task.task_id)
    assert promoted.is_inbox is False
    assert store.read_summary(promoted.task_id) == "summary"
    assert "started" in store.read_journal(promoted.task_id)


def test_todo_tool_add_update_list(tmp_path: Path) -> None:
    TaskStore(tmp_path).create_task("todo", task_id="2026-05-04-11")

    item = add_todo("2026-05-04-11", "write store", data_root=tmp_path)
    updated = update_todo("2026-05-04-11", item.idx, "done", data_root=tmp_path)

    assert updated.status == "done"
    assert [
        todo.content for todo in list_todos("2026-05-04-11", "done", data_root=tmp_path)
    ] == ["write store"]


def test_cron_trigger_requires_permanent_grants(tmp_path: Path) -> None:
    workspace = WorkspaceStore(tmp_path).register(tmp_path)
    schedules = ScheduleStore(tmp_path)
    schedules.create_schedule(
        "daily",
        "0 3 * * *",
        workspace_id=workspace.workspace_id,
        target_task_id="2026-05-04-12",
        required_permanent_grants=(),
    )
    context = make_cron_context("daily", data_root=tmp_path)
    assert context.trigger is Trigger.CRON
    assert context.capability_lease.trigger == "cron"
    assert context.capability_lease.task_id == "2026-05-04-12"
    assert context.capability_lease.capabilities["schedule"] == {
        "required_permanent_grants": []
    }

    # 已保存授权配置损坏时，定时触发仍须拒绝，不能自动当作空授权
    with schedules._db.snapshot() as source:
        daily = source.get("schedule", "daily")
    assert daily is not None
    with schedules._db.transaction() as batch:
        batch.put(
            "schedule",
            "bad",
            {**daily, "schedule_id": "bad", "required_permanent_grants": "invalid"},
            workspace_id=workspace.workspace_id,
        )
    with pytest.raises((ValueError, TypeError)):
        make_cron_context("bad", data_root=tmp_path)


def test_resume_trigger_reads_latest_checkpoint(tmp_path: Path) -> None:
    _record_checkpoint(
        tmp_path,
        Checkpoint(
            task_id="2026-05-04-13",
            session_id="session-13",
            run_id="run-13",
            focus_task_id="2026-05-04-13",
            segment_id="user-1",
            state="PARSING",
            reason="test_checkpoint",
        ),
    )

    context = make_resume_context("2026-05-04-13", data_root=tmp_path)
    assert context.trigger is Trigger.RESUME
    assert context.capability_lease.trigger == "resume"
    assert context.capability_lease.task_id == "2026-05-04-13"
    assert context.parent_segment_id == "user-1"


def test_resume_trigger_can_resume_by_session_id(tmp_path: Path) -> None:
    _record_checkpoint(
        tmp_path,
        Checkpoint(
            task_id="chat-compat",
            session_id="session-abc",
            run_id="run-old",
            compatibility_task_id="chat-compat",
            segment_id="user-1",
            state="PARSING",
            reason="test_checkpoint",
        ),
    )

    context = make_resume_context(data_root=tmp_path, session_id="session-abc")

    assert context.trigger is Trigger.RESUME
    assert context.task_id is None
    assert context.compatibility_task_id == "chat-compat"
    assert context.session_id == "session-abc"
    assert context.capability_lease.trigger == "resume"
    assert context.capability_lease.task_id == "chat-compat"


def test_resume_trigger_preserves_formal_task_when_resuming_by_session(
    tmp_path: Path,
) -> None:
    _record_checkpoint(
        tmp_path,
        Checkpoint(
            task_id="2026-05-09-formal",
            session_id="session-formal",
            run_id="run-formal",
            focus_task_id="2026-05-09-formal",
            segment_id="user-1",
            state="PARSING",
            reason="test_checkpoint",
        ),
    )

    context = make_resume_context(data_root=tmp_path, session_id="session-formal")

    assert context.task_id == "2026-05-09-formal"
    assert context.focus_task_id == "2026-05-09-formal"
    assert context.payload["previous_run_id"] == "run-formal"


def test_resume_trigger_can_resume_by_source_run_id(tmp_path: Path) -> None:
    _record_checkpoint(
        tmp_path,
        Checkpoint(
            task_id="chat-compat",
            session_id="session-abc",
            run_id="run-old",
            compatibility_task_id="chat-compat",
            segment_id="user-1",
            state="PARSING",
            reason="test_checkpoint",
        ),
    )

    context = make_resume_context(data_root=tmp_path, source_run_id="run-old")

    assert context.task_id is None
    assert context.compatibility_task_id == "chat-compat"
    assert context.payload["previous_run_id"] == "run-old"
