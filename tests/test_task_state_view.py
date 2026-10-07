from __future__ import annotations

from runtime.ledger import new_ledger_event
from runtime.task_state_view import TaskStateView, build_task_state_view


def test_task_state_view_no_longer_rebuilds_messages() -> None:
    """消息 owner 收敛到 Session Store 后，任务视图不再重建对话内容。"""
    assert not hasattr(TaskStateView(), "last_user_turn")


def test_task_state_view_rebuilds_latest_summary() -> None:
    events = [
        new_ledger_event(
            "summary.updated",
            "evt-summary-1",
            "test",
            {"summary_kind": "progress", "content": "started"},
            task_id="task-1",
            ts="2026-07-01T00:00:00Z",
        ),
        new_ledger_event(
            "summary.updated",
            "evt-summary-2",
            "test",
            {"summary_kind": "progress", "content": "finished"},
            task_id="task-1",
            ts="2026-07-01T00:00:01Z",
        ),
    ]

    view = build_task_state_view(events)

    assert view.summary == "finished"
    assert view.summary_kind == "progress"
    assert view.summary_event_id == "evt-summary-2"


def test_task_state_view_rebuilds_summary_layers() -> None:
    view = build_task_state_view(
        [
            new_ledger_event(
                "summary.updated",
                "evt-intent",
                "test",
                {"summary_kind": "intent", "content": "do the thing"},
                task_id="task-1",
            ),
            new_ledger_event(
                "summary.updated",
                "evt-progress",
                "test",
                {"summary_kind": "progress", "content": "half done"},
                task_id="task-1",
            ),
            new_ledger_event(
                "summary.updated",
                "evt-resume",
                "test",
                {"summary_kind": "resume_hint", "content": "continue here"},
                task_id="task-1",
            ),
            new_ledger_event(
                "summary.updated",
                "evt-summary",
                "test",
                {"summary_kind": "summary", "content": "aggregate"},
                task_id="task-1",
            ),
        ]
    )

    assert view.intent == "do the thing"
    assert view.progress == "half done"
    assert view.resume_hint == "continue here"
    assert view.summary == "aggregate"
    assert view.summary_kind == "summary"


def test_task_state_view_rebuilds_focus_change() -> None:
    view = build_task_state_view(
        [
            new_ledger_event(
                "task.focus_changed",
                "evt-focus-1",
                "test",
                {
                    "previous_task_id": "task-old",
                    "next_task_id": "task-new",
                    "reason": "user selected task",
                },
                session_id="session-1",
                ts="2026-07-01T00:00:00Z",
            )
        ]
    )

    assert view.task_id == "task-new"
    assert view.previous_task_id == "task-old"
    assert view.focused_task_id == "task-new"
    assert view.focus_reason == "user selected task"
    assert view.focus_event_id == "evt-focus-1"


def test_task_state_view_accepts_mapping_events() -> None:
    view = build_task_state_view(
        [
            {
                "type": "ledger_event",
                "event": "summary.updated",
                "ts": "2026-07-01T00:00:00Z",
                "event_id": "evt-1",
                "task_id": "task-1",
                "source": "test",
                "payload": {"summary_kind": "intent", "content": "from mapping"},
            }
        ]
    )

    assert view.intent == "from mapping"


def test_task_state_view_empty_events_returns_empty_view() -> None:
    assert build_task_state_view([]) == TaskStateView()


def test_task_state_view_ignores_unrelated_events_but_counts_them() -> None:
    view = build_task_state_view(
        [
            new_ledger_event(
                "tool.completed",
                "evt-tool-1",
                "test",
                {"tool_name": "noop", "call_id": "call-1", "status": "ok"},
                task_id="task-1",
                ts="2026-07-01T00:00:00Z",
            )
        ]
    )

    assert view.task_id == "task-1"
    assert view.intent == ""
    assert view.event_count == 1
    assert view.seen_event_ids == ("evt-tool-1",)
