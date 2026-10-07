from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from runtime.ledger import LedgerEvent, LedgerStore, new_ledger_event
from runtime.ledger_writer import (
    CHECKPOINT_SAVED,
    CONTEXT_SEGMENTS_RECORDED,
    GOAL_CONFIRMATION_DECIDED,
    LEDGER_EVENT_NAMES,
    MINIMUM_PAYLOAD_FIELDS,
    MODEL_REQUESTED,
    RUN_LIFECYCLE_CHANGED,
    SUMMARY_UPDATED,
    TASK_FOCUS_CHANGED,
    TOOL_COMPLETED,
    TOOL_REQUESTED,
    LedgerWriter,
)


@pytest.mark.parametrize(
    "change", [{"accepted": "yes"}, {"expected_revision": True}, {"evidence": []}]
)
def test_invalid_confirmation_never_reaches_ledger(
    tmp_path: Path, change: dict[str, object]
) -> None:
    """类型错误的用户决定不落事实源；传参：目录和无效字段；返回：无。"""
    store = LedgerStore(tmp_path)
    payload = {
        "action_id": "action",
        "question_id": "question",
        "source_input_id": "input",
        "expected_revision": 1,
        "branch_anchor": "anchor",
        "branch_id": None,
        "evidence_digest": "hash",
        "evidence": [{"kind": "answer", "reference": "report"}],
        "accepted": True,
    }
    event = new_ledger_event(
        GOAL_CONFIRMATION_DECIDED,
        "decision",
        "user_action",
        {**payload, **change},
        session_id="session",
        task_id="goal",
        run_id="run",
    )
    with pytest.raises(ValueError):
        LedgerWriter(store).record_once(event)
    assert store.read_events() == []


def test_ledger_writer_records_event_with_generated_event_id(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path)
    writer = LedgerWriter(store, source="unit-test")

    event = writer.record_tool_completed(
        "file_read",
        "call-1",
        "ok",
        task_id="task-1",
        session_id="session-1",
        run_id="run-1",
    )

    assert event.event == TOOL_COMPLETED
    assert event.event_id.startswith("ledger-")
    assert event.source == "unit-test"
    assert event.payload == {
        "tool_name": "file_read",
        "call_id": "call-1",
        "status": "ok",
    }
    assert store.read_events() == [event]


def test_ledger_writer_generates_unique_event_ids(tmp_path: Path) -> None:
    writer = LedgerWriter(LedgerStore(tmp_path))

    first = writer.record_tool_requested("file_read", "call-1", {"path": "a"})
    second = writer.record_tool_completed("file_read", "call-1", "ok")

    assert first.event_id.startswith("ledger-")
    assert second.event_id.startswith("ledger-")
    assert first.event_id != second.event_id


def test_ledger_writer_no_longer_owns_message_events() -> None:
    """消息 owner 收敛到 Session Store 后，Ledger 不再定义 turn/response 事件。"""
    assert not hasattr(LedgerWriter, "record_turn")
    assert not hasattr(LedgerWriter, "record_model_responded")
    assert "turn.recorded" not in LEDGER_EVENT_NAMES
    assert "model.responded" not in LEDGER_EVENT_NAMES


def test_ledger_writer_event_contract_lists_all_phase3_events() -> None:
    assert LEDGER_EVENT_NAMES == frozenset(MINIMUM_PAYLOAD_FIELDS)
    assert MINIMUM_PAYLOAD_FIELDS[MODEL_REQUESTED] == frozenset(
        {"provider", "model", "context_segment_names"}
    )
    assert MINIMUM_PAYLOAD_FIELDS[TASK_FOCUS_CHANGED] == frozenset(
        {"previous_task_id", "next_task_id", "reason"}
    )


@pytest.mark.parametrize(
    ("method", "expected_event", "expected_payload"),
    [
        (
            lambda writer: writer.record_model_requested(
                "openai", "gpt-test", ["identity", "conversation"]
            ),
            MODEL_REQUESTED,
            {
                "provider": "openai",
                "model": "gpt-test",
                "context_segment_names": ["identity", "conversation"],
            },
        ),
        (
            lambda writer: writer.record_tool_requested(
                "file_read", "call-1", {"path": "README.md"}
            ),
            TOOL_REQUESTED,
            {
                "tool_name": "file_read",
                "call_id": "call-1",
                "args": {"path": "README.md"},
            },
        ),
        (
            lambda writer: writer.record_tool_completed("file_read", "call-1", "ok"),
            TOOL_COMPLETED,
            {"tool_name": "file_read", "call_id": "call-1", "status": "ok"},
        ),
        (
            lambda writer: writer.record_checkpoint_saved(
                "checkpoint-1", {"state": "paused"}, "approval_wait"
            ),
            CHECKPOINT_SAVED,
            {
                "checkpoint_id": "checkpoint-1",
                "state": {"state": "paused"},
                "reason": "approval_wait",
            },
        ),
        (
            lambda writer: writer.record_summary_updated("progress", "summary text"),
            SUMMARY_UPDATED,
            {"summary_kind": "progress", "content": "summary text"},
        ),
        (
            lambda writer: writer.record_task_focus_changed(
                None, "task-2", "task switch"
            ),
            TASK_FOCUS_CHANGED,
            {
                "previous_task_id": None,
                "next_task_id": "task-2",
                "reason": "task switch",
            },
        ),
        (
            lambda writer: writer.record_context_segments([{"name": "identity"}]),
            CONTEXT_SEGMENTS_RECORDED,
            {"segments": [{"name": "identity"}]},
        ),
        (
            lambda writer: writer.record_run_lifecycle_changed("done"),
            RUN_LIFECYCLE_CHANGED,
            {"status": "done"},
        ),
    ],
)
def test_ledger_writer_records_each_business_event(
    tmp_path: Path,
    method: Callable[[LedgerWriter], LedgerEvent],
    expected_event: str,
    expected_payload: dict[str, object],
) -> None:
    writer = LedgerWriter(LedgerStore(tmp_path), id_factory=lambda: "ledger-fixed")

    event = method(writer)

    assert event.event == expected_event
    assert event.event_id == "ledger-fixed"
    assert event.payload == expected_payload


def test_ledger_writer_rejects_missing_payload_fields(tmp_path: Path) -> None:
    writer = LedgerWriter(LedgerStore(tmp_path))

    with pytest.raises(ValueError, match="missing required fields: status"):
        writer._append_event(
            TOOL_COMPLETED, {"tool_name": "file_read", "call_id": "c1"}
        )


def test_ledger_writer_rejects_blank_payload_text(tmp_path: Path) -> None:
    writer = LedgerWriter(LedgerStore(tmp_path))

    with pytest.raises(ValueError, match="tool_name must be non-empty"):
        writer.record_tool_completed(" ", "call-1", "ok")


def test_ledger_writer_rejects_none_for_non_nullable_fields(tmp_path: Path) -> None:
    writer = LedgerWriter(LedgerStore(tmp_path))

    with pytest.raises(ValueError, match="segments must be present"):
        writer.record_context_segments(None)


def test_ledger_writer_allows_empty_focus_target(tmp_path: Path) -> None:
    writer = LedgerWriter(LedgerStore(tmp_path))

    event = writer.record_task_focus_changed("task-1", None, "clear focus")

    assert event.payload["previous_task_id"] == "task-1"
    assert event.payload["next_task_id"] is None


def test_ledger_writer_rejects_unknown_event(tmp_path: Path) -> None:
    writer = LedgerWriter(LedgerStore(tmp_path))

    with pytest.raises(ValueError, match="unsupported ledger event"):
        writer._append_event("unknown.event", {})


def test_ledger_writer_rejects_empty_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source must be non-empty"):
        LedgerWriter(LedgerStore(tmp_path), source=" ")


def test_ledger_writer_surfaces_append_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LedgerStore(tmp_path)
    writer = LedgerWriter(store)

    def fail_append(_event: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(store, "append", fail_append)

    with pytest.raises(OSError, match="disk full"):
        writer.record_tool_completed("file_read", "call-1", "ok")
