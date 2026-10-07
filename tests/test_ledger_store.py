from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from runtime.ledger import LEDGER_TYPE, LedgerEvent, LedgerStore, new_ledger_event


def test_append_once_serializes_independent_writers(tmp_path: Path) -> None:
    """多个存储实例同时提交同一决定只落一次；传参：临时目录；返回：无。"""
    event = new_ledger_event(
        "decision", "one-action", "user_action", {"accepted": True}
    )

    def append(_: int) -> LedgerEvent:
        """模拟独立宿主实例；传参：并发序号；返回：已提交事件。"""
        return LedgerStore(tmp_path).append_once(event)

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(append, range(24))) == [event] * 24
    store = LedgerStore(tmp_path)
    assert store.read_events() == [event]
    assert store.append_once(replace(event, ts="later")) == event
    with pytest.raises(ValueError, match="different decision"):
        store.append_once(replace(event, payload={"accepted": False}))
    assert store.read_events() == [event]


def test_ledger_waits_for_other_process_writer(tmp_path: Path) -> None:
    """跨进程写者等待提交，发布前没有可恢复的提交记录；参数：根；返回：无。"""
    store = LedgerStore(tmp_path)
    event = new_ledger_event("decision", "action", "user_action", {"accepted": True})
    program = (
        "import sys\nfrom runtime.ledger import LedgerStore,new_ledger_event\n"
        "store=LedgerStore(sys.argv[1])\nprint('ready',flush=True)\n"
        "store.append_once(new_ledger_event('decision','action','user_action',{'accepted':True}))\n"
        "assert len(store.read_events())==1\n"
    )
    child = None
    try:
        with store.database.transaction():
            store.append_once(event)
            child = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-c", program, str(tmp_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert (
                child.stdout is not None and child.stdout.readline().strip() == "ready"
            )
            with pytest.raises(subprocess.TimeoutExpired):
                child.wait(timeout=0.3)
            assert (tmp_path / "commits.jsonl").read_bytes() == b""
        _, error = child.communicate(timeout=10)
        assert child.returncode == 0, error
        assert store.read_events() == [event]
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.communicate()


def test_ledger_store_appends_and_reads_events(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path)
    event = new_ledger_event(
        "turn.recorded",
        "evt-1",
        "test",
        {"role": "user", "content": "hello"},
        task_id="task-1",
        session_id="session-1",
        run_id="run-1",
        ts="2026-07-01T00:00:00Z",
    )

    path = store.append(event)
    rows = store.read_events()

    assert path.name == "events.jsonl"
    assert '"event":"turn.recorded"' in path.read_text(encoding="utf-8")
    assert rows == [event]
    assert rows[0].type == LEDGER_TYPE


def test_ledger_store_accepts_mapping_and_sorts_keys_on_disk(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path)

    store.append(
        {
            "type": LEDGER_TYPE,
            "event": "summary.updated",
            "ts": "2026-07-01T00:00:00Z",
            "event_id": "evt-2",
            "source": "unit-test",
            "task_id": "task-1",
            "payload": {"summary_kind": "progress", "content": "done"},
        }
    )

    event = store.read_events()[0]
    assert event.event == "summary.updated"
    assert event.payload == {"summary_kind": "progress", "content": "done"}


@pytest.mark.parametrize(
    "missing", ["type", "event", "ts", "event_id", "source", "payload"]
)
def test_ledger_event_rejects_missing_required_fields(missing: str) -> None:
    payload: dict[str, object] = {
        "type": LEDGER_TYPE,
        "event": "turn.recorded",
        "ts": "2026-07-01T00:00:00Z",
        "event_id": "evt-1",
        "source": "test",
        "payload": {"role": "user", "content": "hello"},
    }
    del payload[missing]

    with pytest.raises(ValueError, match="missing required fields"):
        LedgerEvent.from_mapping(payload)


def test_ledger_event_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="payload must be an object"):
        LedgerEvent.from_mapping(
            {
                "type": LEDGER_TYPE,
                "event": "turn.recorded",
                "ts": "2026-07-01T00:00:00Z",
                "event_id": "evt-1",
                "source": "test",
                "payload": "not-object",
            }
        )


def test_ledger_event_rejects_wrong_type() -> None:
    with pytest.raises(ValueError, match="ledger event type"):
        LedgerEvent.from_mapping(
            {
                "type": "run_fact",
                "event": "turn.recorded",
                "ts": "2026-07-01T00:00:00Z",
                "event_id": "evt-1",
                "source": "test",
                "payload": {},
            }
        )


def test_ledger_store_filters_by_task_run_and_session(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path)
    store.append(
        new_ledger_event(
            "turn.recorded",
            "evt-task-a",
            "test",
            {"role": "user", "content": "a"},
            task_id="task-a",
            session_id="session-a",
            run_id="run-a",
            ts="2026-07-01T00:00:00Z",
        )
    )
    store.append(
        new_ledger_event(
            "turn.recorded",
            "evt-task-b",
            "test",
            {"role": "user", "content": "b"},
            task_id="task-b",
            session_id="session-b",
            run_id="run-b",
            ts="2026-07-01T00:00:01Z",
        )
    )

    assert [event.event_id for event in store.read_task_events("task-a")] == [
        "evt-task-a"
    ]
    assert [event.event_id for event in store.read_run_events("run-b")] == [
        "evt-task-b"
    ]
    assert [event.event_id for event in store.read_session_events("session-a")] == [
        "evt-task-a"
    ]


def test_ledger_store_rejects_empty_filter_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="valid identity"):
        LedgerStore(tmp_path).read_task_events(" ")


def test_ledger_store_rejects_corrupt_jsonl_line(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path)
    path = store.append(
        new_ledger_event("decision", "action", "user_action", {"accepted": True})
    )
    path.write_bytes(b"broken\n")
    with pytest.raises(ValueError):
        store.read_events()
