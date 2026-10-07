from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from runtime.checkpoint import (
    Checkpoint,
    checkpoint_to_ledger_state,
    list_checkpoints,
    load_latest_checkpoint,
    save_checkpoint,
    save_post_tool_checkpoint,
    save_pre_tool_checkpoint,
)
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.lease import Lease, from_trigger
from runtime.run_facts import RunFactStore
from runtime.types import RunContext, Trigger
from tools.types import ToolError, ToolErrorCategory


def test_runtime_phase1_types_are_import_stable() -> None:
    lease = Lease(max_steps=10)
    ctx = RunContext(
        task_id="2026-05-04-01",
        trigger=Trigger.USER,
        payload={"message": "hi"},
        capability_lease=lease,
    )
    error = ToolError(ToolErrorCategory.UNKNOWN, "failed")

    assert ctx.session_id.startswith("session-")
    assert ctx.run_id.startswith("run-")
    assert ctx.focus_task_id == ctx.task_id
    assert error.partial_state == ""


def _record(store: RunFactStore, checkpoint: Checkpoint) -> None:
    ledger = LedgerWriter(
        LedgerStore(store._data_root),  # noqa: SLF001
        source="tests.runtime_checkpoint",
    )
    ledger.record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        checkpoint.reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )
    store.append_checkpoint_ref(
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
        task_id=None,
        focus_task_id=None,
        compatibility_task_id=checkpoint.compatibility_task_id or checkpoint.task_id,
        segment_id=checkpoint.segment_id,
        checkpoint=checkpoint,
    )


def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    lease = from_trigger("user", task_id="2026-05-04-01")
    ckpt = save_checkpoint(
        Checkpoint(
            task_id="2026-05-04-01",
            session_id="session-x",
            run_id="run-x",
            compatibility_task_id="2026-05-04-01",
            segment_id="user-01",
            state="PARSING",
            working_memory_snapshot={"a": "b"},
            lease_snapshot=asdict(lease),
            reason="unit",
        )
    )
    assert ckpt.checkpoint_id
    assert ckpt.saved_at

    _record(RunFactStore(tmp_path), ckpt)

    loaded = load_latest_checkpoint("2026-05-04-01", data_root=tmp_path)
    assert loaded is not None
    assert loaded.state == "PARSING"
    assert loaded.working_memory_snapshot == {"a": "b"}
    assert loaded.lease_snapshot["trigger"] == "user"
    assert loaded.lease_snapshot["task_id"] == "2026-05-04-01"
    assert loaded.reason == "unit"
    assert loaded.checkpoint_id == ckpt.checkpoint_id

    listed = list_checkpoints("2026-05-04-01", data_root=tmp_path)
    assert [item.checkpoint_id for item in listed] == [ckpt.checkpoint_id]


def test_pre_and_post_tool_checkpoints_are_ordered(tmp_path: Path) -> None:
    store = RunFactStore(tmp_path)
    pre = save_pre_tool_checkpoint(
        "user-01",
        "file_read",
        {"path": "a.txt"},
        "call-1",
        {"step": 1},
        task_id="2026-05-04-02",
        session_id="session-x",
        run_id="run-x",
        compatibility_task_id="2026-05-04-02",
    )
    _record(store, pre)
    post = save_post_tool_checkpoint(
        "user-01",
        {"step": 2},
        task_id="2026-05-04-02",
        session_id="session-x",
        run_id="run-x",
        compatibility_task_id="2026-05-04-02",
    )
    _record(store, post)

    checkpoints = list_checkpoints("2026-05-04-02", data_root=tmp_path)
    reasons = [item.reason for item in checkpoints]
    assert reasons == ["pre_tool", "post_tool"]
    assert checkpoints[0].pending_tool_call is not None
    assert checkpoints[0].pending_tool_call["call_id"] == "call-1"
    assert checkpoints[1].pending_tool_call is None
