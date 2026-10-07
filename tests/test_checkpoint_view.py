from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from runtime.checkpoint import Checkpoint, checkpoint_to_ledger_state, save_checkpoint
from runtime.checkpoint_view import build_checkpoint_view
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.lease import from_trigger


def test_checkpoint_view_projects_latest_checkpoint_from_ledger(tmp_path: Path) -> None:
    """CheckpointView 从 Ledger 重建最新恢复点。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无
    """
    lease = from_trigger("user", task_id="task-1")
    writer = LedgerWriter(
        LedgerStore(tmp_path),
        source="tests.checkpoint_view",
        id_factory=_event_ids(),
    )
    first = save_checkpoint(
        Checkpoint(
            task_id="task-1",
            segment_id="user-1",
            state="PAUSED",
            session_id="session-1",
            run_id="run-1",
            lease_snapshot=asdict(lease),
            reason="waiting_user",
            saved_at="2026-07-02T00:00:00+00:00",
        )
    )
    second = save_checkpoint(
        Checkpoint(
            task_id="task-1",
            segment_id="user-2",
            state="DONE",
            session_id="session-1",
            run_id="run-2",
            lease_snapshot=asdict(lease),
            reason="done",
            saved_at="2026-07-02T00:00:01+00:00",
        )
    )
    for checkpoint in (first, second):
        writer.record_checkpoint_saved(
            checkpoint.checkpoint_id,
            checkpoint_to_ledger_state(checkpoint),
            checkpoint.reason,
            task_id=checkpoint.task_id,
            session_id=checkpoint.session_id,
            run_id=checkpoint.run_id,
        )

    view = build_checkpoint_view(LedgerStore(tmp_path).read_events())
    latest = view.for_task("task-1").latest()

    assert latest is not None
    assert latest.checkpoint_id == second.checkpoint_id
    assert latest.state == "DONE"
    assert latest.lease_snapshot["trigger"] == "user"


def test_checkpoint_view_preserves_pending_tool_call(tmp_path: Path) -> None:
    """CheckpointView 保留等待工具恢复所需的 pending tool call。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无
    """
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id="task-tool",
            segment_id="user-tool",
            state="pre_tool",
            session_id="session-tool",
            run_id="run-tool",
            pending_tool_call={
                "tool_name": "file_write",
                "args": {"path": "a.txt"},
                "call_id": "call-1",
            },
            reason="pre_tool",
        )
    )
    LedgerWriter(
        LedgerStore(tmp_path),
        source="tests.checkpoint_view",
        id_factory=lambda: "ledger-tool",
    ).record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        checkpoint.reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )

    latest = build_checkpoint_view(LedgerStore(tmp_path).read_events()).latest()

    assert latest is not None
    assert latest.pending_tool_call == checkpoint.pending_tool_call


def _event_ids() -> Callable[[], str]:
    """生成测试 Ledger id 工厂。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：无
    返回：每次调用产生递增 id 的函数
    """
    index = 0

    def next_id() -> str:
        """生成下一个 Ledger id。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：无
        返回：Ledger id
        """
        nonlocal index
        index += 1
        return f"ledger-checkpoint-{index}"

    return next_id
