from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_evidence_view import build_run_evidence_view


def test_run_evidence_view_projects_ledger_events_only(tmp_path: Path) -> None:
    """RunEvidenceView 只从 Ledger 事件重建运行证据。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无
    """
    event_index = 0

    def next_event_id() -> str:
        """生成稳定测试事件 id。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：无
        返回：稳定递增的 Ledger 事件 id
        """
        nonlocal event_index
        event_index += 1
        return f"ledger-test-{event_index}"

    writer = LedgerWriter(
        LedgerStore(tmp_path),
        source="tests.run_evidence_view",
        id_factory=next_event_id,
    )
    writer.record_model_requested(
        "test-provider",
        "test-model",
        ["identity", "conversation"],
        task_id="task-1",
        session_id="session-1",
        run_id="run-1",
    )
    writer.record_context_segments(
        [{"name": "identity", "tokens_est": 3}],
        task_id="task-1",
        session_id="session-1",
        run_id="run-1",
    )
    writer.record_tool_requested(
        "list",
        "call-1",
        {"path": "."},
        task_id="task-1",
        session_id="session-1",
        run_id="run-1",
    )
    writer.record_tool_completed(
        "list",
        "call-1",
        "ok",
        content="listed",
        task_id="task-1",
        session_id="session-1",
        run_id="run-1",
    )
    writer.record_run_lifecycle_changed(
        "done",
        task_id="task-1",
        session_id="session-1",
        run_id="run-1",
    )

    view = build_run_evidence_view(LedgerStore(tmp_path).read_run_events("run-1"))

    assert view.run_id == "run-1"
    assert [row["event"] for row in view.model_requests] == ["model.requested"]
    assert [row["event"] for row in view.tool_events] == [
        "tool.requested",
        "tool.completed",
    ]
    context_payload = cast(Mapping[str, object], view.context_segments[0]["payload"])
    context_segments = cast(list[Mapping[str, object]], context_payload["segments"])
    lifecycle_payload = cast(Mapping[str, object], view.lifecycle_events[-1]["payload"])

    assert context_segments[0]["name"] == "identity"
    assert lifecycle_payload["status"] == "done"
    assert view.seen_event_ids == tuple(f"ledger-test-{index}" for index in range(1, 6))


def test_run_evidence_view_ignores_unrelated_ledger_events(tmp_path: Path) -> None:
    """RunEvidenceView 忽略非运行证据事件但保留审计计数。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无
    """
    writer = LedgerWriter(
        LedgerStore(tmp_path),
        source="tests.run_evidence_view",
        id_factory=lambda: "ledger-unrelated",
    )
    writer.record_summary_updated(
        "intent",
        "hello",
        task_id="task-1",
        session_id="session-1",
        run_id="run-1",
    )

    view = build_run_evidence_view(LedgerStore(tmp_path).read_run_events("run-1"))

    assert view.event_count == 1
    assert view.model_requests == ()
    assert view.tool_events == ()
    assert view.context_segments == ()
    assert view.lifecycle_events == ()
