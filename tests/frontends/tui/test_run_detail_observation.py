from __future__ import annotations

from pathlib import Path

from runtime.run_facts import RunFactStore

from frontends.tui.data.run_detail import RunDetailReader, render_run_detail_lines


def test_run_detail_renders_shared_observation_tracks(tmp_path: Path) -> None:
    session_id = "session-detail"
    run_id = "run-detail"
    facts = RunFactStore(tmp_path)
    for payload in [
        {
            "event": "checkpoint:saved",
            "session_id": "session-detail",
            "run_id": "run-detail",
            "checkpoint": {
                "checkpoint_id": "ck-1",
                "state": "PAUSED",
                "reason": "pre_tool",
            },
        },
        {
            "event": "context:built",
            "session_id": "session-detail",
            "run_id": "run-detail",
            "summary": {"tool_history_count": 1},
        },
        {
            "event": "memory:score_breakdown",
            "session_id": "session-detail",
            "run_id": "run-detail",
            "round_id": "round-1",
            "skipped": [{"type": "fact"}],
        },
        {
            "event": "trim:delta",
            "session_id": "session-detail",
            "run_id": "run-detail",
            "reason": "overflow",
            "tokens_before": 100,
            "tokens_after": 80,
            "removed_sections": ["conversation"],
        },
    ]:
        facts.append(payload)

    detail = RunDetailReader(tmp_path).read_run_detail(session_id, run_id)
    lines = render_run_detail_lines(detail)

    assert any("ck-1 | PAUSED | pre_tool" in line for line in lines)
    assert any("score_breakdown: 1" in line for line in lines)
    assert any(
        "overflow | 100->80 | removed=['conversation']" in line for line in lines
    )
