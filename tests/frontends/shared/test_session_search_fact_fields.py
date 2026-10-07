from __future__ import annotations

from pathlib import Path
from typing import Any

from frontends.shared.session_search import SessionSearchService
from runtime.run_facts import RunFactStore
from runtime.session_state import SessionState, SessionStateStore


def _make_session(data_root: Path, session_id: str, run_id: str) -> None:
    SessionStateStore(data_root).save(
        SessionState(
            session_id=session_id,
            updated_at="2026-06-03T00:00:00Z",
            summary="ordinary summary",
            last_run_id=run_id,
        )
    )


def _write_facts(
    data_root: Path,
    *,
    session_id: str,
    run_id: str,
    facts: list[dict[str, Any]],
) -> None:
    store = RunFactStore(data_root)
    for fact in facts:
        store.append({"session_id": session_id, "run_id": run_id, **fact})


def test_deep_scan_ignores_unsupported_top_level_tool_summary(
    tmp_path: Path,
) -> None:
    session_id = "session-" + "a" * 32
    run_id = "run-" + "a" * 32
    _make_session(tmp_path, session_id, run_id)
    _write_facts(
        tmp_path,
        session_id=session_id,
        run_id=run_id,
        facts=[
            {
                "event": "tool:response",
                "tool_summary": "phantommarker",
            }
        ],
    )

    result = SessionSearchService(tmp_path).search("phantommarker", deep_scan=True)

    assert result.hits == []


def test_deep_scan_matches_nested_tool_output_summary(tmp_path: Path) -> None:
    session_id = "session-" + "b" * 32
    run_id = "run-" + "b" * 32
    _make_session(tmp_path, session_id, run_id)
    _write_facts(
        tmp_path,
        session_id=session_id,
        run_id=run_id,
        facts=[
            {
                "event": "tool:response",
                "tool": {
                    "name": "shell",
                    "status": "ok",
                    "output_summary": "nestedoutputmarker",
                },
            }
        ],
    )

    result = SessionSearchService(tmp_path).search(
        "nestedoutputmarker",
        deep_scan=True,
    )

    assert len(result.hits) == 1
    assert result.hits[0].summary_snippet == "nestedoutputmarker"


def test_deep_scan_uses_shared_fact_fields_for_trigger(tmp_path: Path) -> None:
    session_id = "session-" + "c" * 32
    run_id = "run-" + "c" * 32
    _make_session(tmp_path, session_id, run_id)
    _write_facts(
        tmp_path,
        session_id=session_id,
        run_id=run_id,
        facts=[
            {
                "event": "run:start",
                "trigger": "resume",
            }
        ],
    )

    result = SessionSearchService(tmp_path).search("resume", deep_scan=True)

    assert len(result.hits) == 1
    assert result.hits[0].summary_snippet == "resume"
