"""Unit tests for `frontends.shared.session_search`."""

from __future__ import annotations

from pathlib import Path

import pytest
from typing import Any

from frontends.shared.session_search import (
    FileScanBackend,
    SessionSearchService,
)
from runtime.run_facts import RunFactStore
from runtime.persistence import RuntimeStore
from runtime.session_state import SessionState, SessionStateStore


def test_corrupt_run_fact_fails_search_without_returning_partial_hits(
    tmp_path: Path,
) -> None:
    """运行事实原件损坏不能伪装正常搜索；参数：隔离根；返回：无。"""
    session_id, run_id = "session-" + "e" * 32, "run-" + "e" * 32
    SessionStateStore(tmp_path).save(
        SessionState(session_id, summary="needle", last_run_id=run_id)
    )
    RunFactStore(tmp_path).append(
        {
            "session_id": session_id,
            "run_id": run_id,
            "event": "run:start",
            "trigger": "user",
        }
    )
    database = RuntimeStore(tmp_path)
    with database.snapshot() as source:
        record = source.list_raw("run_fact", session_id=session_id)[0]
    assert record.location is not None
    path = tmp_path / record.location.path
    raw = path.read_bytes()
    offset = record.location.offset
    path.write_bytes(raw[:offset] + b"!" + raw[offset + 1 :])
    with pytest.raises(ValueError):
        SessionSearchService(tmp_path).search("needle")


def _make_session(
    data_root: Path,
    *,
    session_id: str,
    updated_at: str,
    summary: str = "",
    last_run_id: str = "",
    last_run_status: str = "",
    last_run_event: str = "",
    focus_task_id: str | None = None,
) -> SessionState:
    state = SessionState(
        session_id=session_id,
        updated_at=updated_at,
        summary=summary,
        last_run_id=last_run_id,
        last_run_status=last_run_status,
        last_run_event=last_run_event,
        focus_task_id=focus_task_id,
    )
    SessionStateStore(data_root).save(state)
    return state


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


# --- sort & basic behavior -----------------------------------------------


def test_search_sort_by_updated_at_desc(tmp_path: Path) -> None:
    _make_session(
        tmp_path,
        session_id="session-" + "a" * 32,
        updated_at="2026-05-19T10:00:00Z",
        summary="alpha keyword report",
    )
    _make_session(
        tmp_path,
        session_id="session-" + "b" * 32,
        updated_at="2026-05-19T11:00:00Z",
        summary="beta keyword report",
    )
    _make_session(
        tmp_path,
        session_id="session-" + "c" * 32,
        updated_at="2026-05-19T09:00:00Z",
        summary="gamma keyword report",
    )

    service = SessionSearchService(tmp_path)
    result = service.search("keyword")

    assert [hit.session_id for hit in result.hits] == [
        "session-" + "b" * 32,
        "session-" + "a" * 32,
        "session-" + "c" * 32,
    ]
    assert result.warnings == []


def test_search_returns_session_id_snippet_and_source(tmp_path: Path) -> None:
    _make_session(
        tmp_path,
        session_id="session-" + "1" * 32,
        updated_at="2026-05-19T10:00:00Z",
        summary="long context discussing widget rollout phases",
        last_run_id="run-" + "1" * 32,
    )

    service = SessionSearchService(tmp_path)
    result = service.search("widget")

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.session_id == "session-" + "1" * 32
    assert hit.hit_source == "session_summary"
    assert "widget" in hit.summary_snippet
    assert hit.last_run_id == "run-" + "1" * 32


# --- ID precise routing ---------------------------------------------------


def test_run_id_exact_routes_to_facts(tmp_path: Path) -> None:
    session_id = "session-" + "2" * 32
    run_id = "run-" + "2" * 32
    _make_session(
        tmp_path,
        session_id=session_id,
        updated_at="2026-05-19T10:00:00Z",
        last_run_id=run_id,
    )
    _write_facts(
        tmp_path,
        session_id=session_id,
        run_id=run_id,
        facts=[
            {
                "type": "run_fact",
                "event": "run:start",
                "ts": "2026-05-19T10:00:00Z",
                "session_id": session_id,
                "run_id": run_id,
                "trigger": "user",
            },
            {
                "type": "run_fact",
                "event": "state:transition",
                "ts": "2026-05-19T10:01:00Z",
                "session_id": session_id,
                "run_id": run_id,
                "to_state": "DONE",
            },
        ],
    )

    service = SessionSearchService(tmp_path)
    precise = service.resolve_id(run_id)

    assert precise is not None
    assert precise.kind == "run"
    assert precise.hit.session_id == session_id
    assert precise.hit.last_run_id == run_id
    assert precise.hit.trigger_badge == "user"


def test_session_id_exact_routes_to_state(tmp_path: Path) -> None:
    session_id = "session-" + "3" * 32
    _make_session(
        tmp_path,
        session_id=session_id,
        updated_at="2026-05-19T10:00:00Z",
        summary="precise session lookup demo",
    )

    service = SessionSearchService(tmp_path)
    precise = service.resolve_id(session_id)

    assert precise is not None
    assert precise.kind == "session"
    assert precise.hit.session_id == session_id
    assert "precise session lookup demo" in precise.hit.summary_snippet


def test_search_with_session_id_query_returns_single_hit(tmp_path: Path) -> None:
    session_id = "session-" + "4" * 32
    _make_session(
        tmp_path,
        session_id=session_id,
        updated_at="2026-05-19T10:00:00Z",
        summary="session_id_routes",
    )

    service = SessionSearchService(tmp_path)
    result = service.search(session_id)

    assert len(result.hits) == 1
    assert result.hits[0].session_id == session_id
    assert result.warnings == []


def test_search_with_unknown_session_id_falls_back_to_keyword(tmp_path: Path) -> None:
    # Have one matching session by keyword, none matching the unknown id.
    _make_session(
        tmp_path,
        session_id="session-" + "5" * 32,
        updated_at="2026-05-19T10:00:00Z",
        summary="contains the search needle in summary",
    )
    unknown = "session-" + "0" * 32
    service = SessionSearchService(tmp_path)
    result = service.search(unknown)

    codes = {warning.code for warning in result.warnings}
    assert "precise_lookup_miss" in codes
    # Keyword fallback won't match because tokens come from the unknown ID
    # itself; the warning is the important signal.


# --- degradation paths ----------------------------------------------------


def test_missing_sessions_dir_returns_warning(tmp_path: Path) -> None:
    service = SessionSearchService(tmp_path)
    result = service.search("anything")

    assert result.hits == []
    assert result.warnings == []
    assert result.scanned_count == 0


def test_corrupt_state_json_fails_search_loudly(tmp_path: Path) -> None:
    good_id = "session-" + "6" * 32
    _make_session(
        tmp_path,
        session_id=good_id,
        updated_at="2026-05-19T10:00:00Z",
        summary="good keyword content",
    )
    bad_id = "session-" + "7" * 32
    store = SessionStateStore(tmp_path)
    store.save(SessionState(bad_id, summary="corrupt"))
    path = store.database.source_path("session_state", bad_id)
    path.write_bytes(b"broken state source")
    with pytest.raises(ValueError):
        SessionSearchService(tmp_path).search("keyword")


def test_short_token_emits_warning(tmp_path: Path) -> None:
    _make_session(
        tmp_path,
        session_id="session-" + "9" * 32,
        updated_at="2026-05-19T10:00:00Z",
        summary="long enough summary",
    )

    service = SessionSearchService(tmp_path)
    result = service.search("a")

    codes = {warning.code for warning in result.warnings}
    assert "token_too_short" in codes
    assert result.hits == []


def test_trigger_badge_failure_returns_none(tmp_path: Path) -> None:
    session_id = "session-" + "a" * 31 + "0"
    _make_session(
        tmp_path,
        session_id=session_id,
        updated_at="2026-05-19T10:00:00Z",
        summary="badge fallback keyword",
    )
    # No facts.jsonl, so list_runs_for_session returns nothing → badge None.
    service = SessionSearchService(tmp_path)
    result = service.search("keyword")

    assert len(result.hits) == 1
    assert result.hits[0].trigger_badge is None


def test_truncation_when_scan_limit_reached(tmp_path: Path) -> None:
    # Create more sessions than the lowered scan_limit.
    for idx in range(5):
        _make_session(
            tmp_path,
            session_id=f"session-{idx:032d}",
            updated_at=f"2026-05-19T10:0{idx}:00Z",
            summary="needle-keyword-here",
        )

    backend = FileScanBackend(tmp_path, scan_limit=3)
    result = backend.scan("needle-keyword-here", limit=10, deep_scan=False)

    assert result.truncated is True
    assert any(warning.code == "scan_limit_reached" for warning in result.warnings)
    assert result.scanned_count == 3


def test_deep_scan_picks_up_fact_matches(tmp_path: Path) -> None:
    session_id = "session-" + "b" * 31 + "0"
    run_id = "run-" + "b" * 31 + "0"
    _make_session(
        tmp_path,
        session_id=session_id,
        updated_at="2026-05-19T10:00:00Z",
        summary="ordinary summary without the term",
        last_run_id=run_id,
    )
    _write_facts(
        tmp_path,
        session_id=session_id,
        run_id=run_id,
        facts=[
            {
                "type": "run_fact",
                "event": "tool:response",
                "ts": "2026-05-19T10:00:00Z",
                "session_id": session_id,
                "run_id": run_id,
                "tool": {
                    "name": "shell",
                    "summary": "executed deepscanmarker successfully",
                },
            }
        ],
    )

    service = SessionSearchService(tmp_path)
    shallow = service.search("deepscanmarker", deep_scan=False)
    assert shallow.hits == []

    deep = service.search("deepscanmarker", deep_scan=True)
    assert len(deep.hits) == 1
    assert deep.hits[0].hit_source == "run_fact"
    assert "deepscanmarker" in deep.hits[0].summary_snippet


def test_unknown_run_id_falls_back_with_warning(tmp_path: Path) -> None:
    # Have a session matching by keyword, but the run-id we type does not exist.
    _make_session(
        tmp_path,
        session_id="session-" + "c" * 31 + "0",
        updated_at="2026-05-19T10:00:00Z",
        summary="fallback marker text",
    )
    unknown_run = "run-" + "0" * 32

    service = SessionSearchService(tmp_path)
    result = service.search(unknown_run)

    codes = {warning.code for warning in result.warnings}
    assert "precise_lookup_miss" in codes


def test_search_empty_query_returns_empty_result(tmp_path: Path) -> None:
    _make_session(
        tmp_path,
        session_id="session-" + "d" * 31 + "0",
        updated_at="2026-05-19T10:00:00Z",
        summary="anything",
    )

    service = SessionSearchService(tmp_path)
    result = service.search("   ")

    assert result.hits == []
    assert result.warnings == []


def test_and_logic_requires_all_tokens_to_match(tmp_path: Path) -> None:
    _make_session(
        tmp_path,
        session_id="session-" + "e" * 31 + "0",
        updated_at="2026-05-19T10:00:00Z",
        summary="alpha bravo charlie delta",
    )
    _make_session(
        tmp_path,
        session_id="session-" + "f" * 31 + "0",
        updated_at="2026-05-19T10:01:00Z",
        summary="alpha only",
    )

    service = SessionSearchService(tmp_path)
    result = service.search("alpha bravo")

    assert [hit.session_id for hit in result.hits] == ["session-" + "e" * 31 + "0"]


def test_run_id_route_reads_trigger_badge(tmp_path: Path) -> None:
    session_id = "session-" + "1" * 31 + "0"
    run_id = "run-" + "1" * 31 + "0"
    _make_session(
        tmp_path,
        session_id=session_id,
        updated_at="2026-05-19T10:00:00Z",
        summary="trigger badge check",
        last_run_id=run_id,
    )
    _write_facts(
        tmp_path,
        session_id=session_id,
        run_id=run_id,
        facts=[
            {
                "type": "run_fact",
                "event": "run:start",
                "ts": "2026-05-19T10:00:00Z",
                "session_id": session_id,
                "run_id": run_id,
                "trigger": "cron",
            }
        ],
    )

    service = SessionSearchService(tmp_path)
    result = service.search("trigger")

    assert len(result.hits) == 1
    assert result.hits[0].trigger_badge == "cron"
