from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.run_facts import RunFactStore, RunSummary
from runtime.session_state import SessionState, SessionStateStore


class FactReader:
    """Thin wrapper over RunFactStore + SessionStateStore for the dashboard."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._facts = RunFactStore(data_root)
        self._sessions = SessionStateStore(data_root)

    @property
    def fact_store(self) -> RunFactStore:
        return self._facts

    @property
    def session_store(self) -> SessionStateStore:
        return self._sessions

    def list_sessions(self, *, limit: int = 50) -> list[SessionState]:
        return self._sessions.list_recent(limit=limit)

    def list_runs(self, session_id: str, *, limit: int = 50) -> list[RunSummary]:
        return self._facts.list_runs_for_session(session_id, limit=limit)

    def list_recent_runs(self, *, limit: int = 50) -> list[RunSummary]:
        return self._facts.list_recent_runs(limit=limit)

    def read_facts(self, run_id: str) -> list[dict[str, Any]]:
        return self._facts.read_run(run_id)

    def find_live_runs(self) -> list[RunSummary]:
        """Return non-terminal runs, most recent first."""
        recent = self._facts.list_recent_runs(limit=20)
        return [r for r in recent if r.status not in {"done", "failed", "paused"}]


__all__ = ["FactReader"]
