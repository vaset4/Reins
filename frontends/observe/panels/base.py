from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from runtime.run_facts import RunFactStore, RunSummary
from runtime.session_state import SessionState, SessionStateStore

from frontends.observe.readers.evidence_reader import EvidenceReader


@dataclass(frozen=True, slots=True)
class PanelDescriptor:
    id: str
    title: str
    section: str
    phase: str


class RunContext:
    """Read-only handle for a single run, passed to ``Panel.build``.

    Lazy: facts.jsonl, summary, and state are loaded on first access only.
    Panels never touch the filesystem directly.
    """

    def __init__(
        self,
        *,
        data_root: Path,
        session_id: str,
        run_id: str,
        fact_store: RunFactStore,
        session_store: SessionStateStore,
        evidence_reader: EvidenceReader,
    ) -> None:
        self._data_root = data_root
        self._session_id = session_id
        self._run_id = run_id
        self._fact_store = fact_store
        self._session_store = session_store
        self._evidence_reader = evidence_reader
        self._facts_cache: list[dict[str, Any]] | None = None
        self._summary_cache: RunSummary | None = None
        self._summary_loaded = False
        self._state_cache: SessionState | None = None
        self._state_loaded = False
        self._errors_cache: list[dict[str, Any]] | None = None

    @property
    def data_root(self) -> Path:
        return self._data_root

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def facts(self) -> list[dict[str, Any]]:
        if self._facts_cache is None:
            self._facts_cache = self._fact_store.read_run(self._run_id)
        return self._facts_cache

    def summary(self) -> RunSummary | None:
        if not self._summary_loaded:
            for item in self._fact_store.list_runs_for_session(self._session_id):
                if item.run_id == self._run_id:
                    self._summary_cache = item
                    break
            self._summary_loaded = True
        return self._summary_cache

    def state(self) -> SessionState | None:
        if not self._state_loaded:
            self._state_cache = self._session_store.load(self._session_id)
            self._state_loaded = True
        return self._state_cache

    def evidence(self, key: str) -> Mapping[str, Any] | None:
        return self._evidence_reader.read_latest(self.facts(), key)

    def errors(self) -> list[dict[str, Any]]:
        if self._errors_cache is None:
            self._errors_cache = self._evidence_reader.read_errors(
                self._session_id, self._run_id
            )
        return self._errors_cache


class Panel:
    """Base class for inspection panels.

    Subclasses must define class-level ``id``, ``title``, ``section``,
    ``phase`` and implement ``build``.
    """

    id: ClassVar[str] = ""
    title: ClassVar[str] = ""
    section: ClassVar[str] = ""
    phase: ClassVar[str] = ""

    def descriptor(self) -> PanelDescriptor:
        return PanelDescriptor(
            id=self.id, title=self.title, section=self.section, phase=self.phase
        )

    def build(self, ctx: RunContext) -> dict[str, Any]:
        raise NotImplementedError("subclass must implement build()")


__all__ = ["Panel", "PanelDescriptor", "RunContext"]
