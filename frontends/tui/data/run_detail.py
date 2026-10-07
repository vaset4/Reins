from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from frontends.tui.data.run_evidence_detail import (
    preview_raw_evidence,
    read_evidence_summaries,
)
from frontends.shared.run_observation import (
    ContextObservation,
    RunObservation,
    build_run_observation,
)
from runtime.run_facts import RunFactStore, RunSummary
from runtime.session_state import SessionStateStore


@dataclass(frozen=True, slots=True)
class RunDetailSection:
    title: str
    summary: list[str]
    raw_path: str = ""
    raw_preview: list[str] = field(default_factory=list)
    expanded: bool = False


@dataclass(frozen=True, slots=True)
class RunDetailView:
    session_id: str
    run_id: str
    task_id: str
    status: str
    started_at: str
    updated_at: str
    session_summary: str
    context_summary: list[str]
    model_input_summary: list[str]
    model_output_summary: list[str]
    call_summary: list[str]
    evidence_summary: list[str]
    sections: list[RunDetailSection]
    checkpoint_summary: list[str] = field(default_factory=list)
    memory_summary: list[str] = field(default_factory=list)
    compression_summary: list[str] = field(default_factory=list)


class RunDetailReader:
    def __init__(self, data_root: Path | str) -> None:
        self._data_root = Path(data_root)
        self._facts = RunFactStore(self._data_root)
        self._sessions = SessionStateStore(self._data_root)

    def list_session_runs(
        self, session_id: str, *, limit: int = 20
    ) -> list[RunSummary]:
        runs = self._facts.list_runs_for_session(session_id, limit=limit)
        return runs

    def read_run_detail(
        self,
        session_id: str,
        run_id: str,
        *,
        expand_raw: bool = False,
    ) -> RunDetailView:
        facts = self._facts.read_run(run_id)
        if not facts:
            raise FileNotFoundError(f"run facts missing for {session_id}/{run_id}")
        summary = _find_summary(self._facts, facts, run_id)
        session = self._sessions.load(session_id)
        observation = build_run_observation(facts)
        evidence = _latest_evidence(facts)
        evidence_summaries = read_evidence_summaries(self._data_root, evidence)
        sections = _sections(self._data_root, evidence, expand_raw=expand_raw)
        return RunDetailView(
            session_id=session_id,
            run_id=run_id,
            task_id=_first_non_empty(facts, "task_id", "focus_task_id") or "(none)",
            status=summary.status if summary is not None else _latest_event(facts),
            started_at=summary.started_at if summary is not None else _first_ts(facts),
            updated_at=summary.updated_at if summary is not None else _first_ts(facts),
            session_summary=session.summary if session is not None else "",
            context_summary=_context_summary(
                observation.context, evidence_summaries.context
            ),
            model_input_summary=evidence_summaries.model_input,
            model_output_summary=evidence_summaries.model_output,
            call_summary=_call_summary(facts),
            evidence_summary=evidence_summaries.evidence,
            sections=sections,
            checkpoint_summary=_checkpoint_summary(observation),
            memory_summary=_memory_summary(observation),
            compression_summary=_compression_summary(observation),
        )


def render_run_detail_lines(detail: RunDetailView) -> list[str]:
    lines = [
        "Run detail",
        f"session: {detail.session_id}",
        f"run: {detail.run_id}",
        f"task: {detail.task_id}",
        f"status: {detail.status}",
        f"started: {detail.started_at}",
        f"updated: {detail.updated_at}",
        "",
        "Context",
        *_indent(detail.context_summary),
        *_optional_block("Checkpoints", detail.checkpoint_summary),
        *_optional_block("Memory", detail.memory_summary),
        *_optional_block("Compression", detail.compression_summary),
        "",
        "Model Input",
        *_indent(detail.model_input_summary),
        "",
        "Model Output",
        *_indent(detail.model_output_summary),
        "",
        "Calls",
        *_indent(detail.call_summary),
        "",
        "Evidence",
        *_indent(detail.evidence_summary),
    ]
    for section in detail.sections:
        lines.extend(_section_lines(section))
    return lines


def _find_summary(
    store: RunFactStore,
    facts: list[dict[str, Any]],
    run_id: str,
) -> RunSummary | None:
    session_id = str(facts[0].get("session_id", "")) if facts else ""
    if not session_id:
        return None
    for summary in store.list_runs_for_session(session_id):
        if summary.run_id == run_id:
            return summary
    return None


def _context_summary(
    context: ContextObservation,
    evidence_lines: list[str],
) -> list[str]:
    return [
        f"context_events: {context.build_count}",
        f"last_tool_history_count: {context.last_tool_history_count}",
    ] + evidence_lines


def _checkpoint_summary(observation: RunObservation) -> list[str]:
    return [
        f"{item.checkpoint_id or '(none)'} | {item.state} | {item.reason}"
        for item in observation.checkpoints
    ]


def _memory_summary(observation: RunObservation) -> list[str]:
    injections = len(observation.memory.injections)
    scores = len(observation.memory.score_breakdowns)
    return (
        [f"injection_explain: {injections}", f"score_breakdown: {scores}"]
        if injections or scores
        else []
    )


def _compression_summary(observation: RunObservation) -> list[str]:
    return [
        (
            f"{event.get('reason') or '(none)'} | "
            f"{event.get('tokens_before')}->{event.get('tokens_after')} | "
            f"removed={event.get('removed_sections', [])}"
        )
        for event in observation.compression.events
    ]


def _call_summary(facts: list[dict[str, Any]]) -> list[str]:
    tool_requests = sum(1 for row in facts if row.get("event") == "tool:request")
    tool_responses = sum(1 for row in facts if row.get("event") == "tool:response")
    llm_rows = [row for row in facts if row.get("event") == "llm:response"]
    observation = _latest_observation(llm_rows)
    error = _latest_error(llm_rows)
    return [
        f"llm_responses: {len(llm_rows)}",
        f"tool_requests: {tool_requests}",
        f"tool_responses: {tool_responses}",
        f"stage: {observation.get('stage', '(none)')}",
        f"provider: {observation.get('provider', '(none)')}",
        f"model: {observation.get('model', '(none)')}",
        f"elapsed_ms: {observation.get('elapsed_ms', '(none)')}",
        f"attempts: {observation.get('attempt_count', '(none)')}",
        f"tokens: {observation.get('prompt_tokens', 0)}/{observation.get('completion_tokens', 0)}",
        f"error: {error.get('category', '(none)') if error else '(none)'}",
        f"total_facts: {len(facts)}",
    ]


def _latest_observation(rows: list[dict[str, Any]]) -> Mapping[str, object]:
    summary = _latest_llm_summary(rows)
    observation = summary.get("observation", {})
    return observation if isinstance(observation, Mapping) else {}


def _latest_error(rows: list[dict[str, Any]]) -> Mapping[str, object]:
    summary = _latest_llm_summary(rows)
    error = summary.get("error", {})
    return error if isinstance(error, Mapping) else {}


def _latest_llm_summary(rows: list[dict[str, Any]]) -> Mapping[str, object]:
    if not rows:
        return {}
    summary = rows[-1].get("summary", {})
    return summary if isinstance(summary, Mapping) else {}


def _sections(
    data_root: Path,
    evidence: Mapping[str, object],
    *,
    expand_raw: bool,
) -> list[RunDetailSection]:
    return [
        _evidence_section(data_root, evidence, "model_request", expand_raw=expand_raw),
        _evidence_section(data_root, evidence, "model_response", expand_raw=expand_raw),
        _evidence_section(data_root, evidence, "parsed_plan", expand_raw=expand_raw),
        _evidence_section(data_root, evidence, "errors", expand_raw=expand_raw),
    ]


def _evidence_section(
    data_root: Path,
    evidence: Mapping[str, object],
    key: str,
    *,
    expand_raw: bool,
) -> RunDetailSection:
    raw_path = str(evidence.get(key, ""))
    summary = [f"path: {raw_path or '(missing)'}"]
    preview = preview_raw_evidence(data_root, raw_path) if expand_raw else []
    return RunDetailSection(
        title=key,
        summary=summary,
        raw_path=raw_path,
        raw_preview=preview,
        expanded=expand_raw,
    )


def _section_lines(section: RunDetailSection) -> list[str]:
    marker = "expanded" if section.expanded else "collapsed"
    lines = ["", f"Raw {section.title} [{marker}]", *_indent(section.summary)]
    if section.expanded:
        lines.extend(_indent(section.raw_preview))
    return lines


def _latest_evidence(facts: list[dict[str, Any]]) -> Mapping[str, object]:
    for row in reversed(facts):
        if row.get("event") != "llm:response":
            continue
        summary = row.get("summary", {})
        if not isinstance(summary, Mapping):
            continue
        evidence = summary.get("evidence", {})
        if isinstance(evidence, Mapping):
            return evidence
    return {}


def _first_non_empty(rows: list[dict[str, Any]], *keys: str) -> str:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _first_ts(rows: list[dict[str, Any]]) -> str:
    return _first_non_empty(rows, "ts")


def _latest_event(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return str(rows[-1].get("event", ""))


def _indent(lines: list[str]) -> list[str]:
    return [f"  {line}" for line in lines]


def _optional_block(title: str, lines: list[str]) -> list[str]:
    return ["", title, *_indent(lines)] if lines else []


__all__ = [
    "RunDetailReader",
    "RunDetailSection",
    "RunDetailView",
    "render_run_detail_lines",
]
