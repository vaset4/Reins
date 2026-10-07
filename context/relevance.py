from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from runtime.run_facts import RunFactStore, RunSummary


@dataclass(frozen=True, slots=True)
class TaskRelevanceDecision:
    include_task_context: bool
    reason: str
    focus_task_id: str | None = None
    run_fact_sources: tuple[str, ...] = ()


def decide_task_relevance(
    *,
    effective_task_id: str | None,
    trigger: object,
    payload: dict[str, object],
    task_relevant: bool | None = None,
    data_root: Path | str,
    session_id: str = "",
    run_id: str = "",
    task_goal: str = "",
    task_tags: Sequence[str] = (),
) -> TaskRelevanceDecision:
    trigger_value = getattr(trigger, "value", str(trigger))
    store: RunFactStore | None = None
    if effective_task_id is None:
        if run_id:
            store = RunFactStore(data_root)
            facts = store.read_run(run_id)
            inferred_task_id = _task_id_from_facts(facts)
            inferred_sources = (f"run_facts:run:{run_id}",) if facts else ()
            if inferred_task_id and trigger_value == "resume":
                return TaskRelevanceDecision(
                    True, "resume_trigger", inferred_task_id, inferred_sources
                )
            if inferred_task_id and (
                payload.get("checkpoint_id")
                or payload.get("pending_tool_call")
                or _facts_have_unfinished_checkpoint(facts)
            ):
                return TaskRelevanceDecision(
                    True,
                    "unfinished_checkpoint_in_current_run",
                    inferred_task_id,
                    inferred_sources,
                )
        return TaskRelevanceDecision(False, "no_focus_task")
    if task_relevant is not None:
        return TaskRelevanceDecision(
            bool(task_relevant),
            "explicit_task_relevance_true"
            if task_relevant
            else "explicit_task_relevance_false",
            effective_task_id,
        )
    if _explicit_other_task(payload, effective_task_id):
        return TaskRelevanceDecision(
            False, "explicit_continue_other_task", effective_task_id
        )
    if str(payload.get("continue_task_id", "")) == effective_task_id:
        return TaskRelevanceDecision(True, "explicit_continue", effective_task_id)
    if _truthy(payload.get("explicit_continue")) or _truthy(
        payload.get("task_relevant")
    ):
        return TaskRelevanceDecision(True, "explicit_continue", effective_task_id)

    if trigger_value == "resume":
        return TaskRelevanceDecision(True, "resume_trigger", effective_task_id)
    if payload.get("checkpoint_id") or payload.get("pending_tool_call"):
        return TaskRelevanceDecision(True, "checkpoint_payload", effective_task_id)

    store = store or RunFactStore(data_root)
    sources: list[str] = []
    if run_id and _run_has_unfinished_checkpoint(store, run_id):
        sources.append(f"run_facts:run:{run_id}")
        return TaskRelevanceDecision(
            True,
            "unfinished_checkpoint_in_current_run",
            effective_task_id,
            tuple(sources),
        )

    recent_runs = store.list_runs_for_session(session_id, limit=5) if session_id else []
    matching_runs = [
        summary
        for summary in recent_runs
        if _summary_matches_task(summary, effective_task_id)
    ]
    sources.extend(f"run_facts:session:{summary.run_id}" for summary in matching_runs)
    if matching_runs and matching_runs[0].status in {"done", "failed"}:
        return TaskRelevanceDecision(
            False,
            "recent_focus_task_terminal",
            effective_task_id,
            tuple(sources),
        )

    user_text = _payload_text(payload)
    if len(matching_runs) >= 2 and (
        not user_text or _text_matches_task(user_text, task_goal, task_tags)
    ):
        return TaskRelevanceDecision(
            True,
            "recent_focus_task_continuity",
            effective_task_id,
            tuple(sources),
        )
    if user_text and not _text_matches_task(user_text, task_goal, task_tags):
        return TaskRelevanceDecision(
            False,
            "topic_switch",
            effective_task_id,
            tuple(sources),
        )
    return TaskRelevanceDecision(
        False,
        "minimal_by_default",
        effective_task_id,
        tuple(sources),
    )


def _run_has_unfinished_checkpoint(store: RunFactStore, run_id: str) -> bool:
    facts = store.read_run(run_id)
    return _facts_have_unfinished_checkpoint(facts)


def _facts_have_unfinished_checkpoint(facts: list[dict[str, object]]) -> bool:
    if not facts:
        return False
    last_terminal = ""
    checkpoint_seen = False
    for fact in facts:
        if fact.get("event") == "checkpoint:saved":
            checkpoint_seen = True
        if fact.get("event") == "state:transition":
            state = str(fact.get("to_state", "")).lower()
            if state in {"done", "failed"}:
                last_terminal = state
    return checkpoint_seen and last_terminal not in {"done", "failed"}


def _task_id_from_facts(facts: list[dict[str, object]]) -> str | None:
    for fact in facts:
        for key in ("focus_task_id", "task_id"):
            value = fact.get(key)
            if value is not None and str(value):
                return str(value)
    return None


def _summary_matches_task(summary: RunSummary, task_id: str) -> bool:
    return summary.task_id == task_id or summary.focus_task_id == task_id


def _explicit_other_task(payload: dict[str, object], effective_task_id: str) -> bool:
    for key in ("continue_task_id", "task_id", "focus_task_id"):
        value = payload.get(key)
        if value is not None and str(value) and str(value) != effective_task_id:
            return True
    return False


def _payload_text(payload: dict[str, object]) -> str:
    parts = []
    for key in ("message", "latest", "input", "query"):
        value = payload.get(key)
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def _text_matches_task(text: str, task_goal: str, task_tags: Sequence[str]) -> bool:
    haystack = set(_tokens(text))
    if not haystack:
        return False
    task_words = set(_tokens(task_goal))
    tag_words = {token for tag in task_tags for token in _tokens(str(tag))}
    return bool(haystack & (task_words | tag_words))


def _tokens(text: str) -> list[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [token for token in normalized.split() if len(token) >= 3]


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


__all__ = ["TaskRelevanceDecision", "decide_task_relevance"]
