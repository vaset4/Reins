from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Mapping

from runtime.run_facts import RunSummary
from runtime.session_state import SessionState

from frontends.observe.api.story_fact_stream import build_fact_stream
from frontends.shared.run_observation import build_run_observation


def build_run_story(
    facts: list[dict[str, Any]],
    summary: RunSummary | None,
    state: SessionState | None,
    read_raw: Callable[[str], Mapping[str, Any]] | None = None,
    run_files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a human-readable run story from persisted run facts."""

    observation = build_run_observation(facts)
    model_calls = _model_calls(facts)
    tool_calls = _tool_calls(facts)
    warnings = _warnings(facts, state, model_calls, tool_calls)
    if read_raw is not None:
        warnings.extend(_evidence_warnings(model_calls, read_raw))
    return {
        "overview": _overview(facts, summary, state, model_calls, tool_calls),
        "conversation": _conversation(facts),
        "lifecycle": asdict(observation.lifecycle),
        "context": asdict(observation.context),
        "checkpoints": [asdict(item) for item in observation.checkpoints],
        "raw_evidence": [asdict(item) for item in observation.raw_evidence],
        "memory": asdict(observation.memory),
        "compression": asdict(observation.compression),
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "fact_stream": build_fact_stream(facts, model_calls),
        "run_files": run_files or [],
        "transitions": _transitions(facts),
        "warnings": warnings,
        "raw": {"facts_count": len(facts)},
    }


def _overview(
    facts: list[dict[str, Any]],
    summary: RunSummary | None,
    state: SessionState | None,
    model_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    start = facts[0].get("ts", "") if facts else ""
    end = facts[-1].get("ts", "") if facts else ""
    return {
        "session_id": _first_text(facts, "session_id"),
        "run_id": _first_text(facts, "run_id"),
        "status": summary.status if summary else _terminal_status(facts),
        "started_at": summary.started_at if summary else start,
        "updated_at": summary.updated_at if summary else end,
        "last_event": summary.last_event if summary else _last_event(facts),
        "model_calls": len(model_calls),
        "tool_calls": len(tool_calls),
        "total_elapsed_ms": sum(_int(c.get("elapsed_ms")) for c in model_calls),
        "session_summary": state.summary if state else "",
        "summary_kind": "latest_writeback",
    }


def _conversation(facts: list[dict[str, Any]]) -> dict[str, Any]:
    goal = ""
    for fact in facts:
        if fact.get("event") != "original_goal_updated":
            continue
        detail = fact.get("detail", {})
        if isinstance(detail, Mapping):
            goal = str(detail.get("goal_preview", ""))
            break
    return {"user_goal": goal or _checkpoint_message(facts)}


def _model_calls(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for fact in facts:
        if fact.get("event") != "llm:response":
            continue
        summary = fact.get("summary", {})
        obs = summary.get("observation", {}) if isinstance(summary, Mapping) else {}
        evidence = summary.get("evidence", {}) if isinstance(summary, Mapping) else {}
        calls.append(_model_call(len(calls) + 1, fact, summary, obs, evidence))
    return calls


def _model_call(
    idx: int,
    fact: Mapping[str, Any],
    summary: Mapping[str, Any],
    obs: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "index": idx,
        "ts": fact.get("ts", ""),
        "stage": obs.get("stage", ""),
        "provider": obs.get("provider", ""),
        "model": obs.get("model", ""),
        "prompt_tokens": _int(obs.get("prompt_tokens")),
        "completion_tokens": _int(obs.get("completion_tokens")),
        "elapsed_ms": _int(obs.get("elapsed_ms")),
        "has_final": bool(summary.get("has_final")),
        "has_run_tools": bool(summary.get("has_run_tools")),
        "error": summary.get("error"),
        "evidence": dict(evidence),
    }


def _tool_calls(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for fact in facts:
        event = fact.get("event")
        if event not in {"tool:request", "tool:response"}:
            continue
        tool = fact.get("tool", {})
        if not isinstance(tool, Mapping):
            continue
        call_id = str(tool.get("call_id", ""))
        if call_id not in rows:
            rows[call_id] = {"call_id": call_id}
            order.append(call_id)
        _merge_tool(rows[call_id], event, fact, tool)
    return [rows[call_id] for call_id in order]


def _merge_tool(
    row: dict[str, Any],
    event: object,
    fact: Mapping[str, Any],
    tool: Mapping[str, Any],
) -> None:
    if event == "tool:request":
        row.update(
            {
                "name": tool.get("name", ""),
                "risk": tool.get("risk", ""),
                "args": tool.get("args_summary", {}),
                "requested_at": fact.get("ts", ""),
            }
        )
        return
    row.update(
        {
            "status": tool.get("status", ""),
            "error": tool.get("error"),
            "summary": tool.get("summary", ""),
            "output_summary": tool.get("output_summary", ""),
            "responded_at": fact.get("ts", ""),
        }
    )
    row["elapsed_seconds"] = _elapsed_seconds(row)


def _transitions(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ts": fact.get("ts", ""),
            "from": fact.get("from_state", ""),
            "to": fact.get("to_state", ""),
            "close_reason": fact.get("close_reason", ""),
        }
        for fact in facts
        if fact.get("event") == "state:transition"
    ]


def _warnings(
    facts: list[dict[str, Any]],
    state: SessionState | None,
    model_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, str]]:
    warnings = []
    if state and state.summary:
        warnings.append(
            _warning("summary", "会话摘要是最近一次写回，不等于完整会话总结。")
        )
    if any(_int(t.get("elapsed_seconds")) >= 60 for t in tool_calls):
        warnings.append(_warning("latency", "至少一个工具调用耗时超过 60 秒。"))
    if _has_secondary_source_only(tool_calls):
        warnings.append(
            _warning("source", "结论可能依赖二手网页来源，建议核对一手来源。")
        )
    if not model_calls:
        warnings.append(_warning("model", "没有记录到模型响应事实。"))
    return warnings


def _evidence_warnings(
    model_calls: list[dict[str, Any]],
    read_raw: Callable[[str], Mapping[str, Any]],
) -> list[dict[str, str]]:
    warnings = []
    for call in model_calls:
        evidence = call.get("evidence", {})
        if not isinstance(evidence, Mapping):
            continue
        parsed_path = str(evidence.get("parsed_plan", ""))
        if not parsed_path or parsed_path.startswith("("):
            continue
        raw = read_raw(parsed_path)
        data = raw.get("data") if isinstance(raw, Mapping) else None
        if _looks_like_protocol_text(data):
            warnings.append(_warning("protocol", "最终输出可能包含工具协议文本。"))
            break
    return warnings


def _looks_like_protocol_text(data: Any) -> bool:
    if not isinstance(data, Mapping):
        return False
    text = str(data.get("final_output", "")).lstrip()
    return (
        text.startswith("[tool_result]")
        or text.startswith("return {")
        or text.startswith('[{"type":"final"')
    )


def _has_secondary_source_only(tool_calls: list[dict[str, Any]]) -> bool:
    joined = " ".join(str(t.get("output_summary", "")) for t in tool_calls)
    return "juejin.cn" in joined and "gist.github.com" not in joined


def _warning(kind: str, message: str) -> dict[str, str]:
    return {"kind": kind, "message": message}


def _elapsed_seconds(row: Mapping[str, Any]) -> int:
    start = _parse_time(str(row.get("requested_at", "")))
    end = _parse_time(str(row.get("responded_at", "")))
    return max(0, end - start) if start and end else 0


def _parse_time(value: str) -> int:
    try:
        from datetime import datetime

        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def _checkpoint_message(facts: list[dict[str, Any]]) -> str:
    for fact in facts:
        checkpoint = fact.get("checkpoint", {})
        if not isinstance(checkpoint, Mapping):
            continue
        snapshot = checkpoint.get("working_memory_snapshot", {})
        payload = snapshot.get("payload", {}) if isinstance(snapshot, Mapping) else {}
        if isinstance(payload, Mapping) and payload.get("message"):
            return str(payload["message"])
    return ""


def _terminal_status(facts: list[dict[str, Any]]) -> str:
    observation = build_run_observation(facts).lifecycle
    if observation.lifecycle:
        return observation.lifecycle
    for fact in reversed(facts):
        if fact.get("event") == "state:transition":
            to_state = str(fact.get("to_state", ""))
            if to_state in {"DONE", "PAUSED", "FAILED"}:
                return to_state.lower()
    return "running"


def _last_event(facts: list[dict[str, Any]]) -> str:
    return str(facts[-1].get("event", "")) if facts else ""


def _first_text(facts: list[dict[str, Any]], key: str) -> str:
    for fact in facts:
        value = fact.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def session_state_payload(state: SessionState | None) -> dict[str, Any]:
    return asdict(state) if state is not None else {}


__all__ = ["build_run_story", "session_state_payload"]
