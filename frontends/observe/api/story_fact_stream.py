from __future__ import annotations

from typing import Any, Mapping


def build_fact_stream(
    facts: list[dict[str, Any]],
    model_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model_by_ts = {str(call.get("ts", "")): call for call in model_calls}
    return [_fact_row(index, fact, model_by_ts) for index, fact in enumerate(facts, 1)]


def _fact_row(
    index: int,
    fact: Mapping[str, Any],
    model_by_ts: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    event = str(fact.get("event", ""))
    row = {
        "index": index,
        "event": event,
        "timestamp": str(fact.get("ts", "")),
        "title": _fact_title(event),
        "detail": _fact_detail(event, fact),
        "raw": dict(fact),
    }
    if event == "llm:response":
        row["model_call"] = model_by_ts.get(str(fact.get("ts", "")))
    return row


def _fact_title(event: str) -> str:
    titles = {
        "checkpoint:saved": "Checkpoint",
        "context:built": "上下文构建",
        "context:segments": "上下文分段",
        "llm:response": "模型响应",
        "original_goal_updated": "目标更新",
        "run:start": "运行开始",
        "run:lifecycle": "生命周期边界",
        "state:transition": "历史状态迁移",
        "tool:request": "工具请求",
        "tool:response": "工具响应",
    }
    return titles.get(event, event or "事件")


def _fact_detail(event: str, fact: Mapping[str, Any]) -> str:
    if event == "run:lifecycle":
        lifecycle = fact.get("lifecycle", "—")
        reason = fact.get("reason", "—")
        return f"{lifecycle} / {reason}"
    if event == "state:transition":
        return f"{fact.get('from_state', '—')} -> {fact.get('to_state', '—')}"
    if event == "checkpoint:saved":
        return _checkpoint_detail(fact)
    if event == "context:segments":
        return _segments_detail(fact)
    if event.startswith("tool:"):
        return _tool_detail(fact)
    if event == "llm:response":
        return _model_detail(fact)
    return _short_value(fact.get("summary") or fact.get("detail") or "")


def _checkpoint_detail(fact: Mapping[str, Any]) -> str:
    checkpoint = fact.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        return "— / —"
    return f"{checkpoint.get('state', '—')} / {checkpoint.get('reason', '—')}"


def _segments_detail(fact: Mapping[str, Any]) -> str:
    raw_segments = fact.get("segments")
    segments = (
        [s for s in raw_segments if isinstance(s, Mapping)]
        if isinstance(raw_segments, list)
        else []
    )
    tokens = sum(_int(s.get("tokens_est")) for s in segments)
    return f"{len(segments)} 个分段，约 {tokens} tokens"


def _tool_detail(fact: Mapping[str, Any]) -> str:
    tool = fact.get("tool")
    if not isinstance(tool, Mapping):
        return "工具事件"
    value = tool.get("name") or tool.get("status") or tool.get("summary")
    return str(value or "工具事件")


def _model_detail(fact: Mapping[str, Any]) -> str:
    summary = fact.get("summary")
    if not isinstance(summary, Mapping):
        return "模型输出"
    return "请求工具" if summary.get("has_run_tools") else "模型输出"


def _short_value(value: Any) -> str:
    if value is None or value == "":
        return "—"
    text = str(value)
    return text if len(text) <= 120 else f"{text[:117]}..."


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["build_fact_stream"]
