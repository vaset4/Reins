from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from runtime.run_evidence import RunEvidenceStore

PREVIEW_LINES = 10
MAX_INLINE_CHARS = 240
MAX_SUMMARY_LINES = 6


@dataclass(frozen=True, slots=True)
class EvidenceSummaries:
    context: list[str]
    model_input: list[str]
    model_output: list[str]
    evidence: list[str]


@dataclass(frozen=True, slots=True)
class EvidenceRead:
    key: str
    raw_path: str
    status: str
    payload: Mapping[str, Any] | None


def read_evidence_summaries(
    data_root: Path,
    evidence: Mapping[str, object],
) -> EvidenceSummaries:
    request = _read_evidence(data_root, evidence, key="model_request")
    response = _read_evidence(data_root, evidence, key="model_response")
    parsed = _read_evidence(data_root, evidence, key="parsed_plan")
    errors = _read_evidence(data_root, evidence, key="errors")
    return EvidenceSummaries(
        context=_context_summary(request),
        model_input=_input_summary(request),
        model_output=_output_summary(response, parsed, errors),
        evidence=_evidence_summary([request, response, parsed, errors]),
    )


def preview_raw_evidence(data_root: Path, raw_path: str) -> list[str]:
    if not raw_path:
        return ["missing: no evidence path"]
    if raw_path.startswith("("):
        return [f"not_available: {raw_path}"]
    payload = RunEvidenceStore(data_root).read_reference(raw_path)
    return (
        ["missing: evidence not found"]
        if payload is None
        else _preview_text(json.dumps(payload, ensure_ascii=False, indent=2))
    )


def _read_evidence(
    data_root: Path,
    evidence: Mapping[str, object],
    *,
    key: str,
) -> EvidenceRead:
    raw_path = str(evidence.get(key, "") or "")
    if not raw_path:
        return EvidenceRead(key, "", "missing", None)
    if raw_path.startswith("("):
        return EvidenceRead(key, raw_path, raw_path, None)
    payload = RunEvidenceStore(data_root).read_reference(raw_path)
    return EvidenceRead(
        key, raw_path, "ok" if payload is not None else "missing", payload
    )


def _context_summary(request: EvidenceRead) -> list[str]:
    if request.payload is None:
        return [f"model_request: {request.status}"]
    context = _mapping(request.payload.get("prompt_context"))
    lines = [
        f"stage: {context.get('stage', '(none)')}",
        f"system_reminder: {context.get('system_reminder', '(none)')}",
        f"render_text_chars: {len(str(request.payload.get('render_text_to_model', '')))}",
    ]
    summary = str(context.get("context_summary", "")).strip()
    if not summary:
        return lines + ["context_summary: (empty)"]
    return lines + _summary_lines(summary)


def _input_summary(request: EvidenceRead) -> list[str]:
    if request.payload is None:
        return [f"model_request: {request.status}"]
    request_payload = _mapping(request.payload.get("request"))
    messages = _list(request_payload.get("messages"))
    roles = _message_roles(messages)
    return [
        f"protocol_mode: {request.payload.get('protocol_mode', '(none)')}",
        f"messages: {len(messages)}",
        f"roles: {roles or '(none)'}",
        f"render_text_chars: {len(str(request.payload.get('render_text_to_model', '')))}",
    ]


def _output_summary(
    response: EvidenceRead,
    parsed: EvidenceRead,
    errors: EvidenceRead,
) -> list[str]:
    lines = _response_lines(response)
    lines.extend(_parsed_lines(parsed))
    if errors.status != "missing":
        lines.append(f"errors: {errors.status}")
    return lines


def _response_lines(response: EvidenceRead) -> list[str]:
    if response.payload is None:
        return [f"model_response: {response.status}"]
    payload = _mapping(response.payload.get("response"))
    tool_calls = _list(payload.get("tool_calls"))
    return [
        f"response_ok: {payload.get('ok', '(none)')}",
        f"text_chars: {len(str(payload.get('text', '') or ''))}",
        f"tool_calls: {len(tool_calls)}",
        f"error_message: {payload.get('error_message') or '(none)'}",
    ]


def _parsed_lines(parsed: EvidenceRead) -> list[str]:
    if parsed.payload is None:
        return [f"parsed_plan: {parsed.status}"]
    request = _mapping(parsed.payload.get("run_tools_request"))
    tool_name = request.get("tool_name") or request.get("action") or "(none)"
    return [
        f"parsed_success: {parsed.payload.get('success', '(none)')}",
        f"has_final: {parsed.payload.get('has_final', '(none)')}",
        f"has_run_tools: {parsed.payload.get('has_run_tools', '(none)')}",
        f"tool_name: {tool_name}",
    ]


def _evidence_summary(reads: list[EvidenceRead]) -> list[str]:
    return [
        f"{read.key}: {read.raw_path or '(missing)'} [{read.status}]" for read in reads
    ]


def _summary_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [f"context_summary: {_truncate(line)}" for line in lines[:MAX_SUMMARY_LINES]]


def _message_roles(messages: list[object]) -> str:
    counts: dict[str, int] = {}
    for message in messages:
        role = str(_mapping(message).get("role", "unknown"))
        counts[role] = counts.get(role, 0) + 1
    return ", ".join(f"{role}={count}" for role, count in sorted(counts.items()))


def _preview_text(text: str) -> list[str]:
    lines = text.splitlines()
    preview = [_truncate(line) for line in lines[:PREVIEW_LINES]]
    if len(lines) > PREVIEW_LINES:
        preview.append(f"... ({len(lines) - PREVIEW_LINES} more lines)")
    return preview or ["(empty)"]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _truncate(text: str) -> str:
    if len(text) <= MAX_INLINE_CHARS:
        return text
    return text[: MAX_INLINE_CHARS - 1] + "..."


__all__ = ["EvidenceSummaries", "preview_raw_evidence", "read_evidence_summaries"]
