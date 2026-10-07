"""核对目标完成声明所引用的会话证据。

作者：xxx
时间：2026-09-13 20:00:00
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from llm.messages import (
    AssistantMessage,
    TextPart,
    ToolResultMessage,
    UserMessage,
    agent_message_to_mapping,
)
from runtime.ledger import LedgerEvent
from runtime.session_message_store import MaterializedSession, SessionEntry


def resolve_completion_evidence(
    session: MaterializedSession,
    evidence: Sequence[Mapping[str, object]],
    *,
    task_id: str,
    operation_refs: Mapping[str, str] | None = None,
    confirmations: Sequence[LedgerEvent] = (),
    expected_revision: int | None = None,
    run_id: str | None = None,
) -> list[dict[str, object]]:
    """从当前分支解析完成引用，核对成功结果及原始目标归属。

    传参：session 为会话快照；evidence 为模型引用与说明；task_id 为目标
    返回：带来源和内容摘要的引用；缺失、失败或跨目标证据明确拒绝
    """
    if not evidence:
        raise ValueError("goal completion requires result evidence")
    resolved: list[dict[str, object]] = []
    for item in evidence:
        kind = item.get("kind")
        reference = item.get("reference")
        reason = item.get("reason")
        if kind not in ("tool_result", "answer", "user_confirmation"):
            raise ValueError(
                "completion evidence kind must be tool_result, answer or user_confirmation"
            )
        if (
            not isinstance(reference, str)
            or not reference.strip()
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("completion evidence needs a reference and an explanation")
        lookup = (
            (operation_refs or {}).get(reference, reference)
            if kind == "tool_result"
            else reference
        )
        if kind == "user_confirmation":
            resolved.append(
                _confirmation_evidence(
                    session,
                    item,
                    confirmations,
                    task_id=task_id,
                    expected_revision=expected_revision,
                )
            )
            continue
        if kind == "answer" and reference == "current_answer":
            answers = [
                entry.message.message_id
                for entry in session.entries
                if entry.message is not None
                and entry.task_id == task_id
                and entry.run_id == run_id
                and _is_answer(entry)
            ]
            lookup = answers[-1] if answers else ""
        entry = _find_evidence(session.entries, str(kind), lookup)
        if entry is None or entry.task_id != task_id:
            raise ValueError(
                f"completion evidence is missing or belongs to another goal: {reference}"
            )
        message = entry.message
        if isinstance(message, ToolResultMessage) and message.status != "success":
            raise ValueError(
                f"completion evidence is not a successful result: {reference}"
            )
        if isinstance(message, ToolResultMessage) and message.tool_name == "ask_user":
            raise ValueError(
                "asking a question is not completion evidence or user confirmation"
            )
        assert message is not None
        encoded = json.dumps(
            agent_message_to_mapping(message), ensure_ascii=False, sort_keys=True
        )
        row: dict[str, object] = {
            "kind": kind,
            "reference": message.message_id if kind == "answer" else reference,
            "reason": reason.strip(),
            "session_id": session.session_id,
            "entry_id": entry.entry_id,
            "message_id": message.message_id,
            "run_id": entry.run_id,
            "task_id": task_id,
            "content_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        }
        if isinstance(message, ToolResultMessage):
            row["call_id"] = message.call_id
            row["artifact_refs"] = list(message.artifact_refs)
            if lookup != reference:
                row["operation_id"] = reference
        resolved.append(row)
    return resolved


def _find_evidence(
    entries: tuple[SessionEntry, ...], kind: str, reference: str
) -> SessionEntry | None:
    """按消息或调用身份定位当前分支证据；传参：条目、类别与引用；返回：唯一条目。"""
    matches: list[SessionEntry] = []
    for entry in entries:
        message = entry.message
        if (
            kind == "tool_result"
            and isinstance(message, ToolResultMessage)
            and message.message_id == reference
        ):
            return entry
        if (
            kind == "tool_result"
            and isinstance(message, ToolResultMessage)
            and message.call_id == reference
        ):
            matches.append(entry)
        if (
            kind == "answer"
            and _is_answer(entry)
            and message is not None
            and message.message_id == reference
        ):
            return entry
    if len(matches) > 1:
        raise ValueError(
            f"completion reference is ambiguous; use its message_id: {reference}"
        )
    return matches[0] if matches else None


def _is_answer(entry: SessionEntry) -> bool:
    """只把有真实可见正文的助手消息视为文字成果；传参：会话条目；返回：是否有答复。"""
    return isinstance(entry.message, AssistantMessage) and any(
        isinstance(part, TextPart) and part.text.strip()
        for part in entry.message.content
    )


def _confirmation_evidence(
    session: MaterializedSession,
    item: Mapping[str, object],
    confirmations: Sequence[LedgerEvent],
    *,
    task_id: str,
    expected_revision: int | None,
) -> dict[str, object]:
    """核对用户明确动作与当前目标版本及分支；传参：会话、引用、事件及目标版本；返回：完成凭据。"""
    event = next(
        (row for row in confirmations if row.event_id == item["reference"]), None
    )
    if (
        event is None
        or event.event != "goal.confirmation_decided"
        or event.source != "user_action"
        or event.session_id != session.session_id
        or event.task_id != task_id
    ):
        raise ValueError("completion confirmation requires a recorded user action")
    payload = event.payload
    if (
        payload.get("accepted") is not True
        or payload.get("expected_revision") != expected_revision
    ):
        raise ValueError("completion confirmation was rejected or its revision changed")
    by_id = {entry.entry_id: entry for entry in session.entries}
    source = by_id.get(str(payload.get("source_input_id")))
    branch = next(
        (
            entry.entry_id
            for entry in reversed(session.entries)
            if entry.type == "branch"
        ),
        None,
    )
    if (
        source is None
        or not isinstance(source.message, UserMessage)
        or source.input_source != "user"
        or payload.get("branch_anchor") not in by_id
        or payload.get("branch_id") != branch
    ):
        raise ValueError("completion confirmation source or branch is no longer valid")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or any(
        row.get("kind") == "user_confirmation" for row in evidence
    ):
        raise ValueError("completion confirmation must reference delivered work")
    delivered = resolve_completion_evidence(session, evidence, task_id=task_id)
    digest = hashlib.sha256(
        json.dumps(delivered, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if digest != payload.get("evidence_digest"):
        raise ValueError("completion confirmation evidence changed")
    return {
        "kind": "user_confirmation",
        "reference": event.event_id,
        "reason": item["reason"],
        "session_id": session.session_id,
        "entry_id": source.entry_id,
        "message_id": source.message.message_id,
        "run_id": event.run_id,
        "task_id": task_id,
        "content_sha256": payload["evidence_digest"],
        "question_id": payload["question_id"],
        "expected_revision": expected_revision,
    }
