"""从当前分支的真实输入和操作构造知识出处，模型不能自报用户身份。

作者：xxx
时间：2026-09-15 05:40:00
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast
from dataclasses import replace

from llm.messages import ToolResultMessage, UserMessage, agent_message_to_mapping
from memory.records import MemorySource
from runtime.native_actions import NativeActionContext
from runtime.session_message_store import MaterializedSession, SessionMessageStore
from runtime.session_compaction import source_digest
from runtime.tool_operations import ToolOperation


def knowledge_sources(
    context: NativeActionContext, call: ToolOperation
) -> tuple[MemorySource, ...]:
    """将来源选择解析到真实记录；传参：当前运行依赖与模型操作；返回：可追溯引用。"""
    mode = str(call.args.get("source_mode", "current_input"))
    if mode == "inference":
        return (
            MemorySource(
                "model_inference",
                call.operation_id,
                session_id=context.run.session_id,
                run_id=context.run.run_id,
            ),
        )
    current = knowledge_source_view(context, origin=mode.startswith("origin_"))
    if mode in {"tool_results", "origin_results"}:
        return _tool_sources(context, call, view=current)
    if mode not in {"current_input", "user_inputs", "origin_inputs"}:
        raise ValueError("unknown knowledge source mode")
    agent_ids = {
        entry.entry_id for entry in current.entries if entry.input_source == "agent"
    }
    user_ids = [
        message.message_id
        for message in current.messages
        if isinstance(message, UserMessage) and message.message_id not in agent_ids
    ]
    requested = (
        user_ids[-1:]
        if mode == "current_input"
        else cast(
            list[str],
            call.args.get(
                "source_input_ids", user_ids[-1:] if mode == "origin_inputs" else []
            ),
        )
    )
    entry_messages = {
        entry.entry_id: entry.message.message_id
        for entry in current.entries
        if isinstance(entry.message, UserMessage)
    }
    requested = [entry_messages.get(identity, identity) for identity in requested]
    if not requested or any(identity not in user_ids for identity in requested):
        raise ValueError(
            "knowledge source must be a delivered user input in this branch. "
            "For accepted background work use source_mode=origin_inputs and omit source_input_ids "
            "to cite the latest original user input; use knowledge_read to find other user message IDs. "
            "A branch/source_entry_id anchor is not a user input ID."
        )
    entries = {
        entry.message.message_id: entry
        for entry in current.entries
        if isinstance(entry.message, UserMessage)
    }
    return tuple(
        MemorySource(
            "user_input",
            entries[identity].entry_id,
            session_id=current.session_id,
            run_id=entries[identity].run_id or "",
            observed_at=entries[identity].timestamp,
        )
        for identity in requested
    )


def _tool_sources(
    context: NativeActionContext, call: ToolOperation, *, view: MaterializedSession
) -> tuple[MemorySource, ...]:
    """固定当前分支已经落盘的实际回执，失败也可支持经验；传参：依赖和引用；返回：有状态的原件引用。"""
    requested = cast(list[str], call.args.get("source_operation_ids", []))
    receipts = {
        message.call_id: message
        for message in view.messages
        if isinstance(message, ToolResultMessage)
    }
    records = {
        row["operation_id"]: row
        for row in context.operations.for_session(view.session_id)
        if row["call"]["call_id"] in receipts and row.get("result") is not None
    }
    if not requested or any(identity not in records for identity in requested):
        raise ValueError(
            "knowledge source must have a persisted tool receipt in this branch"
        )
    return tuple(
        MemorySource(
            "tool_result",
            str(identity),
            session_id=view.session_id,
            run_id=str(records[identity]["run_id"]),
            observed_at=str(records[identity]["updated_at"]),
            message_id=receipts[records[identity]["call"]["call_id"]].message_id,
            content_sha256=source_digest(
                (receipts[records[identity]["call"]["call_id"]],)
            ),
            result_status=receipts[records[identity]["call"]["call_id"]].status,
        )
        for identity in requested
    )


def knowledge_source_view(
    context: NativeActionContext, *, origin: bool
) -> MaterializedSession:
    """后台工作只读取接纳时冻结的来源分支，不跟随后来的切支或新输入；传参：运行与来源选择；返回：快照。"""
    if not origin:
        return context.messages.materialize(context.run.session_id)
    inherited = context.run.payload.get("knowledge_origin")
    if isinstance(inherited, dict):
        if inherited["source_session_id"] != context.run.payload.get(
            "source_session_id"
        ) or inherited["source_run_id"] != context.run.payload.get("source_run_id"):
            raise ValueError("knowledge source does not match its accepted work")
        if inherited.get("work_kind") == "knowledge_maintenance":
            return frozen_knowledge_source(context.messages, inherited)
        return context.messages.materialize(
            str(inherited["source_session_id"]),
            at_entry_id=str(inherited["source_entry_id"]),
        )
    source = context.run.payload
    fields = ("source_session_id", "source_run_id", "source_entry_id")
    if not source.get("schedule_id") or any(
        not isinstance(source.get(key), str) or not source[key] for key in fields
    ):
        raise ValueError("this run has no accepted knowledge source")
    return context.messages.materialize(
        str(source["source_session_id"]), at_entry_id=str(source["source_entry_id"])
    )


def frozen_knowledge_source(
    messages: SessionMessageStore, origin: Mapping[str, Any]
) -> MaterializedSession:
    """【知识维护】【来源预检】用同一投影核对冻结来源；参数：消息存储及冻结记录；返回：可回读原文，缺件或变更抛错。"""
    from runtime.workspaces import WorkspaceStore

    view = messages.materialize(
        str(origin["source_session_id"]), at_entry_id=str(origin["source_entry_id"])
    )
    if (
        WorkspaceStore(messages.database.data_root)
        .for_session(view.session_id)
        .workspace_id
        != origin["workspace_id"]
    ):
        raise ValueError("knowledge source workspace changed")
    current = messages.materialize(view.session_id)
    if origin["source_entry_id"] not in {entry.entry_id for entry in current.entries}:
        raise ValueError("knowledge source branch is no longer compatible")
    selected = tuple(
        message
        for message in view.messages
        if message.message_id in origin["message_ids"]
    )
    if [message.message_id for message in selected] != list(origin["message_ids"]):
        raise ValueError("knowledge source frozen messages are missing or out of order")
    if source_digest(selected) != origin["source_sha256"]:
        raise ValueError("knowledge source content changed")
    return replace(view, messages=selected, pending_tool_calls=())


def read_stored_sources(
    context: NativeActionContext, sources: tuple[MemorySource, ...]
) -> list[dict[str, object]]:
    """按已保存知识引用补读跨会话原件，不接受任意外部会话ID；传参：依赖和已验证引用；返回：来源原文。"""
    result: list[dict[str, object]] = []
    for source in sources:
        reference = {
            "kind": source.kind,
            "reference": source.reference,
            "session_id": source.session_id,
            "run_id": source.run_id,
        }
        if source.kind == "user_input":
            entries = context.messages.read_entries(source.session_id)
            entry = next(
                (item for item in entries if item.entry_id == source.reference), None
            )
            if (
                entry is None
                or not isinstance(entry.message, UserMessage)
                or entry.input_source == "agent"
            ):
                raise ValueError(
                    "stored source no longer resolves to its original user input"
                )
            result.append({**reference, "available": True, "entry": entry.to_mapping()})
        elif source.kind == "tool_result" and source.message_id:
            entries = context.messages.read_entries(source.session_id)
            message = next(
                (
                    entry.message
                    for entry in entries
                    if entry.message is not None
                    and entry.message.message_id == source.message_id
                ),
                None,
            )
            if (
                not isinstance(message, ToolResultMessage)
                or source_digest((message,)) != source.content_sha256
            ):
                raise ValueError(
                    "stored tool receipt is missing or its source content changed"
                )
            result.append(
                {
                    **reference,
                    "available": True,
                    "result_status": source.result_status,
                    "message_id": source.message_id,
                    "receipt": agent_message_to_mapping(message),
                }
            )
        elif (
            source.session_id
            and source.run_id
            and source.kind in {"tool_result", "model_inference"}
        ):
            operation = context.operations.load(
                {
                    "session_id": source.session_id,
                    "run_id": source.run_id,
                    "operation_id": source.reference,
                }
            )
            if operation is None:
                raise FileNotFoundError(
                    f"stored source operation is missing: {source.reference}"
                )
            result.append(
                {
                    **reference,
                    "available": True,
                    "state": operation["state"],
                    "tool": operation["call"]["tool_name"],
                    "result": operation.get("result"),
                }
            )
        else:
            result.append(
                {
                    **reference,
                    "available": False,
                    "reason": "no original input or operation reference was recorded",
                }
            )
    return result
