"""在实际请求窗口内投影工具回执，完整来源仍可用现有产物分页读取。

作者：xxx
时间：2026-09-15 05:40:00
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
import path_security

from context.artifact_ref import store_large_output
from context.compaction import RECENT_WINDOW_DIVISOR
from context.materials import ContextMaterial, artifact_reference, fit_materials
from context.selection_store import (
    content_digest,
    material_key,
    prepared_selection,
    selection_candidate,
)
from context.token_estimate import estimate_agent_messages_tokens
from context.window import request_budget
from llm.messages import (
    AgentMessage,
    TextPart,
    ToolResultMessage,
    agent_message_from_mapping,
    agent_message_to_mapping,
    group_tool_call_units,
    model_visible_text,
)
from llm.model_request import ComposedRequest
from llm.prompt_composer import render_bundle_text
from runtime.tool_operations import ToolOperationStore
from runtime.persistence import RuntimeStore
from runtime.lease import Lease
from tools.file_resources import FILE_WRITERS


class ToolResultViews:
    """为已完成调用组分配实际可用窗口，不改变会话原件、调用参数或执行状态。"""

    def __init__(
        self,
        data_root: Path,
        *,
        task_id: str,
        prepare: Callable[[str, Mapping[str, object]], ComposedRequest],
    ) -> None:
        """接入现有请求组装与产物存储；传参：数据根、材料归属和准备函数；返回：无。"""
        self.data_root, self.task_id, self.prepare_request = data_root, task_id, prepare

    def prepare(self, task: str, context: Mapping[str, object]) -> ComposedRequest:
        """按完整请求的指令、工具、协议和输出额度投影结果；传参：任务与上下文；返回：同源请求及证据。"""
        # 1. 【上下文】【材料分配】按工具回执投影后的实际容量选择材料，避免提前移走仍能放下的技能正文
        context = {**context, "data_root": str(self.data_root)}
        composed = fit_materials(
            task, context, self._prepare_result_views, retain=self._retain_material
        )
        request, messages = composed.request, composed.messages
        layers = [
            {
                "name": section.name,
                "cache_tier": section.layer,
                "tokens_est": section.token_count,
                "trimmable": False,
                "contains_selectable_materials": section.name == "runtime_context",
                "source": section.source,
            }
            for section in composed.prompt_sections
            if section.role_target == "system"
        ]
        layers.extend(
            {
                "name": message.message_id,
                "cache_tier": "dynamic",
                "tokens_est": estimate_agent_messages_tokens((message,)),
                "trimmable": message.message_id != context.get("input_message_id"),
                "source": message.kind,
            }
            for message in messages
        )
        final_budget = request_budget(request, composed.context_window)
        layers.extend(
            {
                "name": name,
                "cache_tier": "request",
                "tokens_est": cost,
                "trimmable": False,
                "source": "ModelRequest",
            }
            for name, cost in (
                ("tool_definitions", final_budget.tools),
                ("protocol", final_budget.protocol),
                ("output_reserved", final_budget.output_reserved),
            )
        )
        return replace(
            composed,
            request=request,
            render_text_to_model=render_bundle_text(
                messages, request.instructions, observations=request.observations
            ),
            trim_delta={
                **(composed.trim_delta or {}),
                "layers": layers,
                "task_state_competes_with_history": False,
                "layer_estimate_overhead": final_budget.required_total
                - sum(cast(int, layer["tokens_est"]) for layer in layers),
            },
            token_estimate={
                **composed.token_estimate,
                **request_budget(request, composed.context_window).evidence(),
            },
        )

    def _prepare_result_views(
        self, task: str, context: Mapping[str, object]
    ) -> ComposedRequest:
        """在每次材料选择后重算文件与回执预览；传参：任务和候选上下文；返回：原件可读的同源请求。"""
        composed = self.prepare_request(task, context)
        messages, evidence = self._file_views(
            composed.messages,
            session_id=str(context.get("session_id", "")),
            lease=context.get("capability_lease"),
        )
        saved = prepared_selection(context)
        protected = _current_read_ids(messages)
        protected.update(
            str(row["message_id"])
            for row in evidence
            if row["reason"] == "permission_removed"
        )
        originals = {
            message.message_id: message
            for message in composed.messages
            if isinstance(message, ToolResultMessage)
        }
        changed = {row["message_id"] for row in evidence}
        messages = tuple(
            _selected_tool_view(message, saved, protected, changed)
            for message in messages
        )
        request = replace(composed.request, messages=messages)
        budget = request_budget(request, composed.context_window)
        available = (
            budget.context_window
            - budget.instructions
            - budget.tools
            - budget.protocol
            - budget.output_reserved
        )
        if available > 0:
            groups = group_tool_call_units(messages)
            messages = tuple(
                message
                for group in groups
                for message in self._fit_group(
                    group, available=available, protected=protected
                )
            )
        request = replace(request, messages=messages)
        catalog = [
            tool_material_row(
                originals[message.message_id],
                session_id=str(context.get("session_id", "")),
                projected=message,
                protected=message.message_id in protected,
            )
            for message in messages
            if isinstance(message, ToolResultMessage)
        ]
        return replace(
            composed,
            request=request,
            render_text_to_model=render_bundle_text(
                messages, request.instructions, observations=request.observations
            ),
            material_selection=selection_candidate(saved, catalog),
            trim_delta={
                **(composed.trim_delta or {}),
                "file_reads": evidence,
                "tool_materials": [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "rendered_message"
                    }
                    for row in catalog
                ],
            },
            token_estimate={
                **composed.token_estimate,
                **request_budget(request, composed.context_window).evidence(),
            },
        )

    def _file_views(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        session_id: str,
        lease: object | None = None,
    ) -> tuple[tuple[AgentMessage, ...], list[dict[str, object]]]:
        """按当前分支实际成功读写判断旧读取，容量未满也生效；传参：本轮消息及会话；返回：历史投影与证据。"""
        if not session_id or not any(
            isinstance(message, ToolResultMessage) for message in messages
        ):
            return messages, []
        records = {
            row["call"]["call_id"]: row
            for row in ToolOperationStore(self.data_root).for_session(session_id)
        }
        restores, positions = _restore_observations(self.data_root, session_id, records)
        latest: dict[str, tuple[str, str, tuple[int, int] | None]] = {}
        replacements: dict[str, ToolResultMessage] = {}
        evidence: list[dict[str, object]] = []
        for message in reversed(messages):
            if not isinstance(message, ToolResultMessage):
                continue
            record = records.get(message.call_id)
            resource = (record or {}).get("call", {}).get("resource") or {}
            if (
                isinstance(lease, Lease)
                and resource.get("known")
                and path_security.check_read(
                    Path(resource["path"]), lease, filtered=True
                )
                is not path_security.Decision.ALLOWED
            ):
                # 1. 【上下文】【权限失效】撤销读取权限后实际移除旧正文及附件，不能靠附加忽略提示代替
                replacements[message.message_id] = replace(
                    message,
                    content=(
                        TextPart(
                            "Previous tool content is unavailable under the current read permissions."
                        ),
                    ),
                    error="Tool failed; original error details are unavailable under the current read permissions."
                    if message.error
                    else None,
                    artifact_refs=(),
                )
                evidence.append(
                    {
                        "message_id": message.message_id,
                        "reason": "permission_removed",
                        "representation": "unavailable",
                    }
                )
                continue
            if message.status != "success":
                continue
            observation = _file_observation(record, message)
            if observation is None:
                continue
            path, version, span, writing = observation
            newer = latest.get(path)
            restored = restores.get(path)
            position = positions.get(message.call_id)
            if (
                writing
                or position is None
                or (
                    restored is not None
                    and (restored.sequence <= position or restored.version == version)
                )
            ):
                restored = None
            reason = ""
            if not writing and newer is not None:
                if newer[0] != version:
                    reason = "superseded_file_version"
                elif (
                    span is not None
                    and newer[2] is not None
                    and newer[2][0] <= span[0]
                    and span[1] <= newer[2][1]
                ):
                    reason = "covered_by_later_read"
            # 2. 【文件恢复】【读取失效】按真实提交顺序识别后来的恢复，原会话消息和工具回执保持原样
            if restored is not None:
                reason = (
                    "file_restore_outcome_unknown"
                    if restored.unknown
                    else "file_restored"
                )
            if reason:
                assert newer is not None or restored is not None
                reference = self._retain(message)
                payload: dict[str, Any] = {
                    "file_read_state": "historical",
                    "reason": reason,
                    "message_id": message.message_id,
                    "source_sha256": version,
                    "later_message_id": newer[1]
                    if restored is None and newer
                    else None,
                    "later_sha256": newer[0] if restored is None and newer else None,
                    "read_original": {
                        "tool": "read_artifact",
                        "arguments": {"artifact_id": reference, "mode": "full"},
                    },
                    "read_current": {"tool": "file_read", "arguments": {"path": path}},
                    "notice": "This old read does not describe the latest observed file; read_current performs a new observation.",
                }
                if restored is not None:
                    payload.update(
                        file_restore_operation_id=restored.operation_id,
                        file_restore_sequence=restored.sequence,
                        notice="A later file restore changed this file or left its outcome unconfirmed; read_current checks the actual file.",
                    )
                replacements[message.message_id] = replace(
                    message,
                    content=(TextPart(json.dumps(payload, ensure_ascii=False)),),
                    artifact_refs=(*message.artifact_refs, reference),
                )
                evidence.append(
                    {
                        "message_id": message.message_id,
                        "source": "file_restore_operation"
                        if restored
                        else "ToolOperationStore",
                        "version": version,
                        "representation": "reference",
                        "reason": reason,
                        "original_tokens": estimate_agent_messages_tokens((message,)),
                        "final_tokens": estimate_agent_messages_tokens(
                            (replacements[message.message_id],)
                        ),
                        "read_action": payload["read_original"],
                    }
                )
            if newer is None or (
                newer[0] == version and newer[2] is None and span is not None
            ):
                latest[path] = (version, message.message_id, span)
        return tuple(
            replacements.get(message.message_id, message) for message in messages
        ), evidence

    def _fit_group(
        self, group: tuple[AgentMessage, ...], *, available: int, protected: set[str]
    ) -> tuple[AgentMessage, ...]:
        """同一批调用共同分配近期窗口，小结果优先完整保留；传参：调用组和消息额度；返回：完整配对的投影。"""
        target = available // RECENT_WINDOW_DIVISOR
        results = [
            message for message in group if isinstance(message, ToolResultMessage)
        ]
        if not results or estimate_agent_messages_tokens(group) <= target:
            return group
        remaining = target - estimate_agent_messages_tokens(
            tuple(item for item in group if not isinstance(item, ToolResultMessage))
        )
        chosen: dict[str, ToolResultMessage] = {}
        results.sort(key=lambda item: estimate_agent_messages_tokens((item,)))
        for index, message in enumerate(results):
            allowance = remaining // (len(results) - index)
            projected = message
            if (
                message.message_id not in protected
                and estimate_agent_messages_tokens((message,)) > allowance
            ):
                reference = self._retain(message)
                projected = _fit_preview(
                    message, reference=reference, allowance=allowance
                )
            chosen[message.message_id] = projected
            remaining -= estimate_agent_messages_tokens((projected,))
        return tuple(chosen.get(message.message_id, message) for message in group)

    def _retain(self, message: ToolResultMessage) -> str:
        """仅为被缩短的回执保存稳定原件，恢复或重新组装复用同一引用；传参：原回执；返回：产物编号。"""
        body = json.dumps(agent_message_to_mapping(message), ensure_ascii=False)
        reference = store_large_output(
            self.data_root,
            self.task_id,
            body,
            threshold=0,
            source_id=message.message_id,
            summary=f"{message.tool_name} complete receipt {message.call_id}",
        )
        assert reference is not None
        return reference.artifact_id

    def _retain_material(self, material: ContextMaterial) -> str:
        """在缩成入口之前保存临时扩展原件；传参：已进入请求边界的材料；返回：真实产物读取入口。"""
        reference = store_large_output(
            self.data_root,
            self.task_id,
            material.text,
            threshold=0,
            source_id=material.identity,
            summary=f"{material.source} context material",
        )
        assert reference is not None
        return artifact_reference(material, reference.artifact_id)


def _current_read_ids(messages: tuple[AgentMessage, ...]) -> set[str]:
    """补读原文的当前调用组先保持可用；参数：已选消息；返回：受保护回执身份。"""
    groups = group_tool_call_units(messages)
    latest = groups[-1] if groups else ()
    return {
        message.message_id
        for message in latest
        if isinstance(message, ToolResultMessage)
        and message.tool_name
        in {"read_artifact", "read_history", "memory_query", "skill_read"}
    }


def tool_material_row(
    message: ToolResultMessage,
    *,
    session_id: str,
    projected: ToolResultMessage | None = None,
    protected: bool = False,
) -> dict[str, Any]:
    """建立工具原件与实际表示的精确关系；参数：原回执及投影；返回：目录行，不复制完整原件。"""
    version = content_digest(agent_message_to_mapping(message))
    changed = projected is not None and projected != message
    row: dict[str, Any] = {
        "identity": f"tool:{message.message_id}",
        "source": "tool_result",
        "version": version,
        "source_digest": version,
        "scope": session_id,
        "message_id": message.message_id,
        "representation": "preview" if changed else "full",
        "protected": protected,
        "read_reference": "",
        "reason": "current_read" if protected else "fits_window",
    }
    if changed:
        row["rendered_message"] = agent_message_to_mapping(
            cast(ToolResultMessage, projected)
        )
        body = model_visible_text(cast(ToolResultMessage, projected))
        if body.startswith("{"):
            payload = json.loads(body)
            if payload.get("context_released") is True:
                row.update(
                    representation="reference",
                    reason="explicit_release",
                    read_reference=json.dumps(
                        payload["read_full_message"], ensure_ascii=False
                    ),
                )
    return row


def _selected_tool_view(
    message: AgentMessage,
    saved: Mapping[str, Any],
    protected: set[str],
    changed: set[object],
) -> AgentMessage:
    """优先应用新文件失效证据，再重放同源表示；参数：消息、持久选档及保护集合；返回：实际回执。"""
    if (
        not isinstance(message, ToolResultMessage)
        or message.message_id in protected | changed
    ):
        return message
    row = tool_material_row(message, session_id=saved["scope"]["session_id"])
    previous = saved["decisions"].get(material_key(row))
    if previous is None or "rendered_message" not in previous:
        return message
    projected = agent_message_from_mapping(previous["rendered_message"])
    if (
        not isinstance(projected, ToolResultMessage)
        or projected.call_id != message.call_id
    ):
        raise ValueError("saved tool representation does not match its original")
    return projected


@dataclass(frozen=True, slots=True)
class _RestoreObservation:
    """文件恢复的持久观察位置，不把时间戳或相同字节当作执行归因。"""

    sequence: int
    version: str | None
    operation_id: str
    unknown: bool


def _restore_observations(
    data_root: Path, session_id: str, records: Mapping[str, dict[str, Any]]
) -> tuple[dict[str, _RestoreObservation], dict[str, int]]:
    """读取同工作区真实恢复与原读取的提交位置；参数：数据根、会话、读取记录；返回：路径观察及读取序号。"""
    changes: dict[str, _RestoreObservation] = {}
    positions: dict[str, int] = {}
    with RuntimeStore(data_root).snapshot() as source:
        binding = source.get("session_workspace", session_id)
        if binding is None:
            return changes, positions
        operations = source.list_raw(
            "file_restore_operation", workspace_id=binding["workspace_id"]
        )
        for record in operations:
            operation = source.get("file_restore_operation", record.record_id)
            assert operation is not None and record.location is not None
            for entry in operation["entries"]:
                if entry["effect"] not in {
                    "changed",
                    "changed_with_conflict",
                    "unknown",
                    "publishing",
                }:
                    continue
                position, state = record.location.sequence, None
                if entry.get("restore_point_id"):
                    point_record = source.raw(
                        "file_restore_point", entry["restore_point_id"]
                    )
                    point = source.get("file_restore_point", entry["restore_point_id"])
                    if (
                        point_record is None
                        or point is None
                        or point_record.location is None
                    ):
                        raise ValueError(
                            "restore file observation is missing from committed sources"
                        )
                    position = point_record.location.sequence
                    state = point["entries"][0].get("after")
                version = None
                if state is not None and not state.get("sensitive"):
                    version = (
                        "missing"
                        if state["kind"] == "missing"
                        else state.get("version")
                    )
                path = os.path.normcase(entry["destination"])
                previous = changes.get(path)
                if previous is None or position > previous.sequence:
                    changes[path] = _RestoreObservation(
                        position,
                        version,
                        operation["operation_id"],
                        entry["effect"] in {"unknown", "publishing"},
                    )
        if changes:
            for call_id, row in records.items():
                read_record = source.raw("tool_operation", row["operation_id"])
                if read_record is not None and read_record.location is not None:
                    positions[call_id] = read_record.location.sequence
    return changes, positions


def _fit_preview(
    message: ToolResultMessage, *, reference: str, allowance: int
) -> ToolResultMessage:
    """计算包含包装、错误与原件引用的连续预览；传参：回执、引用及token额度；返回：可发送片段。"""
    text = model_visible_text(message)
    low, high = 0, len(text)
    chosen = _preview(message, text="", reference=reference)
    while low <= high:
        end = (low + high) // 2
        candidate = _preview(message, text=text[:end], reference=reference)
        if estimate_agent_messages_tokens((candidate,)) <= allowance:
            chosen, low = candidate, end + 1
        else:
            high = end - 1
    return chosen


def _file_observation(
    row: dict[str, Any] | None, message: ToolResultMessage
) -> tuple[str, str, tuple[int, int] | None, bool] | None:
    """从持久操作取已证明的资源版本，不用模型正文猜测；传参：实际操作和分支回执；返回：文件观察或无版本证据。"""
    if row is None or row.get("state") not in {"completed", "late_completed"}:
        return None
    call, result = row["call"], row.get("result") or {}
    execution = call.get("execution_request") or {"tool": call["tool_name"]}
    name = execution["tool"]
    resource, meta = call.get("resource") or {}, result.get("meta") or {}
    if (
        name not in FILE_WRITERS | {"file_read"}
        or execution.get("source", "builtin") != "builtin"
        or result.get("status") != "ok"
    ):
        return None
    if (
        not resource.get("known")
        or resource.get("directory")
        or call["call_id"] != message.call_id
    ):
        return None
    path, version = meta.get("resolved_path"), meta.get("content_sha256")
    if not isinstance(path, str) or not isinstance(version, str) or not version:
        return None
    span = None
    if name == "file_read":
        start = meta.get("offset")
        count = meta.get("returned_count")
        # 【文件读取】【范围复用】选定行读完不等于文件读完，优先采用实际交付字符数
        end = (
            start + count
            if type(start) is int and type(count) is int
            else meta.get("next_offset")
        )
        if end is None:
            end = meta.get("total_count", meta.get("total_chars"))
        if type(start) is int and type(end) is int and 0 <= start <= end:
            span = (start, end)
    return os.path.normcase(path), version, span, name in FILE_WRITERS


def _preview(
    message: ToolResultMessage, *, text: str, reference: str
) -> ToolResultMessage:
    """明确标记预览及原始回执分页入口，状态不因缩短而改变；传参：原回执和片段；返回：模型投影。"""
    payload = {
        "tool_name": message.tool_name,
        "status": message.status,
        "output": text,
        "prompt_truncated": True,
        "output_complete": False,
        "read_full_message": {
            "tool": "read_artifact",
            "artifact_id": reference,
            "mode": "full",
            "offset": 0,
        },
        "source_format": "Complete ToolResultMessage JSON; includes original content, metadata, error and artifact references",
    }
    # 1. 【上下文】【回执投影】完整错误也在原件中，模型状态保留失败，避免大错误再次重复占满窗口
    error = (
        "Tool reported an error; read_full_message contains the complete error."
        if message.error
        else None
    )
    parts = tuple(part for part in message.content if not isinstance(part, TextPart))
    return replace(
        message,
        content=(TextPart(json.dumps(payload, ensure_ascii=False)), *parts),
        error=error,
        artifact_refs=(*message.artifact_refs, reference),
    )
