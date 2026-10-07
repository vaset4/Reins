"""【上下文】【主动释放】查看实际采用材料并将可恢复正文降为引用。

作者：xxx
时间：2026-10-01 14:00:00
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from artifacts.store import ArtifactStore
from context.artifact_ref import store_large_output
from context.selection_store import (
    MaterialSelectionStore,
    content_digest,
    selection_scope,
)
from llm.messages import TextPart, ToolResultMessage, agent_message_to_mapping
from runtime.native_actions import NativeActionContext
from runtime.tool_operations import ToolOperation
from runtime.types import RunToolsResult

CONTEXT_ACTIONS = frozenset({"context_inspect", "context_release"})


class ContextActions:
    """复用现有原生操作与来源，释放不改变业务状态或原件。"""

    def __init__(self, context: NativeActionContext, *, data_root: Path) -> None:
        """绑定执行范围；参数：原生依赖和数据根；返回：无。"""
        self.context, self.data_root = context, data_root

    def execute(self, call: ToolOperation) -> RunToolsResult:
        """查看目录或提交同源释放；参数：已授权操作；返回：真实 applied/rejected 状态。"""
        scope = self._scope()
        store = MaterialSelectionStore(self.data_root)
        current = store.read(scope)
        if call.tool_name == "context_inspect":
            rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"rendered_message", "retained_text"}
                }
                for row in current.get("catalog", [])
            ]
            payload = {
                "materials": rows,
                "adopted_request_id": current.get("adopted_request_id"),
                "scope": scope,
            }
        elif call.tool_name == "context_release":
            try:
                row = next(
                    (
                        item
                        for item in current.get("catalog", [])
                        if item["identity"] == call.args.get("identity")
                        and item["version"] == call.args.get("version")
                    ),
                    None,
                )
                if row is None:
                    raise ValueError(
                        "material version is not in the current adopted request"
                    )
                if row["protected"]:
                    raise ValueError(
                        "current requirements and needed original reads cannot be released"
                    )
                released = self._release_row(row, call)
                # 1. 【上下文】【释放发布】大原件读取在锁外，提交前重核分支和权限归属
                with self.context.messages.write_lock(self.context.run.session_id):
                    if scope != self._scope():
                        raise ValueError(
                            "context branch or permissions changed before release"
                        )
                    store.release(scope, released, operation_id=call.operation_id)
                payload = {
                    "status": "applied",
                    "identity": row["identity"],
                    "version": row["version"],
                    "representation": "reference",
                    "read_reference": released["read_reference"],
                }
            except (ValueError, FileNotFoundError) as exc:
                return RunToolsResult.error_result(
                    action=call.tool_name,
                    error=str(exc),
                    meta={"status": "rejected", "reason": str(exc)},
                )
        else:
            raise ValueError("unknown context action")
        return RunToolsResult.ok(
            action=call.tool_name,
            content=json.dumps(payload, ensure_ascii=False),
            meta=payload,
        )

    def _scope(self) -> dict[str, str]:
        """按实际会话分支和权限取当前范围；参数：无；返回：与请求组装相同的范围键。"""
        run = self.context.run
        filesystem = getattr(run.capability_lease, "capabilities", {}).get("fs", {})
        view = self.context.messages.materialize(run.session_id)
        branch = next(
            (
                entry.entry_id
                for entry in reversed(view.entries)
                if entry.type == "branch"
            ),
            view.entries[0].entry_id if view.entries else run.session_id,
        )
        return selection_scope(
            {
                "session_id": run.session_id,
                "context_branch_id": branch,
                "capability_lease": run.capability_lease,
                "tool_working_directory": filesystem.get("project_root", ""),
            }
        )

    def _release_row(
        self, row: Mapping[str, Any], call: ToolOperation
    ) -> dict[str, Any]:
        """释放前核实可恢复原件；参数：已采用来源及操作；返回：引用表示。"""
        if row["source"] != "tool_result":
            if not row["read_reference"] and "retained_text" in row:
                if content_digest(row["retained_text"]) != row["source_digest"]:
                    raise ValueError("retained material representation changed")
                artifact = store_large_output(
                    self.data_root,
                    self.context.run.storage_task_id,
                    row["retained_text"],
                    threshold=0,
                    source_id=row["identity"],
                    summary="Retained context material",
                )
                assert artifact is not None
                reference = {
                    "tool": "read_artifact",
                    "arguments": {"artifact_id": artifact.artifact_id, "mode": "full"},
                }
                return {
                    **row,
                    "representation": "reference",
                    "read_reference": json.dumps(reference),
                }
            self._verify_reference(row)
            return {**row, "representation": "reference"}
        view = self.context.messages.materialize(self.context.run.session_id)
        message = next(
            (item for item in view.messages if item.message_id == row["message_id"]),
            None,
        )
        if (
            not isinstance(message, ToolResultMessage)
            or content_digest(agent_message_to_mapping(message)) != row["version"]
        ):
            raise ValueError("tool original is missing or outside the current branch")
        if view.pending_tool_calls:
            pending_ids = set(view.pending_tool_calls)
            if message.call_id in pending_ids:
                raise ValueError("pending tool groups cannot be released")
        original = json.dumps(agent_message_to_mapping(message), ensure_ascii=False)
        artifact = store_large_output(
            self.data_root,
            self.context.run.storage_task_id,
            original,
            threshold=0,
            source_id=message.message_id,
            summary=f"{message.tool_name} complete receipt {message.call_id}",
        )
        assert artifact is not None
        reference = {
            "tool": "read_artifact",
            "arguments": {"artifact_id": artifact.artifact_id, "mode": "full"},
        }
        from dataclasses import replace

        projected = replace(
            message,
            content=(
                TextPart(
                    json.dumps(
                        {
                            "context_released": True,
                            "tool_name": message.tool_name,
                            "status": message.status,
                            "read_full_message": reference,
                        },
                        ensure_ascii=False,
                    )
                ),
                *(part for part in message.content if not isinstance(part, TextPart)),
            ),
            artifact_refs=(*message.artifact_refs, artifact.artifact_id),
        )
        return {
            **row,
            "representation": "reference",
            "read_reference": json.dumps(reference, ensure_ascii=False),
            "rendered_message": agent_message_to_mapping(projected),
            "operation_id": call.operation_id,
        }

    def _verify_reference(self, row: Mapping[str, Any]) -> None:
        """只读取已有原件，不执行补读业务工具；参数：引用目录行；返回：无，缺失/损坏报错。"""
        if not row["read_reference"]:
            raise ValueError("material has no retained original reference")
        reference = json.loads(row["read_reference"])
        action = reference.get("read_action", reference)
        args = action["arguments"]
        if action["tool"] == "read_artifact":
            ArtifactStore(self.data_root).read_path(args["artifact_id"])
        elif action["tool"] == "memory_query":
            from memory.store import MemoryStore

            MemoryStore(self.data_root).load_memory(
                args["memory_id"], version=args["version"]
            )
        elif action["tool"] == "skill_read":
            from skills.store import SkillStore

            SkillStore(self.data_root).load_skill(
                args["skill_id"], version=args["version"]
            )
        elif action["tool"] == "read_history":
            from runtime.history_reader import read_history_page
            from runtime.session_compaction import SessionCompactionStore

            view = self.context.messages.materialize(self.context.run.session_id)
            read_history_page(
                view,
                call_id="verify-release",
                summaries=SessionCompactionStore(self.context.messages),
                **args,
            )
        else:
            raise ValueError("material has no supported recoverable original")
