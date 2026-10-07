"""把用户明确选择绑定到已发布成果、目标版本和会话分支。

作者：xxx
时间：2026-09-24 18:00:00
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

from llm.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallPart,
    ToolResultMessage,
    model_visible_text,
)
from runtime.goal_completion import resolve_completion_evidence
from runtime.ledger import LedgerEvent, LedgerStore, new_ledger_event
from runtime.ledger_writer import GOAL_CONFIRMATION_DECIDED, LedgerWriter
from runtime.session_message_store import MaterializedSession, SessionMessageStore
from runtime.tool_operations import ToolOperation, ToolOperationStore
from runtime.types import RunContext
from tasks.store import TaskStore

CONFIRMATION_EVENT = GOAL_CONFIRMATION_DECIDED
EVIDENCE_PREVIEW_CHARS = 400


class CompletionConfirmations:
    """完成提案与用户动作的窄接口，正文和事件仍交给原有写者保存。"""

    def __init__(
        self,
        tasks: TaskStore,
        messages: SessionMessageStore,
        operations: ToolOperationStore,
        ledger: LedgerStore,
    ) -> None:
        """注入四个现有事实写者；传参：目标、会话、操作和事件存储；返回：无。"""
        self.tasks, self.messages, self.operations, self.ledger = (
            tasks,
            messages,
            operations,
            ledger,
        )

    def prepare(self, call: ToolOperation, run: RunContext) -> dict[str, Any]:
        """固定用户将看到的目标和成果；传参：提问操作与当前运行；返回：可持久保存的提案。"""
        request = cast(Mapping[str, Any], call.args["completion"])
        task_id, revision = str(request["goal_id"]), request["expected_revision"]
        with self.messages.write_lock(run.session_id):
            target = self.tasks.require_task(task_id)
            if (
                target.is_inbox
                or type(revision) is not int
                or target.revision != revision
                or target.status != "active"
            ):
                raise ValueError(
                    "completion proposal requires the current active goal revision"
                )
            if any(
                item.get("kind") == "user_confirmation" for item in request["evidence"]
            ):
                raise ValueError(
                    "completion proposal must reference delivered work, not another confirmation"
                )
            current = self.messages.materialize(run.session_id)
            references = resolve_completion_evidence(
                current,
                request["evidence"],
                task_id=task_id,
                operation_refs=self._operation_refs(run.session_id, task_id),
                run_id=run.run_id,
            )
            # 1. 【目标确认】【冻结提案】保存真实消息身份，后续不能把 current_answer 指向新的文字
            evidence = [
                {
                    "kind": row["kind"],
                    "reference": row["message_id"],
                    "reason": row["reason"],
                }
                for row in references
            ]
            fixed = resolve_completion_evidence(current, evidence, task_id=task_id)
            return {
                "question_id": call.operation_id,
                "goal_id": task_id,
                "goal_title": target.goal,
                "expected_revision": revision,
                "evidence": evidence,
                "evidence_digest": _digest(fixed),
                "branch_anchor": current.leaf_id,
                "branch_id": _branch_id(current),
                "run_id": run.run_id,
                "question": str(call.args["question"]),
            }

    def pending(self, session_id: str) -> list[dict[str, Any]]:
        """读取仍在当前分支的确认提案，过期原因可展示；传参：会话；返回：提案视图。"""
        if not self.messages.exists(session_id):
            return []
        with self.messages.write_lock(session_id):
            current = self.messages.materialize(session_id)
            calls = _call_ids(current)
            decided = {
                row.payload.get("question_id")
                for row in self.ledger.read_session_events(session_id)
                if row.event == CONFIRMATION_EVENT
            }
            result = []
            for row in self.operations.for_session(session_id):
                proposal = row.get("result", {}).get("meta", {}).get("confirmation")
                if (
                    not proposal
                    or row["state"] != "waiting_user"
                    or row["call"]["call_id"] not in calls
                    or row["operation_id"] in decided
                ):
                    continue
                error = ""
                try:
                    self._validate(proposal, current)
                except ValueError as exc:
                    error = str(exc)
                source_ids = {item["reference"] for item in proposal["evidence"]}
                preview = [
                    _evidence_preview(message)
                    for message in current.messages
                    if message.message_id in source_ids
                ]
                result.append(
                    {
                        **proposal,
                        "valid": not error,
                        "error": error,
                        "evidence_preview": preview,
                    }
                )
            return result

    def decide(
        self, session_id: str, question_id: str, *, action_id: str, accepted: bool
    ) -> LedgerEvent:
        """持久接纳真实前端选择，重复传输只补缺失提交；传参：会话、问题及动作选择；返回：决定事件。"""
        if type(accepted) is not bool or not action_id.strip():
            raise ValueError(
                "completion choice requires an action identity and a boolean decision"
            )
        event_id, input_id = (
            f"confirmation-{action_id}",
            f"confirmation-input-{action_id}",
        )
        with self.messages.write_lock(session_id):
            decisions = self.ledger.read_session_events(session_id)
            existing = next(
                (row for row in decisions if row.event_id == event_id), None
            )
            if existing is not None:
                if (
                    existing.payload.get("question_id") != question_id
                    or existing.payload.get("accepted") is not accepted
                ):
                    raise ValueError(
                        "completion action identity has a different decision"
                    )
                self._save_answer(session_id, question_id, existing)
                return existing
            if any(
                row.event == CONFIRMATION_EVENT
                and row.payload.get("question_id") == question_id
                for row in decisions
            ):
                raise ValueError("completion question already has a decision")
            current = self.messages.materialize(session_id)
            row = next(
                (
                    item
                    for item in self.operations.for_session(session_id)
                    if item["operation_id"] == question_id
                ),
                None,
            )
            if row is None or row["call"]["call_id"] not in _call_ids(current):
                raise ValueError("completion question is not on the current branch")
            proposal = row.get("result", {}).get("meta", {}).get("confirmation")
            if not isinstance(proposal, dict) or row["state"] not in {
                "waiting_user",
                "answered",
            }:
                raise ValueError("operation is not a completion confirmation question")
            if row["state"] == "answered" and row.get("answer_input_id") != input_id:
                raise ValueError(
                    "completion question was superseded by another user reply"
                )
            self._validate(proposal, current)
            if any(
                entry.entry_id != input_id
                for entry in self.messages.pending_inputs(session_id)
            ):
                raise ValueError(
                    "new user input must be considered before completion confirmation"
                )
            # 1. 【目标确认】【接纳选择】正文仅存 Session；事件仅引用身份及被确认的版本
            text = f"{'确认完成' if accepted else '仍需修改'}：{proposal['goal_title']}"
            source = self.messages.accept_input(
                session_id,
                text,
                input_id=input_id,
                task_id=proposal["goal_id"],
                run_id=proposal["run_id"],
                input_source="user",
            )
            event = new_ledger_event(
                CONFIRMATION_EVENT,
                event_id,
                "user_action",
                {
                    "action_id": action_id,
                    "question_id": question_id,
                    "source_input_id": source.entry_id,
                    "expected_revision": proposal["expected_revision"],
                    "branch_anchor": proposal["branch_anchor"],
                    "branch_id": proposal["branch_id"],
                    "evidence_digest": proposal["evidence_digest"],
                    "evidence": proposal["evidence"],
                    "accepted": accepted,
                },
                task_id=proposal["goal_id"],
                session_id=session_id,
                run_id=proposal["run_id"],
            )
            committed = LedgerWriter(self.ledger).record_once(event)
            self._save_answer(session_id, question_id, committed)
            return committed

    def _save_answer(
        self, session_id: str, question_id: str, event: LedgerEvent
    ) -> None:
        """事件已提交后补齐问题回执，失败重传不重复用户动作；传参：会话、问题、事件；返回：无。"""
        row = next(
            item
            for item in self.operations.for_session(session_id)
            if item["operation_id"] == question_id
        )
        identity = {
            key: str(row[key]) for key in ("session_id", "run_id", "operation_id")
        }
        self.operations.write(
            identity,
            {
                **row,
                "state": "answered",
                "answer_input_id": event.payload["source_input_id"],
                "confirmation_event_id": event.event_id,
            },
        )

    def _validate(
        self, proposal: Mapping[str, Any], current: MaterializedSession
    ) -> None:
        """复核用户看到的版本与成果仍有效；传参：固定提案和当前分支；返回：无，失效抛错。"""
        target = self.tasks.require_task(str(proposal["goal_id"]))
        if (
            target.revision != proposal["expected_revision"]
            or target.status != "active"
        ):
            raise ValueError("completion proposal goal revision changed")
        if proposal["branch_id"] != _branch_id(current) or not any(
            entry.entry_id == proposal["branch_anchor"] for entry in current.entries
        ):
            raise ValueError("completion proposal branch changed")
        references = resolve_completion_evidence(
            current, proposal["evidence"], task_id=target.task_id
        )
        if _digest(references) != proposal["evidence_digest"]:
            raise ValueError("completion proposal evidence changed")

    def _operation_refs(self, session_id: str, task_id: str) -> dict[str, str]:
        """把真实成功操作映射回原调用；传参：会话和目标；返回：引用映射。"""
        return {
            row["operation_id"]: row["call"]["call_id"]
            for row in self.operations.for_session(session_id)
            if row["call"].get("task_id") == task_id
            and row.get("result", {}).get("status") == "ok"
            and row["state"] in {"completed", "late_completed"}
        }


def _digest(references: object) -> str:
    """计算成果引用的固定摘要；传参：已核验引用；返回：内容哈希。"""
    return hashlib.sha256(
        json.dumps(references, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _evidence_preview(message: AgentMessage) -> str:
    """展示成果正文，工具信封仅取输出；传参：原消息；返回：有长度上限的预览，不影响完整引用。"""
    text = model_visible_text(message)
    if isinstance(message, ToolResultMessage):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # 【目标确认】【成果展示】旧工具结果允许纯文本，按原正文展示
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("output"), str):
            text = payload["output"]
    return text[:EVIDENCE_PREVIEW_CHARS]


def _branch_id(current: MaterializedSession) -> str | None:
    """读取最近一次分支选择，普通新消息不使确认失效；传参：当前会话；返回：分支身份。"""
    return next(
        (
            entry.entry_id
            for entry in reversed(current.entries)
            if entry.type == "branch"
        ),
        None,
    )


def _call_ids(current: MaterializedSession) -> set[str]:
    """提取当前分支真实发出的调用身份；传参：会话；返回：调用集合。"""
    return {
        part.call_id
        for message in current.messages
        if isinstance(message, AssistantMessage)
        for part in message.content
        if isinstance(part, ToolCallPart)
    }
