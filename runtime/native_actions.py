"""原生工具的运行内实现，依赖由当前执行器注入。

作者：xxx
时间：2026-09-14 11:00:00
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from llm.messages import AssistantMessage, ToolCallPart, UserMessage
from llm.toolset_policy import resolve_toolset_policy
from llm.types import LLMPlan
from runtime.tool_policy import RuntimeToolPolicy
from runtime.capability_catalog import browse_tools
from runtime.completion_confirmation import CompletionConfirmations
from runtime.goal_manager import AmbiguousGoalError, GoalManager, GoalNotFoundError
from runtime.history_reader import DEFAULT_HISTORY_PAGE_SIZE, read_history_page
from runtime.session_compaction import SessionCompactionStore
from runtime.ledger import LedgerStore
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_state import SessionStateStore
from runtime.tool_operations import (
    ToolOperation,
    ToolOperationStore,
    file_changes,
    retry_source,
)
from runtime.types import (
    RunContext,
    RunToolsRequest,
    RunToolsResult,
    TerminalFocusPolicy,
)
from tasks.store import TaskStore
from tools.tool_registry import Idempotent, ToolRegistry
from tools.argument_schema import tool_argument_error


@dataclass(frozen=True, slots=True)
class ResumeTarget:
    """携带旧操作的实际请求与事项归属，恢复调用不能用当前焦点替换它。"""

    request: RunToolsRequest
    task_id: str | None
    session_id: str
    run_id: str
    operation_id: str
    source: str


@dataclass(frozen=True, slots=True)
class NativeActionContext:
    """本轮动作的依赖与归属，不持有另一个生命周期或消息写者。"""

    run: RunContext
    tasks: TaskStore
    messages: SessionMessageStore
    states: SessionStateStore
    operations: ToolOperationStore
    facts: RunFactStore
    registry: ToolRegistry
    toolset_config: Mapping[str, object] | None = None
    ledger: LedgerStore | None = None
    policy: RuntimeToolPolicy | None = None


class NativeActions:
    """提供深而窄的原生动作接口，副作用仍经过共同执行器的 prepare/commit。"""

    def __init__(self, context: NativeActionContext) -> None:
        """绑定本轮依赖；传参：当前运行及现有写者；返回：无。"""
        self.context = context

    def execute(self, call: ToolOperation) -> RunToolsResult | ResumeTarget:
        """兑现已通过共同边界的动作；传参：完整操作；返回：工具结果，必要存储故障向上暴露。"""
        handlers = {
            "capabilities": self._capabilities,
            "ask_user": self._ask_user,
            "goal": self._goal,
            "read_history": self._read_history,
            "operation_status": self._operation_status,
            "resume_operation": self.prepare_resume,
        }
        try:
            return handlers[call.tool_name](call)
        except AmbiguousGoalError as exc:
            return RunToolsResult.error_result(
                action=call.tool_name,
                error=str(exc),
                meta={"candidates": list(exc.candidates)},
            )
        except (ValueError, GoalNotFoundError) as exc:
            return RunToolsResult.error_result(action=call.tool_name, error=str(exc))

    def _capabilities(self, call: ToolOperation) -> RunToolsResult:
        """按当前授权展示目录，加载结果作为后续请求材料；传参：目录动作；返回：目录页或定义。"""
        services = self.context
        if call.args["action"] == "refresh":
            services.registry.refresh_sources(services.run.capability_lease, force=True)
        policy = (
            services.policy.resolve(services.run)
            if services.policy is not None
            else resolve_toolset_policy(
                payload=services.run.payload,
                session_state=services.states.load(services.run.session_id),
                config=services.toolset_config,
                registry=services.registry,
            )
        )
        if isinstance(policy, LLMPlan):
            raise ValueError(str(policy.final_output))
        result = browse_tools(
            services.registry,
            call.args,
            lease=services.run.capability_lease,
            policy=policy,
        )
        return RunToolsResult.ok(
            action=call.tool_name,
            content=json.dumps(result, ensure_ascii=False),
            meta={"loaded_tool_names": result.get("loaded_tool_names", [])},
        )

    def _ask_user(self, call: ToolOperation) -> RunToolsResult:
        """以操作身份保存问题等待关系；传参：问题调用；返回：可关联后续输入的等待结果。"""
        messages = self.context.messages.materialize(
            self.context.run.session_id
        ).messages
        prior_inputs = [
            message.message_id
            for message in messages
            if isinstance(message, UserMessage)
        ]
        confirmation: dict[str, Any] = {}
        if "completion" in call.args:
            if self.context.ledger is None:
                raise RuntimeError("completion confirmation requires the ledger")
            service = CompletionConfirmations(
                self.context.tasks,
                self.context.messages,
                self.context.operations,
                self.context.ledger,
            )
            confirmation = {"confirmation": service.prepare(call, self.context.run)}
        return RunToolsResult.ok(
            action=call.tool_name,
            content=str(call.args["question"]),
            meta={
                "question_id": call.operation_id,
                "execution_state": "waiting_user",
                "prior_input_ids": prior_inputs,
                **confirmation,
            },
        )

    def _goal(self, call: ToolOperation) -> RunToolsResult:
        """新建、切换或核实目标完成；传参：合法目标参数；返回：已提交目标快照。"""
        services, args = self.context, call.args
        run = services.run
        manager = GoalManager(
            services.tasks,
            messages=services.messages,
            operations=services.operations,
            ledger=services.ledger,
        )
        action = str(args["action"])
        if action == "complete":
            decision = manager.complete_goal(
                str(args.get("goal_ref", run.focus_task_id or "")),
                expected_revision=cast(int, args["expected_revision"]),
                summary=str(args["goal_body"]),
                evidence=cast(Sequence[Mapping[str, object]], args["evidence"]),
                session_id=run.session_id,
                run_id=run.run_id,
                operation_id=call.operation_id,
            )
        else:
            decision = (
                manager.open_new_goal(str(args["goal_body"]))
                if action == "new"
                else manager.switch_goal(str(args["goal_ref"]))
            )
            services.states.update_focus(
                run.session_id,
                focus_task_id=decision.target_task_id,
                previous_focus_task_id=run.focus_task_id,
                compatibility_task_id=run.compatibility_task_id,
                summary=f"goal {action} -> {decision.target_task_id}",
            )
            run.focus_task_id = decision.target_task_id
            run.terminal_focus_policy = TerminalFocusPolicy.PRESERVE
        target = services.tasks.require_task(decision.target_task_id)
        snapshot = {
            "task_id": target.task_id,
            "goal": target.goal,
            "status": target.status,
            "revision": target.revision,
        }
        if run.focus_task_id == target.task_id:
            run.focus_task = snapshot
        services.facts.append(
            {
                "event": "goal:completed"
                if action == "complete"
                else "goal_op:focus_decision",
                "session_id": run.session_id,
                "run_id": run.run_id,
                "task_id": call.task_id,
                "target_task_id": target.task_id,
                "goal_op_kind": action,
                "operation_id": call.operation_id,
                "completion": target.completion,
                "created": decision.created,
            }
        )
        # 1. 【目标管理】【执行证据】目标状态属于事项记录，本动作没有派发后台执行
        receipt = {"record_kind": "goal", "execution_dispatched": False, **snapshot}
        return RunToolsResult.ok(
            action=call.tool_name,
            content=json.dumps(receipt, ensure_ascii=False),
            meta=receipt,
        )

    def _read_history(self, call: ToolOperation) -> RunToolsResult:
        """在当前分支中继续读取历史快照；传参：游标或消息身份及页长；返回：原文页面。"""
        current = self.context.messages.materialize(self.context.run.session_id)
        result = read_history_page(
            current,
            call_id=call.call_id,
            before=cast(str | None, call.args.get("before")),
            cursor=cast(str | None, call.args.get("cursor")),
            limit=cast(int, call.args.get("limit", DEFAULT_HISTORY_PAGE_SIZE)),
            summaries=SessionCompactionStore(self.context.messages),
            summary_id=cast(str | None, call.args.get("summary_id")),
            source_ref=cast(str | None, call.args.get("source_ref")),
            query=cast(str | None, call.args.get("query")),
            view_kind=cast(str | None, call.args.get("view")),
            segment_id=cast(str | None, call.args.get("segment_id")),
            level=cast(str | None, call.args.get("level")),
        )
        return RunToolsResult.ok(
            action=call.tool_name,
            content=json.dumps(result, ensure_ascii=False),
            meta={key: result[key] for key in ("next_before", "next_cursor")},
        )

    def _operation_status(self, call: ToolOperation) -> RunToolsResult:
        """返回当前分支的真实操作状态和已提交结果；传参：可选操作编号；返回：状态视图。"""
        records = self._branch_operations()
        identity = call.args.get("operation_id")
        if identity is not None:
            records = [self._require_operation(str(identity))]
        scope = call.args.get("run_id")
        if scope is not None:
            selected_run = self.context.run.run_id if scope == "current" else str(scope)
            records = [row for row in records if row["run_id"] == selected_run]
        payload: dict[str, object] = {"operations": records}
        payload["file_changes"] = file_changes(records)
        legacy = self.context.run.payload.get("_resume_choice_pending_evidence")
        if legacy is not None:
            payload["legacy_pending"] = legacy
        return RunToolsResult.ok(
            action=call.tool_name, content=json.dumps(payload, ensure_ascii=False)
        )

    def prepare_resume(self, call: ToolOperation) -> ResumeTarget | RunToolsResult:
        """只读取恢复候选供整批授权，不认领或执行原操作；传参：恢复调用；返回：固定候选或已有结果。"""
        row = self._require_operation(str(call.args["operation_id"]))
        if call.args["action"] == "skip":
            return RunToolsResult.ok(
                action=call.tool_name,
                content="已保留原操作证据；不再执行该动作",
                meta={"skipped_operation_id": row["operation_id"]},
            )
        execution = row["call"].get("execution_request") or {
            "tool": row["call"]["tool_name"],
            "arguments": row["call"]["args"],
        }
        definition = self.context.registry.get(execution["tool"])
        if definition is None:
            raise ValueError(
                "saved tool is no longer available; inspect the original and choose a current action"
            )
        error = tool_argument_error(definition.parameters, execution["arguments"])
        if error is not None:
            raise ValueError(
                f"saved tool arguments no longer match the current schema; choose a new action: {error}"
            )
        safe_read = definition.action_readonly(execution["arguments"]) and (
            definition.idempotent == Idempotent.YES
            or execution["arguments"].get("action") in definition.readonly_actions
        )
        if row["state"] != "not_started" and not safe_read:
            raise ValueError(
                "operation effects are not known to be absent; inspect actual state or choose another method"
            )
        previous = row.get("retry_operation_id")
        if previous is not None and previous != call.operation_id:
            return RunToolsResult.ok(
                action=call.tool_name,
                content="该操作已有恢复尝试，请查询该尝试的实际结果",
                meta={"retry_operation_id": previous},
            )
        arguments = dict(execution["arguments"])
        task_id = row["call"].get("task_id")
        # 【运行时】【恢复旧目标】完成动作未显式命名目标时，固定原事项，避免切换焦点后误作用于新事项
        if (
            execution["tool"] == "goal"
            and arguments.get("action") == "complete"
            and task_id is not None
        ):
            arguments.setdefault("goal_ref", task_id)
        request = RunToolsRequest(
            action=execution["tool"], tool_name=execution["tool"], arguments=arguments
        )
        return ResumeTarget(
            request,
            task_id,
            str(row["session_id"]),
            str(row["run_id"]),
            str(row["operation_id"]),
            retry_source(row),
        )

    def _branch_operations(self) -> list[dict[str, Any]]:
        """仅将当前分支调用关联到操作记录；传参：无；返回：有来源的状态列表。"""
        current = self.context.messages.materialize(self.context.run.session_id)
        call_ids = {
            part.call_id
            for message in current.messages
            if isinstance(message, AssistantMessage)
            for part in message.content
            if isinstance(part, ToolCallPart)
        }
        entry_ids = {entry.entry_id for entry in current.entries}
        return [
            row
            for row in self.context.operations.for_session(self.context.run.session_id)
            if row["call"]["call_id"] in call_ids
            or (
                row["call"].get("origin") == "extension"
                and row.get("branch_entry_id") in entry_ids
            )
        ]

    def _require_operation(self, operation_id: str) -> dict[str, Any]:
        """验证模型只能操作当前分支的既有身份；传参：编号；返回：记录，缺失显式报错。"""
        for row in self._branch_operations():
            if row["operation_id"] == operation_id:
                return row
        raise ValueError("operation is not in the current session branch")


def link_question_answers(
    messages: SessionMessageStore,
    operations: ToolOperationStore,
    *,
    run: RunContext,
    facts: RunFactStore,
) -> None:
    """把问题之后实际交付的首条输入关联到原等待身份；传参：消息/操作写者和运行；返回：无，不复制正文。"""
    current = messages.materialize(run.session_id)
    agent_inputs = {
        entry.entry_id for entry in current.entries if entry.input_source == "agent"
    }
    delivered = [
        message.message_id
        for message in current.messages
        if isinstance(message, UserMessage) and message.message_id not in agent_inputs
    ]
    call_ids = {
        part.call_id
        for message in current.messages
        if isinstance(message, AssistantMessage)
        for part in message.content
        if isinstance(part, ToolCallPart)
    }
    for row in operations.for_session(run.session_id):
        if row["state"] != "waiting_user" or row["call"]["call_id"] not in call_ids:
            continue
        prior = row["result"]["meta"]["prior_input_ids"]
        answer = next(
            (identity for identity in delivered if identity not in prior), None
        )
        if answer is None:
            continue
        identity = {
            key: str(row[key]) for key in ("session_id", "run_id", "operation_id")
        }
        operations.write(
            identity, {**row, "state": "answered", "answer_input_id": answer}
        )
        facts.append(
            {
                "event": "question:answered",
                "session_id": run.session_id,
                "run_id": run.run_id,
                "question_id": row["operation_id"],
                "input_ids": [answer],
            }
        )
