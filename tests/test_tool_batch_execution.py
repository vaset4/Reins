"""独立工具并行、逐项失败与持久调用配对验证。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from threading import Barrier

from llm.messages import ToolCallPart, ToolResultMessage, model_visible_text
from llm.parser import parse_tool_call_parts
from llm.types import LLMPlan
from runtime.agent_loop import AgentLoop, State
from runtime.lease import from_trigger
from runtime.session_messages import append_user_message, materialize_messages
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
from tools.types import ToolError, ToolErrorCategory


class Plans:
    """使用模型边界的真实上下文驱动确定性计划，不模拟工具执行。"""

    def __init__(self, plans: list[LLMPlan]) -> None:
        """保存计划序列；传参：按请求返回的计划；返回：无。"""
        self.plans = iter(plans)
        self.contexts: list[object] = []

    def plan(self, _task: str, context: object = None) -> LLMPlan:
        """记录模型收到的上下文；传参：目标与上下文；返回：本轮计划。"""
        self.contexts.append(context)
        return next(self.plans)

    def continue_from_run_tools(
        self, task: str, _result: object, context: object = None
    ) -> LLMPlan:
        """继续读取已提交的调用结果；传参：目标、结果与上下文；返回：计划。"""
        return self.plan(task, context)


def make_run(root: Path, registry: ToolRegistry, plans: list[LLMPlan]):
    """组装真实会话和执行器；传参：存储根、注册表与计划；返回：循环、上下文及模型。"""
    with closing(TaskStore(root)) as store:
        task = store.create_task("批量调查")
    client = Plans(plans)
    loop = AgentLoop(root, llm_client=client, tool_registry=registry)
    lease = from_trigger(
        "user",
        task_id=task.task_id,
        capabilities={"fs": {"read": [str(root)], "write": [str(root)]}},
    )
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": "批量调查"},
        capability_lease=lease,
    )
    append_user_message(root, context.session_id, "批量调查")
    return loop, context, client


def test_invalid_call_does_not_discard_independent_results(tmp_path: Path) -> None:
    """一项参数非法、一项业务失败时第三项仍执行并全部回填；传参：临时根；返回：无。"""
    effects: list[str] = []

    def execute(args: dict[str, object]) -> object:
        """执行可核对的业务动作；传参：已校验字段；返回：成功数据或明确失败。"""
        effects.append(str(args["value"]))
        if args["value"] == "failure":
            return ToolError(
                ToolErrorCategory.INVALID_INPUT, "source missing", retryable=False
            )
        return "independent-success-evidence"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "probe",
            "查询",
            {"value": {"type": "string", "required": True}},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=execute,
        )
    )
    calls = tuple(
        ToolCallPart(f"call-{index}", "probe", args)
        for index, args in enumerate(({}, {"value": "failure"}, {"value": "success"}))
    )
    plan = parse_tool_call_parts(calls, allowed_tool_names={"probe"}, registry=registry)
    loop, context, client = make_run(
        tmp_path, registry, [plan, LLMPlan(final_output="使用成功的调查结果")]
    )
    list(loop.run_stream(context))
    results = [
        message
        for message in materialize_messages(tmp_path, context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    assert effects == ["failure", "success"]
    assert [result.call_id for result in results] == ["call-0", "call-1", "call-2"]
    assert [result.status for result in results] == ["error", "error", "success"]
    assert "independent-success-evidence" in str(client.contexts[-1])
    assert loop.state == State.DONE
    payloads = [json.loads(result.content[0].text) for result in results]
    assert len({payload["meta"]["operation_id"] for payload in payloads}) == 3


def test_declared_independent_backends_execute_concurrently(tmp_path: Path) -> None:
    """两个允许并行的后端必须在同一屏障会合；传参：临时根；返回：无。"""
    barrier = Barrier(2, timeout=3)

    def execute(args: dict[str, object]) -> str:
        """只有实际并行才返回成功；传参：查询值；返回：可核对正文。"""
        barrier.wait()
        return str(args["value"])

    registry = ToolRegistry()
    definition = ToolDefinition(
        "independent",
        "独立查询",
        {"value": {"type": "string"}},
        "agent",
        ToolRisk.SAFE,
        True,
        "logical_scope",
        "builtin",
        idempotent=Idempotent.YES,
        executor=execute,
    )
    definition.parallel_safe = True
    registry.register(definition)
    calls = tuple(
        ToolCallPart(f"parallel-{value}", "independent", {"value": value})
        for value in ("A", "B")
    )
    plan = parse_tool_call_parts(
        calls, allowed_tool_names={"independent"}, registry=registry
    )
    loop, context, _client = make_run(
        tmp_path, registry, [plan, LLMPlan(final_output="已使用A和B")]
    )
    list(loop.run_stream(context))
    results = [
        message
        for message in materialize_messages(tmp_path, context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 2
    assert all(result.status == "success" for result in results)


def test_once_approval_does_not_authorize_other_parallel_calls(
    tmp_path: Path, monkeypatch
) -> None:
    """单次授权只能执行用户批准的资源；传参：临时目录与替换器；返回：无。"""
    import approval

    requested: list[str] = []
    effects: list[str] = []

    def decide(request: approval.ApprovalRequest) -> approval.ApprovalDecision:
        """仅批准 A，拒绝 B；传参：当前审批资源；返回：该动作的决定。"""
        value = str(request.args["value"])
        requested.append(value)
        return (
            approval.ApprovalDecision.ONCE
            if value == "A"
            else approval.ApprovalDecision.DENY
        )

    def execute(arguments: dict[str, object]) -> str:
        """记录真实派发的资源；传参：工具参数；返回：资源正文。"""
        value = str(arguments["value"])
        effects.append(value)
        return value

    from tests.support.approval import install_approval

    install_approval(monkeypatch, decide)
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "private_read",
            "读取需授权的资源",
            {"value": {"type": "string"}},
            "agent",
            ToolRisk.CONFIRM,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=execute,
            parallel_safe=True,
        )
    )
    calls = tuple(
        ToolCallPart(f"read-{value}", "private_read", {"value": value})
        for value in ("A", "B")
    )
    plan = parse_tool_call_parts(
        calls, allowed_tool_names={"private_read"}, registry=registry
    )
    loop, context, _client = make_run(
        tmp_path, registry, [plan, LLMPlan(final_output="使用获准资源")]
    )

    list(loop.run_stream(context))

    assert requested == ["A", "B"]
    assert effects == ["A"]


def test_approval_facility_failure_reaches_model_as_facility(
    tmp_path: Path, monkeypatch
) -> None:
    """审批设施故障经真实主循环后模型读到设施故障标识而非未知错误；传参：临时目录与替换器；返回：无。"""
    import approval

    def unavailable(_request: approval.ApprovalRequest) -> approval.ApprovalDecision:
        """模拟审批通道没给出决定；传参：审批请求；返回：不返回，抛出设施故障。"""
        raise approval.ApprovalUnavailable("approval backend failed: channel down")

    from tests.support.approval import install_approval

    install_approval(monkeypatch, unavailable)
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "private_read",
            "读取需授权的资源",
            {"value": {"type": "string"}},
            "agent",
            ToolRisk.CONFIRM,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=lambda _arguments: "SECRET-PAYLOAD",
        )
    )
    calls = (ToolCallPart("read-1", "private_read", {"value": "A"}),)
    plan = parse_tool_call_parts(
        calls, allowed_tool_names={"private_read"}, registry=registry
    )
    loop, context, _client = make_run(
        tmp_path, registry, [plan, LLMPlan(final_output="审批系统故障")]
    )

    list(loop.run_stream(context))

    results = [
        message
        for message in materialize_messages(tmp_path, context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 1
    payload = json.loads(model_visible_text(results[0]))
    assert payload["meta"]["approval_state"] == "unavailable"
    assert payload["meta"]["tool_error_category"] == "transport"
    assert payload["error"].startswith("transport:")
    assert "SECRET-PAYLOAD" not in model_visible_text(results[0])
