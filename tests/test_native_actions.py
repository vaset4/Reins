"""统一工具入口的真实会话往返验证。

作者：xxx
时间：2026-09-14 13:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final, from_test_stub

import json
from dataclasses import replace
from contextlib import closing

from llm.messages import ToolCallPart, ToolResultMessage
from runtime.agent_loop import AgentLoop
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_user_message, materialize_messages
from runtime.types import new_run_id
from schedules.store import ScheduleStore
from tasks.store import TaskStore
from tools.native_actions import register_native_actions
from tools.tool_registry import ToolRegistry
from tools.tool_registry import ToolDefinition, ToolRisk, Idempotent
from runtime.types import RunToolsResult
import pytest
from runtime.tool_executor import ToolBatchExecutor
from tests.test_session_runtime import capture_requests, make_runtime
from tests.test_tool_batch_execution import make_run
from tests.support.approval import install_approval


def native_registry():
    """只装配本轮统一原生工具；传参：无；返回：真实声明注册表。"""
    registry = ToolRegistry()
    register_native_actions(registry)
    return registry


def test_resume_executes_the_authorized_extension_candidate(tmp_path, monkeypatch):
    """恢复目标经过扩展改参后执行已授权候选；传参：目录与审批替换；返回：无。"""
    from approval import ApprovalDecision
    from runtime.extensions import RuntimeExtensions, ToolProposal

    effects = []
    registry = native_registry()
    registry.register(
        ToolDefinition(
            "effect",
            "实际动作",
            {"value": {"type": "string", "required": True}},
            "agent",
            ToolRisk.CONFIRM,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            executor=lambda args: effects.append(args["value"]) or "执行完成",
        )
    )
    install_approval(monkeypatch, lambda _request: ApprovalDecision.DENY)
    loop, context, _client = make_run(tmp_path, registry, [])
    loop.llm_client = from_test_native_tool_then_final(
        [ToolCallPart("original", "effect", {"value": "old"})], "等待恢复"
    )
    list(loop.run_stream(context))
    original = loop.operations.for_session(context.session_id)[0]
    assert original["state"] == "not_started"
    install_approval(monkeypatch, lambda _request: ApprovalDecision.ONCE)
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "resume",
                "resume_operation",
                {"operation_id": original["operation_id"], "action": "retry"},
            )
        ],
        "恢复完成",
    )

    def change(proposal, _token):
        """只调整真实目标参数，外层恢复身份不变；传参：候选和取消；返回：待授权候选。"""
        return (
            ToolProposal("effect", {"value": "authorized-new"}, proposal.operation_id)
            if proposal.tool == "effect"
            else proposal
        )

    resumed = AgentLoop(
        tmp_path,
        llm_client=client,
        tool_registry=registry,
        extensions=RuntimeExtensions(before_tool=(change,)),
    )
    list(resumed.run_stream(replace(context, run_id=new_run_id(), segment_id="")))
    assert effects == ["authorized-new"]
    row = next(
        row
        for row in resumed.operations.for_session(context.session_id)
        if row["call"]["call_id"] == "resume"
    )
    assert row["result"]["meta"]["resumed_execution_request"]["arguments"] == {
        "value": "authorized-new"
    }


def test_question_answer_keeps_identity_across_runs(tmp_path, monkeypatch):
    """提问先配对并等待，下一输入关联原等待身份且只保存一次；传参：临时根；返回：无。"""
    client = from_test_native_tool_then_final(
        [ToolCallPart("question-call", "ask_user", {"question": "哪一天？"})],
        "按周五办理",
    )
    requests = capture_requests(client, monkeypatch)
    runtime, runs = make_runtime(tmp_path, client, native_registry())
    runtime.submit("安排一次会议", input_id="question-origin")
    assert runtime.wait_idle(10)
    from runtime.tool_operations import ToolOperationStore

    operations = ToolOperationStore(tmp_path)
    question = operations.for_session("session-live")[0]
    assert question["state"] == "waiting_user"
    assert question["result"]["meta"]["question_id"] == question["operation_id"]
    runtime.submit("周五", input_id="answer")
    assert runtime.wait_idle(10)
    assert len(runs) == 2
    assert "周五" in str(requests[-1])
    question = operations.for_session("session-live")[0]
    assert question["state"] == "answered"
    assert question["answer_input_id"] == "answer"
    runtime.submit("周五", input_id="answer")
    assert runtime.wait_idle(2)
    assert len(runs) == 2


def test_goal_uses_common_call_result_and_keeps_long_term_state(tmp_path, monkeypatch):
    """目标操作进入下一真实请求，普通终稿不把目标标完成；传参：临时根；返回：无。"""
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "goal-call", "goal", {"action": "new", "goal_body": "核对两份材料"}
            ),
        ],
        "已建事项，等待材料",
    )
    requests = capture_requests(client, monkeypatch)
    runtime, runs = make_runtime(tmp_path, client, native_registry())
    runtime.submit("核对两份材料")
    assert runtime.wait_idle(10)
    result = next(
        item for item in requests[-1].messages if isinstance(item, ToolResultMessage)
    )
    receipt = json.loads(json.loads(result.content[0].text)["output"])
    target = receipt["task_id"]
    assert receipt["record_kind"] == "goal"
    assert receipt["execution_dispatched"] is False
    assert len(runs) == 1
    with closing(ScheduleStore(tmp_path)) as schedules:
        assert schedules.list_all_schedules() == []
    store = TaskStore(tmp_path)
    try:
        assert store.require_task(target).status == "active"
        assert runs[0].focus_task_id == target
        assert target in str(requests[-1])
    finally:
        store.close()


def test_history_pages_follow_message_cursor_without_repeating_tool(tmp_path):
    """历史游标向前翻页，未知游标明确返回失败；传参：临时根；返回：无。"""
    registry = native_registry()
    loop, context, _client = make_run(tmp_path, registry, [])
    ids = [
        append_user_message(tmp_path, context.session_id, f"旧材料-{index}")
        for index in range(4)
    ]
    client = from_test_native_tool_then_final(
        [
            ToolCallPart("history-1", "read_history", {"before": ids[-1], "limit": 2}),
        ],
        "已找到历史",
    )
    loop.llm_client = client
    list(loop.run_stream(context))
    result = next(
        item
        for item in materialize_messages(tmp_path, context.session_id)
        if isinstance(item, ToolResultMessage)
    )
    page = json.loads(json.loads(result.content[0].text)["output"])
    assert [item["message_id"] for item in page["messages"]] == ids[1:3]
    assert page["next_before"] == ids[1]
    next_client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "history-2", "read_history", {"before": page["next_before"], "limit": 2}
            ),
        ],
        "已读更早部分",
    )
    list(
        AgentLoop(tmp_path, llm_client=next_client, tool_registry=registry).run_stream(
            replace(context, run_id=new_run_id())
        )
    )
    results = [
        item
        for item in SessionMessageStore(tmp_path)
        .materialize(context.session_id)
        .messages
        if isinstance(item, ToolResultMessage)
    ]
    earlier = json.loads(json.loads(results[-1].content[0].text)["output"])
    assert ids[0] in str(earlier)
    assert ids[1] not in str(earlier)


def test_resume_keeps_original_task_for_approval_after_focus_changes(
    tmp_path, monkeypatch
):
    """恢复旧操作使用原事项授权范围，恢复请求仍归当前事项；传参：目录、替换器；返回：无。"""
    from approval import ApprovalDecision
    from tests.test_runtime_extensions import setup_loop
    from runtime.extensions import RuntimeExtensions

    effects, approvals = [], []

    def approve(request):
        """首次拒绝、恢复时单次批准，并记录真实授权归属；传参：审批；返回：决定。"""
        approvals.append(request.lease.task_id)
        return ApprovalDecision.DENY if len(approvals) == 1 else ApprovalDecision.ONCE

    install_approval(monkeypatch, approve)
    loop, context = setup_loop(
        tmp_path,
        RuntimeExtensions(),
        lambda _args: effects.append("written") or "已修改",
    )
    register_native_actions(loop.tool_registry)
    context.capability_lease.capabilities["fs"]["write"] = []
    list(loop.run_stream(context))
    original = loop.operations.for_session(context.session_id)[0]
    assert original["state"] == "not_started"
    with closing(TaskStore(tmp_path)) as store:
        current = store.create_task("另一个事项")
    resume = replace(
        context,
        run_id=new_run_id(),
        segment_id="",
        focus_task_id=current.task_id,
        capability_lease=replace(context.capability_lease, task_id=current.task_id),
    )
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "resume-old-owner",
                "resume_operation",
                {"operation_id": original["operation_id"], "action": "retry"},
            ),
        ],
        "继续原事项的动作",
    )
    list(
        AgentLoop(
            tmp_path, llm_client=client, tool_registry=loop.tool_registry
        ).run_stream(resume)
    )
    assert approvals == [context.task_id, context.task_id]
    assert effects == ["written"]
    results = [
        message
        for message in materialize_messages(tmp_path, context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    assert (
        json.loads(results[-1].content[0].text)["meta"]["resumed_execution_request"][
            "task_id"
        ]
        == context.task_id
    )


@pytest.mark.parametrize("rewind", [False, True])
def test_saved_unannounced_action_can_resume_only_on_its_branch(
    tmp_path, monkeypatch, rewind
):
    """后处理接纳后中断仍可显式继续，其他分支不能执行；传参：目录、故障注入、分支选择；返回：无。"""
    from runtime.extensions import ActionRequest, RuntimeExtensions
    from tests.test_runtime_extensions import setup_loop

    effects = []
    action = ActionRequest(
        "pending-action", "write", {"path": str(tmp_path / "later.txt")}
    )
    loop, context = setup_loop(
        tmp_path,
        RuntimeExtensions(after_run=(lambda _event: (action,),)),
        lambda _args: effects.append(True) or "已完成",
    )
    register_native_actions(loop.tool_registry)
    loop.llm_client = from_test_stub("主运行答复已保存")

    def fail(*_args, **_kwargs):
        """意图已接纳，在派发公告前中断；传参：调用参数；返回：抛错。"""
        raise OSError("before action announcement")

    with monkeypatch.context() as failure:
        failure.setattr(ToolBatchExecutor, "_announce_tool_batch", fail)
        with pytest.raises(OSError, match="before action announcement"):
            list(loop.run_stream(context))
    original = loop.operations.for_session(context.session_id)[0]
    if rewind:
        messages = SessionMessageStore(tmp_path)
        first_entry = messages.materialize(context.session_id).entries[0].entry_id
        messages.branch(context.session_id, first_entry)
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "inspect-pending",
                "operation_status",
                {"operation_id": original["operation_id"]},
            ),
            ToolCallPart(
                "resume-pending",
                "resume_operation",
                {"operation_id": original["operation_id"], "action": "retry"},
            ),
        ],
        "按可见操作状态继续",
    )
    list(
        AgentLoop(
            tmp_path, llm_client=client, tool_registry=loop.tool_registry
        ).run_stream(replace(context, run_id=new_run_id(), segment_id=""))
    )
    results = [
        message
        for message in materialize_messages(tmp_path, context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    assert [item.status for item in results[-2:]] == (
        ["error", "error"] if rewind else ["success", "success"]
    )
    assert effects == ([] if rewind else [True])


@pytest.mark.parametrize(
    "original_state", ["not_started", "unknown", "completed", "readonly"]
)
def test_resume_uses_effect_evidence_and_deduplicates_attempt(
    tmp_path, monkeypatch, original_state
):
    """仅未开始或明确幂等读取可重试，同一操作的恢复不会重复执行；传参：目录/替换器/状态；返回：无。"""
    from approval import ApprovalDecision

    effects = []
    registry = native_registry()

    def execute(_args):
        """留下可核对效果并报告相应执行证据；传参：参数；返回：实际结果。"""
        effects.append(True)
        if original_state == "unknown":
            return RunToolsResult.error_result(
                action="effect",
                error="backend cannot confirm effect",
                meta={"execution_state": "unknown"},
            )
        return "已执行结果"

    readonly = original_state == "readonly"
    registry.register(
        ToolDefinition(
            "effect",
            "实际动作",
            {},
            "agent",
            ToolRisk.CONFIRM,
            readonly,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES if readonly else Idempotent.NO,
            executor=execute,
        )
    )
    install_approval(
        monkeypatch,
        lambda _req: (
            ApprovalDecision.DENY
            if original_state == "not_started"
            else ApprovalDecision.ONCE
        ),
    )
    loop, context, _client = make_run(tmp_path, registry, [])
    loop.llm_client = from_test_native_tool_then_final(
        [ToolCallPart("original-effect", "effect", {})], "保留原结果"
    )
    list(loop.run_stream(context))
    row = loop.operations.for_session(context.session_id)[0]
    install_approval(monkeypatch, lambda _req: ApprovalDecision.ONCE)
    retry = {"operation_id": row["operation_id"], "action": "retry"}
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(f"retry-{index}", "resume_operation", retry)
            for index in range(2)
        ],
        "已核实恢复结果",
    )
    resumed = AgentLoop(tmp_path, llm_client=client, tool_registry=registry)
    list(resumed.run_stream(replace(context, run_id=new_run_id(), segment_id="")))
    assert len(effects) == (2 if readonly else 1)
    results = [
        item
        for item in materialize_messages(tmp_path, context.session_id)
        if isinstance(item, ToolResultMessage) and item.tool_name == "resume_operation"
    ]
    if original_state in {"unknown", "completed"}:
        assert all(item.status == "error" for item in results)
        assert "effects are not known to be absent" in str(results)
    else:
        assert all(item.status == "success" for item in results)
        assert "retry_operation_id" in str(results[-1])
