"""真实运行与持久消息边界中的未完成输出关闭。

作者：xxx
时间：2026-09-30 11:00:00
"""

from contextlib import closing

import pytest

from app.background.events import EventBuffer
from app.background.sessions import session_history
from frontends.tui.projection import ConversationProjection
from llm.types import LLMPlan, ModelError, ModelOutputDelta, ModelRetryNotice
from runtime.agent_loop import AgentLoop
from runtime.cancellation import ExecutionCancelled, RunBudgetExceeded
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.stream_events import (
    AssistantStreamClosed,
    AssistantTextDelta,
    AssistantTurnComplete,
)
from runtime.types import RunContext, Trigger
from scripts.testing.llm import from_test_sequence
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry


def prepared_run(root, monkeypatch, stream):
    """仅替换供应商输出，运行和存储使用真实实现；参数：隔离根、补丁和输出流；返回：运行生成器与身份。"""
    with closing(TaskStore(root)) as store:
        task = store.create_task("测试流生命周期", is_inbox=True)
    messages = SessionMessageStore(root)
    messages.accept_input(
        "session-stream", "原始请求", input_id="input-original", task_id=task.task_id
    )
    context = RunContext(
        session_id="session-stream",
        compatibility_task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": "原始请求", "input_message_id": "input-original"},
        capability_lease=from_trigger(
            "user", task_id=task.task_id, capabilities={"fs": {"read": [str(root)]}}
        ),
    )
    client = from_test_sequence(["未使用的适配器响应"])
    monkeypatch.setattr(client, "plan_stream", stream)
    monkeypatch.setattr(client, "continue_stream", stream)
    loop = AgentLoop(root, llm_client=client, tool_registry=ToolRegistry())
    return loop.run_stream(context), context


def project_event(buffer, projection, context, event):
    """把真实运行事件交到后台和前端；参数：缓存、投影、运行与事件；返回：无。"""
    buffer.emit(event, run_id=context.run_id)
    projection.apply(
        {"session_id": context.session_id, **buffer.read(projection.cursor)}
    )


def initial_projection(buffer, context):
    """建立已握手的空投影；参数：缓存与运行；返回：界面投影。"""
    projection = ConversationProjection()
    projection.apply(
        {
            "session_id": context.session_id,
            "history": [],
            **buffer.read(None, include_streams=True),
        }
    )
    return projection


def test_new_input_closes_superseded_output_and_late_delta_cannot_revive_it(
    tmp_path, monkeypatch
):
    """生成中真实接纳纠正后，旧输出不提交且迟到增量不能复活；参数：隔离根；返回：无。"""
    requests = []

    def stream(*args, **kwargs):
        """首请求生成时接纳新输入，下一请求回答新要求；参数：模型调用；返回：真实计划。"""
        requests.append(kwargs["context"]["model_request_id"])
        if len(requests) == 1:
            yield ModelOutputDelta("text", "旧片段")
            SessionMessageStore(tmp_path).accept_input(
                "session-stream", "改用新要求", input_id="input-correction"
            )
            return LLMPlan(final_output="旧结论")
        yield ModelOutputDelta("text", "新结论")
        return LLMPlan(final_output="新结论")

    runtime, context = prepared_run(tmp_path, monkeypatch, stream)
    buffer = EventBuffer()
    projection = initial_projection(buffer, context)
    for event in runtime:
        project_event(buffer, projection, context, event)
    closures = [
        row
        for row in RunFactStore(tmp_path).read_run(context.run_id)
        if row["event"] == "llm:stream_closed"
    ]
    assert len(closures) == 1 and closures[0]["reason"] == "superseded"
    assert (
        closures[0]["request_id"] == requests[0] and closures[0]["committed"] is False
    )
    assert any(
        card.role == "status" and "新输入替代" in card.text
        for card in projection.cards.values()
    )
    assert [
        card.text for card in projection.cards.values() if card.role == "assistant"
    ] == ["新结论"]
    buffer.emit(
        AssistantTextDelta("迟到旧输出", message_id=requests[0]), run_id=context.run_id
    )
    assert not buffer.read(None, include_streams=True)["streams"]
    restarted = ConversationProjection()
    restarted.apply(
        {
            "session_id": context.session_id,
            "history": session_history(tmp_path, context.session_id),
            **buffer.read(None, include_streams=True),
        }
    )
    assert not any("旧" in card.text for card in restarted.cards.values())


@pytest.mark.parametrize(
    "error,reason",
    [
        (ExecutionCancelled("stop"), "cancelled"),
        (RunBudgetExceeded("auxiliary budget"), "budget_exhausted"),
    ],
)
def test_interrupted_output_is_not_recovered_as_completed_answer(
    tmp_path, monkeypatch, error, reason
):
    """关闭原因落事实，局部内容只在本连接标记中断；参数：隔离根与停止原因；返回：无。"""

    def stream(*args, **kwargs):
        """模拟实际供应商流在部分输出后中断；参数：调用；返回：中断异常。"""
        yield ModelOutputDelta("text", "尚未完成的内容")
        raise error

    runtime, context = prepared_run(tmp_path, monkeypatch, stream)
    buffer = EventBuffer()
    projection = initial_projection(buffer, context)
    seen = []
    for event in runtime:
        seen.append(event)
        project_event(buffer, projection, context, event)
    closed = next(event for event in seen if isinstance(event, AssistantStreamClosed))
    assert closed.reason == reason
    assert not buffer.read(None, include_streams=True)["streams"]
    if reason == "cancelled":
        assert any(
            "未保存" in card.text and "尚未完成的内容" in card.text
            for card in projection.cards.values()
        )
    facts = RunFactStore(tmp_path).read_run(context.run_id)
    assert any(
        row["event"] == "llm:stream_closed" and row["reason"] == reason for row in facts
    )
    restarted_buffer = EventBuffer()
    projection.apply(
        {
            "session_id": context.session_id,
            "history": session_history(tmp_path, context.session_id),
            **restarted_buffer.read(None, include_streams=True),
        }
    )
    assert not any("尚未完成" in card.text for card in projection.cards.values())


def test_retry_clears_failed_attempt_and_reconnect_retains_only_current_attempt(
    tmp_path, monkeypatch
):
    """重试中重连只恢复当前尝试，不拼接失败内容；参数：隔离根；返回：无。"""

    def stream(*args, **kwargs):
        """同逻辑请求内先失败再返回新内容；参数：调用；返回：成功计划。"""
        yield ModelOutputDelta("text", "失败尝试")
        yield ModelOutputDelta("thinking", "过期思考")
        yield ModelRetryNotice(2, 3, 0.1, "transport_error")
        yield ModelOutputDelta("text", "重试成功")
        return LLMPlan(final_output="重试成功")

    runtime, context = prepared_run(tmp_path, monkeypatch, stream)
    buffer = EventBuffer()
    projection = initial_projection(buffer, context)
    reconnected = False
    for event in runtime:
        project_event(buffer, projection, context, event)
        if isinstance(event, AssistantTextDelta) and event.text == "重试成功":
            projection = initial_projection(buffer, context)
            card = next(
                card for card in projection.cards.values() if card.role == "assistant"
            )
            assert card.text == "重试成功" and card.reasoning == ""
            reconnected = True
    assert reconnected
    assert [
        card.text for card in projection.cards.values() if card.role == "assistant"
    ] == ["重试成功"]
    assert not buffer.read(None, include_streams=True)["streams"]


def test_recoverable_model_error_closes_its_request_before_corrected_answer(
    tmp_path, monkeypatch
):
    """可恢复解析错误的回执带原请求身份，下一回答不继承局部思考；参数：隔离根；返回：无。"""
    attempts = []

    def stream(*args, **kwargs):
        """首轮返回协议错误，后续模型独立修正；参数：调用；返回：错误或有效计划。"""
        attempts.append(kwargs["context"]["model_request_id"])
        if len(attempts) == 1:
            yield ModelOutputDelta("text", "错误片段")
            yield ModelOutputDelta("thinking", "首轮思考")
            return LLMPlan(
                model_error=ModelError(
                    "invalid_model_protocol", "fixture malformed output"
                )
            )
        yield ModelOutputDelta("text", "修正回答")
        return LLMPlan(final_output="修正回答")

    runtime, context = prepared_run(tmp_path, monkeypatch, stream)
    buffer = EventBuffer()
    completed = []
    for event in runtime:
        buffer.emit(event, run_id=context.run_id)
        if isinstance(event, AssistantTurnComplete):
            completed.append(event)
            assert not buffer.read(None, include_streams=True)["streams"]
    assert [event.message_id for event in completed] == attempts
    assert completed[-1].entry_id
    assert completed[0].stop_reason == "model_error:invalid_model_protocol"
    projection = initial_projection(buffer, context)
    projection.apply(
        {
            "session_id": context.session_id,
            "history": session_history(tmp_path, context.session_id),
            **buffer.read(None, include_streams=True),
        }
    )
    assert [
        card.text for card in projection.cards.values() if card.role == "assistant"
    ] == ["修正回答"]


def test_failure_after_stream_before_plan_handoff_closes_without_hiding_error(
    tmp_path, monkeypatch
):
    """供应商返回后证据处理失败也关闭局部流，原异常仍暴露；参数：隔离根；返回：无。"""

    def stream(*args, **kwargs):
        """提供未提交片段；参数：调用；返回：候选计划。"""
        yield ModelOutputDelta("text", "尚未交接")
        return LLMPlan(final_output="尚未交接")

    def fail(*args):
        """模拟真实状态提交失败；参数：调用；返回：抛出原失败。"""
        raise OSError("fixture state write failed")

    monkeypatch.setattr(
        "runtime.model_execution.ModelRequestRunner._record_prompt_snapshot_state", fail
    )
    runtime, context = prepared_run(tmp_path, monkeypatch, stream)
    buffer = EventBuffer()
    with pytest.raises(OSError, match="fixture state write failed"):
        for event in runtime:
            buffer.emit(event, run_id=context.run_id)
    assert not buffer.read(None, include_streams=True)["streams"]
    closures = [
        row
        for row in RunFactStore(tmp_path).read_run(context.run_id)
        if row["event"] == "llm:stream_closed"
    ]
    assert closures[-1]["reason"] == "model_request_failed"
    assert not any(
        row["role"] == "assistant"
        for row in session_history(tmp_path, context.session_id)
    )
