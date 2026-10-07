"""Multi-turn conversation injection tests (V2.1 §6.4 segment 8).

Verifies that the second prompt's LLM call sees the first prompt's
user/assistant rows in its `messages` array, so the model can answer
follow-up questions that depend on prior context.

The REPL flow (next phase) will call `agent_loop._call_model` more than once
per task. Without history injection, the model would hallucinate or refuse
because each turn looks isolated. These tests pin the contract.
"""

from __future__ import annotations

from typing import Any

from runtime.session_messages import (
    append_assistant_message,
    append_user_message,
)
from runtime.watchdog import Watchdog

from llm.client import _extract_history
from llm.types import LLMPlan
from llm.messages import (
    AssistantMessage,
    TextPart,
    UserMessage,
    model_visible_text,
)
from llm.model_request import compose_model_request
from tools.tool_registry import ToolRegistry


def test_extract_history_passes_canonical_messages_through() -> None:
    """历史已由 Builder 选好并经 Store 校验，取用时原样透传不改写。"""
    history = (
        UserMessage("u1", (TextPart("hi"),)),
        AssistantMessage("a1", (TextPart("hello"),)),
    )
    assert _extract_history({"conversation_history": history}) == history


def test_extract_history_drops_non_message_entries() -> None:
    """非 canonical 消息一律不进请求：模型只该看到 Store 校验过的事实。"""
    valid = UserMessage("u1", (TextPart("ok"),))
    context = {
        "conversation_history": [
            valid,
            {"role": "user", "content": "raw dict"},
            ("assistant", "raw tuple"),
            "not a row",
            None,
        ]
    }
    assert _extract_history(context) == (valid,)


def test_extract_history_returns_empty_for_non_dict_context() -> None:
    assert _extract_history(None) == ()
    assert _extract_history("string context") == ()
    assert _extract_history(42) == ()


def test_prompt_composition_preserves_history_after_standalone_task() -> None:
    bundle = compose_model_request(
        task="next question",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={
            "conversation_history": (
                UserMessage("u1", (TextPart("first question"),)),
                AssistantMessage("a1", (TextPart("first answer"),)),
            )
        },
        registry=ToolRegistry(),
        context_window=30000,
    )
    messages = bundle.messages
    assert [item.kind for item in messages] == ["user", "user", "assistant"]
    assert model_visible_text(messages[0]).startswith("Task: next question")
    assert model_visible_text(messages[1]) == "first question"
    assert model_visible_text(messages[2]) == "first answer"
    # 指令不再伪装成消息，改由 instructions 承载
    assert bundle.request.instructions
    assert "system" not in [item.kind for item in messages]


def test_prompt_composition_without_history_keeps_single_user_message() -> None:
    bundle = compose_model_request(
        task="single shot",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={},
        registry=ToolRegistry(),
        context_window=30000,
    )
    assert [item.kind for item in bundle.messages] == ["user"]
    assert bundle.request.instructions


class _CapturingClient:
    """A drop-in that mirrors RealLLMClient.plan/continue signatures and
    captures the context for assertion."""

    def __init__(self) -> None:
        self.captured_contexts: list[Any] = []

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        self.captured_contexts.append(
            {"stage": "plan", "task": task, "context": context}
        )
        return LLMPlan()

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: object,
        context: object | None = None,
    ) -> LLMPlan:
        self.captured_contexts.append(
            {
                "stage": "continue",
                "task": task,
                "context": context,
            }
        )
        return LLMPlan()


def test_agent_loop_call_model_injects_conversation_history(tmp_path) -> None:
    """Integration: agent_loop._call_model reads Ledger conversation facts and
    forwards a `conversation_history` entry through the LLM context."""
    from runtime.agent_loop import AgentLoop
    from runtime.lease import from_trigger
    from runtime.types import RunContext, Trigger
    from tasks.store import TaskStore

    data_root = tmp_path / "data"
    store = TaskStore(data_root)
    record = store.create_task("multi-turn case")
    session_id = "session-multiturn"
    append_user_message(data_root, session_id, "first question")
    append_assistant_message(data_root, session_id, "first answer")

    capture = _CapturingClient()
    lease = from_trigger("user", task_id=record.task_id)
    loop = AgentLoop(data_root, llm_client=capture)  # type: ignore[arg-type]
    context = RunContext(
        task_id=record.task_id,
        session_id=session_id,
        trigger=Trigger.USER,
        payload={"message": "second question"},
        capability_lease=lease,
        segment_id="user-test",
    )

    # _call_model 现在是生成器（边生成边发增量），不排空则函数体一行都不执行
    loop._prepare_turn_core(context).store.close()
    list(
        loop._call_model(  # accessing internal helper is intentional under test
            task="second question",
            context=context,
            last_tool_result=None,
            watchdog=Watchdog(context.capability_lease),
        )
    )

    assert len(capture.captured_contexts) == 1
    captured = capture.captured_contexts[0]
    history = captured["context"]["conversation_history"]
    assert [item.kind for item in history] == ["user", "assistant"]
    assert [model_visible_text(item) for item in history] == [
        "first question",
        "first answer",
    ]


def _recovery_result(error_text: str) -> Any:
    """造一份可恢复错误观测，形状与 agent_loop 回灌用的一致。

    作者：xxx
    时间：2026-09-01 00:00:00
    传参：error_text 为 "分类: 原码" 形态的错误说明
    返回：status=error 且 meta.recoverable_observation 为真的 RunToolsResult
    """
    from runtime.types import RunToolsResult

    return RunToolsResult.error_result(
        action="model",
        tool_name="model",
        error=error_text,
        summary="recoverable error observation",
        meta={
            "recoverable_observation": True,
            "source": "model_parse",
            "error_type": "invalid_model_protocol",
            "error_message": "tool_arguments_must_be_object",
            "retryable": True,
        },
    )


def _loop_context(tmp_path, session_id: str):
    """搭一个能直接调 _call_model 的最小 loop 与运行上下文。

    作者：xxx
    时间：2026-09-01 00:00:00
    传参：tmp_path 为隔离目录；session_id 为会话 id
    返回：(loop, context, capture) 三元组
    """
    from runtime.agent_loop import AgentLoop
    from runtime.lease import from_trigger
    from runtime.types import RunContext, Trigger
    from tasks.store import TaskStore

    data_root = tmp_path / "data"
    record = TaskStore(data_root).create_task("recovery feedback case")
    append_user_message(data_root, session_id, "do the thing")
    capture = _CapturingClient()
    loop = AgentLoop(data_root, llm_client=capture)  # type: ignore[arg-type]
    context = RunContext(
        task_id=record.task_id,
        session_id=session_id,
        trigger=Trigger.USER,
        payload={"message": "do the thing"},
        capability_lease=from_trigger("user", task_id=record.task_id),
        segment_id="user-test",
    )
    return loop, context, capture


def test_recoverable_error_reason_reaches_model_context(tmp_path) -> None:
    """重试轮必须把可恢复错误的原因交给模型，否则模型只能重复犯同一个错。

    模型发出参数非 object 的工具调用后，流装配层拦下并归成
    invalid_model_protocol；循环把错误观测当 last_tool_result 传回来重试。
    这条链只要断在任何一环，模型第二轮读到的东西和第一轮完全一样。
    """
    loop, context, capture = _loop_context(tmp_path, "session-recovery-reason")

    loop._prepare_turn_core(context).store.close()
    list(
        loop._call_model(
            task="do the thing",
            context=context,
            last_tool_result=_recovery_result(
                "invalid_model_protocol: tool_arguments_must_be_object"
            ),
            watchdog=Watchdog(context.capability_lease),
        )
    )

    notice = capture.captured_contexts[0]["context"].get("recoverable_error_notice")
    assert notice, "重试轮的 model_context 必须带上错误原因"
    assert "tool_arguments_must_be_object" in notice, "原码要原样交给模型自己判断"


def test_recoverable_error_notice_is_absent_on_first_turn(tmp_path) -> None:
    """首轮没有错误可回灌，不得凭空造出一段说明。"""
    loop, context, capture = _loop_context(tmp_path, "session-recovery-first")

    loop._prepare_turn_core(context).store.close()
    list(
        loop._call_model(
            task="do the thing",
            context=context,
            last_tool_result=None,
            watchdog=Watchdog(context.capability_lease),
        )
    )

    assert "recoverable_error_notice" not in capture.captured_contexts[0]["context"]


def test_recoverable_error_notice_does_not_add_session_message(tmp_path) -> None:
    """错误说明是本轮请求的说明，不是对话里发生过的一轮（PRD R3）。"""
    from runtime.session_messages import read_history_rows

    session_id = "session-recovery-no-entry"
    loop, context, _ = _loop_context(tmp_path, session_id)
    data_root = tmp_path / "data"
    before = len(read_history_rows(data_root, session_id, limit=50))

    loop._prepare_turn_core(context).store.close()
    list(
        loop._call_model(
            task="do the thing",
            context=context,
            last_tool_result=_recovery_result(
                "invalid_model_protocol: tool_arguments_must_be_object"
            ),
            watchdog=Watchdog(context.capability_lease),
        )
    )

    assert len(read_history_rows(data_root, session_id, limit=50)) == before


def test_recoverable_error_notice_reaches_rendered_prompt() -> None:
    """错误说明必须真的渲染进交给模型的正文，落到 model_context 不算送达。

    只断言 model_context 的键会漏掉渲染这一层：把渲染短路掉，键仍然在，
    模型却什么也读不到——这条守的就是那个缺口。
    """
    from llm.client import ModelInput, render_model_input_text

    notice = "invalid_model_protocol: tool_arguments_must_be_object"
    rendered = render_model_input_text(
        ModelInput(
            task="do the thing", stage="continue", recoverable_error_notice=notice
        )
    )

    assert notice in rendered, "错误原因没进模型正文，回灌链断在渲染层"


def test_rendered_prompt_differs_between_retry_turns() -> None:
    """重试轮的正文必须与首轮不同，否则模型只会原样再犯一次（PRD 缺口本体）。"""
    from llm.client import ModelInput, render_model_input_text

    first = render_model_input_text(ModelInput(task="do the thing", stage="continue"))
    retry = render_model_input_text(
        ModelInput(
            task="do the thing",
            stage="continue",
            recoverable_error_notice="invalid_model_protocol: tool_arguments_must_be_object",
        )
    )

    assert first != retry, "两轮正文逐字节相同，等于没告诉模型上轮错在哪"
