"""验证网络重试的用户反馈、工具效果与明确停止。

作者：xxx
时间：2026-09-29 10:40:00
"""

from __future__ import annotations

from contextlib import closing
from functools import partial
from pathlib import Path

import pytest

from runtime.agent_loop import AgentLoop, State
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_messages import append_user_message
from runtime.stream_events import LifecycleChanged, SegmentPaused
from runtime.types import RunContext, Trigger
from scripts.testing.llm import (
    ScriptedTurnOptions,
    from_test_turns,
    scripted_provider_error,
)
from tasks.store import TaskStore
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


def _write_receipt(path: Path, _arguments: dict[str, object]) -> str:
    """留下不可重复的工具执行回执；参数：文件和工具参数；返回：完成说明。"""
    with path.open("a", encoding="utf-8") as receipt:
        receipt.write("executed\n")
    return "receipt saved"


def _context(data_root: Path) -> RunContext:
    """创建隔离会话的真实输入；参数：运行目录；返回：可执行的运行上下文。"""
    with closing(TaskStore(data_root)) as store:
        task = store.create_task("记录一次回执并说明结果")
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": task.goal},
        capability_lease=from_trigger("user", task_id=task.task_id),
    )
    append_user_message(data_root, context.session_id, task.goal)
    return context


def test_retry_is_visible_and_does_not_repeat_completed_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """工具成功后模型断流只重试模型；参数：隔离目录和替换器；返回：无。"""
    from approval import ApprovalDecision
    from tests.support.approval import install_approval

    install_approval(monkeypatch, lambda _request: ApprovalDecision.ONCE)
    monkeypatch.setattr("llm.client.compute_wait", lambda *_args, **_kwargs: 0.01)
    receipt = tmp_path / "receipt.txt"
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "write_receipt",
            "保存一次回执",
            {},
            "agent",
            ToolRisk.SAFE,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            executor=partial(_write_receipt, receipt),
        )
    )
    client = from_test_turns(
        [
            '{"type":"run_tools","tool":"write_receipt","arguments":{}}',
            scripted_provider_error(
                "transport_error", "Upstream stream disconnected", retryable=True
            ),
            '{"type":"final","content":"回执已保存"}',
        ],
        options=ScriptedTurnOptions(protocol_mode="text_json"),
    )
    data_root = tmp_path / "data"
    context = _context(data_root)
    loop = AgentLoop(data_root, llm_client=client, tool_registry=registry)
    events = list(loop.run_stream(context))
    retries = [
        event for event in events if type(event).__name__ == "ModelRetryScheduled"
    ]
    assert len(retries) == 1, "等待重试必须作为独立状态到达界面"
    assert loop.state == State.DONE and loop.last_output == "回执已保存"
    assert receipt.read_text(encoding="utf-8") == "executed\n"
    rows = RunFactStore(data_root).read_run(context.run_id)
    attempts = [row for row in rows if row.get("event") == "llm:attempt"]
    assert [row["success"] for row in attempts] == [True, False, True]
    assert attempts[1]["request_id"] == attempts[2]["request_id"]
    assert len([row for row in rows if row.get("event") == "tool:response"]) == 1


def test_exhausted_network_retries_keep_network_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """网络持续失败要保留原因而不是显示用户停止；参数：隔离目录和替换器；返回：无。"""
    monkeypatch.setattr("llm.client.compute_wait", lambda *_args, **_kwargs: 0.01)
    client = from_test_turns(
        [
            scripted_provider_error(
                "transport_error", "connection interrupted", retryable=True
            ),
        ],
        options=ScriptedTurnOptions(repeat=True),
    )
    context = _context(tmp_path)
    loop = AgentLoop(tmp_path, llm_client=client, tool_registry=ToolRegistry())
    events = list(loop.run_stream(context))
    assert (
        len(
            [event for event in events if type(event).__name__ == "ModelRetryScheduled"]
        )
        == 2
    )
    assert loop.state == State.FAILED
    terminal = [event for event in events if isinstance(event, LifecycleChanged)][-1]
    assert terminal.reason == "transport_error"
    assert not any(isinstance(event, SegmentPaused) for event in events)


def test_user_stop_during_retry_wait_prevents_another_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """用户在等待提示后停止时不得再次派发模型；参数：隔离目录和替换器；返回：无。"""
    monkeypatch.setattr("llm.client.compute_wait", lambda *_args, **_kwargs: 10.0)
    client = from_test_turns(
        [
            scripted_provider_error(
                "transport_error", "connection interrupted", retryable=True
            ),
            "must not run",
        ]
    )
    context = _context(tmp_path)
    loop = AgentLoop(tmp_path, llm_client=client, tool_registry=ToolRegistry())
    events = []
    for event in loop.run_stream(context):
        events.append(event)
        if type(event).__name__ == "ModelRetryScheduled":
            loop.cancellation.cancel()
    assert loop.state == State.PAUSED
    assert any(
        isinstance(event, SegmentPaused) and event.reason == "user stop requested"
        for event in events
    )
    attempts = [
        row
        for row in RunFactStore(tmp_path).read_run(context.run_id)
        if row.get("event") == "llm:attempt"
    ]
    assert len(attempts) == 1


def test_retry_feedback_survives_background_event_transport() -> None:
    """后台传输保留等待状态的全部信息；参数：无；返回：无。"""
    from app.background.events import EventBuffer, decode_event
    from runtime.stream_events import ModelRetryScheduled

    event = ModelRetryScheduled(
        attempt_index=2,
        max_attempts=3,
        wait_seconds=2.5,
        error_category="transport_error",
    )
    buffer = EventBuffer()
    buffer.emit(event)
    encoded = buffer.read(0)["events"]
    assert len(encoded) == 1
    assert decode_event(encoded[0]) == event


def test_retry_feedback_separates_partial_answer_and_keeps_final_error() -> None:
    """半截回答与重试分开显示，末次失败仍可见；参数：无；返回：无。"""
    from app.repl.console import capture_console
    from app.repl.render import EventRenderer
    from runtime.stream_events import (
        AssistantTextDelta,
        AssistantTurnComplete,
        ModelRetryScheduled,
    )

    with capture_console() as console:
        renderer = EventRenderer()
        renderer.render(AssistantTextDelta(text="上一次的半截回答"))
        renderer.render(
            ModelRetryScheduled(
                attempt_index=2,
                max_attempts=3,
                wait_seconds=2.5,
                error_category="transport_error",
            )
        )
        renderer.render(AssistantTextDelta(text="重试后的半截回答"))
        renderer.render(
            AssistantTurnComplete(
                content="MODEL_TRANSPORT_ERROR: connection interrupted",
                stop_reason="model_error:transport_error",
            )
        )
        output = console.export_text()
    assert "模型连接中断" in output and "2.5 秒" in output and "2/3" in output
    assert "user stop requested" not in output
    assert "MODEL_TRANSPORT_ERROR" in output
    assert "unknown event" not in output


def test_projection_keeps_retry_status_out_of_answer() -> None:
    """正式显示投影分开失败片段、重试状态和新回答；参数：无；返回：无。"""
    from app.background.events import EventBuffer
    from frontends.tui.projection import ConversationProjection
    from runtime.stream_events import (
        AssistantStreamClosed,
        AssistantTextDelta,
        ModelRetryScheduled,
    )

    projection, events = ConversationProjection(), EventBuffer()
    projection.apply({"session_id": "session", "history": [], **events.read(None)})
    events.emit(AssistantTextDelta(text="部分回答", message_id="answer"))
    projection.apply({"session_id": "session", **events.read(projection.cursor)})
    assert any(card.text == "部分回答" for card in projection.cards.values())
    events.emit(
        AssistantStreamClosed(message_id="answer", reason="retry", retrying=True)
    )
    events.emit(
        ModelRetryScheduled(
            attempt_index=2, max_attempts=3, wait_seconds=2.5, error_category="timeout"
        )
    )
    projection.apply({"session_id": "session", **events.read(projection.cursor)})
    assert (
        "timeout" in projection.activity_status()
        and "2/3" in projection.activity_status()
    )
    assert not any(card.role == "assistant" for card in projection.cards.values())
    events.emit(AssistantTextDelta(text="新的回答", message_id="answer"))
    projection.apply({"session_id": "session", **events.read(projection.cursor)})
    answers = [
        card.text for card in projection.cards.values() if card.role == "assistant"
    ]
    assert answers == ["新的回答"]
