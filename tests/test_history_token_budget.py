from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from llm.messages import (
    AgentMessage,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    validate_message_sequence,
)
from runtime.agent_loop import AgentLoop
from runtime.session_messages import (
    ToolExchange,
    append_assistant_message,
    append_tool_exchange,
    append_user_message,
)

SESSION_ID = "session-budget"


def _loop_with_window(tmp_path: Path, window: int) -> AgentLoop:
    """An AgentLoop whose llm_client only needs to expose a context window."""
    return AgentLoop(tmp_path, llm_client=SimpleNamespace(context_window=window))


def _users(messages: tuple[AgentMessage, ...]) -> list[AgentMessage]:
    return [item for item in messages if isinstance(item, UserMessage)]


def _seed_user_turns(data_root: Path, texts: list[str]) -> None:
    """把若干用户消息写进唯一消息 owner。

    参数：data_root 为 data 根目录；texts 为按时间顺序的用户输入
    返回：无
    """
    for text in texts:
        append_user_message(data_root, SESSION_ID, text)


def test_raw_history_does_not_change_with_model_window(tmp_path: Path) -> None:
    """模型变小不能直接删除未总结的历史；传参：临时存储；返回：无。"""
    bulk = "word " * 400
    _seed_user_turns(tmp_path, [f"{index} {bulk}" for index in range(40)])

    small = _loop_with_window(tmp_path, 4_000)._read_conversation_history(SESSION_ID)
    large = _loop_with_window(tmp_path, 200_000)._read_conversation_history(SESSION_ID)

    assert large.messages == small.messages
    assert len(_users(small.messages)) == 40
    assert small.truncated is False
    assert small.retained_count == len(small.messages)


def test_history_floor_keeps_single_oversized_message(tmp_path: Path) -> None:
    _seed_user_turns(tmp_path, ["word " * 5_000])

    selection = _loop_with_window(tmp_path, 1_000)._read_conversation_history(
        SESSION_ID
    )

    assert len(_users(selection.messages)) == 1  # never collapses to empty


def test_explicit_limit_preserves_legacy_count_behaviour(tmp_path: Path) -> None:
    _seed_user_turns(tmp_path, [f"message {index}" for index in range(25)])

    # No client window configured, explicit limit honoured (back-compat path).
    selection = AgentLoop(tmp_path)._read_conversation_history(SESSION_ID, limit=20)

    assert len(_users(selection.messages)) == 20
    assert selection.truncated is True  # truncation marker


def test_budget_trim_keeps_tool_pair_intact(tmp_path: Path) -> None:
    """预算裁尾不得切开 assistant(tool_calls) 与它的 tool 结果——F5 的裁剪永远交不出被
    Provider 拒绝的请求，靠的是整组保留，而不是靠下游补造一条假的调用公告。"""
    append_assistant_message(tmp_path, SESSION_ID, "word " * 2_000)
    append_tool_exchange(
        tmp_path,
        SESSION_ID,
        ToolExchange(
            call_id="call-1", tool_name="list", rendered="[tool_result] ok", status="ok"
        ),
    )
    append_user_message(tmp_path, SESSION_ID, "latest")

    messages = (
        _loop_with_window(tmp_path, 800)._read_conversation_history(SESSION_ID).messages
    )
    # 裁剪结果本身必须是合法调用图，不依赖下游补救
    validate_message_sequence(messages)

    for index, message in enumerate(messages):
        if not isinstance(message, ToolResultMessage):
            continue
        assert index > 0, "tool result cannot be first"
        prev = messages[index - 1]
        calls = [part for part in prev.content if isinstance(part, ToolCallPart)]
        assert any(part.call_id == message.call_id for part in calls)
        # 公告是真实工具名，不是补造出来的字面量 "tool"
        assert calls[0].tool_name == "list"
