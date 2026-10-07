from __future__ import annotations

from pathlib import Path

import pytest

from llm.messages import (
    AssistantMessage,
    StopReason,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from runtime.session_message_store import (
    SessionMessageStore,
    SessionMessageStoreError,
)

SESSION_ID = "session-1"


def _user(message_id: str, text: str) -> UserMessage:
    """构造测试所需的规范用户消息。

    参数：message_id 为消息标识；text 为模型可见文本
    返回：不可变 UserMessage
    """
    return UserMessage(message_id, (TextPart(text),))


def _assistant_call(message_id: str, call_id: str, tool_name: str) -> AssistantMessage:
    """构造以工具调用结束的规范助手消息。

    参数：message_id/call_id/tool_name 为消息、调用和工具标识
    返回：停止原因为 tool_call 的 AssistantMessage
    """
    return AssistantMessage(
        message_id,
        (ToolCallPart(call_id, tool_name, {"path": "README.md"}),),
        stop_reason=StopReason.TOOL_CALL,
    )


def _tool_result(message_id: str, call_id: str, tool_name: str) -> ToolResultMessage:
    """构造与工具调用关联的成功结果。

    参数：message_id/call_id/tool_name 为消息、调用和工具标识
    返回：包含模型可见文本的 ToolResultMessage
    """
    return ToolResultMessage(
        message_id,
        call_id,
        tool_name,
        (TextPart("done"),),
        "success",
    )


def test_append_creates_header_and_materializes_messages(tmp_path: Path) -> None:
    """首次追加创建 header，后续 Entry 认当前 leaf 为父节点。"""
    store = SessionMessageStore(tmp_path)

    first = store.append_message(
        SESSION_ID,
        _user("msg-1", "hello"),
        run_id="run-1",
        task_id="task-1",
    )
    second = store.append_message(
        SESSION_ID,
        _user("msg-2", "next"),
    )

    assert first.parent_id is None
    assert second.parent_id == first.entry_id
    materialized = store.materialize(SESSION_ID)
    assert materialized.leaf_id == second.entry_id
    assert tuple(message.message_id for message in materialized.messages) == (
        "msg-1",
        "msg-2",
    )
    assert store.read_entries(SESSION_ID)[0].run_id == "run-1"
    assert store.list_session_ids() == (SESSION_ID,)
    assert not (tmp_path / "sessions").exists()


def test_branch_keeps_old_history_and_survives_restart(tmp_path: Path) -> None:
    """回退只追加 branch Entry，重启后当前路径为 A 到 C 且旧 B 仍在数据库。"""
    store = SessionMessageStore(tmp_path)
    first = store.append_message(SESSION_ID, _user("msg-a", "A"))
    old_branch = store.append_message(SESSION_ID, _user("msg-b", "B"))
    before_branch = store.read_entries(SESSION_ID)

    branch = store.branch(SESSION_ID, first.entry_id)
    new_branch = store.append_message(SESSION_ID, _user("msg-c", "C"))

    source = store.read_entries(SESSION_ID)
    assert source[: len(before_branch)] == before_branch
    assert branch.parent_id == first.entry_id
    assert old_branch in source
    restarted = SessionMessageStore(tmp_path).materialize(SESSION_ID)
    assert restarted.leaf_id == new_branch.entry_id
    assert tuple(entry.entry_id for entry in restarted.entries) == (
        first.entry_id,
        branch.entry_id,
        new_branch.entry_id,
    )
    assert tuple(message.message_id for message in restarted.messages) == (
        "msg-a",
        "msg-c",
    )


def test_message_id_uniqueness_is_scoped_to_each_materialized_path(
    tmp_path: Path,
) -> None:
    """不同保留分支可复用 message_id，但任一根到叶路径内仍保持唯一。"""
    store = SessionMessageStore(tmp_path)
    first = store.append_message(SESSION_ID, _user("msg-a", "A"))
    branch_parent = store.append_message(
        SESSION_ID,
        _user("msg-branch", "B"),
    )
    branch = store.branch(SESSION_ID, first.entry_id)

    new_branch = store.append_message(
        SESSION_ID,
        _user("msg-branch", "C"),
    )

    restored = SessionMessageStore(tmp_path).materialize(SESSION_ID)
    assert tuple(message.message_id for message in restored.messages) == (
        "msg-a",
        "msg-branch",
    )
    assert tuple(entry.entry_id for entry in store.read_entries(SESSION_ID)) == (
        first.entry_id,
        branch_parent.entry_id,
        branch.entry_id,
        new_branch.entry_id,
    )


def test_pending_tool_call_is_recoverable_and_result_closes_it(tmp_path: Path) -> None:
    """当前路径可停在 pending tool call，追加正确结果后 pending 标识清空。"""
    store = SessionMessageStore(tmp_path)
    store.append_message(
        SESSION_ID,
        _assistant_call("msg-call", "call-1", "read_file"),
    )

    pending = store.materialize(SESSION_ID)
    assert pending.pending_tool_calls == ("call-1",)

    store.append_message(
        SESSION_ID,
        _tool_result("msg-result", "call-1", "read_file"),
    )
    assert store.materialize(SESSION_ID).pending_tool_calls == ()


@pytest.mark.parametrize(
    "message",
    [
        _tool_result("msg-result", "missing", "read_file"),
        _tool_result("msg-result", "call-1", "write_file"),
    ],
)
def test_rejects_orphan_or_wrong_name_tool_result(
    tmp_path: Path,
    message: ToolResultMessage,
) -> None:
    """孤立结果和错名结果在 Store append 边界显式失败。"""
    store = SessionMessageStore(tmp_path)
    if message.call_id == "call-1":
        store.append_message(
            SESSION_ID,
            _assistant_call("msg-call", "call-1", "read_file"),
        )
    else:
        store.create_session(SESSION_ID)
    original = store.read_entries(SESSION_ID)
    with pytest.raises(SessionMessageStoreError) as raised:
        store.append_message(SESSION_ID, message)

    assert raised.value.code == "invalid_message_sequence"
    assert store.read_entries(SESSION_ID) == original


def test_pending_call_must_be_on_terminal_assistant_message(tmp_path: Path) -> None:
    """pending call 后若已出现新用户消息则不是可恢复的末尾工具状态。"""
    store = SessionMessageStore(tmp_path)
    store.append_message(
        SESSION_ID,
        _assistant_call("msg-call", "call-1", "read_file"),
    )
    original = store.read_entries(SESSION_ID)
    with pytest.raises(SessionMessageStoreError) as raised:
        store.append_message(
            SESSION_ID,
            _user("msg-user", "continue"),
        )

    assert raised.value.code == "invalid_message_sequence"
    assert store.read_entries(SESSION_ID) == original


def test_duplicate_tool_result_fails_before_append(tmp_path: Path) -> None:
    """同一 call_id 的第二条工具结果必须失败且不修改已闭合路径。"""
    store = SessionMessageStore(tmp_path)
    store.append_message(
        SESSION_ID,
        _assistant_call("msg-call", "call-1", "read_file"),
    )
    result = _tool_result("msg-result", "call-1", "read_file")
    store.append_message(SESSION_ID, result)
    original = store.read_entries(SESSION_ID)

    with pytest.raises(SessionMessageStoreError) as raised:
        store.append_message(
            SESSION_ID,
            _tool_result("msg-result-2", "call-1", "read_file"),
        )

    assert raised.value.code == "invalid_message_sequence"
    assert store.read_entries(SESSION_ID) == original


def test_duplicate_message_id_on_same_path_fails_before_append(tmp_path: Path) -> None:
    """同一 materialized 路径内重复 message_id 必须失败且不追加。"""
    store = SessionMessageStore(tmp_path)
    store.append_message(SESSION_ID, _user("msg-1", "one"))
    original = store.read_entries(SESSION_ID)

    with pytest.raises(SessionMessageStoreError) as raised:
        store.append_message(
            SESSION_ID,
            _user("msg-1", "duplicate"),
        )

    assert raised.value.code == "invalid_message_sequence"
    assert store.read_entries(SESSION_ID) == original


def test_sessions_are_isolated(tmp_path: Path) -> None:
    """不同 session 使用独立身份和当前 leaf，不共享消息事实。"""
    store = SessionMessageStore(tmp_path)
    store.append_message(SESSION_ID, _user("msg-1", "one"))
    store.append_message("session-2", _user("msg-2", "two"))

    assert set(store.list_session_ids()) == {SESSION_ID, "session-2"}
    assert store.materialize(SESSION_ID).messages[0].message_id == "msg-1"
    assert store.materialize("session-2").messages[0].message_id == "msg-2"
