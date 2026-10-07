"""后台持久事实到终端卡片的显示合同。

作者：xxx
时间：2026-09-29 21:00:00
"""

import pytest

from frontends.tui.projection import ConversationProjection


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("success", "完成"),
        ("error", "失败"),
        ("partial", "部分完成"),
        (None, "状态未知"),
    ],
)
def test_history_tool_status_and_artifacts_survive_reconnect(status, expected):
    """重连后不把失败或缺失状态当成功，完整错误及产物仍可读；传参：状态；返回：无。"""
    projection = ConversationProjection()
    row = {
        "entry_id": "result",
        "role": "tool",
        "tool_call_id": "call",
        "tool_name": "file_write",
        "text": "已保存的结果",
        "error": "具体失败原因",
        "artifact_refs": ["artifact:output"],
    }
    if status is not None:
        row["status"] = status
    projection.apply({"session_id": "session", "history": [row], "cursor": 0})
    card = projection.cards["tool:session::call"]
    assert card.state == expected
    assert "具体失败原因" in card.text
    assert "artifact:output" in card.text


def test_tools_with_reused_call_ids_stay_in_their_run():
    """相同工具调用编号在不同运行中显示两条，重复poll不追加；传参：无；返回：无。"""
    projection = ConversationProjection()
    projection.apply(
        {"session_id": "s", "event_epoch": "host", "history": [], "cursor": 0}
    )
    events = [
        {
            "sequence": index,
            "run_id": run,
            "type": "ToolExecutionCompleted",
            "data": {
                "call_id": "same",
                "tool_name": "read",
                "output": run,
                "is_error": False,
            },
        }
        for index, run in enumerate(["run-a", "run-b"], 1)
    ]
    snapshot = {"session_id": "s", "event_epoch": "host", "events": events, "cursor": 2}
    projection.apply(snapshot)
    assert {card.text for card in projection.cards.values()} == {"run-a", "run-b"}
    assert not projection.apply(snapshot)


def test_restart_with_equal_cursor_requires_snapshot_and_rejects_late_old_epoch():
    """宿主重启即使游标相等也建立新基线，旧实例迟到事件不覆盖；传参：无；返回：无。"""
    projection = ConversationProjection()
    projection.apply(
        {"session_id": "s", "event_epoch": "old", "history": [], "cursor": 7}
    )
    with pytest.raises(ValueError, match="快照"):
        projection.apply({"session_id": "s", "event_epoch": "new", "cursor": 7})
    fresh = {
        "session_id": "s",
        "event_epoch": "new",
        "cursor": 7,
        "history": [{"entry_id": "saved", "role": "assistant", "text": "已保存回答"}],
    }
    projection.apply(fresh)
    stale = {
        "session_id": "s",
        "event_epoch": "old",
        "cursor": 8,
        "events": [
            {
                "sequence": 8,
                "type": "AssistantTextDelta",
                "data": {"text": "迟到旧内容"},
            }
        ],
    }
    assert not projection.apply(stale)
    assert projection.snapshot is fresh
    assert [card.text for card in projection.cards.values()] == ["已保存回答"]


def test_events_from_previous_session_cannot_switch_current_session():
    """迟到的另一会话增量不能切换当前会话；传参：无；返回：无。"""
    projection = ConversationProjection()
    projection.apply({"session_id": "current", "history": [], "cursor": 0})
    assert not projection.apply({"session_id": "previous", "events": [], "cursor": 1})
    assert projection.session_id == "current"
