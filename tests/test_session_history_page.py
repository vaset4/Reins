"""历史分页的稳定锚点、读取范围和工具配对。

作者：xxx
时间：2026-09-29 23:00:00
"""

import pytest

from llm.messages import (
    AssistantMessage,
    StopReason,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from runtime.session_message_store import SessionMessageStore, SessionMessageStoreError


def test_pages_stay_on_original_branch_after_new_input_and_branch_change(
    tmp_path, monkeypatch
):
    """分页过程中新增内容和切换分支不改变已固定的历史；传参：隔离根；返回：无。"""
    store = SessionMessageStore(tmp_path)
    entries = [
        store.append_message(
            "s", UserMessage(f"message-{index}", (TextPart(str(index)),))
        )
        for index in range(12)
    ]
    first = store.history_page("s", limit=3)
    assert first.entries == tuple(entries[-3:])
    store.branch("s", entries[0].entry_id)
    store.append_message("s", UserMessage("new-branch", (TextPart("新分支"),)))

    def reject_full_read(*args, **kwargs):
        """禁止分页退回全历史读取；传参：调用参数；返回：测试失败。"""
        pytest.fail("history pagination must not read all entries")

    monkeypatch.setattr(store, "read_entries", reject_full_read)
    collected = list(first.entries)
    before = first.next_before
    while before is not None:
        page = store.history_page("s", leaf_id=first.leaf_id, before=before, limit=3)
        collected = [*page.entries, *collected]
        before = page.next_before
    assert collected == entries
    with pytest.raises(SessionMessageStoreError, match="cursor"):
        store.history_page("s", before=entries[-1].entry_id, limit=3)


def test_tool_results_are_paged_with_their_call(tmp_path):
    """页首向前补齐工具调用，不出现无调用的结果；传参：隔离根；返回：无。"""
    store = SessionMessageStore(tmp_path)
    user = store.append_message("s", UserMessage("user", (TextPart("操作"),)))
    call = store.append_message(
        "s",
        AssistantMessage(
            "call",
            (ToolCallPart("a", "read", {}), ToolCallPart("b", "read", {})),
            stop_reason=StopReason.TOOL_CALL,
        ),
    )
    first = store.append_message(
        "s", ToolResultMessage("one", "a", "read", (TextPart("甲"),), "success")
    )
    second = store.append_message(
        "s", ToolResultMessage("two", "b", "read", (TextPart("乙"),), "success")
    )
    page = store.history_page("s", limit=1)
    assert page.entries == (call, first, second)
    older = store.history_page(
        "s", leaf_id=page.leaf_id, before=page.next_before, limit=1
    )
    assert older.entries == (user,)
    assert older.next_before is None


def test_page_decodes_only_requested_records(tmp_path, monkeypatch):
    """分页只解码所需正文，不先读全量再切片；传参：隔离根；返回：无。"""
    store = SessionMessageStore(tmp_path)
    for index in range(10):
        store.append_message("s", UserMessage(f"m-{index}", (TextPart("正文"),)))
    decoded = []
    original = store._parse_entry

    def record_decode(value, session_id, line_number):
        """记录真实正文读取次数；传参：连接及记录；返回：原解析值。"""
        decoded.append(value)
        return original(value, session_id, line_number)

    monkeypatch.setattr(store, "_parse_entry", record_decode)
    assert len(store.history_page("s", limit=2).entries) == 2
    assert len(decoded) == 2


def test_background_history_page_returns_stable_cursor_and_visible_rows(tmp_path):
    """后台查询入口保留分页锚点和完整正文；传参：隔离根；返回：无。"""
    from types import SimpleNamespace
    from app.background.server import BackgroundServer

    store = SessionMessageStore(tmp_path)
    first = store.append_message("s", UserMessage("one", (TextPart("第一条"),)))
    second = store.append_message("s", UserMessage("two", (TextPart("第二条"),)))
    from runtime.workspaces import WorkspaceStore

    server = BackgroundServer(
        SimpleNamespace(
            services=SimpleNamespace(data_root=tmp_path),
            workspaces=WorkspaceStore(tmp_path),
        ),
        "test-token",
    )
    try:
        page = server.dispatch("history_page", {"session_id": "s", "limit": 1})
        assert (
            page["leaf_id"] == second.entry_id
            and page["next_before"] == second.entry_id
        )
        assert [row["text"] for row in page["history"]] == ["第二条"]
        older = server.dispatch(
            "history_page",
            {
                "session_id": "s",
                "leaf_id": page["leaf_id"],
                "before": page["next_before"],
                "limit": 1,
            },
        )
        assert older["history"][0]["entry_id"] == first.entry_id
        with pytest.raises(ValueError, match="positive integer"):
            server.dispatch("history_page", {"session_id": "s", "limit": "1"})
    finally:
        server.server_close()
