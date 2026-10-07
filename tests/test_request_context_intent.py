"""历史补读工具通过同一结果消息进入后续真实模型请求。

作者：xxx
时间：2026-09-14 15:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

import json

from llm.messages import ToolCallPart, ToolResultMessage
from runtime.session_messages import append_user_message, materialize_messages
from tests.test_native_actions import native_registry
from tests.test_session_runtime import capture_requests
from tests.test_tool_batch_execution import make_run


def test_history_page_reaches_next_actual_model_request(tmp_path, monkeypatch):
    """按消息身份补读的原文进入下一实际请求；传参：目录/替换器；返回：无。"""
    registry = native_registry()
    loop, context, _client = make_run(tmp_path, registry, [])
    ids = [
        append_user_message(
            tmp_path, context.session_id, f"材料-{index} " + "长材料 " * 100
        )
        for index in range(8)
    ]
    client = from_test_native_tool_then_final(
        [ToolCallPart("history-read", "read_history", {"before": ids[3], "limit": 2})],
        "已结合旧材料",
    )
    loop.llm_client = client
    requests = capture_requests(client, monkeypatch)
    list(loop.run_stream(context))
    result = next(
        item for item in requests[-1].messages if isinstance(item, ToolResultMessage)
    )
    payload = json.loads(json.loads(result.content[0].text)["output"])
    assert [row["message_id"] for row in payload["messages"]] == ids[1:3]
    assert payload["next_before"] == ids[1]
    assert "材料-1" in str(result)
    assert "材料-1" in str(requests[0].messages)


def test_history_default_page_excludes_its_own_call(tmp_path):
    """默认游标从当前公告前开始，避免回读自己并占用页长；传参：目录；返回：无。"""
    loop, context, _client = make_run(tmp_path, native_registry(), [])
    loop.llm_client = from_test_native_tool_then_final(
        [ToolCallPart("history-read", "read_history", {})], "已读"
    )
    list(loop.run_stream(context))
    result = next(
        item
        for item in materialize_messages(tmp_path, context.session_id)
        if isinstance(item, ToolResultMessage)
    )
    page = json.loads(json.loads(result.content[0].text)["output"])
    assert len(page["messages"]) == 1
    assert page["messages"][0]["kind"] == "user"
    assert page["next_before"] is None


def test_unknown_history_cursor_returns_explicit_error(tmp_path):
    """未知游标不静默回到第一页；传参：目录；返回：无。"""
    loop, context, _client = make_run(tmp_path, native_registry(), [])
    loop.llm_client = from_test_native_tool_then_final(
        [ToolCallPart("history-read", "read_history", {"before": "missing"})],
        "需要修正游标",
    )
    list(loop.run_stream(context))
    result = next(
        item
        for item in materialize_messages(tmp_path, context.session_id)
        if isinstance(item, ToolResultMessage)
    )
    assert result.status == "error"
    assert "cursor is not in the current branch" in str(result)
