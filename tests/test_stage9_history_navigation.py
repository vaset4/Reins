"""【上下文】【原文导航】区分尚无分段目录与无命中，并检索工具真实输入。

作者：xxx
时间：2026-10-01 14:30:06
"""

from __future__ import annotations

from dataclasses import replace

from runtime.history_reader import read_history_page
from runtime.session_compaction import SessionCompactionStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import (
    ToolExchange,
    append_tool_exchange,
    append_user_message,
)
from tests.test_stage9_history import segment_content, source_history


def test_legacy_structured_summary_reports_missing_segment_directory(tmp_path):
    """旧条目摘要可读，但不能假报已经建立四档目录；参数：隔离空间；返回：无。"""
    messages, source, first = source_history(tmp_path)
    content = replace(segment_content(source, first), segments=(), checks=())
    store = SessionCompactionStore(messages)
    saved = store.publish(
        source, content.render(), content=content, request_ids=("generation",)
    )
    page = read_history_page(
        messages.materialize("long"),
        call_id="read",
        summaries=store,
        view_kind="segments",
    )
    assert page["directory_state"] == "not_provided"
    assert page["summaries_without_directory"] == [saved.summary_id]
    assert page["segments"] == []
    original = read_history_page(
        messages.materialize("long"),
        call_id="read",
        summaries=store,
        summary_id=saved.summary_id,
        query="500",
    )
    assert original["returned_count"] > 0


def test_original_search_matches_tool_arguments_and_returns_complete_exchange(tmp_path):
    """工具参数中的定位词也能命中，返回完整调用和结果；参数：隔离空间；返回：无。"""
    append_tool_exchange(
        tmp_path,
        "tool-source",
        ToolExchange(
            "read-config",
            "file_read",
            rendered="配置读取失败",
            args={"path": "needletoken/config.toml"},
            status="error",
        ),
    )
    append_user_message(tmp_path, "tool-source", "继续定位")
    view = SessionMessageStore(tmp_path).materialize("tool-source")
    page = read_history_page(view, call_id="search", query="needletoken")
    assert page["returned_count"] == 2
    call, result = page["messages"]
    assert call["content"][0]["arguments"] == {"path": "needletoken/config.toml"}
    assert result["status"] == "error"
