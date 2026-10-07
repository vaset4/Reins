"""验证压缩后的话题定位、精确原文与冻结分页。

作者：xxx
时间：2026-09-25 20:00:00
"""

from dataclasses import replace

import pytest

from context.summary_entries import SummaryContent, SummaryTopic
from runtime.history_reader import read_history_page
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_assistant_message, append_user_message


def history_with_topics(root):
    """建立两次压缩且已切换话题的会话；传参：隔离根；返回：原文与摘要所有者、快照。"""
    owner = SessionMessageStore(root)
    first = append_user_message(root, "topics", "数据库只用本地 SQLite，不选云方案")
    append_assistant_message(root, "topics", "离线使用是选择原因，导入方式待确认")
    append_user_message(root, "topics", "接着讨论界面")
    store = SessionCompactionStore(owner)
    topic = SummaryTopic(
        "database", "数据库方案", "本地 SQLite，原因是离线要求", (first,)
    )
    content = SummaryContent(topics=(topic,))
    record = store.publish(
        SummarySource(owner.materialize("topics"), 2),
        "",
        request_ids=("request-one",),
        content=content,
    )
    append_assistant_message(root, "topics", "界面使用浅色")
    second = owner.materialize("topics").messages[-1].message_id
    append_user_message(root, "topics", "现在回到数据库")
    content = SummaryContent(topics=(SummaryTopic("ui", "界面", "浅色", (second,)),))
    latest = store.publish(
        SummarySource(owner.materialize("topics"), 4, record),
        "",
        request_ids=("request-two",),
        content=content,
    )
    return owner, store, record, latest


def test_topic_scan_reports_unscanned_history_and_reads_exact_originals(tmp_path):
    """首屏未命中不代表旧话题不存在，续页能定位原话；传参：临时根；返回：无。"""
    owner, store, original, latest = history_with_topics(tmp_path)
    first = read_history_page(
        owner.materialize("topics"),
        call_id="none",
        summaries=store,
        view_kind="topics",
        query="数据库",
        limit=1,
    )
    assert first["topics"] == [] and not first["scan_complete"]
    append_user_message(tmp_path, "topics", "查询期间新增内容")
    next_page = read_history_page(
        owner.materialize("topics"),
        call_id="none",
        summaries=store,
        cursor=first["next_cursor"],
        limit=1,
    )
    assert next_page["scan_complete"]
    topic = next_page["topics"][0]
    assert topic["summary_id"] == original.summary_id and "离线" in topic["text"]
    source = read_history_page(
        owner.materialize("topics"),
        call_id="none",
        summaries=store,
        source_ref=topic["source_ref"],
    )
    assert "SQLite" in str(source["messages"]) and "查询期间新增" not in str(source)
    assert len(latest.new_message_ids) == 2 and len(latest.message_ids) == 4
    with pytest.raises(ValueError, match="cannot change"):
        read_history_page(
            owner.materialize("topics"),
            call_id="none",
            summaries=store,
            cursor=first["next_cursor"],
            query="界面",
        )


def test_summary_pages_do_not_include_uncovered_messages_or_cross_branches(tmp_path):
    """摘要编号限定累计原文范围，回退后拒绝旧来源；传参：临时根；返回：无。"""
    owner, store, original, _ = history_with_topics(tmp_path)
    view = owner.materialize("topics")
    first = read_history_page(
        view, call_id="none", summaries=store, summary_id=original.summary_id, limit=1
    )
    second = read_history_page(
        view, call_id="none", summaries=store, cursor=first["next_cursor"], limit=1
    )
    ids = {
        message["message_id"]
        for page in (first, second)
        for message in page["messages"]
    }
    assert ids == set(original.message_ids) and second["scan_complete"]
    owner.branch("topics", view.entries[0].entry_id)
    with pytest.raises(ValueError, match="branch"):
        read_history_page(
            owner.materialize("topics"),
            call_id="none",
            summaries=store,
            cursor=first["next_cursor"],
        )


def test_old_summary_without_topics_can_search_original_text(tmp_path):
    """旧格式没有目录仍可查询原文，并区分原件损坏；传参：临时根；返回：无。"""
    owner = SessionMessageStore(tmp_path)
    append_user_message(tmp_path, "legacy", "数据库采用 SQLite")
    append_user_message(tmp_path, "legacy", "稍后再讨论")
    store = SessionCompactionStore(owner)
    record = store.publish(
        SummarySource(owner.materialize("legacy"), 1),
        "旧摘要",
        request_ids=("old-request",),
    )
    page = read_history_page(
        owner.materialize("legacy"), call_id="none", summaries=store, view_kind="topics"
    )
    assert page["directory_state"] == "not_provided" and page[
        "summaries_without_directory"
    ] == [record.summary_id]
    original = read_history_page(
        owner.materialize("legacy"),
        call_id="none",
        summaries=store,
        summary_id=record.summary_id,
        query="SQLite",
    )
    assert original["returned_count"] == 1
    view = owner.materialize("legacy")
    changed = replace(
        view,
        messages=(
            replace(view.messages[0], message_id="changed-id"),
            *view.messages[1:],
        ),
    )
    with pytest.raises(ValueError, match="range changed"):
        store.read(record.summary_id, changed)
