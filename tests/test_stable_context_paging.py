"""历史分支与产物版本游标，以及大文件分页的内存边界。

作者：xxx
时间：2026-09-14 20:00:00
"""

from __future__ import annotations

import tracemalloc
from contextlib import closing

import pytest

from artifacts.store import ArtifactStore
from context.artifact_ref import store_large_output
from runtime.lease import from_trigger
from runtime.file_records import SourceCorruptionError, record_key
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_user_message
from tasks.store import TaskStore
from tools.read_artifact import read_artifact_page


def _artifact(root, text):
    """创建实际可回取的保存结果；传参：根与正文；返回：产物引用。"""
    with closing(TaskStore(root)) as store:
        task = store.create_task("保存长资料")
    ref = store_large_output(root, task.task_id, text)
    assert ref is not None
    return ref


def test_artifact_cursor_continues_exact_content_and_rejects_another_artifact(tmp_path):
    """游标绑定产物、模式与版本，连续读取不重复；传参：临时根；返回：无。"""
    text = "甲🙂\r\n乙\n" * 1000
    first = _artifact(tmp_path, text)
    second = _artifact(tmp_path, text + "另一个版本")
    page = read_artifact_page(tmp_path, first.artifact_id, lease=None, limit=601)
    cursor = page["meta"]["next_cursor"]
    pieces = [page["content"]]
    while cursor:
        page = read_artifact_page(
            tmp_path, first.artifact_id, lease=None, cursor=cursor, limit=601
        )
        pieces.append(page["content"])
        cursor = page["meta"]["next_cursor"]
    assert "".join(pieces) == text
    original = read_artifact_page(tmp_path, first.artifact_id, lease=None, limit=5)[
        "meta"
    ]["next_cursor"]
    with pytest.raises(ValueError, match="another artifact"):
        read_artifact_page(tmp_path, second.artifact_id, lease=None, cursor=original)
    with pytest.raises(ValueError, match="mode"):
        read_artifact_page(
            tmp_path, first.artifact_id, lease=None, cursor=original, mode="summary"
        )
    denied = from_trigger(
        "user", capabilities={"fs": {"read": [str(tmp_path / "denied")]}}
    )
    with pytest.raises(PermissionError):
        read_artifact_page(tmp_path, first.artifact_id, lease=denied, cursor=original)


def test_artifact_cursor_rejects_changed_content_without_manual_hash(tmp_path):
    """仅传回游标也能发现原文版本变化；传参：临时存储；返回：无。"""
    ref = _artifact(tmp_path, "原始内容" * 1000)
    first = read_artifact_page(tmp_path, ref.artifact_id, lease=None, limit=7)
    with closing(ArtifactStore(tmp_path)) as store:
        record = store.load_artifact(ref.artifact_id)
    assert record is not None
    path = tmp_path / record.path
    path.write_text("新原文" * 1000, encoding="utf-8")
    preserved = read_artifact_page(
        tmp_path, ref.artifact_id, lease=None, cursor=first["meta"]["next_cursor"]
    )
    assert preserved["content"] == ("原始内容" * 1000)[7:]
    assert record.retained_path is not None
    (tmp_path / record.retained_path).write_text("新原文" * 1000, encoding="utf-8")
    with pytest.raises(ValueError):
        read_artifact_page(
            tmp_path, ref.artifact_id, lease=None, cursor=first["meta"]["next_cursor"]
        )


def test_large_artifact_page_uses_bounded_memory(tmp_path):
    """读取小页面不会把整个Unicode文件加载进内存；传参：临时存储；返回：无。"""
    text = "甲🙂\r\n乙\n" * 500000
    ref = _artifact(tmp_path, text)
    tracemalloc.start()
    try:
        page = read_artifact_page(
            tmp_path, ref.artifact_id, lease=None, offset=400001, limit=1000
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert page["content"] == text[400001:401001]
    assert page["meta"]["total_count"] == len(text)
    assert peak < 6 * 1024 * 1024


def test_history_cursor_keeps_snapshot_after_append_and_rejects_rewind(tmp_path):
    """新增消息不改旧页范围，回退到共同祖先时游标明确失效；传参：临时存储；返回：无。"""
    from runtime.history_reader import read_history_page

    owner = SessionMessageStore(tmp_path)
    ids = [
        append_user_message(tmp_path, "paging-session", f"消息{index}")
        for index in range(6)
    ]
    first = read_history_page(
        owner.materialize("paging-session"), call_id="none", limit=2
    )
    append_user_message(tmp_path, "paging-session", "分页后新到的纠正")
    second = read_history_page(
        owner.materialize("paging-session"),
        call_id="none",
        limit=2,
        cursor=first["next_cursor"],
    )
    assert [message["message_id"] for message in first["messages"]] == ids[4:]
    assert [message["message_id"] for message in second["messages"]] == ids[2:4]
    last = read_history_page(
        owner.materialize("paging-session"),
        call_id="none",
        limit=2,
        cursor=second["next_cursor"],
    )
    assert [message["message_id"] for message in last["messages"]] == ids[:2]
    assert last["next_cursor"] is None and last["next_before"] is None
    original = owner.materialize("paging-session")
    owner.branch("paging-session", original.entries[3].entry_id)
    with pytest.raises(ValueError, match="branch"):
        read_history_page(
            owner.materialize("paging-session"),
            call_id="none",
            cursor=first["next_cursor"],
        )


def test_head_page_reports_unread_original_and_continues(tmp_path):
    """head只限制页长，未读原文仍有正确范围和游标；传参：临时根；返回：无。"""
    text = "资料原文\r\n" * 2000
    ref = _artifact(tmp_path, text)
    first = read_artifact_page(tmp_path, ref.artifact_id, lease=None, mode="head")
    assert first["content"] == text[:4096]
    assert first["meta"]["total_count"] == len(text)
    assert first["meta"]["truncated"]
    second = read_artifact_page(
        tmp_path, ref.artifact_id, lease=None, cursor=first["meta"]["next_cursor"]
    )
    assert second["content"] == text[4096:8192]


def test_history_cursor_rejects_rewritten_source(tmp_path):
    """同一分支的源正文变化也使游标失效；传参：临时根；返回：无。"""
    from runtime.history_reader import read_history_page

    owner = SessionMessageStore(tmp_path)
    for text in ("第一条原文", "后续第二条", "后续第三条"):
        append_user_message(tmp_path, "history-version", text)
    view = owner.materialize("history-version")
    entry = view.entries[0]
    page = read_history_page(view, call_id="none", limit=1)
    path = owner.database.source_path(
        "session_entry", record_key(entry.session_id, entry.entry_id)
    )
    original = path.read_bytes()
    corrupted = original.replace(
        "第一条原文".encode("utf-8"), "被改写原文".encode("utf-8")
    )
    assert corrupted != original
    path.write_bytes(corrupted)
    with pytest.raises(SourceCorruptionError, match="committed source corrupt"):
        read_history_page(
            owner.materialize("history-version"),
            call_id="none",
            cursor=page["next_cursor"],
        )
    assert path.read_bytes() == corrupted
