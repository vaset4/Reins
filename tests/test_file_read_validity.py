"""用真实文件操作验证请求中的旧版本读取，不修改持久历史。

作者：xxx
时间：2026-09-25 20:00:00
"""

from scripts.testing.llm import _from_scripted
import json
import os

import pytest

from scripts.testing.llm import _ScriptedTurn
from llm.messages import ToolCallPart, ToolResultMessage, model_visible_text
from runtime.agent_loop import State
from runtime.session_messages import materialize_messages
from tests.test_file_batch_tracking import _runtime
from tests.test_session_runtime import capture_requests
from tools.file_persistence import content_sha256
from tools.readonly_inspection import DEFAULT_READ_MAX_CHARS


@pytest.mark.parametrize("final_read", [True, False])
def test_later_write_makes_all_old_reads_historical(tmp_path, monkeypatch, final_read):
    """成功写入后旧读取不冒充现状，最后重读才有当前正文；传参：隔离根、替换器、是否重读；返回：无。"""
    path = tmp_path / "sample.txt"
    path.write_text("version-one", encoding="utf-8")
    calls = [
        ToolCallPart("read-one", "file_read", {"path": str(path)}),
        ToolCallPart(
            "write-two",
            "file_write",
            {
                "path": str(path),
                "content": "version-two",
                "expected_sha256": content_sha256(b"version-one"),
            },
        ),
        ToolCallPart("read-two", "file_read", {"path": str(path)}),
        ToolCallPart(
            "write-three",
            "file_write",
            {
                "path": str(path),
                "content": "version-three",
                "expected_sha256": content_sha256(b"version-two"),
            },
        ),
    ]
    if final_read:
        calls.append(ToolCallPart("read-three", "file_read", {"path": str(path)}))
    loop, context, _ = _runtime(tmp_path, [])
    client = _from_scripted(
        [
            *(_ScriptedTurn(calls=(call,)) for call in calls),
            _ScriptedTurn(text="完成观察"),
        ]
    )
    requests = capture_requests(client, monkeypatch)
    loop.llm_client = client
    assert loop.run(context) is State.DONE
    reads = [
        message
        for message in requests[-1].messages
        if isinstance(message, ToolResultMessage) and message.tool_name == "file_read"
    ]
    assert len(reads) == 2 + int(final_read)
    for message in reads[:2]:
        payload = json.loads(model_visible_text(message))
        assert (
            payload["file_read_state"] == "historical"
            and payload["reason"] == "superseded_file_version"
        )
        assert (
            payload["read_original"]["arguments"]["artifact_id"]
            in message.artifact_refs
        )
    if final_read:
        assert "version-three" in model_visible_text(
            reads[-1]
        ) and "file_read_state" not in model_visible_text(reads[-1])
    durable = materialize_messages(loop.data_root, context.session_id)
    assert all(
        "file_read_state" not in model_visible_text(message) for message in durable
    )
    assert path.read_text(encoding="utf-8") == "version-three"


@pytest.mark.parametrize(
    "new_content, expected_hash",
    [("original", content_sha256(b"original")), ("unpublished", "bad-version")],
)
def test_unchanged_or_failed_write_does_not_invent_a_new_version(
    tmp_path, monkeypatch, new_content, expected_hash
):
    """无变化与失败写入不废弃已有读取；传参：根、替换器、候选和写前版本；返回：无。"""
    path = tmp_path / "stable.txt"
    path.write_text("original", encoding="utf-8")
    loop, context, _ = _runtime(tmp_path, [])
    client = _from_scripted(
        [
            _ScriptedTurn(
                calls=(ToolCallPart("read", "file_read", {"path": str(path)}),)
            ),
            _ScriptedTurn(
                calls=(
                    ToolCallPart(
                        "write",
                        "file_write",
                        {
                            "path": str(path),
                            "content": new_content,
                            "expected_sha256": expected_hash,
                        },
                    ),
                )
            ),
            _ScriptedTurn(text="已核对实际结果"),
        ]
    )
    requests = capture_requests(client, monkeypatch)
    loop.llm_client = client
    assert loop.run(context) is State.DONE
    read = next(
        message
        for message in requests[-1].messages
        if isinstance(message, ToolResultMessage) and message.tool_name == "file_read"
    )
    assert "original" in model_visible_text(
        read
    ) and "file_read_state" not in model_visible_text(read)


def test_complementary_pages_keep_both_ranges_and_replace_only_repeated_page(
    tmp_path, monkeypatch
):
    """同版本互补页保留，只有被新读取完整覆盖的旧页改引用；传参：隔离根和替换器；返回：无。"""
    path = tmp_path / "pages.txt"
    path.write_text(
        "A" * (DEFAULT_READ_MAX_CHARS - 1) + "\n最后一页标记",
        encoding="utf-8",
        newline="\n",
    )
    calls = [
        ToolCallPart("first", "file_read", {"path": str(path)}),
        ToolCallPart("tail", "file_read", {"path": str(path), "start_line": 2}),
        ToolCallPart("tail-again", "file_read", {"path": str(path), "start_line": 2}),
    ]
    loop, context, _ = _runtime(tmp_path, [])
    client = _from_scripted(
        [
            *(_ScriptedTurn(calls=(call,)) for call in calls),
            _ScriptedTurn(text="核对完毕"),
        ]
    )
    requests = capture_requests(client, monkeypatch)
    loop.llm_client = client
    assert loop.run(context) is State.DONE
    reads = {
        message.call_id: json.loads(model_visible_text(message))
        for message in requests[-1].messages
        if isinstance(message, ToolResultMessage) and message.tool_name == "file_read"
    }
    assert "file_read_state" not in reads["first"]
    assert reads["tail"]["reason"] == "covered_by_later_read"
    assert "file_read_state" not in reads["tail-again"]
    assert "最后一页标记" in str(reads["tail-again"])


def test_selected_first_line_does_not_replace_a_distinct_later_line(
    tmp_path, monkeypatch
):
    """指定行已读完不等于整文件读齐，互补来源不能被错误去重；参数：隔离根与请求捕获器；返回：无。"""
    path = tmp_path / "selected-lines.txt"
    path.write_text("首行约定\n中间材料\n最后约定\n", encoding="utf-8", newline="\n")
    calls = [
        ToolCallPart(
            "last-line",
            "file_read",
            {"path": str(path), "start_line": 3, "line_count": 1},
        ),
        ToolCallPart(
            "first-line",
            "file_read",
            {"path": str(path), "start_line": 1, "line_count": 1},
        ),
    ]
    loop, context, _ = _runtime(tmp_path, [])
    client = _from_scripted(
        [
            *(_ScriptedTurn(calls=(call,)) for call in calls),
            _ScriptedTurn(text="两项约定保持"),
        ]
    )
    requests = capture_requests(client, monkeypatch)
    loop.llm_client = client
    assert loop.run(context) is State.DONE
    reads = {
        row.call_id: json.loads(model_visible_text(row))
        for row in requests[-1].messages
        if isinstance(row, ToolResultMessage) and row.tool_name == "file_read"
    }
    assert "file_read_state" not in reads["last-line"]
    assert "最后约定" in str(reads["last-line"]) and "首行约定" in str(
        reads["first-line"]
    )


@pytest.mark.parametrize("alias_kind", ["relative", "hardlink"])
def test_alias_write_tracks_actual_replacement_without_invalidating_other_file(
    tmp_path, monkeypatch, alias_kind
):
    """路径别名影响同一路径，无法保留硬链接拓扑时明确拒绝写入；传参：隔离根和别名方式；返回：无。"""
    path = tmp_path / "source.txt"
    path.write_text("old-body", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    if alias_kind == "hardlink":
        os.link(path, alias)
    else:
        alias = tmp_path / "." / "source.txt"
    alias_argument = str(alias) if alias_kind == "hardlink" else "source.txt"
    calls = [
        ToolCallPart("original", "file_read", {"path": str(path)}),
        ToolCallPart(
            "write-alias",
            "file_write",
            {
                "path": alias_argument,
                "content": "new-body",
                "expected_sha256": content_sha256(b"old-body"),
            },
        ),
        ToolCallPart("current", "file_read", {"path": str(path)}),
    ]
    loop, context, _ = _runtime(tmp_path, [])
    client = _from_scripted(
        [
            *(_ScriptedTurn(calls=(call,)) for call in calls),
            _ScriptedTurn(text="核对完毕"),
        ]
    )
    requests = capture_requests(client, monkeypatch)
    loop.llm_client = client
    assert loop.run(context) is State.DONE
    reads = {
        message.call_id: json.loads(model_visible_text(message))
        for message in requests[-1].messages
        if isinstance(message, ToolResultMessage) and message.tool_name == "file_read"
    }
    expected = (
        "covered_by_later_read"
        if alias_kind == "hardlink"
        else "superseded_file_version"
    )
    assert reads["original"]["reason"] == expected
    assert path.read_text(encoding="utf-8") == (
        "old-body" if alias_kind == "hardlink" else "new-body"
    )
    assert alias.read_text(encoding="utf-8") == (
        "old-body" if alias_kind == "hardlink" else "new-body"
    )
    if alias_kind == "hardlink":
        write = next(
            row
            for row in loop.operations.for_session(context.session_id)
            if row["call"]["call_id"] == "write-alias"
        )
        assert "hard_link_semantics_not_supported" in json.dumps(write)
        assert path.samefile(alias) and path.stat().st_nlink == 2
    assert "file_read_state" not in reads["current"]
