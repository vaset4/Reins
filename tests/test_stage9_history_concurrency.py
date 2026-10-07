"""【上下文】【后台一致性】生成中的取消、分支、前驱和工具追加保持真实结果。

作者：xxx
时间：2026-10-01 14:30:06
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from llm.messages import (
    AssistantMessage,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    model_visible_text,
)
from runtime.context_compaction_jobs import ContextCompactionJobs
from runtime.context_compaction_worker import execute_compaction
from runtime.session_compaction import SessionCompactionStore
from runtime.workspaces import Workspace
from tests.test_stage9_history import (
    background_request,
    segment_content,
    source_history,
)
from tools.tool_registry import ToolRegistry


@pytest.mark.parametrize(
    "change,expected",
    [("cancel", "paused"), ("branch", "failed"), ("predecessor", "failed")],
)
def test_generation_conflicts_preserve_originals_and_do_not_publish_stale_work(
    tmp_path, monkeypatch, change, expected
):
    """生成中发生真实变更后拒绝旧候选；参数：隔离空间、冲突和预期；返回：无。"""
    jobs, accepted, request, client, calls, messages = background_request(
        tmp_path, monkeypatch
    )
    before = messages.read_entries("long")
    store = SessionCompactionStore(messages)
    adapter = client._adapter_registry.require("scripted_test")
    original_stream = adapter.stream
    competitor = []

    def stream(model_request, **options):
        """首个模型调用期间注入状态变更；参数：真实准备请求；返回：有来源的响应。"""
        if not calls:
            if change == "cancel":
                jobs.cancel(accepted["job_id"])
            elif change == "branch":
                messages.branch("long", before[0].entry_id)
            else:
                material = jobs.material(accepted)
                content = segment_content(
                    material.source, material.source.view.messages[0].message_id
                )
                competitor.append(
                    store.publish(
                        material.source,
                        content.render(),
                        content=content,
                        request_ids=("generate", "audit"),
                    )
                )
        return original_stream(model_request, **options)

    monkeypatch.setattr(adapter, "stream", stream)
    result = execute_compaction(
        request,
        data_root=tmp_path,
        llm_factory=lambda _: client,
        registry_factory=ToolRegistry,
    )
    assert result.status == expected and calls
    row = jobs.load(accepted["job_id"])
    assert row["summary_id"] is None
    assert row["status"] == ("cancelled" if change == "cancel" else "failed")
    assert messages.read_entries("long")[: len(before)] == before
    current = store.current(messages.materialize("long"))
    assert (current.summary_id if current else None) == (
        competitor[0].summary_id if competitor else None
    )
    expected_error = {
        "cancel": "cancel",
        "branch": "branch",
        "predecessor": "superseded",
    }
    assert expected_error[change] in result.error


def test_tool_result_appended_during_generation_remains_exact_after_restart(
    tmp_path, monkeypatch
):
    """生成期间新增工具原件不被摘要覆盖或重放；参数：隔离空间和模型替换；返回：无。"""
    jobs, accepted, request, client, calls, messages = background_request(
        tmp_path, monkeypatch
    )
    messages.append_message(
        "long",
        AssistantMessage(
            "pending-read",
            (ToolCallPart("new-result", "file_read", {"path": "missing.txt"}),),
        ),
    )
    adapter = client._adapter_registry.require("scripted_test")
    original_stream = adapter.stream

    def stream(model_request, **options):
        """模型整理冻结来源时追加失败工具回执；参数：请求；返回：辅助响应。"""
        if not calls:
            messages.append_message(
                "long",
                ToolResultMessage(
                    "new-result-message",
                    "new-result",
                    "file_read",
                    (TextPart("ENOENT：文件不存在，尚未读取"),),
                    "error",
                    error="ENOENT",
                ),
            )
        return original_stream(model_request, **options)

    monkeypatch.setattr(adapter, "stream", stream)
    outcome = execute_compaction(
        request,
        data_root=tmp_path,
        llm_factory=lambda _: client,
        registry_factory=ToolRegistry,
    )
    assert outcome.status == "done", outcome.error
    source_ids = {
        message.message_id for message in jobs.material(accepted).source.view.messages
    }
    saved = SessionCompactionStore(messages).current(messages.materialize("long"))
    assert saved and set(saved.message_ids).issubset(source_ids)
    before = messages.read_entries("long")
    count = len(calls)
    restored = ContextCompactionJobs(tmp_path)
    restored.update(accepted["job_id"], status="running", summary_id=None)
    outcome = execute_compaction(
        request,
        data_root=tmp_path,
        llm_factory=lambda _: client,
        registry_factory=ToolRegistry,
    )
    assert outcome.status == "done" and len(calls) == count
    assert messages.read_entries("long") == before
    results = [
        message
        for message in messages.materialize("long").messages
        if isinstance(message, ToolResultMessage) and message.call_id == "new-result"
    ]
    assert len(results) == 1 and results[0].status == "error"
    assert model_visible_text(results[0]) == "ENOENT：文件不存在，尚未读取"


def test_unavailable_workspace_is_persisted_as_a_failed_job(tmp_path, monkeypatch):
    """后台重启发现目录不可用时记录失败而非永久排队；参数：隔离空间；返回：无。"""
    jobs, accepted, request, client, calls, _ = background_request(
        tmp_path, monkeypatch
    )

    def unavailable(workspace: Workspace):
        """模拟原目录已移动；参数：冻结工作区；返回：不返回。"""
        missing = replace(workspace, project_root=tmp_path / "moved-project")
        raise FileNotFoundError(f"原工作区不可用：{missing.project_root}")

    monkeypatch.setattr(Workspace, "require_available", unavailable)
    outcome = execute_compaction(
        request,
        data_root=tmp_path,
        llm_factory=lambda _: client,
        registry_factory=ToolRegistry,
    )
    assert outcome.status == "failed" and not calls
    row = jobs.load(accepted["job_id"])
    assert row["status"] == "failed" and "FileNotFoundError" in row["error"]


def test_same_length_source_change_is_rejected_before_publication(
    tmp_path, monkeypatch
):
    """消息身份和长度未变仍须按真实内容拒绝旧摘要；参数：隔离空间与读取替换；返回：无。"""
    messages, source, first = source_history(tmp_path)
    content = segment_content(source, first)
    original = source.view.messages[0]
    changed = replace(original, content=(TextPart("费用不得超过900，禁止上传"),))
    assert len(model_visible_text(changed)) == len(model_visible_text(original))
    current = replace(source.view, messages=(changed, *source.view.messages[1:]))

    def changed_source(_session_id):
        """模拟发布时来源字节已经变化；参数：会话；返回：同身份但不同正文的当前视图。"""
        return current

    monkeypatch.setattr(messages, "materialize", changed_source)
    store = SessionCompactionStore(messages)
    with pytest.raises(ValueError, match="source changed"):
        store.publish(
            source, content.render(), content=content, request_ids=("generate", "audit")
        )
    with store.database.snapshot() as snapshot:
        assert not snapshot.list("session_compaction", session_id="long")
