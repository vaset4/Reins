"""【文件恢复】【请求投影】已恢复文件的旧读取失效，持久历史保持原样。

作者：xxx
时间：2026-09-30 23:00:00
"""

from __future__ import annotations

import hashlib
import json
from contextlib import closing

from llm.messages import (
    AssistantMessage,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    model_visible_text,
)
from runtime.tool_operations import ToolOperationStore
from runtime.tool_result_views import ToolResultViews
from runtime.workspaces import WorkspaceStore
from scripts.testing.llm import from_test_turns
from tasks.store import TaskStore
from tests.test_stage8_restore_service import capture, execute, preview
from tools.tool_registry import ToolRegistry

pytest_plugins = ("tests.test_stage8_restore_service",)


def observed_read(service, path, *, call_id="read-before", session_id="session-test"):
    """登记真实文件字节的读取事实；参数：服务、路径、调用和会话；返回：原消息与操作身份。"""
    raw = path.read_bytes()
    identity = {
        "session_id": session_id,
        "run_id": "read-run",
        "operation_id": "op-" + call_id,
    }
    ToolOperationStore(service.snapshots.data_root).create(
        identity,
        {
            "state": "completed",
            "call": {
                "call_id": call_id,
                "tool_name": "file_read",
                "args": {"path": str(path)},
                "resource": {"known": True, "directory": False},
            },
            "result": {
                "status": "ok",
                "content": raw.decode(),
                "meta": {
                    "resolved_path": str(path),
                    "content_sha256": hashlib.sha256(raw).hexdigest(),
                    "offset": 0,
                    "total_chars": len(raw),
                },
            },
        },
    )
    message = ToolResultMessage(
        "message-" + call_id, call_id, "file_read", (TextPart(raw.decode()),), "success"
    )
    return message, identity


def compose_read(service, message, *, session_id="session-test"):
    """调用正式请求组装，保留完整工具调用组；参数：服务、原消息、会话；返回：实际发送的读取投影。"""
    data = service.snapshots.data_root
    with closing(TaskStore(data)) as tasks:
        task = tasks.create_task("核对恢复后的工作文件")
    client = from_test_turns([])
    views = ToolResultViews(data, task_id=task.task_id, prepare=client.prepare_request)
    context = {
        "session_id": session_id,
        "tool_registry": ToolRegistry(data_root=data),
        "conversation_history": (
            UserMessage("input", (TextPart("继续核对文件"),)),
            AssistantMessage("call", (ToolCallPart(message.call_id, "file_read", {}),)),
            message,
        ),
    }
    composed = views.prepare("继续", context)
    return next(row for row in composed.messages if isinstance(row, ToolResultMessage))


def test_restore_invalidates_previous_read_only_after_real_effect(restore):
    """预览不废弃读取，实际恢复后旧正文改引用且原件不改；参数：隔离环境；返回：无。"""
    service, root, *_ = restore
    path = root / "read.txt"
    path.write_bytes(b"original")
    point = capture(restore, {"read.txt": b"tool-result-version"})
    message, identity = observed_read(service, path)
    original_record = ToolOperationStore(service.snapshots.data_root).load(identity)
    plan = preview(restore, point)
    assert model_visible_text(compose_read(service, message)) == "tool-result-version"
    operation = execute(service, plan)
    projected = compose_read(service, message)
    payload = json.loads(model_visible_text(projected))
    assert payload["file_read_state"] == "historical"
    assert payload["file_restore_operation_id"] == operation["operation_id"]
    assert (
        payload["read_current"]["arguments"]["path"].casefold() == str(path).casefold()
    )
    assert "tool-result-version" not in model_visible_text(projected)
    assert model_visible_text(message) == "tool-result-version"
    assert (
        ToolOperationStore(service.snapshots.data_root).load(identity)
        == original_record
    )


def test_new_read_after_restore_survives_later_operation_status_update(
    restore, monkeypatch
):
    """同秒新读取按提交位置保持有效，后续状态登记不反向废弃；参数：环境与同秒时钟；返回：无。"""
    import runtime.file_restore_records as records
    import runtime.tool_operations as operations

    monkeypatch.setattr(records, "utc_now", lambda: "2026-09-30T12:00:00+00:00")
    monkeypatch.setattr(operations, "utc_now", lambda: "2026-09-30T12:00:00+00:00")
    service, root, *_ = restore
    path = root / "read.txt"
    path.write_bytes(b"original")
    point = capture(restore, {"read.txt": b"tool-result-version"})
    old, _ = observed_read(service, path)
    operation = execute(service, preview(restore, point))
    fresh, _ = observed_read(service, path, call_id="read-after")
    service.records.update(operation["operation_id"], {"error": None})
    assert model_visible_text(compose_read(service, fresh)) == "original"
    assert (
        json.loads(model_visible_text(compose_read(service, old)))["file_read_state"]
        == "historical"
    )


def test_restore_invalidates_reads_in_other_session_of_same_workspace(restore):
    """物理工作区的恢复也废弃另一会话旧读取；参数：隔离环境；返回：无。"""
    service, root, *_ = restore
    WorkspaceStore(service.snapshots.data_root).bind_session("other-session", root)
    path = root / "read.txt"
    path.write_bytes(b"original")
    point = capture(restore, {"read.txt": b"tool-result-version"})
    message, _ = observed_read(service, path, session_id="other-session")
    execute(service, preview(restore, point))
    projected = compose_read(service, message, session_id="other-session")
    assert json.loads(model_visible_text(projected))["file_read_state"] == "historical"


def test_unknown_restore_effect_marks_old_read_uncertain_without_claiming_success(
    restore, monkeypatch
):
    """发布后登记不完整时旧读取不再冒充现状，也不伪称成功；参数：环境与发布边界；返回：无。"""
    from runtime.file_restore_execution import RestoreExecution

    service, root, *_ = restore
    path = root / "read.txt"
    path.write_bytes(b"original")
    point = capture(restore, {"read.txt": b"tool-result-version"})
    message, _ = observed_read(service, path)
    publish = RestoreExecution._publish

    def fail_after_publish(self, *args, **kwargs):
        """保留真正发布效果后模拟未登记；参数：真实发布参数；返回：无。"""
        publish(self, *args, **kwargs)
        raise OSError("receipt unavailable after publication")

    monkeypatch.setattr(RestoreExecution, "_publish", fail_after_publish)
    operation = execute(service, preview(restore, point))
    assert operation["status"] == "needs_reconciliation"
    payload = json.loads(model_visible_text(compose_read(service, message)))
    assert payload["reason"] == "file_restore_outcome_unknown"
    assert payload["file_restore_operation_id"] == operation["operation_id"]
