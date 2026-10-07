"""从后台组合根验证会话接纳、等待、恢复和可断开显示。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final, from_test_sequence

from dataclasses import replace
from functools import partial
from threading import Event

import approval
import pytest

from app.background.sessions import (
    BackgroundSession,
    SessionRecord,
    SessionServices,
    load_session_records,
)
from llm.messages import ToolCallPart
from runtime.session_message_store import SessionMessageStore
from schedules.notifications import NotificationStore
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
from tests.test_session_runtime import capture_requests


def session_for(root, client, *, registry=None):
    """装配真实后台会话和可控模型边界；传参：隔离目录及依赖；返回：后台会话。"""
    factory = (
        (lambda: registry)
        if registry is not None
        else partial(build_tool_registry, repo_root=root, data_root=root)
    )
    services = SessionServices(root, root, lambda _options: client, factory)
    return BackgroundSession(SessionRecord("session-background"), services)


def test_background_acceptance_and_reconnect_do_not_repeat_body_or_write(
    tmp_path, monkeypatch
):
    """确认丢失重投和重连不会再次写文件；传参：隔离目录与请求捕获；返回：无。"""
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "file", "file_write", {"path": "result.txt", "content": "后台实际完成"}
            ),
        ],
        "文件已写入",
    )
    requests = capture_requests(client, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "data"
    services = SessionServices(
        project,
        data_root,
        lambda _options: client,
        partial(build_tool_registry, repo_root=project, data_root=data_root),
    )
    session = BackgroundSession(SessionRecord("session-background"), services)
    session.submit("写入结果", input_id="input-once", model_config={})
    assert session.runtime.wait_idle(10)
    before = (project / "result.txt").stat().st_mtime_ns
    session.submit("写入结果", input_id="input-once", model_config={})
    saved = load_session_records(data_root)[0]
    restarted = BackgroundSession(saved, session.services)
    restarted.recover()
    assert restarted.runtime.wait_idle(2)
    assert (project / "result.txt").stat().st_mtime_ns == before
    assert len(requests) == 2 and saved.status == "done"
    assert (
        len(
            [
                row
                for row in SessionMessageStore(data_root).read_entries(saved.session_id)
                if row.type == "inbound"
            ]
        )
        == 1
    )
    assert NotificationStore(data_root).list_all()[0].delivery_status == "pending"


def test_background_input_and_frozen_model_fail_as_one_commit(tmp_path, monkeypatch):
    """冻结配置失败时输入也未接纳，不唤醒模型；传参：隔离根与故障器；返回：无。"""
    client = from_test_sequence(["不应调用"])
    requests = capture_requests(client, monkeypatch)
    session = session_for(tmp_path, client, registry=ToolRegistry())

    def fail_model_snapshot(*_args, **_kwargs):
        """模拟正文准备后保存冻结配置失败；传参：配置写入；返回：不返回。"""
        raise OSError("model snapshot unavailable")

    monkeypatch.setattr(session.records, "save_input", fail_model_snapshot)
    with pytest.raises(OSError, match="model snapshot unavailable"):
        session.submit("保存这条输入", input_id="unaccepted", model_config={})
    assert not session.runtime.active and requests == []
    assert all(
        row.entry_id != "unaccepted"
        for row in session.messages.read_entries(session.record.session_id)
    )
    assert session.records.load(session.record.session_id) is None


def test_completed_run_lost_host_ack_is_observed_before_resume(tmp_path, monkeypatch):
    """运行终态已落盘而宿主确认丢失时只补交付；传参：隔离目录与捕获器；返回：无。"""
    client = from_test_sequence(["保存完成"])
    requests = capture_requests(client, monkeypatch)
    session = session_for(tmp_path, client, registry=ToolRegistry())
    session.submit("整理材料", input_id="input-summary", model_config={})
    assert session.runtime.wait_idle(10)
    restarted = BackgroundSession(
        replace(session.record, status="running"), session.services
    )
    restarted.recover()
    assert restarted.record.status == "done" and not restarted.runtime.active
    assert len(requests) == 1
    assert len(NotificationStore(tmp_path).list_all()) == 1


def test_waiting_question_does_not_run_until_new_user_input(tmp_path, monkeypatch):
    """等待答复在重连后保持等待，新输入继续原问题；传参：隔离目录与捕获器；返回：无。"""
    client = from_test_native_tool_then_final(
        [
            ToolCallPart("question", "ask_user", {"question": "使用哪份材料？"}),
        ],
        "已使用用户选择的材料",
    )
    requests = capture_requests(client, monkeypatch)
    session = session_for(tmp_path, client)
    session.submit("分析材料", input_id="input-question", model_config={})
    assert session.runtime.wait_idle(10) and session.record.status == "paused"
    assert session.snapshot(history=True)["questions"] == ["使用哪份材料？"]
    assert NotificationStore(tmp_path).list_all()[0].message == "使用哪份材料？"
    restarted = BackgroundSession(load_session_records(tmp_path)[0], session.services)
    restarted.recover()
    assert not restarted.runtime.active and len(requests) == 1
    restarted.submit("使用本地材料 A", input_id="input-answer", model_config={})
    assert restarted.runtime.wait_idle(10)
    assert "使用本地材料 A" in str(requests[-1]) and len(requests) == 2


def test_live_approval_remains_available_after_display_reconnect(tmp_path):
    """显示连接断开不取消待审批，原编号仍能决定一次真实动作；传参：隔离目录；返回：无。"""
    reached, effects = Event(), []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "confirm_action",
            "动作",
            {},
            "agent",
            ToolRisk.CONFIRM,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            executor=lambda _: effects.append(True) or "已执行",
        )
    )
    client = from_test_native_tool_then_final(
        [ToolCallPart("approval", "confirm_action", {})], "已处理"
    )
    session = session_for(tmp_path, client, registry=registry)
    original = session._present_approval

    def present(identity, request):
        """以真实审批等待建立时序；传参：编号和请求；返回：无。"""
        original(identity, request)
        reached.set()

    session.approvals._present = present
    approval.register_approval_backend(session.approve)
    try:
        session.submit("执行动作", input_id="input-approve", model_config={})
        assert reached.wait(10)
        first, reconnected = session.snapshot(), session.snapshot(history=True)
        assert (
            first["approval"]["identity"] == reconnected["approval"]["identity"]
            and effects == []
        )
        session.approvals.answer(f"/approve {first['approval']['identity']} once")
        assert session.runtime.wait_idle(10) and effects == [True]
    finally:
        session.cancel()
        approval.register_approval_backend(None)


def test_explicit_stop_is_not_automatically_resumed(tmp_path, monkeypatch):
    """停止意图持久保存，后台重启不重新执行已接纳输入；传参：目录与捕获器；返回：无。"""
    client = from_test_sequence(["不会发出请求"])
    requests = capture_requests(client, monkeypatch)
    session = session_for(tmp_path, client, registry=ToolRegistry())
    session.messages.accept_input(
        session.record.session_id, "保留待办", input_id="queued"
    )
    session.cancel()
    restarted = BackgroundSession(load_session_records(tmp_path)[0], session.services)
    restarted.recover()
    assert not restarted.runtime.active and requests == []


def test_background_rejects_credentials_in_public_metadata(tmp_path):
    """公开配置不能成为新的密钥存储；传参：隔离目录；返回：无。"""
    session = session_for(tmp_path, from_test_sequence(["unused"]))
    with pytest.raises(ValueError, match="public fields"):
        session.submit(
            "输入", input_id="credential", model_config={"api_key": "do-not-persist"}
        )
    assert session.records.load(session.record.session_id) is None


def test_notification_handoff_failure_recovers_without_rerunning_model(
    tmp_path, monkeypatch
):
    """运行完成而通知未接纳时重启只补通知，不丢交付或重做模型；传参：目录与故障注入；返回：无。"""
    client = from_test_sequence(["已整理完毕"])
    requests = capture_requests(client, monkeypatch)
    session = session_for(tmp_path, client, registry=ToolRegistry())
    enqueue = NotificationStore.enqueue

    def fail(*_args, **_kwargs):
        """模拟终态提交后 outbox 写入失败；传参：通知；返回：无。"""
        raise OSError("outbox interrupted")

    monkeypatch.setattr(NotificationStore, "enqueue", fail)
    session.submit("整理", input_id="input-notice", model_config={})
    with pytest.raises(RuntimeError, match="remain recoverable"):
        session.runtime.wait_idle(10)
    assert session.record.status == "done" and session.record.pending_notice
    monkeypatch.setattr(NotificationStore, "enqueue", enqueue)
    restarted = BackgroundSession(load_session_records(tmp_path)[0], session.services)
    restarted.recover()
    assert (
        len(requests) == 1
        and not restarted.runtime.active
        and restarted.record.pending_notice is None
    )
    notices = NotificationStore(tmp_path).list_all()
    assert len(notices) == 1 and notices[0].message == "已整理完毕"


def test_reconnecting_mid_response_displays_the_complete_answer(tmp_path, monkeypatch):
    """接回流式答复中途时显示完整正文，不只显示余下片段；传参：目录与连接替换器；返回：无。"""
    import time
    from rich.console import Console
    from app.background.frontend import BackgroundSessionHost
    from app.repl.console import reset_console_for_tests
    from app.repl.session_host import SessionHostConfig
    from app.repl.slash_commands import ReplState

    snapshot = {
        "session_id": "session-display",
        "project_root": str(tmp_path),
        "current_task_id": None,
        "compatibility_task_id": None,
        "current_run_id": "run-display",
        "status": "running",
        "active": True,
        "error": None,
        "approval": None,
        "cursor": 10,
        "gap": False,
        "history": [],
        "events": [],
    }

    class Connection:
        """重现打开窗口时模型已经开始发送的网络边界。"""

        def call(self, method, **_params):
            """首次只返回流的后半段及完整轮末；传参：RPC 请求；返回：边界事件。"""
            if method == "status":
                return {"unread_notifications": 0, "errors": {}}
            if method == "poll" and snapshot["active"]:
                snapshot.update(active=False, status="done", cursor=12)
                return {
                    **snapshot,
                    "events": [
                        {"type": "AssistantTextDelta", "data": {"text": "answer tail"}},
                        {
                            "type": "AssistantTurnComplete",
                            "data": {"content": "Full answer head and answer tail"},
                        },
                    ],
                }
            return snapshot

    monkeypatch.setattr(BackgroundSessionHost, "_connect", lambda _: Connection())
    console = Console(record=True, force_terminal=False)
    reset_console_for_tests(console)
    host = BackgroundSessionHost(
        SessionHostConfig(
            ReplState(),
            ToolRegistry(),
            from_test_sequence(["unused"]),
            tmp_path,
            tmp_path,
        )
    )
    try:
        deadline = time.monotonic() + 3
        while (
            "Full answer" not in console.export_text(clear=False)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        shown = console.export_text(clear=False)
        assert (
            "Full answer head and answer tail" in shown
            and shown.count("answer tail") == 1
        )
    finally:
        host.close()
        reset_console_for_tests(None)
