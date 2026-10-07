"""活动读取与控制复用生产宿主，验证真实状态和会话隔离。

作者：xxx
时间：2026-09-30 12:00:00
"""

from contextlib import closing
from functools import partial
from threading import Event, RLock
from types import SimpleNamespace

import pytest

from app.background.frontend import BackgroundSessionHost
from app.background.server import BackgroundServer
from app.background.service import BackgroundService
from app.background.sessions import SessionServices
from app.repl.dashboard import _read_todo
from runtime.collaboration_store import CollaborationStore
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.workspaces import WorkspaceStore
from scripts.testing.llm import from_test_sequence, from_test_stub, _scripted_events
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.todo_tool import add_todo, update_todo


def make_service(root, client):
    """只注入可控模型，其他边界使用实际实现；参数：数据根与模型；返回：宿主。"""
    return BackgroundService(
        SessionServices(
            root,
            root,
            lambda _: client,
            partial(build_tool_registry, repo_root=root, data_root=root),
        )
    )


def attach_session(service, identity):
    """为活动场景先发布真实归属再连接；参数：宿主及会话编号；返回：后台会话。"""
    WorkspaceStore(service.services.data_root).bind_session(
        identity, service.services.project_root
    )
    return service.attach(identity)


def test_activity_reads_current_plan_and_scoped_child_outcomes(tmp_path):
    """计划读同库，旧运行及其他会话子代理不混入当前活动；参数：隔离根；返回：无。"""
    service = make_service(tmp_path, from_test_sequence(["已核对当前资料"]))
    with closing(TaskStore(tmp_path)) as tasks:
        tasks.create_task("核对当前资料", task_id="goal")
    item = add_todo("goal", "读取原始凭据", data_root=tmp_path)
    update_todo("goal", item.idx, "done", data_root=tmp_path)
    session = attach_session(service, "session-activity")
    server = BackgroundServer(service, "fixture-token")
    try:
        session.submit(
            "请核对", task_id="goal", input_id="input-activity", model_config={}
        )
        assert session.runtime.wait_idle(10)
        current = session.snapshot()["current_run_id"]
        messages = SessionMessageStore(tmp_path)
        group = CollaborationStore(messages, "session-activity")
        member = {
            "agent_id": "agent-current",
            "name": "核对者",
            "session_id": "session-child",
            "run_id": "run-child",
            "budget_run_id": current,
            "task": "核对日期",
            "backend": "internal",
        }
        group.add_member(member)
        group.add_member(
            {
                **member,
                "agent_id": "agent-old",
                "name": "旧运行成员",
                "budget_run_id": "run-old",
            }
        )
        CollaborationStore(messages, "session-other").add_member(
            {**member, "name": "其他会话成员"}
        )
        facts = RunFactStore(tmp_path)
        facts.append(
            {
                "event": "agent:finished",
                "session_id": "session-child",
                "run_id": "run-child",
                "status": "failed",
                "output": "缺少原始凭据",
            }
        )
        view = server.dispatch("activity", {"session_id": "session-activity"})
        assert view["task"]["goal"] == "核对当前资料"
        assert view["task"]["todos"] == [
            {"idx": item.idx, "content": "读取原始凭据", "status": "done"}
        ]
        assert [row["name"] for row in view["members"]] == ["核对者"]
        assert view["members"][0]["status"] == "failed"
        assert view["members"][0]["output"] == "缺少原始凭据"
        assert not view["current"]["active"]
        assert "[done] 读取原始凭据" in _read_todo(tmp_path, "goal").plain
    finally:
        server.server_close()
        service.close()


def test_activity_cancel_rejects_previous_run_and_does_not_stop_other_session(tmp_path):
    """旧浮层不能停止新运行，正确选择只作用于所属会话；参数：隔离根；返回：无。"""
    service = make_service(tmp_path, from_test_sequence(["完成甲", "完成乙"]))
    selected, other = (
        attach_session(service, "session-selected"),
        attach_session(service, "session-other"),
    )
    server = BackgroundServer(service, "fixture-token")
    try:
        selected.submit("甲", input_id="input-one", model_config={})
        assert selected.runtime.wait_idle(10)
        stale = selected.snapshot()["current_run_id"]
        selected.submit("乙", input_id="input-two", model_config={})
        assert selected.runtime.wait_idle(10)
        current = selected.snapshot()["current_run_id"]
        assert current != stale
        with pytest.raises(ValueError, match="运行已变化"):
            server.dispatch(
                "cancel", {"session_id": "session-selected", "expected_run_id": stale}
            )
        assert not selected.record.stopped and not other.record.stopped
        server.dispatch(
            "cancel", {"session_id": "session-selected", "expected_run_id": current}
        )
        assert selected.record.stopped and not other.record.stopped
    finally:
        server.server_close()
        service.close()


def test_frontend_activity_continues_selected_session_using_real_submit(tmp_path):
    """接续沿同一后台接纳链，旧会话选择不能发给新会话；参数：隔离根；返回：无。"""
    service = make_service(tmp_path, from_test_sequence(["处理完成"]))
    server = BackgroundServer(service, "fixture-token")
    session = attach_session(service, "session-control")
    host = object.__new__(BackgroundSessionHost)
    host._connection_lock = RLock()
    host.session_id = "session-control"
    host.config = SimpleNamespace(
        state=SimpleNamespace(session_id=host.session_id, current_task_id=None),
        llm_client=object(),
    )
    host.client = SimpleNamespace(
        call=lambda method, **params: server.dispatch(method, params)
    )
    host._completion_menu = SimpleNamespace(choose=lambda _: None)
    host._input_error = None
    host._sync = lambda _: None
    host._present = lambda _: None
    choice = {
        "action": "submit",
        "session_id": "session-control",
        "run_id": "",
        "text": "继续并核对凭据",
    }
    try:
        result = host.control_activity(choice)
        assert session.runtime.wait_idle(10)
        entry = next(
            row
            for row in session.messages.read_entries("session-control")
            if row.entry_id == result["input_id"]
        )
        assert entry.input_source == "user"
        assert entry.message.content[0].text == "继续并核对凭据"
        host.session_id = "session-new-selection"
        with pytest.raises(ValueError, match="所选会话已变化"):
            host.control_activity(choice)
        assert (
            len(
                session.messages.pending_inputs(
                    "session-control", include_delivered=True
                )
            )
            == 1
        )
    finally:
        server.server_close()
        service.close()


def test_activity_cancel_reaches_actual_delegated_model_execution(
    tmp_path, monkeypatch
):
    """活动取消通过真实父循环抵达子代理请求，兄弟会话不受影响；参数：隔离根与替换器；返回：无。"""
    from approval import ApprovalDecision
    from llm.messages import UserMessage
    from runtime.cancellation import ExecutionCancelled
    from schedules.notifications import NotificationStore
    from runtime.session_state import SessionState, SessionStateStore
    from tests.test_collaboration_runtime import successful_results, tool

    child_started, child_cancelled = Event(), Event()
    client = from_test_stub("unused")

    def stream(request, *, model, connection, cancellation=None, prepared_body=None):
        """模型主动委派，子供应商请求等待真实取消；参数：请求及供应商依赖；返回：事件。"""
        first = next(item for item in request.messages if isinstance(item, UserMessage))
        if first.message_id != "input-parent":
            child_started.set()
            assert cancellation is not None and cancellation.wait(10)
            child_cancelled.set()
            raise ExecutionCancelled("子请求实际收到取消")
        spawned = any(
            item.tool_name == "delegate" for item in successful_results(request)
        )
        turn = (
            tool("agent_wait", targets=["核对者"], timeout_seconds=3)
            if spawned
            else tool("delegate", name="核对者", task="核对资料")
        )
        return iter(_scripted_events(turn, model))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    monkeypatch.setattr("approval._backend", lambda _: ApprovalDecision.ONCE)
    service = make_service(tmp_path, client)
    server = BackgroundServer(service, "fixture-token")
    session, other = (
        attach_session(service, "session-parent"),
        attach_session(service, "session-unrelated"),
    )
    SessionStateStore(tmp_path).save(
        SessionState(session_id="session-parent", toolsets_enabled=["full"])
    )
    try:
        session.submit("自主组织核对", input_id="input-parent", model_config={})
        assert child_started.wait(10), session.snapshot()
        view = server.dispatch("activity", {"session_id": "session-parent"})
        assert view["members"] and view["members"][0]["status"] == "unknown"
        server.dispatch(
            "cancel",
            {
                "session_id": "session-parent",
                "expected_run_id": view["current"]["run_id"],
            },
        )
        assert child_cancelled.wait(5)
        assert session.runtime.wait_idle(10)
        assert session.record.stopped and not other.record.stopped
        notice = NotificationStore(tmp_path).load(
            f"background-{view['current']['run_id']}"
        )
        assert (
            notice.message == "已请求停止"
            and notice.source["run_id"] == view["current"]["run_id"]
        )
    finally:
        session.cancel()
        server.server_close()
        service.close()
