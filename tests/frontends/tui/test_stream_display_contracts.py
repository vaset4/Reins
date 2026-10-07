"""当前全屏入口的流显示和真实后台桥接回归，作者：xxx。"""

from contextlib import contextmanager

from app.background.events import EventBuffer
from frontends.tui.projection import ConversationProjection
from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurnComplete,
    LifecycleChanged,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)


@contextmanager
def connected_tui_bridge(
    project_root, data_root, client, registry, monkeypatch, *, session_id, task_id
):
    """只替换网络传输，接通正式界面桥与后台；参数：目录、模型、工具和会话；返回：桥、后台会话、界面通知。"""
    from app.background.frontend import BackgroundSessionHost
    from app.background.server import BackgroundServer
    from app.background.service import BackgroundService
    from app.background.sessions import SessionServices
    from app.repl.slash_commands import ReplState
    from frontends.tui.bridge import TuiBridge

    service = BackgroundService(
        SessionServices(project_root, data_root, lambda _: client, lambda: registry)
    )
    service.workspaces.bind_session(session_id, project_root)
    session = service.attach(session_id)
    server = BackgroundServer(service, "test-tui-control-token")
    notices = []

    class Connection:
        """在进程内调用同一生产 RPC 分派器。"""

        def call(self, method, **params):
            """转发请求到真实服务；参数：方法及参数；返回：服务回执。"""
            return server.dispatch(method, params)

    monkeypatch.setattr(BackgroundSessionHost, "_connect", lambda _: Connection())
    bridge = TuiBridge(
        project_root=project_root,
        data_root=data_root,
        llm_client=client,
        registry=registry,
        event_sink=lambda kind, data: notices.append((kind, data)),
    )
    bridge.state = ReplState(session_id=session_id, current_task_id=task_id)
    try:
        bridge.start()
        yield bridge, session, notices
    finally:
        bridge.close()
        session.close()
        server.server_close()


def test_stream_merges_reasoning_and_answer_without_mixing_lifecycle():
    """正文与思考分别合并，生命周期状态不混入回答；参数：无；返回：无。"""
    projection, events = ConversationProjection(), EventBuffer()
    projection.apply({"session_id": "session", "history": [], **events.read(None)})
    events.emit(
        LifecycleChanged(
            lifecycle="running", reason="", checkpoint_id=None, segment_id="segment"
        )
    )
    events.emit(AssistantReasoningDelta(text="先看清", message_id="answer"))
    events.emit(AssistantReasoningDelta(text="用户要什么", message_id="answer"))
    events.emit(AssistantTextDelta(text="hello ", message_id="answer"))
    events.emit(AssistantTextDelta(text="world", message_id="answer"))
    projection.apply({"session_id": "session", **events.read(projection.cursor)})
    cards = list(projection.cards.values())
    assert len(cards) == 1
    assert cards[0].reasoning == "先看清用户要什么"
    assert cards[0].text == "hello world"
    events.emit(
        AssistantTurnComplete(
            content="hello world",
            message_id="answer",
            entry_id="saved-answer",
            usage={"input_tokens": 1, "output_tokens": 2},
            stop_reason="end_turn",
        )
    )
    projection.apply({"session_id": "session", **events.read(projection.cursor)})
    assert list(projection.cards) == ["saved-answer"]
    assert projection.cards["saved-answer"].text == "hello world"
    assert projection.usage == {"input_tokens": 1, "output_tokens": 2}


def test_tool_card_preserves_start_arguments_and_completed_result():
    """同一工具卡由真实开始事件更新为成功结果；参数：无；返回：无。"""
    projection, events = ConversationProjection(), EventBuffer()
    projection.apply({"session_id": "session", "history": [], **events.read(None)})
    events.emit(
        ToolExecutionStarted(
            tool_name="list", args={"path": "tools"}, call_id="call", risk="safe"
        ),
        run_id="run",
    )
    projection.apply({"session_id": "session", **events.read(projection.cursor)})
    pending = list(projection.cards.values())
    assert len(pending) == 1 and pending[0].state == "执行中"
    assert pending[0].title == "list" and '"tools"' in pending[0].args
    events.emit(
        ToolExecutionCompleted(tool_name="list", output="tools/", call_id="call"),
        run_id="run",
    )
    projection.apply({"session_id": "session", **events.read(projection.cursor)})
    completed = list(projection.cards.values())
    assert len(completed) == 1 and completed[0].key == pending[0].key
    assert completed[0].text == "tools/" and completed[0].state == "完成"
