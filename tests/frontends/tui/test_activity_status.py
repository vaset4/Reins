"""运行活动等待与重连的可见合同；作者：xxx。"""

from app.background.events import EventBuffer
from frontends.tui.projection import ConversationProjection
from runtime.stream_events import (
    ModelRequestStarted,
    ModelRetryScheduled,
    AssistantTextDelta,
    LifecycleChanged,
)


def test_waiting_retry_and_reconnect_preserve_real_activity(monkeypatch):
    """后台空轮询和重连不清除等待，重试确实开始后切换状态；参数：时间替换；返回：无。"""
    clock = [100.0]
    monkeypatch.setattr("app.background.events.time", lambda: clock[0])
    monkeypatch.setattr("frontends.tui.projection.time.time", lambda: clock[0])
    events = EventBuffer()
    events.emit(ModelRequestStarted("request", 1, 3), run_id="run")
    view = ConversationProjection()
    view.apply(
        {
            "session_id": "session",
            "status": "running",
            "history": [],
            **events.read(None),
        }
    )
    clock[0] = 172.0
    view.apply(
        {"session_id": "session", "status": "running", **events.read(view.cursor)}
    )
    assert "等待模型响应" in view.activity_status()
    assert "72s" in view.activity_status()
    events.emit(ModelRetryScheduled(2, 3, 7, "transport_error"), run_id="run")
    clock[0] += 3
    view.apply(
        {"session_id": "session", "status": "running", **events.read(view.cursor)}
    )
    assert "4.0s 后重试" in view.activity_status()
    assert "transport_error" in view.activity_status()
    events.emit(ModelRequestStarted("request", 2, 3), run_id="run")
    view.apply(
        {"session_id": "session", "status": "running", **events.read(view.cursor)}
    )
    assert "尝试 2/3" in view.activity_status()
    events.emit(AssistantTextDelta("回答", "request"), run_id="run")
    view.apply(
        {"session_id": "session", "status": "running", **events.read(view.cursor)}
    )
    assert "正在接收回答" in view.activity_status()
    events.emit(LifecycleChanged("failed", "provider error", "segment"), run_id="run")
    view.apply(
        {"session_id": "session", "status": "failed", **events.read(view.cursor)}
    )
    assert view.activity_status() == "失败 · provider error"


def test_terminal_snapshot_error_wins_over_last_activity():
    """后台异常结束不会遗留正在等待的假状态；参数：无；返回：无。"""
    view = ConversationProjection()
    view.apply(
        {
            "session_id": "session",
            "history": [],
            "active": False,
            "error": "empty_chat_stream",
        }
    )
    assert view.activity_status() == "失败 · empty_chat_stream"
