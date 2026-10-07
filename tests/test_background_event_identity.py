"""后台事件的运行归属与实例身份合同。

作者：xxx
时间：2026-09-29 22:00:00
"""

from app.background.events import EventBuffer, decode_event
from runtime.stream_events import AssistantTextDelta


def test_equal_sequence_buffers_have_distinct_epochs_and_keep_run_identity():
    """相同游标不代表相同实例，序列化仍可还原事件；传参：无；返回：无。"""
    old, new = EventBuffer(), EventBuffer()
    for buffer in (old, new):
        buffer.emit(AssistantTextDelta("文字"), run_id="run-scope")
    first, second = old.read(0), new.read(0)
    assert first["cursor"] == second["cursor"]
    assert first["event_epoch"] != second["event_epoch"]
    assert first["events"][0]["run_id"] == "run-scope"
    assert decode_event(first["events"][0]) == AssistantTextDelta("文字")


def test_real_background_events_identify_the_executing_run(tmp_path):
    """真实后台执行入口给所有事件绑定所属运行；传参：隔离目录；返回：无。"""
    from scripts.testing.llm import from_test_sequence
    from tests.test_background_sessions import session_for

    session = session_for(tmp_path, from_test_sequence(["执行完成"]))
    try:
        session.submit("开始", input_id="identity-input", model_config={})
        assert session.runtime.wait_idle(10)
        snapshot = session.snapshot(after=0)
        assert snapshot["events"]
        assert {row["run_id"] for row in snapshot["events"]} == {
            snapshot["current_run_id"]
        }
        assert snapshot["event_epoch"] == session.snapshot(history=True)["event_epoch"]
        complete = next(
            row["data"]
            for row in snapshot["events"]
            if row["type"] == "AssistantTurnComplete"
        )
        saved = next(
            row
            for row in session.snapshot(history=True)["history"]
            if row["role"] == "assistant"
        )
        assert complete["message_id"] == saved["message_id"]
        assert complete["entry_id"] == saved["entry_id"]
    finally:
        session.close()


def test_context_failure_event_points_to_the_persisted_error_answer(
    tmp_path, monkeypatch
):
    """上下文读取失败的流式回执指向真实错误正文；传参：隔离根及故障注入；返回：无。"""
    from scripts.testing.llm import from_test_sequence
    from tests.test_background_sessions import session_for
    from runtime.ledger import LedgerStore

    def unavailable(self, task_id):
        """注入真实读取边界故障；传参：记录器及目标；返回：抛出IO错误。"""
        raise OSError("summary store unavailable")

    monkeypatch.setattr(LedgerStore, "read_task_events", unavailable)
    session = session_for(tmp_path, from_test_sequence(["不应调用模型"]))
    try:
        session.submit("开始", input_id="failure-input", model_config={})
        assert session.runtime.wait_idle(10)
        snapshot = session.snapshot(after=0)
        complete = next(
            row["data"]
            for row in snapshot["events"]
            if row["type"] == "AssistantTurnComplete"
        )
        saved = next(
            row
            for row in session.snapshot(history=True)["history"]
            if row["role"] == "assistant"
        )
        assert complete["stop_reason"].startswith("context_error:")
        assert complete["entry_id"] == saved["entry_id"]
        assert complete["content"] == saved["text"]
    finally:
        session.close()


def test_frontend_rehandshakes_on_epoch_change_even_at_equal_cursor():
    """另一界面切换空事件分支时，当前连接仍重新读取完整基线；传参：无；返回：无。"""
    from threading import Event, RLock
    from types import SimpleNamespace
    from app.background.frontend import BackgroundSessionHost

    attached = []
    host = object.__new__(BackgroundSessionHost)
    host._closed = Event()
    host._connection_lock = RLock()
    host.session_id = "s"
    host.config = SimpleNamespace(state=SimpleNamespace(session_id="s"))
    host._cursor = 0
    host._snapshot = {"event_epoch": "old"}
    host.client = SimpleNamespace(
        call=lambda *args, **kwargs: {"event_epoch": "new", "cursor": 0}
    )

    def attach(identity):
        """记录重新握手并结束本次监视；传参：会话；返回：无。"""
        attached.append(identity)
        host._closed.set()

    host._attach = attach
    host._poll()
    assert attached == ["s"]
