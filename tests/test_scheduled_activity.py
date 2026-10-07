"""定时活动的执行身份、事件和设置必须来自实际发生。

作者：xxx
时间：2026-09-30 12:00:00
"""

from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.background.server import BackgroundServer
from app.scheduled_run import create_scheduler
from runtime.stream_events import AssistantTextDelta
from schedules.occurrences import OccurrenceStore
from tests.test_scheduled_execution import TEST_NOW, register_work
from tests.test_scheduled_session_handoff import service_for


def test_scheduled_events_keep_original_run_after_resume(tmp_path, monkeypatch):
    """接续创建新运行后，迟到回调仍归原运行；参数：隔离根与替换器；返回：无。"""
    register_work(tmp_path)
    callbacks = []

    def execute(context, **options):
        """保留真实装配交给执行器的回调；参数：运行与选项；返回：暂停边界。"""
        callbacks.append((context.run_id, options["event_sink"]))
        options["event_sink"](AssistantTextDelta("执行中", f"message-{context.run_id}"))
        return SimpleNamespace(
            status="paused", output="需要用户继续", task_id=context.material_task_id
        )

    monkeypatch.setattr("app.scheduled_run.execute_context", execute)
    service = service_for(tmp_path, object())
    with closing(
        create_scheduler(project_root=tmp_path, data_root=tmp_path)
    ) as scheduler:
        occurrence = scheduler.accept_due(now=TEST_NOW)[0]
    view = service.attach(occurrence.session_id)
    with closing(service._scheduler(session=view)) as scheduler:
        scheduler.run_occurrence(occurrence.occurrence_id, now=TEST_NOW)
        view.submit("继续核对", input_id="input-continue", model_config={})
        scheduler.run_occurrence(occurrence.occurrence_id, now=TEST_NOW)
    assert len(callbacks) == 2 and callbacks[0][0] != callbacks[1][0]
    callbacks[0][1](AssistantTextDelta("迟到内容", "message-late"))
    events = view.snapshot(after=0)["events"]
    assert [event["run_id"] for event in events] == [
        callbacks[0][0],
        callbacks[1][0],
        callbacks[0][0],
    ]
    assert view.snapshot()["current_run_id"] == callbacks[1][0]
    before_cancel = view.snapshot()
    with pytest.raises(ValueError, match="定时运行已变化"):
        view.cancel(expected_run_id=callbacks[0][0])
    assert view.snapshot() == before_cancel


def test_scheduled_snapshot_exposes_pause_and_unknown_execution_settings(tmp_path):
    """明确停止保留计划暂停状态，配置不能冒充执行连接；参数：隔离根；返回：无。"""
    register_work(tmp_path)
    service = service_for(tmp_path, object())
    with closing(
        create_scheduler(project_root=tmp_path, data_root=tmp_path)
    ) as scheduler:
        occurrence = scheduler.accept_due(now=TEST_NOW)[0]
    store = OccurrenceStore(tmp_path)
    store.save(replace(occurrence, status="settled", result_status="paused"))
    view = service.attach(occurrence.session_id)
    view.cancel()
    assert view.snapshot()["stopped"] is True
    server = BackgroundServer(service, "fixture-token")
    try:
        settings = server.dispatch("settings", {"session_id": occurrence.session_id})
        assert settings["execution_attached"] is False
        assert settings["model"] is None and settings["mcp"] is None
    finally:
        server.server_close()
