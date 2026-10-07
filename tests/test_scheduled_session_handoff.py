"""验证定时会话的直接答复仍由原发生执行者处理。

作者：xxx
时间：2026-09-15 00:20:00
"""

from __future__ import annotations
from scripts.testing.llm import (
    _from_scripted,
    from_test_native_tool_then_final,
    from_test_sequence,
)

import time
from contextlib import closing
from functools import partial
from threading import Event

from app.background.service import BackgroundService
from app.background.sessions import SessionServices
from app.background.scheduled_session import ScheduledSession
from app.scheduled_run import create_scheduler
from llm.messages import ToolCallPart
from runtime.session_message_store import SessionMessageStore
from runtime.checkpoint import load_latest_checkpoint_for_run
from runtime.ledger import LedgerStore
from schedules.occurrences import OccurrenceStore
from tests.test_scheduled_execution import TEST_NOW, register_work
from tests.test_session_runtime import capture_requests
from tools.builtin_tools import build_tool_registry
from tasks.store import TaskStore


def test_scheduled_confirmation_resumes_original_occurrence_with_cron_permissions(
    tmp_path,
):
    """定时成果经真实用户确认后接续，原定时权限与发生不变；传参：隔离目录；返回：无。"""
    from scripts.testing.llm import _ScriptedTurn

    register_work(tmp_path)
    with closing(TaskStore(tmp_path)) as tasks:
        tasks.create_task("核对资料", task_id="goal")
    client = _from_scripted(
        [
            _ScriptedTurn(
                calls=(
                    ToolCallPart(
                        "focus", "goal", {"action": "switch", "goal_ref": "goal"}
                    ),
                )
            ),
            _ScriptedTurn(
                text="核对结果：两份资料的日期与数值一致",
                calls=(
                    ToolCallPart(
                        "ask",
                        "ask_user",
                        {
                            "question": "是否认可核对结果？",
                            "completion": {
                                "goal_id": "goal",
                                "expected_revision": 1,
                                "evidence": [
                                    {
                                        "kind": "answer",
                                        "reference": "current_answer",
                                        "reason": "已交付核对正文",
                                    }
                                ],
                            },
                        },
                    ),
                ),
            ),
            _ScriptedTurn(text="已收到确认"),
        ]
    )
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: client
        )
    ) as scheduler:
        paused = scheduler.run_due_jobs(now=TEST_NOW)[0]
    service = service_for(tmp_path, client)
    view = service.attach(paused.session_id)
    assert isinstance(view, ScheduledSession)
    proposal = view.snapshot(history=True)["completion_requests"][0]
    source = view.confirm_completion(
        question_id=proposal["question_id"],
        action_id="confirm-scheduled",
        accepted=True,
    )
    assert (
        view.confirm_completion(
            question_id=proposal["question_id"],
            action_id="confirm-scheduled",
            accepted=True,
        )
        == source
    )
    service._dispatch_occurrences()
    wait_done(view)
    record = OccurrenceStore(tmp_path).load(paused.occurrence_id)
    assert len(record.run_ids) == 2 and not service._sessions
    checkpoint = load_latest_checkpoint_for_run(record.run_id, data_root=tmp_path)
    assert checkpoint.lease_snapshot["trigger"] == "cron"
    decisions = [
        row
        for row in LedgerStore(tmp_path).read_session_events(paused.session_id)
        if row.event == "goal.confirmation_decided"
    ]
    assert len(decisions) == 1 and decisions[0].payload["source_input_id"] == source


def service_for(root, client):
    """装配生产会话/调度器，模型边界可控；传参：目录与客户端；返回：宿主。"""
    return BackgroundService(
        SessionServices(
            root,
            root,
            lambda _options: client,
            partial(build_tool_registry, repo_root=root, data_root=root),
        )
    )


def wait_done(view):
    """等待原发生的执行和持久交接完成；传参：连接视图；返回：最终状态。"""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = view.snapshot()
        if not status["active"] and status["status"] == "done":
            return status
        time.sleep(0.05)
    raise AssertionError(view.snapshot())


def test_direct_reply_resumes_original_occurrence_without_second_session_owner(
    tmp_path, monkeypatch
):
    """打开定时会话回答问题，只接续原发生且保存一次真实用户输入；传参：目录和捕获器；返回：无。"""
    register_work(tmp_path)
    client = from_test_native_tool_then_final(
        [ToolCallPart("ask", "ask_user", {"question": "选哪份材料？"})], "选择 B 已处理"
    )
    requests = capture_requests(client, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: client
        )
    ) as scheduler:
        paused = scheduler.run_due_jobs(now=TEST_NOW)[0]
    service = service_for(tmp_path, client)
    view = service.attach(paused.session_id)
    assert isinstance(view, ScheduledSession)
    assert view.snapshot(history=True)["questions"] == ["选哪份材料？"]
    view.submit("使用材料 B", input_id="input-direct-answer", model_config={})
    service._dispatch_occurrences()
    result = wait_done(view)
    assert result["session_id"] == paused.session_id and "使用材料 B" in str(
        requests[-1]
    )
    record = OccurrenceStore(tmp_path).load(paused.occurrence_id)
    assert len(record.run_ids) == 2 and not service._sessions
    inputs = [
        row
        for row in SessionMessageStore(tmp_path).read_entries(paused.session_id)
        if row.input_source == "user"
    ]
    assert len(inputs) == 1 and inputs[0].entry_id == "input-direct-answer"


def test_live_scheduled_correction_reaches_same_executor(tmp_path, monkeypatch):
    """模型等待期间的直接纠正进入同一工作下一次请求；传参：目录与捕获器；返回：无。"""
    register_work(tmp_path)
    entered, release = Event(), Event()
    client = from_test_sequence(["旧结论", "按新条件处理"])

    def blocked_request():
        """在真实模型请求边界暂停；传参：无；返回：无。"""
        entered.set()
        assert release.wait(10)

    requests = capture_requests(client, monkeypatch, before_first=blocked_request)
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: client
        )
    ) as scheduler:
        occurrence = scheduler.accept_due(now=TEST_NOW)[0]
    service = service_for(tmp_path, client)
    service._dispatch_occurrences()
    try:
        assert entered.wait(10)
        view = service.attach(occurrence.session_id)
        assert isinstance(view, ScheduledSession)
        view.submit(
            "范围更正：只处理材料 C", input_id="input-correction", model_config={}
        )
    finally:
        release.set()
    wait_done(view)
    assert len(requests) == 2 and "范围更正：只处理材料 C" in str(requests[-1])
    assert len(OccurrenceStore(tmp_path).load(occurrence.occurrence_id).run_ids) == 1


def test_input_at_completion_handoff_is_not_lost_or_repeated(tmp_path, monkeypatch):
    """终态写入前后到达的未交付输入会接续，已交付输入不会再启动；传参：目录与捕获器；返回：无。"""
    register_work(tmp_path)
    client = from_test_sequence(["首轮结束", "已处理补充要求"])
    requests = capture_requests(client, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: client
        )
    ) as scheduler:
        first = scheduler.run_due_jobs(now=TEST_NOW)[0]
        SessionMessageStore(tmp_path).accept_input(
            first.session_id, "补充：再说明来源", input_id="input-late"
        )
        second = scheduler.run_due_jobs(now=TEST_NOW)[0]
        assert second.session_id == first.session_id and second.run_id != first.run_id
        assert scheduler.run_due_jobs(now=TEST_NOW) == []
    assert len(requests) == 2 and "补充：再说明来源" in str(requests[-1])
