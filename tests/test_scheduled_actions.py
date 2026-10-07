"""验证模型动作与持久计划、发生和通知的真实接线。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

from contextlib import closing
from datetime import datetime, timezone

from app.run_task import run_task
from app.scheduled_run import create_scheduler
from llm.messages import ToolCallPart
from runtime.cron import CronScheduler
from runtime.workspaces import WorkspaceStore
from schedules.notifications import NotificationStore
from schedules.store import ScheduleStore
from tools.builtin_tools import build_tool_registry
from tests.test_session_runtime import capture_requests
from tests.support.approval import install_approval
from approval import ApprovalDecision


def action_registry(root):
    """显式加载测试要调用的动作，不绕过生产执行器；传参：目录；返回：工具目录。"""
    registry = build_tool_registry(repo_root=root, data_root=root)
    definitions = [
        registry.get(name)
        for name in ("schedule", "notification_send", "notification_status")
    ]
    for definition in definitions:
        assert definition is not None
        definition.deferred = False
    registry.publish(definitions, replace_names=tuple(row.name for row in definitions))
    return registry


def test_native_schedule_and_notification_preserve_operation_sources(
    tmp_path, monkeypatch
):
    """模型创建提醒和通知都有实际操作归属，接纳不冒充已发送；传参：隔离目录/捕获器；返回：无。"""
    approved = []

    def approve(request):
        """显式授权用户发起的定时控制动作；传参：审批；返回：单次允许。"""
        approved.append(request.tool)
        return ApprovalDecision.ONCE

    install_approval(monkeypatch, approve)
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "remind",
                "schedule",
                {
                    "action": "create",
                    "name": "取文件",
                    "prompt": "去取打印文件",
                    "time": "at:2030-01-01T10:00:00+08:00",
                    "timezone": "Asia/Shanghai",
                    "kind": "reminder",
                },
            ),
            ToolCallPart(
                "notify",
                "notification_send",
                {"title": "已接纳", "message": "提醒计划已保存"},
            ),
        ],
        "已保存提醒",
    )
    requests = capture_requests(client, monkeypatch)
    response = run_task(
        "保存提醒",
        tmp_path,
        data_root=tmp_path,
        llm_client=client,
        tool_registry=action_registry(tmp_path),
    )
    assert response.status == "done"
    with closing(ScheduleStore(tmp_path)) as store:
        records = [
            record for record in store.list_all_schedules() if record.kind == "reminder"
        ]
    assert len(records) == 1 and records[0].source_run_id == response.run_id
    assert records[0].kind == "reminder" and records[0].timezone_name == "Asia/Shanghai"
    notices = NotificationStore(tmp_path).list_all()
    assert len(notices) == 1 and notices[0].source["operation_id"]
    assert notices[0].delivery_status == "pending" and notices[0].read_at is None
    assert "accepted" in str(requests[-1])
    assert approved == ["schedule"]


def test_editing_future_schedule_does_not_change_accepted_occurrence(tmp_path):
    """接纳后的提醒使用原内容，后续修改只影响未来；传参：隔离目录；返回：无。"""
    now = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
    with closing(ScheduleStore(tmp_path)) as store:
        store.create_schedule(
            "original",
            "interval:60",
            next_run_at=now.isoformat(),
            prompt="原提醒",
            workspace_id=WorkspaceStore(tmp_path).register(tmp_path).workspace_id,
            kind="reminder",
            timezone_name="Asia/Shanghai",
        )
    with closing(CronScheduler(project_root=tmp_path, data_root=tmp_path)) as scheduler:
        occurrence = scheduler.accept_due(now=now)[0]
        with closing(ScheduleStore(tmp_path)) as store:
            store.update("original", prompt="新提醒")
        result = scheduler.run_occurrence(occurrence.occurrence_id, now=now)
    assert NotificationStore(tmp_path).load(result.notification_id).message == "原提醒"


def test_model_can_resume_waiting_occurrence_with_sourced_reply(tmp_path, monkeypatch):
    """等待问题可由显式接续输入恢复，仍使用 CRON 权限；传参：隔离目录与捕获器；返回：无。"""
    from tests.test_scheduled_execution import TEST_NOW, register_work

    install_approval(monkeypatch, lambda _: ApprovalDecision.ONCE)

    register_work(tmp_path)
    worker = from_test_native_tool_then_final(
        [ToolCallPart("ask", "ask_user", {"question": "使用哪个版本？"})],
        "已使用版本 B",
    )
    requests = capture_requests(worker, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: worker
        )
    ) as scheduler:
        paused = scheduler.run_due_jobs(now=TEST_NOW)[0]
        assert paused.status == "paused"
    coordinator = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "resume",
                "schedule",
                {
                    "action": "resume_occurrence",
                    "occurrence_id": paused.occurrence_id,
                    "message": "用户选择版本 B，继续核对",
                },
            ),
        ],
        "已交给原工作接续",
    )
    run_task(
        "让刚才的工作使用版本 B",
        tmp_path,
        data_root=tmp_path,
        llm_client=coordinator,
        tool_registry=action_registry(tmp_path),
    )
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: worker
        )
    ) as scheduler:
        resumed = scheduler.run_due_jobs(now=TEST_NOW)[0]
        occurrence = scheduler.occurrences.load(paused.occurrence_id)
        assert (
            resumed.status == "succeeded" and scheduler.run_due_jobs(now=TEST_NOW) == []
        )
    assert resumed.session_id == paused.session_id and resumed.run_id != paused.run_id
    assert len(occurrence.run_ids) == 2 and occurrence.budget_run_id == resumed.run_id
    assert "用户选择版本 B" in str(requests[-1])
