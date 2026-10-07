"""从生产调度装配验证模型、实际文件效果和中断交接。

作者：xxx
时间：2026-09-14 19:16:10
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final, from_test_sequence

from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from app.scheduled_run import create_scheduler, scheduled_context
from llm.messages import AssistantMessage, ToolCallPart, model_visible_text
from llm.profiles import ModelProfile, ModelProfilesConfig
from runtime.cancellation import CancellationToken
from runtime.checkpoint import load_latest_checkpoint_for_run
from runtime.cron import CronExecution, CronScheduler
from runtime.default_capabilities import build_local_agent_capabilities
from runtime.session_message_store import SessionMessageStore
from runtime.types import Trigger, new_run_id
from runtime.workspaces import WorkspaceStore
from schedules.notifications import NotificationStore
from schedules.store import ScheduleStore
from tests.test_session_runtime import capture_requests

TEST_NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)


def register_work(
    root, *, kind="work", prompt="将核对结果写入 report.txt", project_root=None
):
    """保存受控时钟和本地授权的计划；传参：隔离根、类型与提示；返回：计划。"""
    project = project_root if project_root is not None else root
    capabilities = build_local_agent_capabilities(project, root)
    capabilities["mcp"] = {"enabled": False, "allow_servers": []}
    with closing(ScheduleStore(root)) as store:
        return store.create_schedule(
            "scheduled-check",
            "at:2026-09-14T19:00:00+08:00",
            kind=kind,
            workspace_id=WorkspaceStore(root).register(project).workspace_id,
            prompt=prompt,
            name="核对材料",
            timezone_name="Asia/Shanghai",
            capabilities=capabilities,
            model_config={"model": "chosen-scheduled-model"},
        )


def test_production_factory_passes_selected_model_and_runs_real_file_tool(
    tmp_path, monkeypatch
):
    """所选模型进入工厂，文件工具和CRON身份走真实链路；传参：隔离目录与替换器；返回：无。"""
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "data"
    register_work(data_root, project_root=project)
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "write-report",
                "file_write",
                {"path": "report.txt", "content": "已核对：318.40元"},
            ),
        ],
        "核对结果已写入 report.txt",
    )
    requests = capture_requests(client, monkeypatch)
    configurations = []

    def make_model(options):
        """记录调度传入的模型选择；传参：公开配置；返回：受控外部模型。"""
        configurations.append(options)
        return client

    with closing(
        create_scheduler(
            project_root=project, data_root=data_root, llm_factory=make_model
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=TEST_NOW)[0]
        assert result.status == "succeeded"
        assert scheduler.run_due_jobs(now=TEST_NOW) == []
    assert (project / "report.txt").read_text(encoding="utf-8") == "已核对：318.40元"
    assert (
        configurations
        == [{"model": "chosen-scheduled-model", "reasoning_effort": "default"}]
        and len(requests) == 2
    )
    checkpoint = load_latest_checkpoint_for_run(result.run_id, data_root=data_root)
    assert checkpoint is not None and checkpoint.lease_snapshot["trigger"] == "cron"
    inbound = [
        entry
        for entry in SessionMessageStore(data_root).read_entries(result.session_id)
        if entry.type == "inbound"
    ]
    assert len(inbound) == 1 and inbound[0].input_source == "agent"
    notice = NotificationStore(data_root).load(result.notification_id)
    assert notice.delivery_status == "pending" and notice.read_at is None


def test_completion_before_ack_recovers_without_repeating_write(tmp_path, monkeypatch):
    """文件写入与通知接纳完成后丢失确认，重启不再调用模型或工具；传参：目录与替换器；返回：无。"""
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "data"
    register_work(data_root, project_root=project)
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "write-once",
                "file_write",
                {"path": "report.txt", "content": "只写一次"},
            ),
        ],
        "已写入，只需交付结果",
    )
    original = NotificationStore.enqueue

    def lose_ack(store, *args, **kwargs):
        """在真实通知提交后模拟进程丢失确认；传参：原接纳调用；返回：无。"""
        original(store, *args, **kwargs)
        raise OSError("notification acknowledgement lost")

    with closing(
        create_scheduler(
            project_root=project, data_root=data_root, llm_factory=lambda _: client
        )
    ) as scheduler:
        with monkeypatch.context() as failure:
            failure.setattr(NotificationStore, "enqueue", lose_ack)
            with pytest.raises(OSError, match="acknowledgement"):
                scheduler.run_due_jobs(now=TEST_NOW)
        occurrence = scheduler.occurrences.list_all()[0]
    before = (project / "report.txt").stat().st_mtime_ns

    def unexpected_model(_options):
        """已经完成的工作不应重建模型；传参：配置；返回：无。"""
        raise AssertionError("completed work must not run again")

    with closing(
        create_scheduler(
            project_root=project, data_root=data_root, llm_factory=unexpected_model
        )
    ) as restarted:
        result = restarted.run_due_jobs(now=TEST_NOW)[0]
        assert result.run_id == occurrence.run_id
        assert len(restarted.occurrences.list_all()) == 1
    assert (project / "report.txt").stat().st_mtime_ns == before
    assert len(NotificationStore(data_root).list_all()) == 1


def test_waiting_for_user_stays_paused_and_resume_keeps_cron_scope(tmp_path):
    """等待用户不能被自动当成完成或改成用户触发授权；传参：隔离目录；返回：无。"""
    job = register_work(tmp_path)
    client = from_test_native_tool_then_final(
        [
            ToolCallPart("question", "ask_user", {"question": "应核对哪张发票？"}),
        ],
        "不应自动执行这段后续",
    )
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: client
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=TEST_NOW)[0]
        assert result.status == "paused" and scheduler.run_due_jobs(now=TEST_NOW) == []
        occurrence = scheduler.occurrences.list_all()[0]
    resumed = replace(occurrence, run_id=new_run_id())
    with closing(ScheduleStore(tmp_path)) as store:
        store.update(
            job.schedule_id,
            capabilities={
                "fs": {"read": [str(tmp_path.parent)], "write": [str(tmp_path.parent)]}
            },
        )
    request = CronExecution(
        job, resumed, TEST_NOW, CancellationToken(), occurrence.run_id
    )
    context = scheduled_context(request, data_root=tmp_path)
    assert (
        context.trigger is Trigger.CRON and context.capability_lease.trigger == "cron"
    )
    assert context.capability_lease.capabilities["fs"] == job.capabilities["fs"]
    assert context.payload["previous_run_id"] == occurrence.run_id


def test_overdue_reminder_is_persisted_without_a_model_or_fake_run(tmp_path):
    """过期提醒补交付，不伪造模型运行或已读回执；传参：隔离目录；返回：无。"""
    register_work(tmp_path, kind="reminder", prompt="核对账单")

    def unexpected_model(_options):
        """单纯提醒无需消耗模型；传参：配置；返回：无。"""
        raise AssertionError("reminders must not call a model")

    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=unexpected_model
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=TEST_NOW)[0]
    assert (
        result.status == "notification_queued"
        and result.run_id is None
        and result.session_id is None
    )
    notification = NotificationStore(tmp_path).load(result.notification_id)
    assert (
        notification.message == "核对账单" and notification.delivery_status == "pending"
    )
    assert notification.source["scheduled_at"].startswith("2026-09-14T11:00:00")


def test_named_model_selection_does_not_follow_frontend_default(tmp_path, monkeypatch):
    """保存的命名模型覆盖界面当前默认值，凭据仍从密钥库解析；传参：目录和替换器；返回：无。"""
    from app import cli

    scheduled = ModelProfile(
        "scheduled",
        "custom",
        "https://scheduled.example/v1",
        "scheduled-model",
        "scheduled-key",
    )
    foreground = ModelProfile(
        "foreground",
        "custom",
        "https://foreground.example/v1",
        "foreground-model",
        "foreground-key",
    )
    profiles = ModelProfilesConfig(
        tmp_path / "models.yaml",
        active="foreground",
        profiles={"scheduled": scheduled, "foreground": foreground},
    )
    vault = Mock()
    vault.get.return_value = "test-only-placeholder"
    monkeypatch.setattr(cli, "load_model_profiles", lambda: profiles)
    monkeypatch.setattr(cli, "load_project_llm_defaults", lambda _: {})
    monkeypatch.setattr(
        cli, "load_saved_config", lambda: {"model": "other-saved-model"}
    )
    monkeypatch.setattr(cli, "SecretsVault", lambda: vault)
    client = cli.build_llm_client({"profile_name": "scheduled"}, project_root=tmp_path)
    assert client.resolved_target.model == "scheduled-model"
    assert client.resolved_target.base_url == "https://scheduled.example/v1"
    assert client.resolved_target.profile_name == "scheduled"
    vault.get.assert_called_once_with("scheduled-key")


def test_failed_runtime_result_keeps_error_in_schedule_and_notification(tmp_path):
    """后台模型失败的原始说明进入状态和通知；传参：隔离目录；返回：无。"""
    register_work(tmp_path)
    client = from_test_sequence(['{"type":"unsupported"}', '{"type":"unsupported"}'])
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: client
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=TEST_NOW)[0]
        occurrence = scheduler.occurrences.list_all()[0]
    assert result.status == "failed"
    outputs = [
        model_visible_text(entry.message)
        for entry in SessionMessageStore(tmp_path).read_entries(result.session_id)
        if entry.run_id == result.run_id and isinstance(entry.message, AssistantMessage)
    ]
    assert result.error == outputs[-1] and "invalid_model_protocol" in result.error
    assert occurrence.error == result.error
    with closing(ScheduleStore(tmp_path)) as store:
        assert store.load_schedule(result.job_id).last_error == result.error
    assert (
        NotificationStore(tmp_path).load(result.notification_id).message
        == f"失败\n{result.error}"
    )


def test_failure_before_receipt_recovers_error_without_repeating_model(
    tmp_path, monkeypatch
):
    """失败已写入会话但尚未交接时，重启补交原因而不重跑模型；传参：目录与替换器；返回：无。"""
    register_work(tmp_path)
    client = from_test_sequence(['{"type":"unsupported"}', '{"type":"unsupported"}'])
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: client
        )
    ) as scheduler:
        with monkeypatch.context() as crash:
            crash.setattr(
                CronScheduler,
                "_complete",
                Mock(side_effect=OSError("handoff interrupted")),
            )
            with pytest.raises(OSError, match="handoff interrupted"):
                scheduler.run_due_jobs(now=TEST_NOW)
        occurrence = scheduler.occurrences.list_all()[0]
        assert occurrence.result_status is None
    unexpected_model = Mock(
        side_effect=AssertionError("failed run must not execute again")
    )
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=unexpected_model
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=TEST_NOW)[0]
        assert result.run_id == occurrence.run_id
    unexpected_model.assert_not_called()
    assert result.status == "failed"
    outputs = [
        model_visible_text(entry.message)
        for entry in SessionMessageStore(tmp_path).read_entries(result.session_id)
        if entry.run_id == result.run_id and isinstance(entry.message, AssistantMessage)
    ]
    assert result.error == outputs[-1] and "invalid_model_protocol" in result.error
    with closing(ScheduleStore(tmp_path)) as store:
        assert store.load_schedule(result.job_id).last_error == result.error
    assert (
        NotificationStore(tmp_path).load(result.notification_id).message
        == f"失败\n{result.error}"
    )
