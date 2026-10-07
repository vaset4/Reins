"""验证知识提交与交付回执分离，索引失败和宿主恢复保留真实版本。

作者：xxx
时间：2026-10-01 15:20:00
"""

from contextlib import closing
from datetime import datetime, timezone
import sqlite3

import pytest

from app.scheduled_run import create_scheduler
from app.run_task import run_task
from llm.messages import ToolCallPart
from memory.store import MemoryStore
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.workspaces import WorkspaceStore
from schedules.notifications import NotificationStore
from scripts.testing.llm import from_test_native_tool_then_final, from_test_stub
from tests.test_automatic_knowledge import start_source


def saving_worker():
    """提供真实工具调用的本地模型响应，不发生网络调用；参数：无；返回：客户端。"""
    return from_test_native_tool_then_final(
        [
            ToolCallPart("read", "knowledge_read", {"action": "messages"}),
            ToolCallPart(
                "save",
                "memory_manage",
                {
                    "action": "create",
                    "type": "rule",
                    "kind": "fact",
                    "memory_scope": "project",
                    "subject": "金额",
                    "fact_key": "小数位",
                    "content": "本项目金额保留两位小数",
                    "source_mode": "origin_inputs",
                },
            ),
            ToolCallPart(
                "finish",
                "knowledge_finish",
                {"outcome": "completed", "reason": "原用户证据已核对并写入"},
            ),
        ],
        "完成",
    )


def test_committed_index_failure_is_recorded_without_republishing(
    tmp_path, monkeypatch
):
    """索引首次写入失败保留已提交原件，修复不发布第二版本；参数：隔离根/注入器；返回：无。"""
    _, work = start_source(tmp_path)
    import memory.store as memory_module

    original = memory_module.upsert_memory_index
    failures = []

    def fail_once(*args, **kwargs):
        """模拟一次派生索引故障；参数：原索引写入；返回：首个失败后恢复真实执行。"""
        if not failures:
            failures.append(True)
            raise sqlite3.OperationalError("injected knowledge index failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(memory_module, "upsert_memory_index", fail_once)
    with closing(
        create_scheduler(
            project_root=tmp_path,
            data_root=tmp_path,
            llm_factory=lambda _: saving_worker(),
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=datetime.now(timezone.utc))[0]
    assert result.status == "succeeded"
    row = KnowledgeMaintenance(tmp_path).load(work["work_id"])
    assert row["commits"][0]["index_state"] == "stale"
    with closing(MemoryStore(tmp_path)) as store:
        records = store.list_memories()
        assert len(records) == 1 and records[0].revision == 1
        assert records[0].version == row["commits"][0]["version"]


def test_completed_maintenance_delivery_recovery_does_not_call_model(
    tmp_path, monkeypatch
):
    """通知回执丢失后按同一工作事实交付，不重跑模型/记忆写入；参数：隔离根/注入器；返回：无。"""
    _, work = start_source(tmp_path)
    original = NotificationStore.enqueue

    def lost_ack(self, *args, **kwargs):
        """保存通知后丢失交付回执；参数：原通知；返回：明确IO错误。"""
        original(self, *args, **kwargs)
        raise OSError("lost knowledge delivery ack")

    with monkeypatch.context() as fault:
        fault.setattr(NotificationStore, "enqueue", lost_ack)
        with closing(
            create_scheduler(
                project_root=tmp_path,
                data_root=tmp_path,
                llm_factory=lambda _: saving_worker(),
            )
        ) as scheduler:
            with pytest.raises(OSError, match="delivery ack"):
                scheduler.run_due_jobs(now=datetime.now(timezone.utc))
    row = KnowledgeMaintenance(tmp_path).load(work["work_id"])
    assert row["state"] == "completed"

    def no_model(_options):
        """完成工作恢复不得新建模型；参数：公开配置；返回：如调用直接失败。"""
        raise AssertionError("completed knowledge must not call model again")

    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=no_model
        )
    ) as scheduler:
        assert (
            scheduler.run_due_jobs(now=datetime.now(timezone.utc))[0].status
            == "succeeded"
        )
    with closing(MemoryStore(tmp_path)) as store:
        records = store.list_memories()
        assert len(records) == 1 and records[0].version == row["commits"][0]["version"]


def test_model_setup_failure_records_knowledge_work_failure(tmp_path):
    """模型配置失败也形成知识工作的真实失败状态，不长期显示排队；参数：隔离目录；返回：无。"""
    _, work = start_source(tmp_path)

    def unavailable_model(_options):
        """模拟凭据或配置不能装配模型；参数：冻结模型选项；返回：明确配置错误。"""
        raise ValueError("saved model configuration unavailable")

    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=unavailable_model
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=datetime.now(timezone.utc))[0]
    row = KnowledgeMaintenance(tmp_path).load(work["work_id"])
    assert result.status == "failed"
    assert row["state"] == "failed" and "configuration unavailable" in row["error"]
    assert row["outcome"] is None and row["read_message_ids"] == []


def test_missing_workspace_records_failure_before_building_model(tmp_path):
    """接纳后工作区被移动时记录真实失败且不调用模型；参数：隔离目录；返回：无。"""
    project, data = tmp_path / "project", tmp_path / "data"
    project.mkdir()
    run_task(
        "记录项目客户编号规则",
        project,
        data_root=data,
        llm_client=from_test_stub("已了解"),
    )
    manager = KnowledgeMaintenance(data)
    work = manager.status()["works"][0]
    project.rename(tmp_path / "moved-project")
    model_calls = []

    def track_model(options):
        """记录不应发生的模型装配；参数：模型配置；返回：本地测试客户端。"""
        model_calls.append(options)
        return from_test_stub("unused")

    with closing(
        create_scheduler(project_root=project, data_root=data, llm_factory=track_model)
    ) as scheduler:
        result = scheduler.run_due_jobs(now=datetime.now(timezone.utc))[0]
    row = manager.load(work["work_id"])
    assert result.status == "failed" and row["state"] == "failed"
    assert row["error"] and row["outcome"] is None and not model_calls


@pytest.mark.parametrize(
    "lifecycle,cancelled", [("done", False), ("failed", False), ("paused", True)]
)
def test_restart_reconciles_terminal_run_without_claiming_knowledge_completed(
    tmp_path, lifecycle, cancelled
):
    """模型终态已保存但知识交接丢失时只对账，不重跑或伪报核验完成；参数：隔离根、终态、取消；返回：无。"""
    _, work = start_source(tmp_path)
    manager = KnowledgeMaintenance(tmp_path)
    with closing(
        create_scheduler(project_root=tmp_path, data_root=tmp_path)
    ) as scheduler:
        occurrence = scheduler.occurrences.begin(scheduler.accept_due()[0])
    WorkspaceStore(tmp_path).bind_session(occurrence.session_id, tmp_path)
    manager.update(
        work["work_id"],
        state="running",
        worker_session_id=occurrence.session_id,
        run_id=occurrence.run_id,
    )
    if cancelled:
        manager.cancel(work["work_id"])
    RunFactStore(tmp_path).append_lifecycle(
        lifecycle=lifecycle,
        reason="terminal boundary before lost acknowledgement",
        session_id=occurrence.session_id,
        run_id=occurrence.run_id,
        segment_id="worker-segment",
    )
    originals = SessionMessageStore(tmp_path).read_entries(work["source_session_id"])

    def no_model(_options):
        """已记录运行终态仅允许恢复交付；参数：原模型配置；返回：误调用明确失败。"""
        raise AssertionError("terminal recovery must not dispatch another model")

    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=no_model
        )
    ) as scheduler:
        result = scheduler.run_occurrence(occurrence.occurrence_id)
        assert result.status == ("paused" if cancelled else "failed")
        assert scheduler.run_due_jobs() == []
    row = manager.load(work["work_id"])
    assert row["state"] == ("cancelled" if cancelled else "failed")
    assert row["outcome"] is None and row["read_message_ids"] == []
    assert (
        SessionMessageStore(tmp_path).read_entries(work["source_session_id"])
        == originals
    )
