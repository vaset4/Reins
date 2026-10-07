"""【文件原件】【领域恢复】验证跨领域提交、未知效果与冻结内容。

作者：xxx
时间：2026-09-30 16:30:00
"""

from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from artifacts.store import ArtifactStore
from runtime.ledger import LedgerStore
from runtime.cron import CronScheduler
from runtime.persistence import RuntimeStore
from runtime.tool_operations import ToolOperationStore
from runtime.workspaces import WorkspaceStore
from schedules.occurrences import OccurrenceStore
from schedules.store import ScheduleStore
from tasks.store import TaskStore


def test_occurrence_and_session_binding_publish_together(tmp_path, monkeypatch):
    """发生写入失败时不暴露孤立会话，重试仍只有一次发生；传参：隔离根与故障器；返回：无。"""
    from runtime import file_journal

    data = tmp_path / "data"
    workspaces = WorkspaceStore(data)
    workspace = workspaces.register(tmp_path)
    schedule = ScheduleStore(data).create_schedule(
        "daily", "interval:60", workspace_id=workspace.workspace_id
    )
    before = (data / "commits.jsonl").read_bytes()
    original = file_journal.append_synced

    def fail_occurrence(path, content):
        """模拟会话已准备、发生日志尚未同步时磁盘失败；传参：路径和字节；返回：原写入长度。"""
        if path.name == "events.jsonl" and b'"kind":"schedule_occurrence"' in content:
            raise OSError("occurrence disk failure")
        return original(path, content)

    with monkeypatch.context() as fault:
        fault.setattr(file_journal, "append_synced", fail_occurrence)
        with pytest.raises(OSError, match="occurrence disk failure"):
            OccurrenceStore(data).accept(
                "daily",
                "2026-09-30T00:00:00Z",
                next_run_at=None,
                schedule_snapshot=asdict(schedule),
            )
    assert (data / "commits.jsonl").read_bytes() == before
    assert OccurrenceStore(data).list_all() == []
    with RuntimeStore(data).snapshot() as source:
        assert source.list("session") == source.list("session_workspace") == ()
    accepted = OccurrenceStore(data).accept(
        "daily",
        "2026-09-30T00:00:00Z",
        next_run_at=None,
        schedule_snapshot=asdict(schedule),
    )
    assert workspaces.for_session(accepted.session_id) == workspace
    restored = OccurrenceStore(data).list_all()
    assert len(restored) == 1 and restored[0].occurrence_id == accepted.occurrence_id
    assert restored[0].run_id == accepted.run_id
    assert ScheduleStore(data).load_schedule("daily") == schedule


def test_task_summary_and_ledger_failure_preserve_previous_commit(
    tmp_path, monkeypatch
):
    """摘要和账本不可半提交；传参：隔离根与故障器；返回：无。"""
    tasks = TaskStore(tmp_path)
    tasks.create_task("核对资料", task_id="goal")
    tasks.update_summary_layers("goal", intent="原目标", progress="已读取")
    before = tasks.read_summary_layers("goal")
    events = LedgerStore(tmp_path).read_events()

    def fail_event(*_args, **_kwargs):
        """模拟摘要准备后的回执失败；传参：账本事件；返回：不返回。"""
        raise OSError("ledger unavailable")

    monkeypatch.setattr(LedgerStore, "append", fail_event)
    with pytest.raises(OSError, match="ledger unavailable"):
        tasks.update_summary_layers("goal", progress="新进展")
    assert TaskStore(tmp_path).read_summary_layers("goal") == before
    assert LedgerStore(tmp_path).read_events() == events


def test_due_occurrence_and_schedule_cursor_publish_together(tmp_path, monkeypatch):
    """推进游标失败时发生及子会话也不发布；传参：隔离根与故障器；返回：无。"""
    data = tmp_path / "data"
    workspace = WorkspaceStore(data).register(tmp_path)
    now = datetime(2026, 9, 30, tzinfo=timezone.utc)
    schedule = ScheduleStore(data).create_schedule(
        "due",
        "interval:60",
        workspace_id=workspace.workspace_id,
        prompt="核对",
        next_run_at=now.isoformat(),
    )
    scheduler = CronScheduler(project_root=tmp_path, data_root=data)

    def fail_cursor(*_args, **_kwargs):
        """在准备好发生后模拟计划写入失败；传参：计划更新；返回：不返回。"""
        raise OSError("cursor unavailable")

    with monkeypatch.context() as failure:
        failure.setattr(scheduler._store, "update_next_run_at", fail_cursor)
        with pytest.raises(OSError, match="cursor unavailable"):
            scheduler.accept_due(now=now)
    assert OccurrenceStore(data).list_all() == []
    assert ScheduleStore(data).load_schedule("due") == schedule
    assert len(scheduler.accept_due(now=now)) == 1
    assert scheduler.accept_due(now=now) == []
    scheduler.close()


def test_index_loss_preserves_global_task_and_tool_effect_state(tmp_path):
    """索引丢失不能让完成工具变成待执行，也不能丢掉全局目标；传参：隔离根；返回：无。"""
    tasks = TaskStore(tmp_path)
    tasks.create_task("跨项目目标", task_id="goal")
    tasks.append_journal("goal", "已核对第一份")
    tasks.update_summary_layers("goal", progress="待核对第二份")
    operations = ToolOperationStore(tmp_path)
    identity = {"session_id": "session-a", "run_id": "run-a", "operation_id": "done"}
    operations.write(
        identity, {"state": "completed", "result": {"output": "已真实写入"}}
    )
    unknown = {**identity, "operation_id": "unknown"}
    operations.write(unknown, {"state": "started"})
    (tmp_path / "index.sqlite").unlink()
    assert TaskStore(tmp_path).require_task("goal").goal == "跨项目目标"
    assert "已核对第一份" in TaskStore(tmp_path).read_journal("goal")
    assert TaskStore(tmp_path).read_summary("goal") == "待核对第二份"
    restarted = ToolOperationStore(tmp_path)
    assert restarted.load(identity)["state"] == "completed"
    assert restarted.load(unknown)["state"] == "started"
    restarted.write(identity, {"state": "late_completed"})
    restarted.write(identity, {"state": "unknown"})
    assert restarted.load(identity)["state"] == "late_completed"
    assert (
        RuntimeStore(tmp_path).source_path("task", "goal")
        == tmp_path / "global" / "events.jsonl"
    )


def test_artifact_original_is_frozen_in_its_workspace_and_rebuildable(tmp_path):
    """项目产物冻结后不受当前文件和索引变化影响；传参：隔离目录；返回：无。"""
    project, data = tmp_path / "project", tmp_path / "data"
    project.mkdir()
    workspaces = WorkspaceStore(data)
    workspace = workspaces.bind_session("artifact-session", project)
    original = project / "report.bin"
    original.write_bytes(b"real binary\0" * 10000)
    store = ArtifactStore(data)
    first = store.create_artifact(
        "global-goal",
        "output",
        str(original),
        "核对原件",
        0,
        session_id="artifact-session",
    )
    second = store.create_artifact(
        "other-goal",
        "output",
        str(original),
        "复用相同字节",
        0,
        workspace_id=workspace.workspace_id,
    )
    assert first.retained_path == second.retained_path
    assert first.workspace_id == workspace.workspace_id
    retained = store.read_path(first.artifact_id)
    assert retained.is_relative_to(
        RuntimeStore(data).workspace_directory(workspace.workspace_id) / "objects"
    )
    original.write_bytes(b"later edit")
    (data / "index.sqlite").unlink()
    assert (
        ArtifactStore(data).read_path(first.artifact_id).read_bytes()
        == b"real binary\0" * 10000
    )
    retained.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="content missing or corrupt"):
        ArtifactStore(data).read_path(first.artifact_id)


def test_artifact_cannot_capture_another_workspace_file(tmp_path):
    """显式产物身份不能越过原工作区读取另一目录；传参：隔离目录；返回：无。"""
    project, other, data = tmp_path / "project", tmp_path / "other", tmp_path / "data"
    project.mkdir()
    other.mkdir()
    WorkspaceStore(data).bind_session("session", project)
    secret = other / "private.txt"
    secret.write_text("other project", encoding="utf-8")
    with pytest.raises(ValueError, match="outside its data storage and workspace"):
        ArtifactStore(data).create_artifact(
            "goal", "output", str(secret), "越界", 0, session_id="session"
        )
