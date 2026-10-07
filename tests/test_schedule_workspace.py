"""验证统一资料库中的定时工作始终属于创建时的工作区。

作者：xxx
时间：2026-09-30 11:00:00
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from runtime.cron import CronScheduler
from runtime.workspaces import WorkspaceStore
from schedules.store import ScheduleRecord, ScheduleStore
from triggers.cron import make_run_context


def test_schedule_requires_a_known_workspace(tmp_path: Path) -> None:
    """拒绝不存在的工作区身份；参数：隔离资料库；返回：无。"""
    with closing(ScheduleStore(tmp_path / "data")) as store:
        with pytest.raises(ValueError, match="workspace"):
            store.create_schedule(
                "unknown", "interval:60", workspace_id="missing", prompt="核对资料"
            )
        assert store.load_schedule("unknown") is None


@pytest.mark.parametrize("kind", ["work", "reminder"])
def test_accepted_work_keeps_original_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """从另一目录接纳工作或提醒仍绑定原目录；参数：隔离目录及计划种类；返回：无。"""
    data_root = tmp_path / "data"
    original, other = tmp_path / "a", tmp_path / "b"
    original.mkdir()
    other.mkdir()
    workspaces = WorkspaceStore(data_root)
    workspace = workspaces.register(original)
    alternative = workspaces.register(other)
    now = datetime(2026, 9, 30, 3, tzinfo=timezone.utc)
    with closing(ScheduleStore(data_root)) as store:
        store.create_schedule(
            "original",
            "interval:60",
            workspace_id=workspace.workspace_id,
            prompt="核对原目录",
            next_run_at=now.isoformat(),
            kind=kind,
        )
    monkeypatch.chdir(other)
    with closing(CronScheduler(project_root=other, data_root=data_root)) as scheduler:
        occurrence = scheduler.accept_due(now=now)[0]
        assert occurrence.schedule_snapshot["workspace_id"] == workspace.workspace_id
        assert (
            workspaces.for_session(occurrence.session_id).project_root
            == original.resolve()
        )
        with closing(ScheduleStore(data_root)) as store:
            changed = store.update("original", prompt="下一次核对")
            assert changed.workspace_id == workspace.workspace_id
            with pytest.raises(ValueError, match="workspace"):
                store.update("original", workspace_id=alternative.workspace_id)
        restored = scheduler.occurrences.load(occurrence.occurrence_id)
        assert (
            restored is not None
            and restored.schedule_snapshot["prompt"] == "核对原目录"
        )


def test_cron_context_uses_saved_workspace(tmp_path: Path) -> None:
    """定时运行上下文从计划读取工作区，不从调用目录推断；参数：隔离目录；返回：无。"""
    data_root = tmp_path / "data"
    workspace = WorkspaceStore(data_root).register(tmp_path)
    with closing(ScheduleStore(data_root)) as store:
        job = store.create_schedule(
            "context",
            "interval:60",
            workspace_id=workspace.workspace_id,
            target_task_id="task",
            prompt="核对",
        )
    context = make_run_context(
        job.schedule_id,
        data_root=data_root,
        session_id="scheduled-session",
        run_id="scheduled-run",
        schedule_data=asdict(job),
    )
    assert context.payload["workspace_id"] == workspace.workspace_id


def test_schedule_without_workspace_is_not_treated_as_current_data() -> None:
    """旧记录缺少归属时明确失败，不能套用后来启动的目录；参数：无；返回：无。"""
    with pytest.raises(ValueError, match="workspace"):
        ScheduleRecord.from_dict({"schedule_id": "old", "cron": "interval:60"})


def test_scheduled_file_write_uses_original_workspace_from_another_startup(
    tmp_path: Path,
) -> None:
    """从B启动调度仍将A的任务产物写入A；参数：两个隔离工作区及统一资料库；返回：无。"""
    from app.scheduled_run import create_scheduler
    from llm.messages import ToolCallPart
    from runtime.default_capabilities import build_local_agent_capabilities
    from scripts.testing.llm import from_test_native_tool_then_final

    original, other, data_root = tmp_path / "a", tmp_path / "b", tmp_path / "data"
    original.mkdir()
    other.mkdir()
    workspace = WorkspaceStore(data_root).register(original)
    capabilities = build_local_agent_capabilities(original, data_root)
    capabilities["mcp"] = {"enabled": False, "allow_servers": []}
    now = datetime(2026, 9, 30, 3, tzinfo=timezone.utc)
    with closing(ScheduleStore(data_root)) as store:
        store.create_schedule(
            "write-original",
            "interval:60",
            workspace_id=workspace.workspace_id,
            prompt="把结果写入 report.txt",
            next_run_at=now.isoformat(),
            capabilities=capabilities,
            model_config={"model": "scheduled-test", "reasoning_effort": "default"},
        )
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "write", "file_write", {"path": "report.txt", "content": "原工作区结果"}
            ),
        ],
        "结果已保存",
    )
    with closing(
        create_scheduler(
            project_root=other, data_root=data_root, llm_factory=lambda _: client
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=now)[0]
    assert result.status == "succeeded"
    assert (original / "report.txt").read_text(encoding="utf-8") == "原工作区结果"
    assert not (other / "report.txt").exists()
    assert (
        WorkspaceStore(data_root).for_session(result.session_id).workspace_id
        == workspace.workspace_id
    )
