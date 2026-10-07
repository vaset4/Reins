"""
沉淀补扫查漏测试
测试 list_missing_sediment / backfill_sediment / batch_backfill_sediment
覆盖 AC1-AC14

作者: LKX
时间: 2026-07-28 15:30:00
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tasks.records import TaskRecord
from runtime.run_facts import RunFactStore
from runtime.persistence import RuntimeStore
from tasks.sediment_cli import (
    backfill_sediment,
    batch_backfill_sediment,
    list_missing_sediment,
)
from tasks.store import TaskStore


@pytest.fixture
def temp_data_root(tmp_path: Path) -> Path:
    """临时数据根目录"""
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "tasks").mkdir()
    (data_root / "memory" / "active").mkdir(parents=True)
    (data_root / "memory" / "draft").mkdir(parents=True)
    (data_root / "skills" / "active").mkdir(parents=True)
    (data_root / "skills" / "manual").mkdir(parents=True)
    return data_root


@pytest.fixture
def mock_proposer():
    """Mock proposer 返回正确的结构（每次调用返回不同内容避免冲突检测）"""
    call_count = [0]

    def proposer(*args, **kwargs):
        call_count[0] += 1
        return {
            "memory": {
                "content": f"test memory {call_count[0]}",
                "tags": ["test"],
                "type": "fact",
            },
            "skill": {
                "skill_id": f"test-skill-{call_count[0]}",
                "name": f"test-skill-{call_count[0]}",
                "description": f"test skill description {call_count[0]}",
                "body": f"echo test {call_count[0]}",
                "validation": "test validation",
                "tags": ["test"],
            },
        }

    mock = MagicMock()
    mock.side_effect = proposer
    return mock


def _create_task(
    store: TaskStore,
    task_id: str,
    status: str,
    *,
    sediment_done: bool = False,
    sediment_attempts: int = 0,
    sediment_failed: bool = False,
) -> TaskRecord:
    """使用实际领域写者建立任务、摘要和运行事实；传参：目标状态；返回：保存后的目标。"""
    store.create_task(f"Test task {task_id}", task_id=task_id)
    store.update_task_status(task_id, status)
    task = store.update_sediment_status(
        task_id, done=sediment_done, attempts=sediment_attempts, failed=sediment_failed
    )
    store.update_summary(task_id, "Test task summary")
    for step in range(5):
        RunFactStore(store._data_root).append(
            {
                "event": "step",
                "task_id": task_id,
                "session_id": f"session-{task_id}",
                "run_id": f"run-{task_id}",
                "step": step,
            }
        )
    return task


# ========== AC1-AC3: 查漏函数 ==========


def test_list_missing_sediment_empty(temp_data_root: Path):
    """AC2: 全部已沉淀时返回空列表"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=True)

    missing = list_missing_sediment(temp_data_root)
    assert missing == []


def test_list_missing_sediment_basic(temp_data_root: Path):
    """AC1: 返回未沉淀任务清单，含必要字段"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=False)
    _create_task(store, "task-002", "failed", sediment_done=False, sediment_attempts=2)
    _create_task(store, "task-003", "done", sediment_done=True)  # 已沉淀，应过滤

    missing = list_missing_sediment(temp_data_root)

    assert len(missing) == 2
    assert all(
        k in missing[0]
        for k in [
            "task_id",
            "status",
            "sediment_done",
            "sediment_attempts",
            "sediment_failed",
        ]
    )
    assert missing[0]["sediment_done"] is False
    assert missing[0]["steps"] is None  # with_steps=False 默认


def test_list_missing_sediment_with_steps(temp_data_root: Path):
    """AC3: with_steps=True 时统计 events.jsonl 行数"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=False)

    missing = list_missing_sediment(temp_data_root, with_steps=True)

    assert len(missing) == 1
    assert missing[0]["steps"] == 5  # _create_task 写了 5 行


def test_list_missing_sediment_status_filter(temp_data_root: Path):
    """查漏支持按状态过滤"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=False)
    _create_task(store, "task-002", "failed", sediment_done=False)

    missing_done = list_missing_sediment(temp_data_root, status="done")
    assert len(missing_done) == 1
    assert missing_done[0]["status"] == "done"

    missing_failed = list_missing_sediment(temp_data_root, status="failed")
    assert len(missing_failed) == 1
    assert missing_failed[0]["status"] == "failed"


def test_list_missing_sediment_limit(temp_data_root: Path):
    """查漏支持 LIMIT"""
    store = TaskStore(temp_data_root)
    for i in range(10):
        _create_task(store, f"task-{i:03d}", "done", sediment_done=False)

    missing = list_missing_sediment(temp_data_root, limit=3)
    assert len(missing) == 3


# ========== AC4-AC7: 单任务补扫 ==========


def test_backfill_sediment_success(temp_data_root: Path, mock_proposer):
    """AC4: 补扫成功返回 written 且标 sediment_done=True"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=False)

    mock_client = MagicMock()

    with patch(
        "tasks.sediment_cli.build_reflection_proposer", return_value=mock_proposer
    ):
        result = backfill_sediment(temp_data_root, "task-001", mock_client)

    assert result["status"] == "written"
    assert "memory_id" in result
    assert "skill_id" in result

    # 验证标志位
    task = store.load_task("task-001")
    assert task.sediment_done is True


def test_backfill_sediment_active_rejected(temp_data_root: Path, mock_proposer):
    """AC5: active/paused 任务抛 ValueError"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "active")

    mock_client = MagicMock()

    with pytest.raises(ValueError, match="only done/failed tasks"):
        backfill_sediment(temp_data_root, "task-001", mock_client)


def test_backfill_sediment_skip_done(temp_data_root: Path, mock_proposer):
    """AC6: 已沉淀任务返回 skipped"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=True)

    mock_client = MagicMock()

    result = backfill_sediment(temp_data_root, "task-001", mock_client)

    assert result["status"] == "skipped"
    assert "already_done" in result["reason"]


def test_backfill_sediment_force_override(temp_data_root: Path, mock_proposer):
    """强制整理追加新快照并保留旧依据；参数：独立数据根与提议器；返回：无。"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=True)
    store.append_reflection("task-001", {"old": "data"})
    with RuntimeStore(temp_data_root).snapshot() as source:
        old = source.list("task_reflection", filters={"task_id": "task-001"})[0]

    with patch(
        "tasks.sediment_cli.build_reflection_proposer", return_value=mock_proposer
    ):
        result = backfill_sediment(temp_data_root, "task-001", MagicMock(), force=True)

    assert result["status"] == "written"
    with RuntimeStore(temp_data_root).snapshot() as source:
        reflections = source.list("task_reflection", filters={"task_id": "task-001"})
    assert len(reflections) == 2
    assert (
        next(row for row in reflections if row["reflection_id"] == old["reflection_id"])
        == old
    )
    assert old["payload"] == {"old": "data"}
    assert any(row["payload"].get("memory") for row in reflections)


# ========== AC8-AC9: 批量补扫 ==========


def test_batch_backfill_success(temp_data_root: Path, mock_proposer):
    """AC8: 批量补扫返回统计"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=False)
    _create_task(store, "task-002", "done", sediment_done=False)

    mock_client = MagicMock()

    with patch(
        "tasks.sediment_cli.build_reflection_proposer", return_value=mock_proposer
    ):
        result = batch_backfill_sediment(temp_data_root, mock_client, limit=10)

    assert result["success"] == 2
    assert result["skipped"] == 0
    assert result["failed"] == 0


def test_batch_backfill_partial_failure(temp_data_root: Path, mock_proposer):
    """AC9: 批量补扫中某任务失败，不影响后续"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=False)
    _create_task(store, "task-002", "done", sediment_done=False)
    _create_task(store, "task-003", "done", sediment_done=False)

    mock_client = MagicMock()

    # 让 proposer 第二次调用失败（用计数器生成不同内容避免冲突）
    call_count = [0]

    def failing_proposer(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("proposer failed")
        return {
            "memory": {
                "content": f"ok memory {call_count[0]}",
                "tags": [],
                "type": "fact",
            },
            "skill": {
                "skill_id": f"ok-{call_count[0]}",
                "name": f"ok-{call_count[0]}",
                "description": f"ok skill {call_count[0]}",
                "body": f"echo {call_count[0]}",
                "validation": "validated",
                "tags": [],
            },
        }

    failing_mock = MagicMock()
    failing_mock.side_effect = failing_proposer

    with patch(
        "tasks.sediment_cli.build_reflection_proposer", return_value=failing_mock
    ):
        result = batch_backfill_sediment(temp_data_root, mock_client, limit=10)

    assert result["success"] == 2
    assert result["failed"] == 1


# ========== AC10: dry-run ==========


def test_backfill_dry_run(temp_data_root: Path):
    """AC10: dry-run 入口拦截，不调 proposer"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=False)

    mock_client = MagicMock()
    mock_proposer = MagicMock()

    with patch(
        "tasks.sediment_cli.build_reflection_proposer", return_value=mock_proposer
    ):
        result = backfill_sediment(
            temp_data_root, "task-001", mock_client, dry_run=True
        )

    assert result["status"] == "dry_run"
    assert result["would_run"] is True
    mock_proposer.assert_not_called()  # proposer 不应被调


def test_batch_backfill_dry_run(temp_data_root: Path):
    """批量补扫 dry-run"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=False)
    _create_task(store, "task-002", "done", sediment_done=False)

    mock_client = MagicMock()

    result = batch_backfill_sediment(temp_data_root, mock_client, dry_run=True)

    # dry-run 全部计入 skipped
    assert result["skipped"] == 2
    assert result["success"] == 0


# ========== AC11: 失败计数 ==========


def test_backfill_failure_count(temp_data_root: Path):
    """AC11: 补扫失败计入 sediment_attempts"""
    store = TaskStore(temp_data_root)
    _create_task(store, "task-001", "done", sediment_done=False, sediment_attempts=0)

    mock_client = MagicMock()
    failing_proposer = MagicMock(side_effect=RuntimeError("proposer failed"))

    with patch(
        "tasks.sediment_cli.build_reflection_proposer", return_value=failing_proposer
    ):
        result = backfill_sediment(temp_data_root, "task-001", mock_client)

    assert result["status"] == "failed"

    # 验证 attempts 增加（由 run_sediment 内部 _record_failure 处理）
    task_payload = store.load_task_payload("task-001")
    assert task_payload.get("sediment_attempts", 0) == 1
