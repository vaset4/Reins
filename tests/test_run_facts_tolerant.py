"""【运行事实】【读取边界】区分未提交尾部与已经损坏的原件。

作者：xxx
时间：2026-09-30 16:30:00
"""

from pathlib import Path

import pytest

from runtime.run_facts import RunFactStore


def test_uncommitted_tail_is_not_visible_as_a_run_fact(tmp_path: Path) -> None:
    """未发布的残尾不能作为完成事实，读取不改源；传参：隔离根；返回：无。"""
    store = RunFactStore(tmp_path)
    path = store.append(
        {"event": "run:start", "session_id": "session", "run_id": "run"}
    )
    with path.open("ab") as handle:
        handle.write(b'{"event":"run:lifecycle","lifecycle":"done"')
    before = path.read_bytes()
    tolerant = store.read_run_tolerant("run")
    assert len(tolerant.facts) == 1 and tolerant.facts[0]["event"] == "run:start"
    assert tolerant.warnings == ()
    assert store.read_latest_lifecycle("run") == {}
    assert path.read_bytes() == before


def test_tolerant_queries_do_not_hide_missing_committed_original(
    tmp_path: Path,
) -> None:
    """观察入口也必须报告已提交原件丢失；传参：隔离根；返回：无。"""
    store = RunFactStore(tmp_path)
    path = store.append(
        {"event": "run:start", "session_id": "session", "run_id": "run"}
    )
    path.unlink()
    with pytest.raises(ValueError):
        store.read_run_tolerant("run")
    with pytest.raises(ValueError):
        store.list_runs_for_session_tolerant("session", limit=1)


def test_all_run_fact_queries_reject_corrupt_committed_source(tmp_path: Path) -> None:
    """所有事实入口拒绝已提交范围的坏字节；传参：隔离根；返回：无。"""
    store = RunFactStore(tmp_path)
    path = store.append(
        {
            "event": "run:start",
            "session_id": "session",
            "run_id": "run",
            "task_id": "task",
        }
    )
    path.write_bytes(b"broken\n")
    queries = (
        lambda: store.read_run("run"),
        lambda: store.list_runs_for_session("session"),
        store.list_recent_runs,
        lambda: store.list_runs_for_task("task"),
    )
    for query in queries:
        with pytest.raises(ValueError):
            query()
