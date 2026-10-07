"""【Reins】【收件箱查询】验证预览不会改写持久任务。

作者：xxx
时间：2026-09-29 11:30:00
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import pytest

from scripts.inbox_scan import scan, write_report
from runtime.persistence import RuntimeStore
from tasks.store import TaskStore


def test_scan_and_promotion_preview_preserve_records(tmp_path: Path) -> None:
    """预览正确区分事项并保留记录；参数：临时根；返回：无。"""
    with closing(TaskStore(tmp_path)) as store:
        store.create_task("正式事项", task_id="formal")
        store.create_task("暂存的想法", task_id="inbox", is_inbox=True)
        before = [store.load_task_payload(key) for key in ("formal", "inbox")]
    report = scan(tmp_path, dry_run=True)
    assert report.summary == {"inbox_count": 1, "formal_count": 1, "issues_count": 0}
    assert [(row.task_id, row.goal) for row in report.promote_candidates] == [
        ("inbox", "暂存的想法")
    ]
    assert [(row.from_path, row.to_path) for row in report.promote_plan] == [
        ("inbox:inbox", "task:inbox")
    ]
    with closing(TaskStore(tmp_path)) as store:
        assert [store.load_task_payload(key) for key in ("formal", "inbox")] == before
    assert not (tmp_path / "tasks").exists()


def test_scan_does_not_turn_missing_originals_into_empty_success(
    tmp_path: Path,
) -> None:
    """丢失文件原件不能伪装成空收件箱；参数：临时根；返回：无。"""
    with pytest.raises(
        FileNotFoundError, match="source identity or commit journal missing"
    ):
        scan(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_scan_reads_originals_without_requiring_or_rebuilding_index(
    tmp_path: Path,
) -> None:
    """索引丢失不影响只读预览，也不写新索引；参数：临时根；返回：无。"""
    TaskStore(tmp_path).create_task("保留原件", task_id="inbox", is_inbox=True)
    (tmp_path / "index.sqlite").unlink()
    before = (tmp_path / "commits.jsonl").read_bytes()
    report = scan(tmp_path)
    assert report.promote_candidates[0].goal == "保留原件"
    assert not report.index_available and not (tmp_path / "index.sqlite").exists()
    assert (tmp_path / "commits.jsonl").read_bytes() == before


def test_scan_rejects_corrupt_committed_original(tmp_path: Path) -> None:
    """已提交原件损坏必须报告；参数：临时根；返回：无。"""
    TaskStore(tmp_path).create_task("保留原件", task_id="inbox", is_inbox=True)
    RuntimeStore(tmp_path).source_path("task", "inbox").write_bytes(b"broken\n")
    with pytest.raises(ValueError):
        scan(tmp_path)


def test_rebuild_cli_recovers_derived_views_without_changing_sources(
    tmp_path, monkeypatch, capsys
):
    """维护入口从文件重建任务和记忆索引，原件字节不变；参数：临时根和命令行；返回：无。"""
    import sys
    from memory.store import MemoryStore
    from scripts.rebuild_index import main

    TaskStore(tmp_path).create_task("全局事项", task_id="task")
    memory = MemoryStore(tmp_path).create_memory("fact", "Markdown原件", [])
    before = (tmp_path / "commits.jsonl").read_bytes()
    (tmp_path / "index.sqlite").unlink()
    monkeypatch.setattr(sys, "argv", ["rebuild-index", "--data-dir", str(tmp_path)])
    assert main() == 0
    assert "'tasks': 1" in capsys.readouterr().out
    assert (tmp_path / "commits.jsonl").read_bytes() == before
    assert MemoryStore(tmp_path).load_memory(memory).content == "Markdown原件"


def test_export_report_contains_the_actual_preview(tmp_path: Path) -> None:
    """显式导出包含可读查询结果；参数：临时根；返回：无。"""
    with closing(TaskStore(tmp_path)) as store:
        store.create_task("检查预算", task_id="budget", is_inbox=True)
    report = scan(tmp_path, dry_run=True)
    path = write_report(report, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["promote_candidates"][0]["goal"] == "检查预算"
    assert payload["promote_plan"][0]["from"] == "inbox:budget"
