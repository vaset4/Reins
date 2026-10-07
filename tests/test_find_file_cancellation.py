"""文件搜索的模式兼容与真实协作停止回归。

作者：xxx
时间：2026-09-30 12:00:00
"""

from __future__ import annotations

from contextlib import contextmanager
import time

import pytest

from runtime.lease import from_trigger
from runtime.cancellation import CancellationToken
from runtime.watchdog import Watchdog
from tools.builtin_tools import build_tool_registry
from tools.readonly_inspection import ReadOnlyInspectionExecutor
from runtime.types import ReadOnlyInspectionRequest
from tools.types import ToolError, ToolErrorCategory


@pytest.mark.parametrize(
    "query",
    [
        "pdf",
        "*.pdf",
        "*.PDF",
        "[ab]?.pdf",
        "sub/*.pdf",
        "sub/a",
        "**/*.pdf",
        "sub/**/a*.pdf",
        "**/**/a*.pdf",
        "sub/../*.pdf",
    ],
)
def test_find_file_keeps_pathlib_results(tmp_path, query):
    """裸名、路径片段、多级 glob、隐藏目录保持原有结果；参数：临时目录、查询；返回：无。"""
    for name in (
        "a1.pdf",
        "sub/a2.pdf",
        "sub/deep/a3.pdf",
        ".hidden/b4.PDF",
        "sub/note.txt",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("sample", encoding="utf-8")
    pattern = query if any(char in query for char in "*?[") else f"*{query}*"
    expected = [
        item.relative_to(tmp_path).as_posix()
        for item in sorted(tmp_path.rglob(pattern))
        if item.is_file()
    ]
    executor = ReadOnlyInspectionExecutor(tmp_path, 10, 100, 100)
    result = executor.execute(
        ReadOnlyInspectionRequest(action="find_path", target_path=".", query=query)
    )
    assert result.status == "ok"
    assert result.output.splitlines() == expected
    assert result.meta["total_count"] == len(expected)


def test_timeout_stops_search_and_closes_directory_iterator(tmp_path, monkeypatch):
    """真实注册表超时后搜索退出并关闭扫描器，不能报告未知或迟到成功；参数：目录、替换器；返回：无。"""
    import tools.readonly_inspection as inspection

    for index in range(10):
        (tmp_path / f"{index}.txt").write_text("sample", encoding="utf-8")
    original = inspection.os.scandir
    visited = []
    closed = []

    @contextmanager
    def slow_scan(path):
        """保留真实目录句柄，仅延缓条目交付模拟慢盘；参数：目录；返回：迭代器。"""
        with original(path) as entries:
            try:
                yield delayed_entries(entries)
            finally:
                closed.append(path)

    def delayed_entries(entries):
        """按固定延迟提供真实条目；参数：目录条目；返回：条目流。"""
        for entry in entries:
            time.sleep(0.03)
            visited.append(entry.name)
            yield entry

    registry = build_tool_registry(repo_root=tmp_path)
    lease = from_trigger(
        "user",
        capabilities={"fs": {"project_root": str(tmp_path), "read": [str(tmp_path)]}},
    )
    watchdog = Watchdog(lease, tool_timeout_seconds=0.05)
    monkeypatch.setattr(inspection.os, "scandir", slow_scan)
    result = registry.execute_tool(
        "find_path", {"query": "*.pdf"}, lease, watchdog=watchdog
    )
    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.TIMEOUT
    assert result.details["execution_state"] == "completed"
    assert result.details["stopped"] is True
    assert len(closed) == 1
    count = len(visited)
    assert count < 10
    time.sleep(0.06)
    assert len(visited) == count


def test_find_file_cancelled_before_scan_never_opens_directory(tmp_path, monkeypatch):
    """已取消搜索不启动磁盘遍历；参数：目录、替换器；返回：无。"""
    import tools.readonly_inspection as inspection

    def unexpected_scan(path):
        """拒绝取消后继续访问目录；参数：目录；返回：无。"""
        raise AssertionError("cancelled search entered scandir")

    monkeypatch.setattr(inspection.os, "scandir", unexpected_scan)
    token = CancellationToken()
    token.cancel("user_stop")
    executor = ReadOnlyInspectionExecutor(tmp_path, 10, 100, 2)
    result = executor.execute(
        ReadOnlyInspectionRequest(action="find_path", target_path=".", query="*.pdf"),
        cancellation=token,
    )
    assert result.status == "error"
    assert result.meta["error_category"] == "cancelled"
    assert token.backend_evidence()["stopped"] is True


def test_find_file_preserves_sorted_window_and_total(tmp_path):
    """窗口截断保留准确总数和路径排序，不提前退出伪造总数；参数：目录；返回：无。"""
    for name in ("z.pdf", "a.pdf", "m.pdf"):
        (tmp_path / name).write_text("sample", encoding="utf-8")
    executor = ReadOnlyInspectionExecutor(tmp_path, 10, 100, 2)
    result = executor.execute(
        ReadOnlyInspectionRequest(action="find_path", target_path=".", query="*.pdf")
    )
    assert result.output.splitlines() == ["a.pdf", "m.pdf"]
    assert result.meta["total_count"] == 3
    assert result.meta["truncated"] is True


def test_find_file_surfaces_unreadable_directory(tmp_path, monkeypatch):
    """无法读取目录必须返回真实错误，不能伪装没有匹配；参数：目录、替换器；返回：无。"""
    import tools.readonly_inspection as inspection

    def denied_scan(path):
        """模拟操作系统拒绝遍历；参数：目录；返回：无。"""
        raise PermissionError("directory access denied")

    monkeypatch.setattr(inspection.os, "scandir", denied_scan)
    executor = ReadOnlyInspectionExecutor(tmp_path, 10, 100, 2)
    result = executor.execute(
        ReadOnlyInspectionRequest(action="find_path", target_path=".", query="*.pdf")
    )
    assert result.status == "error"
    assert "PermissionError" in result.output
    assert "directory access denied" in result.output
