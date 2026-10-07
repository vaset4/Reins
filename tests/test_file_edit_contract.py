"""文件编辑的现实效果、版本冲突与发布故障验证。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from runtime.lease import Lease, from_trigger
from tools import file_persistence, restore_protection
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import ToolRegistry
from tools.types import ToolError, ToolErrorCategory

_WAIT_SECONDS = 5


def _file_api(root: Path) -> tuple[ToolRegistry, Lease]:
    """装配真实文件工具及隔离目录授权；传参：根目录；返回：注册表与租约。"""
    registry = build_tool_registry(repo_root=root, data_root=root / "data")
    lease = from_trigger(
        "user",
        task_id="file-edit",
        capabilities={
            "fs": {
                "project_root": str(root),
                "read": [str(root)],
                "write": [str(root)],
            },
        },
    )
    return registry, lease


def _patch(version: str, **overrides: object) -> dict[str, object]:
    """建立有版本依据的编辑；传参：版本与覆盖字段；返回：工具参数。"""
    return {
        "path": "sample.txt",
        "old_text": "old",
        "new_text": "new",
        "expected_sha256": version,
        **overrides,
    }


def test_repeated_matches_require_explicit_batch_edit(tmp_path: Path) -> None:
    """重复片段默认不修改，显式批量返回真实次数；传参：隔离目录；返回：无。"""
    registry, lease = _file_api(tmp_path)
    path = tmp_path / "sample.txt"
    path.write_bytes(b"old\r\nold\r\n")
    read = registry.execute_tool("file_read", {"path": "sample.txt"}, lease)
    assert read["content"] == "old\r\nold\r\n"
    version = read["meta"]["content_sha256"]
    rejected = registry.execute_tool("file_patch", _patch(version), lease)
    assert isinstance(rejected, ToolError)
    assert "2 locations" in rejected.message
    assert path.read_bytes() == b"old\r\nold\r\n"
    result = registry.execute_tool(
        "file_patch", _patch(version, replace_all=True), lease
    )
    assert result["meta"]["changed_count"] == 2
    assert path.read_bytes() == b"new\r\nnew\r\n"


def test_stale_file_version_does_not_overwrite_external_edit(tmp_path: Path) -> None:
    """读取后发生修改时保留后来内容；传参：隔离目录；返回：无。"""
    registry, lease = _file_api(tmp_path)
    path = tmp_path / "sample.txt"
    path.write_bytes(b"old")
    read = registry.execute_tool("file_read", {"path": "sample.txt"}, lease)
    path.write_bytes(b"old plus external changes")
    result = registry.execute_tool(
        "file_patch", _patch(read["meta"]["content_sha256"]), lease
    )
    assert isinstance(result, ToolError)
    assert "version conflict" in result.message
    assert path.read_bytes() == b"old plus external changes"


def test_publish_failure_keeps_original_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """发布失败不会损坏原文件或伪报成功；传参：目录与故障替换器；返回：无。"""
    registry, lease = _file_api(tmp_path)
    path = tmp_path / "sample.txt"
    path.write_bytes(b"old")
    version = file_persistence.content_sha256(path.read_bytes())

    def fail_replace(*_args: object) -> None:
        """模拟实际替换失败；传参：源和目标；返回：无，抛出IO错误。"""
        raise OSError("disk publication failed")

    monkeypatch.setattr(file_persistence._KERNEL, "ReplaceFileW", fail_replace)
    result = registry.execute_tool("file_patch", _patch(version), lease)
    assert isinstance(result, ToolError)
    assert "disk publication failed" in result.message
    assert path.read_bytes() == b"old"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["data", "sample.txt"]


def test_edit_during_temporary_write_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """暂存期间被外部修改时提交再次检查版本；传参：目录与替换器；返回：无。"""
    registry, lease = _file_api(tmp_path)
    path = tmp_path / "sample.txt"
    path.write_bytes(b"old")
    write_private_file = restore_protection.write_private_file

    def external_edit(temporary: Path, content: bytes) -> None:
        """在暂存写入完成时修改现实文件；传参：文件句柄；返回：无。"""
        write_private_file(temporary, content)
        if temporary.parent.parent == tmp_path:
            path.write_bytes(b"external")

    monkeypatch.setattr(restore_protection, "write_private_file", external_edit)
    result = registry.execute_tool(
        "file_patch", _patch(file_persistence.content_sha256(b"old")), lease
    )
    assert isinstance(result, ToolError)
    assert "changed during edit" in result.message
    assert path.read_bytes() == b"external"


def test_two_writers_cannot_commit_from_the_same_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一文件有活动写者时拒绝竞争修改，成功结果对应真实文件；传参：目录与替换器；返回：无。"""
    registry, lease = _file_api(tmp_path)
    path = tmp_path / "sample.txt"
    path.write_bytes(b"old")
    entered, release = Event(), Event()
    original_publish = file_persistence.replace_prepared_file

    def held_publish(target: Path, temporary: Path, *, backup_path: Path) -> None:
        """把第一写者停在提交前；传参：路径与字节；返回：无。"""
        entered.set()
        assert release.wait(_WAIT_SECONDS)
        original_publish(target, temporary, backup_path=backup_path)

    monkeypatch.setattr(file_persistence, "replace_prepared_file", held_publish)
    arguments = _patch(file_persistence.content_sha256(b"old"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(registry.execute_tool, "file_patch", arguments, lease)
        try:
            assert entered.wait(_WAIT_SECONDS)
            second = registry.execute_tool("file_patch", arguments, lease)
            assert isinstance(second, ToolError)
            assert "workspace write scope is busy" in second.message
        finally:
            release.set()
        result = first.result(timeout=_WAIT_SECONDS)
    assert result["meta"]["changed_count"] == 1
    assert path.read_bytes() == b"new"


def test_overwriting_requires_version_and_empty_content_is_supported(
    tmp_path: Path,
) -> None:
    """已有文件不能无依据覆盖，明确版本后可清空；传参：隔离目录；返回：无。"""
    registry, lease = _file_api(tmp_path)
    path = tmp_path / "sample.txt"
    path.write_bytes(b"old")
    arguments = {"path": "sample.txt", "content": ""}
    refused = registry.execute_tool("file_write", arguments, lease)
    assert isinstance(refused, ToolError)
    assert refused.category is ToolErrorCategory.INVALID_INPUT
    assert path.read_bytes() == b"old"
    result = registry.execute_tool(
        "file_write",
        {**arguments, "expected_sha256": file_persistence.content_sha256(b"old")},
        lease,
    )
    assert result["meta"]["byte_count"] == 0
    assert path.read_bytes() == b""


@pytest.mark.parametrize("absolute_path", [False, True])
def test_write_receipt_identifies_the_actual_file(
    tmp_path: Path, absolute_path: bool
) -> None:
    """相对和绝对路径的回执都能定位实际文件；传参：隔离目录、路径形式；返回：无。"""
    root = tmp_path / "tool-workspace"
    root.mkdir()
    registry, lease = _file_api(root)
    destination = root / "reports" / "result.json"
    requested = str(destination) if absolute_path else "reports/result.json"

    # 1. 【文件工具】【产物定位】写入成功必须说明实际落点，不能只回显用户传入的路径
    written = registry.execute_tool(
        "file_write", {"path": requested, "content": "{}"}, lease
    )
    assert not isinstance(written, ToolError)
    assert written["meta"]["resolved_path"] == str(destination.resolve())
    assert destination.read_bytes() == b"{}"

    # 2. 【文件工具】【产物定位】读取和修改使用同一文件身份，工作目录不参与路径猜测
    read = registry.execute_tool("file_read", {"path": requested}, lease)
    assert not isinstance(read, ToolError)
    assert read["meta"]["resolved_path"] == written["meta"]["resolved_path"]
    patched = registry.execute_tool(
        "file_patch",
        {
            "path": requested,
            "old_text": "{}",
            "new_text": '{"net":"64.10"}',
            "expected_sha256": read["meta"]["content_sha256"],
        },
        lease,
    )
    assert not isinstance(patched, ToolError)
    assert patched["meta"]["resolved_path"] == written["meta"]["resolved_path"]
    assert destination.read_text(encoding="utf-8") == '{"net":"64.10"}'
