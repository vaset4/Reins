from __future__ import annotations

from contextlib import closing
from pathlib import Path

from runtime.checkpoint import Checkpoint, save_checkpoint
from runtime.types import RunToolsRequest


def test_checkpoint_round_trip_preserves_structured_run_tools_request(
    tmp_path: Path,
) -> None:
    del tmp_path
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id="task_demo",
            segment_id="segment_demo",
            state="EXECUTING_TOOL",
            pending_tool_call={
                "tool_name": "inspect",
                "args": {"payload": "file app/cli.py"},
                "call_id": "call_demo",
            },
        )
    )

    assert checkpoint.pending_tool_call is not None
    assert checkpoint.pending_tool_call["tool_name"] == "inspect"
    assert checkpoint.pending_tool_call["args"] == {"payload": "file app/cli.py"}
    assert checkpoint.checkpoint_id
    assert checkpoint.saved_at


def test_file_executor_returns_unified_result_for_structured_request(
    tmp_path: Path,
) -> None:
    """验证正式文件读取返回正文、状态和目标路径；参数：隔离目录；返回：无。"""
    from tools.readonly_file_tools import ReadOnlyFileToolExecutor
    from tools.readonly_inspection import (
        DEFAULT_READ_MAX_CHARS,
        ReadOnlyInspectionExecutor,
    )

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "cli.py").write_text("print('hi')\n", encoding="utf-8")

    executor = ReadOnlyFileToolExecutor(
        ReadOnlyInspectionExecutor(tmp_path, 50, DEFAULT_READ_MAX_CHARS, 50)
    )
    result = executor.execute(
        RunToolsRequest(
            action="file_read",
            tool_name="file_read",
            arguments={"path": "app/cli.py"},
            target_scope="app/cli.py",
        )
    )

    assert result.status == "ok"
    assert result.content is not None
    assert "print('hi')" in result.content
    assert result.summary == "tool executed"
    assert result.error is None
    assert result.target_scope == "app/cli.py"
    assert "print('hi')" in result.output


def test_registry_normalizes_alias_arguments_for_real_file_write(
    tmp_path: Path,
) -> None:
    """验证注册表规范化参数别名后实际创建文件；参数：隔离目录；返回：无。"""
    from tools.builtin_tools import build_tool_registry
    from tools.write_file_tools import WriteFileToolExecutor

    with closing(
        build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    ) as registry:
        request = registry.normalize_request(
            RunToolsRequest(
                action="file_write",
                tool_name="file_write",
                arguments={
                    "target_path": "notes.txt",
                    "body": "hello from alias",
                },
            )
        )
        assert request.arguments == {"path": "notes.txt", "content": "hello from alias"}
        assert request.target_scope == "notes.txt"
        result = WriteFileToolExecutor(tmp_path).execute(request)

    assert result.status == "ok"
    assert result.summary == "tool executed"
    assert result.target_scope == "notes.txt"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello from alias"
