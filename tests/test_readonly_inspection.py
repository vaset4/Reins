from __future__ import annotations

from pathlib import Path

from runtime.types import ReadOnlyInspectionRequest


def test_list_dir_returns_directory_entries(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "cli.py").write_text("print('hi')", encoding="utf-8")
    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=10,
        max_chars=100,
        max_matches=10,
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(action="list_dir", target_path="app")
    )

    assert result.status == "ok"
    assert "cli.py" in result.output
    assert result.meta["truncated"] is False


def test_read_file_rejects_path_outside_repo(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=10,
        max_chars=100,
        max_matches=10,
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(action="read_file", target_path="../outside.txt")
    )

    assert result.status == "rejected"
    assert result.output.startswith("REJECTED_PATH")


def test_grep_text_returns_matching_lines(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "agent_loop.py").write_text(
        "class AgentLoop:\n    pass\n",
        encoding="utf-8",
    )
    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=10,
        max_chars=100,
        max_matches=10,
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(
            action="grep_text",
            target_path="runtime",
            query="AgentLoop",
        )
    )

    assert result.status == "ok"
    assert "AgentLoop" in result.output


def test_read_file_truncates_large_output(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    (tmp_path / "notes.txt").write_text("0123456789abcdef", encoding="utf-8")
    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=10,
        max_chars=6,
        max_matches=10,
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(action="read_file", target_path="notes.txt")
    )

    assert result.status == "ok"
    assert result.meta["truncated"] is True
    assert result.output == "012345"
    assert result.meta["returned_count"] == 6
    assert result.meta["total_count"] == 16
    assert result.meta["next_offset"] == 6


def test_read_file_uses_offset_for_followup_chunk(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    (tmp_path / "notes.txt").write_text("0123456789abcdef", encoding="utf-8")
    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=10,
        max_chars=6,
        max_matches=10,
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(
            action="read_file",
            target_path="notes.txt",
            offset=6,
        )
    )

    assert result.status == "ok"
    assert result.output == "6789ab"
    assert result.meta["offset"] == 6
    assert result.meta["next_offset"] == 12


def test_grep_text_skips_non_utf8_files_and_returns_matches(tmp_path: Path) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.txt").write_text("needle here\n", encoding="utf-8")
    (tmp_path / "src" / "bad.bin").write_bytes(b"\xe0\x80\x80")
    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=10,
        max_chars=100,
        max_matches=10,
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(
            action="grep_text",
            target_path="src",
            query="needle",
        )
    )

    assert result.status == "ok"
    assert "ok.txt" in result.output
    assert result.meta["skipped_decode_count"] == 1
    assert result.meta["skipped_decode_examples"] == ["src\\bad.bin"] or result.meta[
        "skipped_decode_examples"
    ] == ["src/bad.bin"]


def test_executor_reports_internal_errors_structurally(
    tmp_path: Path, monkeypatch
) -> None:
    from tools.readonly_inspection import ReadOnlyInspectionExecutor

    executor = ReadOnlyInspectionExecutor(
        repo_root=tmp_path,
        max_entries=10,
        max_chars=100,
        max_matches=10,
    )
    monkeypatch.setattr(
        executor,
        "_list_dir",
        lambda _path, _request: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = executor.execute(
        ReadOnlyInspectionRequest(action="list_dir", target_path=".")
    )

    assert result.status == "error"
    assert result.output == "list_dir failed: RuntimeError: boom"
    assert result.error == "readonly_inspection_execution_failed"
    assert result.meta["exception_type"] == "RuntimeError"
    assert result.meta["message"] == "boom"
