from __future__ import annotations

from runtime.types import ReadOnlyInspectionRequest, ReadOnlyInspectionResult


def test_readonly_inspection_types_can_be_constructed() -> None:
    request = ReadOnlyInspectionRequest(
        action="read_file",
        target_path="app/cli.py",
        query=None,
    )
    result = ReadOnlyInspectionResult(
        action="read_file",
        status="ok",
        output="from __future__ import annotations",
        meta={"resolved_path": "D:/repo/app/cli.py", "truncated": False},
    )

    assert request.action == "read_file"
    assert request.target_path == "app/cli.py"
    assert result.status == "ok"
    assert result.meta["truncated"] is False


def test_parse_inspect_dir_payload() -> None:
    from tools.inspection_parser import parse_inspection_payload

    request = parse_inspection_payload("dir runtime")
    assert request == ReadOnlyInspectionRequest(
        action="list_dir",
        target_path="runtime",
        query=None,
    )


def test_parse_inspect_file_payload() -> None:
    from tools.inspection_parser import parse_inspection_payload

    request = parse_inspection_payload("file app/cli.py")
    assert request == ReadOnlyInspectionRequest(
        action="read_file",
        target_path="app/cli.py",
        query=None,
    )


def test_parse_inspect_search_payload() -> None:
    from tools.inspection_parser import parse_inspection_payload

    request = parse_inspection_payload("search runtime for AgentLoop")
    assert request == ReadOnlyInspectionRequest(
        action="grep_text",
        target_path="runtime",
        query="AgentLoop",
    )


def test_parse_workspace_alias_as_repo_root() -> None:
    from tools.inspection_parser import parse_inspection_payload

    request = parse_inspection_payload("workspace")
    assert request == ReadOnlyInspectionRequest(
        action="list_dir",
        target_path=".",
        query=None,
    )


def test_parse_invalid_payload_returns_none() -> None:
    from tools.inspection_parser import parse_inspection_payload

    assert parse_inspection_payload("please inspect something useful") is None
