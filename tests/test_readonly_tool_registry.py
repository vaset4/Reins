from __future__ import annotations


def test_default_tool_registry_exposes_readonly_file_and_web_tools() -> None:
    from tools.tool_registry import get_default_tool_registry

    registry = get_default_tool_registry()

    model_visible = registry.list_tool_names(model_visible_only=True)

    assert "file_read" in model_visible
    assert "list" in model_visible
    assert "grep" in model_visible
    assert "web_search" in model_visible
    assert "web_fetch" in model_visible
    assert "web_scan" in model_visible
    assert registry.get("inspect") is not None
    assert "inspect" not in model_visible


def test_workspace_file_and_artifact_tool_descriptions_are_disambiguated() -> None:
    from tools.builtin_tools import build_tool_registry

    registry = build_tool_registry(repo_root=".")
    file_read = registry.get("file_read")
    read_artifact = registry.get("read_artifact")

    assert "PDF" in file_read.description
    assert "ordinary workspace filenames" in read_artifact.description
    artifact_id = read_artifact.parameters["properties"]["artifact_id"]
    assert isinstance(artifact_id, dict)
    assert "not a filename or path" in artifact_id["description"]
