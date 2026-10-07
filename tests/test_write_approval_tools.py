from __future__ import annotations


def test_default_tool_registry_exposes_write_tools_with_write_risk() -> None:
    from tools.tool_registry import TOOL_RISK_WRITE, get_default_tool_registry

    registry = get_default_tool_registry()

    assert registry.get("file_write") is not None
    assert registry.get("file_write").risk_level == TOOL_RISK_WRITE
    assert registry.get("file_patch") is not None
    assert registry.get("file_patch").risk_level == TOOL_RISK_WRITE
    assert "file_write" in registry.list_tool_names(model_visible_only=True)
    assert "file_patch" in registry.list_tool_names(model_visible_only=True)
