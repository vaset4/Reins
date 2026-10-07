from __future__ import annotations

import pytest

from tools.tool_registry import (
    Idempotent,
    MCPToolRiskError,
    TARGET_SCOPE_LOGICAL,
    TOOL_SOURCE_MCP_RESERVED,
    TOOLSET_AGENT,
    ToolDefinition,
    ToolRegistry,
    ToolRisk,
)


def _definition(name: str, risk: ToolRisk) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="mcp tool",
        parameters={},
        toolset=TOOLSET_AGENT,
        risk_level=risk,
        readonly=True,
        target_scope_rule=TARGET_SCOPE_LOGICAL,
        source=TOOL_SOURCE_MCP_RESERVED,
        idempotent=Idempotent.YES,
    )


def test_mcp_tool_cannot_register_safe_risk() -> None:
    with pytest.raises(MCPToolRiskError):
        ToolRegistry().register(_definition("mcp_test_tool", ToolRisk.SAFE))


def test_mcp_tool_allows_confirm_and_deny() -> None:
    registry = ToolRegistry()

    registry.register(_definition("mcp_confirm_tool", ToolRisk.CONFIRM))
    registry.register(_definition("mcp_deny_tool", ToolRisk.DENY))

    assert registry.get("mcp_confirm_tool") is not None
    assert registry.get("mcp_deny_tool") is not None
