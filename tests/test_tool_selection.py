from __future__ import annotations

from llm.tool_selection import canonical_tool_schema_hash, select_tools
from llm.toolset_policy import ToolsetPolicy
from runtime.lease import Lease
from tools.tool_registry import (
    IDEMPOTENT_YES,
    TARGET_SCOPE_LOGICAL,
    TOOL_RISK_SAFE,
    TOOL_RISK_CONFIRM,
    TOOL_SOURCE_BUILTIN,
    TOOL_SOURCE_MCP_RESERVED,
    TOOLSET_AGENT,
    TOOLSET_FILE,
    TOOLSET_WEB,
    ToolDefinition,
    ToolRegistry,
    ToolRisk,
)


def test_legacy_default_selection_reports_selected_and_excluded_tools() -> None:
    registry = ToolRegistry()
    visible = _tool("visible_tool")
    hidden = _tool("hidden_tool", model_visible=False)
    unavailable = _tool("unavailable_tool", available=False)
    registry.register(visible)
    registry.register(hidden)
    registry.register(unavailable)

    selection = select_tools(registry)

    assert selection.mode == "legacy_default"
    assert selection.policy_source == "legacy_default"
    assert [tool.name for tool in selection.selected_definitions] == ["visible_tool"]
    assert selection.allowed_tool_names == frozenset({"visible_tool"})
    assert selection.selected[0].schema_hash == canonical_tool_schema_hash(visible)
    assert selection.selected[0].schema_hash.startswith("sha256:")
    assert {item.name: item.reason for item in selection.excluded} == {
        "hidden_tool": "hidden",
        "unavailable_tool": "unavailable",
    }
    assert selection.summary["selected"] == 1
    assert selection.summary["excluded"] == 2


def test_hybrid_policy_selects_enabled_toolsets_and_reports_expansion() -> None:
    registry = ToolRegistry()
    registry.register(_tool("file_read", toolset=TOOLSET_FILE))
    registry.register(_tool("web_search", toolset=TOOLSET_WEB))

    selection = select_tools(
        registry,
        policy=ToolsetPolicy(enabled_toolsets=("file",), source="payload"),
    )

    assert selection.mode == "hybrid"
    assert selection.policy_source == "payload"
    assert [item.name for item in selection.selected] == ["file_read"]
    assert {item.name: item.reason for item in selection.excluded} == {
        "web_search": "toolset_not_enabled",
    }
    assert selection.summary["expanded_tool_names"]["enabled"] == ["file_read"]


def test_hybrid_policy_disabled_toolsets_subtract_from_candidates() -> None:
    registry = ToolRegistry()
    registry.register(_tool("file_read", toolset=TOOLSET_FILE))
    registry.register(_tool("web_search", toolset=TOOLSET_WEB))

    selection = select_tools(
        registry,
        policy=ToolsetPolicy(disabled_toolsets=("web",), source="session"),
    )

    assert [item.name for item in selection.selected] == ["file_read"]
    assert {item.name: item.reason for item in selection.excluded} == {
        "web_search": "toolset_disabled",
    }


def test_hybrid_hard_boundaries_exclude_lease_and_deny_risk() -> None:
    registry = ToolRegistry()
    registry.register(_tool("terminal_tool", risk=TOOL_RISK_CONFIRM))
    registry.register(_tool("code_execution_tool", risk=TOOL_RISK_CONFIRM))
    registry.register(_tool("web_search", toolset=TOOLSET_WEB))
    registry.register(_tool("browser_click", toolset=TOOLSET_WEB))
    registry.register(
        _tool(
            "mcp_alpha_fetch", source=TOOL_SOURCE_MCP_RESERVED, risk=TOOL_RISK_CONFIRM
        )
    )
    registry.register(_tool("denied_tool", risk=ToolRisk.DENY))
    lease = Lease(
        capabilities={
            "fs": {"read": [], "write": []},
            "terminal": {"enabled": False},
            "network": {"enabled": False},
            "browser": {"enabled": False},
            "mcp": {"enabled": False, "allow_servers": []},
        }
    )

    selection = select_tools(
        registry,
        policy=ToolsetPolicy(enabled_toolsets=("full",), source="payload"),
        lease=lease,
    )

    assert selection.selected == ()
    assert {item.name: item.reason for item in selection.excluded} == {
        "browser_click": "lease_disallowed",
        "code_execution_tool": "lease_disallowed",
        "denied_tool": "risk_denied",
        "mcp_alpha_fetch": "lease_disallowed",
        "terminal_tool": "lease_disallowed",
        "web_search": "lease_disallowed",
    }


def test_fs_enabled_false_excludes_file_tools() -> None:
    registry = ToolRegistry()
    registry.register(_tool("file_read", toolset=TOOLSET_FILE))
    lease = Lease(capabilities={"fs": {"enabled": False, "read": [], "write": []}})

    selection = select_tools(
        registry,
        policy=ToolsetPolicy(enabled_toolsets=("full",), source="payload"),
        lease=lease,
    )

    assert selection.selected == ()
    assert {item.name: item.reason for item in selection.excluded} == {
        "file_read": "lease_disallowed",
    }


def test_allowed_actions_excludes_unlisted_tools() -> None:
    registry = ToolRegistry()
    registry.register(_tool("file_read", toolset=TOOLSET_FILE))
    registry.register(_tool("file_write", toolset=TOOLSET_FILE))

    selection = select_tools(registry, allowed_actions=("file_read",))

    assert [item.name for item in selection.selected] == ["file_read"]
    assert {item.name: item.reason for item in selection.excluded} == {
        "file_write": "action_not_allowed",
    }
    assert selection.summary["allowed_actions"] == ["file_read"]


def test_confirm_risk_tool_stays_selected_with_approval_status() -> None:
    registry = ToolRegistry()
    registry.register(_tool("terminal_tool", risk=TOOL_RISK_CONFIRM))

    selection = select_tools(
        registry,
        policy=ToolsetPolicy(enabled_toolsets=("full",), source="payload"),
        lease=Lease(),
    )

    assert selection.selected[0].name == "terminal_tool"
    assert selection.selected[0].status == "selected_with_approval_required"


def test_mcp_authorization_does_not_advertise_unloaded_definitions() -> None:
    registry = ToolRegistry()
    registry.register(_tool("file_read", toolset=TOOLSET_FILE))
    lease = Lease(
        capabilities={
            "fs": {"read": [], "write": []},
            "mcp": {"enabled": True, "allow_servers": ["echo"]},
        }
    )

    selection = select_tools(
        registry,
        policy=ToolsetPolicy(enabled_toolsets=("full",), source="payload"),
        lease=lease,
    )

    assert selection.allowed_tool_names == frozenset({"file_read"})


def _tool(
    name: str,
    *,
    model_visible: bool = True,
    available: bool = True,
    toolset: str = TOOLSET_AGENT,
    risk: object = TOOL_RISK_SAFE,
    source: str = TOOL_SOURCE_BUILTIN,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} description",
        parameters={
            "value": {
                "type": "string",
                "description": "test value",
                "required": True,
            }
        },
        toolset=toolset,
        risk_level=risk,
        readonly=True,
        target_scope_rule=TARGET_SCOPE_LOGICAL,
        source=source,
        model_visible=model_visible,
        idempotent=IDEMPOTENT_YES,
        executor=lambda _args: {"ok": True},
        availability_check=lambda: (available, None if available else "not installed"),
    )
