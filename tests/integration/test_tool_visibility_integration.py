"""集成测试：工具可见性治理在实际场景中的行为"""

from __future__ import annotations

import pytest

from runtime.lease import Lease
from tools.tool_registry import (
    IDEMPOTENT_YES,
    TARGET_SCOPE_LOGICAL,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_AGENT,
    ToolDefinition,
    ToolRegistry,
)
from tools.types import ToolError, ToolVisibilityStatus


def _dummy_executor(args: dict[str, object]) -> object:
    return {"status": "ok"}


def _always_available() -> tuple[bool, str | None]:
    return True, None


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def lease() -> Lease:
    return Lease(
        task_id="test-task",
        trigger="user",
        capabilities={},
    )


def test_browser_dependency_missing_scenario(registry: ToolRegistry) -> None:
    """测试浏览器依赖缺失场景"""

    def _playwright_unavailable() -> tuple[bool, str | None]:
        return (
            False,
            "browser_unavailable: Playwright optional dependency is not installed",
        )

    # 注册浏览器工具（模拟 Playwright 不可用）
    registry.register(
        ToolDefinition(
            name="browser_navigate",
            description="Navigate to URL",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_playwright_unavailable,
        )
    )

    # 注册一个可用工具作为对照
    registry.register(
        ToolDefinition(
            name="available_tool",
            description="Available tool",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_always_available,
        )
    )

    # 模拟 Session 启动：刷新缓存
    registry.refresh_availability_cache("session-1")

    # 验证模型工具列表不包含浏览器工具
    model_visible_tools = registry.list_definitions(
        model_visible_only=True, available_only=True
    )
    tool_names = {tool.name for tool in model_visible_tools}
    assert "browser_navigate" not in tool_names
    assert "available_tool" in tool_names

    # 验证 get_visibility_report 显示 UNAVAILABLE
    reports, summary = registry.get_visibility_report()
    report_map = {r.name: r for r in reports}

    assert (
        report_map["browser_navigate"].visibility_status
        == ToolVisibilityStatus.UNAVAILABLE
    )
    assert "Playwright" in (report_map["browser_navigate"].reason or "")
    assert (
        report_map["available_tool"].visibility_status == ToolVisibilityStatus.VISIBLE
    )


def test_mcp_not_configured_scenario(registry: ToolRegistry) -> None:
    """测试 MCP 未配置场景"""

    def _mcp_not_configured() -> tuple[bool, str | None]:
        return False, "MCP is not configured"

    # 注册 MCP 工具（模拟 MCP 未配置）
    registry.register(
        ToolDefinition(
            name="mcp_server1_tool1",
            description="MCP tool",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level="confirm",
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source="mcp_reserved",
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_mcp_not_configured,
        )
    )

    # 注册一个可用工具作为对照
    registry.register(
        ToolDefinition(
            name="available_tool",
            description="Available tool",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_always_available,
        )
    )

    # 模拟 Session 启动：刷新缓存
    registry.refresh_availability_cache("session-1")

    # 验证模型工具列表不包含 MCP 工具
    model_visible_tools = registry.list_definitions(
        model_visible_only=True, available_only=True
    )
    tool_names = {tool.name for tool in model_visible_tools}
    assert "mcp_server1_tool1" not in tool_names
    assert "available_tool" in tool_names

    # 验证 get_visibility_report 显示 NOT_CONFIGURED
    reports, summary = registry.get_visibility_report()
    report_map = {r.name: r for r in reports}

    assert (
        report_map["mcp_server1_tool1"].visibility_status
        == ToolVisibilityStatus.NOT_CONFIGURED
    )
    assert (
        report_map["available_tool"].visibility_status == ToolVisibilityStatus.VISIBLE
    )


def test_visibility_reports_every_available_tool_without_count_limit(
    registry: ToolRegistry,
) -> None:
    """可用工具完整显示，数量本身不构成故障；传参：注册表；返回：无。"""
    # 注册 35 个可见工具
    for i in range(35):
        registry.register(
            ToolDefinition(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters={},
                toolset=TOOLSET_AGENT,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_YES,
                executor=_dummy_executor,
                availability_check=_always_available,
            )
        )

    # 调用 get_visibility_report
    reports, summary = registry.get_visibility_report()
    assert {report.name for report in reports} == {
        f"tool_{index}" for index in range(35)
    }
    assert summary.total_tools == summary.visible_tools == 35
    assert summary.hidden_tools == summary.unavailable_tools == 0
    assert summary.warnings == []


def test_cache_invalidation_on_tool_execution_failure(
    registry: ToolRegistry, lease: Lease
) -> None:
    """测试工具首次调用失败后缓存失效"""
    call_count = 0
    tool_available = True

    def _dynamic_availability() -> tuple[bool, str | None]:
        nonlocal call_count
        call_count += 1
        if tool_available:
            return True, None
        return False, "tool became unavailable"

    # 注册一个动态可用性的工具
    registry.register(
        ToolDefinition(
            name="dynamic_tool",
            description="Tool with dynamic availability",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_dynamic_availability,
        )
    )

    # 启动 Session（工具可用）
    call_count = 0
    registry.refresh_availability_cache("session-1")
    assert call_count == 1

    # 验证工具在模型列表中
    tools = registry.list_definitions(model_visible_only=True, available_only=True)
    assert any(t.name == "dynamic_tool" for t in tools)

    # 模拟工具变为不可用
    tool_available = False

    # 调用工具失败（通过 execute_tool 触发）
    result = registry.execute_tool("dynamic_tool", {}, lease)
    assert isinstance(result, ToolError)
    assert "unavailable" in result.message

    # 验证缓存失效（call_count 应该增加，因为 execute 会重新检查）
    assert call_count == 2

    # 下一轮生成工具列表时，该工具被过滤
    tools_after_failure = registry.list_definitions(
        model_visible_only=True, available_only=True
    )
    assert not any(t.name == "dynamic_tool" for t in tools_after_failure)
