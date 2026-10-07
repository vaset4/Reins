"""单元测试：工具可见性治理"""

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
from tools.types import ToolVisibilityStatus


def _always_available() -> tuple[bool, str | None]:
    return True, None


def _never_available() -> tuple[bool, str | None]:
    return False, "test unavailable reason"


def _dummy_executor(args: dict[str, object]) -> object:
    return {"status": "ok"}


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def lease() -> Lease:
    return Lease(
        task_id="test-task",
        session_id="test-session",
        trigger="user",
        capabilities={},
    )


def test_list_definitions_available_only_filters_unavailable_tools(
    registry: ToolRegistry,
) -> None:
    """测试 list_definitions(available_only=True) 过滤不可用工具"""
    # 注册可用工具
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

    # 注册不可用工具
    registry.register(
        ToolDefinition(
            name="unavailable_tool",
            description="Unavailable tool",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_never_available,
        )
    )

    # 不过滤时应该返回所有工具
    all_tools = registry.list_definitions(model_visible_only=True)
    assert len(all_tools) == 2
    assert {tool.name for tool in all_tools} == {"available_tool", "unavailable_tool"}

    # 过滤时只返回可用工具
    available_tools = registry.list_definitions(
        model_visible_only=True, available_only=True
    )
    assert len(available_tools) == 1
    assert available_tools[0].name == "available_tool"


def test_availability_cache_mechanism(registry: ToolRegistry) -> None:
    """测试 availability 缓存机制"""
    call_count = 0

    def _counting_check() -> tuple[bool, str | None]:
        nonlocal call_count
        call_count += 1
        return True, None

    registry.register(
        ToolDefinition(
            name="cached_tool",
            description="Tool with counting check",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_counting_check,
        )
    )

    # 刷新缓存应该执行 availability_check
    call_count = 0
    registry.refresh_availability_cache("session-1")
    assert call_count == 1

    # 再次调用 list_definitions 应该使用缓存
    registry.list_definitions(available_only=True)
    assert call_count == 1  # 没有增加

    # 使缓存失效后再次调用应该重新执行
    registry.invalidate_tool_availability("cached_tool")
    registry.list_definitions(available_only=True)
    assert call_count == 2

    # 切换 session 应该清空缓存
    registry.refresh_availability_cache("session-2")
    assert call_count == 3


def test_get_visibility_report_all_statuses(registry: ToolRegistry) -> None:
    """测试 get_visibility_report 能识别所有状态"""
    # VISIBLE
    registry.register(
        ToolDefinition(
            name="visible_tool",
            description="Visible tool",
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

    # HIDDEN
    registry.register(
        ToolDefinition(
            name="hidden_tool",
            description="Hidden tool",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            model_visible=False,  # 隐藏
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_always_available,
        )
    )

    # UNAVAILABLE
    registry.register(
        ToolDefinition(
            name="unavailable_tool",
            description="Unavailable tool",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_never_available,
        )
    )

    reports, summary = registry.get_visibility_report()

    # 验证报告
    assert len(reports) == 3
    assert summary.total_tools == 3
    assert summary.visible_tools == 1
    assert summary.hidden_tools == 1
    assert summary.unavailable_tools == 1

    # 验证各工具状态
    report_map = {r.name: r for r in reports}
    assert report_map["visible_tool"].visibility_status == ToolVisibilityStatus.VISIBLE
    assert report_map["hidden_tool"].visibility_status == ToolVisibilityStatus.HIDDEN
    assert (
        report_map["unavailable_tool"].visibility_status
        == ToolVisibilityStatus.UNAVAILABLE
    )
    assert report_map["unavailable_tool"].reason == "test unavailable reason"


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

    reports, summary = registry.get_visibility_report()
    assert {report.name for report in reports} == {
        f"tool_{index}" for index in range(35)
    }
    assert summary.total_tools == summary.visible_tools == 35
    assert summary.hidden_tools == summary.unavailable_tools == 0
    assert summary.warnings == []


def test_mcp_tool_visibility_status(registry: ToolRegistry) -> None:
    """测试 MCP 工具的可见性状态识别"""

    def _mcp_not_configured() -> tuple[bool, str | None]:
        return False, "MCP is not configured"

    def _mcp_not_discovered() -> tuple[bool, str | None]:
        return False, "MCP tool not discovered"

    def _mcp_connection_failed() -> tuple[bool, str | None]:
        return False, "connection failed"

    # NOT_CONFIGURED (MCP 工具必须使用 CONFIRM 风险级别)
    registry.register(
        ToolDefinition(
            name="mcp_server1_tool1",
            description="MCP tool not configured",
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

    # NOT_DISCOVERED
    registry.register(
        ToolDefinition(
            name="mcp_server2_tool2",
            description="MCP tool not discovered",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level="confirm",
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source="mcp_reserved",
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_mcp_not_discovered,
        )
    )

    # UNAVAILABLE (连接失败)
    registry.register(
        ToolDefinition(
            name="mcp_server3_tool3",
            description="MCP tool connection failed",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level="confirm",
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source="mcp_reserved",
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=_dummy_executor,
            availability_check=_mcp_connection_failed,
        )
    )

    reports, _summary = registry.get_visibility_report()
    report_map = {r.name: r for r in reports}

    assert (
        report_map["mcp_server1_tool1"].visibility_status
        == ToolVisibilityStatus.NOT_CONFIGURED
    )
    assert (
        report_map["mcp_server2_tool2"].visibility_status
        == ToolVisibilityStatus.NOT_DISCOVERED
    )
    assert (
        report_map["mcp_server3_tool3"].visibility_status
        == ToolVisibilityStatus.UNAVAILABLE
    )
