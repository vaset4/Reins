from __future__ import annotations

import importlib.util

from tools.browser import playwright_adapter
from tools.tool_registry import (
    IDEMPOTENT_CONDITIONAL,
    IDEMPOTENT_NO,
    IDEMPOTENT_YES,
    TARGET_SCOPE_DOMAIN,
    TARGET_SCOPE_LOGICAL,
    TOOL_RISK_CONFIRM,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_WEB,
    ToolDefinition,
    ToolRegistry,
)


def register_tools(registry: ToolRegistry) -> None:
    if registry.get("browser_navigate") is None:
        registry.register(
            ToolDefinition(
                name="browser_navigate",
                description=(
                    "Open one HTTP or HTTPS URL in the active browser profile. "
                    "For page evidence, call browser_extract after navigation and "
                    "browser_screenshot after extraction."
                ),
                parameters={
                    "url": {
                        "type": "string",
                        "required": True,
                        "description": "HTTP, HTTPS, or allowed local file URL to open first.",
                    }
                },
                toolset=TOOLSET_WEB,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_DOMAIN,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_CONDITIONAL,
                executor=playwright_adapter.navigate_executor,
                availability_check=_playwright_available,
            )
        )

    if registry.get("browser_click") is None:
        registry.register(
            ToolDefinition(
                name="browser_click",
                description="Click an element by CSS selector or exact text.",
                parameters={
                    "selector": {"type": "string", "required": False},
                    "text": {"type": "string", "required": False},
                    "expect_download": {"type": "string", "required": False},
                },
                toolset=TOOLSET_WEB,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_NO,
                executor=playwright_adapter.click_executor,
                availability_check=_playwright_available,
            )
        )

    if registry.get("browser_type") is None:
        registry.register(
            ToolDefinition(
                name="browser_type",
                description="Fill text into one element by CSS selector.",
                parameters={
                    "selector": {"type": "string", "required": True},
                    "text": {"type": "string", "required": True},
                },
                toolset=TOOLSET_WEB,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_NO,
                executor=playwright_adapter.type_executor,
                availability_check=_playwright_available,
            )
        )

    if registry.get("browser_screenshot") is None:
        registry.register(
            ToolDefinition(
                name="browser_screenshot",
                description=(
                    "Save a screenshot of the current browser page as an artifact "
                    "after browser_navigate and browser_extract have captured text evidence."
                ),
                parameters={
                    "full_page": {
                        "type": "string",
                        "required": False,
                        "description": "Use true to capture the full page.",
                    }
                },
                toolset=TOOLSET_WEB,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_YES,
                executor=playwright_adapter.screenshot_executor,
                availability_check=_playwright_available,
            )
        )

    if registry.get("browser_extract") is None:
        registry.register(
            ToolDefinition(
                name="browser_extract",
                description=(
                    "Extract visible text from the current browser page after "
                    "browser_navigate; then call browser_screenshot to preserve visual evidence."
                ),
                parameters={
                    "selector": {
                        "type": "string",
                        "required": False,
                        "description": "Optional CSS selector; omit to extract the page body.",
                    }
                },
                toolset=TOOLSET_WEB,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_YES,
                executor=playwright_adapter.extract_executor,
                availability_check=_playwright_available,
            )
        )


def _playwright_available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("playwright") is None:
        return (
            False,
            "browser_unavailable: Playwright optional dependency is not installed",
        )
    return True, None


__all__ = ["register_tools"]
