from __future__ import annotations

import importlib.util
import shutil

from tools.tool_registry import (
    IDEMPOTENT_CONDITIONAL,
    IDEMPOTENT_YES,
    TARGET_SCOPE_LOGICAL,
    TOOL_RISK_CONFIRM,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    ToolDefinition,
    ToolRegistry,
)
from tools.vision import ocr as ocr_tool
from tools.vision import redact as redact_tool
from tools.vision import screenshot as screenshot_tool

TOOLSET_VISION = "vision"


def register_tools(registry: ToolRegistry) -> None:
    if registry.get("screenshot") is None:
        available, _reason = _screenshot_available()
        registry.register(
            ToolDefinition(
                name="screenshot",
                description="Capture the Windows desktop and save it as a screenshot artifact.",
                parameters={"monitor": {"type": "string", "required": False}},
                toolset=TOOLSET_VISION,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=available,
                idempotent=IDEMPOTENT_YES,
                executor=screenshot_tool.screenshot_executor,
                availability_check=_screenshot_available,
            )
        )

    if registry.get("ocr") is None:
        available, _reason = _ocr_available()
        registry.register(
            ToolDefinition(
                name="ocr",
                description="Run local Tesseract OCR on a screenshot artifact.",
                parameters={
                    "artifact_id": {"type": "string", "required": True},
                    "lang": {"type": "string", "required": False},
                },
                toolset=TOOLSET_VISION,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=available,
                idempotent=IDEMPOTENT_YES,
                executor=ocr_tool.ocr_executor,
                availability_check=_ocr_available,
            )
        )

    if registry.get("redact") is None:
        available, _reason = _redact_available()
        registry.register(
            ToolDefinition(
                name="redact",
                description="Create a redacted copy of a screenshot artifact.",
                parameters={
                    "artifact_id": {"type": "string", "required": True},
                    "regions": {"type": "array", "required": False},
                    "lang": {"type": "string", "required": False},
                },
                toolset=TOOLSET_VISION,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=available,
                idempotent=IDEMPOTENT_CONDITIONAL,
                executor=redact_tool.redact_executor,
                availability_check=_redact_available,
            )
        )


def _screenshot_available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("mss") is None:
        return False, "mss optional dependency is not installed"
    if importlib.util.find_spec("mss.tools") is None:
        return False, "mss tools module is not available"
    return True, None


def _ocr_available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("pytesseract") is None:
        return False, "pytesseract optional dependency is not installed"
    if shutil.which("tesseract") is None:
        return False, "tesseract binary is not installed"
    return True, None


def _redact_available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("PIL") is None:
        return False, "Pillow optional dependency is not installed"
    return True, None


__all__ = ["register_tools"]
