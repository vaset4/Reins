from __future__ import annotations

import importlib
import importlib.util

from tools.tool_registry import (
    IDEMPOTENT_CONDITIONAL,
    IDEMPOTENT_YES,
    TARGET_SCOPE_LOGICAL,
    TOOL_RISK_CONFIRM,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_AGENT,
    ToolDefinition,
    ToolRegistry,
)
from tools.types import ToolError, ToolErrorCategory


def register_tools(registry: ToolRegistry) -> None:
    if registry.get("clipboard_read") is None:
        available, _reason = _pyperclip_available()
        registry.register(
            ToolDefinition(
                name="clipboard_read",
                description="Read plain text from the Windows clipboard.",
                parameters={},
                toolset=TOOLSET_AGENT,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=available,
                idempotent=IDEMPOTENT_YES,
                executor=clipboard_read_executor,
                availability_check=_pyperclip_available,
            )
        )

    if registry.get("clipboard_write") is None:
        available, _reason = _pyperclip_available()
        registry.register(
            ToolDefinition(
                name="clipboard_write",
                description="Write plain text to the Windows clipboard.",
                parameters={"text": {"type": "string", "required": True}},
                toolset=TOOLSET_AGENT,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=available,
                idempotent=IDEMPOTENT_CONDITIONAL,
                executor=clipboard_write_executor,
                availability_check=_pyperclip_available,
            )
        )


def clipboard_read() -> dict[str, str]:
    pyperclip = importlib.import_module("pyperclip")
    paste = getattr(pyperclip, "paste", None)
    if not callable(paste):
        raise RuntimeError("pyperclip paste missing")
    return {"text": str(paste())}


def clipboard_write(text: str) -> dict[str, object]:
    pyperclip = importlib.import_module("pyperclip")
    copy = getattr(pyperclip, "copy", None)
    if not callable(copy):
        raise RuntimeError("pyperclip copy missing")
    copy(text)
    return {"written": True, "chars": len(text)}


def clipboard_read_executor(_args: dict[str, object]) -> object:
    try:
        return clipboard_read()
    except Exception as exc:
        return ToolError(ToolErrorCategory.UNKNOWN, str(exc), retryable=False)


def clipboard_write_executor(args: dict[str, object]) -> object:
    try:
        return clipboard_write(str(args.get("text", "")))
    except Exception as exc:
        return ToolError(ToolErrorCategory.UNKNOWN, str(exc), retryable=False)


def _pyperclip_available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("pyperclip") is None:
        return False, "pyperclip optional dependency is not installed"
    return True, None


__all__ = [
    "clipboard_read",
    "clipboard_read_executor",
    "clipboard_write",
    "clipboard_write_executor",
    "register_tools",
]
