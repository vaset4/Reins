"""Console bootstrap for the Reins REPL.

Two responsibilities:

1. Force the Windows console into UTF-8 mode so Chinese text and box-drawing
   characters render correctly under cmd / PowerShell. This mirrors EvoHarness
   `harness/console.py::enable_utf8_console` because the same bug is the same
   on every Windows machine.
2. Provide a single shared `rich.Console` instance for the renderer, slash
   commands, and dashboard so styling stays consistent and tests can capture
   output through one entry point.

Linux / macOS get a no-op for the UTF-8 helper since their default consoles
already speak UTF-8. The V2.1 design (§11 永远不做) restricts MVP-0/1 to
Windows, but we keep the no-op branch so unit tests can run in CI on any
platform without raising.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TextIO

from rich.console import Console


_CONSOLE: Console | None = None


def _reconfigure_stream(stream: Any, *, errors: str) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(encoding="utf-8", errors=errors)
    except Exception:
        # Reconfigure is best-effort. If it fails (e.g., already closed
        # stream during pytest capture), the renderer still works because
        # rich does its own encoding sniffing.
        pass


def enable_utf8_console() -> None:
    """Switch the Windows console code page to UTF-8 (65001) and reconfigure
    stdio streams to use UTF-8 with a forgiving error mode. No-op elsewhere."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    _reconfigure_stream(sys.stdin, errors="replace")
    _reconfigure_stream(sys.stdout, errors="replace")
    _reconfigure_stream(sys.stderr, errors="backslashreplace")


def get_console() -> Console:
    """Return the shared Console instance, creating it on first call."""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(soft_wrap=False, highlight=False)
    return _CONSOLE


def reset_console_for_tests(console: Console | None = None) -> None:
    """Replace (or clear) the shared console. Tests use this to inject a
    capturing console; production code never calls it."""
    global _CONSOLE
    _CONSOLE = console


@contextmanager
def capture_console(
    width: int = 120, *, file: TextIO | None = None
) -> Iterator[Console]:
    """Temporarily capture helpers that write to the shared console."""
    global _CONSOLE
    previous = _CONSOLE
    recorder = Console(
        file=file,
        record=True,
        width=width,
        force_terminal=False,
        color_system=None,
    )
    _CONSOLE = recorder
    try:
        yield recorder
    finally:
        _CONSOLE = previous


__all__ = [
    "capture_console",
    "enable_utf8_console",
    "get_console",
    "reset_console_for_tests",
]
