from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine

from approval import ApprovalDecision, ApprovalRequest, ApprovalUnavailable

_Presenter = Callable[[ApprovalRequest], Coroutine[object, object, ApprovalDecision]]
logger = logging.getLogger(__name__)


class ApprovalBridge:
    def __init__(self, loop: asyncio.AbstractEventLoop, presenter: _Presenter) -> None:
        self._loop = loop
        self._presenter = presenter

    def request(self, req: ApprovalRequest, timeout: float = 300.0) -> ApprovalDecision:
        """等待界面决定并取消已失效交互；传参：请求/等待时长；返回：决定，故障抛错。"""
        if self._loop.is_closed() or not self._loop.is_running():
            raise ApprovalUnavailable("approval event loop is not running")
        coro = self._presenter(req)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except BaseException:
            future.cancel()
            raise


_TUI_BRIDGE: ApprovalBridge | None = None
_TUI_BRIDGE_LOCK = threading.Lock()


def register_tui_bridge(bridge: ApprovalBridge | None) -> None:
    global _TUI_BRIDGE
    with _TUI_BRIDGE_LOCK:
        _TUI_BRIDGE = bridge


def get_tui_bridge() -> ApprovalBridge | None:
    with _TUI_BRIDGE_LOCK:
        return _TUI_BRIDGE


def request_via_tui(req: ApprovalRequest) -> ApprovalDecision | None:
    bridge = get_tui_bridge()
    if bridge is None:
        return None
    try:
        return bridge.request(req)
    except Exception as exc:
        raise ApprovalUnavailable(
            f"TUI approval bridge failed for {req.tool}: {exc}"
        ) from exc
