from __future__ import annotations

import asyncio
import pytest
import threading
from pathlib import Path

from approval import ApprovalDecision, ApprovalRequest, ApprovalUnavailable
from approval.bridge import ApprovalBridge, register_tui_bridge, request_via_tui
from runtime.lease import from_trigger


def test_bridge_unlocks_sync_caller() -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
    thread.start()
    try:
        bridge = ApprovalBridge(loop, _present_task)
        decision = bridge.request(_request())
        assert decision is ApprovalDecision.TASK
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_request_via_tui_uses_registered_bridge() -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
    thread.start()
    try:
        register_tui_bridge(ApprovalBridge(loop, _present_once))
        result = request_via_tui(_request())
        assert result is ApprovalDecision.ONCE
    finally:
        register_tui_bridge(None)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_request_via_tui_returns_none_without_bridge() -> None:
    register_tui_bridge(None)
    assert request_via_tui(_request()) is None


def test_request_via_tui_logs_presenter_failure_and_fails_closed(caplog) -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
    thread.start()
    try:
        register_tui_bridge(ApprovalBridge(loop, _present_raises))
        with pytest.raises(ApprovalUnavailable, match="presenter failed"):
            request_via_tui(_request())
    finally:
        register_tui_bridge(None)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


async def _present_task(_req: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision.TASK


async def _present_once(_req: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision.ONCE


async def _present_raises(_req: ApprovalRequest) -> ApprovalDecision:
    raise RuntimeError("presenter failed")


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        tool="file_write",
        args={"path": "out.txt"},
        risk="confirm",
        lease=from_trigger("user", task_id="task-1"),
        data_root=Path("."),
        message="write file",
    )
