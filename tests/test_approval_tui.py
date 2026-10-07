from __future__ import annotations

from pathlib import Path

import pytest

from approval import ApprovalDecision, ApprovalRequest
from approval.tui import ApprovalModal, present_approval_modal, tui_request_approval
from runtime.lease import from_trigger


def test_tui_request_approval_prefers_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "approval.tui.request_via_tui", lambda _req: ApprovalDecision.TASK
    )
    monkeypatch.setattr(
        "approval.tui.cli_request_approval", lambda _req: ApprovalDecision.DENY
    )

    assert tui_request_approval(_request()) is ApprovalDecision.TASK


def test_tui_request_approval_falls_back_to_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("approval.tui.request_via_tui", lambda _req: None)
    monkeypatch.setattr(
        "approval.tui.cli_request_approval", lambda _req: ApprovalDecision.PERMANENT
    )

    assert tui_request_approval(_request()) is ApprovalDecision.PERMANENT


@pytest.mark.asyncio
async def test_present_approval_modal_returns_dismiss_result() -> None:
    req = _request()
    captured: dict[str, object] = {}

    class FakeApp:
        def push_screen(self, modal: object, callback: object) -> None:
            captured["modal"] = modal
            if callable(callback):
                callback(ApprovalDecision.ONCE)

    result = await present_approval_modal(FakeApp(), req)
    assert result is ApprovalDecision.ONCE
    assert isinstance(captured["modal"], ApprovalModal)


def test_approval_modal_dismiss_actions() -> None:
    modal = ApprovalModal(_request())
    values: list[ApprovalDecision] = []

    def _dismiss(value: ApprovalDecision | None) -> None:
        if isinstance(value, ApprovalDecision):
            values.append(value)

    setattr(modal, "dismiss", _dismiss)

    modal.action_choose_once()
    modal.action_choose_task()
    modal.action_choose_permanent()
    modal.action_choose_deny()

    assert values == [
        ApprovalDecision.ONCE,
        ApprovalDecision.TASK,
        ApprovalDecision.PERMANENT,
        ApprovalDecision.DENY,
    ]


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        tool="file_write",
        args={"path": "out.txt"},
        risk="confirm",
        lease=from_trigger("user", task_id="task-1"),
        data_root=Path("."),
        message="write file",
    )
