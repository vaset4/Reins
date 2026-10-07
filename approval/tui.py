from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from approval import ApprovalDecision, ApprovalRequest, ApprovalUnavailable
from approval.bridge import request_via_tui
from approval.cli import cli_request_approval

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.screen import ModalScreen as _ModalScreenBase
    from textual.widgets import Button as _ButtonBase
else:
    ComposeResult = Any
    _ModalScreenBase = object
    _ButtonBase = object


class ApprovalModal(_ModalScreenBase):  # type: ignore[type-arg]
    BINDINGS = [
        ("1", "choose_once", "Once"),
        ("2", "choose_task", "Task"),
        ("3", "choose_permanent", "Permanent"),
        ("4", "choose_deny", "Deny"),
        ("escape", "choose_cancel", "Cancel"),
    ]

    def __init__(self, req: ApprovalRequest) -> None:
        super().__init__()
        self._req = req

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical
        from textual.widgets import Button, Pretty, Static

        yield Vertical(
            Static("Approval Required", id="approval-title"),
            Static(self._req.message or f"tool={self._req.tool}", id="approval-msg"),
            Static(f"tool: {self._req.tool}"),
            Static(f"risk: {self._req.risk}"),
            Static(f"task: {self._req.lease.task_id}"),
            Static(
                f"resource: {self._req.resource}"
                if self._req.resource is not None
                else "scope: exact arguments"
            ),
            Pretty(dict(self._req.args), id="approval-args"),
            Button("Once (1)", id="once", variant="primary"),
            Button("Task (2)", id="task"),
            Button("Permanent (3)", id="permanent"),
            Button("Deny (4 / Esc)", id="deny", variant="error"),
            id="approval-dialog",
        )

    def on_button_pressed(self, event: Any) -> None:
        button_id = getattr(getattr(event, "button", None), "id", None)
        if button_id == "once":
            self.dismiss(ApprovalDecision.ONCE)
        elif button_id == "task":
            self.dismiss(ApprovalDecision.TASK)
        elif button_id == "permanent":
            self.dismiss(ApprovalDecision.PERMANENT)
        elif button_id == "deny":
            self.dismiss(ApprovalDecision.DENY)

    def action_choose_once(self) -> None:
        self.dismiss(ApprovalDecision.ONCE)

    def action_choose_task(self) -> None:
        self.dismiss(ApprovalDecision.TASK)

    def action_choose_permanent(self) -> None:
        self.dismiss(ApprovalDecision.PERMANENT)

    def action_choose_deny(self) -> None:
        self.dismiss(ApprovalDecision.DENY)

    def action_choose_cancel(self) -> None:
        """关闭交互只记录取消，不伪造用户拒绝；传参：无；返回：无。"""
        self.dismiss(ApprovalDecision.CANCELLED)


async def present_approval_modal(app: Any, req: ApprovalRequest) -> ApprovalDecision:
    try:
        # Lazy import to keep this module importable when textual extra is not installed.
        from textual.app import App  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ApprovalUnavailable("approval UI dependency is unavailable") from exc

    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[ApprovalDecision] = loop.create_future()

    def _on_result(result: ApprovalDecision | None) -> None:
        if result_future.done():
            return
        if isinstance(result, ApprovalDecision):
            result_future.set_result(result)
            return
        result_future.set_result(ApprovalDecision.CANCELLED)

    app.push_screen(ApprovalModal(req), _on_result)
    return await result_future


def tui_request_approval(req: ApprovalRequest) -> ApprovalDecision:
    decision = request_via_tui(req)
    if decision is not None:
        return decision
    return cli_request_approval(req)
