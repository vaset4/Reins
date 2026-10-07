from __future__ import annotations

import approval


def parse_approval_decision(text: str) -> approval.ApprovalDecision | None:
    lowered = text.strip().casefold()
    if lowered in {"/deny", "deny", "n", "no"}:
        return approval.ApprovalDecision.DENY
    if lowered in {"/approve", "/approve once", "approve", "y", "yes"}:
        return approval.ApprovalDecision.ONCE
    if lowered == "/approve task":
        return approval.ApprovalDecision.TASK
    if lowered == "/approve permanent":
        return approval.ApprovalDecision.PERMANENT
    return None


def approval_body(req: approval.ApprovalRequest) -> str:
    return (
        f"{req.message}\ntool: {req.tool}\nrisk: {req.risk}\n"
        f"args: {dict(req.args)}\ntype /approve once, /approve task, or /deny"
    )


__all__ = ["approval_body", "parse_approval_decision"]
