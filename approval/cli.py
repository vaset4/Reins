from __future__ import annotations

from approval import ApprovalDecision, ApprovalRequest
from approval.batch_types import (
    ApprovalBatch,
    BatchDecision,
    format_batch,
    parse_batch_decision,
)


def cli_request_approval(req: ApprovalRequest) -> ApprovalDecision:
    print(req.message or f"{req.tool} requires approval")
    if req.resource is not None:
        print(f"授权动作及精确资源：{req.resource.action} {req.resource.target}")
    print("1. once")
    print("2. task")
    print("3. permanent")
    print("4. deny")
    choices = {
        "1": ApprovalDecision.ONCE,
        "once": ApprovalDecision.ONCE,
        "2": ApprovalDecision.TASK,
        "task": ApprovalDecision.TASK,
        "3": ApprovalDecision.PERMANENT,
        "permanent": ApprovalDecision.PERMANENT,
        "4": ApprovalDecision.DENY,
        "deny": ApprovalDecision.DENY,
    }
    while True:
        value = input("approval> ").strip().lower()
        decision = choices.get(value)
        if decision is not None:
            return decision
        print("choose 1, 2, 3, or 4")


def cli_request_batch(batch: ApprovalBatch) -> BatchDecision:
    """集中展示全部待决定项后接收一次完整提交；传参：批次；返回：逐项决定或取消。"""
    print(format_batch(batch))
    while True:
        try:
            return parse_batch_decision(batch, input("approval> ").strip())
        except ValueError as exc:
            print(str(exc))
