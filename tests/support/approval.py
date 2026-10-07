"""将测试中的逐项决定接入生产整批审批边界。

作者：xxx
时间：2026-09-24 12:00:00
"""

from collections.abc import Callable

from pytest import MonkeyPatch

from approval import ApprovalDecision, ApprovalRequest
from approval.batch_types import ApprovalBatch, ApprovalChoice, BatchDecision


def install_approval(
    monkeypatch: MonkeyPatch, decide: Callable[[ApprovalRequest], ApprovalDecision]
) -> None:
    """只替换用户决定来源，保留整批验证和真实派发；传参：替换器及逐项策略；返回：无。"""

    def batch_decision(batch: ApprovalBatch) -> BatchDecision:
        """为每个已展示操作独立取得选择；传参：实际批次；返回：完整回执。"""
        choices = tuple(
            ApprovalChoice(request.operation_id, decide(request))
            for request in batch.requests
        )
        if any(choice.decision is ApprovalDecision.CANCELLED for choice in choices):
            return BatchDecision(cancelled=True)
        return BatchDecision(choices)

    monkeypatch.setattr("approval._backend", decide)
    monkeypatch.setattr("approval.batch._batch_backend", batch_decision)
