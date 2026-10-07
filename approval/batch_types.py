"""整批审批的不可变展示与逐项决定。

作者：xxx
时间：2026-09-24 12:00:00
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from approval import ApprovalDecision, ApprovalRequest, ApprovalUnavailable


@dataclass(frozen=True, slots=True)
class ApprovalChoice:
    """用户对一个已展示动作选定的复用范围，directory 仅指展示的父目录。"""

    operation_id: str
    decision: ApprovalDecision
    resource_scope: str = "exact"


@dataclass(frozen=True, slots=True)
class ApprovalBatch:
    """一个交互中展示的完整待决定集合，未选择项目不能被推断为批准。"""

    batch_id: str
    requests: tuple[ApprovalRequest, ...]

    def __post_init__(self) -> None:
        """冻结展示集合，不共享调用方的可变列表；传参：构造字段；返回：无。"""
        object.__setattr__(self, "requests", tuple(self.requests))


@dataclass(frozen=True, slots=True)
class BatchDecision:
    """整批提交的用户动作；取消时不携带逐项授权。"""

    choices: tuple[ApprovalChoice, ...] = ()
    cancelled: bool = False
    action_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        """冻结一次提交的全部项目；传参：构造字段；返回：无。"""
        object.__setattr__(self, "choices", tuple(self.choices))


def validate_batch_decision(batch: ApprovalBatch, decision: BatchDecision) -> None:
    """完整校验界面回执后才允许保存任何授权；传参：原批次与决定；返回：无，错误抛设施异常。"""
    if (
        not isinstance(decision, BatchDecision)
        or not decision.action_id.strip()
        or type(decision.cancelled) is not bool
    ):
        raise ApprovalUnavailable("approval batch response is invalid")
    if decision.cancelled:
        if decision.choices:
            raise ApprovalUnavailable("cancelled approval batch cannot contain grants")
        return
    expected = {request.operation_id for request in batch.requests}
    actual = [choice.operation_id for choice in decision.choices]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ApprovalUnavailable(
            "approval batch response has missing, duplicate or unknown operations"
        )
    for choice in decision.choices:
        if (
            not isinstance(choice.decision, ApprovalDecision)
            or choice.decision is ApprovalDecision.CANCELLED
        ):
            raise ApprovalUnavailable(
                "each operation requires an explicit grant scope or denial"
            )
        if choice.resource_scope not in {"exact", "directory"}:
            raise ApprovalUnavailable("approval resource scope is invalid")
        request = next(
            item for item in batch.requests if item.operation_id == choice.operation_id
        )
        if request.force_confirmation and choice.decision not in {
            ApprovalDecision.ONCE,
            ApprovalDecision.DENY,
        }:
            raise ApprovalUnavailable(
                "protected replacement requires a single-operation confirmation"
            )
        if choice.resource_scope == "directory" and (
            request.resource is None
            or request.resource.kind != "file"
            or choice.decision is ApprovalDecision.ONCE
        ):
            raise ApprovalUnavailable(
                "directory approval requires a displayed file and reusable scope"
            )


def parse_batch_decision(batch: ApprovalBatch, text: str) -> BatchDecision:
    """解析已展示项目的数字选择，一次提交全部决定；传参：批次和输入；返回：完整回执。"""
    if text.strip() == "cancel":
        return BatchDecision(cancelled=True)
    choices = []
    for selection in text.split():
        number, separator, value = selection.partition("=")
        if (
            not separator
            or not number.isdecimal()
            or not 1 <= int(number) <= len(batch.requests)
        ):
            raise ValueError(
                "逐项输入 编号=once|session|task|permanent|deny，例如 1=once 2=deny；cancel 取消整批"
            )
        scope, _, resource_scope = value.partition(":")
        choices.append(
            ApprovalChoice(
                batch.requests[int(number) - 1].operation_id,
                ApprovalDecision(scope),
                resource_scope or "exact",
            )
        )
    result = BatchDecision(tuple(choices))
    try:
        validate_batch_decision(batch, result)
    except ApprovalUnavailable as exc:
        raise ValueError(str(exc)) from exc
    return result


def format_batch(batch: ApprovalBatch) -> str:
    """展示本次每个动作和可选范围，不预选允许；传参：固定批次；返回：用户可读的整批说明。"""
    from pathlib import Path

    lines = ["请逐项选择本批操作的授权范围："]
    for index, request in enumerate(batch.requests, 1):
        resource = request.resource
        target = resource.target if resource is not None else str(dict(request.args))
        lines.append(f"[{index}] {request.tool} · {target}")
        if (
            resource is not None
            and resource.kind == "file"
            and not request.force_confirmation
        ):
            lines.append(
                f"    可选目录范围：{Path(resource.target).parent}（在范围后加 :directory）"
            )
        if request.force_confirmation:
            lines.append("    此操作必须单次明确确认，仅可选择 once 或 deny")
    lines.extend(
        [
            "once=仅这次；session=本会话；task=本任务；permanent=永久；deny=拒绝",
            "一次提交所有项目，例如：1=once 2=deny；cancel 取消整批。没有选择的项目不会默认批准。",
        ]
    )
    return "\n".join(lines)
