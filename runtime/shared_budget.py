"""父子运行共用派发额度，预留和实际消费各有独立凭据。

作者：xxx
时间：2026-09-15 01:00:00
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition
from typing import Any
from uuid import uuid4

from llm.types import (
    ModelAttemptEvent,
    ModelUsage,
    usage_from_mapping,
    usage_to_mapping,
)
from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.lease import Lease
from runtime.run_facts import RunFactStore


@dataclass(frozen=True, slots=True)
class ModelReservation:
    """一次实际请求的预留信息；attempt_id关联结算，input_tokens是估算，output_tokens是输出限额。"""

    attempt_id: str
    input_tokens: int = 0
    output_tokens: int | None = None
    request_id: str = ""
    minimum_output_tokens: int = 1

    def __post_init__(self) -> None:
        """拒绝无法计量的预留参数；传参：请求估算及额度；返回：无。"""
        if (
            not self.attempt_id
            or self.input_tokens < 0
            or self.minimum_output_tokens < 1
        ):
            raise ValueError("invalid model budget reservation")
        if (
            self.output_tokens is not None
            and self.output_tokens < self.minimum_output_tokens
        ):
            raise ValueError("output reservation is below its required minimum")


@dataclass(frozen=True, slots=True)
class BudgetOwner:
    """用量归属与本地限额；session_id/run_id指向执行者，lease只允许收窄父额度。"""

    session_id: str
    run_id: str
    lease: Lease


def known_tokens(usage: ModelUsage) -> tuple[int, bool, bool]:
    """区分实际零、已知下界与未知；传参：供应商用量；返回：已知token、是否有计量、是否缺项。"""
    total = usage.total_tokens.value
    if total is not None:
        return total, True, False
    components = (usage.input_tokens.value, usage.output_tokens.value)
    return (
        sum(value for value in components if value is not None),
        any(value is not None for value in components),
        any(value is None for value in components),
    )


class SharedRunBudget:
    """同一父运行的唯一账目，所有执行者在同一把锁内预留和结算。"""

    def __init__(self, owner: BudgetOwner, facts: RunFactStore) -> None:
        """绑定父运行和持久事实写者；传参：父身份/额度与事实存储；返回：无。"""
        self.owner, self._facts = owner, facts
        self._condition = Condition()
        self._owners: dict[str, BudgetOwner] = {}
        self._accounts: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, tuple[str, int]] = {}
        self._settled: set[str] = set()
        self._carried = _empty_account()
        self.register(owner)

    @classmethod
    def restore(cls, owner: BudgetOwner, facts: RunFactStore) -> SharedRunBudget:
        """自动接续时从根运行事实继承消费，失联请求保留未知预留；传参：原预算身份与事实；返回：剩余额度。"""
        budget = cls(owner, facts)
        rows = facts.read_run(owner.run_id)
        reservations = {
            str(row["attempt_id"]): row
            for row in rows
            if row.get("event") == "budget:reserved"
        }
        settlements = {
            str(row["attempt_id"]): row
            for row in rows
            if row.get("event") == "budget:settled"
        }
        if not settlements.keys() <= reservations.keys():
            raise ValueError("persisted budget settlement is missing its reservation")
        carried = _empty_account()
        carried["steps_used"] = len(
            {
                row["dispatch_id"]
                for row in rows
                if row.get("event") == "budget:dispatch"
            }
        ) + len(reservations)
        carried["model_attempts"] = len(reservations)
        for identity, reservation in reservations.items():
            if identity not in settlements:
                carried["unknown_usage_attempts"] += 1
                carried["uncertain_tokens_reserved"] += int(
                    reservation["tokens_reserved"]
                )
                continue
            settled = settlements[identity]
            known, measured, incomplete = known_tokens(
                usage_from_mapping(settled["usage"])
            )
            carried["tokens_used"] += known
            carried["has_token_usage"] = carried["has_token_usage"] or measured
            carried["unknown_usage_attempts"] += int(incomplete)
            carried["uncertain_tokens_reserved"] += int(
                settled["uncertain_tokens_reserved"]
            )
        budget._carried = carried
        budget._settled = set(settlements)
        return budget

    def register(self, owner: BudgetOwner) -> None:
        """登记执行者而不复制父余额；传参：身份和本地限额；返回：无，重复身份不能改变额度。"""
        with self._condition:
            previous = self._owners.get(owner.run_id)
            if previous is not None:
                if previous != owner:
                    raise ValueError("budget owner identity or limits changed")
                return
            self._owners[owner.run_id] = owner
            self._accounts[owner.run_id] = _empty_account()

    def reserve_tool(self, run_id: str, *, operation_id: str = "") -> None:
        """记下一次真实工具派发的步数；传参：执行者及原操作身份；返回：无，用量不再拦派发。"""
        with self._condition:
            self._record(
                "budget:dispatch",
                run_id,
                dispatch_id=f"dispatch-{uuid4().hex}",
                operation_id=operation_id,
                kind="tool",
            )
            self._accounts[run_id]["steps_used"] += 1

    def reserve_model(
        self, run_id: str, request: ModelReservation, *, cancellation: CancellationToken
    ) -> None:
        """记下一次模型派发的输入估算与输出上限；传参：执行者、请求及停止信号；返回：无，用量不再拦派发。"""
        with self._condition:
            if (
                request.attempt_id in self._pending
                or request.attempt_id in self._settled
            ):
                raise ValueError("model attempt already reserved")
            # 【运行账目】【派发前停止边界】取消仍在派发前生效，与用量无关
            if cancellation.cancelled:
                raise ExecutionCancelled("cancelled before model dispatch")
            output = request.output_tokens or 0
            reserved = request.input_tokens + output
            self._record(
                "budget:reserved",
                run_id,
                attempt_id=request.attempt_id,
                request_id=request.request_id,
                input_estimate=request.input_tokens,
                output_limit=output,
                tokens_reserved=reserved,
            )
            self._pending[request.attempt_id] = (run_id, reserved)
            account = self._accounts[run_id]
            account["steps_used"] += 1
            account["model_attempts"] += 1

    def settle(self, run_id: str, attempt: ModelAttemptEvent) -> None:
        """按尝试身份结算，失败和取消同样入账；传参：执行者与完成事件；返回：无，未知额度继续显式占用。"""
        if attempt.phase != "finished":
            return
        with self._condition:
            if attempt.attempt_id in self._settled:
                return
            reservation = self._pending.get(attempt.attempt_id)
            if reservation is None or reservation[0] != run_id:
                raise ValueError("model settlement has no matching budget reservation")
            known, measured, incomplete = known_tokens(attempt.usage)
            uncertain = max(0, reservation[1] - known) if incomplete else 0
            self._record(
                "budget:settled",
                run_id,
                attempt_id=attempt.attempt_id,
                request_id=attempt.request_id,
                usage=usage_to_mapping(attempt.usage),
                known_tokens=known,
                uncertain_tokens_reserved=uncertain,
                error=attempt.error.category if attempt.error else None,
            )
            account = self._accounts[run_id]
            account["tokens_used"] += known
            account["has_token_usage"] = account["has_token_usage"] or measured
            account["unknown_usage_attempts"] += int(incomplete)
            account["uncertain_tokens_reserved"] += uncertain
            self._settled.add(attempt.attempt_id)
            del self._pending[attempt.attempt_id]

    def snapshot(self, run_id: str) -> dict[str, object]:
        """导出共同余额和当前执行者消费；传参：执行者；返回：独立视图，不把未采集费用补零。"""
        with self._condition:
            total = self._totals()
            return {
                **total,
                "steps_limit": self.owner.lease.max_steps,
                "tokens_limit": self.owner.lease.max_tokens,
                "tokens_reserved": sum(value[1] for value in self._pending.values()),
                "carried_usage": dict(self._carried),
                "budget_session_id": self.owner.session_id,
                "budget_run_id": self.owner.run_id,
                "owner": {"run_id": run_id, **self._accounts[run_id]},
            }

    def _totals(self) -> dict[str, Any]:
        """汇总唯一账目；传参：无；返回：共同消费与缺失计量。"""
        keys = (
            "steps_used",
            "tokens_used",
            "model_attempts",
            "unknown_usage_attempts",
            "uncertain_tokens_reserved",
        )
        accounts = (self._carried, *self._accounts.values())
        return {
            **{key: sum(account[key] for account in accounts) for key in keys},
            "has_token_usage": any(account["has_token_usage"] for account in accounts),
        }

    def _record(self, event: str, run_id: str, **detail: object) -> None:
        """先提交预算事实再开放真实派发；传参：事件、消费归属与证据；返回：无，持久化失败直接暴露。"""
        owner = self._owners[run_id]
        self._facts.append(
            {
                "event": event,
                "session_id": self.owner.session_id,
                "run_id": self.owner.run_id,
                "owner_session_id": owner.session_id,
                "owner_run_id": owner.run_id,
                **detail,
            }
        )


def _empty_account() -> dict[str, Any]:
    """建立尚未发生消费的账户；传参：无；返回：独立计量字段。"""
    return {
        "steps_used": 0,
        "tokens_used": 0,
        "model_attempts": 0,
        "unknown_usage_attempts": 0,
        "uncertain_tokens_reserved": 0,
        "has_token_usage": False,
    }
