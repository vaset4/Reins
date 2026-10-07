"""验证并发执行者使用一份父预算及可追溯结算。

作者：xxx
时间：2026-09-15 01:00:00
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from llm.provider_result import reported
from llm.types import ModelAttemptEvent, ModelUsage
from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.shared_budget import BudgetOwner, ModelReservation, SharedRunBudget


def family(tmp_path, *, steps=6, tokens=100):
    """构造真实文件账目与两个消费归属；传参：目录和总额度；返回：预算。"""
    lease = from_trigger("user", max_steps=steps, max_tokens=tokens)
    budget = SharedRunBudget(
        BudgetOwner("session-parent", "run-parent", lease), RunFactStore(tmp_path)
    )
    budget.register(
        BudgetOwner("session-child", "run-child", replace(lease, trigger="delegate"))
    )
    return budget


def finished(identity, usage):
    """构造已观察到的实际尝试证据；传参：预留ID和用量；返回：完成事件。"""
    return ModelAttemptEvent(
        "finished",
        "request-shared",
        identity,
        0,
        "test",
        "model",
        "2026-09-14T17:00:00Z",
        usage=usage,
    )


def test_restart_preserves_consumption_and_unknown_reservations(tmp_path):
    """后台自动接续不能获得新总预算，断线用量保持未知；传参：隔离目录；返回：无。"""
    budget = family(tmp_path, steps=5, tokens=100)
    budget.reserve_tool("run-child", operation_id="written-once")
    budget.reserve_model(
        "run-parent",
        ModelReservation("settled", 10, 20),
        cancellation=CancellationToken(),
    )
    budget.settle(
        "run-parent", finished("settled", ModelUsage(total_tokens=reported(15)))
    )
    budget.reserve_model(
        "run-child", ModelReservation("lost", 10, 20), cancellation=CancellationToken()
    )
    restored = SharedRunBudget.restore(budget.owner, RunFactStore(tmp_path))
    restored.register(BudgetOwner("session-parent", "run-resumed", budget.owner.lease))
    snapshot = restored.snapshot("run-resumed")
    assert snapshot["steps_used"] == 3 and snapshot["tokens_used"] == 15
    assert (
        snapshot["unknown_usage_attempts"] == 1
        and snapshot["uncertain_tokens_reserved"] == 30
    )
    restored.reserve_tool("run-resumed", operation_id="next-operation")
    again = SharedRunBudget.restore(budget.owner, RunFactStore(tmp_path))
    assert again.snapshot("run-parent")["steps_used"] == 4


def test_restart_keeps_unsettled_reservation_recorded(tmp_path):
    """重启后未结算的预留仍留在账上，但不再拦截后续消费；传参：隔离目录；返回：无。"""
    budget = family(tmp_path, tokens=100)
    budget.reserve_model(
        "run-parent", ModelReservation("lost", 20, 80), cancellation=CancellationToken()
    )
    restored = SharedRunBudget.restore(budget.owner, RunFactStore(tmp_path))
    snapshot = restored.snapshot("run-parent")
    assert snapshot["unknown_usage_attempts"] == 1
    assert snapshot["uncertain_tokens_reserved"] == 100
    # 预留占用只作记录，下一次派发不再被它挡住
    assert (
        restored.reserve_model(
            "run-parent",
            ModelReservation("new", 1, 1),
            cancellation=CancellationToken(),
        )
        is None
    )
    assert restored.snapshot("run-parent")["steps_used"] == 2


def test_parallel_children_dispatches_are_recorded_against_their_owner(tmp_path):
    """并发派发全部记账且归属可追溯，互不拦截；传参：目录；返回：无。"""
    budget, barrier = family(tmp_path, steps=3), Barrier(2)

    def spend(owner):
        """并发写入同一父账目；传参：执行者；返回：成功派发次数。"""
        barrier.wait()
        for _ in range(3):
            budget.reserve_tool(owner)
        return 3

    with ThreadPoolExecutor(2) as pool:
        counts = list(pool.map(spend, ("run-parent", "run-child")))
    assert sum(counts) == 6
    assert budget.snapshot("run-parent")["steps_used"] == 6
    rows = RunFactStore(tmp_path).read_run("run-parent")
    assert len(rows) == 6
    assert {row["owner_run_id"] for row in rows} == {"run-parent", "run-child"}


def test_reserved_tokens_are_recorded_until_real_settlement(tmp_path):
    """预留金额先上账，结算后换成真实用量，不做可用额度扣减；传参：目录；返回：无。"""
    budget = family(tmp_path)
    assert (
        budget.reserve_model(
            "run-parent",
            ModelReservation("first", 20, 80),
            cancellation=CancellationToken(),
        )
        is None
    )
    assert budget.snapshot("run-parent")["tokens_reserved"] == 100

    # 第二个执行者不再等待首个请求结算，直接记账
    assert (
        budget.reserve_model(
            "run-child",
            ModelReservation("second", 10, 80),
            cancellation=CancellationToken(),
        )
        is None
    )
    attempt = finished("first", ModelUsage(total_tokens=reported(35)))
    budget.settle("run-parent", attempt)
    budget.settle("run-parent", attempt)
    snapshot = budget.snapshot("run-child")
    assert snapshot["tokens_used"] == 35
    assert snapshot["tokens_reserved"] == 90
    assert snapshot["steps_used"] == 2


def test_missing_usage_is_recorded_but_not_treated_as_zero(tmp_path):
    """用量缺失时保留预留和未知计数；传参：目录；返回：无。"""
    budget = family(tmp_path)
    budget.reserve_model(
        "run-child",
        ModelReservation("partial", 20, 30),
        cancellation=CancellationToken(),
    )
    budget.settle(
        "run-child", finished("partial", ModelUsage(input_tokens=reported(12)))
    )
    snapshot = budget.snapshot("run-parent")
    assert snapshot["tokens_used"] == 12
    assert snapshot["uncertain_tokens_reserved"] == 38
    assert snapshot["unknown_usage_attempts"] == 1
    # 未知用量只记录，不再折算成可用额度扣减
    assert (
        budget.reserve_model(
            "run-parent",
            ModelReservation("next", 10, 80),
            cancellation=CancellationToken(),
        )
        is None
    )


def test_cancelled_signal_refuses_dispatch_before_recording(tmp_path):
    """停止信号仍在派发前生效，被拒的派发不留预留；传参：目录；返回：无。"""
    budget, parent = family(tmp_path), CancellationToken()
    child = CancellationToken(parent)
    parent.cancel()
    with pytest.raises(ExecutionCancelled):
        budget.reserve_model(
            "run-child", ModelReservation("blocked", 10, 20), cancellation=child
        )
    snapshot = budget.snapshot("run-child")
    assert snapshot["steps_used"] == 0
    assert snapshot["tokens_reserved"] == 0
