"""【阶段九评估】【并发计量】真实线程及Windows进程不能竞争突破总额度。

作者：xxx
时间：2026-10-01 20:00:00
"""

import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from scripts.eval_stage9_meter import EvaluationBudget, EvaluationLimitExceeded
from tests.scripts.test_eval_stage9 import budget_at

PARTICIPANTS = 4
RESERVATIONS_PER_PARTICIPANT = 5
ATTEMPT_LIMIT = 16


def reserve_batch(path, barrier):
    """同步起跑后竞争同一持久预算；参数：总账/跨执行者屏障；返回：成功和拒绝次数。"""
    budget = EvaluationBudget(Path(path))
    barrier.wait(timeout=15)
    accepted = rejected = 0
    for number in range(RESERVATIONS_PER_PARTICIPANT):
        try:
            budget.reserve(
                {"attempts": 1, "output_tokens": 3}, identity=f"attempt-{number}"
            )
            accepted += 1
        except EvaluationLimitExceeded:
            rejected += 1
    return accepted, rejected


def process_batch(path, barrier, results):
    """把独立进程的竞争结果交还测试；参数：总账/屏障/队列；返回：无。"""
    results.put(reserve_batch(path, barrier))


def assert_budget(path, results):
    """同时核对成功次数和原件总账，拒绝不能消费输出额度；参数：原件/结果；返回：无。"""
    assert sum(result[0] for result in results) == ATTEMPT_LIMIT
    assert (
        sum(result[1] for result in results)
        == PARTICIPANTS * RESERVATIONS_PER_PARTICIPANT - ATTEMPT_LIMIT
    )
    state = json.loads(path.read_text(encoding="utf-8"))
    assert (
        state["attempts"] == ATTEMPT_LIMIT
        and state["output_tokens"] == ATTEMPT_LIMIT * 3
    )


def test_threads_reserve_exactly_sixteen_attempts(tmp_path):
    """前后台线程不能覆盖对方的额度预留；参数：隔离总账；返回：无。"""
    budget = budget_at(tmp_path)
    barrier = Barrier(PARTICIPANTS)
    with ThreadPoolExecutor(max_workers=PARTICIPANTS) as pool:
        futures = [
            pool.submit(reserve_batch, budget.path, barrier)
            for _ in range(PARTICIPANTS)
        ]
        results = [future.result(timeout=20) for future in futures]
    assert_budget(budget.path, results)


def test_processes_reserve_exactly_sixteen_attempts(tmp_path):
    """独立Windows执行进程共用同一系统锁，拒绝第17次派发；参数：隔离总账；返回：无。"""
    budget = budget_at(tmp_path)
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(PARTICIPANTS), context.Queue()
    processes = [
        context.Process(target=process_batch, args=(str(budget.path), barrier, results))
        for _ in range(PARTICIPANTS)
    ]
    try:
        for process in processes:
            process.start()
        observed = [results.get(timeout=25) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert_budget(budget.path, observed)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
