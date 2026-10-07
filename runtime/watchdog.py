from __future__ import annotations

import importlib
import json
import logging
import time
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from functools import partial
from threading import RLock, Thread
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, cast

from llm.retry_utils import retry_delays
from llm.types import TokenUsage, ModelAttemptEvent
from runtime.lease import Lease
from runtime.cancellation import CancellationToken
from runtime.shared_budget import BudgetOwner, ModelReservation, SharedRunBudget
from tools.types import ToolError, ToolErrorCategory

T = TypeVar("T")

_RETRYABLE_CATEGORIES = frozenset({"transport", "timeout", "rate_limited"})
_LOG = logging.getLogger(__name__)
_CANCEL_POLL_SECONDS = 0.05
_BACKEND_STOP_GRACE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class WatchdogDecision:
    paused: bool
    reason: str = ""
    message: str = ""


@dataclass(slots=True)
class Watchdog:
    lease: Lease
    task_id: str = ""
    data_root: Path | str | None = None
    segment_id: str = ""
    tool_timeout_seconds: float = 30.0
    steps_taken: int = 0
    tokens_used: int = 0
    consecutive_llm_failures: int = 0
    tool_failures: dict[str, int] = field(default_factory=dict)
    _pause_requested: bool = False
    _paused_reason: str = ""
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    model_attempts: int = 0
    unknown_usage_attempts: int = 0
    has_token_usage: bool = False
    _settled_attempts: set[str] = field(default_factory=set)
    _budget_lock: RLock = field(default_factory=RLock, repr=False)
    shared_budget: SharedRunBudget | None = None
    budget_owner: BudgetOwner | None = None

    def __post_init__(self) -> None:
        """把当前执行者接到父运行账目；传参：构造字段；返回：无，缺失归属明确报错。"""
        if self.shared_budget is not None:
            if self.budget_owner is None:
                raise ValueError("shared budget requires a run owner")
            self.shared_budget.register(self.budget_owner)

    def tick(
        self,
        *,
        steps_taken: int | None = None,
        tokens_used: int | None = None,
        token_usage: TokenUsage | None = None,
    ) -> WatchdogDecision:
        if self._paused_reason:
            return WatchdogDecision(True, self._paused_reason)
        if self._pause_requested:
            return self._pause("manual_pause", "pause requested")
        if steps_taken is None:
            self.steps_taken += 1
        else:
            self.steps_taken = steps_taken
        if tokens_used is not None:
            self.tokens_used = tokens_used
            self.has_token_usage = True
        if token_usage is not None:
            self.tokens_used += _token_total(token_usage)
            self.has_token_usage = True
        return WatchdogDecision(False)

    def reserve_tool_step(self, *, operation_id: str = "") -> WatchdogDecision:
        """在真实工具调用开始前记下一个 step

        作者：LKX
        时间：2026-08-16 00:00:00
        传参：operation_id 为本次工具操作身份
        返回：WatchdogDecision；只有手动暂停给出 paused=True，用量本身不拦派发
        """
        with self._budget_lock:
            if self.shared_budget is not None and self.budget_owner is not None:
                self.shared_budget.reserve_tool(
                    self.budget_owner.run_id, operation_id=operation_id
                )
            return self.tick(steps_taken=self.steps_taken + 1)

    def reserve_model_attempt(self, request: ModelReservation | None = None) -> None:
        """记下每次真实 Provider 派发的用量归属；传参：本次尝试的预留信息；返回：无，用量不拦派发。"""
        with self._budget_lock:
            if self.shared_budget is not None and self.budget_owner is not None:
                if request is None:
                    raise ValueError(
                        "shared model budget requires attempt identity and input estimate"
                    )
                self.shared_budget.reserve_model(
                    self.budget_owner.run_id, request, cancellation=self.cancellation
                )
            self.steps_taken += 1
            self.model_attempts += 1

    def budget_evidence(self) -> dict[str, object]:
        """提供主请求和辅助摘要共用的实时预算视图；传参：无；返回：已知消费、未知记录及额度。"""
        with self._budget_lock:
            if self.shared_budget is not None and self.budget_owner is not None:
                return self.shared_budget.snapshot(self.budget_owner.run_id)
            return {
                "steps_used": self.steps_taken,
                "steps_limit": self.lease.max_steps,
                "tokens_used": self.tokens_used,
                "tokens_limit": self.lease.max_tokens,
                "has_token_usage": self.has_token_usage or self.tokens_used > 0,
                "unknown_usage_attempts": self.unknown_usage_attempts,
                "model_attempts": self.model_attempts,
            }

    def settle_model_attempt(self, attempt: ModelAttemptEvent) -> None:
        """结算每次尝试的已知消耗，未知部分保持独立计数；传参：已完成尝试；返回：无。"""
        with self._budget_lock:
            if (
                attempt.phase != "finished"
                or attempt.attempt_id in self._settled_attempts
            ):
                return
            if self.shared_budget is not None and self.budget_owner is not None:
                self.shared_budget.settle(self.budget_owner.run_id, attempt)
            self._settled_attempts.add(attempt.attempt_id)
            usage = attempt.usage
            known = usage.total_tokens.value
            components = (usage.input_tokens.value, usage.output_tokens.value)
            self.has_token_usage = (
                self.has_token_usage
                or known is not None
                or any(value is not None for value in components)
            )
            if known is None:
                known = sum(value for value in components if value is not None)
                if any(value is None for value in components):
                    self.unknown_usage_attempts += 1
            self.tokens_used += known

    def request_pause(self) -> None:
        self._pause_requested = True
        self.cancellation.cancel("manual_pause")

    def register_pause_hotkey(self, shortcut: str = "ctrl+alt+p") -> bool:
        try:
            keyboard = importlib.import_module("keyboard")
            add_hotkey = getattr(keyboard, "add_hotkey", None)
            if not callable(add_hotkey):
                _LOG.error("watchdog pause hotkey unavailable: add_hotkey missing")
                return False
            add_hotkey(shortcut, self.request_pause)
        except Exception as exc:
            _LOG.error("watchdog pause hotkey registration failed: %s", exc)
            return False
        return True

    def run_tool_with_timeout(
        self,
        operation: Callable[[], T],
        *,
        cancellation: CancellationToken | None = None,
        on_late: Callable[[object], None] | None = None,
    ) -> T | ToolError:
        """在停止边界等待后端证据，无法中断的线程保留迟到结果；传参：操作、信号与接收者；返回：真实结果。"""
        token = cancellation or CancellationToken(self.cancellation)
        if token.cancelled:
            return ToolError(
                ToolErrorCategory.CANCELLED,
                "cancelled before dispatch",
                retryable=False,
                details={"execution_state": "not_started"},
            )
        future: Future[T] = Future()
        token.begin_tool_phase("tool")
        started = time.monotonic()
        Thread(target=_run_future, args=(operation, future), daemon=True).start()
        while not token.cancelled:
            # 1. 【执行器】【时间归属】捕获、命令和后态保存各自受原时限约束，前后处理不冒充命令耗时
            phase = token.tool_phase()
            deadline = (
                phase[1] if phase is not None else started
            ) + self.tool_timeout_seconds
            if time.monotonic() >= deadline:
                break
            try:
                return future.result(
                    timeout=min(
                        _CANCEL_POLL_SECONDS, max(0, deadline - time.monotonic())
                    )
                )
            except FutureTimeoutError:
                if future.done():
                    return future.result()
                continue
        stopped = self._stop_tool(future, token, on_late=on_late)
        return stopped

    def _stop_tool(
        self,
        future: Future[T],
        token: CancellationToken,
        *,
        on_late: Callable[[object], None] | None,
    ) -> T | ToolError:
        """请求真实停止并交接未完成结果；参数：执行句柄/取消/迟到接收者；返回：结果或明确未知状态。"""
        phase = token.tool_phase()
        category = (
            ToolErrorCategory.CANCELLED
            if token.cancelled
            else ToolErrorCategory.TIMEOUT
        )
        token.cancel(category.value)
        # 【执行器】【停止交接】唤醒后端注册的真实关闭/取消入口；发出请求仍不代表副作用已经停止
        token.close_backend()
        grace = (
            _BACKEND_STOP_GRACE_SECONDS
            if token.backend_evidence().get("supports_stop")
            else 0
        )
        try:
            return future.result(timeout=grace)
        except FutureTimeoutError:
            if future.done():
                return future.result()
            if on_late is not None:
                future.add_done_callback(
                    partial(_deliver_late_result, receiver=on_late)
                )
            return ToolError(
                category,
                f"tool {category.value}; execution outcome unknown",
                retryable=False,
                partial_state="stop not confirmed; side effects unknown",
                details={
                    "execution_state": "unknown",
                    "phase": phase[0] if phase else "tool",
                    **token.backend_evidence(),
                },
            )

    def record_tool_failure(
        self, tool: str, args: Mapping[str, object]
    ) -> WatchdogDecision:
        key = failure_key(tool, args)
        count = self.tool_failures.get(key, 0) + 1
        self.tool_failures[key] = count
        # 【执行器】【失败计量】失败次数是证据；是否换方法由模型决定，真实派发仍受租约预算约束
        return WatchdogDecision(False)

    def record_llm_failure(self) -> WatchdogDecision:
        self.consecutive_llm_failures += 1
        if self.consecutive_llm_failures >= 3:
            return self._pause("llm_failure_budget", "segment llm failure budget hit")
        return WatchdogDecision(False)

    def record_llm_success(self) -> None:
        self.consecutive_llm_failures = 0

    def _pause(self, reason: str, message: str) -> WatchdogDecision:
        """记录详细暂停决定并接纳稳定通知；传参：原因和当前边界说明；返回：暂停决定。"""
        self._paused_reason = reason
        _notify_pause(reason, data_root=self.data_root, owner=self.budget_owner)
        return WatchdogDecision(True, reason, message)


def retry_with_budget(
    operation: Callable[[], T],
    *,
    max_retries: int = 3,
    retryable_categories: Collection[str] = _RETRYABLE_CATEGORIES,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    attempts = 0
    delays = retry_delays(max_retries=max_retries)
    while True:
        try:
            result = operation()
        except Exception as exc:
            category = _exception_category(exc)
            if attempts >= max_retries or category not in retryable_categories:
                raise
            sleep(delays[attempts])
            attempts += 1
            continue
        if isinstance(result, ToolError) and result.retryable:
            category = _tool_error_category(result.category)
            if attempts >= max_retries or category not in retryable_categories:
                return cast(T, result)
            sleep(delays[attempts])
            attempts += 1
            continue
        return result


def failure_key(tool: str, args: Mapping[str, object]) -> str:
    return json.dumps(
        {"tool": tool, "args": dict(args)}, ensure_ascii=False, sort_keys=True
    )


def _exception_category(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "transport"
    return "unknown"


def _tool_error_category(category: ToolErrorCategory | str) -> str:
    value = category.value if isinstance(category, ToolErrorCategory) else str(category)
    if value == "transport_error":
        return "transport"
    return value


def _token_total(usage: TokenUsage) -> int:
    return (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_read_input_tokens
        + usage.cache_creation_input_tokens
    )


def _notify_pause(
    reason: str, *, data_root: Path | str | None, owner: BudgetOwner | None
) -> None:
    """同一运行的同一暂停原因仅交付一条稳定通知；传参：原因与运行；返回：无。"""
    if data_root is None or owner is None:
        return
    from schedules.notifications import NotificationStore

    # 1. 【后台】【暂停通知】预算检查位置可能不同，通知按同一原因去重，详细边界说明仍在运行决定中
    NotificationStore(data_root).enqueue(
        f"pause-{owner.run_id}-{reason}",
        title="Reins 工作暂停",
        message=f"工作已暂停，原因：{reason}。打开对应会话可查看详细状态。",
        source={
            "session_id": owner.session_id,
            "run_id": owner.run_id,
            "reason": reason,
        },
    )


def _run_future(operation: Callable[[], T], future: Future[T]) -> None:
    """在可脱离的线程运行后端并交还异常；传参：操作与结果槽；返回：无。"""
    try:
        future.set_result(operation())
    except BaseException as exc:
        future.set_exception(exc)


def _deliver_late_result(
    future: Future[object], *, receiver: Callable[[object], None]
) -> None:
    """将迟到结果交回原操作写者；传参：结果与写者；返回：无，保存失败记录明确错误。"""
    try:
        result = future.result()
    except BaseException as exc:
        result = ToolError(ToolErrorCategory.UNKNOWN, str(exc), retryable=False)
    try:
        receiver(result)
    except Exception:
        _LOG.exception("【执行器】【迟到结果】必要结果保存失败")
