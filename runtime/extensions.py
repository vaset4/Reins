"""区分事实观察、可等待介入、结果视图与动作请求。

作者：xxx
时间：2026-09-14 12:00:00
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import TypeVar, cast

from llm.messages import JsonValue, freeze_json_object
from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.watchdog import Watchdog
from tools.types import ToolError, ToolErrorCategory

T = TypeVar("T")
_POLL_SECONDS = 0.05
_OBSERVING: ContextVar[bool] = ContextVar("runtime_observing", default=False)


@dataclass(frozen=True, slots=True)
class ToolProposal:
    """可改写的候选工具参数，不含租约、预算或实际执行入口。"""

    tool: str
    arguments: Mapping[str, JsonValue]
    operation_id: str = ""

    def __post_init__(self) -> None:
        """冻结入参避免介入者原地改写执行事实；传参：构造字段；返回：无。"""
        object.__setattr__(
            self,
            "arguments",
            freeze_json_object(self.arguments, path="tool_proposal.arguments"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    """只读事实快照，返回值不参与业务决策。"""

    event: str
    detail: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        """冻结事实字段；传参：构造字段；返回：无。"""
        object.__setattr__(
            self, "detail", freeze_json_object(self.detail, path="observation.detail")
        )


@dataclass(frozen=True, slots=True)
class ResultView:
    """派生模型视图与注释，不含可改写真实status/error的字段。"""

    content: str | None = None
    annotations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """后处理只提交动作意图，执行另走共同校验/授权/预算边界。"""

    request_id: str
    tool: str
    arguments: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        """校验可去重身份并冻结参数；传参：请求字段；返回：无。"""
        if (
            not self.request_id
            or any(char in self.request_id for char in "/\\:")
            or self.request_id in {".", ".."}
        ):
            raise ValueError("action request_id must be a single storage name")
        object.__setattr__(
            self,
            "arguments",
            freeze_json_object(self.arguments, path="action_request.arguments"),
        )


BeforeTool = Callable[
    [ToolProposal, CancellationToken], ToolProposal | Awaitable[ToolProposal]
]
Observer = Callable[[RuntimeObservation], None]
ViewHook = Callable[[RuntimeObservation], ResultView | Awaitable[ResultView]]
ContextHook = Callable[[RuntimeObservation], str | Awaitable[str]]
AfterRun = Callable[
    [RuntimeObservation],
    tuple[ActionRequest, ...] | Awaitable[tuple[ActionRequest, ...]],
]


@dataclass(frozen=True, slots=True)
class RuntimeExtensions:
    """运行局部的扩展依赖；所有入口默认没有扩展。"""

    before_tool: tuple[BeforeTool, ...] = ()
    observers: tuple[Observer, ...] = ()
    result_views: tuple[ViewHook, ...] = ()
    context_sources: tuple[ContextHook, ...] = ()
    after_run: tuple[AfterRun, ...] = ()

    def prepare(
        self, proposal: ToolProposal, cancellation: CancellationToken
    ) -> ToolProposal:
        """等待前置介入完成，输出仍只是需要重新校验的建议；传参：候选/取消信号；返回：候选。"""
        for hook in self.before_tool:
            proposal = resolve_hook(hook(proposal, cancellation), cancellation)
            if not isinstance(proposal, ToolProposal):
                raise TypeError("before-tool hook must return ToolProposal")
        return proposal

    def observe(self, observation: RuntimeObservation) -> tuple[str, ...]:
        """逐个通知观察者，失败作为独立诊断返回；传参：已提交事实；返回：观察错误，不改变原事实。"""
        errors: list[str] = []
        boundary = _OBSERVING.set(True)
        try:
            for observer in self.observers:
                try:
                    observer(observation)
                except Exception as exc:
                    errors.append(
                        f"{getattr(observer, '__qualname__', type(observer).__name__)}: {type(exc).__name__}: {exc}"
                    )
        finally:
            _OBSERVING.reset(boundary)
        return tuple(errors)

    @property
    def observing(self) -> bool:
        """识别观察回调内的递归派发；传参：无；返回：当前是否只读观察边界。"""
        return _OBSERVING.get()


def run_hook(
    operation: Callable[[], T | Awaitable[T]],
    watchdog: Watchdog,
    cancellation: CancellationToken,
) -> T:
    """在可取消边界调用同步或异步扩展；传参：回调/等待边界/运行信号；返回：完成值，迟到建议无效。"""
    token = CancellationToken(cancellation)
    value = watchdog.run_tool_with_timeout(
        partial(_invoke_hook, operation, token), cancellation=token
    )
    if isinstance(value, ToolError):
        if value.category == ToolErrorCategory.CANCELLED:
            raise ExecutionCancelled("extension cancelled while waiting")
        raise TimeoutError("extension did not finish before its execution timeout")
    return value


def _invoke_hook(
    operation: Callable[[], T | Awaitable[T]], cancellation: CancellationToken
) -> T:
    """在受控等待线程内调用扩展，阻塞不占用运行线程；传参：回调/取消；返回：完成值。"""
    return resolve_hook(operation(), cancellation)


def resolve_hook(value: T | Awaitable[T], cancellation: CancellationToken) -> T:
    """同步宿主可等待协作式异步Hook，取消不产生有效建议；传参：返回值/信号；返回：真实完成值。"""
    if not inspect.isawaitable(value):
        if cancellation.cancelled:
            raise ExecutionCancelled("extension cancelled before completion")
        return value
    return cast(T, asyncio.run(_await_hook(value, cancellation)))


async def _await_hook(value: Awaitable[T], cancellation: CancellationToken) -> T:
    """将运行取消传到可等待Hook；传参：等待对象/信号；返回：完成值。"""
    task = asyncio.ensure_future(value)
    while not task.done():
        if cancellation.cancelled:
            task.cancel()
            raise ExecutionCancelled("extension cancelled while waiting")
        await asyncio.wait({task}, timeout=_POLL_SECONDS)
    return task.result()
