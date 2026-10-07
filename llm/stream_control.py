"""在网络阻塞期间传递取消，保留已收到事件且不执行迟到的模型决策。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any, TypeVar

from runtime.cancellation import CancellationToken, ExecutionCancelled
from llm.provider_adapter import ProviderAdapterError

T = TypeVar("T")
_POLL_SECONDS = 0.05


def interruptible_events(
    events: Iterator[T], cancellation: CancellationToken | None
) -> Iterator[T]:
    """把停止请求从读取线程传到网络流；传参：原事件流与信号；返回：保序事件流。"""
    if cancellation is None:
        yield from events
        return
    queue: Queue[tuple[str, Any]] = Queue(maxsize=1)
    Thread(target=_pump_events, args=(events, queue, cancellation), daemon=True).start()
    while not cancellation.cancelled:
        try:
            kind, value = queue.get(timeout=_POLL_SECONDS)
        except Empty:
            continue
        if kind == "end":
            return
        if kind == "error":
            raise value
        yield value
    cancellation.close_backend()
    raise ExecutionCancelled("model request cancelled; remote usage may be incomplete")


def _pump_events(
    events: Iterator[T], queue: Queue[tuple[str, Any]], token: CancellationToken
) -> None:
    """受背压控制读取网络，停止后退出投递；传参：流、队列和信号；返回：无。"""
    try:
        for event in events:
            if not _put_event(queue, ("event", event), token):
                return
    except BaseException as exc:
        _put_event(queue, ("error", exc), token)
    finally:
        _put_event(queue, ("end", None), token)


def _put_event(
    queue: Queue[tuple[str, Any]], item: tuple[str, Any], token: CancellationToken
) -> bool:
    """只在接收方仍运行时投递事件；传参：队列、事件与信号；返回：是否投递。"""
    while not token.cancelled:
        try:
            queue.put(item, timeout=_POLL_SECONDS)
            return True
        except Full:
            continue
    return False


def mapped_sdk_stream(
    stream: Iterable[object],
    mapper: Callable[[Any], Mapping[str, object]],
    token: CancellationToken | None,
) -> Iterator[Mapping[str, object]]:
    """绑定并关闭实际SDK流，取消到达连接层；传参：SDK流、映射与信号；返回：原始响应事件。"""
    closer = getattr(stream, "close", None)
    if token is not None and callable(closer):
        token.register_closer(closer)
    try:
        response = getattr(stream, "response", None)
        if response is not None:
            content_type = (
                response.headers.get("content-type", "")
                .split(";", maxsplit=1)[0]
                .strip()
                .lower()
            )
            if content_type != "text/event-stream":
                raise ProviderAdapterError(
                    f"invalid_provider_response: received {content_type or 'no content type'} "
                    f"(HTTP {response.status_code}); expected text/event-stream"
                )
        for item in stream:
            yield mapper(item)
    finally:
        if callable(closer):
            closer()
