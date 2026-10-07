"""将停止请求传给真实执行边界，不把取消请求当作停止证明。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

from threading import Event, RLock, Thread
from collections.abc import Callable
import logging
import time

_POLL_SECONDS = 0.05


class CancellationToken:
    """共享停止信号；父信号停止整个运行，子信号只停止当前操作。"""

    def __init__(self, parent: CancellationToken | None = None) -> None:
        """创建未取消信号；传参：可选父运行；返回：无。"""
        self._event = Event()
        self._parent = parent
        self.reason = ""
        self._lock = RLock()
        self._backend: dict[str, object] = {}
        self._closers: list[Callable[[], object]] = []
        self._tool_phase: tuple[str, float] | None = None

    @property
    def cancelled(self) -> bool:
        """查询自身或父运行是否要求停止；传参：无；返回：是否停止新派发。"""
        return self._event.is_set() or (
            self._parent is not None and self._parent.cancelled
        )

    def cancel(self, reason: str = "user_stop") -> None:
        """只记录停止请求，执行端另报真实结果；传参：停止原因；返回：无。"""
        self.reason = reason
        self._event.set()

    def wait(self, seconds: float) -> bool:
        """等待退避或父运行取消；传参：最多等待秒数；返回：是否已取消。"""
        deadline = time.monotonic() + seconds
        while not self.cancelled and time.monotonic() < deadline:
            self._event.wait(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
        return self.cancelled

    def report_backend(self, **evidence: object) -> None:
        """记录执行端已观察的停止证据；传参：实际后端状态；返回：无。"""
        with self._lock:
            self._backend.update(evidence)

    def backend_evidence(self) -> dict[str, object]:
        """读取后端证据快照；传参：无；返回：独立映射，不把请求当成证明。"""
        with self._lock:
            return dict(self._backend)

    def begin_tool_phase(self, name: str) -> None:
        """记录宿主实际进入的捕获或执行阶段；参数：阶段名；返回：无，不清除取消信号。"""
        with self._lock:
            self._tool_phase = (name, time.monotonic())

    def tool_phase(self) -> tuple[str, float] | None:
        """读取实际阶段及起点供时间归属使用；参数：无；返回：阶段快照或未分阶段。"""
        with self._lock:
            return self._tool_phase

    def register_closer(self, close: Callable[[], object]) -> None:
        """绑定实际网络流或连接的关闭入口；传参：关闭函数；返回：无。"""
        with self._lock:
            self._closers.append(close)
        if self.cancelled:
            self.close_backend()

    def close_backend(self) -> None:
        """异步请求关闭实际连接，不宣称远端计算已停止；传参：无；返回：无。"""
        with self._lock:
            closers, self._closers = self._closers, []
        for close in closers:
            Thread(target=_close_backend, args=(close,), daemon=True).start()


class ExecutionCancelled(RuntimeError):
    """执行已收到停止请求，调用方必须保留未知用量和已发生效果。"""


class RunBudgetExceeded(RuntimeError):
    """本次真实派发被上游额度拒绝；租约额度已取消，调用方须保留已发生用量。"""


def _close_backend(close: Callable[[], object]) -> None:
    """关闭实际执行连接并暴露失败；传参：后端关闭入口；返回：无。"""
    try:
        close()
    except Exception:
        logging.getLogger(__name__).exception(
            "【执行器】【取消连接】后端关闭失败，远端结果仍未知"
        )
