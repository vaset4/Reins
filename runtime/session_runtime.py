"""协调同一会话的持久接纳、活跃运行与退出交接。

作者：xxx
时间：2026-09-14 10:00:00
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Condition, RLock, Thread
from llm.messages import UserContentPart

from runtime.cancellation import CancellationToken
from runtime.run_facts import RunFactStore, latest_lifecycle_from_facts
from runtime.session_message_store import SessionEntry, SessionMessageStore


@dataclass(frozen=True, slots=True)
class RecoveryIntent:
    """引用中断运行的恢复意图；传参：稳定身份、原运行/输入、原因和检查点；返回：不可变引用。"""

    recovery_id: str
    previous_run_id: str
    input_id: str
    reason: str
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        """拒绝没有真实来源的恢复；传参：无；返回：无，空身份直接报错。"""
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.recovery_id,
                self.previous_run_id,
                self.input_id,
                self.reason,
            )
        ):
            raise ValueError(
                "recovery requires stable identity, original run/input and reason"
            )


@dataclass(frozen=True, slots=True)
class SessionRun:
    """传给宿主的运行请求，只引用 Session 正文；cancellation 为本轮停止信号。"""

    input_id: str
    cancellation: CancellationToken
    recovery: RecoveryIntent | None = None


class SessionRuntime:
    """一个宿主内每会话一个协调器，正文和处理凭据仍由现有持久存储负责。"""

    def __init__(
        self,
        session_id: str,
        *,
        messages: SessionMessageStore,
        facts: RunFactStore,
        run: Callable[[SessionRun], None],
        parent_cancellation: CancellationToken | None = None,
    ) -> None:
        """绑定会话与执行依赖；传参：身份、消息/事实写者及运行入口；返回：无。"""
        self.session_id = session_id
        self._messages, self._facts, self._run = messages, facts, run
        self._condition = Condition(RLock())
        self._active = False
        self._closed = False
        self._parent_cancellation = parent_cancellation
        self._cancellation = CancellationToken(parent_cancellation)
        self._error: BaseException | None = None
        self._recovery: RecoveryIntent | None = None

    @property
    def active(self) -> bool:
        """查询是否仍有运行持有会话；传参：无；返回：活跃状态。"""
        with self._condition:
            return self._active

    def submit(
        self,
        text: str,
        *,
        input_id: str | None = None,
        task_id: str | None = None,
        content: tuple[UserContentPart, ...] | None = None,
    ) -> str:
        """先持久接纳再唤醒运行；传参：正文与可选去重身份/目标；返回：已保存的输入编号。"""
        with self._condition:
            if self._closed:
                raise RuntimeError("session runtime is closed")
            entry = self._messages.accept_input(
                self.session_id,
                text,
                input_id=input_id,
                task_id=task_id,
                content=content,
            )
            self._start_locked()
            return entry.entry_id

    def resume(self, recovery: RecoveryIntent | None = None) -> None:
        """接纳有持久来源的恢复或未处理输入；传参：可选恢复引用；返回：无，不追加会话正文。"""
        with self._condition:
            if self._closed:
                raise RuntimeError("session runtime is closed")
            if recovery is not None:
                if self._recovery is not None and self._recovery != recovery:
                    raise ValueError(
                        "another recovery is already pending for this session"
                    )
                if not self._accept_recovery(recovery):
                    return
                self._recovery = recovery
            self._start_locked()

    def _accept_recovery(self, recovery: RecoveryIntent) -> bool:
        """在原运行记录中接纳恢复身份；传参：恢复意图；返回：是否仍需执行，重投不重复接纳。"""
        rows = self._facts.read_run(recovery.previous_run_id)
        if not rows or any(row.get("session_id") != self.session_id for row in rows):
            raise ValueError(
                "recovery source run is missing or belongs to another session"
            )
        accepted = [
            row
            for row in rows
            if row.get("event") == "recovery:accepted"
            and row.get("recovery_id") == recovery.recovery_id
        ]
        payload = asdict(recovery)
        if accepted and any(
            any(row.get(key) != value for key, value in payload.items())
            for row in accepted
        ):
            raise ValueError(
                "recovery identity was already accepted with different sources"
            )
        if any(
            row.get("event") == "recovery:handled"
            and row.get("recovery_id") == recovery.recovery_id
            for row in rows
        ):
            return False
        if latest_lifecycle_from_facts(rows):
            raise ValueError(
                "a terminal or waiting run cannot be automatically resumed"
            )
        entries = self._messages.materialize(self.session_id).entries
        if not any(
            row.entry_id == recovery.input_id and row.type == "inbound"
            for row in entries
        ):
            raise ValueError("recovery input is not on the current session branch")
        if not accepted:
            self._facts.append(
                {
                    "event": "recovery:accepted",
                    "session_id": self.session_id,
                    "run_id": recovery.previous_run_id,
                    **payload,
                }
            )
        return True

    def cancel(self) -> None:
        """停止活跃运行的真实执行边界；传参：无；返回：无，停止证明由后端记录。"""
        with self._condition:
            self._cancellation.cancel()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """等待会话交接并暴露执行失败；传参：等待秒数；返回：是否已空闲。"""
        with self._condition:
            idle = self._condition.wait_for(lambda: not self._active, timeout)
            if idle and self._error is not None:
                raise RuntimeError(
                    "session execution failed; accepted inputs remain recoverable"
                ) from self._error
            return idle

    def close(self, *, cancel: bool = False) -> None:
        """关闭接纳入口并按用户停止信号结束当前工作；传参：是否取消；返回：无。"""
        with self._condition:
            self._closed = True
            if cancel:
                self._cancellation.cancel()

    def _start_locked(self) -> None:
        """在接纳与交接共用锁内启动唯一执行者；传参：无；返回：无。"""
        if self._active or (self._recovery is None and not self._unhandled_inputs()):
            return
        self._error = None
        self._active = True
        self._cancellation = CancellationToken(self._parent_cancellation)
        Thread(
            target=self._drain, name=f"session-{self.session_id}", daemon=True
        ).start()

    def _drain(self) -> None:
        """连续处理交接边界的新输入，失败/暂停不自行重放本轮输入；传参：无；返回：无。"""
        attempted: set[str] = set()
        try:
            while True:
                with self._condition:
                    pending = tuple(
                        item
                        for item in self._unhandled_inputs()
                        if item.entry_id not in attempted
                    )
                    recovery = self._recovery
                    if (
                        not pending and recovery is None
                    ) or self._cancellation.cancelled:
                        self._active = False
                        self._condition.notify_all()
                        return
                    if recovery is not None:
                        request = SessionRun(
                            recovery.input_id, self._cancellation, recovery
                        )
                        attempted.add(recovery.input_id)
                    else:
                        attempted.update(item.entry_id for item in pending)
                        request = SessionRun(pending[0].entry_id, self._cancellation)
                # 【会话执行】【连接生命周期】复用当前线程连接，每条事实仍独立事务提交
                with self._messages.database.connection_scope():
                    self._run(request)
                if recovery is not None:
                    with self._condition:
                        self._facts.append(
                            {
                                "event": "recovery:handled",
                                "session_id": self.session_id,
                                "run_id": recovery.previous_run_id,
                                "recovery_id": recovery.recovery_id,
                            }
                        )
                        self._recovery = None
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._active = False
                self._condition.notify_all()

    def _unhandled_inputs(self) -> tuple[SessionEntry, ...]:
        """按持久处理凭据推导待处理引用，交付记录本身不算处理成功；传参：无；返回：待处理入站。"""
        handled = self._facts.handled_input_ids(self.session_id)
        return tuple(
            item
            for item in self._messages.pending_inputs(
                self.session_id, include_delivered=True
            )
            if item.entry_id not in handled
        )
