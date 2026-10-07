"""审批与普通纠正共用一个输入接收通道，工作线程不读取 stdin。

作者：xxx
时间：2026-09-14 10:00:00
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Condition
from uuid import uuid4
from typing import cast

from approval import ApprovalDecision, ApprovalRequest
from approval.batch_types import ApprovalBatch, BatchDecision, parse_batch_decision

_POLL_SECONDS = 0.05


class ApprovalChannel:
    """以请求编号关联决定，旧审批答案不能批准另一个动作。"""

    def __init__(
        self,
        present: Callable[[str, ApprovalRequest], None],
        *,
        present_batch: Callable[[str, ApprovalBatch], None] | None = None,
    ) -> None:
        """绑定界面展示函数；传参：编号/请求的展示入口；返回：无。"""
        self._present = present
        self._present_batch = present_batch
        self._batch: ApprovalBatch | None = None
        self._condition = Condition()
        self._pending: str | None = None
        self._decision: ApprovalDecision | BatchDecision | None = None
        self._closed = False

    def request(self, request: ApprovalRequest) -> ApprovalDecision:
        """等待同一输入通道的明确决定；传参：审批内容；返回：批准、拒绝或交互取消。"""
        return cast(ApprovalDecision, self._wait(request))

    def request_batch(self, request: ApprovalBatch) -> BatchDecision:
        """等待一个编号对应的整批逐项提交；传参：全部待决定项；返回：完整决定。"""
        return cast(BatchDecision, self._wait(request))

    def _wait(
        self, request: ApprovalRequest | ApprovalBatch
    ) -> ApprovalDecision | BatchDecision:
        """共用现有输入接收通道，不在工作线程读取stdin；传参：展示对象；返回：用户回执。"""
        with self._condition:
            if self._closed:
                return (
                    BatchDecision(cancelled=True)
                    if isinstance(request, ApprovalBatch)
                    else ApprovalDecision.CANCELLED
                )
            if self._pending is not None:
                raise RuntimeError("another approval is already pending")
            identity = uuid4().hex[:12]
            self._pending, self._decision = identity, None
            self._batch = request if isinstance(request, ApprovalBatch) else None
            try:
                if isinstance(request, ApprovalBatch):
                    if self._present_batch is None:
                        raise RuntimeError(
                            "this approval channel has no batch presenter"
                        )
                    self._present_batch(identity, request)
                else:
                    self._present(identity, request)
                while self._decision is None:
                    requests = (
                        request.requests
                        if isinstance(request, ApprovalBatch)
                        else (request,)
                    )
                    if any(_interrupted(item) for item in requests):
                        return (
                            BatchDecision(cancelled=True)
                            if self._batch
                            else ApprovalDecision.CANCELLED
                        )
                    self._condition.wait(_POLL_SECONDS)
                return self._decision
            finally:
                self._pending, self._decision = None, None
                self._batch = None

    def answer(self, command: str) -> None:
        """提交带编号的审批命令；传参：/approve 编号 once/task/permanent/deny；返回：无。"""
        fields = command.split(maxsplit=2)
        if len(fields) != 3 or fields[0] != "/approve":
            raise ValueError("用法：/approve 请求编号 once|task|permanent|deny")
        with self._condition:
            if fields[1] != self._pending or self._decision is not None:
                raise ValueError("该审批已结束或编号不匹配")
            self._decision = (
                parse_batch_decision(self._batch, fields[2])
                if self._batch
                else ApprovalDecision(fields[2])
            )
            self._condition.notify_all()

    def interrupt(self) -> None:
        """普通纠正或停止使待审批动作不再派发；传参：无；返回：无，不产生拒绝或授权。"""
        with self._condition:
            if self._pending is not None:
                self._decision = (
                    BatchDecision(cancelled=True)
                    if self._batch
                    else ApprovalDecision.CANCELLED
                )
                self._condition.notify_all()

    def close(self) -> None:
        """入口关闭后释放等待者；传参：无；返回：无。"""
        with self._condition:
            self._closed = True
            self.interrupt()


def _interrupted(request: ApprovalRequest) -> bool:
    """检查原请求是否被停止或新要求替代；传参：申请；返回：是否中断。"""
    return (request.cancellation is not None and request.cancellation.cancelled) or (
        request.superseded is not None and request.superseded()
    )
