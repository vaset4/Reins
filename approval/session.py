"""宿主会话内的授权范围；重启不从审计日志恢复临时权限。

作者：xxx
时间：2026-09-24 12:00:00
"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from threading import RLock
from uuid import uuid4


class ApprovalMode(str, Enum):
    READ_ONLY = "read_only"
    WORKSPACE = "workspace"
    AUTO = "auto"


class ApprovalSession:
    """由实际宿主持有的一份临时权限，持久会话编号不等于此实例身份。"""

    def __init__(self) -> None:
        """创建默认工作区模式和独立实例；传参：无；返回：无。"""
        self.instance_id = uuid4().hex
        self._mode = ApprovalMode.WORKSPACE
        self._grants: dict[str, dict[str, object]] = {}
        self._revoked: set[str] = set()
        self._lock = RLock()

    @property
    def mode(self) -> ApprovalMode:
        """读取当前模式；传参：无；返回：会话模式。"""
        with self._lock:
            return self._mode

    def set_mode(self, mode: ApprovalMode) -> None:
        """接纳真实界面的显式模式选择；传参：枚举值；返回：无。"""
        if not isinstance(mode, ApprovalMode):
            raise ValueError("invalid approval mode")
        with self._lock:
            self._mode = mode

    def remember(self, grant: dict[str, object]) -> None:
        """保存事件已提交的临时授权；传参：完整授权；返回：无，身份冲突明确报错。"""
        identity = str(grant["grant_id"])
        with self._lock:
            previous = self._grants.get(identity)
            if previous is not None and previous != grant:
                raise ValueError("approval grant identity has different content")
            if identity in self._revoked:
                raise ValueError("approval grant has been revoked")
            self._grants[identity] = deepcopy(grant)

    def grants(self) -> tuple[dict[str, object], ...]:
        """返回独立权限快照；传参：无；返回：不共享可变内容的授权。"""
        with self._lock:
            return tuple(
                deepcopy(value)
                for key, value in self._grants.items()
                if key not in self._revoked
            )

    def revoke(self, grant_id: str) -> None:
        """先阻止权限复用，审计失败也不恢复允许；传参：授权身份；返回：无。"""
        with self._lock:
            self._revoked.add(grant_id)
            self._grants.pop(grant_id, None)

    def is_revoked(self, grant_id: str) -> bool:
        """派发前复核本宿主立即撤销；传参：授权身份；返回：是否撤销。"""
        with self._lock:
            return grant_id in self._revoked

    def close(self) -> None:
        """会话结束时撤回全部内存权限；传参：无；返回：无。"""
        with self._lock:
            self._revoked.update(self._grants)
            self._grants.clear()
            self._mode = ApprovalMode.READ_ONLY
