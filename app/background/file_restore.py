"""【本机后台】【文件恢复】持有恢复作业，界面断开不取消已确认操作。

作者：xxx
时间：2026-09-30 16:00:00
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from threading import RLock, Thread
from typing import TYPE_CHECKING, Any

from app.background.sessions import BackgroundSession
from app.session_assembly import user_lease
from approval.session import ApprovalMode
from runtime.file_restore_records import RestoreAuthority
from runtime.session_state import SessionStateStore

if TYPE_CHECKING:
    from app.background.service import BackgroundService
    from app.background.scheduled_session import ScheduledSession

_LOG = logging.getLogger(__name__)


class FileRestoreJobs:
    """独立作业容器；与会话运行共享权限和物理写入协调，不需要模型客户端。"""

    def __init__(self, owner: BackgroundService) -> None:
        """绑定后台与恢复领域；参数：现有后台；返回：无，不执行历史工作。"""
        from runtime.file_restore import FileRestoreService

        self.owner = owner
        self.service = FileRestoreService(
            owner.services.data_root, authority=self.authority, changed=self._changed
        )
        self._lock = RLock()
        self._jobs: dict[str, Thread] = {}
        self._closed = False

    def start(self) -> None:
        """启动时只对账未结束原件；参数：无；返回：无，不重放已启动文件动作。"""
        self.service.reconcile()

    def query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """转交不改工作文件的查询；参数：已认证载荷；返回：列表、详情、计划或状态。"""
        return self.service.query(payload)

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """先持久接纳再由后台线程执行；参数：计划和确认；返回：同计划唯一操作。"""
        with self._lock:
            if self._closed or self.owner.stopping.is_set():
                raise ValueError("后台正在停止，不能接纳新的文件恢复")
            operation, created = self.service.accept(payload)
            identity = operation["operation_id"]
            if created:
                thread = Thread(
                    target=self._run, args=(identity,), name=f"file-{identity}"
                )
                self._jobs[identity] = thread
                thread.start()
        return operation

    def cancel(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """取消后续项，保留已完成效果；参数：操作身份；返回：当前进度。"""
        identity = payload.get("operation_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("operation_id must be non-empty text")
        self.service.status(payload)
        return self.service.cancel(identity)

    def close(self) -> None:
        """宿主退出时请求停止后续项并等当前发布结束；参数：无；返回：无。"""
        with self._lock:
            self._closed = True
            jobs = tuple(self._jobs.items())
        for identity, _thread in jobs:
            self.service.cancel(identity)
        for _identity, thread in jobs:
            thread.join()

    def authority(self, session_id: str) -> RestoreAuthority:
        """读取当前宿主权限及可能写入的活动；参数：当前用户会话；返回：实时权限快照。"""
        session = self.owner.attach(session_id)
        workspace = self.owner.workspaces.for_session(session_id)
        state = SessionStateStore(self.owner.services.data_root).load(session_id)
        task_id = (
            (state.focus_task_id or state.compatibility_task_id)
            if state is not None
            else None
        )
        lease = user_lease(
            task_id=task_id or "file-restore",
            project_root=workspace.project_root,
            data_root=self.owner.services.data_root,
        )
        mode = (
            session.approval_session.mode
            if isinstance(session, BackgroundSession)
            else ApprovalMode.READ_ONLY
        )
        instance = (
            session.approval_session.instance_id
            if isinstance(session, BackgroundSession)
            else "scheduled"
        )
        return RestoreAuthority(
            lease, mode, instance, self._active_writers(workspace.project_root)
        )

    def _active_writers(self, root: Path) -> tuple[dict[str, Any], ...]:
        """描述同范围仍可能写文件的运行；参数：物理根；返回：原停止入口需要的身份。"""
        with self.owner._lock:
            sessions: tuple[BackgroundSession | ScheduledSession, ...] = (
                *self.owner._sessions.values(),
                *self.owner._scheduled_sessions.values(),
            )
        blocked = []
        for session in sessions:
            snapshot = session.snapshot()
            if not snapshot["active"]:
                continue
            workspace = self.owner.workspaces.for_session(snapshot["session_id"])
            other = workspace.project_root
            if root == other or root in other.parents or other in root.parents:
                blocked.append(
                    {
                        "session_id": snapshot["session_id"],
                        "run_id": snapshot.get("current_run_id"),
                        "status": snapshot["status"],
                        "project_root": str(other),
                        "stop_method": "cancel",
                    }
                )
        return tuple(blocked)

    def _changed(self, workspace_id: str, paths: tuple[str, ...]) -> None:
        """撤销本工作区旧敏感视图；参数：归属和实际变化路径；返回：无，旧消息不改写。"""
        with self.owner._lock:
            sessions = tuple(self.owner._sessions.values())
        for session in sessions:
            if session.workspace.workspace_id == workspace_id:
                session.redacted_files.invalidate_paths(paths)

    def _run(self, identity: str) -> None:
        """持有一次真实执行并报告无法登记的故障；参数：操作身份；返回：无。"""
        try:
            self.service.run(identity)
        except Exception:
            _LOG.exception(
                "【本机后台】【文件恢复】恢复结果需要重新对账，操作=%s", identity
            )
        finally:
            with self._lock:
                self._jobs.pop(identity, None)
