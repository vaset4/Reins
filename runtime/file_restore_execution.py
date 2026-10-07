"""【文件恢复】【后台执行】复核完整计划、保存真实移走字节并逐项登记效果。

作者：xxx
时间：2026-09-30 22:00:00
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runtime.file_restore import ensure_path, parent_versions, write_allowed
from runtime.file_restore_records import TERMINAL_STATES, operation_view
from runtime.file_snapshots import new_point
from tasks.ids import utc_now
from tools.file_persistence import (
    FileEditConflict,
    file_edit_lock,
    publish_prepared_file,
)
from tools.restore_protection import (
    create_private_staging,
    handle_security,
    move_handle_file,
    private_handle_security,
    protected_replace_prepared_file,
    security_handle,
    set_handle_security,
    verify_private_acl,
)
from tools.workspace_coordination import workspace_write_window

if TYPE_CHECKING:
    from runtime.file_restore import FileRestoreService

_LOG = logging.getLogger(__name__)


class RestoreExecution:
    """一个计划的真实执行责任，磁盘操作不占运行数据提交锁。"""

    def __init__(self, service: FileRestoreService) -> None:
        """绑定恢复领域；参数：服务；返回：无。"""
        self.service = service
        self.records, self.snapshots = service.records, service.snapshots
        self.created_parents: dict[str, dict[str, Any]] = {}

    def run(self, identity: str) -> dict[str, Any]:
        """只执行首次排队操作；参数：身份；返回：逐项实际结果。"""
        with file_edit_lock(self.snapshots.data_root / (identity + ".execution")):
            operation = self.records.require("file_restore_operation", identity)
            if operation["status"] != "queued":
                return operation_view(operation)
            plan = self.records.require("file_restore_plan", operation["plan_id"])
            root = Path(plan["workspace_root"])
            try:
                with workspace_write_window(
                    root, subtree=True, wait=False, owner=identity
                ):
                    self._preflight(plan)
                    self.records.update(identity, {"status": "running"})
                    for entry in plan["entries"]:
                        if not entry["selected"]:
                            continue
                        latest = self.records.require(
                            "file_restore_operation", identity
                        )
                        if latest["cancel_requested"]:
                            self.records.update(
                                identity,
                                {"status": "cancelled"},
                                entry_id=entry["entry_id"],
                            )
                            continue
                        self._execute_entry(identity, plan, entry)
            except (OSError, ValueError) as exc:
                current = self.records.require("file_restore_operation", identity)
                for entry in current["entries"]:
                    if entry["status"] == "pending":
                        self.records.update(
                            identity,
                            {
                                "status": "conflict",
                                "error": str(exc),
                                "effect": "not_changed",
                            },
                            entry_id=entry["entry_id"],
                        )
                self.records.update(identity, {"error": str(exc)})
            operation = self._finish(identity)
            paths = tuple(
                entry["destination"]
                for entry in operation["entries"]
                if entry["effect"] in {"changed", "changed_with_conflict", "unknown"}
            )
            if paths and self.service.changed is not None:
                self.service.changed(plan["workspace_id"], paths)
            return operation_view(operation)

    def _preflight(self, plan: dict[str, Any]) -> None:
        """第一次写入前统一检查全部选择；参数：计划；返回：无，一项变化使全计划失效。"""
        authority = self.service.authority(plan["session_id"])
        if authority.evidence() != plan["authority"] or authority.blocked_by:
            raise FileEditConflict("当前权限或仍活动的写入者与预览不同，请重新预览")
        for entry in plan["entries"]:
            if entry["selected"]:
                self._validate_entry(plan, entry)

    def _validate_entry(
        self, plan: dict[str, Any], entry: dict[str, Any]
    ) -> dict[str, Any]:
        """复核路径、权限和真实版本；参数：固定计划和项；返回：当前完整前态。"""
        path, root = Path(entry["destination"]), Path(plan["workspace_root"])
        authority = self.service.authority(plan["session_id"])
        if authority.evidence() != plan["authority"] or authority.blocked_by:
            raise FileEditConflict("权限或活动已变化，请重新预览")
        ensure_path(path, root)
        if not write_allowed(path, authority, entry["sensitive"]):
            raise PermissionError("当前权限不允许恢复此目标")
        self.snapshots.validate_state(entry["target"])
        parents = [
            self.created_parents.get(row["path"], row) for row in entry["parents"]
        ]
        if parent_versions(path, root) != parents:
            raise FileEditConflict("父目录身份或权限已变化，请重新预览")
        current = self.snapshots.capture_file(
            path, authority.lease, plan["workspace_id"]
        )
        if not self.snapshots.same_state(current, entry["expected"], identity=True):
            raise FileEditConflict("文件在预览后发生变化，请重新预览")
        self.snapshots.validate_state(current)
        if entry["choice"] == "copy":
            original = self.snapshots.capture_file(
                Path(entry["path"]), authority.lease, plan["workspace_id"]
            )
            if not self.snapshots.same_state(original, entry["current"], identity=True):
                raise FileEditConflict("原文件在预览后发生变化，请重新预览副本选择")
        return current

    def _prepare_parent(self, plan: dict[str, Any], entry: dict[str, Any]) -> None:
        """仅创建计划列出的缺失祖先，保留其他目录资料；参数：计划和项；返回：无。"""
        root = Path(plan["workspace_root"])
        for row in reversed(entry["parents"]):
            if row["identity"] is not None or row["path"] in self.created_parents:
                continue
            path = Path(row["path"])
            ensure_path(path, root)
            path.mkdir()
            own = parent_versions(path / "child", root)[0]
            self.created_parents[row["path"]] = own

    def _execute_entry(
        self, identity: str, plan: dict[str, Any], entry: dict[str, Any]
    ) -> None:
        """保存前态后执行一项，真实失败不触发回滚；参数：操作、计划、文件；返回：无。"""
        path, entry_id = Path(entry["destination"]), entry["entry_id"]
        changed = False
        try:
            with file_edit_lock(path):
                before = self._validate_entry(plan, entry)
                if entry["choice"] != "copy" and self.snapshots.same_state(
                    before, entry["target"]
                ):
                    self.records.update(
                        identity,
                        {"status": "unchanged", "effect": "not_changed"},
                        entry_id=entry_id,
                    )
                    return
                self._prepare_parent(plan, entry)
                stage = create_private_staging(path.parent)
                temporary, backup = stage / "target.tmp", stage / "displaced.tmp"
                # 1. 【文件恢复】【执行意图】路径和完整前态先持久化，中断后仍能定位同卷保护副本
                self.records.update(
                    identity,
                    {
                        "status": "started",
                        "effect": "not_changed",
                        "before_restore": before,
                        "temporary_path": str(temporary),
                        "backup_path": str(backup),
                        "started_at": utc_now(),
                    },
                    entry_id=entry_id,
                )
                self._stage_target(temporary, entry)
                if self.records.require("file_restore_operation", identity)[
                    "cancel_requested"
                ]:
                    self.records.update(
                        identity, {"status": "cancelled"}, entry_id=entry_id
                    )
                    temporary.unlink(missing_ok=True)
                    stage.rmdir()
                    return
                current = self._validate_entry(plan, entry)
                if not self.snapshots.same_state(before, current, identity=True):
                    raise FileEditConflict("文件在发布前发生变化，没有执行替换")
                self.records.update(
                    identity, {"effect": "publishing"}, entry_id=entry_id
                )
                self._publish(path, temporary, backup, before=before, entry=entry)
                changed = True
                # 2. 【文件恢复】【效果核验】使用真正被移走的字节，不用预检快照冒充最后版本
                displaced = self._displaced(plan, entry, backup, before)
                authority = self.service.authority(plan["session_id"])
                after = self.snapshots.capture_file(
                    path, authority.lease, plan["workspace_id"]
                )
                conflict = not self.snapshots.same_state(
                    displaced, before
                ) or not self.snapshots.same_state(after, entry["target"])
                point = self._save_point(
                    plan, identity, entry, displaced, after=after, conflict=conflict
                )
                self.records.update(
                    identity,
                    {
                        "status": "conflict" if conflict else "restored",
                        "effect": "changed_with_conflict" if conflict else "changed",
                        "actual_displaced": displaced,
                        "restore_point_id": point["point_id"],
                        "after_restore": after,
                        "error": "发布时检测到外部竞争，实际移走内容已保存"
                        if conflict
                        else None,
                    },
                    entry_id=entry_id,
                )
                temporary.unlink(missing_ok=True)
                backup.unlink(missing_ok=True)
                stage.rmdir()
        except (OSError, ValueError) as exc:
            latest = next(
                row
                for row in self.records.require("file_restore_operation", identity)[
                    "entries"
                ]
                if row["entry_id"] == entry_id
            )
            backup_exists = bool(
                latest.get("backup_path") and Path(latest["backup_path"]).exists()
            )
            unknown = changed or backup_exists or latest.get("effect") == "publishing"
            self.records.update(
                identity,
                {
                    "status": "unknown"
                    if unknown
                    else "conflict"
                    if isinstance(exc, FileEditConflict)
                    else "failed",
                    "effect": "unknown" if unknown else "not_changed",
                    "error": str(exc),
                },
                entry_id=entry_id,
            )

    def _stage_target(self, temporary: Path, entry: dict[str, Any]) -> None:
        """流式暂存目标，明文只在私有目录；参数：已记录位置和目标；返回：无。"""
        if entry["target"]["kind"] == "missing":
            return
        verify_private_acl(temporary.parent)
        with temporary.open("xb") as handle:
            for chunk in self.snapshots.iter_state(entry["target"]):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        verify_private_acl(temporary)

    def _publish(
        self,
        path: Path,
        temporary: Path,
        backup: Path,
        *,
        before: dict[str, Any],
        entry: dict[str, Any],
    ) -> None:
        """同卷发布或移走，保留实际被替换字节及当前权限；参数：路径、暂存、备份、前态、项；返回：无。"""
        from tools.file_persistence import replace_prepared_file

        if before["kind"] == "missing":
            target_metadata = self.snapshots.state_metadata(entry["target"])
            os.chmod(temporary, target_metadata["mode"])
            with security_handle(temporary) as replacement:
                publish_prepared_file(path, temporary)
                set_handle_security(replacement, target_metadata["security"])
            return
        if entry["target"]["kind"] != "missing":
            if entry["sensitive"]:
                protected_replace_prepared_file(path, temporary, backup_path=backup)
            else:
                replace_prepared_file(path, temporary, backup_path=backup)
            return
        with security_handle(path, rename=True) as handle:
            security = handle_security(handle)
            private_handle_security(handle)
            published = False
            try:
                move_handle_file(handle, backup)
                published = True
            finally:
                if not published:
                    set_handle_security(handle, security)
            verify_private_acl(backup)

    def _displaced(
        self,
        plan: dict[str, Any],
        entry: dict[str, Any],
        backup: Path,
        before: dict[str, Any],
    ) -> dict[str, Any]:
        """归档真实移走版本，敏感判定沿原路径；参数：计划、项、备份、前态；返回：准确旧版本。"""
        if not backup.exists():
            if before["kind"] != "missing":
                raise FileNotFoundError("实际被替换文件备份缺失，结果需要核对")
            return before
        authority = self.service.authority(plan["session_id"])
        state = self.snapshots.capture_displaced(
            backup, before, authority.lease, workspace_id=plan["workspace_id"]
        )
        self.snapshots.validate_state(state)
        return state

    def _save_point(
        self,
        plan: dict[str, Any],
        identity: str,
        entry: dict[str, Any],
        before: dict[str, Any],
        *,
        after: dict[str, Any],
        conflict: bool,
    ) -> dict[str, Any]:
        """把本次恢复也留为可预览的动作；参数：归属、前后态及竞争；返回：已发布点。"""
        point = new_point(
            {
                "workspace_id": plan["workspace_id"],
                "workspace_root": plan["workspace_root"],
                "session_id": plan["session_id"],
                "run_id": "",
                "input_id": identity,
                "operation_id": identity,
            },
            scope="files",
            attribution="unknown" if conflict else "exact",
        )
        point["point_id"] = "point-" + identity + "-" + entry["entry_id"]
        with self.snapshots.database.snapshot() as source:
            previous = source.get("file_restore_point", point["point_id"])
        if previous is not None:
            return previous
        point.update(
            status="complete",
            completed_at=utc_now(),
            entries=[
                {
                    "path": entry["destination"],
                    "before": before,
                    "after": after,
                    "attribution": point["attribution"],
                }
            ],
        )
        self.snapshots.publish_point(point)
        return point

    def _finish(self, identity: str) -> dict[str, Any]:
        """汇总真实逐项结果，不把部分恢复称全成；参数：操作；返回：最终记录。"""
        operation = self.records.require("file_restore_operation", identity)
        states = {
            entry["status"] for entry in operation["entries"] if entry["selected"]
        }
        if "unknown" in states or "started" in states:
            status = "needs_reconciliation"
        elif states <= {"restored", "unchanged"}:
            status = "completed"
        elif "restored" in states or "unchanged" in states:
            status = "partial"
        elif states <= {"cancelled"}:
            status = "cancelled"
        else:
            status = "failed"
        return self.records.update(identity, {"status": status})

    def reconcile(self) -> None:
        """重启核对全部未完成操作，绝不再次执行；参数：无；返回：无。"""
        with self.snapshots.database.snapshot() as source:
            operations = source.list("file_restore_operation")
        for operation in operations:
            if (
                operation["status"] in TERMINAL_STATES
                and operation["status"] != "needs_reconciliation"
            ):
                continue
            identity = operation["operation_id"]
            try:
                with file_edit_lock(
                    self.snapshots.data_root / (identity + ".execution")
                ):
                    self._reconcile_operation(identity)
            except FileEditConflict:
                _LOG.info(
                    "【文件恢复】【重启对账】操作仍由其他执行者持有，保留当前进度与暂存：%s",
                    identity,
                )

    def _reconcile_operation(self, identity: str) -> None:
        """获得原操作独占后重新读取事实；参数：操作身份；返回：无，活作业不能被清理。"""
        operation = self.records.require("file_restore_operation", identity)
        if (
            operation["status"] in TERMINAL_STATES
            and operation["status"] != "needs_reconciliation"
        ):
            return
        plan = self.records.require("file_restore_plan", operation["plan_id"])
        for entry in operation["entries"]:
            if entry["status"] in {"started", "unknown"}:
                self._reconcile_entry(identity, plan, entry)
            elif entry["status"] == "pending":
                self.records.update(
                    identity,
                    {
                        "status": "cancelled",
                        "effect": "not_started",
                        "error": "宿主已重启，未执行项需要重新预览确认",
                    },
                    entry_id=entry["entry_id"],
                )
        self._finish(identity)

    def _reconcile_entry(
        self, identity: str, plan: dict[str, Any], entry: dict[str, Any]
    ) -> None:
        """按当前磁盘和暂存证据对账，不把内容相同当成功归因；参数：操作、计划、项；返回：无。"""
        evidence: dict[str, Any] = {"automatic_replay": False}
        try:
            authority = self.service.authority(plan["session_id"])
            current = self.snapshots.capture_file(
                Path(entry["destination"]), authority.lease, plan["workspace_id"]
            )
            evidence["disk_matches"] = (
                "target"
                if self.snapshots.same_state(current, entry["target"])
                else (
                    "before"
                    if self.snapshots.same_state(current, entry["before_restore"])
                    else "other"
                )
            )
            backup = Path(entry["backup_path"])
            if backup.exists():
                displaced = self._displaced(
                    plan, entry, backup, entry["before_restore"]
                )
                self.records.update(
                    identity,
                    {"actual_displaced": displaced},
                    entry_id=entry["entry_id"],
                )
                if not entry.get("restore_point_id"):
                    point = self._save_point(
                        plan, identity, entry, displaced, after=current, conflict=True
                    )
                    self.records.update(
                        identity,
                        {"restore_point_id": point["point_id"]},
                        entry_id=entry["entry_id"],
                    )
                evidence["backup_archived"] = True
                backup.unlink()
            elif not entry.get("restore_point_id"):
                point = self._save_point(
                    plan,
                    identity,
                    entry,
                    entry["before_restore"],
                    after=current,
                    conflict=True,
                )
                self.records.update(
                    identity,
                    {"restore_point_id": point["point_id"]},
                    entry_id=entry["entry_id"],
                )
            temporary = Path(entry["temporary_path"])
            temporary.unlink(missing_ok=True)
            if temporary.parent.exists():
                temporary.parent.rmdir()
        except (OSError, ValueError) as exc:
            evidence["error"] = str(exc)
        self.records.update(
            identity,
            {
                "status": "unknown",
                "effect": "unknown",
                "reconciliation": evidence,
                "error": "启动过的文件仅完成对账，无法证明由本操作成功发布；继续需新预览",
            },
            entry_id=entry["entry_id"],
        )
