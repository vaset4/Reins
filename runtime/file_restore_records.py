"""【文件恢复】【执行原件】固定预览与逐项作业状态的短提交边界。

作者：xxx
时间：2026-09-30 16:00:00
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from approval.session import ApprovalMode
from runtime.lease import Lease
from runtime.file_records import ContentReference
from runtime.persistence import RuntimeStore
from tasks.ids import utc_now

TERMINAL_STATES = frozenset(
    {"completed", "partial", "cancelled", "failed", "needs_reconciliation"}
)


@dataclass(frozen=True, slots=True)
class RestoreAuthority:
    """当前宿主的权限事实，不能从旧运行租约或界面参数重建。"""

    lease: Lease
    mode: ApprovalMode
    instance_id: str
    blocked_by: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def evidence(self) -> dict[str, Any]:
        """冻结与文件权限相关的当前依据；参数：无；返回：无模型配置的权限材料。"""
        return {
            "mode": self.mode.value,
            "instance_id": self.instance_id,
            "task_id": self.lease.task_id,
            "fs": self.lease.capabilities.get("fs", {}),
        }


class RestoreRecords:
    """复用 RuntimeStore 原件，不持有文件 IO 或后台作业的全局提交锁。"""

    def __init__(self, database: RuntimeStore) -> None:
        """绑定已有数据空间；参数：存储；返回：无。"""
        self.database = database

    def require(self, kind: str, identity: str) -> dict[str, Any]:
        """读取固定身份的原件；参数：领域及身份；返回：独立记录，缺失明确报错。"""
        with self.database.snapshot() as source:
            row = source.get(kind, identity)
        if row is None:
            raise ValueError(f"恢复记录不存在：{identity}")
        return row

    def save(
        self, kind: str, identity: str, row: Mapping[str, Any], *, create: bool = False
    ) -> None:
        """先在锁外冻结载荷，再短批次发布；参数：身份及完整事实；返回：无。"""
        prepared = self.database.prepare_payload(row, workspace_id=row["workspace_id"])
        references = self.original_references(row)
        with (
            self.database.prepare_references(references) as originals,
            self.database.transaction() as batch,
        ):
            batch.put(
                kind,
                identity,
                prepared,
                session_id=row["session_id"],
                workspace_id=row["workspace_id"],
                expected_revision=0 if create else None,
            )
            for original in originals:
                batch.reference_prepared_content(original)

    def accept(
        self, plan: Mapping[str, Any], confirmation: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """同一计划只接纳一个操作；参数：原计划和明确确认；返回：操作及是否首次接纳。"""
        identity = "restore-" + plan["plan_id"]
        entries = [
            {
                **entry,
                "status": "pending" if entry["selected"] else "unchanged",
                "effect": "not_started",
            }
            for entry in plan["entries"]
        ]
        row = {
            "operation_id": identity,
            "plan_id": plan["plan_id"],
            "session_id": plan["session_id"],
            "workspace_id": plan["workspace_id"],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "confirmation": dict(confirmation),
            "status": "queued",
            "cancel_requested": False,
            "entries": entries,
            "error": None,
        }
        prepared = self.database.prepare_payload(row, workspace_id=row["workspace_id"])
        references = self.original_references(row)
        with (
            self.database.prepare_references(references) as originals,
            self.database.transaction() as batch,
        ):
            existing = batch.get("file_restore_operation", identity)
            if existing is not None:
                return existing, False
            batch.put(
                "file_restore_operation",
                identity,
                prepared,
                session_id=row["session_id"],
                workspace_id=row["workspace_id"],
                expected_revision=0,
            )
            for original in originals:
                batch.reference_prepared_content(original)
        return row, True

    def update(
        self, identity: str, changes: Mapping[str, Any], *, entry_id: str | None = None
    ) -> dict[str, Any]:
        """短批次更新操作或一项，不丢并发取消；参数：操作、变化及可选项；返回：新原件。"""
        references = self.original_references(changes)
        with (
            self.database.prepare_references(references) as originals,
            self.database.transaction() as batch,
        ):
            row = batch.get("file_restore_operation", identity)
            if row is None:
                raise ValueError("恢复操作不存在")
            if entry_id is None:
                updated = {**row, **changes, "updated_at": utc_now()}
            else:
                entries = [
                    {**entry, **changes} if entry["entry_id"] == entry_id else entry
                    for entry in row["entries"]
                ]
                updated = {**row, "entries": entries, "updated_at": utc_now()}
            batch.put(
                "file_restore_operation",
                identity,
                updated,
                session_id=row["session_id"],
                workspace_id=row["workspace_id"],
            )
            for original in originals:
                batch.reference_prepared_content(original)
        return updated

    def original_references(self, row: Mapping[str, Any]) -> list[ContentReference]:
        """收集领域原件引用，调用方锁外固定句柄直到提交；参数：计划或逐项变更；返回：去重引用。"""
        fields = (
            "before",
            "after",
            "target",
            "current",
            "expected",
            "before_restore",
            "actual_displaced",
            "after_restore",
        )
        entries = [
            entry
            for entry in row.get("entries", [])
            if entry.get("selected") and entry.get("executable")
        ]
        references = {}
        for entry in [row, *entries]:
            for field_name in fields:
                state = entry.get(field_name) or {}
                reference = state.get("protected_content") or state.get("content")
                if reference:
                    references[reference["path"]] = ContentReference.from_mapping(
                        reference
                    )
        return list(references.values())

    def cancel(self, identity: str) -> dict[str, Any]:
        """取消仅标记尚未开始的责任；参数：操作身份；返回：当前事实，已完成项不回滚。"""
        row = self.require("file_restore_operation", identity)
        if row["status"] in TERMINAL_STATES:
            return row
        return self.update(identity, {"cancel_requested": True})


def operation_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """输出进度白名单，避免内部敏感摘要和权限材料泄漏；参数：原件；返回：界面状态。"""
    fields = (
        "entry_id",
        "path",
        "destination",
        "choice",
        "action",
        "status",
        "effect",
        "error",
        "backup_path",
        "temporary_path",
        "reconciliation",
        "restore_point_id",
    )
    return {
        key: row[key]
        for key in (
            "operation_id",
            "plan_id",
            "session_id",
            "workspace_id",
            "status",
            "created_at",
            "updated_at",
            "cancel_requested",
            "error",
        )
    } | {
        "entries": [
            {key: entry[key] for key in fields if key in entry}
            for entry in row["entries"]
        ]
    }
