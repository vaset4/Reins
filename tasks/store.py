from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.persistence import RuntimeStore, record_key
from tasks.ids import new_task_id, new_ulid, utc_now
from tasks.persistence import (
    TaskRevisionConflict as TaskRevisionConflict,
    validate_task_id,
)
from tasks.records import (
    TASK_STATUS_ACTIVE as TASK_STATUS_ACTIVE,
    TASK_STATUS_DONE as TASK_STATUS_DONE,
    TASK_STATE_VERSION,
    TaskRecord,
    TaskSummaryLayers,
    record_payload,
    render_compatible_summary,
)

MIN_SEDIMENT_ATTEMPTS = 0


class TaskStore:
    def __init__(self, data_root: Path | str) -> None:
        """绑定目标事实与派生索引；传参：数据根；返回：无。"""
        self._data_root = Path(data_root)
        self._db = RuntimeStore(data_root)

    def create_task(
        self,
        goal: str,
        *,
        task_id: str | None = None,
        is_inbox: bool = False,
    ) -> TaskRecord:
        """创建新目标或会话收件箱；传参：目标、可选身份与类别；返回：新记录。"""
        now = utc_now()
        record = TaskRecord(
            task_id=task_id or new_task_id(),
            goal=goal,
            created_at=now,
            updated_at=now,
            is_inbox=is_inbox,
        )
        with self._db.transaction():
            if self.load_task(record.task_id) is not None:
                raise FileExistsError(f"task already exists: {record.task_id}")
            self._write_task_record(record)
        return record

    def close(self) -> None:
        """短连接自动释放；传参：无；返回：无。"""

    def load_task(self, task_id: str) -> TaskRecord | None:
        """读取规范目标；传参：目标身份；返回：记录或不存在。"""
        validate_task_id(task_id)
        with self._db.snapshot() as source:
            row = source.get("task", task_id)
            return TaskRecord.from_dict(row) if row is not None else None

    def load_task_payload(self, task_id: str) -> dict[str, Any]:
        """读取含未知扩展字段的完整目标；传参：身份；返回：新映射。"""
        validate_task_id(task_id)
        with self._db.snapshot() as source:
            payload = source.get("task", task_id)
            if payload is None:
                raise FileNotFoundError(task_id)
            if not isinstance(payload, dict):
                raise ValueError(f"stored task payload is not an object: {task_id}")
            return payload

    def update_task_status(self, task_id: str, status: str) -> TaskRecord:
        """显式更新状态并保留未知字段，状态修改本身不构成完成验收。

        传参：task_id/status 为目标及状态；返回：更新后的任务
        """
        with self._db.transaction():
            payload = self.load_task_payload(task_id)
            record = TaskRecord.from_dict(payload)
            now = utc_now()
            updated = {
                **payload,
                "status": status,
                "updated_at": now,
                "state_version": TASK_STATE_VERSION,
                "revision": record.revision + 1,
                "status_source": "explicit_status_change",
                "completion": None,
            }
            if status == TASK_STATUS_DONE and not record.done_at:
                updated["done_at"] = now
            self._write_task_payload(updated)
            return TaskRecord.from_dict(updated)

    def complete_task(
        self,
        task_id: str,
        completion: dict[str, Any],
        *,
        expected_revision: int,
    ) -> TaskRecord:
        """在同一写入边界核对修订并提交目标完成及其依据。

        传参：task_id 为目标；completion 为已核实引用；expected_revision 为所见修订
        返回：完成后的任务；版本冲突时不修改原始记录
        """
        with self._db.transaction():
            payload = self.load_task_payload(task_id)
            record = TaskRecord.from_dict(payload)
            if record.revision != expected_revision:
                raise TaskRevisionConflict(
                    f"goal revision changed: expected {expected_revision}, current {record.revision}"
                )
            now = utc_now()
            updated = {
                **payload,
                "status": TASK_STATUS_DONE,
                "updated_at": now,
                "done_at": record.done_at or now,
                "state_version": TASK_STATE_VERSION,
                "revision": record.revision + 1,
                "status_source": "goal_completion",
                "completion": {**completion, "version": 1, "recorded_at": now},
            }
            result = TaskRecord.from_dict(updated)
            self._write_task_payload(updated)
            return result

    def update_sediment_status(
        self,
        task_id: str,
        *,
        done: bool | None = None,
        attempts: int | None = None,
        failed: bool | None = None,
    ) -> TaskRecord:
        """更新经验后处理状态；传参：目标和本次变更字段；返回：保留完成依据的新修订。"""
        updates: dict[str, Any] = {}
        if done is not None:
            updates["sediment_done"] = done
        if attempts is not None:
            if attempts < MIN_SEDIMENT_ATTEMPTS:
                raise ValueError("sediment attempts must be >= 0")
            updates["sediment_attempts"] = attempts
        if failed is not None:
            updates["sediment_failed"] = failed
        return self._update_metadata(task_id, updates)

    def update_task_refs(
        self,
        task_id: str,
        *,
        spec_refs: list[str] | None = None,
        skill_refs: list[str] | None = None,
    ) -> TaskRecord:
        """更新目标引用；传参：目标和资料/方法引用；返回：保留其他字段的新修订。"""
        updates: dict[str, Any] = {}
        if spec_refs is not None:
            updates["spec_refs"] = [str(ref) for ref in spec_refs]
        if skill_refs is not None:
            updates["skill_refs"] = [str(ref) for ref in skill_refs]
        return self._update_metadata(task_id, updates)

    def append_grant(self, task_id: str, grant: dict[str, object]) -> TaskRecord:
        """把已作出的授权加入真实目标，授权写入不能覆盖其他写者的完成依据。

        传参：task_id/grant 为目标与授权记录；返回：保存授权后的新修订
        """
        with self._db.transaction():
            payload = self.load_task_payload(task_id)
            record = TaskRecord.from_dict(payload)
            if grant.get("grant_id"):
                existing = next(
                    (
                        item
                        for item in record.grants
                        if item.get("grant_id") == grant["grant_id"]
                    ),
                    None,
                )
                if existing is not None:
                    if existing != grant:
                        raise ValueError("task grant identity has different content")
                    return record
            updated = {
                **payload,
                "grants": [*record.grants, dict(grant)],
                "updated_at": utc_now(),
                "revision": record.revision + 1,
            }
            self._write_task_payload(updated)
            return TaskRecord.from_dict(updated)

    def revoke_grant(self, task_id: str, grant: dict[str, object]) -> None:
        """撤回精确匹配的授权而保留目标其他字段；传参：目标与原权限；返回：无。"""
        with self._db.transaction():
            payload = self.load_task_payload(task_id)
            record = TaskRecord.from_dict(payload)
            grants = [item for item in record.grants if item != grant]
            if grants == record.grants:
                return
            self._write_task_payload(
                {
                    **payload,
                    "grants": grants,
                    "updated_at": utc_now(),
                    "revision": record.revision + 1,
                }
            )

    def _update_metadata(self, task_id: str, updates: dict[str, Any]) -> TaskRecord:
        """锁内重读后提交元数据变更；传参：目标及字段；返回：更新后的记录。"""
        with self._db.transaction():
            payload = self.load_task_payload(task_id)
            record = TaskRecord.from_dict(payload)
            updated = {
                **payload,
                **updates,
                "updated_at": utc_now(),
                "revision": record.revision + 1,
            }
            self._write_task_payload(updated)
            return TaskRecord.from_dict(updated)

    def list_tasks(
        self, status: str | None = None, limit: int | None = None
    ) -> list[TaskRecord]:
        """列出数据空间内的目标；传参：可选状态和数量；返回：最近更新优先的完整目标。"""
        with self._db.snapshot() as source:
            rows = sorted(
                source.list_raw(
                    "task", filters=None if status is None else {"status": status}
                ),
                key=lambda row: (row.payload["updated_at"], row.record_id),
                reverse=True,
            )
            result = []
            for row in rows[:limit]:
                payload = source.get("task", row.record_id)
                assert payload is not None
                result.append(TaskRecord.from_dict(payload))
            return result

    def get_inbox_tasks(self) -> list[TaskRecord]:
        """读取已有会话收件箱；传参：无；返回：最近更新优先的收件箱。"""
        return [record for record in self.list_tasks() if record.is_inbox]

    def require_task(self, task_id: str) -> TaskRecord:
        record = self.load_task(task_id)
        if record is None:
            raise FileNotFoundError(task_id)
        return record

    def task_dir(self, task_id: str) -> Path:
        """返回目标实际工作材料目录；传参：目标身份；返回：工作目录，不存聊天状态。"""
        validate_task_id(task_id)
        return self._data_root / "workspace" / task_id

    def append_journal(self, task_id: str, text: str) -> None:
        """追加目标进展正文；传参：目标及正文；返回：无。"""
        with self._db.transaction() as batch:
            identity = new_ulid()
            batch.put(
                "task_journal",
                identity,
                {
                    "entry_id": identity,
                    "task_id": task_id,
                    "content": f"\n## {utc_now()}\n\n{text}\n",
                },
                expected_revision=0,
            )

    def read_journal(self, task_id: str) -> str:
        """读取已提交的进展日志；传参：目标身份；返回：按顺序连接的正文。"""
        with self._db.snapshot() as source:
            return "".join(
                row["content"]
                for row in source.list("task_journal", filters={"task_id": task_id})
            )

    def read_summary(self, task_id: str) -> str:
        """读取进展摘要；传参：目标身份；返回：正文。"""
        layers = self.read_summary_layers(task_id)
        return layers.progress or layers.summary

    def update_summary(self, task_id: str, new_summary: str) -> None:
        """更新进展摘要；传参：目标及正文；返回：无。"""
        self.update_summary_layers(task_id, progress=new_summary.strip())

    def read_summary_layers(self, task_id: str) -> TaskSummaryLayers:
        """读取三个规范摘要槽位，组合视图不重复保存；传参：目标；返回：摘要视图。"""
        with self._db.snapshot() as source:
            row = source.get("task_summary", task_id)
        layers = (
            TaskSummaryLayers()
            if row is None
            else TaskSummaryLayers(
                intent=row["intent"],
                progress=row["progress"],
                resume_hint=row["resume_hint"],
            )
        )
        return TaskSummaryLayers(
            intent=layers.intent,
            progress=layers.progress,
            resume_hint=layers.resume_hint,
            summary=render_compatible_summary(layers),
        )

    def update_summary_layers(
        self,
        task_id: str,
        *,
        intent: str | None = None,
        progress: str | None = None,
        resume_hint: str | None = None,
    ) -> TaskSummaryLayers:
        """一次提交摘要和 Ledger 引用；传参：目标及变更槽位；返回：一致的新摘要。"""
        with self._db.transaction() as batch:
            old = self.read_summary_layers(task_id)
            values = [
                old.intent if intent is None else intent.strip(),
                old.progress if progress is None else progress.strip(),
                old.resume_hint if resume_hint is None else resume_hint.strip(),
            ]
            batch.put(
                "task_summary",
                task_id,
                {
                    "task_id": task_id,
                    "intent": values[0],
                    "progress": values[1],
                    "resume_hint": values[2],
                },
            )
            layers = self.read_summary_layers(task_id)
            self._record_summary_events(
                task_id,
                intent=intent,
                progress=progress,
                resume_hint=resume_hint,
                summary=layers.summary,
            )
            return layers

    def append_reflection(self, task_id: str, payload: dict[str, object]) -> None:
        """保留每次知识整理的原始依据；传参：目标及反思材料；返回：无。"""
        with self._db.transaction() as batch:
            identity = new_ulid()
            batch.put(
                "task_reflection",
                identity,
                {"reflection_id": identity, "task_id": task_id, "payload": payload},
                expected_revision=0,
            )

    def _record_summary_events(
        self,
        task_id: str,
        *,
        intent: str | None,
        progress: str | None,
        resume_hint: str | None,
        summary: str,
    ) -> None:
        from runtime.ledger import LedgerStore
        from runtime.ledger_writer import LedgerWriter

        writer = LedgerWriter(LedgerStore(self._data_root), source="tasks.store")
        for summary_kind, content in (
            ("intent", intent),
            ("progress", progress),
            ("resume_hint", resume_hint),
            ("summary", summary),
        ):
            if content is not None and content.strip():
                writer.record_summary_updated(
                    summary_kind,
                    content.strip(),
                    task_id=task_id,
                )

    def promote_inbox_to_task(
        self, inbox_id: str, new_task_id: str | None = None
    ) -> TaskRecord:
        """原子转正目标和附属记录；传参：收件箱与可选新身份；返回：正式目标。"""
        target_id = new_task_id or inbox_id
        validate_task_id(target_id)
        with self._db.transaction() as batch:
            payload = self.load_task_payload(inbox_id)
            record = TaskRecord.from_dict(payload)
            if not record.is_inbox:
                raise ValueError(f"not an inbox task: {inbox_id}")
            if target_id != inbox_id and self.load_task(target_id) is not None:
                raise FileExistsError(target_id)
            updated = {
                **payload,
                "task_id": target_id,
                "is_inbox": False,
                "updated_at": utc_now(),
                "revision": record.revision + 1,
            }
            self._write_task_payload(updated)
            if target_id != inbox_id:
                # 1. 【目标管理】【收件箱转正】目标身份和附属材料在同批次换绑，不留下半份目标
                self._move_attached_records(inbox_id, target_id)
                batch.delete("task", inbox_id)
            return TaskRecord.from_dict(updated)

    def cleanup_workspace(self, task_id: str) -> None:
        """工作材料保留供用户使用；传参：目标身份；返回：无。"""
        validate_task_id(task_id)

    def _write_task_record(self, record: TaskRecord) -> None:
        """发布新目标记录；传参：类型化目标；返回：无。"""
        self._write_task_payload(record_payload(record))

    def _write_task_payload(self, payload: dict[str, Any]) -> None:
        """发布数据空间级目标原件；传参：完整记录；返回：无，索引由提交服务派生。"""
        with self._db.transaction() as batch:
            batch.put("task", payload["task_id"], payload)

    def _move_attached_records(self, inbox_id: str, target_id: str) -> None:
        """随收件箱转正换绑摘要及日志；传参：原、新目标身份；返回：无，借用调用方批次。"""
        with self._db.transaction() as batch:
            summary = batch.get("task_summary", inbox_id)
            if summary is not None:
                batch.put("task_summary", target_id, {**summary, "task_id": target_id})
                batch.delete("task_summary", inbox_id)
            for row in batch.list("task_todo", filters={"task_id": inbox_id}):
                batch.put(
                    "task_todo",
                    record_key(target_id, str(row["idx"])),
                    {**row, "task_id": target_id},
                )
                batch.delete("task_todo", record_key(inbox_id, str(row["idx"])))
            for kind, identity_key in (
                ("task_journal", "entry_id"),
                ("task_reflection", "reflection_id"),
            ):
                for row in batch.list(kind):
                    if row["task_id"] == inbox_id:
                        batch.put(
                            kind, row[identity_key], {**row, "task_id": target_id}
                        )
