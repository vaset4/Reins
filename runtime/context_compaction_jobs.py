"""【上下文】【后台整理】冻结来源并复用持久调度，提交身份与恢复结果可对账。

作者：xxx
时间：2026-10-01 12:00:00
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from context.compaction import CompactionMaterial
from llm.base import LLMClient
from llm.public_config import ephemeral_api_key, public_model_config
from runtime.persistence import RuntimeStore
from runtime.session_compaction import (
    SessionCompactionStore,
    SummarySource,
    source_digest,
)
from runtime.session_message_store import SessionMessageStore
from runtime.types import RunContext
from runtime.workspaces import WorkspaceStore
from schedules.store import ScheduleStore

CONTROL_ID = "automatic-history-control"


class ContextCompactionJobs:
    """管理整理来源和结果；模型调用及等待始终位于会话写锁之外。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定已有空间；参数：数据根；返回：无。"""
        self.data_root = Path(data_root)
        self.database = RuntimeStore(data_root)

    def accept(
        self,
        material: CompactionMaterial,
        context: Mapping[str, object],
        *,
        run: RunContext,
        client: LLMClient,
    ) -> dict[str, Any]:
        """冻结来源和当前模型后接纳一次后台整理；参数：材料、模型范围、运行和客户端；返回：持久工作状态。"""
        if ephemeral_api_key(client) is not None:
            return {
                "status": "not_accepted",
                "reason": "background_compaction_requires_saved_model_credentials",
            }
        capability = run.capability_lease.capabilities.get("background_run")
        if not isinstance(capability, Mapping) or capability.get("enabled") is not True:
            return {
                "status": "not_accepted",
                "reason": "background_run_permission_disabled",
            }
        source = material.source
        model = public_model_config(client)
        frozen = {
            "source_session_id": source.view.session_id,
            "source_entry_id": source.view.leaf_id,
            "covered_count": source.covered_count,
            "source_sha256": source_digest(
                source.view.messages[: source.covered_count]
            ),
            "first_kept_message_id": source.view.messages[
                source.covered_count
            ].message_id,
            "previous_summary_id": source.previous.summary_id
            if source.previous
            else None,
        }
        identity = hashlib.sha256(
            json.dumps({**frozen, "model": model}, sort_keys=True).encode()
        ).hexdigest()
        job_id = f"compaction-{identity}"
        workspace = WorkspaceStore(self.data_root).for_session(run.session_id)
        with self.database.transaction() as batch:
            control = batch.get("context_compaction_job", CONTROL_ID)
            if control is not None and control["enabled"] is False:
                return {"status": "not_accepted", "reason": "automatic_history_paused"}
            existing = batch.get("context_compaction_job", job_id)
            if existing is not None:
                return existing
            # 1. 【上下文】【后台接纳】同一前驱只允许一个生产者，新尾部留待下一次已发布版本
            pending = [
                row
                for row in batch.list(
                    "context_compaction_job", session_id=run.session_id
                )
                if row.get("record_type") == "job"
                and row["status"] in {"queued", "running"}
                and row["previous_summary_id"] == frozen["previous_summary_id"]
            ]
            if pending:
                return pending[0]
            record: dict[str, object] = {
                "schema_version": 1,
                "record_type": "job",
                "job_id": job_id,
                **frozen,
                "status": "queued",
                "source_run_id": run.run_id,
                "workspace_id": workspace.workspace_id,
                "budget_run_id": run.budget_run_id or run.run_id,
                "lease": asdict(run.capability_lease),
                "accepted_at": datetime.now(timezone.utc).isoformat(),
                "summary_id": None,
                "request_ids": [],
                "error": None,
                "history_representation_version": 1,
                "cancel_requested": False,
                "schedule_id": f"schedule-{job_id}",
            }
            batch.put(
                "context_compaction_job", job_id, record, session_id=run.session_id
            )
            schedules = ScheduleStore(self.data_root)
            schedules.create_schedule(
                f"schedule-{job_id}",
                f"at:{datetime.now(timezone.utc).isoformat()}",
                workspace_id=workspace.workspace_id,
                name="整理会话历史",
                prompt="整理已冻结的旧历史并核对四级表示",
                source_session_id=run.session_id,
                source_run_id=run.run_id,
                source_entry_id=source.view.leaf_id,
                model_config=model,
                capabilities=dict(run.capability_lease.capabilities),
                max_steps=run.capability_lease.max_steps,
                max_tokens=run.capability_lease.max_tokens,
                context_compaction_origin={"job_id": job_id},
            )
            schedules.close()
        return record

    def status(self, *, session_id: str | None = None) -> dict[str, Any]:
        """只读查看自动整理和工作状态；参数：可选会话；返回：默认启用及真实工作，不创建控制记录。"""
        if not (self.data_root / "space.json").exists():
            return {"enabled": True, "jobs": []}
        with self.database.snapshot() as source:
            control = source.get("context_compaction_job", CONTROL_ID)
            jobs = [
                row
                for row in source.list("context_compaction_job", session_id=session_id)
                if row.get("record_type") == "job"
            ]
        return {
            "enabled": control["enabled"] if control is not None else True,
            "jobs": jobs,
        }

    def configure(self, *, enabled: bool) -> dict[str, Any]:
        """暂停只影响后续接纳，已接纳工作继续；参数：自动整理开关；返回：持久控制与工作状态。"""
        if type(enabled) is not bool:
            raise ValueError("automatic context compaction must be boolean")
        with self.database.transaction() as batch:
            batch.put(
                "context_compaction_job",
                CONTROL_ID,
                {"schema_version": 1, "record_type": "control", "enabled": enabled},
            )
        return self.status()

    def cancel(self, job_id: str) -> dict[str, Any]:
        """取消排队工作或请求运行中停止；参数：工作身份；返回：实际状态，运行中不提前称已停止。"""
        with self.database.transaction() as batch:
            row = self.load(job_id)
            if row["status"] in {"published", "cancelled", "failed"}:
                return row
            schedules = ScheduleStore(self.data_root)
            schedules.update(row["schedule_id"], enabled=False)
            schedules.close()
            running = any(
                item["schedule_id"] == row["schedule_id"]
                and item["status"] == "running"
                for item in batch.list("schedule_occurrence")
            )
            status = (
                row["status"] if running or row["status"] == "running" else "cancelled"
            )
            return self.update(job_id, cancel_requested=True, status=status)

    def cancellation_requested(self, job_id: str) -> bool:
        """执行边界重读用户取消意图；参数：工作身份；返回：是否停止新模型派发及发布。"""
        return self.load(job_id).get("cancel_requested") is True

    def load(self, job_id: str) -> dict[str, Any]:
        """读取真实工作状态；参数：工作身份；返回：记录，缺失直接失败。"""
        with self.database.snapshot() as source:
            row = source.get("context_compaction_job", job_id)
        if row is None:
            raise FileNotFoundError(job_id)
        if row.get("schema_version") != 1 or row.get("job_id") != job_id:
            raise ValueError("invalid context compaction job")
        return row

    def update(self, job_id: str, **changes: object) -> dict[str, Any]:
        """更新工作回执而不移动来源水位；参数：身份和状态变动；返回：已保存记录。"""
        with self.database.transaction() as batch:
            row = self.load(job_id)
            updated = {**row, **changes}
            batch.put(
                "context_compaction_job",
                job_id,
                updated,
                session_id=row["source_session_id"],
            )
        return updated

    def material(self, row: Mapping[str, Any]) -> CompactionMaterial:
        """重启后按冻结分支读取原件，不重放工具；参数：持久工作；返回：核对过来源版本的整理材料。"""
        messages = SessionMessageStore(self.data_root)
        view = messages.materialize(
            row["source_session_id"], at_entry_id=row["source_entry_id"]
        )
        count = row["covered_count"]
        if type(count) is not int or not 0 < count < len(view.messages):
            raise ValueError("background compaction source range changed")
        if (
            source_digest(view.messages[:count]) != row["source_sha256"]
            or view.messages[count].message_id != row["first_kept_message_id"]
        ):
            raise ValueError("background compaction original content changed")
        store = SessionCompactionStore(messages)
        previous = (
            store.read(row["previous_summary_id"], view)
            if row["previous_summary_id"]
            else None
        )
        return CompactionMaterial(SummarySource(view, count, previous), "")

    def published(self, row: Mapping[str, Any]) -> str | None:
        """区分已提交与显示回执未更新；参数：工作；返回：已发布身份，不再次调用模型。"""
        with self.database.snapshot() as source:
            records = source.list(
                "session_compaction", session_id=row["source_session_id"]
            )
        return next(
            (
                record["summary_id"]
                for record in records
                if record.get("compaction_job_id") == row["job_id"]
            ),
            None,
        )
