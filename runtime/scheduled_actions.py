"""以运行身份接纳时间意图和通知，不接受模型自造权限快照。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from llm.base import LLMClient
from llm.profiles import load_model_profiles
from llm.public_config import (
    PUBLIC_MODEL_FIELDS,
    ephemeral_api_key,
    public_model_config,
)
from runtime.session_message_store import SessionMessageStore
from runtime.tool_operations import ToolOperation
from runtime.types import RunContext, RunToolsResult
from runtime.workspaces import WorkspaceStore
from schedules.notifications import NotificationStore
from schedules.occurrences import OccurrenceRecord, OccurrenceStore
from schedules.persistence import SCHEDULE_DISPATCH_LOCK, claim_file
from schedules.store import ScheduleRecord, ScheduleStore
from schedules.timing import TimeIntent, format_instant

SCHEDULED_ACTIONS = frozenset({"schedule", "notification_send", "notification_status"})


class ScheduledActions:
    """只负责持久意图和查询，何时调用以及工作内容仍由当前模型决定。"""

    def __init__(self, run: RunContext, *, data_root: Path, client: LLMClient) -> None:
        """绑定真实运行和模型配置；传参：上下文、存储与客户端；返回：无。"""
        self.run, self.data_root, self.client = run, data_root, client

    def execute(self, call: ToolOperation) -> RunToolsResult:
        """处理已经通过执行边界的动作；传参：操作身份和合法参数；返回：持久接纳或实际状态。"""
        try:
            if call.tool_name.startswith("notification_"):
                result = self._notification(call)
            elif call.args["action"] == "list":
                result = self._list(call.args.get("schedule_id"))
            else:
                capability = self.run.capability_lease.capabilities.get(
                    "background_run"
                )
                if (
                    not isinstance(capability, Mapping)
                    or capability.get("enabled") is not True
                ):
                    raise ValueError(
                        "this run has no permission to create or change persistent background work"
                    )
                with claim_file(self.data_root / SCHEDULE_DISPATCH_LOCK) as acquired:
                    if not acquired:
                        raise ValueError(
                            "schedule dispatch is in progress; inspect and retry this management action"
                        )
                    result = self._change(call)
        except (ValueError, FileNotFoundError) as exc:
            return RunToolsResult.error_result(action=call.tool_name, error=str(exc))
        return RunToolsResult.ok(
            action=call.tool_name,
            content=json.dumps(result, ensure_ascii=False),
            meta=result,
        )

    def _change(self, call: ToolOperation) -> dict[str, Any]:
        """在调度交接锁内变更未来意图或显式接续；传参：当前操作；返回：实际记录。"""
        args = call.args
        with closing(ScheduleStore(self.data_root)) as store:
            action = args["action"]
            if action == "create":
                return asdict(self._create(store, call))
            if action == "resume_occurrence":
                return asdict(self._resume_occurrence(store, call))
            identity = str(args["schedule_id"])
            job = store.load_schedule(identity)
            if job is None:
                raise FileNotFoundError(identity)
            if action == "update":
                return asdict(store.update(identity, **self._changes(job, args)))
            changes = {
                "pause": {"paused": True},
                "resume": {"enabled": True, "paused": False},
                "cancel": {"enabled": False},
            }
            return asdict(store.update(identity, **changes[str(action)]))

    def _create(self, store: ScheduleStore, call: ToolOperation) -> ScheduleRecord:
        """用当前真实权限创建幂等计划，模型只提供工作与时间；传参：写者和操作；返回：计划。"""
        identity = f"schedule-{call.operation_id}"
        existing = store.load_schedule(identity)
        if existing is not None:
            return existing
        args = call.args
        model = self._model_config(args) if args["kind"] == "work" else {}
        # 1. 【定时工作】【保存归属】工作区来自已接纳会话，模型参数不能改写文件执行位置
        workspace = WorkspaceStore(self.data_root).for_session(self.run.session_id)
        return store.create_schedule(
            identity,
            str(args["time"]),
            name=str(args["name"]),
            prompt=str(args["prompt"]),
            workspace_id=workspace.workspace_id,
            kind=str(args["kind"]),
            timezone_name=str(args["timezone"]),
            target_task_id=None
            if args.get("knowledge_origin")
            else self.run.focus_task_id,
            source_session_id=self.run.session_id,
            source_run_id=self.run.run_id,
            model_config=model,
            source_entry_id=self._source_entry_id(),
            capabilities=json.loads(json.dumps(self.run.capability_lease.capabilities)),
            max_steps=self.run.capability_lease.max_steps,
            max_tokens=self.run.capability_lease.max_tokens,
            knowledge_origin=cast(dict[str, Any] | None, args.get("knowledge_origin")),
        )

    def _source_entry_id(self) -> str:
        """冻结接纳动作之前的真实会话分支；传参：无；返回：来源边界，不用后来的输入补造来源。"""
        view = SessionMessageStore(self.data_root).materialize(self.run.session_id)
        anchor = view.entries[-1].parent_id if view.entries else None
        if anchor is None:
            raise ValueError("scheduled work requires prior session evidence")
        return anchor

    def _model_config(self, args: Mapping[str, object]) -> dict[str, object]:
        """固定当前或用户选定的已保存模型，临时密钥不能进入计划；传参：公开参数；返回：公开配置。"""
        name = args.get("profile_name")
        if name is not None:
            profiles = load_model_profiles()
            if str(name) not in profiles.profiles:
                raise ValueError(f"model profile not found: {name}")
            profile = profiles.profiles[str(name)]
            return {
                **{
                    key: value
                    for key, value in profile.as_config().items()
                    if key in PUBLIC_MODEL_FIELDS
                },
                "profile_name": str(name),
            }
        if ephemeral_api_key(self.client) is not None:
            raise ValueError(
                "scheduled model work needs a saved model profile; temporary CLI credentials cannot be persisted"
            )
        return public_model_config(self.client)

    def _changes(
        self, job: ScheduleRecord, args: Mapping[str, object]
    ) -> dict[str, Any]:
        """校验未来计划的新内容，已接纳发生保留旧快照；传参：计划和变更；返回：允许字段。"""
        mapping = {
            "name": "name",
            "prompt": "prompt",
            "kind": "kind",
            "time": "cron",
            "timezone": "timezone_name",
        }
        changes = {
            destination: args[source]
            for source, destination in mapping.items()
            if source in args
        }
        if "prompt" in args:
            source = {
                "source_session_id": self.run.session_id,
                "source_run_id": self.run.run_id,
                "source_entry_id": self._source_entry_id(),
            }
            changes.update(source)
            if job.knowledge_origin is not None:
                changes["knowledge_origin"] = {
                    **job.knowledge_origin,
                    **source,
                    "objective": str(args["prompt"]),
                }
        if "profile_name" in args:
            changes["model_config"] = self._model_config(args)
        if "time" in args or "timezone" in args:
            intent = TimeIntent(
                str(args.get("time", job.cron)),
                str(args.get("timezone", job.timezone_name)),
            )
            intent.validate()
            changes["next_run_at"] = format_instant(
                intent.first_at(datetime.now(timezone.utc))
            )
        if changes.get("kind") == "work" and job.kind == "reminder":
            changes["model_config"] = self._model_config(args)
            changes["capabilities"] = json.loads(
                json.dumps(self.run.capability_lease.capabilities)
            )
        return changes

    def _resume_occurrence(
        self, store: ScheduleStore, call: ToolOperation
    ) -> OccurrenceRecord:
        """向原发生交付带来源的接续输入，不冒充用户的独立确认；传参：计划写者和操作；返回：排队发生。"""
        occurrences = OccurrenceStore(self.data_root)
        identity = str(call.args["occurrence_id"])
        record = occurrences.load(identity)
        if record is None:
            raise FileNotFoundError(identity)
        if record.resume_request_id == call.operation_id:
            return record
        if record.status != "settled" or record.result_status not in {
            "paused",
            "failed",
            "timeout",
        }:
            raise ValueError(
                "only paused or failed occurrences can be explicitly resumed"
            )
        input_id = f"schedule-reply-{call.operation_id}"
        message = f"来自会话 {self.run.session_id} 的接续说明（输入引用 {self.run.payload.get('input_message_id', '')}）：\n{call.args['message']}"
        SessionMessageStore(self.data_root).accept_input(
            record.session_id, message, input_id=input_id, input_source="agent"
        )
        updated = occurrences.request_resume(
            identity, request_id=call.operation_id, input_id=input_id
        )
        store.update(record.schedule_id, enabled=True, paused=False)
        return updated

    def _list(self, schedule_id: object) -> dict[str, Any]:
        """读取计划、发生及结果引用，暂停与通知状态独立；传参：可选计划编号；返回：实际记录。"""
        with closing(ScheduleStore(self.data_root)) as store:
            jobs = store.list_all_schedules()
        records = OccurrenceStore(self.data_root).list_all()
        return {
            "local_now": datetime.now().astimezone().isoformat(),
            "utc_now": datetime.now(timezone.utc).isoformat(),
            "schedules": [
                asdict(job)
                for job in jobs
                if schedule_id is None or job.schedule_id == schedule_id
            ],
            "occurrences": [
                asdict(row)
                for row in records
                if schedule_id is None or row.schedule_id == schedule_id
            ],
        }

    def _notification(self, call: ToolOperation) -> dict[str, Any]:
        """持久接纳通知或返回渠道回执，不能由模型替用户标已读；传参：真实操作；返回：记录。"""
        store = NotificationStore(self.data_root)
        if call.tool_name == "notification_send":
            record = store.enqueue(
                f"notice-{call.operation_id}",
                title=str(call.args["title"]),
                message=str(call.args["message"]),
                source={
                    "session_id": self.run.session_id,
                    "run_id": self.run.run_id,
                    "operation_id": call.operation_id,
                },
            )
            return {"accepted": True, **asdict(record)}
        identity = call.args.get("notification_id")
        if identity is None:
            return {
                "notifications": [
                    asdict(row) for row in store.list_all(unread_only=True)
                ]
            }
        found = store.load(str(identity))
        if found is None:
            raise FileNotFoundError(str(identity))
        return asdict(found)
