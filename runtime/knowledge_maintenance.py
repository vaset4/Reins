"""【知识维护】【来源进度】复用定时交付保存自动维护的冻结来源与实际处理结论。

作者：xxx
时间：2026-10-01 14:00:00
"""

from __future__ import annotations

from contextlib import closing
from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from pathlib import Path

from approval.session import ApprovalMode, ApprovalSession
from llm.base import LLMClient
from llm.messages import AssistantMessage, ToolCallPart, ToolResultMessage, UserMessage
from llm.public_config import ephemeral_api_key, public_model_config
from runtime.persistence import RuntimeStore, record_key
from runtime.session_compaction import source_digest
from runtime.session_message_store import MaterializedSession, SessionMessageStore
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from schedules.store import ScheduleStore
from tasks.ids import new_ulid, utc_now

KIND = "knowledge_maintenance"
CONTROL_ID = "enable-boundary"
WORK_KIND = "knowledge_maintenance"
FINISHED = frozenset({"completed", "no_op"})
PROMPT = (
    "处理已冻结的新来源及相关知识变化。用knowledge_read读取来源，按需用memory_query查重及核对旧版，"
    "可用file_read补读当前文件。自主决定哪些内容值得长期保存、如何验证，以及是否修订或替代。"
    "用户偏好/规则必须引用真实用户输入，工具观察与推断保留各自来源，不把临时进度升级为长期事实。"
    "仅维护本工作区或原会话知识；不编辑项目文件、文档、技能，不按时间删除。"
    "记忆写入复用memory_manage的expected_version；冲突先读新版，不能覆盖用户编辑。"
    "原来源用origin_inputs/origin_results，当前file_read证据用tool_results。"
    "提交knowledge_finish前需读齐本工作冻结的全部message_ids，包括非用户消息。"
    "knowledge_read(action=messages)提供完整冻结消息集合；user_inputs只读真实用户消息子集，operations不推进消息覆盖。"
    "若冻结集合全是用户消息且已读齐，无须额外读取messages；知识核验和写入顺序由你决定。"
    "完成后必须调用knowledge_finish，说明已核验来源及结论。没有值得保存的知识可明确no_op；"
    "仅回答文本、格式错误、未读完来源或写入失败都不表示完成。索引失败但committed=true先对账，勿重复写。"
)


def automatic_origin(context: RunContext) -> dict[str, Any] | None:
    """识别宿主已接纳的维护工作；参数：运行；返回：冻结来源或空，不接受模型参数。"""
    origin = context.payload.get("knowledge_origin")
    return (
        origin
        if isinstance(origin, dict) and origin.get("work_kind") == WORK_KIND
        else None
    )


class KnowledgeMaintenance:
    """来源和处理进度的唯一写者；执行、预算与恢复仍由既有调度器持有。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定数据空间；参数：数据根；返回：无，不启动线程或模型。"""
        self.root = Path(data_root)
        self.database = RuntimeStore(data_root)
        self.messages = SessionMessageStore(data_root)

    def initialize(self, context: RunContext | None = None) -> None:
        """首次启用只冻结边界，当前已接纳真实输入仍属于新来源；参数：可选当前运行；返回：无。"""
        if context is not None and context.trigger != Trigger.USER:
            return
        with self.database.transaction() as batch:
            existing = batch.get(KIND, CONTROL_ID)
            if existing is not None and (
                not existing["enabled"] or existing["sequence"] is not None
            ):
                return
            boundary = batch.sequence
            if context is not None and context.payload.get("input_message_id"):
                input_id = str(context.payload["input_message_id"])
                raw = batch.raw(
                    "session_entry", record_key(context.session_id, input_id)
                )
                if raw is None:
                    entries = self.messages.materialize(context.session_id).entries
                    entry = next(
                        (
                            entry
                            for entry in entries
                            if isinstance(entry.message, UserMessage)
                            and entry.message.message_id == input_id
                            and entry.input_source != "agent"
                        ),
                        None,
                    )
                    if entry is not None:
                        raw = batch.raw(
                            "session_entry",
                            record_key(context.session_id, entry.entry_id),
                        )
                if raw is None or raw.location is None:
                    raise ValueError(
                        "knowledge enable boundary cannot resolve its accepted user input"
                    )
                boundary = raw.location.sequence - 1
            batch.put(
                KIND,
                CONTROL_ID,
                {
                    "record_type": "boundary",
                    "enabled_at": utc_now(),
                    "sequence": boundary,
                    "enabled": True,
                },
            )

    def configure(self, *, enabled: bool) -> dict[str, Any]:
        """暂停只影响后续接纳，恢复不改首次边界；参数：启用状态；返回：持久控制状态。"""
        if type(enabled) is not bool:
            raise ValueError("automatic knowledge maintenance must be boolean")
        with self.database.transaction() as batch:
            control = batch.get(KIND, CONTROL_ID)
            if control is None:
                control = {
                    "record_type": "boundary",
                    "enabled_at": None,
                    "sequence": None,
                }
            batch.put(KIND, CONTROL_ID, {**control, "enabled": enabled})
        if enabled:
            self.initialize()
        return self.status()

    def observe(
        self,
        context: RunContext,
        *,
        client: LLMClient,
        approval_session: ApprovalSession | None = None,
    ) -> dict[str, Any] | None:
        """在真实运行的新证据边界合并来源，仅接纳工作不调用模型；参数：当前运行/模型；返回：工作或空。"""
        if context.trigger != Trigger.USER or automatic_origin(context) is not None:
            return None
        if (
            approval_session is not None
            and approval_session.mode is ApprovalMode.READ_ONLY
        ):
            return None
        capability = context.capability_lease.capabilities.get("background_run")
        if not isinstance(capability, Mapping) or capability.get("enabled") is not True:
            return None
        if not self.messages.exists(context.session_id) and not context.payload.get(
            "input_message_id"
        ):
            return None
        self.initialize(context)
        with self.database.snapshot() as source:
            control = source.get(KIND, CONTROL_ID)
            assert control is not None
            if not control["enabled"]:
                return None
        view = complete_source(self.messages.materialize(context.session_id))
        if not view.messages:
            return None
        projected_ids = {message.message_id for message in view.messages}
        with self.database.snapshot() as source:
            eligible = set()
            for entry in view.entries:
                raw = source.raw(
                    "session_entry", record_key(view.session_id, entry.entry_id)
                )
                if (
                    entry.message is not None
                    and entry.message.message_id in projected_ids
                    and raw is not None
                    and raw.location is not None
                ):
                    if (
                        raw.location.sequence > control["sequence"]
                        and entry.input_source != "agent"
                    ):
                        eligible.add(entry.message.message_id)
        return self.admit_sources(
            context,
            client=client,
            view=view,
            message_ids=eligible,
            reason="new_source",
            automatic=True,
            approval_session=approval_session,
        )

    def admit_sources(
        self,
        context: RunContext,
        *,
        client: LLMClient,
        view: MaterializedSession,
        message_ids: set[str],
        reason: str,
        automatic: bool = False,
        approval_session: ApprovalSession | None = None,
    ) -> dict[str, Any] | None:
        """接纳模型/用户明确选定旧来源或新增范围，排队源合并且已冻结源不改写；参数：来源与原因；返回：工作。"""
        if view.session_id != context.session_id:
            raise ValueError(
                "knowledge sources must belong to the current authorized session"
            )
        capability = context.capability_lease.capabilities.get("background_run")
        if not isinstance(capability, Mapping) or capability.get("enabled") is not True:
            raise ValueError(
                "this run cannot accept persistent background knowledge work"
            )
        if not message_ids.issubset({message.message_id for message in view.messages}):
            raise ValueError("selected knowledge sources are outside the frozen branch")
        with (
            self.database.transaction() as batch,
            closing(ScheduleStore(self.root)) as schedules,
        ):
            # 1. 【知识维护】【接纳权限】新建和扩展来源均使用宿主当前模式，既有工作不被追溯取消
            if (
                approval_session is not None
                and approval_session.mode is ApprovalMode.READ_ONLY
            ):
                return None
            rows = batch.list(KIND, session_id=context.session_id)
            # 1. 【知识维护】【失败处置】自动接纳不复活旧失败；显式选源可基于修复后的锚点重新核验
            covered = {
                identity
                for row in rows
                if automatic
                or row.get("state") in FINISHED | {"queued", "running", "cancelling"}
                for identity in row.get("message_ids", [])
            }
            selected = message_ids - covered
            targets = [
                row
                for row in rows
                if row.get("record_type") == "evidence" and not row.get("work_id")
            ]
            if not selected and not targets:
                return None
            workspace = WorkspaceStore(self.root).for_session(context.session_id)
            occurrences = {
                row["schedule_id"] for row in batch.list("schedule_occurrence")
            }
            pending = next(
                (
                    row
                    for row in rows
                    if row.get("state") == "queued"
                    and row["schedule_id"] not in occurrences
                    and set(row["message_ids"]).issubset(
                        {message.message_id for message in view.messages}
                    )
                ),
                None,
            )
            if pending is not None:
                selected.update(pending["message_ids"])
            if ephemeral_api_key(client) is not None:
                if automatic:
                    self.record_unaccepted(context, message_ids=selected)
                    return None
                raise ValueError(
                    "automatic knowledge maintenance needs a saved model credential"
                )
            model_config = public_model_config(client) if pending is None else None
            identity = pending["work_id"] if pending else f"knowledge-{new_ulid()}"
            chosen = tuple(
                message for message in view.messages if message.message_id in selected
            )
            row = {
                "record_type": "work",
                "work_id": identity,
                "schedule_id": identity,
                "state": "queued",
                "source_session_id": context.session_id,
                "source_run_id": context.run_id,
                "source_entry_id": view.leaf_id,
                "source_sha256": source_digest(chosen),
                "message_ids": [message.message_id for message in chosen],
                "workspace_id": workspace.workspace_id,
                "model_config": pending["model_config"] if pending else model_config,
                "reasons": sorted(
                    {
                        reason,
                        *(pending["reasons"] if pending else []),
                        *(["file_source_changed"] if targets else []),
                    }
                ),
                "created_at": pending["created_at"] if pending else utc_now(),
                "updated_at": utc_now(),
                "read_message_ids": [],
                "outcome": None,
                "error": None,
                "work_kind": WORK_KIND,
            }
            row["verification_targets"] = [
                *(pending.get("verification_targets", []) if pending else []),
                *targets,
            ]
            batch.put(KIND, identity, row, session_id=context.session_id)
            for target in targets:
                batch.put(
                    KIND,
                    target["evidence_id"],
                    {**target, "work_id": identity},
                    session_id=context.session_id,
                )
            if pending is not None:
                schedules.update(
                    identity,
                    source_entry_id=view.leaf_id,
                    source_run_id=row["source_run_id"],
                    knowledge_origin=row,
                )
            else:
                schedules.create_schedule(
                    identity,
                    f"at:{utc_now()}",
                    name="自动知识维护",
                    prompt=PROMPT,
                    workspace_id=workspace.workspace_id,
                    source_session_id=context.session_id,
                    source_run_id=context.run_id,
                    source_entry_id=view.leaf_id,
                    model_config=model_config,
                    capabilities=maintenance_capabilities(context),
                    knowledge_origin=row,
                    max_steps=context.capability_lease.max_steps,
                    max_tokens=context.capability_lease.max_tokens,
                )
            admission_id = record_key("admission", context.session_id)
            admission = batch.get(KIND, admission_id)
            if automatic and admission is not None:
                batch.put(
                    KIND,
                    admission_id,
                    {
                        **admission,
                        "state": "accepted",
                        "reason": None,
                        "work_id": identity,
                        "updated_at": utc_now(),
                    },
                    session_id=context.session_id,
                )
            return row

    def record_unaccepted(self, context: RunContext, *, message_ids: set[str]) -> None:
        """临时凭据只阻止后台接纳，不消费来源或中止主聊天；参数：真实运行/待处理消息；返回：无。"""
        identity = record_key("admission", context.session_id)
        row = {
            "record_type": "admission",
            "state": "not_accepted",
            "reason": "background_knowledge_requires_saved_model_credentials",
            "source_session_id": context.session_id,
            "source_run_id": context.run_id,
            "source_count": len(message_ids),
            "source_message_ids": sorted(message_ids),
        }
        with self.database.transaction() as batch:
            previous = batch.get(KIND, identity)
            if previous is not None and all(
                previous.get(key) == value for key, value in row.items()
            ):
                return
            batch.put(
                KIND,
                identity,
                {**row, "updated_at": utc_now()},
                session_id=context.session_id,
            )

    def load(self, work_id: str) -> dict[str, Any]:
        """读取实际工作；参数：身份；返回：记录，不存在明确失败。"""
        with self.database.snapshot() as source:
            row = source.get(KIND, work_id)
        if row is None or row.get("record_type") != "work":
            raise ValueError(f"knowledge work not found: {work_id}")
        return row

    def update(self, work_id: str, **changes: Any) -> dict[str, Any]:
        """短事务记录运行或恢复进度；参数：工作与真实变化；返回：最新记录。"""
        with self.database.transaction() as batch:
            current = self.load(work_id)
            if changes.get("state") == "running" and current.get("cancel_requested"):
                return current
            row = {**current, **changes, "updated_at": utc_now()}
            # 【知识维护】【并发阅读】冻结来源的已读范围只累积，迟到页面不能覆盖另一页已保存的进度
            if "read_message_ids" in changes:
                row["read_message_ids"] = sorted(
                    {*current["read_message_ids"], *changes["read_message_ids"]}
                )
            batch.put(KIND, work_id, row, session_id=row["source_session_id"])
            return row

    def status(self, *, session_id: str | None = None) -> dict[str, Any]:
        """查看首启边界与工作，不将旧未处理来源计为no-op；参数：可选会话；返回：真实进度。"""
        with self.database.snapshot() as source:
            boundary = source.get(KIND, CONTROL_ID)
            works = [
                row
                for row in source.list(KIND, session_id=session_id)
                if row.get("record_type") == "work"
            ]
            admissions = [
                row
                for row in source.list(KIND, session_id=session_id)
                if row.get("record_type") == "admission"
            ]
        from runtime.knowledge_worker import committed_changes
        from runtime.tool_operations import ToolOperationStore

        for row in works:
            if row.get("worker_session_id"):
                commits, errors = committed_changes(
                    ToolOperationStore(self.root).for_session(row["worker_session_id"]),
                    resolutions=row.get("candidate_resolutions", []),
                )
                row.update(commits=commits, failed_operation_ids=errors)
        return {
            "boundary": boundary,
            "works": works,
            "admissions": admissions,
            "historical_sources": "unprocessed_unless_explicitly_selected",
        }

    def cancel(self, work_id: str) -> dict[str, Any]:
        """持久取消指定工作，宿主传播给已有执行者；参数：工作身份；返回：状态，不改变覆盖结论。"""
        with (
            self.database.transaction(),
            closing(ScheduleStore(self.root)) as schedules,
        ):
            row = self.load(work_id)
            if row["state"] in FINISHED:
                return row
            schedules.update(row["schedule_id"], enabled=False)
            state = (
                "cancelling"
                if row["state"] in {"running", "cancelling"}
                else "cancelled"
            )
            return self.update(
                work_id,
                state=state,
                cancel_requested=True,
                error="cancellation requested by user",
            )


def complete_source(view: MaterializedSession) -> MaterializedSession:
    """选择最后完整工具交互之前的来源，未完成当前尾部不阻塞旧前缀；参数：分支；返回：完整前缀。"""
    pending: set[str] = set()
    complete = 0
    for index, message in enumerate(view.messages):
        if isinstance(message, AssistantMessage):
            pending.update(
                part.call_id
                for part in message.content
                if isinstance(part, ToolCallPart)
            )
        elif isinstance(message, ToolResultMessage):
            pending.discard(message.call_id)
        if not pending:
            complete = index + 1
    messages = view.messages[:complete]
    # 1. 【知识维护】【冻结来源】用户正文在delivery时进入模型，不能把来源锚点倒退到inbound
    last_id = messages[-1].message_id if messages else None
    end = 0
    for index, entry in enumerate(view.entries):
        emitted = None
        if entry.type == "message" and entry.message is not None:
            emitted = entry.message.message_id
        elif entry.type == "delivery":
            emitted = entry.input_id
        if last_id is not None and emitted == last_id:
            end = index + 1
            break
    entries = view.entries[:end]
    anchor = entries[-1].entry_id if entries else None
    return replace(
        view, messages=messages, entries=entries, pending_tool_calls=(), leaf_id=anchor
    )


def maintenance_capabilities(context: RunContext) -> dict[str, object]:
    """收窄到原有文件只读权限，移除后台自触发和所有外部副作用；参数：原租约；返回：子工作能力。"""
    fs = context.capability_lease.capabilities.get("fs")
    return {
        "fs": {**fs, "write": []} if isinstance(fs, dict) else {},
        "background_run": {"enabled": False},
        "mcp": {"enabled": False},
    }
