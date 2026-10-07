"""定时工作复用生产模型、工具、权限和统一运行入口。

作者：xxx
时间：2026-09-14 19:16:10
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

from app.run_task import execute_context
from llm.base import LLMClient
from llm.public_config import restore_model_config
from runtime.checkpoint import load_latest_checkpoint_for_run
from runtime.cron import CronExecution, CronOutcome, CronScheduler
from runtime.lease import from_trigger, load_snapshot
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.shared_budget import BudgetOwner, SharedRunBudget
from runtime.stream_events import StreamEvent
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import ToolRegistry
from triggers.cron import make_run_context as make_cron_context
from triggers.resume import make_run_context as make_resume_context


def create_scheduler(
    *,
    project_root: Path,
    data_root: Path,
    llm_factory: Callable[[dict[str, object]], LLMClient] | None = None,
    registry_factory: Callable[[], ToolRegistry] | None = None,
    event_sink: Callable[[str, StreamEvent], None] | None = None,
) -> CronScheduler:
    """装配生产调度；参数：目录、工厂和接收运行身份及事件的回调；返回：调度器。"""
    execute = partial(
        execute_scheduled,
        data_root=data_root,
        llm_factory=llm_factory,
        registry_factory=registry_factory,
        event_sink=event_sink,
    )
    return CronScheduler(
        project_root=project_root, data_root=data_root, execute=execute
    )


def execute_scheduled(
    request: CronExecution,
    *,
    data_root: Path,
    llm_factory: Callable[[dict[str, object]], LLMClient] | None,
    registry_factory: Callable[[], ToolRegistry] | None,
    event_sink: Callable[[str, StreamEvent], None] | None = None,
) -> CronOutcome:
    """执行已认领发生；参数：发生、依赖和接收运行身份及事件的回调；返回：真实运行结果。"""
    from app.cli import build_llm_client

    if request.schedule.context_compaction_origin is not None:
        from runtime.context_compaction_worker import execute_compaction

        return execute_compaction(
            request,
            data_root=data_root,
            llm_factory=llm_factory,
            registry_factory=registry_factory,
            event_sink=event_sink,
        )
    origin = request.schedule.knowledge_origin
    if origin is not None and origin.get("work_kind") == "knowledge_maintenance":
        return _execute_maintenance(
            request,
            data_root=data_root,
            llm_factory=llm_factory,
            registry_factory=registry_factory,
            event_sink=event_sink,
        )

    workspace = WorkspaceStore(data_root).get(request.schedule.workspace_id)
    workspace.require_available()
    make_llm = llm_factory or partial(
        build_llm_client, project_root=workspace.project_root
    )
    make_registry = registry_factory or partial(
        build_tool_registry, repo_root=workspace.project_root, data_root=data_root
    )
    context = scheduled_context(request, data_root=data_root)
    client = make_llm(restore_model_config(request.schedule.model_config))
    budget = None
    if (
        request.previous_run_id is not None
        and request.occurrence.budget_run_id != context.run_id
    ):
        root = BudgetOwner(
            context.session_id,
            request.occurrence.budget_run_id or request.occurrence.run_ids[0],
            context.capability_lease,
        )
        budget = SharedRunBudget.restore(root, RunFactStore(data_root))
    with closing(make_registry()) as registry:
        # 【定时工作】【事件归属】1. 冻结实际认领后的运行身份，接续后的迟到回调仍归原运行
        publish = (
            partial(event_sink, context.run_id) if event_sink is not None else None
        )
        response = execute_context(
            context,
            data_root=data_root,
            llm_client=client,
            registry=registry,
            cancellation=request.cancellation,
            shared_budget=budget,
            event_sink=publish,
        )
    # 1. 【后台工作】【失败交付】共同执行器已返回失败说明，计划和通知需要保留同一原因
    error = response.output if response.status == "failed" else None
    return CronOutcome(response.status, response.output, response.task_id, error=error)


def scheduled_context(request: CronExecution, *, data_root: Path) -> RunContext:
    """构造独立会话并在恢复时保留CRON授权边界；传参：发生与存储；返回：统一上下文。"""
    job, occurrence = request.schedule, request.occurrence
    compatibility = _ensure_scheduled_storage(request, data_root=data_root)
    context = make_cron_context(
        job.schedule_id,
        data_root=data_root,
        session_id=occurrence.session_id,
        run_id=occurrence.run_id,
        compatibility_task_id=compatibility,
        schedule_data={
            **asdict(job),
            "required_permanent_grants": list(job.required_permanent_grants),
        },
    )
    message = f"执行已接纳的定时工作：{job.prompt}\n原定时刻：{occurrence.scheduled_at}\n当前时刻：{request.now.isoformat()}\n"
    message += "若已过原定时刻，先核对工作是否仍有效；沿用原授权范围，核查已有结果再决定下一步。"
    input_id = f"scheduled-{occurrence.occurrence_id}"
    if request.previous_run_id is not None:
        checkpoint = load_latest_checkpoint_for_run(
            request.previous_run_id, data_root=data_root
        )
        if checkpoint is not None:
            resumed = make_resume_context(
                data_root=data_root, checkpoint=checkpoint, run_id=occurrence.run_id
            )
            if not checkpoint.lease_snapshot:
                raise ValueError(
                    "automatic schedule recovery requires its original permission snapshot"
                )
            previous_lease = load_snapshot(checkpoint.lease_snapshot)
            lease = from_trigger(
                "cron",
                task_id=previous_lease.task_id,
                capabilities=previous_lease.capabilities,
                max_steps=previous_lease.max_steps,
                max_tokens=previous_lease.max_tokens,
            )
            context = replace(resumed, trigger=Trigger.CRON, capability_lease=lease)
        context.payload["previous_run_id"] = request.previous_run_id
        context.payload["resume_action"] = "inspect"
        input_id = f"scheduled-resume-{occurrence.run_id}"
        reason = (
            "收到显式继续请求"
            if occurrence.budget_run_id == occurrence.run_id
            else "宿主中断后恢复"
        )
        message += f"\n{reason}，接续原运行 {request.previous_run_id}；先读取已记录效果，不重复执行已完成操作。"
    if occurrence.resume_input_id is not None:
        message += "\n本次发生收到显式接续输入，先阅读该输入与原问题，再判断下一步。"
    if job.source_entry_id is not None:
        context.payload["source_entry_id"] = job.source_entry_id
        message += (
            "\n本工作保留了接纳时的原会话来源。可用knowledge_read补读原要求；"
            "记忆或方法引用原用户输入时使用source_mode=origin_inputs，不把本条调度说明当作用户确认。"
        )
    if job.knowledge_origin is not None:
        context.payload["knowledge_origin"] = dict(job.knowledge_origin)
        if job.knowledge_origin.get("verification_targets"):
            import json

            message += "\n当前相关知识来源变化，需核验：" + json.dumps(
                job.knowledge_origin["verification_targets"], ensure_ascii=False
            )
    context.payload.update(
        message=message,
        schedule_id=job.schedule_id,
        occurrence_id=occurrence.occurrence_id,
        scheduled_at=occurrence.scheduled_at,
        observed_at=request.now.isoformat(),
        source_session_id=job.source_session_id,
        source_run_id=job.source_run_id,
    )
    # 【定时工作】【输入归属】定时触发是系统交付，不能冒充新的用户确认或扩大权限
    if (
        job.knowledge_origin is not None
        and job.knowledge_origin.get("work_kind") == "knowledge_maintenance"
    ):
        # 【知识维护】【输入归属】辅助任务只有系统工作材料，不新增用户输入或推进主会话输入水位
        context.payload.pop("input_message_id", None)
        return context
    entry = SessionMessageStore(data_root).accept_input(
        context.session_id,
        message,
        input_id=input_id,
        run_id=context.run_id,
        task_id=context.material_task_id,
        input_source="agent",
    )
    context.payload["input_message_id"] = entry.entry_id
    return context


def _execute_maintenance(
    request: CronExecution,
    *,
    data_root: Path,
    llm_factory: Callable[[dict[str, object]], LLMClient] | None,
    registry_factory: Callable[[], ToolRegistry] | None,
    event_sink: Callable[[str, StreamEvent], None] | None,
) -> CronOutcome:
    """原发生恢复先对账已提交处理结果，模型只用受限工具；参数：调度依赖；返回：真实工作结果。"""
    from app.cli import build_llm_client
    from runtime.knowledge_maintenance import FINISHED, KnowledgeMaintenance
    from runtime.knowledge_sources import frozen_knowledge_source
    from runtime.knowledge_worker import maintenance_registry, provide_frozen_sources
    from runtime.session_message_store import SessionMessageStore

    manager = KnowledgeMaintenance(data_root)
    origin = request.schedule.knowledge_origin
    assert origin is not None
    row = manager.load(origin["work_id"])
    if row["state"] in FINISHED:
        return CronOutcome("done", str(row["reason"]))
    if row["state"] in {"cancelled", "cancelling"} or request.cancellation.cancelled:
        manager.update(row["work_id"], state="cancelled")
        return CronOutcome("paused", "知识维护已取消")
    try:
        # 【知识维护】【启动对账】工作区、模型或恢复准备失败也属于本次已接纳工作的失败
        workspace = WorkspaceStore(data_root).get(request.schedule.workspace_id)
        workspace.require_available()
        # 1. 【知识维护】【来源预检】坏冻结源由宿主明确失败，不消耗模型调用反复猜测读取参数
        source = frozen_knowledge_source(SessionMessageStore(data_root), origin)
        context = provide_frozen_sources(
            scheduled_context(request, data_root=data_root), source
        )
        make_llm = llm_factory or partial(
            build_llm_client, project_root=workspace.project_root
        )
        make_registry = registry_factory or partial(
            build_tool_registry, repo_root=workspace.project_root, data_root=data_root
        )
        client = make_llm(restore_model_config(request.schedule.model_config))
        started = manager.update(
            row["work_id"],
            state="running",
            worker_session_id=context.session_id,
            run_id=context.run_id,
        )
        if started.get("cancel_requested"):
            manager.update(row["work_id"], state="cancelled")
            return CronOutcome("paused", "知识维护已取消")
        publish = (
            partial(event_sink, context.run_id) if event_sink is not None else None
        )
        budget = None
        if (
            request.previous_run_id is not None
            and request.occurrence.budget_run_id != context.run_id
        ):
            owner = BudgetOwner(
                context.session_id,
                request.occurrence.budget_run_id or request.occurrence.run_ids[0],
                context.capability_lease,
            )
            budget = SharedRunBudget.restore(owner, RunFactStore(data_root))
        with (
            closing(make_registry()) as full,
            closing(maintenance_registry(full, data_root=data_root)) as registry,
        ):
            response = execute_context(
                context,
                data_root=data_root,
                llm_client=client,
                registry=registry,
                cancellation=request.cancellation,
                event_sink=publish,
                shared_budget=budget,
            )
        current = manager.load(row["work_id"])
        if current["state"] in FINISHED:
            return CronOutcome("done", str(current["reason"]), response.task_id)
        error = (
            response.output
            if response.status != "done"
            else "knowledge worker ended without verified source coverage"
        )
        state = (
            "cancelled"
            if request.cancellation.cancelled or current.get("cancel_requested")
            else "failed"
        )
        manager.update(row["work_id"], state=state, error=error)
        return CronOutcome("failed", error, response.task_id, error=error)
    except Exception as exc:
        current = manager.load(row["work_id"])
        if current.get("cancel_requested"):
            manager.update(
                row["work_id"], state="cancelled", error=f"{type(exc).__name__}: {exc}"
            )
        elif current["state"] not in FINISHED | {"cancelled"}:
            manager.update(
                row["work_id"], state="failed", error=f"{type(exc).__name__}: {exc}"
            )
        raise


def _ensure_scheduled_storage(request: CronExecution, *, data_root: Path) -> str | None:
    """有正式目标则引用原目标，否则使用一次发生的稳定收件箱；传参：发生与存储；返回：兼容身份。"""
    job = request.schedule
    with closing(TaskStore(data_root)) as store:
        if job.target_task_id is not None:
            store.require_task(job.target_task_id)
            return None
        identity = f"scheduled-{request.occurrence.occurrence_id}"
        if store.load_task(identity) is None:
            assert job.prompt and job.prompt.strip(), (
                "scheduled prompt must be non-empty"
            )
            store.create_task(job.prompt, task_id=identity, is_inbox=True)
        return identity
