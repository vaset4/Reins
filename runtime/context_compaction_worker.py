"""【上下文】【后台执行】通过现有模型证据与预算边界整理冻结历史，不创建会话输入。

作者：xxx
时间：2026-10-01 12:00:00
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import cast

from llm.base import LLMClient
from llm.public_config import restore_model_config
from runtime.cancellation import ExecutionCancelled
from runtime.context_compaction_jobs import ContextCompactionJobs
from runtime.context_preparation import ContextCompactor, _summary_plan
from runtime.cron import CronExecution, CronOutcome
from runtime.extension_execution import ExtensionExecution
from runtime.extensions import RuntimeExtensions
from runtime.lease import from_trigger, load_snapshot
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.model_evidence import ModelEvidenceWriter
from runtime.model_execution import ModelRequestRunner
from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore
from runtime.session_compaction import SessionCompactionStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_state import SessionStateStore
from runtime.shared_budget import BudgetOwner, SharedRunBudget
from runtime.stream_events import StreamEvent
from runtime.types import RunContext, Trigger
from runtime.watchdog import Watchdog
from runtime.workspaces import WorkspaceStore
from tools.tool_registry import ToolRegistry


def execute_compaction(
    request: CronExecution,
    *,
    data_root: Path,
    llm_factory: Callable[[dict[str, object]], LLMClient] | None = None,
    registry_factory: Callable[[], ToolRegistry] | None = None,
    event_sink: Callable[[str, StreamEvent], None] | None = None,
) -> CronOutcome:
    """执行调度器认领的整理工作；参数：冻结发生及装配依赖；返回：真实发布、取消或失败结果。"""
    from app.cli import build_llm_client
    from tools.builtin_tools import build_tool_registry

    del event_sink
    origin = request.schedule.context_compaction_origin
    if origin is None:
        raise ValueError("scheduled work is missing its compaction origin")
    jobs = ContextCompactionJobs(data_root)
    row = jobs.load(str(origin["job_id"]))
    published = jobs.published(row)
    if published is not None:
        jobs.update(row["job_id"], status="published", summary_id=published, error=None)
        return CronOutcome("done", f"历史已发布：{published}")
    jobs.update(row["job_id"], status="running", run_id=request.occurrence.run_id)
    try:
        if request.cancellation.cancelled or jobs.cancellation_requested(row["job_id"]):
            raise ExecutionCancelled("background compaction cancelled")
        # 1. 【上下文】【后台执行】工作区不可用也属于本次工作的失败，须落盘供重启与活动查看
        workspace = WorkspaceStore(data_root).get(request.schedule.workspace_id)
        workspace.require_available()
        make_llm = llm_factory or partial(
            build_llm_client, project_root=workspace.project_root
        )
        make_registry = registry_factory or partial(
            build_tool_registry, repo_root=workspace.project_root, data_root=data_root
        )
        material = jobs.material(row)
        store = SessionCompactionStore(SessionMessageStore(data_root))
        current = store.messages.materialize(row["source_session_id"])
        if row["source_entry_id"] not in {entry.entry_id for entry in current.entries}:
            raise ValueError("background compaction source branch is no longer active")
        active = store.current(current)
        if (active.summary_id if active else None) != row["previous_summary_id"]:
            raise ValueError("background compaction predecessor was superseded")
        client = make_llm(restore_model_config(request.schedule.model_config))
        with closing(make_registry()) as registry:
            compactor, context = _execution(
                request, data_root=data_root, client=client, registry=registry
            )
            model_context = {
                "session_id": context.session_id,
                "run_id": context.run_id,
                "segment_id": context.segment_id,
                "tool_registry": registry,
                "history_representation_version": 1,
            }
            content, requests = compactor._summarize(
                material, model_context, context_window=0
            )
            record = store.publish(
                material.source,
                content.render(),
                content=content,
                request_ids=requests,
                cancelled=lambda: (
                    request.cancellation.cancelled
                    or jobs.cancellation_requested(row["job_id"])
                ),
                compaction_job_id=row["job_id"],
            )
        jobs.update(
            row["job_id"],
            status="published",
            summary_id=record.summary_id,
            request_ids=list(requests),
            error=None,
        )
        return CronOutcome(
            "done", f"历史已发布：{record.summary_id}；后续请求按当前来源采用"
        )
    except Exception as exc:
        published = jobs.published(row)
        status = (
            "published"
            if published
            else "cancelled"
            if isinstance(exc, ExecutionCancelled)
            else "failed"
        )
        jobs.update(
            row["job_id"],
            status=status,
            summary_id=published,
            error=f"{type(exc).__name__}: {exc}",
        )
        return CronOutcome(
            "paused" if status == "cancelled" else "failed", str(exc), error=str(exc)
        )


def _execution(
    request: CronExecution,
    *,
    data_root: Path,
    client: LLMClient,
    registry: ToolRegistry,
) -> tuple[ContextCompactor, RunContext]:
    """复用模型运行器和共享资源账目；参数：发生、客户端和工具目录；返回：仅摘要执行能力与归属。"""
    job = ContextCompactionJobs(data_root).load(
        str(
            cast(dict[str, object], request.schedule.context_compaction_origin)[
                "job_id"
            ]
        )
    )
    lease = from_trigger(
        "cron",
        capabilities=request.schedule.capabilities,
        max_steps=request.schedule.max_steps,
        max_tokens=request.schedule.max_tokens,
    )
    context = RunContext(
        trigger=Trigger.CRON,
        payload={"context_compaction_job_id": job["job_id"]},
        capability_lease=lease,
        session_id=request.occurrence.session_id,
        run_id=request.occurrence.run_id,
        segment_id=f"compaction-{request.occurrence.run_id}",
        parent_session_id=job["source_session_id"],
        parent_run_id=job["source_run_id"],
        budget_run_id=job["budget_run_id"],
    )
    facts, evidence = RunFactStore(data_root), RunEvidenceStore(data_root)
    owner = BudgetOwner(
        job["source_session_id"], job["budget_run_id"], load_snapshot(job["lease"])
    )
    budget = SharedRunBudget.restore(owner, facts)
    watchdog = Watchdog(
        lease,
        task_id=context.storage_task_id,
        data_root=data_root,
        cancellation=request.cancellation,
        shared_budget=budget,
        budget_owner=BudgetOwner(context.session_id, context.run_id, lease),
    )
    runner = ModelRequestRunner(
        client,
        registry=registry,
        cancellation=request.cancellation,
        evidence=ModelEvidenceWriter(evidence, facts),
        facts=facts,
        run_evidence=evidence,
        states=SessionStateStore(data_root),
        ledger=LedgerWriter(
            LedgerStore(data_root), source="runtime.context_compaction_worker"
        ),
        extensions=ExtensionExecution(RuntimeExtensions(), request.cancellation, facts),
    )
    prepare = getattr(client, "prepare_request", None)
    if not callable(prepare):
        raise ValueError("background compaction requires a model request preparer")
    compactor = ContextCompactor(
        SessionCompactionStore(SessionMessageStore(data_root)),
        prepare,
        partial(_summary_plan, runner=runner, context=context, watchdog=watchdog),
        cancelled=lambda: (
            request.cancellation.cancelled
            or ContextCompactionJobs(data_root).cancellation_requested(job["job_id"])
        ),
    )
    return compactor, context
