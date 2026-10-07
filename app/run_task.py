from __future__ import annotations

from collections.abc import Callable
from contextlib import closing, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llm.base import LLMClient
from llm.client import MissingConfigurationLLMClient
from runtime.agent_loop import AgentLoop, State
from approval.session import ApprovalSession
from runtime.extensions import RuntimeExtensions
from runtime.cancellation import CancellationToken
from runtime.shared_budget import SharedRunBudget
from runtime.stream_events import StreamEvent
from runtime.checkpoint import (
    Checkpoint,
    load_checkpoint,
    load_checkpoint_by_id,
    load_latest_checkpoint,
    load_latest_checkpoint_for_run,
    load_latest_checkpoint_for_session,
)
from runtime.lease import Lease
from runtime.default_capabilities import build_local_agent_capabilities
from runtime.lease import from_trigger
from runtime.session_message_store import SessionMessageStore
from runtime.session_state import SessionStateStore
from runtime.workspaces import WorkspaceStore
from runtime.types import RunContext, Trigger
from tasks.ids import new_ulid
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.mcp_client.registry import attach_mcp_registry
from tools.tool_registry import ToolRegistry
from triggers.resume import make_run_context as make_resume_context
from app.startup import resolve_startup_identity


@dataclass(slots=True)
class RunTaskResponse:
    task_id: str
    segment_id: str
    status: str
    output: str
    run_id: str = ""


_RESUMABLE_CHECKPOINT_STATES = frozenset(
    {"PAUSED", "FAILED", "WAITING_USER", "WAITING_APPROVAL", "PRE_TOOL", "POST_TOOL"}
)


def run_task(
    task: str,
    project_root: Path,
    *,
    data_root: Path | str | None = None,
    llm_client: LLMClient | None = None,
    tool_registry: ToolRegistry | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    extensions: RuntimeExtensions | None = None,
) -> RunTaskResponse:
    """创建正式目标、保存当前输入并执行一次运行。

    传参：task/project_root 为目标和项目；其余为存储、身份及注入依赖；返回：运行结果
    """
    root = resolve_startup_identity(
        project_root=project_root, data_root=data_root
    ).data_root
    with closing(TaskStore(root)) as store:
        record = store.create_task(task)
        active_llm_client = llm_client or MissingConfigurationLLMClient()
        registry = tool_registry or build_tool_registry(
            repo_root=project_root,
            data_root=root,
        )
        with closing(registry) if tool_registry is None else nullcontext(registry):
            lease = _default_chat_lease(
                task_id=record.task_id,
                project_root=project_root,
                data_root=root,
            )
            context = RunContext(
                session_id=session_id or "",
                run_id=run_id or "",
                task_id=record.task_id,
                focus_task_id=record.task_id,
                focus_task={
                    "task_id": record.task_id,
                    "goal": record.goal,
                    "status": record.status,
                },
                trigger=Trigger.USER,
                payload={"message": task},
                capability_lease=lease,
                segment_id=f"user-{new_ulid()}",
            )
            # 一次性 CLI 的用户输入落成 canonical 消息；Ledger 只保留运行证据
            WorkspaceStore(root).bind_session(
                context.session_id, project_root
            ).require_available()
            context.payload["input_message_id"] = (
                SessionMessageStore(root)
                .accept_input(
                    context.session_id,
                    task,
                    run_id=context.run_id,
                    task_id=record.task_id,
                )
                .entry_id
            )
            return execute_context(
                context,
                data_root=root,
                llm_client=active_llm_client,
                registry=registry,
                extensions=extensions,
            )


def inspect_resume(
    checkpoint_id: str,
    project_root: Path,
    *,
    data_root: Path | str | None = None,
) -> RunTaskResponse:
    """只读检查一个可恢复 checkpoint

    参数：checkpoint_id 为 task/run/session/checkpoint 身份；project_root 为项目根；data_root 为可选数据根
    返回：不包含新 run/segment 身份的恢复摘要
    """
    root = resolve_startup_identity(
        project_root=project_root, data_root=data_root
    ).data_root
    checkpoint, source = _resolve_resume_checkpoint(checkpoint_id, root)
    task_id = checkpoint.task_id
    return RunTaskResponse(
        task_id=task_id,
        run_id="",
        segment_id="",
        status=_checkpoint_status(checkpoint.state),
        output=_format_resume_output(source, checkpoint, task_id, data_root=root),
    )


def execute_resume(
    checkpoint_id: str,
    project_root: Path,
    *,
    data_root: Path | str | None = None,
    decision: Literal["skip", "replay"] | None = None,
    llm_client: LLMClient | None = None,
    tool_registry: ToolRegistry | None = None,
    extensions: RuntimeExtensions | None = None,
) -> RunTaskResponse:
    """从指定 checkpoint 创建新的 resume run 并进入统一 AgentLoop

    参数：checkpoint_id/project_root 定位恢复点；data_root 指定数据根；decision 为 pending 决定；其余为可注入运行依赖
    返回：本次真实 run/segment 身份、终态和模型输出
    """
    root = resolve_startup_identity(
        project_root=project_root, data_root=data_root
    ).data_root
    checkpoint, source = _resolve_resume_checkpoint(checkpoint_id, root)
    _validate_resume_checkpoint(checkpoint)
    workspace = WorkspaceStore(root).for_session(checkpoint.session_id)
    workspace.require_available()
    registry = tool_registry or build_tool_registry(
        repo_root=workspace.project_root, data_root=root
    )
    with closing(registry) if tool_registry is None else nullcontext(registry):
        resume_action = _resolve_resume_action(checkpoint, registry, decision)

        # 【CLI Resume】【统一执行】1. 使用同一已解析 checkpoint 构造上下文，避免 latest 二次漂移
        context = make_resume_context(data_root=root, checkpoint=checkpoint)
        context.payload["message"] = _format_resume_execute_message(source, checkpoint)
        context.payload["resume_action"] = resume_action
        context.payload["input_message_id"] = (
            SessionMessageStore(root)
            .accept_input(
                context.session_id,
                str(context.payload["message"]),
                run_id=context.run_id,
                task_id=context.material_task_id,
            )
            .entry_id
        )
        # 【CLI Resume】【统一执行】2. pending、审批、工具与终态全部交给共享 AgentLoop
        active_llm_client = llm_client or MissingConfigurationLLMClient()
        return execute_context(
            context,
            data_root=root,
            llm_client=active_llm_client,
            registry=registry,
            extensions=extensions,
        )


def resolve_resume_identity(checkpoint_id: str, data_root: Path) -> tuple[str, Path]:
    """在模型装配前冻结恢复点及原工作区；参数：恢复选择和数据空间；返回：精确恢复点与原目录。"""
    checkpoint, _source = _resolve_resume_checkpoint(checkpoint_id, data_root)
    workspace = WorkspaceStore(data_root).for_session(checkpoint.session_id)
    workspace.require_available()
    return checkpoint.checkpoint_id, workspace.project_root


def execute_context(
    context: RunContext,
    *,
    data_root: Path,
    llm_client: LLMClient,
    registry: ToolRegistry,
    extensions: RuntimeExtensions | None = None,
    runtime_config: dict[str, object] | None = None,
    cancellation: CancellationToken | None = None,
    shared_budget: SharedRunBudget | None = None,
    event_sink: Callable[[StreamEvent], None] | None = None,
    approval_session: ApprovalSession | None = None,
) -> RunTaskResponse:
    """从新请求或恢复入口执行同一运行循环并返回真实状态。

    传参：context 为已保存输入的运行；其余为运行依赖；返回：身份、状态与输出
    """
    # 【运行装配】【借用依赖】连接归宿主所有；共同执行只使用依赖，不在此关闭共享资源
    from runtime.knowledge_maintenance import KnowledgeMaintenance, automatic_origin

    maintenance = KnowledgeMaintenance(data_root)
    maintenance.initialize(context)
    if automatic_origin(context) is None:
        attach_mcp_registry(context.capability_lease, registry)
    loop = AgentLoop(
        data_root,
        llm_client=llm_client,
        tool_registry=registry,
        extensions=extensions,
        cancellation=cancellation,
        shared_budget=shared_budget,
        approval_session=approval_session,
        runtime_config=runtime_config,
    )
    if event_sink is None:
        state = loop.run(context)
    else:
        for event in loop.run_stream(context):
            event_sink(event)
        if loop.state is None:
            raise RuntimeError("run exited without a lifecycle boundary")
        state = loop.state
    maintenance.observe(context, client=llm_client, approval_session=approval_session)
    return RunTaskResponse(
        task_id=context.focus_task_id or context.storage_task_id,
        run_id=context.run_id,
        segment_id=context.segment_id,
        status=_status(state),
        output=loop.last_output,
    )


def _resolve_resume_checkpoint(target: str, data_root: Path) -> tuple[Checkpoint, str]:
    if "::" in target:
        task_id, checkpoint_id = target.split("::", maxsplit=1)
        checkpoint = load_checkpoint(task_id, checkpoint_id, data_root=data_root)
        source = "task checkpoint"
    elif target.startswith("run-"):
        checkpoint = load_latest_checkpoint_for_run(target, data_root=data_root)
        source = "run"
    elif target.startswith("session-"):
        checkpoint = load_latest_checkpoint_for_session(target, data_root=data_root)
        source = "session"
    else:
        checkpoint = load_latest_checkpoint(target, data_root=data_root)
        source = "task"
        if checkpoint is None:
            checkpoint = load_checkpoint_by_id(target, data_root=data_root)
            source = "checkpoint"
    if checkpoint is None:
        raise FileNotFoundError(target)
    return checkpoint, source


def _validate_resume_checkpoint(checkpoint: Checkpoint) -> None:
    """拒绝没有恢复语义的 checkpoint 状态

    参数：checkpoint 为精确解析后的来源恢复点
    返回：无；不支持时抛出 ValueError
    """
    state = checkpoint.state.strip().upper()
    if state not in _RESUMABLE_CHECKPOINT_STATES:
        raise ValueError(
            f"checkpoint is not resumable: {checkpoint.checkpoint_id} ({state})"
        )


def _resolve_resume_action(
    checkpoint: Checkpoint,
    registry: ToolRegistry,
    decision: Literal["skip", "replay"] | None,
) -> str:
    """校验显式 pending 决定或计算现有默认恢复策略

    参数：checkpoint 为来源恢复点；registry 提供工具幂等性；decision 为可选显式决定
    返回：写入 RunContext.payload 的 replay/skip/ask
    """
    if decision is None:
        return "inspect"
    if decision not in {"skip", "replay"}:
        raise ValueError(f"invalid resume decision: {decision}")
    if checkpoint.pending_tool_call is None:
        raise ValueError("resume decision requires a pending tool")
    return decision


def _format_resume_execute_message(source: str, checkpoint: Checkpoint) -> str:
    """生成模型可见的 resume 继续执行消息

    参数：source 为解析来源；checkpoint 为精确恢复点
    返回：不包含伪成功语义的继续执行提示
    """
    return (
        f"Continue the interrupted goal from {source} checkpoint "
        f"{checkpoint.checkpoint_id}. Recovery reason: {checkpoint.reason or '(none)'}"
    )


def _format_resume_output(
    source: str,
    checkpoint: Checkpoint,
    task_id: str,
    *,
    data_root: Path,
) -> str:
    focus = checkpoint.focus_task_id
    if focus is None and checkpoint.compatibility_task_id != checkpoint.task_id:
        focus = checkpoint.task_id
    session_summary = ""
    if checkpoint.session_id:
        state = SessionStateStore(data_root).load(checkpoint.session_id)
        if state is not None:
            session_summary = state.summary
    lines = [
        "RESUME_READY",
        f"source: {source}",
        f"session: {checkpoint.session_id or '(none)'}",
        f"run: {checkpoint.run_id or '(none)'}",
        f"checkpoint: {checkpoint.checkpoint_id} ({checkpoint.state})",
        f"reason: {checkpoint.reason or '(none)'}",
        f"focus_task: {focus or '(none)'}",
        f"compatibility_task: {checkpoint.compatibility_task_id or '(none)'}",
        f"storage_task: {task_id}",
    ]
    if session_summary:
        lines.append(f"session_summary: {session_summary}")
    return "\n".join(lines)


def _status(state: State) -> str:
    if state is State.DONE:
        return "done"
    if state is State.PAUSED:
        return "paused"
    return "failed"


def _checkpoint_status(state: str) -> str:
    upper = state.upper()
    if upper == State.DONE.value:
        return "done"
    if upper in {
        State.PAUSED.value,
        "WAITING_USER",
        "WAITING_APPROVAL",
        "PRE_TOOL",
        "POST_TOOL",
    }:
        return "paused"
    return "failed"


def _default_chat_lease(
    *,
    task_id: str,
    project_root: Path,
    data_root: Path,
) -> Lease:
    return from_trigger(
        "user",
        task_id=task_id,
        capabilities=build_local_agent_capabilities(project_root, data_root),
    )
