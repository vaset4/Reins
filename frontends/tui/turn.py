from __future__ import annotations

from contextlib import closing
from pathlib import Path
from approval.session import ApprovalSession

from functools import partial
from app.run_task import execute_context
from app.session_assembly import (
    SessionBinding,
    current_session_binding,
    ensure_session_binding,
    record_run_error,
    user_context,
    user_lease,
)
from runtime.session_message_store import SessionMessageStore
from runtime.session_state import SessionStateStore
from runtime.stream_events import StreamEvent
from app.repl.slash_commands import ReplState
from llm.base import LLMClient
from runtime.cancellation import CancellationToken
from runtime.types import new_run_id
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry

from frontends.tui.event_adapter import TuiEventAdapter
from frontends.tui.transcript import TranscriptBuffer


def drive_agent_turn(
    *,
    state: ReplState,
    transcript: TranscriptBuffer,
    project_root: Path,
    data_root: Path,
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    user_message: str,
    invalidate: object,
    cancellation: CancellationToken | None = None,
    input_id: str | None = None,
    approval_session: ApprovalSession | None = None,
) -> None:
    """用全屏宿主持有的目录执行一轮；传参：会话、展示、输入及执行依赖；返回：无，连接由宿主释放。"""
    with closing(TaskStore(data_root)) as store:
        binding = ensure_session_binding(
            SessionBinding(
                state.session_id, state.current_task_id, state.compatibility_task_id
            ),
            store,
            user_message,
        )
        binding = current_session_binding(binding, data_root)
        state.session_id = binding.session_id
        state.current_task_id = binding.focus_task_id
        state.compatibility_task_id = binding.compatibility_task_id
        storage_task_id = state.current_task_id or state.compatibility_task_id
        if storage_task_id is None:
            transcript.append(
                "error", "storage", "No task storage is available.", state="error"
            )
            return
        run_id = new_run_id()
        state.current_run_id = run_id
        transcript.append(
            "status",
            "run started",
            f"run={run_id} task={storage_task_id}",
            state="pending",
        )
        lease = user_lease(
            task_id=storage_task_id, project_root=project_root, data_root=data_root
        )
        if input_id is None:
            input_id = (
                SessionMessageStore(data_root)
                .accept_input(
                    state.session_id,
                    user_message,
                    run_id=run_id,
                    task_id=storage_task_id,
                )
                .entry_id
            )
        context = user_context(
            binding, lease, data_root=data_root, run_id=run_id, input_id=input_id
        )
        adapter = TuiEventAdapter(transcript, trace_on=state.trace_on)
        try:
            execute_context(
                context,
                data_root=data_root,
                llm_client=llm_client,
                registry=tool_registry,
                cancellation=cancellation,
                approval_session=approval_session,
                event_sink=partial(_render_event, adapter, invalidate),
            )
        except Exception as exc:
            record_run_error(data_root, context, exc)
            raise
        finally:
            saved = SessionStateStore(data_root).load(state.session_id)
            state.current_task_id = saved.focus_task_id if saved else None


def _render_event(
    adapter: TuiEventAdapter, invalidate: object, event: StreamEvent
) -> None:
    """展示共同执行器的事件并刷新界面；传参：适配器/刷新入口/真实事件；返回：无。"""
    adapter.render(event)
    if callable(invalidate):
        invalidate()


__all__ = ["drive_agent_turn"]
