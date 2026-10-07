"""本地聊天的显示适配；运行装配与执行归公共应用边界。

作者：xxx
时间：2026-09-28 16:55:00
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from app.repl.render import EventRenderer
from app.repl.slash_commands import ReplState
from app.repl.status import explain_exception_status, format_status_hint
from app.run_task import execute_context
from app.session_assembly import (
    SessionBinding,
    current_session_binding,
    record_run_error,
    user_context,
    user_lease,
)
from approval.session import ApprovalSession
from llm.base import LLMClient
from runtime.cancellation import CancellationToken
from runtime.extensions import RuntimeExtensions
from runtime.session_message_store import SessionMessageStore
from runtime.session_state import SessionStateStore
from runtime.types import new_run_id
from tools.tool_registry import ToolRegistry


def render_user_turn(
    *,
    console: Console,
    state: ReplState,
    registry: ToolRegistry,
    llm_client: LLMClient,
    project_root: Path,
    data_root: Path,
    user_message: str,
    runtime_config: dict[str, object] | None = None,
    on_first_output: Callable[[], None] | None = None,
    input_id: str | None = None,
    cancellation: CancellationToken | None = None,
    extensions: RuntimeExtensions | None = None,
    approval_session: ApprovalSession | None = None,
) -> None:
    """显示一轮用户执行并回流焦点；传参：界面状态、依赖和输入引用；返回：无，错误保存并展示。"""
    binding = current_session_binding(
        SessionBinding(
            state.session_id, state.current_task_id, state.compatibility_task_id
        ),
        data_root,
    )
    storage_task = binding.focus_task_id or binding.compatibility_task_id
    if storage_task is None:
        raise ValueError("chat entry requires an established session binding")
    run_id = new_run_id()
    state.current_run_id = run_id
    # 【本地会话】【输入接纳】直接脚本入口在此接纳，宿主已接纳的输入仅传递引用
    if input_id is None:
        input_id = (
            SessionMessageStore(data_root)
            .accept_input(
                binding.session_id,
                user_message,
                run_id=run_id,
                task_id=storage_task,
            )
            .entry_id
        )
    lease = user_lease(
        task_id=storage_task, project_root=project_root, data_root=data_root
    )
    context = user_context(
        binding, lease, data_root=data_root, run_id=run_id, input_id=input_id
    )
    renderer = EventRenderer(trace_on=state.trace_on, on_first_output=on_first_output)
    try:
        execute_context(
            context,
            data_root=data_root,
            llm_client=llm_client,
            registry=registry,
            runtime_config=runtime_config,
            cancellation=cancellation,
            extensions=extensions,
            approval_session=approval_session,
            event_sink=renderer.render,
        )
    except Exception as exc:
        record_run_error(data_root, context, exc)
        hint = format_status_hint(explain_exception_status(exc))
        console.print(
            Panel(
                Text.assemble(
                    (f"agent loop error: {exc}\n\n", "red"), (hint, "yellow")
                ),
                title="error",
                border_style="red",
                expand=False,
            )
        )
    finally:
        saved = SessionStateStore(data_root).load(binding.session_id)
        state.current_task_id = saved.focus_task_id if saved else None
