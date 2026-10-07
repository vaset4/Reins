"""两个聊天入口共用持久会话协调器和审批输入通道。

作者：xxx
时间：2026-09-14 10:00:00
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterator
from contextlib import (
    AbstractContextManager,
    ExitStack,
    closing,
    contextmanager,
    nullcontext,
)
from pathlib import Path

import approval
from approval.batch import register_batch_backend
from rich.text import Text
from rich.status import Status

from app.repl.console import get_console
from app.repl.turn import render_user_turn
from app.session_assembly import SessionBinding, ensure_session_binding
from app.completion_ui import CompletionMenu, completion_service
from app.approval_ui import approval_control
from app.repl.slash_commands import ReplState
from approval import ApprovalRequest
from approval.channel import ApprovalChannel
from approval.batch_types import ApprovalBatch, format_batch
from approval.session import ApprovalSession
from llm.base import LLMClient
from llm.messages import UserMessage, model_visible_text
from runtime.run_facts import RunFactStore
from runtime.extensions import RuntimeExtensions
from runtime.session_message_store import SessionMessageStore
from runtime.session_runtime import SessionRun, SessionRuntime
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry
from tools.builtin_tools import build_tool_registry


@dataclass(frozen=True, slots=True)
class SessionHostConfig:
    """宿主依赖；数据库连接由执行线程自己创建和释放。"""

    state: ReplState
    registry: ToolRegistry
    llm_client: LLMClient | None
    project_root: Path
    data_root: Path
    show_working: bool = False
    extensions: RuntimeExtensions | None = None


class SessionHost:
    """让读取输入独立于模型/工具等待，使用同一执行入口保留现有展示。"""

    def __init__(self, config: SessionHostConfig) -> None:
        """保存入口依赖；传参：宿主配置；返回：无。"""
        if config.llm_client is None:
            raise ValueError("本地执行会话需要模型配置；只读历史请连接后台")
        self.config = config
        with ExitStack() as cleanup:
            self.approvals = ApprovalChannel(
                _present_approval, present_batch=_present_batch
            )
            cleanup.callback(self.approvals.close)
            self.approval_session = ApprovalSession()
            cleanup.callback(self.approval_session.close)
            self._runtime: SessionRuntime | None = None
            self._completion_menu = CompletionMenu()
            self._refresh_confirmations()
            # 【本地会话】【装配交接】初始化成功后资源归宿主，失败则按创建顺序逆序释放
            cleanup.pop_all()

    @property
    def active(self) -> bool:
        """查询当前会话是否执行中；传参：无；返回：状态。"""
        return self._runtime is not None and self._runtime.active

    def submit(self, text: str) -> str:
        """接纳正文后中断旧审批，让模型使用新约束；传参：正文；返回：输入身份。"""
        config = self.config
        with closing(TaskStore(config.data_root)) as store:
            binding = ensure_session_binding(
                SessionBinding(
                    config.state.session_id,
                    config.state.current_task_id,
                    config.state.compatibility_task_id,
                ),
                store,
                text,
            )
            config.state.session_id = binding.session_id
            config.state.current_task_id = binding.focus_task_id
            config.state.compatibility_task_id = binding.compatibility_task_id
        WorkspaceStore(config.data_root).bind_session(
            config.state.session_id, config.project_root
        ).require_available()
        if self._runtime is None or self._runtime.session_id != config.state.session_id:
            if self._runtime is not None:
                self.approval_session.close()
                self.approval_session = ApprovalSession()
            self._runtime = SessionRuntime(
                config.state.session_id,
                messages=SessionMessageStore(config.data_root),
                facts=RunFactStore(config.data_root),
                run=self._run,
            )
        choice = self._completion_menu.choose(text)
        if choice is not None:
            with completion_service(config.data_root) as confirmations:
                event = confirmations.decide(config.state.session_id, **choice)
            self._runtime.resume()
            identity = str(event.payload["source_input_id"])
        else:
            identity = self._runtime.submit(text, task_id=config.state.current_task_id)
        get_console().print("[dim]已接纳，将在下一次决策中使用。[/dim]")
        return identity

    def handle_control(self, text: str) -> bool:
        """处理共用审批和停止命令；传参：输入；返回：是否已消费。"""
        if text.split(maxsplit=1)[0] in {"/mode", "/revoke"}:
            try:
                message = approval_control(
                    text,
                    self.approval_session,
                    data_root=self.config.data_root,
                    session_id=self.config.state.session_id,
                    task_id=self.config.state.current_task_id,
                )
                get_console().print(Text(message))
                if len(text.split()) > 1:
                    self.approvals.interrupt()
            except (OSError, ValueError) as exc:
                get_console().print(Text(str(exc), style="red"))
            return True
        if text == "/stop":
            self.cancel()
            get_console().print("[dim]已请求停止，正在确认实际执行状态。[/dim]")
            return True
        if text.startswith("/approve"):
            try:
                self.approvals.answer(text)
            except ValueError as exc:
                get_console().print(Text(str(exc), style="red"))
            return True
        return False

    def cancel(self) -> None:
        """把停止传给模型、工具及审批等待；传参：无；返回：无。"""
        if self._runtime is not None:
            self._runtime.cancel()
        self.approvals.interrupt()

    def wait_idle(self) -> None:
        """切换会话或退出前完成运行交接；传参：无；返回：无。"""
        if self._runtime is not None:
            self._runtime.wait_idle()

    def prepare_command(self, text: str) -> None:
        """界面退出可立即断开，其他本地命令先交接当前运行；传参：命令；返回：无。"""
        if text.split(maxsplit=1)[0] not in {"/exit", "/quit"}:
            self.approvals.interrupt()
            self.wait_idle()

    def close(self) -> None:
        """关闭交互后等待已接纳工作释放宿主；传参：无；返回：无。"""
        # 【本地会话】【资源释放】保持等待已接纳工作的顺序；前一资源关闭失败仍清理其余自有资源
        with ExitStack() as cleanup:
            cleanup.callback(self.config.registry.close)
            if self._runtime is not None:
                cleanup.callback(self._runtime.wait_idle)
                cleanup.callback(self._runtime.close)
            cleanup.callback(self.approval_session.close)
            cleanup.callback(self.approvals.close)

    def _run(self, request: SessionRun) -> None:
        """在执行线程创建运行依赖并引用唯一正文；传参：输入身份/取消信号；返回：无。"""
        config = self.config
        if config.llm_client is None:
            raise ValueError("本地执行会话没有模型配置")
        entries = SessionMessageStore(config.data_root).read_entries(
            config.state.session_id
        )
        entry = next(item for item in entries if item.entry_id == request.input_id)
        assert isinstance(entry.message, UserMessage)
        status: AbstractContextManager[Status | None]
        if config.show_working:
            status = get_console().status("[cyan]Working...[/cyan]", spinner="dots")
        else:
            status = nullcontext(None)
        with status as spinner:
            render_user_turn(
                console=get_console(),
                state=config.state,
                registry=config.registry,
                llm_client=config.llm_client,
                project_root=config.project_root,
                data_root=config.data_root,
                user_message=model_visible_text(entry.message),
                input_id=request.input_id,
                cancellation=request.cancellation,
                extensions=config.extensions,
                approval_session=self.approval_session,
                on_first_output=spinner.stop if spinner is not None else None,
            )
        self._refresh_confirmations()

    def _refresh_confirmations(self) -> None:
        """展示可恢复的目标确认选项；传参：无；返回：无。"""
        if not self.config.state.session_id or not SessionMessageStore(
            self.config.data_root
        ).exists(self.config.state.session_id):
            return
        with completion_service(self.config.data_root) as confirmations:
            text = self._completion_menu.update(
                confirmations.pending(self.config.state.session_id)
            )
        if text:
            get_console().print(Text(text, style="yellow"))


def _present_approval(identity: str, request: ApprovalRequest) -> None:
    """展示具体动作/资源和带编号决定方式；传参：编号与审批；返回：无，不读取输入。"""
    scope = (
        f"{request.resource.action}: {request.resource.target}"
        if request.resource
        else str(dict(request.args))
    )
    get_console().print(
        Text(
            f"\n需要授权：{request.tool}\n范围：{scope}\n"
            f"输入 /approve {identity} once（仅这次）、task（本任务相同范围）、permanent（永久相同范围）或 deny（拒绝）\n"
            "直接输入新的要求会取消这次待审批动作。",
            style="yellow",
        )
    )


def _present_batch(identity: str, request: ApprovalBatch) -> None:
    """同一次界面展示整批动作，逐项选择后一次提交；传参：交互编号和批次；返回：无。"""
    get_console().print(
        Text(
            f"{format_batch(request)}\n输入 /approve {identity} 后跟全部选择；直接输入新要求会取消整批。",
            style="yellow",
        )
    )


@contextmanager
def local_session_resources(
    state: ReplState,
    *,
    project_root: Path,
    data_root: Path,
    llm_client: LLMClient | None,
    tool_registry: ToolRegistry | None = None,
    show_working: bool = False,
    extensions: RuntimeExtensions | None = None,
    host_factory: Callable[[SessionHostConfig], SessionHost] | None = None,
) -> Iterator[tuple[TaskStore, ToolRegistry, SessionHost]]:
    """管理本地显示入口的完整资源范围；传参：身份、目录及注入依赖；返回：存储、注册表和宿主。"""
    with ExitStack() as cleanup:
        store = cleanup.enter_context(closing(TaskStore(data_root)))
        with ExitStack() as pending_host:
            registry = tool_registry or build_tool_registry(
                repo_root=project_root, data_root=data_root
            )
            pending_host.callback(registry.close)
            host = (host_factory or SessionHost)(
                SessionHostConfig(
                    state,
                    registry,
                    llm_client,
                    project_root,
                    data_root,
                    show_working=show_working,
                    extensions=extensions,
                )
            )
            # 【本地会话】【所有权交接】构造成功后注册表归宿主；构造失败由入口清理，不重复关闭
            pending_host.pop_all()
        cleanup.callback(register_batch_backend, None)
        cleanup.callback(approval.register_approval_backend, None)
        cleanup.callback(host.close)
        approval.register_approval_backend(host.approvals.request)
        register_batch_backend(host.approvals.request_batch)
        yield store, registry, host
