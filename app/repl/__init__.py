"""REPL package for the Reins CLI.

Provides the interactive read-eval-print loop with rich rendering and slash
commands. The REPL remains a script, compatibility, and diagnostic entry;
`python -m app.cli` lands here when no subcommand is given. The primary
interactive terminal entry lives under `frontends/tui/`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.repl.session_host import SessionHost, SessionHostConfig

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from app.repl.console import enable_utf8_console, get_console
from app.session_assembly import user_lease
from app.repl.slash_commands import (
    ReplState,
    SlashCommandContext,
    SlashCommandRegistry,
    create_default_registry,
    format_prompt_label,
    format_session_banner,
)
from approval import ApprovalDecision, ApprovalRequest
from llm.base import LLMClient
from runtime.extensions import RuntimeExtensions
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry


PromptFn = Callable[[str], str]


def run_repl(
    *,
    project_root: Path,
    data_root: Path,
    llm_client: LLMClient,
    tool_registry: ToolRegistry | None = None,
    prompt_fn: PromptFn | None = None,
    initial_task_id: str | None = None,
    initial_message: str | None = None,
    max_turns: int | None = None,
    extensions: RuntimeExtensions | None = None,
    host_factory: Callable[[SessionHostConfig], SessionHost] | None = None,
) -> int:
    """Run the interactive REPL until the user types `/exit` or hits EOF.

    `prompt_fn` is overridable for tests so they can drive the loop with a
    scripted iterator without needing real stdin. Defaults to the built-in
    `input(...)` for production.

    `initial_message` and `max_turns` are also test-friendly hooks: an
    initial message is processed before the first prompt, and `max_turns`
    caps the number of REPL iterations (None = unbounded)."""
    from app.repl.session_host import local_session_resources

    enable_utf8_console()
    state = ReplState(current_task_id=initial_task_id)
    with local_session_resources(
        state,
        project_root=project_root,
        data_root=data_root,
        llm_client=llm_client,
        tool_registry=tool_registry,
        extensions=extensions,
        host_factory=host_factory,
    ) as (store, registry, host):
        context = SlashCommandContext(
            repl_state=state,
            store=store,
            registry=registry,
            llm_client=llm_client,
            project_root=project_root,
            data_root=data_root,
            prompt_fn=prompt_fn or input,
        )
        _print_repl_banner(context, warn=True)
        try:
            return _run_repl_loop(
                context, host, initial_message=initial_message, max_turns=max_turns
            )
        except KeyboardInterrupt:
            host.cancel()
    return 0


def _print_repl_banner(context: SlashCommandContext, *, warn: bool = False) -> None:
    """展示会话身份和模型配置；传参：命令上下文及首次提示标志；返回：无。"""
    console, state = get_console(), context.repl_state
    client = context.llm_client
    console.print(
        format_session_banner(
            project_root=context.project_root,
            data_root=context.data_root,
            model=_describe_model(client),
            provider=_describe_provider(client),
            session_id=state.session_id,
            current_run_id=state.current_run_id,
            focus_task_id=state.current_task_id,
            compatibility_task_id=state.compatibility_task_id,
        )
    )
    if not warn:
        return
    if _describe_model(client) == "(unconfigured)":
        console.print(
            "[bold yellow]⚠ model/provider 未配置 — LLM 功能不可用，仅 slash 命令可用。"
            "请设置 API key 和 model。[/bold yellow]\n"
        )
    else:
        target = getattr(client, "resolved_target", None)
        if target is not None and getattr(target, "context_window_defaulted", False):
            console.print(
                f"[bold yellow]⚠ 上下文窗口在用默认值 {target.context_window} — "
                "未按模型真实窗口配置，较长历史可能被过早截断。"
                "建议在 models.json/profile 设置 context_window。[/bold yellow]\n"
            )


def _run_repl_loop(
    context: SlashCommandContext,
    host: SessionHost,
    *,
    initial_message: str | None,
    max_turns: int | None,
) -> int:
    """接纳正文并处理界面命令；传参：上下文/宿主/初始输入/测试轮数；返回：退出码。"""
    pending_message, turns = initial_message, 0
    commands = create_default_registry()
    while max_turns is None or turns < max_turns:
        context.project_root = host.config.project_root
        context.registry = host.config.registry
        context.llm_client = host.config.llm_client
        if pending_message is not None:
            raw, pending_message = pending_message, None
        else:
            try:
                raw = context.prompt_fn(format_prompt_label(context.repl_state))
            except EOFError:
                get_console().print("\n[dim]EOF received, exiting.[/dim]")
                break
        turns += 1
        raw = raw.strip()
        if not raw or host.handle_control(raw):
            continue
        if raw.startswith("/"):
            if _dispatch_repl_command(context, host, commands, raw):
                return 0
        else:
            host.submit(raw)
    return 0


def _dispatch_repl_command(
    context: SlashCommandContext,
    host: SessionHost,
    commands: SlashCommandRegistry,
    text: str,
) -> bool:
    """处理现有本地命令并返回退出意图；传参：上下文/宿主/命令表/文本；返回：是否退出。"""
    host.prepare_command(text)
    result = commands.dispatch(text, context)
    if result is None:
        return False
    if context.repl_state.session_id != getattr(
        host, "session_id", context.repl_state.session_id
    ):
        from app.background.frontend import BackgroundSessionHost

        if isinstance(host, BackgroundSessionHost):
            host.attach_session(context.repl_state.session_id)
            context.project_root = host.config.project_root
            context.registry = host.config.registry
            context.llm_client = host.config.llm_client
    if result.clear_screen:
        _clear_screen()
        _print_repl_banner(context)
    if result.message:
        get_console().print(result.message)
    if result.session_input is not None:
        host.submit(result.session_input)
    if result.enter_dashboard:
        _render_dashboard_view(
            context.repl_state,
            context.store,
            context.registry,
            context.project_root,
            context.data_root,
        )
    return result.should_exit


# ---------------------------------------------------------------------------
# Internal helpers


def _make_cli_approval_backend() -> Callable[[ApprovalRequest], ApprovalDecision]:
    """Build a stateless approval backend that prompts the user inline through
    rich, rendering only the ApprovalRequest fields (tool/args/risk/message)."""

    def backend(req: ApprovalRequest) -> ApprovalDecision:
        console = get_console()
        body = Text.assemble(
            (f"tool: {req.tool}\n", "bold"),
            (f"args: {dict(req.args)}\n", ""),
            (f"risk: {req.risk}\n", "yellow"),
            (f"reason: {req.message}", ""),
        )
        console.print(
            Panel(body, title="approval required", border_style="red", expand=False)
        )
        if not Confirm.ask("Approve?", default=False, console=console):
            return ApprovalDecision.DENY
        scope = Prompt.ask(
            "Scope",
            choices=["once", "task", "permanent"],
            default="once",
            console=console,
        )
        return ApprovalDecision(scope)

    return backend


def _render_dashboard_view(
    state: ReplState,
    store: TaskStore,
    registry: ToolRegistry,
    project_root: Path,
    data_root: Path,
) -> None:
    # Imported lazily to avoid pulling rich.Layout into startup latency
    # for users who never invoke the dashboard view.
    from app.repl.dashboard import render_dashboard

    render_dashboard(
        state=state,
        store=store,
        registry=registry,
        project_root=project_root,
        data_root=data_root,
        lease=user_lease(
            task_id=state.current_task_id or state.compatibility_task_id or "",
            project_root=project_root,
            data_root=data_root,
        ),
    )


def _describe_model(client: LLMClient | None) -> str:
    """展示当前工作区模型或未配置状态；参数：可为空的模型客户端；返回：模型显示名称。"""
    target = getattr(client, "resolved_target", None)
    if target is not None:
        model = getattr(target, "model", None)
        if model:
            return str(model)
    config = getattr(client, "config", None)
    if config is None:
        return "(unconfigured)"
    return getattr(config, "model", None) or "(unconfigured)"


def _describe_provider(client: LLMClient | None) -> str:
    """展示当前工作区供应商或未配置状态；参数：可为空的模型客户端；返回：供应商显示名称。"""
    target = getattr(client, "resolved_target", None)
    if target is not None:
        provider = getattr(target, "provider", None)
        if provider:
            return str(provider)
    config = getattr(client, "config", None)
    if config is None:
        return "(unconfigured)"
    base = getattr(config, "base_url", None) or "(unconfigured)"
    return str(base)


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


__all__ = ["run_repl"]
