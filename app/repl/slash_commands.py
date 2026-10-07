"""Slash command registry for the Reins REPL.

Mirrors EvoHarness `harness/slash_commands.py` in shape (registry, dispatch,
SlashCommandResult) but the command set is tailored to V2.1 surfaces:
tasks, lease, watchdog, conversation summary, MCP, memory, skills, and
trace-level toggling.

The command list (10 MVP-0 commands per the new task PRD R11) intentionally
stays small. A future task can add `/redact /trace memory add /skill show
/mcp enable /approve /deny` once their dependencies stabilize.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.table import Table

from app.repl.console import get_console
from app.repl.model_commands import handle_model_command
from app.repl.status import (
    classify_run_status,
    read_latest_run_errors,
)
from llm.base import LLMClient
from frontends.shared.session_search import (
    SearchHit,
    SearchResult,
    SessionSearchService,
)
from runtime.checkpoint import (
    Checkpoint,
    load_checkpoint,
    load_checkpoint_by_id,
    load_latest_checkpoint,
    load_latest_checkpoint_for_run,
    load_latest_checkpoint_for_session,
    list_recent_checkpoints,
    summarize_checkpoint,
)
from runtime.run_facts import RunFactStore, RunSummary
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import read_history_rows
from runtime.session_state import SessionState, SessionStateStore
from runtime.types import new_session_id
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry


@dataclass(slots=True)
class SlashCommandResult:
    """Outcome of dispatching one command line.

    `should_exit` ends the REPL. `clear_screen` clears the terminal
    before printing `message`. `enter_dashboard` requests a one-shot
    switch to the dashboard view; the REPL main loop is responsible
    for actually rendering it (keeps render imports out of this
    module to avoid cycles)."""

    message: str | None = None
    should_exit: bool = False
    clear_screen: bool = False
    enter_dashboard: bool = False
    new_task_id: str | None = None
    pause_requested: bool = False
    model_config_pending_restart: bool = False
    model_config_applied_to_active_client: bool = False
    session_input: str | None = None


@dataclass(slots=True)
class SlashCommandContext:
    """State the REPL passes to each command handler. Not every
    handler needs every field; missing fields stay None and the
    handlers handle that case explicitly."""

    repl_state: "ReplState"
    store: TaskStore
    registry: ToolRegistry
    llm_client: LLMClient | None
    project_root: Path
    data_root: Path
    prompt_fn: Callable[[str], str]


@dataclass(slots=True)
class ReplState:
    """Mutable state the REPL holds across the prompt loop. Commands may
    read or write it; the REPL main loop owns the canonical instance."""

    current_task_id: str | None = None
    compatibility_task_id: str | None = None
    session_id: str = ""
    current_run_id: str = ""
    trace_on: bool = False
    toolsets_enabled: list[str] | None = None
    toolsets_disabled: list[str] | None = None
    defer_session: bool = False

    def __post_init__(self) -> None:
        if not self.session_id and not self.defer_session:
            self.ensure_session()

    def ensure_session(self) -> str:
        if not self.session_id:
            self.session_id = new_session_id()
        return self.session_id


@dataclass(slots=True)
class _ResumeResolution:
    source: str
    checkpoint: Checkpoint | None = None
    task_id: str | None = None
    message: str = ""


CommandHandler = Callable[[str, SlashCommandContext], SlashCommandResult]


@dataclass(slots=True)
class SlashCommand:
    name: str
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()


class SlashCommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._alias_to_name: dict[str, str] = {}

    def register(self, command: SlashCommand) -> None:
        if command.name in self._commands:
            raise ValueError(f"slash command already registered: {command.name}")
        self._commands[command.name] = command
        for alias in command.aliases:
            if alias in self._alias_to_name or alias in self._commands:
                raise ValueError(f"slash command alias collision: {alias}")
            self._alias_to_name[alias] = command.name

    def dispatch(
        self, raw_input: str, context: SlashCommandContext
    ) -> SlashCommandResult | None:
        """Return None if `raw_input` is not a slash command; return a
        SlashCommandResult otherwise (including unknown-command errors,
        which surface as `message` so the REPL can print them)."""
        if not raw_input.startswith("/"):
            return None
        body = raw_input[1:]
        name, _, args = body.partition(" ")
        name = name.strip()
        if not name:
            return SlashCommandResult(message="Type /help for available commands.")
        canonical = self._alias_to_name.get(name, name)
        command = self._commands.get(canonical)
        if command is None:
            return SlashCommandResult(message=f"Unknown command: /{name} (try /help)")
        return command.handler(args.strip(), context)

    def visible_commands(self) -> list[SlashCommand]:
        return sorted(self._commands.values(), key=lambda item: item.name)


def create_default_registry() -> SlashCommandRegistry:
    """Build the MVP-0 registry. Handlers close over no state — the REPL
    passes a SlashCommandContext per dispatch so the same registry serves
    multiple concurrent REPL sessions in tests."""
    from app.repl.toolsets_command import handle_toolsets

    registry = SlashCommandRegistry()

    def _help(_args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        del ctx
        lines = ["Available commands:"]
        for cmd in registry.visible_commands():
            alias_text = (
                f" (aliases: {', '.join('/' + a for a in cmd.aliases)})"
                if cmd.aliases
                else ""
            )
            lines.append(f"  /{cmd.name:<11} {cmd.description}{alias_text}")
        return SlashCommandResult(message="\n".join(lines))

    def _exit(_args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        del ctx
        return SlashCommandResult(message="Bye.", should_exit=True)

    def _clear(_args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        del ctx
        return SlashCommandResult(clear_screen=True)

    def _status(_args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        return SlashCommandResult(message=_render_status(ctx))

    def _tasks(_args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        records = [
            record for record in ctx.store.list_tasks(limit=None) if not record.is_inbox
        ][:20]
        if not records:
            return SlashCommandResult(message="No active tasks yet.")
        table = Table(title=f"Active tasks ({len(records)})", show_lines=False)
        table.add_column("task_id", style="cyan")
        table.add_column("status")
        table.add_column("goal")
        table.add_column("updated_at", style="dim")
        for record in records:
            marker = " *" if record.task_id == ctx.repl_state.current_task_id else ""
            table.add_row(
                record.task_id + marker,
                record.status,
                _truncate(record.goal, 60),
                record.updated_at,
            )
        get_console().print(table)
        return SlashCommandResult()

    def _task(args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        sub, _, rest = args.partition(" ")
        sub = sub.strip()
        rest = rest.strip()
        if not sub:
            current = ctx.repl_state.current_task_id or "(none)"
            compat = ctx.repl_state.compatibility_task_id or "(none)"
            return SlashCommandResult(
                message=(
                    f"Current task: {current}\n"
                    f"Focus task: {current}\n"
                    f"Compatibility task: {compat}"
                )
            )
        if sub in {"clear", "none", "unfocus"}:
            previous_task_id = ctx.repl_state.current_task_id
            previous = previous_task_id or "(none)"
            ctx.repl_state.current_task_id = None
            ctx.repl_state.ensure_session()
            SessionStateStore(ctx.data_root).update_focus(
                ctx.repl_state.session_id,
                focus_task_id=None,
                previous_focus_task_id=previous_task_id,
                compatibility_task_id=ctx.repl_state.compatibility_task_id,
                summary=f"Focus task cleared. Previous focus: {previous}",
            )
            return SlashCommandResult(
                message=f"Focus task cleared. Previous focus: {previous}"
            )
        if sub == "new":
            if not rest:
                return SlashCommandResult(message="Usage: /task new <goal>")
            previous_task_id = ctx.repl_state.current_task_id
            created = ctx.store.create_task(rest)
            ctx.repl_state.current_task_id = created.task_id
            ctx.repl_state.ensure_session()
            SessionStateStore(ctx.data_root).update_focus(
                ctx.repl_state.session_id,
                focus_task_id=created.task_id,
                previous_focus_task_id=previous_task_id,
                compatibility_task_id=ctx.repl_state.compatibility_task_id,
                summary=f"Focus task set to {created.task_id}: {rest}",
            )
            return SlashCommandResult(
                message=f"Created task {created.task_id}\nGoal: {rest}",
                new_task_id=created.task_id,
            )
        # Otherwise treat the first token as a task_id to switch to.
        target_id = sub
        record = ctx.store.load_task(target_id)
        if record is None:
            return SlashCommandResult(message=f"Task not found: {target_id}")
        previous_task_id = ctx.repl_state.current_task_id
        ctx.repl_state.current_task_id = record.task_id
        ctx.repl_state.ensure_session()
        SessionStateStore(ctx.data_root).update_focus(
            ctx.repl_state.session_id,
            focus_task_id=record.task_id,
            previous_focus_task_id=previous_task_id,
            compatibility_task_id=ctx.repl_state.compatibility_task_id,
            summary=f"Focus task set to {record.task_id}: {record.goal}",
        )
        return SlashCommandResult(
            message=f"Switched to task {record.task_id}\nGoal: {record.goal}",
            new_task_id=record.task_id,
        )

    def _resume(args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        target = args.strip() or "latest"
        resolution = _resolve_resume_target(target, ctx)
        if resolution is None:
            return SlashCommandResult(message=_resume_failure_message(target, ctx))
        _apply_resume_resolution(ctx, resolution)
        return SlashCommandResult(
            message=_format_resume_message(resolution, ctx),
            new_task_id=ctx.repl_state.current_task_id,
        )

    def _pause(_args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        if ctx.repl_state.current_task_id is None:
            return SlashCommandResult(message="No active task to pause.")
        return SlashCommandResult(
            message=(
                f"Pause requested. Task {ctx.repl_state.current_task_id} will pause "
                "after the current segment finishes."
            ),
            pause_requested=True,
        )

    def _compact(_args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        """将显式压缩请求交给共用会话运行；传参：命令与会话；返回：接纳请求，不预先修改原文。"""
        owner = SessionMessageStore(ctx.data_root)
        session_id = ctx.repl_state.session_id
        if not session_id or not owner.exists(session_id):
            return SlashCommandResult(message="会话为空，没有需要整理的历史。")
        if len(owner.materialize(session_id).messages) < 2:
            return SlashCommandResult(message="会话历史较短，尚无可整理的完整片段。")
        if ctx.llm_client is None:
            return SlashCommandResult(message="请先配置模型，再生成会话摘要。")
        return SlashCommandResult(session_input="/compact")

    def _dashboard(_args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        del ctx
        return SlashCommandResult(enter_dashboard=True)

    def _model(args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        return handle_model_command(args, ctx)

    def _search(args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        return SlashCommandResult(message=_render_search(args, ctx))

    def _toolsets(args: str, ctx: SlashCommandContext) -> SlashCommandResult:
        result = handle_toolsets(args, ctx)
        if isinstance(result, SlashCommandResult):
            return result
        return SlashCommandResult(message=str(result))

    registry.register(
        SlashCommand("help", "List slash commands", _help, aliases=("?",))
    )
    registry.register(SlashCommand("exit", "Exit the REPL", _exit, aliases=("quit",)))
    registry.register(SlashCommand("clear", "Clear the screen", _clear))
    registry.register(
        SlashCommand(
            "status", "Show session, run, focus task, and recovery summary", _status
        )
    )
    registry.register(SlashCommand("tasks", "List active tasks", _tasks))
    registry.register(
        SlashCommand("task", "Show, switch, clear, or create focus task", _task)
    )
    registry.register(
        SlashCommand("resume", "Resume a session, run, checkpoint, or task", _resume)
    )
    registry.register(
        SlashCommand("pause", "Request a pause after current segment", _pause)
    )
    registry.register(
        SlashCommand(
            "compact", "Manually update summary.md from the recent tail", _compact
        )
    )
    registry.register(
        SlashCommand("dashboard", "Switch to the full dashboard view", _dashboard)
    )
    registry.register(
        SlashCommand("model", "Show or set model/provider configuration", _model)
    )
    registry.register(
        SlashCommand("search", "Search sessions by keyword or ID", _search)
    )
    registry.register(
        SlashCommand("requests", "查看实际模型请求、尝试、工具结果及导出", _requests)
    )
    registry.register(SlashCommand("restore", "选择文件、预览差异并确认恢复", _restore))
    registry.register(
        SlashCommand(
            "toolsets", "Show or update model-visible toolset policy", _toolsets
        )
    )
    return registry


def _restore(args: str, ctx: SlashCommandContext) -> SlashCommandResult:
    """指向独立文件恢复界面，不直接写文件；参数：命令参数与会话；返回：入口说明。"""
    if args:
        return SlashCommandResult(
            message="用法：/restore；先选择文件、预览差异，再明确确认恢复"
        )
    return SlashCommandResult(
        message=f"会话 {ctx.repl_state.session_id} 的文件恢复：在 reins-tui 按 F10 或使用 /restore"
    )


def _requests(args: str, ctx: SlashCommandContext) -> SlashCommandResult:
    """提供共享请求查看入口说明；参数：命令参数和会话；返回：正式界面定位说明。"""
    if args:
        return SlashCommandResult(
            message="用法：/requests；在请求记录面板选择运行、请求和实际尝试"
        )
    return SlashCommandResult(
        message=f"会话 {ctx.repl_state.session_id} 的实际请求记录：在 reins-tui 按 F9 或使用 /requests 查看及导出"
    )


def _resolve_resume_target(
    target: str, ctx: SlashCommandContext
) -> _ResumeResolution | None:
    if target == "latest":
        return _resolve_latest_resume(ctx)
    if "::" in target:
        task_id, checkpoint_id = target.split("::", maxsplit=1)
        checkpoint = load_checkpoint(task_id, checkpoint_id, data_root=ctx.data_root)
        if checkpoint is None:
            return None
        return _ResumeResolution(source="task checkpoint", checkpoint=checkpoint)
    if target.startswith("run-"):
        checkpoint = load_latest_checkpoint_for_run(target, data_root=ctx.data_root)
        if checkpoint is None:
            return None
        return _ResumeResolution(source="run", checkpoint=checkpoint)
    if target.startswith("session-"):
        checkpoint = load_latest_checkpoint_for_session(target, data_root=ctx.data_root)
        if checkpoint is None:
            return None
        return _ResumeResolution(source="session", checkpoint=checkpoint)
    record = ctx.store.load_task(target)
    if record is not None:
        checkpoint = load_latest_checkpoint(record.task_id, data_root=ctx.data_root)
        source = "task checkpoint" if checkpoint is not None else "task"
        return _ResumeResolution(
            source=source, checkpoint=checkpoint, task_id=record.task_id
        )
    checkpoint = load_checkpoint_by_id(target, data_root=ctx.data_root)
    if checkpoint is not None:
        return _ResumeResolution(source="checkpoint", checkpoint=checkpoint)
    return None


def _resolve_latest_resume(ctx: SlashCommandContext) -> _ResumeResolution | None:
    session_state = (
        SessionStateStore(ctx.data_root).load(ctx.repl_state.session_id)
        if ctx.repl_state.session_id
        else None
    )
    checkpoint = None
    if ctx.repl_state.session_id:
        if session_state is not None and session_state.last_run_id:
            checkpoint = load_latest_checkpoint_for_run(
                session_state.last_run_id, data_root=ctx.data_root
            )
        if checkpoint is not None:
            return _ResumeResolution(source="current session", checkpoint=checkpoint)
        checkpoint = load_latest_checkpoint_for_session(
            ctx.repl_state.session_id, data_root=ctx.data_root
        )
        if checkpoint is not None:
            return _ResumeResolution(source="current session", checkpoint=checkpoint)
    if ctx.repl_state.current_run_id:
        checkpoint = load_latest_checkpoint_for_run(
            ctx.repl_state.current_run_id, data_root=ctx.data_root
        )
        if checkpoint is not None:
            return _ResumeResolution(source="current run", checkpoint=checkpoint)
    for run in RunFactStore(ctx.data_root).list_recent_runs(limit=10):
        checkpoint = load_latest_checkpoint_for_run(run.run_id, data_root=ctx.data_root)
        if checkpoint is not None:
            return _ResumeResolution(source="recent run", checkpoint=checkpoint)
    records = [
        record
        for record in ctx.store.list_tasks(status="paused", limit=None)
        if not record.is_inbox
    ]
    if not records:
        return None
    record = records[0]
    checkpoint = load_latest_checkpoint(record.task_id, data_root=ctx.data_root)
    return _ResumeResolution(
        source="paused task", checkpoint=checkpoint, task_id=record.task_id
    )


def _apply_resume_resolution(
    ctx: SlashCommandContext, resolution: _ResumeResolution
) -> None:
    checkpoint = resolution.checkpoint
    if checkpoint is None:
        ctx.repl_state.current_task_id = resolution.task_id
        return
    if checkpoint.session_id:
        ctx.repl_state.session_id = checkpoint.session_id
    if checkpoint.run_id:
        ctx.repl_state.current_run_id = checkpoint.run_id
    ctx.repl_state.compatibility_task_id = checkpoint.compatibility_task_id
    ctx.repl_state.current_task_id = _formal_focus_from_checkpoint(ctx, checkpoint)


def _formal_focus_from_checkpoint(
    ctx: SlashCommandContext, checkpoint: Checkpoint
) -> str | None:
    candidates = [checkpoint.focus_task_id, checkpoint.task_id]
    for task_id in candidates:
        if not task_id:
            continue
        record = ctx.store.load_task(task_id)
        if record is not None and not record.is_inbox:
            return record.task_id
    return None


def _format_resume_message(
    resolution: _ResumeResolution, ctx: SlashCommandContext
) -> str:
    checkpoint = resolution.checkpoint
    if checkpoint is None:
        return "\n".join(
            [
                f"Resumed legacy task {resolution.task_id}.",
                "No checkpoint was found for that task; the next prompt will continue with it as focus.",
            ]
        )
    summary = summarize_checkpoint(checkpoint)
    focus = ctx.repl_state.current_task_id or "(none)"
    compat = ctx.repl_state.compatibility_task_id or "(none)"
    lines = [
        f"Resume ready from {resolution.source}.",
        f"Session    : {summary.session_id or '(none)'}",
        f"Run        : {summary.run_id or '(none)'}",
        f"Checkpoint : {summary.checkpoint_id} ({summary.state})",
        f"Focus task : {focus}",
        f"Compat task: {compat}",
    ]
    lines.extend(_format_resume_transcript(checkpoint, ctx))
    lines.append("Type your next prompt to continue from this recovery point.")
    return "\n".join(lines)


def _format_resume_transcript(
    checkpoint: Checkpoint, ctx: SlashCommandContext
) -> list[str]:
    # 恢复回显的对话来自 checkpoint 记下的会话，消息 owner 按 session 归档
    session_id = checkpoint.session_id
    if not session_id:
        return ["", "Conversation: (no source session found)"]
    rows = read_history_rows(ctx.data_root, session_id, limit=0)
    lines = ["", f"Conversation ({session_id}):"]
    if not rows:
        lines.append("  (no conversation rows found)")
        return lines
    for index, row in enumerate(rows, start=1):
        lines.extend(_format_conversation_row(index, row))
    return lines


def _format_conversation_row(index: int, row: dict[str, object]) -> list[str]:
    role = str(row.get("role", "?"))
    content = str(row.get("content", ""))
    return [f"  {index}. [{role}]", _indent_conversation_content(content)]


def _indent_conversation_content(content: str) -> str:
    lines = content.splitlines() or [""]
    return "\n".join(f"     {line}" for line in lines)


def _resume_failure_message(target: str, ctx: SlashCommandContext) -> str:
    if target == "latest":
        return (
            "No recovery point found for the current session or recent runs, "
            "and no paused task is available."
        )
    if target.startswith("session-"):
        return f"No checkpoint found for session: {target}"
    if target.startswith("run-"):
        return f"No checkpoint found for run: {target}"
    if "::" in target:
        task_id, checkpoint_id = target.split("::", maxsplit=1)
        return f"No checkpoint {checkpoint_id} found for task: {task_id}"
    if ctx.store.load_task(target) is None:
        return f"Task, run, session, or checkpoint not found: {target}"
    return f"No checkpoint found for task: {target}"


def _render_search(args: str, ctx: SlashCommandContext) -> str:
    deep_scan, limit, query = _parse_search_args(args)
    if not query:
        return (
            "Usage: /search <query> | /search --deep <query> | "
            "/search --limit N <query>"
        )
    sessions_root = ctx.data_root / "sessions"
    if not sessions_root.is_dir():
        return "暂无会话记录。运行一次任务后即可搜索历史会话。"
    service = SessionSearchService(ctx.data_root)
    result = service.search(query, limit=limit, deep_scan=deep_scan)
    return _format_search_result(query, result, limit=limit)


def _parse_search_args(args: str) -> tuple[bool, int, str]:
    """Parse `/search` args. Returns (deep_scan, limit, remaining_query)."""
    tokens = args.split()
    deep_scan = False
    limit = 10
    remaining: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--deep":
            deep_scan = True
            idx += 1
            continue
        if token == "--limit":
            if idx + 1 < len(tokens):
                try:
                    limit = max(1, int(tokens[idx + 1]))
                except ValueError:
                    pass
                idx += 2
                continue
            idx += 1
            continue
        if token.startswith("--limit="):
            try:
                limit = max(1, int(token.split("=", 1)[1]))
            except ValueError:
                pass
            idx += 1
            continue
        remaining.append(token)
        idx += 1
    return deep_scan, limit, " ".join(remaining).strip()


def _format_search_result(query: str, result: "SearchResult", *, limit: int) -> str:
    lines: list[str] = [f'搜索: "{query}"']
    for warning in result.warnings:
        lines.append(f"提示: {warning.message}")
    if not result.hits:
        if not any(
            warning.code in {"no_sessions_dir", "token_too_short"}
            for warning in result.warnings
        ):
            lines.append("未找到匹配会话")
        return "\n".join(lines)
    lines.append(f"匹配 {len(result.hits)} 个会话:")
    lines.append("")
    for hit in result.hits:
        lines.extend(_format_search_hit(hit))
    if result.truncated or len(result.hits) >= limit:
        lines.append(f"还有更多结果，用 /search --limit {limit * 3} {query} 查看更多")
    return "\n".join(lines)


def _format_search_hit(hit: "SearchHit") -> list[str]:
    run_label = hit.last_run_id or "(none)"
    status_label = hit.status or "(none)"
    focus_label = hit.focus_task_id or "(none)"
    header = (
        f"{hit.session_id} | run={run_label} | status={status_label} | "
        f"focus={focus_label}"
    )
    rows = [header, f"  来源: {hit.hit_source}"]
    if hit.summary_snippet:
        rows.append(f"  摘要: {hit.summary_snippet}")
    if hit.trigger_badge:
        rows.append(f"  trigger: {hit.trigger_badge}")
    return rows


def _render_status(ctx: SlashCommandContext) -> str:
    session_store = SessionStateStore(ctx.data_root)
    session_state = session_store.load(ctx.repl_state.session_id)
    focus_task_id = _status_focus_task_id(ctx, session_state)
    focus_record = ctx.store.load_task(focus_task_id) if focus_task_id else None
    recovery = _latest_recovery_line(ctx)
    recent_run = _recent_run_line(ctx)
    run_summary = _recent_run_summary(ctx, session_state)
    run_facts = (
        RunFactStore(ctx.data_root).read_run(run_summary.run_id) if run_summary else []
    )
    run_errors = (
        read_latest_run_errors(
            ctx.data_root,
            session_id=run_summary.session_id,
            run_id=run_summary.run_id,
        )
        if run_summary is not None
        else []
    )
    run_state = classify_run_status(
        run_facts,
        errors=run_errors,
        checkpoint_state=session_state.last_checkpoint_state if session_state else "",
        session_status=session_state.last_run_status if session_state else "",
    )
    lines = [
        f"Session    : {_session_display(ctx.repl_state)}",
        f"Current run: {ctx.repl_state.current_run_id or '(none)'}",
        f"Recent run : {recent_run}",
        f"Run state  : {run_state.label} ({run_state.category})",
        f"Meaning    : {run_state.detail}",
        f"Next step  : {run_state.next_step}",
        f"Focus task : {ctx.repl_state.current_task_id or '(none)'}",
        f"Compat task: {ctx.repl_state.compatibility_task_id or '(none)'}",
        f"Recovery   : {recovery}",
        f"Trace      : {'on' if ctx.repl_state.trace_on else 'off'}",
    ]
    if session_state is not None:
        lines.append(
            f"Session sum: {_truncate(session_state.summary or '(empty)', 200)}"
        )
    if focus_task_id is not None and focus_record is None:
        lines.append(f"Focus task missing on disk: {focus_task_id}")
    if focus_record is not None:
        layers = ctx.store.read_summary_layers(focus_record.task_id)
        history_count = len(
            read_history_rows(ctx.data_root, ctx.repl_state.session_id, limit=200)
        )
        lines.extend(
            [
                f"Task status: {focus_record.status}",
                f"Task goal  : {_truncate(focus_record.goal, 80)}",
                f"Updated    : {focus_record.updated_at}",
                f"Conv rows  : {history_count}",
                f"Intent     : {_truncate(layers.intent or '(empty)', 200)}",
                f"Resume hint: {_truncate(layers.resume_hint or '(empty)', 200)}",
                f"Progress   : {_truncate(layers.progress or layers.summary or '(empty)', 200)}",
            ]
        )
    lines.extend(
        [
            f"Project    : {ctx.project_root}",
            f"Data root  : {ctx.data_root}",
        ]
    )
    return "\n".join(lines)


def _status_focus_task_id(
    ctx: SlashCommandContext,
    session_state: SessionState | None,
) -> str | None:
    return (
        ctx.repl_state.current_task_id
        or ctx.repl_state.compatibility_task_id
        or (session_state.focus_task_id if session_state is not None else None)
        or (session_state.compatibility_task_id if session_state is not None else None)
    )


def _latest_recovery_line(ctx: SlashCommandContext) -> str:
    session_state = (
        SessionStateStore(ctx.data_root).load(ctx.repl_state.session_id)
        if ctx.repl_state.session_id
        else None
    )
    checkpoint = None
    if (
        ctx.repl_state.session_id
        and session_state is not None
        and session_state.last_run_id
    ):
        checkpoint = load_latest_checkpoint_for_run(
            session_state.last_run_id, data_root=ctx.data_root
        )
    source = "session"
    if checkpoint is None and ctx.repl_state.session_id:
        checkpoint = load_latest_checkpoint_for_session(
            ctx.repl_state.session_id, data_root=ctx.data_root
        )
    if checkpoint is None and ctx.repl_state.current_run_id:
        checkpoint = load_latest_checkpoint_for_run(
            ctx.repl_state.current_run_id, data_root=ctx.data_root
        )
        source = "run"
    if checkpoint is None:
        recent = list_recent_checkpoints(data_root=ctx.data_root, limit=1)
        if not recent:
            return "(none)"
        summary = recent[0]
        return (
            f"{summary.checkpoint_id} ({summary.state}, latest, "
            f"run={summary.run_id or '(none)'})"
        )
    summary = summarize_checkpoint(checkpoint)
    return (
        f"{summary.checkpoint_id} ({summary.state}, {source}, "
        f"run={summary.run_id or '(none)'})"
    )


def _recent_run_line(ctx: SlashCommandContext) -> str:
    summary = _recent_run_summary(
        ctx,
        SessionStateStore(ctx.data_root).load(ctx.repl_state.session_id),
    )
    if summary is not None:
        return _format_run_summary(summary)
    return "(none)"


def _recent_run_summary(
    ctx: SlashCommandContext,
    session_state: SessionState | None,
) -> RunSummary | None:
    if ctx.repl_state.session_id:
        if session_state is not None and session_state.last_run_id:
            return RunSummary(
                run_id=session_state.last_run_id,
                session_id=session_state.session_id,
                focus_task_id=session_state.focus_task_id,
                compatibility_task_id=session_state.compatibility_task_id,
                updated_at=session_state.updated_at,
                status=session_state.last_run_status,
                last_event=session_state.last_run_event,
            )
        runs = RunFactStore(ctx.data_root).list_runs_for_session(
            ctx.repl_state.session_id, limit=1
        )
        if runs:
            return runs[0]
    runs = RunFactStore(ctx.data_root).list_recent_runs(limit=1)
    if not runs:
        return None
    return runs[0]


def _format_run_summary(run: RunSummary) -> str:
    status = f", status={run.status}" if run.status else ""
    focus = f", focus={run.focus_task_id}" if run.focus_task_id else ""
    return f"{run.run_id} (session={run.session_id}{status}{focus})"


def _format_session_state_run(state: SessionState) -> str:
    status = f", status={state.last_run_status}" if state.last_run_status else ""
    focus = f", focus={state.focus_task_id}" if state.focus_task_id else ""
    return f"{state.last_run_id} (session={state.session_id}{status}{focus})"


def format_session_banner(
    project_root: Path,
    data_root: Path,
    *,
    model: str = "(unconfigured)",
    provider: str = "(unconfigured)",
    session_id: str = "",
    current_run_id: str = "",
    focus_task_id: str | None = None,
    compatibility_task_id: str | None = None,
) -> str:
    width = max(60, min(_terminal_width(), 100))
    bar = "=" * width
    dash = "-" * width
    lines = [
        bar,
        "Reins  |  V2.1 MVP-0 chat REPL",
        "type a request to send it, or use slash commands to steer the session",
        "/help  /tasks  /task new <goal>  /task clear  /resume  /status  /search  /compact  /dashboard  /exit",
        dash,
        f"project   : {project_root}",
        f"data root : {data_root}",
        f"model     : {model}",
        f"provider  : {provider}",
    ]
    if session_id:
        lines.extend(
            [
                f"session   : {session_id}",
                f"run       : {current_run_id or '(none)'}",
                f"focus task: {focus_task_id or '(none)'}",
                f"compat    : {compatibility_task_id or '(none)'}",
            ]
        )
    lines.append(bar)
    return "\n".join(lines)


def format_prompt_label(state: ReplState) -> str:
    if state.current_task_id:
        focus = state.current_task_id
    elif state.current_run_id:
        focus = f"run:{state.current_run_id}"
    else:
        focus = state.session_id or "not-started"
    short = focus[:18] + "..." if len(focus) > 21 else focus
    trace = "trace" if state.trace_on else "-"
    return f"[{short} | {trace}] > "


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _terminal_width() -> int:
    try:
        return shutil.get_terminal_size((80, 20)).columns
    except Exception:
        return 80


def _session_display(state: ReplState) -> str:
    return state.session_id or "(not started)"


__all__ = [
    "ReplState",
    "SlashCommand",
    "SlashCommandContext",
    "SlashCommandRegistry",
    "SlashCommandResult",
    "create_default_registry",
    "format_prompt_label",
    "format_session_banner",
]
