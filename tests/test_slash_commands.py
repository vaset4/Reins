"""Tests for the slash command registry."""

from __future__ import annotations
from scripts.testing.llm import from_test_stub

from collections.abc import Generator
from pathlib import Path

import pytest

from app.repl.console import reset_console_for_tests
from app.repl.slash_commands import (
    ReplState,
    SlashCommandContext,
    SlashCommandRegistry,
    SlashCommandResult,
    create_default_registry,
    format_prompt_label,
    format_session_banner,
)
from rich.console import Console
from runtime.checkpoint import Checkpoint, checkpoint_to_ledger_state, save_checkpoint
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore
from runtime.session_messages import append_assistant_message, append_user_message
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry


@pytest.fixture()
def recorded_console() -> Generator[Console, None, None]:
    console = Console(record=True, width=120, force_terminal=False, color_system=None)
    reset_console_for_tests(console)
    yield console
    reset_console_for_tests(None)


@pytest.fixture()
def context(tmp_path: Path) -> SlashCommandContext:
    project = tmp_path / "project"
    project.mkdir()
    data_root = project / ".reins" / "data"
    data_root.mkdir(parents=True)
    store = TaskStore(data_root)
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    return SlashCommandContext(
        repl_state=ReplState(),
        store=store,
        registry=registry,
        llm_client=None,
        project_root=project,
        data_root=data_root,
        prompt_fn=lambda _prompt: "",
    )


def test_dispatch_returns_none_for_non_slash() -> None:
    registry = create_default_registry()
    state = ReplState()
    ctx = SlashCommandContext(
        repl_state=state,
        store=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        llm_client=None,
        project_root=Path("."),
        data_root=Path("."),
        prompt_fn=lambda _: "",
    )
    assert registry.dispatch("hello", ctx) is None


def test_dispatch_unknown_command(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/notreal", context)
    assert result is not None
    assert "Unknown command" in (result.message or "")


def test_help_lists_all_commands(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/help", context)
    assert result is not None
    assert "Available commands" in (result.message or "")
    for name in (
        "help",
        "exit",
        "clear",
        "status",
        "tasks",
        "task",
        "resume",
        "pause",
        "compact",
        "dashboard",
        "toolsets",
    ):
        assert f"/{name}" in (result.message or "")


def test_help_alias_question_mark(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/?", context)
    assert result is not None
    assert "Available commands" in (result.message or "")


def test_exit_and_quit_request_exit(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    exit_result = registry.dispatch("/exit", context)
    quit_result = registry.dispatch("/quit", context)
    assert exit_result is not None
    assert quit_result is not None
    assert exit_result.should_exit is True
    assert quit_result.should_exit is True


def test_clear_signals_clear_screen(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/clear", context)
    assert result is not None
    assert result.clear_screen is True


def test_status_without_task(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/status", context)
    assert result is not None
    assert "Session" in (result.message or "")
    assert "Focus task : (none)" in (result.message or "")


def test_status_with_task(context: SlashCommandContext) -> None:
    record = context.store.create_task("inspect tools")
    context.repl_state.current_task_id = record.task_id
    registry = create_default_registry()
    result = registry.dispatch("/status", context)
    assert result is not None
    assert record.task_id in (result.message or "")
    assert "inspect tools" in (result.message or "")


def test_status_explains_recent_run_and_summary_layers(
    context: SlashCommandContext,
) -> None:
    record = context.store.create_task("describe Reins", task_id="task-status")
    context.repl_state.current_task_id = record.task_id
    context.store.update_summary_layers(
        record.task_id,
        intent="Introduce the current Reins project.",
        progress="Need to inspect the repository before writing.",
        resume_hint="Continue by reading project files.",
    )
    RunFactStore(context.data_root).append(
        {
            "event": "run:start",
            "session_id": context.repl_state.session_id,
            "run_id": "run-status",
            "task_id": record.task_id,
            "focus_task_id": record.task_id,
        }
    )
    RunFactStore(context.data_root).append(
        {
            "event": "state:transition",
            "session_id": context.repl_state.session_id,
            "run_id": "run-status",
            "task_id": record.task_id,
            "focus_task_id": record.task_id,
            "to_state": "FAILED",
        }
    )
    RunEvidenceStore(context.data_root).append_error(
        session_id=context.repl_state.session_id,
        run_id="run-status",
        error={
            "category": "invalid_model_protocol",
            "message": "MODEL_PROTOCOL_ERROR: invalid JSON",
        },
    )

    registry = create_default_registry()
    result = registry.dispatch("/status", context)

    message = result.message or ""
    assert result is not None
    assert "recoverable_failure" in message
    assert "Introduce the current Reins project." in message
    assert "Continue by reading project files." in message
    assert "Need to inspect the repository" in message


def test_tasks_lists_active_tasks(
    context: SlashCommandContext, recorded_console: Console
) -> None:
    context.store.create_task("first goal")
    context.store.create_task("second goal")
    registry = create_default_registry()
    result = registry.dispatch("/tasks", context)
    rendered = recorded_console.export_text()
    assert "first goal" in rendered
    assert "second goal" in rendered
    assert result is not None
    assert result.message is None  # output went straight to console


def test_tasks_empty_message(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/tasks", context)
    assert result is not None
    assert "No active tasks" in (result.message or "")


def test_tasks_hides_inbox_compatibility_tasks(
    context: SlashCommandContext, recorded_console: Console
) -> None:
    inbox = context.store.create_task(
        "plain chat", task_id="chat-compat", is_inbox=True
    )
    context.store.create_task("formal task", task_id="formal-task")

    registry = create_default_registry()
    result = registry.dispatch("/tasks", context)
    rendered = recorded_console.export_text()

    assert result is not None
    assert "formal task" in rendered
    assert inbox.task_id not in rendered
    assert "plain chat" not in rendered


def test_task_show_current(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/task", context)
    assert result is not None
    assert "Current task: (none)" in (result.message or "")


def test_task_new_creates_and_switches(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/task new explore filesystem", context)
    assert result is not None
    assert result.new_task_id is not None
    assert context.repl_state.current_task_id == result.new_task_id
    record = context.store.load_task(result.new_task_id)
    assert record is not None
    assert record.goal == "explore filesystem"


def test_task_new_requires_goal(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/task new", context)
    assert result is not None
    assert "Usage" in (result.message or "")


def test_task_switch_to_existing(context: SlashCommandContext) -> None:
    record = context.store.create_task("existing goal")
    registry = create_default_registry()
    result = registry.dispatch(f"/task {record.task_id}", context)
    assert result is not None
    assert context.repl_state.current_task_id == record.task_id
    assert "existing goal" in (result.message or "")


def test_task_switch_unknown(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/task does-not-exist", context)
    assert result is not None
    assert "Task not found" in (result.message or "")


def test_resume_no_paused(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/resume latest", context)
    assert result is not None
    assert "No recovery point" in (result.message or "")


def test_resume_latest_does_not_fallback_to_inbox_paused_task(
    context: SlashCommandContext,
) -> None:
    inbox = context.store.create_task(
        "plain chat", task_id="chat-compat", is_inbox=True
    )
    context.store.update_task_status(inbox.task_id, "paused")

    registry = create_default_registry()
    result = registry.dispatch("/resume latest", context)

    assert result is not None
    assert context.repl_state.current_task_id is None
    assert "No recovery point" in (result.message or "")


def test_resume_specific(context: SlashCommandContext) -> None:
    record = context.store.create_task("paused task")
    context.store.update_task_status(record.task_id, "paused")
    registry = create_default_registry()
    result = registry.dispatch(f"/resume {record.task_id}", context)
    assert result is not None
    assert context.repl_state.current_task_id == record.task_id
    assert "legacy task" in (result.message or "")


def test_resume_latest_prefers_current_session_checkpoint(
    context: SlashCommandContext,
) -> None:
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id="chat-compat",
            session_id=context.repl_state.session_id,
            run_id="run-session",
            compatibility_task_id="chat-compat",
            segment_id="user-1",
            state="PAUSED",
        )
    )
    RunFactStore(context.data_root).append_checkpoint_ref(
        session_id=context.repl_state.session_id,
        run_id="run-session",
        task_id=None,
        focus_task_id=None,
        compatibility_task_id="chat-compat",
        segment_id="user-1",
        checkpoint=checkpoint,
    )
    _record_ledger_checkpoint(context.data_root, checkpoint)

    registry = create_default_registry()
    result = registry.dispatch("/resume latest", context)

    assert result is not None
    assert context.repl_state.current_run_id == "run-session"
    assert context.repl_state.current_task_id is None
    assert context.repl_state.compatibility_task_id == "chat-compat"
    assert "current session" in (result.message or "")


def test_resume_run_and_session_targets_update_repl_state(
    context: SlashCommandContext,
) -> None:
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id="chat-compat",
            session_id="session-target",
            run_id="run-target",
            compatibility_task_id="chat-compat",
            segment_id="user-1",
            state="PAUSED",
        )
    )
    RunFactStore(context.data_root).append_checkpoint_ref(
        session_id="session-target",
        run_id="run-target",
        task_id=None,
        focus_task_id=None,
        compatibility_task_id="chat-compat",
        segment_id="user-1",
        checkpoint=checkpoint,
    )
    _record_ledger_checkpoint(context.data_root, checkpoint)

    registry = create_default_registry()
    by_run = registry.dispatch("/resume run-target", context)
    assert by_run is not None
    assert context.repl_state.session_id == "session-target"
    assert context.repl_state.current_run_id == "run-target"
    assert context.repl_state.current_task_id is None

    context.repl_state.session_id = "session-other"
    context.repl_state.current_run_id = ""
    by_session = registry.dispatch("/resume session-target", context)
    assert by_session is not None
    assert context.repl_state.session_id == "session-target"
    assert context.repl_state.current_run_id == "run-target"


def test_resume_session_renders_source_conversation(
    context: SlashCommandContext,
) -> None:
    record = context.store.create_task("chat transcript", is_inbox=True)
    # 回显取自 checkpoint 记录的会话，故种子落在同一 session 上
    append_user_message(context.data_root, "session-transcript", "first question")
    append_assistant_message(context.data_root, "session-transcript", "first answer")
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id=record.task_id,
            session_id="session-transcript",
            run_id="run-transcript",
            compatibility_task_id=record.task_id,
            segment_id="user-1",
            state="PAUSED",
        )
    )
    RunFactStore(context.data_root).append_checkpoint_ref(
        session_id="session-transcript",
        run_id="run-transcript",
        task_id=None,
        focus_task_id=None,
        compatibility_task_id=record.task_id,
        segment_id="user-1",
        checkpoint=checkpoint,
    )
    _record_ledger_checkpoint(context.data_root, checkpoint)

    registry = create_default_registry()
    result = registry.dispatch("/resume session-transcript", context)

    assert result is not None
    message = result.message or ""
    assert "Resume ready from session." in message
    assert "Conversation (session-transcript):" in message
    assert "1. [user]" in message
    assert "first question" in message
    assert "2. [assistant]" in message
    assert "first answer" in message


def _record_ledger_checkpoint(data_root: Path, checkpoint: Checkpoint) -> None:
    reason = checkpoint.reason or checkpoint.state.lower()
    LedgerWriter(
        LedgerStore(data_root),
        source="tests.slash_commands",
    ).record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )


def test_task_clear_unfocuses_without_dropping_compat_task(
    context: SlashCommandContext,
) -> None:
    record = context.store.create_task("focused")
    context.repl_state.current_task_id = record.task_id
    context.repl_state.compatibility_task_id = "chat-compat"

    registry = create_default_registry()
    result = registry.dispatch("/task clear", context)

    assert result is not None
    assert context.repl_state.current_task_id is None
    assert context.repl_state.compatibility_task_id == "chat-compat"
    assert "Focus task cleared" in (result.message or "")


def test_pause_without_task(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/pause", context)
    assert result is not None
    assert "No active task" in (result.message or "")


def test_pause_with_task(context: SlashCommandContext) -> None:
    record = context.store.create_task("running")
    context.repl_state.current_task_id = record.task_id
    registry = create_default_registry()
    result = registry.dispatch("/pause", context)
    assert result is not None
    assert result.pause_requested is True
    assert "Pause requested" in (result.message or "")


def test_compact_without_history(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/compact", context)
    assert result is not None
    assert "会话为空" in (result.message or "")


def test_compact_queues_semantic_summary_without_overwriting_task_layers(
    context: SlashCommandContext,
) -> None:
    """命令进入实际运行后才生成摘要，任务材料不被摘录覆盖；传参：会话上下文；返回：无。"""

    record = context.store.create_task("conv heavy")
    context.repl_state.current_task_id = record.task_id
    session_id = context.repl_state.session_id
    append_user_message(context.data_root, session_id, "first turn")
    append_assistant_message(context.data_root, session_id, "answer 1")
    before = context.store.read_summary_layers(record.task_id)
    context.llm_client = from_test_stub("尚未调用")
    registry = create_default_registry()
    result = registry.dispatch("/compact", context)
    assert result is not None
    assert result.session_input == "/compact"
    layers = context.store.read_summary_layers(record.task_id)
    assert layers == before


def test_dashboard_signals_enter(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/dashboard", context)
    assert result is not None
    assert result.enter_dashboard is True


def test_empty_slash(context: SlashCommandContext) -> None:
    registry = create_default_registry()
    result = registry.dispatch("/", context)
    assert result is not None
    assert "/help" in (result.message or "")


def test_toolsets_enable_persists_session_policy(
    context: SlashCommandContext,
) -> None:
    registry = create_default_registry()

    result = registry.dispatch("/toolsets enable web,file", context)

    assert result is not None
    assert "enabled : web, file" in (result.message or "")
    from runtime.session_state import SessionStateStore

    state = SessionStateStore(context.data_root).load(context.repl_state.session_id)
    assert state is not None
    assert state.toolsets_enabled == ["web", "file"]
    assert context.repl_state.toolsets_enabled == ["web", "file"]


def test_toolsets_rejects_unknown_without_saving(
    context: SlashCommandContext,
) -> None:
    registry = create_default_registry()

    result = registry.dispatch("/toolsets enable missing", context)

    assert result is not None
    assert "Unknown toolset" in (result.message or "")
    from runtime.session_state import SessionStateStore

    assert (
        SessionStateStore(context.data_root).load(context.repl_state.session_id) is None
    )


def test_toolsets_reset_clears_policy(context: SlashCommandContext) -> None:
    registry = create_default_registry()

    registry.dispatch("/toolsets enable web", context)
    result = registry.dispatch("/toolsets reset", context)

    assert result is not None
    from runtime.session_state import SessionStateStore

    state = SessionStateStore(context.data_root).load(context.repl_state.session_id)
    assert state is not None
    assert state.toolsets_enabled is None
    assert state.toolsets_disabled is None


def test_toolsets_show_describes_the_real_default_not_inference(
    context: SlashCommandContext,
) -> None:
    # 默认态早已不是"按任务文本推断"，文案还那么写就是假话。同时把拿 opt-in 工具的
    # 正确姿势写出来——enable 的语义是收窄不是追加，敲 enable secret 会只剩两个工具
    registry = create_default_registry()

    shown = registry.dispatch("/toolsets show", context)
    reset = registry.dispatch("/toolsets reset", context)

    assert shown is not None and reset is not None
    assert "inference" not in (shown.message or "")
    assert "inference" not in (reset.message or "")
    assert "all tools except the opt-in group" in (shown.message or "")
    assert "/toolsets enable full" in (shown.message or "")


def test_register_disallows_duplicate_names() -> None:
    registry = SlashCommandRegistry()
    from app.repl.slash_commands import SlashCommand

    cmd = SlashCommand("foo", "first", lambda _a, _c: SlashCommandResult())
    registry.register(cmd)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(cmd)


def test_register_disallows_alias_collision() -> None:
    registry = SlashCommandRegistry()
    from app.repl.slash_commands import SlashCommand

    registry.register(
        SlashCommand("alpha", "x", lambda _a, _c: SlashCommandResult(), aliases=("a",))
    )
    with pytest.raises(ValueError, match="alias collision"):
        registry.register(
            SlashCommand(
                "beta", "y", lambda _a, _c: SlashCommandResult(), aliases=("a",)
            )
        )


def test_format_session_banner_contains_workspace_basics(tmp_path: Path) -> None:
    banner = format_session_banner(
        project_root=tmp_path / "project",
        data_root=tmp_path / "project" / ".reins" / "data",
        model="claude-sonnet-4",
        provider="anthropic",
    )
    assert "Reins" in banner
    assert "claude-sonnet-4" in banner
    assert "anthropic" in banner
    assert "/help" in banner


def test_format_prompt_label_uses_state() -> None:
    label = format_prompt_label(
        ReplState(current_task_id="2026-05-06-01HK", trace_on=True)
    )
    assert "2026-05-06-01HK" in label
    assert "trace" in label

    label_empty = format_prompt_label(ReplState())
    assert "session-" in label_empty
