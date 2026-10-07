"""Dashboard view for the Reins REPL.

A one-shot snapshot of the eight V2.1 subsystems the user is meant to "see at
a glance" — task inventory, current goal, lease, watchdog, MCP servers, and
recent trajectory. Triggered by `/dashboard`. Does not auto-refresh; the user
returns to the chat stream by typing a prompt or `/dashboard` again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.repl.console import get_console
from app.repl.slash_commands import ReplState
from runtime.lease import Lease
from runtime.run_facts import RunFactStore
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry
from tools.todo_tool import list_todos


def render_dashboard(
    *,
    state: ReplState,
    store: TaskStore,
    registry: ToolRegistry,
    project_root: Path,
    data_root: Path,
    lease: Lease | None = None,
) -> None:
    console = get_console()
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=1),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="middle"),
        Layout(name="right"),
    )

    layout["header"].update(
        Panel(
            _header_text(state, store),
            border_style="blue",
            title="Reins Dashboard",
        )
    )
    layout["left"].update(_active_tasks_panel(state, store))
    layout["middle"].update(_middle_panel(state, store, data_root))
    layout["right"].update(
        _right_panel(state, store, registry, project_root, data_root, lease)
    )
    layout["footer"].update(
        Text("[type any prompt to return to chat — /dashboard refreshes]", style="dim")
    )
    console.print(layout)


def _header_text(state: ReplState, store: TaskStore) -> Text:
    task_id = state.current_task_id or "(no task)"
    record = store.load_task(state.current_task_id) if state.current_task_id else None
    status = record.status if record else "-"
    goal = record.goal if record else "-"
    trace = "on" if state.trace_on else "off"
    return Text.assemble(
        (f"task: {task_id}  ", "bold cyan"),
        (f"status: {status}  ", "green"),
        (f"trace: {trace}\n", "dim"),
        (f"goal: {goal}", ""),
    )


def _active_tasks_panel(state: ReplState, store: TaskStore) -> Panel:
    records = store.list_tasks(limit=10)
    if not records:
        return Panel(Text("(no tasks yet)", style="dim"), title="Active Tasks")
    table = Table.grid(padding=(0, 1))
    table.add_column("id", style="cyan")
    table.add_column("status")
    table.add_column("goal")
    for record in records:
        marker = "▶" if record.task_id == state.current_task_id else " "
        table.add_row(
            f"{marker} {record.task_id[-12:]}",
            record.status,
            _truncate(record.goal, 30),
        )
    return Panel(table, title=f"Active Tasks [{len(records)}]")


def _middle_panel(state: ReplState, store: TaskStore, data_root: Path) -> Panel:
    if state.current_task_id is None:
        return Panel(
            Text("Use /task new <goal> to start a task.", style="dim"),
            title="Current Task",
        )
    task_id = state.current_task_id
    todo = _read_todo(data_root, task_id)
    summary = store.read_summary(task_id)
    trajectory_tail = _read_trajectory_tail(data_root, task_id, limit=5)

    sections: list[Any] = []
    sections.append(Text("TODO", style="bold"))
    sections.append(todo or Text("(no saved plan yet)", style="dim"))
    sections.append(Text("\nSUMMARY", style="bold"))
    sections.append(Text(_truncate(summary or "(empty)", 240), style="dim"))
    sections.append(Text("\nRECENT TRAJECTORY", style="bold"))
    if trajectory_tail:
        traj_table = Table.grid(padding=(0, 1))
        traj_table.add_column("event", style="magenta")
        traj_table.add_column("detail")
        for row in trajectory_tail:
            traj_table.add_row(
                str(row.get("event", row.get("type", "?"))),
                _truncate(json.dumps(row, ensure_ascii=False), 60),
            )
        sections.append(traj_table)
    else:
        sections.append(Text("(no trajectory yet)", style="dim"))

    body = Text()
    for section in sections:
        if isinstance(section, Text):
            body.append(section)
            body.append("\n")
    # Mix in non-text sections as separate console renderables would need a
    # full Group; for simplicity we render only text when available, and fall
    # back to a Group when a Table appears.
    from rich.console import Group

    return Panel(Group(*sections), title="Current Task")


def _right_panel(
    state: ReplState,
    store: TaskStore,
    registry: ToolRegistry,
    project_root: Path,
    data_root: Path,
    lease: Lease | None = None,
) -> Panel:
    lines: list[str] = []
    lines.append(
        "LEASE  (current user lease)" if lease is not None else "LEASE  (unknown)"
    )
    lines.append(f"  fs.project: {_truncate(str(project_root), 40)}")
    lines.append(f"  fs.data:    {_truncate(str(data_root), 40)}")
    lines.extend(_capability_lines(lease))
    lines.append("")
    lines.extend(_watchdog_lines(data_root, state.current_task_id))
    lines.append("")
    lines.extend(_mcp_lines(registry))
    return Panel("\n".join(lines), title="Lease / Watchdog / MCP")


def _capability_lines(lease: Lease | None) -> list[str]:
    if lease is None:
        return [
            "  terminal:   unknown",
            "  browser:    unknown",
            "  network:    unknown",
            "  mcp:        unknown",
        ]
    capabilities = lease.capabilities
    return [
        _terminal_line(capabilities.get("terminal")),
        _browser_line(capabilities.get("browser")),
        _enabled_line("network", capabilities.get("network")),
        _mcp_line(capabilities.get("mcp")),
    ]


def _terminal_line(value: object) -> str:
    if not isinstance(value, dict):
        return "  terminal:   unknown"
    status = _enabled_status(value)
    allow = value.get("allow_commands")
    allow_text = (
        json.dumps(allow, ensure_ascii=False) if isinstance(allow, list) else "unknown"
    )
    return f"  terminal:   {status} allow={allow_text}"


def _browser_line(value: object) -> str:
    if not isinstance(value, dict):
        return "  browser:    unknown"
    status = _enabled_status(value)
    headless = value.get("headless")
    mode = (
        "headless"
        if headless is True
        else "headed"
        if headless is False
        else "headless=unknown"
    )
    return f"  browser:    {status} {mode}"


def _mcp_line(value: object) -> str:
    if not isinstance(value, dict):
        return "  mcp:        unknown"
    status = _enabled_status(value)
    servers = value.get("allow_servers")
    servers_text = (
        json.dumps(servers, ensure_ascii=False)
        if isinstance(servers, list)
        else "unknown"
    )
    return f"  mcp:        {status} allow_servers={servers_text}"


def _enabled_line(name: str, value: object) -> str:
    return f"  {name}:    {_enabled_status(value)}"


def _enabled_status(value: object) -> str:
    if not isinstance(value, dict) or "enabled" not in value:
        return "unknown"
    return "enabled" if value.get("enabled") is True else "disabled"


def _watchdog_lines(data_root: Path, task_id: str | None) -> list[str]:
    if task_id is None:
        return ["WATCHDOG", "  idle (no task)"]
    rows = _read_trajectory_tail(data_root, task_id, limit=200)
    tool_failures = sum(
        1
        for row in rows
        if row.get("event") == "tool:response" and row.get("status") != "ok"
    )
    llm_failures = sum(
        1
        for row in rows
        if row.get("event") == "llm:response" and isinstance(row.get("error"), dict)
    )
    return [
        "WATCHDOG",
        f"  tool failures: {tool_failures}",
        f"  llm failures:  {llm_failures}",
    ]


def _mcp_lines(registry: ToolRegistry) -> list[str]:
    mcp_tools = [
        d
        for d in registry.list_definitions(model_visible_only=False)
        if d.name.startswith("mcp_")
    ]
    if not mcp_tools:
        return ["MCP SERVERS", "  (no MCP tools registered)"]
    server_set = {d.name.split("_", maxsplit=2)[1] for d in mcp_tools}
    return ["MCP SERVERS"] + [f"  {name}" for name in sorted(server_set)]


def _read_todo(data_root: Path, task_id: str) -> Text | None:
    """读取模型实际维护的计划；参数：数据根与目标；返回：待办文本或无计划。"""
    items = list_todos(task_id, data_root=data_root)
    if not items:
        return None
    text = "\n".join(
        f"{item.idx + 1}. [{item.status}] {item.content}" for item in items
    )
    return Text(_truncate(text, 320), style="dim")


def _read_trajectory_tail(
    data_root: Path, task_id: str | None, *, limit: int = 5
) -> list[dict[str, Any]]:
    if task_id is None:
        return []
    facts = RunFactStore(data_root).read_task_facts(task_id)
    return [dict(fact) for fact in facts[-limit:]]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


__all__ = ["render_dashboard"]
