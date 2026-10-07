from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from frontends.shared.session_search import (
    QueryWarning,
    SearchResult,
    SessionSearchService,
)
from frontends.shared.run_observation import build_watchdog_observation
from frontends.tui.data.run_detail import RunDetailReader, RunDetailView
from memory.store import MEMORY_STATE_ACTIVE, MemoryStore
from runtime.checkpoint import RecoveryPointSummary, list_recent_checkpoints
from runtime.lease import LEASE_SEGMENT_END, Lease, from_trigger, load_snapshot
from runtime.run_facts import RunFactStore, RunSummary
from runtime.persistence import RuntimeStore
from runtime.session_state import SessionState, SessionStateStore
from skills.store import SkillStore
from tasks.store import TaskStore
from tools.todo_tool import TODO_IN_PROGRESS, TODO_PENDING, list_todos


@dataclass(frozen=True, slots=True)
class TaskListItem:
    task_id: str
    created_at: str
    updated_at: str
    status: str
    goal: str


@dataclass(frozen=True, slots=True)
class WatchdogView:
    llm_failures: int
    tool_failures: int
    unique_failure_patterns: int
    paused: bool


@dataclass(frozen=True, slots=True)
class TaskDetail:
    task_id: str
    todo: list[str]
    lease: Lease
    watchdog: WatchdogView
    trajectory_tail: list[str]


@dataclass(frozen=True, slots=True)
class TaskPage:
    items: list[TaskListItem]
    page: int
    page_size: int
    has_more: bool
    warnings: list[QueryWarning]


@dataclass(frozen=True, slots=True)
class SessionOverview:
    sessions: list[SessionState]
    recent_runs: list[RunSummary]
    recovery_points: list[RecoveryPointSummary]


class TaskQueryService:
    def __init__(self, data_root: Path | str | None = None) -> None:
        self._data_root = _resolve_data_root(data_root)
        self._task_store = TaskStore(self._data_root)
        self._memory_store = MemoryStore(self._data_root)
        self._skill_store = SkillStore(self._data_root)
        self._run_facts = RunFactStore(self._data_root)
        self._run_detail = RunDetailReader(self._data_root)
        self._session_states = SessionStateStore(self._data_root)

    @property
    def data_root(self) -> Path:
        return self._data_root

    def list_tasks_page(self, *, page: int, page_size: int = 30) -> TaskPage:
        """按派生目录分页定位目标原件；传参：页号和容量；返回：完整目标页，索引失败明确抛错。"""
        safe_page = max(1, page)
        safe_size = max(1, page_size)
        offset = (safe_page - 1) * safe_size
        store = RuntimeStore(self._data_root)
        # 1. 【任务目录】【分页】固定索引代次和身份范围，后台写入不改变本页截止位置
        with store.index_connection() as connection:
            connection.execute("BEGIN")
            sequence = int(
                connection.execute(
                    "SELECT value FROM runtime_meta WHERE key='sequence'"
                ).fetchone()[0]
            )
            identities = connection.execute(
                """
                SELECT record_id FROM records WHERE kind='task' AND json_extract(payload,'$.is_inbox')=0
                ORDER BY json_extract(payload,'$.created_at') DESC,record_id DESC LIMIT ? OFFSET ?
            """,
                (safe_size + 1, offset),
            ).fetchall()
        # 2. 【任务目录】【读取原件】仅展开本页正文，丢失或损坏的已保存目标不能冒充空列表
        rows = []
        with store.snapshot(sequence) as source:
            for identity in identities[:safe_size]:
                task = source.get("task", identity[0])
                if task is None:
                    raise ValueError(f"indexed task source is missing: {identity[0]}")
                rows.append(
                    TaskListItem(
                        task["task_id"],
                        task["created_at"],
                        task["updated_at"],
                        task["status"],
                        task["goal"],
                    )
                )
        return TaskPage(
            items=rows,
            page=safe_page,
            page_size=safe_size,
            has_more=len(identities) > safe_size,
            warnings=[],
        )

    def list_active_memories(self, *, limit: int = 100) -> list[str]:
        items = self._memory_store.list_memories(state=MEMORY_STATE_ACTIVE)
        return [
            f"{item.memory_id} | {item.type} | {item.content[:72]}"
            for item in items[:limit]
        ]

    def list_active_skills(self, *, limit: int = 100) -> list[str]:
        skills = self._skill_store.list_skills(active_only=True)
        return [
            f"{item.skill_id} | {item.frontmatter.name} | {item.frontmatter.role}"
            for item in skills[:limit]
        ]

    def read_session_overview(self, *, limit: int = 5) -> SessionOverview:
        safe_limit = max(1, limit)
        return SessionOverview(
            sessions=self._session_states.list_recent(limit=safe_limit),
            recent_runs=self._run_facts.list_recent_runs(limit=safe_limit),
            recovery_points=list_recent_checkpoints(
                data_root=self._data_root, limit=safe_limit
            ),
        )

    def render_session_overview_lines(self, *, limit: int = 5) -> list[str]:
        overview = self.read_session_overview(limit=limit)
        lines: list[str] = ["Recent sessions"]
        if not overview.sessions:
            lines.append("(none)")
        for session in overview.sessions:
            status = (
                f", status={session.last_run_status}" if session.last_run_status else ""
            )
            focus = f", focus={session.focus_task_id}" if session.focus_task_id else ""
            lines.append(
                f"{session.session_id} | run={session.last_run_id or '(none)'}"
                f"{status}{focus}"
            )
            if session.summary:
                lines.append(f"summary: {session.summary[:120]}")
        lines.append("Recent runs")
        if not overview.recent_runs:
            lines.append("(none)")
        for run in overview.recent_runs:
            status = f", status={run.status}" if run.status else ""
            focus = f", focus={run.focus_task_id}" if run.focus_task_id else ""
            lines.append(f"{run.run_id} | session={run.session_id}{status}{focus}")
        lines.append("Recovery points")
        if not overview.recovery_points:
            lines.append("(none)")
        for item in overview.recovery_points:
            lines.append(
                f"{item.checkpoint_id} | state={item.state} | "
                f"session={item.session_id or '(none)'} | run={item.run_id or '(none)'}"
            )
        return lines

    def read_session_overview_filtered(
        self, query: str, *, limit: int = 10
    ) -> SearchResult:
        """Search sessions by keyword or ID.

        Empty query returns an empty SearchResult (caller should fall back
        to `read_session_overview`).
        """
        service = SessionSearchService(self._data_root)
        return service.search(query, limit=limit)

    def render_session_overview_lines_filtered(
        self, query: str, *, limit: int = 10
    ) -> list[str]:
        """Render session search results as lines for the sidebar.

        Empty query returns the default overview lines.
        """
        if not query.strip():
            return self.render_session_overview_lines()
        result = self.read_session_overview_filtered(query, limit=limit)
        lines: list[str] = ["Recent sessions"]
        for warning in result.warnings:
            lines.append(f"提示: {warning.message}")
        if not result.hits:
            lines.append("未找到匹配会话")
        for hit in result.hits:
            status = f", status={hit.status}" if hit.status else ""
            focus = f", focus={hit.focus_task_id}" if hit.focus_task_id else ""
            lines.append(
                f"{hit.session_id} | run={hit.last_run_id or '(none)'}{status}{focus}"
            )
            if hit.summary_snippet:
                lines.append(f"  摘要: {hit.summary_snippet}")
        # Keep "Recent runs" and "Recovery points" blocks from default overview.
        overview = self.read_session_overview(limit=5)
        lines.append("Recent runs")
        if not overview.recent_runs:
            lines.append("(none)")
        for run in overview.recent_runs:
            status = f", status={run.status}" if run.status else ""
            focus = f", focus={run.focus_task_id}" if run.focus_task_id else ""
            lines.append(f"{run.run_id} | session={run.session_id}{status}{focus}")
        lines.append("Recovery points")
        if not overview.recovery_points:
            lines.append("(none)")
        for item in overview.recovery_points:
            lines.append(
                f"{item.checkpoint_id} | state={item.state} | "
                f"session={item.session_id or '(none)'} | run={item.run_id or '(none)'}"
            )
        return lines

    def list_session_runs(
        self, session_id: str, *, limit: int = 20
    ) -> list[RunSummary]:
        return self._run_detail.list_session_runs(session_id, limit=limit)

    def read_run_detail(
        self,
        session_id: str,
        run_id: str,
        *,
        expand_raw: bool = False,
    ) -> RunDetailView:
        return self._run_detail.read_run_detail(
            session_id,
            run_id,
            expand_raw=expand_raw,
        )

    def read_task_detail(
        self,
        task_id: str,
        *,
        trajectory_limit: int = 10,
    ) -> TaskDetail:
        """从原任务待办和运行事实读取详情；传参：目标和轨迹数量；返回：当前可追溯视图。"""
        self._task_store.require_task(task_id)
        todo = [
            f"- [ ] {item.content}"
            if item.status == TODO_PENDING
            else f"in_progress: {item.content}"
            for item in list_todos(task_id, data_root=self._data_root)
            if item.status in {TODO_PENDING, TODO_IN_PROGRESS}
        ]
        facts = self._run_facts.read_task_facts(task_id)
        lease = _latest_lease(facts, task_id)
        watchdog = _watchdog_view(facts)
        trajectory_tail = _render_trajectory_tail(facts, trajectory_limit)
        return TaskDetail(
            task_id=task_id,
            todo=todo,
            lease=lease,
            watchdog=watchdog,
            trajectory_tail=trajectory_tail,
        )


def _resolve_data_root(data_root: Path | str | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    return Path.home() / ".reins" / "data"


def _latest_lease(rows: list[dict[str, object]], task_id: str) -> Lease:
    for row in reversed(rows):
        if row.get("event") == "run:start":
            payload = row.get("lease_summary")
            if isinstance(payload, dict):
                return load_snapshot(payload)
    return from_trigger("user", task_id=task_id)


def _watchdog_view(rows: list[dict[str, object]]) -> WatchdogView:
    observed = build_watchdog_observation(rows)
    return WatchdogView(
        llm_failures=observed.llm_failures,
        tool_failures=observed.tool_failures,
        unique_failure_patterns=observed.unique_failure_patterns,
        paused=observed.paused,
    )


def _render_trajectory_tail(rows: list[dict[str, object]], limit: int) -> list[str]:
    out: list[str] = []
    for row in rows[-limit:]:
        event = str(row.get("event", ""))
        if event == "tool:response":
            tool_info = row.get("tool")
            tool_name = (
                str(tool_info.get("name", "tool"))
                if isinstance(tool_info, dict)
                else "tool"
            )
            status = (
                str(tool_info.get("status", "")) if isinstance(tool_info, dict) else ""
            )
            if status != "ok":
                out.append(f"tool_failure | {tool_name}")
            else:
                out.append(f"tool:response | {tool_name}")
        elif event == "llm:response":
            summary = row.get("summary")
            if isinstance(summary, dict) and summary.get("error"):
                out.append("llm_failure")
            else:
                out.append("llm:response")
        elif event == "run:start":
            lease_summary = row.get("lease_summary")
            trigger = "unknown"
            if isinstance(lease_summary, dict):
                trigger = str(lease_summary.get("trigger", "unknown"))
            else:
                trigger = str(row.get("trigger", "unknown"))
            out.append(f"lease_snapshot | trigger={trigger}")
        elif event == "run:lifecycle":
            lifecycle = str(row.get("lifecycle", ""))
            reason = str(row.get("reason", ""))
            out.append(f"lifecycle | {lifecycle} | {reason}")
        elif event == "state:transition":
            close_reason = row.get("close_reason")
            if close_reason:
                out.append(f"legacy_segment_close | {close_reason}")
            else:
                from_s = str(row.get("from_state", ""))
                to_s = str(row.get("to_state", ""))
                out.append(f"legacy_state:transition | {from_s}->{to_s}")
        elif event == "checkpoint:saved":
            ckpt = row.get("checkpoint")
            reason = str(ckpt.get("reason", "")) if isinstance(ckpt, dict) else ""
            out.append(f"checkpoint | {reason}" if reason else "checkpoint")
        else:
            detail = event if event else str(row.get("type", "event"))
            out.append(detail)
    return out


def render_lease_lines(lease: Lease) -> list[str]:
    expires = lease.expires_at
    if expires == LEASE_SEGMENT_END:
        expires = "segment_end"
    return [
        f"trigger: {lease.trigger}",
        f"expires_at: {expires}",
        f"max_steps: {lease.max_steps}",
        f"max_tokens: {lease.max_tokens}",
        f"capabilities: {json.dumps(lease.capabilities, ensure_ascii=False)}",
    ]
