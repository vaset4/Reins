from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from runtime.checkpoint import RecoveryPointSummary
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.types import RunContext, TerminalFocusPolicy
from runtime.persistence import RuntimeStore
from tasks.ids import utc_now


@dataclass(frozen=True, slots=True)
class WritebackTargets:
    session: bool = True
    run_facts: bool = True
    task: bool = False


@dataclass(slots=True)
class SessionState:
    session_id: str
    updated_at: str = ""
    last_run_id: str = ""
    last_run_status: str = ""
    last_run_event: str = ""
    last_checkpoint_id: str = ""
    last_checkpoint_state: str = ""
    last_checkpoint_reason: str = ""
    last_checkpoint_at: str = ""
    focus_task_id: str | None = None
    compatibility_task_id: str | None = None
    summary: str = ""
    recent_run_ids: list[str] = field(default_factory=list)
    writeback_targets: dict[str, bool] = field(default_factory=dict)
    consecutive_readonly_count: int = 0
    original_user_goal: str = ""
    hint_injection_count: int = 0
    toolsets_enabled: list[str] | None = None
    toolsets_disabled: list[str] | None = None
    toolsets_updated_at: str = ""
    stable_prompt_hash: str = ""
    stable_prompt_updated_at: str = ""


class SessionStateStore:
    def __init__(self, data_root: Path | str) -> None:
        """初始化会话投影；参数：数据根；返回：无。"""
        self._data_root = Path(data_root)
        self.database = RuntimeStore(data_root)

    def load(self, session_id: str) -> SessionState | None:
        """读取会话当前状态；参数：会话编号；返回：状态或尚不存在。"""
        if not session_id:
            return None
        with self.database.snapshot() as source:
            raw = source.get("session_state", session_id)
        if raw is None:
            return None
        if not isinstance(raw, dict) or raw.get("session_id") != session_id:
            raise ValueError("invalid session state identity")
        return _state_from_mapping(raw, session_id=session_id)

    def save(self, state: SessionState) -> Path:
        """原子保存状态，摘要仅有一份正文；参数：完整状态；返回：会话原件路径。"""
        if not state.session_id:
            raise ValueError("session state requires identity")
        payload = asdict(state)
        payload.pop("consecutive_readonly_count", None)
        payload.pop("hint_injection_count", None)
        with self.database.transaction() as batch:
            batch.put(
                "session_state", state.session_id, payload, session_id=state.session_id
            )
        return self.database.source_path("session_state", state.session_id)

    def list_recent(self, *, limit: int | None = None) -> list[SessionState]:
        """查询最近会话状态；参数：可选条数；返回：按最近时间倒序的状态。"""
        with self.database.snapshot() as source:
            rows = sorted(
                source.list("session_state"),
                key=lambda row: (row["updated_at"], row["session_id"]),
                reverse=True,
            )
        return [
            _state_from_mapping(row, session_id=row["session_id"])
            for row in rows[:limit]
        ]

    def record_checkpoint(
        self,
        checkpoint: RecoveryPointSummary,
        *,
        summary: str = "",
    ) -> SessionState | None:
        with self.database.transaction():
            if not checkpoint.session_id:
                return None
            state = self.load(checkpoint.session_id) or SessionState(
                session_id=checkpoint.session_id
            )
            state.last_run_id = checkpoint.run_id or state.last_run_id
            state.last_checkpoint_id = checkpoint.checkpoint_id
            state.last_checkpoint_state = checkpoint.state
            state.last_checkpoint_reason = checkpoint.reason
            state.last_checkpoint_at = checkpoint.saved_at
            state.focus_task_id = checkpoint.focus_task_id
            state.compatibility_task_id = checkpoint.compatibility_task_id
            state.summary = _compact_summary(summary or state.summary)
            state.updated_at = checkpoint.saved_at or utc_now()
            _remember_run(state, checkpoint.run_id)
            self.save(state)
            return state

    def record_run_terminal(
        self,
        context: RunContext,
        status: str,
        *,
        last_event: str = "state:transition",
        summary: str = "",
        preserve_summary: bool = False,
    ) -> SessionState:
        with self.database.transaction():
            state = self.load(context.session_id) or SessionState(
                session_id=context.session_id
            )
            targets = select_writeback_targets(context)
            state.last_run_id = context.run_id
            state.last_run_status = status
            state.last_run_event = last_event
            state.focus_task_id = (
                context.focus_task_id
                if context.terminal_focus_policy is TerminalFocusPolicy.PRESERVE
                else None
            )
            state.compatibility_task_id = context.compatibility_task_id
            if preserve_summary and not summary:
                state.summary = _compact_summary(state.summary)
            else:
                state.summary = _compact_summary(
                    summary or _default_summary(context=context, status=status)
                )
            state.writeback_targets = asdict(targets)
            state.updated_at = utc_now()
            _remember_run(state, context.run_id)
            self.save(state)
            return state

    def update_focus(
        self,
        session_id: str,
        *,
        focus_task_id: str | None,
        previous_focus_task_id: str | None = None,
        compatibility_task_id: str | None = None,
        summary: str = "",
    ) -> SessionState | None:
        with self.database.transaction():
            if not session_id:
                return None
            state = self.load(session_id) or SessionState(session_id=session_id)
            previous_focus = (
                previous_focus_task_id
                if previous_focus_task_id is not None
                else state.focus_task_id
            )
            state.focus_task_id = focus_task_id
            if compatibility_task_id is not None:
                state.compatibility_task_id = compatibility_task_id
            if summary:
                state.summary = _compact_summary(summary)
            state.updated_at = utc_now()
            self.save(state)
            LedgerWriter(
                LedgerStore(self._data_root), source="runtime.session_state"
            ).record_task_focus_changed(
                previous_focus,
                focus_task_id,
                summary or "focus updated",
                session_id=session_id,
            )
            return state


def select_writeback_targets(context: RunContext) -> WritebackTargets:
    return WritebackTargets(task=context.has_formal_task)


def _state_from_mapping(raw: Mapping[str, object], *, session_id: str) -> SessionState:
    recent_run_ids = _str_list(raw.get("recent_run_ids"))
    raw_targets = raw.get("writeback_targets", {})
    target_items = raw_targets.items() if isinstance(raw_targets, dict) else ()
    retired_targets = {"compatibility_task", "memory", "skill", "artifact"}
    writeback_targets = {
        str(key): bool(value)
        for key, value in target_items
        if key not in retired_targets
    }
    return SessionState(
        session_id=str(raw.get("session_id", session_id)),
        updated_at=str(raw.get("updated_at", "")),
        last_run_id=str(raw.get("last_run_id", "")),
        last_run_status=str(raw.get("last_run_status", "")),
        last_run_event=str(raw.get("last_run_event", "")),
        last_checkpoint_id=str(raw.get("last_checkpoint_id", "")),
        last_checkpoint_state=str(raw.get("last_checkpoint_state", "")),
        last_checkpoint_reason=str(raw.get("last_checkpoint_reason", "")),
        last_checkpoint_at=str(raw.get("last_checkpoint_at", "")),
        focus_task_id=_optional_str(raw.get("focus_task_id")),
        compatibility_task_id=_optional_str(raw.get("compatibility_task_id")),
        summary=str(raw.get("summary", "")),
        recent_run_ids=recent_run_ids,
        writeback_targets=writeback_targets,
        consecutive_readonly_count=_safe_int(raw.get("consecutive_readonly_count")),
        original_user_goal=str(raw.get("original_user_goal", "")),
        hint_injection_count=_safe_int(raw.get("hint_injection_count")),
        toolsets_enabled=_optional_str_list(raw.get("toolsets_enabled")),
        toolsets_disabled=_optional_str_list(raw.get("toolsets_disabled")),
        toolsets_updated_at=str(raw.get("toolsets_updated_at", "")),
        stable_prompt_hash=str(raw.get("stable_prompt_hash", "")),
        stable_prompt_updated_at=str(raw.get("stable_prompt_updated_at", "")),
    )


def _remember_run(state: SessionState, run_id: str) -> None:
    if not run_id:
        return
    state.recent_run_ids = [
        run_id,
        *[item for item in state.recent_run_ids if item != run_id],
    ][:10]


def _default_summary(*, context: RunContext, status: str) -> str:
    focus = context.focus_task_id or "none"
    compat = context.compatibility_task_id or "none"
    return (
        f"last run {context.run_id} ended as {status}; "
        f"focus_task={focus}; compatibility_task={compat}"
    )


def _compact_summary(value: str, *, limit: int = 500) -> str:
    text = " ".join(value.strip().split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _safe_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _str_list(value: object) -> list[str]:
    return (
        [str(item) for item in value if isinstance(item, str) and item]
        if isinstance(value, list)
        else []
    )


def _optional_str_list(value: object) -> list[str] | None:
    return [str(item) for item in value] if isinstance(value, list) else None


def session_state_payload(state: SessionState | None) -> dict[str, Any]:
    if state is None:
        return {}
    return {
        "session_id": state.session_id,
        "last_run_id": state.last_run_id,
        "last_run_status": state.last_run_status,
        "last_checkpoint_id": state.last_checkpoint_id,
        "last_checkpoint_state": state.last_checkpoint_state,
        "focus_task_id": state.focus_task_id,
        "compatibility_task_id": state.compatibility_task_id,
        "summary": state.summary,
    }
