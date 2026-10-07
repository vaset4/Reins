from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from runtime.lease import Lease


@dataclass(slots=True)
class RunToolsRequest:
    action: str
    payload: str = ""
    tool_name: str = ""
    arguments: dict[str, object] = field(default_factory=dict)
    target_scope: str | None = None
    call_id: str = ""
    validation_error: str | None = None


@dataclass(slots=True)
class RunToolsResult:
    action: str
    output: str
    status: str = "ok"
    tool_name: str = ""
    content: str | None = None
    error: str | None = None
    summary: str = ""
    target_scope: str | None = None
    meta: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)
    model_view: str | None = None
    annotations: tuple[str, ...] = ()

    @classmethod
    def ok(
        cls,
        *,
        action: str,
        content: str,
        tool_name: str = "",
        summary: str = "",
        target_scope: str | None = None,
        meta: dict[str, object] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> "RunToolsResult":
        return cls(
            action=action,
            output=content,
            status="ok",
            tool_name=tool_name or action,
            content=content,
            summary=summary,
            target_scope=target_scope,
            meta=dict(meta or {}),
            diagnostics=dict(diagnostics or {}),
        )

    @classmethod
    def error_result(
        cls,
        *,
        action: str,
        error: str,
        tool_name: str = "",
        summary: str = "",
        target_scope: str | None = None,
        meta: dict[str, object] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> "RunToolsResult":
        return cls(
            action=action,
            output=error,
            status="error",
            tool_name=tool_name or action,
            error=error,
            summary=summary,
            target_scope=target_scope,
            meta=dict(meta or {}),
            diagnostics=dict(diagnostics or {}),
        )

    @classmethod
    def denied(
        cls,
        *,
        action: str,
        error: str,
        tool_name: str = "",
        summary: str = "",
        target_scope: str | None = None,
        meta: dict[str, object] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> "RunToolsResult":
        return cls(
            action=action,
            output=error,
            status="denied",
            error=error,
            tool_name=tool_name or action,
            summary=summary,
            target_scope=target_scope,
            meta=dict(meta or {}),
            diagnostics=dict(diagnostics or {}),
        )


@dataclass(slots=True)
class ReadOnlyInspectionRequest:
    action: str
    target_path: str
    query: str | None = None
    offset: int = 0
    cursor: str | None = None
    limit: int | None = None
    start_line: int | None = None
    line_count: int | None = None
    path_kind: str = "file"


@dataclass(slots=True)
class ReadOnlyInspectionResult:
    action: str
    status: str
    output: str
    error: str | None = None
    meta: dict[str, object] = field(default_factory=dict)


class Trigger(str, Enum):
    USER = "user"
    CRON = "cron"
    RESUME = "resume"
    DELEGATE = "delegate"
    IDLE = "idle"


class TerminalFocusPolicy(str, Enum):
    """控制 run 终态是否保留当前 session focus。

    作者：xxx
    时间：2026-08-17 00:00:00
    """

    CLEAR = "clear"
    PRESERVE = "preserve"


@dataclass(slots=True, kw_only=True)
class RunContext:
    trigger: Trigger
    payload: dict[str, object]
    capability_lease: Lease
    session_id: str = ""
    run_id: str = ""
    task_id: str | None = None
    focus_task_id: str | None = None
    focus_task: dict[str, object] = field(default_factory=dict)
    terminal_focus_policy: TerminalFocusPolicy = TerminalFocusPolicy.PRESERVE
    compatibility_task_id: str | None = None
    segment_id: str = ""
    parent_segment_id: str | None = None
    parent_session_id: str | None = None
    parent_run_id: str | None = None
    budget_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = new_session_id()
        if not self.run_id:
            self.run_id = new_run_id()
        if self.focus_task_id is None and self.task_id is not None:
            self.focus_task_id = self.task_id

    @property
    def storage_task_id(self) -> str:
        return self.task_id or self.compatibility_task_id or self.run_id

    @property
    def material_task_id(self) -> str:
        """取得当前材料和进度所属事项，运行原始存储身份保持不变。

        传参：无；返回：当前焦点任务，或无正式任务时的兼容存储身份
        """
        return self.focus_task_id or self.storage_task_id

    @property
    def has_formal_task(self) -> bool:
        return self.task_id is not None


def new_session_id() -> str:
    return f"session-{uuid4().hex}"


def new_run_id() -> str:
    return f"run-{uuid4().hex}"
