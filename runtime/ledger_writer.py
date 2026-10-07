from __future__ import annotations

import uuid
from typing import Callable, Mapping

from runtime.ledger import LedgerEvent, LedgerStore, new_ledger_event

MODEL_REQUESTED = "model.requested"
TOOL_REQUESTED = "tool.requested"
TOOL_COMPLETED = "tool.completed"
CHECKPOINT_SAVED = "checkpoint.saved"
SUMMARY_UPDATED = "summary.updated"
TASK_FOCUS_CHANGED = "task.focus_changed"
CONTEXT_SEGMENTS_RECORDED = "context.segments.recorded"
RUN_LIFECYCLE_CHANGED = "run.lifecycle.changed"
GOAL_CONFIRMATION_DECIDED = "goal.confirmation_decided"
APPROVAL_REQUESTED = "approval.requested"
APPROVAL_DECIDED = "approval.decided"
APPROVAL_REVOKED = "approval.revoked"

LEDGER_EVENT_NAMES = frozenset(
    {
        MODEL_REQUESTED,
        TOOL_REQUESTED,
        TOOL_COMPLETED,
        CHECKPOINT_SAVED,
        SUMMARY_UPDATED,
        TASK_FOCUS_CHANGED,
        CONTEXT_SEGMENTS_RECORDED,
        RUN_LIFECYCLE_CHANGED,
        GOAL_CONFIRMATION_DECIDED,
        APPROVAL_REQUESTED,
        APPROVAL_DECIDED,
        APPROVAL_REVOKED,
    }
)

MINIMUM_PAYLOAD_FIELDS: Mapping[str, frozenset[str]] = {
    MODEL_REQUESTED: frozenset({"provider", "model", "context_segment_names"}),
    TOOL_REQUESTED: frozenset({"tool_name", "call_id", "args"}),
    TOOL_COMPLETED: frozenset({"tool_name", "call_id", "status"}),
    CHECKPOINT_SAVED: frozenset({"checkpoint_id", "state", "reason"}),
    SUMMARY_UPDATED: frozenset({"summary_kind", "content"}),
    TASK_FOCUS_CHANGED: frozenset({"previous_task_id", "next_task_id", "reason"}),
    CONTEXT_SEGMENTS_RECORDED: frozenset({"segments"}),
    RUN_LIFECYCLE_CHANGED: frozenset({"status"}),
    GOAL_CONFIRMATION_DECIDED: frozenset(
        {
            "action_id",
            "question_id",
            "source_input_id",
            "expected_revision",
            "branch_anchor",
            "branch_id",
            "evidence_digest",
            "evidence",
            "accepted",
        }
    ),
    APPROVAL_REQUESTED: frozenset(
        {"batch_id", "operation_id", "tool", "intent_digest", "definition_version"}
    ),
    APPROVAL_DECIDED: frozenset(
        {
            "action_id",
            "batch_id",
            "operation_id",
            "tool",
            "intent_digest",
            "definition_version",
            "decision",
            "scope",
        }
    ),
    APPROVAL_REVOKED: frozenset({"action_id", "grant_id", "source_input_id"}),
}

_NULLABLE_PAYLOAD_FIELDS = frozenset({"previous_task_id", "next_task_id", "branch_id"})


class LedgerWriter:
    """Ledger 业务事件写入器。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：store 为 LedgerStore，source 为写入来源，id_factory 用于测试稳定生成事件 id
    返回：各 record 方法返回已写入的 LedgerEvent
    """

    def __init__(
        self,
        store: LedgerStore,
        *,
        source: str = "runtime.ledger_writer",
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        """初始化 LedgerWriter。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：store 为底层追加写入存储，source 为事件来源，id_factory 为事件 id 工厂
        返回：无
        """
        if not source.strip():
            raise ValueError("ledger writer source must be non-empty")
        self._store = store
        self._source = source
        self._id_factory = id_factory or _new_event_id

    def record_once(self, event: LedgerEvent) -> LedgerEvent:
        """校验类型化决定并按既有动作身份追加；传参：完整事件；返回：首次提交的事件。"""
        _validate_payload(event.event, event.payload)
        if event.event == GOAL_CONFIRMATION_DECIDED:
            if (
                event.source != "user_action"
                or not event.task_id
                or not event.session_id
                or not event.run_id
            ):
                raise ValueError(
                    "goal confirmation requires a user action and goal/session/run identities"
                )
            if (
                type(event.payload["accepted"]) is not bool
                or type(event.payload["expected_revision"]) is not int
            ):
                raise ValueError(
                    "goal confirmation requires a boolean decision and integer revision"
                )
            if (
                not isinstance(event.payload["evidence"], list)
                or not event.payload["evidence"]
            ):
                raise ValueError("goal confirmation requires delivered evidence")
        return self._store.append_once(event)

    def record_model_requested(
        self,
        provider: str,
        model: str,
        context_segment_names: object,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> LedgerEvent:
        """记录模型请求事实。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：provider 为供应商，model 为模型名，context_segment_names 为上下文段名集合
        返回：已写入 Ledger 的事件
        """
        return self._append_event(
            MODEL_REQUESTED,
            {
                "provider": provider,
                "model": model,
                "context_segment_names": context_segment_names,
            },
            task_id=task_id,
            session_id=session_id,
            run_id=run_id,
        )

    def record_tool_requested(
        self,
        tool_name: str,
        call_id: str,
        args: object,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> LedgerEvent:
        """记录工具调用请求事实。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：tool_name 为工具名，call_id 为调用 id，args 为原始参数结构
        返回：已写入 Ledger 的事件
        """
        return self._append_event(
            TOOL_REQUESTED,
            {"tool_name": tool_name, "call_id": call_id, "args": args},
            task_id=task_id,
            session_id=session_id,
            run_id=run_id,
        )

    def record_tool_completed(
        self,
        tool_name: str,
        call_id: str,
        status: str,
        *,
        content: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> LedgerEvent:
        """记录工具完成事实。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：tool_name 为工具名，call_id 为调用 id，status 为执行状态，content 为模型可见结果
        返回：已写入 Ledger 的事件
        """
        payload: dict[str, object] = {
            "tool_name": tool_name,
            "call_id": call_id,
            "status": status,
        }
        if content is not None:
            payload["content"] = content
        return self._append_event(
            TOOL_COMPLETED,
            payload,
            task_id=task_id,
            session_id=session_id,
            run_id=run_id,
        )

    def record_checkpoint_saved(
        self,
        checkpoint_id: str,
        state: object,
        reason: str,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> LedgerEvent:
        """记录 checkpoint 保存事实。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：checkpoint_id 为检查点 id，state 为原始状态结构，reason 为保存原因
        返回：已写入 Ledger 的事件
        """
        return self._append_event(
            CHECKPOINT_SAVED,
            {"checkpoint_id": checkpoint_id, "state": state, "reason": reason},
            task_id=task_id,
            session_id=session_id,
            run_id=run_id,
        )

    def record_summary_updated(
        self,
        summary_kind: str,
        content: str,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> LedgerEvent:
        """记录 summary 更新事实。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：summary_kind 为摘要类型，content 为摘要内容，task/session/run 为关联 id
        返回：已写入 Ledger 的事件
        """
        return self._append_event(
            SUMMARY_UPDATED,
            {"summary_kind": summary_kind, "content": content},
            task_id=task_id,
            session_id=session_id,
            run_id=run_id,
        )

    def record_task_focus_changed(
        self,
        previous_task_id: str | None,
        next_task_id: str | None,
        reason: str,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> LedgerEvent:
        """记录 task focus 切换事实。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：previous_task_id 为原 focus，next_task_id 为新 focus，reason 为切换原因
        返回：已写入 Ledger 的事件
        """
        return self._append_event(
            TASK_FOCUS_CHANGED,
            {
                "previous_task_id": previous_task_id,
                "next_task_id": next_task_id,
                "reason": reason,
            },
            task_id=next_task_id,
            session_id=session_id,
            run_id=run_id,
        )

    def record_context_segments(
        self,
        segments: object,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> LedgerEvent:
        """记录上下文段事实。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：segments 为上下文段原始结构，task/session/run 为关联 id
        返回：已写入 Ledger 的事件
        """
        return self._append_event(
            CONTEXT_SEGMENTS_RECORDED,
            {"segments": segments},
            task_id=task_id,
            session_id=session_id,
            run_id=run_id,
        )

    def record_run_lifecycle_changed(
        self,
        status: str,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> LedgerEvent:
        """记录 run 生命周期变化事实。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：status 为生命周期状态，task/session/run 为关联 id
        返回：已写入 Ledger 的事件
        """
        return self._append_event(
            RUN_LIFECYCLE_CHANGED,
            {"status": status},
            task_id=task_id,
            session_id=session_id,
            run_id=run_id,
        )

    def _append_event(
        self,
        event: str,
        payload: Mapping[str, object],
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> LedgerEvent:
        """校验并追加 Ledger 事件。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：event 为事件名，payload 为业务载荷，task/session/run 为关联 id
        返回：已写入 Ledger 的事件
        """
        _validate_payload(event, payload)
        ledger_event = new_ledger_event(
            event,
            self._id_factory(),
            self._source,
            payload,
            task_id=task_id,
            session_id=session_id,
            run_id=run_id,
        )
        self._store.append(ledger_event)
        return ledger_event


def _new_event_id() -> str:
    """生成 Ledger 事件 id。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：无
    返回：带 ledger- 前缀的唯一事件 id
    """
    return f"ledger-{uuid.uuid4().hex}"


def _validate_payload(event: str, payload: Mapping[str, object]) -> None:
    """校验业务事件最小 payload 合同。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：event 为事件名，payload 为待写入载荷
    返回：无，校验失败时抛出 ValueError
    """
    if event not in LEDGER_EVENT_NAMES:
        raise ValueError(f"unsupported ledger event: {event}")
    required_fields = MINIMUM_PAYLOAD_FIELDS[event]
    missing = sorted(field for field in required_fields if field not in payload)
    if missing:
        raise ValueError(
            f"ledger payload missing required fields: {', '.join(missing)}"
        )
    for field_name in sorted(required_fields):
        _validate_payload_value(field_name, payload[field_name])


def _validate_payload_value(field_name: str, value: object) -> None:
    """校验单个 payload 字段值。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：field_name 为字段名，value 为字段值
    返回：无，校验失败时抛出 ValueError
    """
    if value is None and field_name not in _NULLABLE_PAYLOAD_FIELDS:
        raise ValueError(f"ledger payload {field_name} must be present")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"ledger payload {field_name} must be non-empty")


__all__ = [
    "GOAL_CONFIRMATION_DECIDED",
    "CHECKPOINT_SAVED",
    "CONTEXT_SEGMENTS_RECORDED",
    "LEDGER_EVENT_NAMES",
    "MINIMUM_PAYLOAD_FIELDS",
    "MODEL_REQUESTED",
    "RUN_LIFECYCLE_CHANGED",
    "SUMMARY_UPDATED",
    "TASK_FOCUS_CHANGED",
    "TOOL_COMPLETED",
    "TOOL_REQUESTED",
    "LedgerWriter",
]
