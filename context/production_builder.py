from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from context.engine import recall_context_materials
from context.materials import ContextMaterial, render_materials
from context.history_view import history_context
from context.token_estimate import estimate_agent_messages_tokens, estimate_tokens
from llm.messages import (
    AgentMessage,
    group_tool_call_units,
    model_visible_text,
    recent_user_text,
)
from runtime.ledger import LedgerStore
from runtime.session_message_store import MaterializedSession, SessionMessageStore
from runtime.session_compaction import SessionCompactionStore, SessionSummary
from runtime.task_state_view import build_task_state_view
from runtime.types import RunContext
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry
from runtime.capability_catalog import loaded_tools_from_operations


class ContextReadError(RuntimeError):
    def __init__(self, category: str, task_id: str, cause: Exception) -> None:
        self.category = category
        self.task_id = task_id
        self.cause_type = type(cause).__name__
        self.message = f"{category}: {self.cause_type}: {cause}"
        super().__init__(self.message)

    def render_output(self) -> str:
        return f"CONTEXT_READ_ERROR: {self.category}; stopped before model call."

    def meta(self) -> dict[str, object]:
        return {"task_id": self.task_id, "cause_type": self.cause_type}


@dataclass(frozen=True, slots=True)
class HistorySelection:
    """本轮模型可见的历史尾部，以及尾部之前是否还有被截掉的更早对话。

    作者：LKX
    时间：2026-08-30 14:20:00
    传参：messages 为选中的 canonical 消息；truncated 表示更早历史被截断
    返回：不可变历史选择结果

    截断事实与消息本身分开携带，因为"更早历史被截掉了"是关于本轮上下文的说明，不是对话
    里发生过的一轮。它由 prompt 组装渲染进 instructions，模型据此选择 read_history 继续补读，
    不再伪装成一条 system 消息混在历史序列里。
    """

    messages: tuple[AgentMessage, ...]
    truncated: bool
    summary: SessionSummary | None = None
    loaded_tools: frozenset[str] = frozenset()
    recall_user_text: str = ""
    line_limit: int | None = None
    line_omitted_count: int = 0
    history_context: Mapping[str, object] = field(default_factory=dict)

    @property
    def retained_count(self) -> int:
        """本轮保留的历史消息条数，用于截断说明里如实报数。"""
        return len(self.messages)


@dataclass(frozen=True, slots=True)
class ContextSegmentEvidence:
    name: str
    source: str
    authority: str
    layer: str
    token_est: int
    included_reason: str
    order: int
    meta: Mapping[str, object] = field(default_factory=dict)

    def to_run_fact_segment(self) -> dict[str, object]:
        segment = {
            "name": self.name,
            "tokens_est": self.token_est,
            "order": self.order,
            "layer": self.layer,
            "source": self.source,
            "authority": self.authority,
            "included_reason": self.included_reason,
        }
        segment.update(dict(self.meta))
        return segment


@dataclass(frozen=True, slots=True)
class ProductionContextBundle:
    model_task: str
    model_context: Mapping[str, object]
    segments: tuple[ContextSegmentEvidence, ...]


class ProductionContextBuilder:
    def __init__(
        self,
        data_root: Path | str,
        *,
        system_prompt_provider: Callable[[], str],
    ) -> None:
        self.data_root = Path(data_root)
        self._system_prompt_provider = system_prompt_provider

    def build(
        self,
        *,
        task: str,
        context: RunContext,
        toolset_policy: Mapping[str, object],
        tool_registry: ToolRegistry,
        system_reminder: str | None = None,
        recoverable_error_notice: str = "",
        no_progress_observation: str = "",
    ) -> ProductionContextBundle:
        store = TaskStore(self.data_root)
        task_summary = self.task_summary_context(context.material_task_id, store=store)
        history = self.read_conversation_history(
            context.session_id,
            store=store,
            input_message_id=str(context.payload.get("input_message_id", "")),
        )
        recall = self.recall_for_context(context, history=history, store=store)
        model_task = _model_task(task, context, task_summary)
        model_context = self._model_context(
            context=context,
            toolset_policy=toolset_policy,
            tool_registry=tool_registry,
            history=history,
            task_summary=task_summary,
            recall_context="\n\n".join(
                part for part in (recall[0], render_materials(recall[1], {})) if part
            ),
            system_reminder=system_reminder,
            recoverable_error_notice=recoverable_error_notice,
            no_progress_observation=no_progress_observation,
        )
        # 【工具体系】【材料入口】引用材料携带可执行读取入口；这只选择上下文，不增加权限
        required_tools = set(history.loaded_tools)
        if history.truncated or history.summary is not None:
            required_tools.add("read_history")
        if any(item.source == "skill" for item in recall[1]):
            required_tools.add("skill_read")
        if context.payload.get("recovery_intent") or context.payload.get(
            "_resume_choice_pending_evidence"
        ):
            required_tools.update(("operation_status", "resume_operation"))
        model_context["loaded_tools"] = frozenset(required_tools)
        model_context["context_materials"] = recall[1]
        model_context["recall_notices"] = recall[0]
        if context.focus_task_id is not None:
            goal = store.require_task(context.focus_task_id)
            model_context["focus_goal"] = {
                "task_id": goal.task_id,
                "goal": goal.goal,
                "status": goal.status,
                "revision": goal.revision,
                "status_source": goal.status_source,
            }
        return ProductionContextBundle(
            model_task=model_task,
            model_context=model_context,
            segments=self.segment_evidence(model_context, model_task=model_task),
        )

    def task_summary_context(
        self, task_id: str, *, store: TaskStore | None = None
    ) -> dict[str, str]:
        del store
        try:
            view = build_task_state_view(
                LedgerStore(self.data_root).read_task_events(task_id)
            )
        except Exception as exc:
            raise ContextReadError("context_summary_read_failed", task_id, exc) from exc
        return {
            "intent": view.intent,
            "resume_hint": view.resume_hint,
            "progress": view.progress,
            "summary": view.summary,
        }

    def read_conversation_history(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        store: TaskStore | None = None,
        input_message_id: str = "",
    ) -> HistorySelection:
        """从唯一消息owner取当前分支及已发布摘要；未总结的原文不能被预算静默删除。

        作者：LKX
        时间：2026-08-30 14:20:00
        传参：session_id 为会话标识；limit 为条数上限；input_message_id 为本轮已保存输入
        返回：选中的 canonical 消息与截断事实；读取失败暴露为 ContextReadError
        """
        del store
        try:
            owner = SessionMessageStore(self.data_root)
            view = (
                owner.materialize(session_id)
                if owner.exists(session_id)
                else MaterializedSession(session_id, None, (), ())
            )
            summaries = SessionCompactionStore(owner)
            summary = summaries.current(view)
            history_state = history_context(view, summaries)
            messages = view.messages
            projected = messages[len(summary.message_ids) :] if summary else messages
        except Exception as exc:
            raise ContextReadError(
                "conversation_history_read_failed", session_id, exc
            ) from exc
        selected = (
            _tail_within_count(projected, max_messages=limit)
            if limit is not None
            else list(projected)
        )
        # 【上下文】【本轮输入】裁剪历史后仍保留当前输入，只按消息身份判断，不按正文去重
        if input_message_id and not any(
            message.message_id == input_message_id for message in selected
        ):
            current = next(
                (
                    message
                    for message in messages
                    if message.message_id == input_message_id
                ),
                None,
            )
            if current is None:
                raise ContextReadError(
                    "current_input_missing",
                    session_id,
                    ValueError(
                        f"input message not in current branch: {input_message_id}"
                    ),
                )
            selected.insert(0, current)
        return HistorySelection(
            messages=tuple(selected),
            truncated=len(selected) < len(messages),
            summary=summary,
            loaded_tools=loaded_tools_from_operations(
                self.data_root, session_id, messages
            ),
            recall_user_text=_recall_user_text(view),
            line_limit=limit,
            line_omitted_count=max(0, len(projected) - len(selected)),
            history_context=history_state,
        )

    def recall_for_context(
        self,
        context: RunContext,
        *,
        history: HistorySelection,
        store: TaskStore | None = None,
    ) -> tuple[str, tuple[ContextMaterial, ...]]:
        """按当前运行和已接纳维护范围召回材料；参数：运行/历史/任务存储；返回：提示及材料。"""
        from runtime.knowledge_maintenance import automatic_origin

        owner = store or TaskStore(self.data_root)
        record = owner.load_task(context.material_task_id)
        if record is None:
            return "", ()
        return recall_context_materials(
            self.data_root,
            task_summary=record.goal,
            task_tags=record.tags,
            skill_refs=record.skill_refs,
            recent_user_text=history.recall_user_text,
            session_id=context.session_id,
            run_id=context.run_id,
            task_id=context.storage_task_id,
            focus_task_id=context.focus_task_id,
            knowledge_origin=automatic_origin(context),
        )

    def segment_evidence(
        self, model_context: Mapping[str, object], *, model_task: str
    ) -> tuple[ContextSegmentEvidence, ...]:
        segments = _base_segments(self._system_prompt_provider())
        segments = _append_optional_segments(segments, model_context)
        segments.append(_recall_segment(model_context, len(segments)))
        segments.append(_user_task_segment(model_task, len(segments)))
        return tuple(segments)

    def _model_context(
        self,
        *,
        context: RunContext,
        toolset_policy: Mapping[str, object],
        tool_registry: ToolRegistry,
        history: HistorySelection,
        task_summary: Mapping[str, str],
        recall_context: str,
        system_reminder: str | None,
        recoverable_error_notice: str = "",
        no_progress_observation: str = "",
    ) -> dict[str, object]:
        """组装本次请求的材料和运行边界；传参：运行与已选材料；返回：模型上下文。"""
        model_context: dict[str, object] = {
            "task_id": context.task_id,
            "storage_task_id": context.storage_task_id,
            "session_id": context.session_id,
            "run_id": context.run_id,
            "data_root": str(self.data_root),
            "focus_task_id": context.focus_task_id,
            "input_message_id": context.payload.get("input_message_id", ""),
            "segment_id": context.segment_id,
            "trigger": context.trigger.value,
            "capability_lease": context.capability_lease,
            "toolset_policy": dict(toolset_policy),
            "tool_registry": tool_registry,
            "loaded_tools": history.loaded_tools,
            "artifact_output_dir": f".reins/workspace/{context.material_task_id}/outputs",
            "conversation_history": history.messages,
            "history_selection": {
                "line_limit": history.line_limit,
                "line_limit_applied": history.line_limit is not None,
                "line_omitted_count": history.line_omitted_count,
                "summary_covered_count": len(history.summary.message_ids)
                if history.summary
                else 0,
            },
            "task_summary_layers": dict(task_summary),
            "recall_context": recall_context,
            **_runtime_context_evidence(context),
            **history.history_context,
        }
        # 【上下文】【历史裁剪】告知模型保留范围，使其可以选择补读更早历史
        if history.truncated:
            model_context["history_truncated_retained"] = history.retained_count
        if history.summary is not None:
            model_context["session_summary"] = history.summary.model_view()
        if system_reminder:
            model_context["system_reminder"] = system_reminder
        # 【上下文】【错误反馈】协议错误单独传递，避免与其他运行提示互相覆盖
        if recoverable_error_notice:
            model_context["recoverable_error_notice"] = recoverable_error_notice
        if no_progress_observation:
            model_context["no_progress_observation"] = no_progress_observation
        return model_context


def _recall_user_text(view: MaterializedSession) -> str:
    """从当前分支的完整消息及来源选择用户问题；传参：同次读取的快照；返回：用户正文或空串。"""
    # 1. 【上下文】【召回来源】后台与协作输入仍保留在对话中，但不能替用户更换检索主题
    agent_ids = {
        entry.message.message_id
        for entry in view.entries
        if entry.input_source == "agent" and entry.message is not None
    }
    # 2. 【上下文】【召回来源】在压缩和显示裁剪之前取用户消息，排除尚未交付的入站输入
    return recent_user_text(
        tuple(
            message for message in view.messages if message.message_id not in agent_ids
        )
    )


def _runtime_context_evidence(context: RunContext) -> dict[str, object]:
    """投影真实工具目录、预算、待决操作和会话命令；传参：运行；返回：本轮上下文字段。"""
    result: dict[str, object] = {}
    recovery = context.payload.get("recovery_intent")
    if isinstance(recovery, dict):
        result["recovery_intent"] = dict(recovery)
    # 1. 【上下文】【工具目录】相对路径以实际工具项目为准，宿主进程的cwd不能替代该目录
    filesystem = context.capability_lease.capabilities.get("fs")
    if isinstance(filesystem, Mapping) and "project_root" in filesystem:
        project_root = filesystem["project_root"]
        if not isinstance(project_root, (str, Path)) or not str(project_root).strip():
            raise ValueError("tool filesystem project_root must be a non-empty path")
        result["tool_working_directory"] = str(Path(project_root).resolve())
    for source, target in (
        ("_resume_choice_pending_evidence", "resume_choice_pending_evidence"),
        ("_runtime_budget_evidence", "budget_evidence"),
    ):
        evidence = context.payload.get(source)
        if isinstance(evidence, dict):
            result[target] = dict(evidence)
    if context.payload.get("message") == "/compact":
        result["context_purpose"] = "compaction_confirmation"
        result["system_reminder"] = (
            "用户通过/compact请求整理当前会话。带来源的会话摘要将在本次请求前完成发布，"
            "原始记录保留。简要报告整理结果，随后等待用户继续，不执行其他业务动作。"
        )
    return result


def _model_task(task: str, context: RunContext, task_summary: Mapping[str, str]) -> str:
    if not _is_continue_message(str(context.payload.get("message", ""))):
        return task
    resume_hint = str(task_summary.get("resume_hint", "")).strip()
    return f"{resume_hint}\n\nUser said: {task}" if resume_hint else task


def _base_segments(system_prompt: str) -> list[ContextSegmentEvidence]:
    return [
        ContextSegmentEvidence(
            name="system_prompt",
            source="prompt_composer",
            authority="prompt_material",
            layer="stable",
            token_est=estimate_tokens(system_prompt),
            included_reason="base runtime instructions for every model request",
            order=0,
        )
    ]


def _append_optional_segments(
    segments: list[ContextSegmentEvidence], model_context: Mapping[str, object]
) -> list[ContextSegmentEvidence]:
    _append_system_reminder(segments, model_context)
    _append_recoverable_error_notice(segments, model_context)
    _append_no_progress_observation(segments, model_context)
    _append_toolset_policy(segments, model_context)
    _append_conversation_history(segments, model_context)
    _append_session_summary(segments, model_context)
    _append_summary_layers(segments, model_context)
    _append_filesystem_locations(segments, model_context)
    return segments


def _append_system_reminder(
    segments: list[ContextSegmentEvidence], model_context: Mapping[str, object]
) -> None:
    reminder = str(model_context.get("system_reminder", ""))
    if not reminder:
        return
    segments.append(
        ContextSegmentEvidence(
            name="system_reminder",
            source="runtime_convergence",
            authority="prompt_material",
            layer="ephemeral",
            token_est=estimate_tokens(reminder),
            included_reason="runtime injected a one-turn convergence reminder",
            order=len(segments),
        )
    )


def _append_recoverable_error_notice(
    segments: list[ContextSegmentEvidence], model_context: Mapping[str, object]
) -> None:
    """把上一轮的可恢复错误原因登记为本轮的一次性 prompt 材料。

    作者：xxx
    时间：2026-09-01 00:00:00
    传参：segments 为累积的 section 证据；model_context 为本轮模型上下文
    返回：无；有错误说明时原地追加一个 ephemeral section
    """
    notice = str(model_context.get("recoverable_error_notice", ""))
    if not notice:
        return
    segments.append(
        ContextSegmentEvidence(
            name="recoverable_error_notice",
            source="runtime_recovery",
            authority="prompt_material",
            layer="ephemeral",
            token_est=estimate_tokens(notice),
            included_reason="previous turn failed with a recoverable protocol error",
            order=len(segments),
        )
    )


def _append_no_progress_observation(
    segments: list[ContextSegmentEvidence], model_context: Mapping[str, object]
) -> None:
    """登记本轮无进展观察的一次性提示；传参：段落证据与模型上下文；返回：无。"""
    observation = str(model_context.get("no_progress_observation", ""))
    if not observation:
        return
    segments.append(
        ContextSegmentEvidence(
            name="no_progress_observation",
            source="runtime_progress",
            authority="prompt_material",
            layer="ephemeral",
            token_est=estimate_tokens(observation),
            included_reason="previous read-only action produced no new evidence",
            order=len(segments),
        )
    )


def _append_toolset_policy(
    segments: list[ContextSegmentEvidence], model_context: Mapping[str, object]
) -> None:
    if not isinstance(model_context.get("toolset_policy"), Mapping):
        return
    segments.append(
        ContextSegmentEvidence(
            name="toolset_policy",
            source="toolset_runtime_config",
            authority="prompt_material",
            layer="dynamic",
            token_est=0,
            included_reason="tool policy selects the model-visible tool set",
            order=len(segments),
        )
    )


def _append_conversation_history(
    segments: list[ContextSegmentEvidence], model_context: Mapping[str, object]
) -> None:
    history = model_context.get("conversation_history", ())
    if not isinstance(history, (list, tuple)):
        return
    hist_text = " ".join(model_visible_text(message) for message in history)
    segments.append(
        ContextSegmentEvidence(
            name="conversation_history",
            source="SessionMessageStore",
            authority="fact",
            layer="dynamic",
            token_est=estimate_agent_messages_tokens(history),
            included_reason="selected conversation tail for this model request",
            order=len(segments),
            meta={
                "content_tokens_est": estimate_tokens(hist_text),
                "token_estimate_kind": "message_with_overhead",
                "message_count": len(history),
            },
        )
    )


def _append_session_summary(
    segments: list[ContextSegmentEvidence], model_context: Mapping[str, object]
) -> None:
    """记录派生摘要及原文范围，避免把摘要误称真人指令；传参：证据和上下文；返回：无。"""
    summary = model_context.get("session_summary")
    if not isinstance(summary, Mapping):
        return
    segments.append(
        ContextSegmentEvidence(
            name="session_summary",
            source="SessionCompactionStore",
            authority="projection",
            layer="dynamic",
            token_est=estimate_tokens(json.dumps(dict(summary), ensure_ascii=False)),
            included_reason="published semantic summary covers the earlier session messages",
            order=len(segments),
            meta={
                "summary_id": summary["summary_id"],
                "summary_source": summary["source"],
                "first_kept_message_id": summary["first_kept_message_id"],
            },
        )
    )


def _append_summary_layers(
    segments: list[ContextSegmentEvidence], model_context: Mapping[str, object]
) -> None:
    summary_layers = model_context.get("task_summary_layers", {})
    if not isinstance(summary_layers, Mapping) or not any(summary_layers.values()):
        return
    layers_text = " ".join(str(v) for v in summary_layers.values() if v)
    segments.append(
        ContextSegmentEvidence(
            name="task_summary_layers",
            source="TaskStateView",
            authority="projection",
            layer="semi_stable",
            token_est=estimate_tokens(layers_text),
            included_reason="current task summary projection is non-empty",
            order=len(segments),
        )
    )


def _append_filesystem_locations(
    segments: list[ContextSegmentEvidence], model_context: Mapping[str, object]
) -> None:
    """分别说明工具工作目录和内部产物目录的真实来源；传参：观测列表与上下文；返回：无。"""
    for name, source in (
        ("artifact_output_dir", "run_workspace"),
        ("tool_working_directory", "capability_lease.fs.project_root"),
    ):
        location = str(model_context.get(name, "")).strip()
        if location:
            segments.append(
                ContextSegmentEvidence(
                    name=name,
                    source=source,
                    authority="prompt_material",
                    layer="dynamic",
                    token_est=estimate_tokens(location),
                    included_reason="actual filesystem location supplied to the model",
                    order=len(segments),
                )
            )


def _recall_segment(
    model_context: Mapping[str, object], order: int
) -> ContextSegmentEvidence:
    recall_context = str(model_context.get("recall_context", ""))
    return ContextSegmentEvidence(
        name="recall_context",
        source="memory_store+skill_store",
        authority="prompt_material",
        layer="dynamic",
        token_est=estimate_tokens(recall_context),
        included_reason="auto-recalled memory and skill bodies for the task",
        order=order,
    )


def _user_task_segment(model_task: str, order: int) -> ContextSegmentEvidence:
    return ContextSegmentEvidence(
        name="user_task",
        source="run_context.payload.message",
        authority="prompt_material",
        layer="dynamic",
        token_est=estimate_tokens(model_task),
        included_reason="current model task text after resume hint handling",
        order=order,
    )


def _tail_within_count(
    messages: Sequence[AgentMessage], *, max_messages: int
) -> list[AgentMessage]:
    """按条数上限取历史尾部，且不切开任何工具调用组。

    作者：LKX
    时间：2026-08-30 14:20:00
    传参：messages 为时间顺序的历史消息；max_messages 为保留条数上限，非正数表示全量
    返回：新的消息 list；实际条数可少于上限，绝不多于

    上限落在某个工具调用组中间时整组丢弃，而不是留下半组：留下的半组要么是没有公告的
    孤立工具结果，要么是没有结果的悬空调用，两种都会让模型读到不成立的调用图。
    """
    if max_messages <= 0:
        return list(messages)
    selected: list[tuple[AgentMessage, ...]] = []
    kept = 0
    for group in reversed(group_tool_call_units(messages)):
        if kept + len(group) > max_messages:
            break
        selected.append(group)
        kept += len(group)
    selected.reverse()
    return [message for group in selected for message in group]


def _is_continue_message(message: str) -> bool:
    return message.strip().lower() in {"继续", "continue", "resume", "接着"}


__all__ = [
    "ContextReadError",
    "ContextSegmentEvidence",
    "HistorySelection",
    "ProductionContextBundle",
    "ProductionContextBuilder",
]
