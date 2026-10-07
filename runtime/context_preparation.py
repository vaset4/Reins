"""在真实模型边界中生成、独立核对并原子发布增量接续摘要。

作者：xxx
时间：2026-09-25 12:00:00
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

from context.compaction import (
    CompactionMaterial,
    auxiliary_window,
    compaction_material,
    fit_fragment,
    original_sources,
    source_groups,
    split_source_group,
)
from context.production_builder import (
    ContextReadError,
    ProductionContextBuilder,
    ProductionContextBundle,
)
from context.summary_entries import SummaryContent, apply_delta, parse_delta
from context.history_segments import segment_digest, validate_segment_coverage
from context.token_estimate import estimate_tokens
from context.window import ContextWindowExceeded, request_budget
from llm.messages import group_tool_call_units
from llm.model_request import ComposedRequest
from llm.types import LLMPlan
from runtime.cancellation import (
    CancellationToken,
    ExecutionCancelled,
    RunBudgetExceeded,
)
from runtime.session_compaction import SessionCompactionStore

from llm.toolset_policy import policy_to_mapping
from runtime.types import RunContext, RunToolsResult
from runtime.model_execution import ModelRequestRunner
from runtime.extension_execution import ExtensionExecution
from runtime.tool_policy import RuntimeToolPolicy
from runtime.tool_result_views import ToolResultViews
from runtime.session_message_store import SessionMessageStore
from runtime.progress import ProgressGuard
from runtime.collaboration import CollaborationRuntime
from runtime.watchdog import Watchdog
from tools.tool_registry import ToolRegistry

PRESSURE_RATIO = 0.8
TARGET_RATIO = 0.6


class SummaryProviderOverflow(ValueError):
    """表示供应商拒绝片段容量；传参：错误说明；返回：允许按原文边界继续缩小的明确错误。"""


class ContextCompactor:
    """统一协调材料、摘要模型与原文写者，不获得记忆或业务操作权限。"""

    def __init__(
        self,
        store: SessionCompactionStore,
        prepare: Callable[[str, Mapping[str, object]], ComposedRequest],
        summarize: Callable[[ProductionContextBundle], LLMPlan],
        *,
        cancelled: Callable[[], bool] | None = None,
        enqueue: Callable[
            [CompactionMaterial, Mapping[str, object]], Mapping[str, object]
        ]
        | None = None,
        proactive: bool = True,
    ) -> None:
        """注入执行和取消边界；传参：存储、准备器、模型及取消查询；返回：无。"""
        self.store, self.prepare, self.summarize = store, prepare, summarize
        self.cancelled = cancelled
        self.enqueue = enqueue
        self.proactive = proactive

    def fit(
        self,
        bundle: ProductionContextBundle,
        *,
        rebuild: Callable[[], ProductionContextBundle],
        force: bool = False,
    ) -> ProductionContextBundle:
        """有压力时整理早期原文，提交后重建当前尾部；传参：上下文与重建器；返回：可发送材料。"""
        current = bundle
        compacted = False
        started = time.perf_counter()
        while True:
            prepared = self.prepare(current.model_task, current.model_context)
            budget = request_budget(prepared.request, prepared.context_window)
            fits = budget.required_total <= budget.context_window
            fixed = (
                budget.instructions
                + budget.tools
                + budget.protocol
                + budget.output_reserved
            )
            available = max(0, budget.context_window - fixed)
            pressured = budget.messages > available * PRESSURE_RATIO
            if (
                not force
                and fits
                and (not self.proactive or compacted or not pressured)
            ):
                evidence = {
                    **cast(
                        Mapping[str, object],
                        current.model_context.get("compaction_evidence", {}),
                    ),
                    "after": budget.evidence(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "target_met": budget.messages <= available * TARGET_RATIO,
                    "applied": compacted,
                }
                return self._ready(current, prepared, evidence)
            session_id = str(current.model_context["session_id"])
            view = self.store.messages.materialize(session_id)
            previous = self.store.current(view)
            begin = len(previous.message_ids) if previous else 0
            if len(group_tool_call_units(view.messages[begin:])) < 2 and (
                previous is None or previous.content is not None
            ):
                if force:
                    raise ValueError(
                        "not enough completed history to compact; the latest message group must remain readable"
                    )
                if not fits:
                    raise ContextWindowExceeded(budget)
                return self._ready(
                    current,
                    prepared,
                    {
                        "applied": False,
                        "reason": "no_new_completed_history",
                        "after": budget.evidence(),
                    },
                )
            fixed_context = {**current.model_context, "session_summary": None}
            if previous is not None and previous.content is None:
                # 1. 【上下文】【旧摘要重建】旧正文即将由同源片段替换，不作为不可替换的固定成本重复扣除
                fixed_context["history_materials"] = ()
            fixed_request = self.prepare(current.model_task, fixed_context)
            selection = replace(
                budget,
                instructions=request_budget(
                    fixed_request.request, fixed_request.context_window
                ).instructions,
            )
            material = compaction_material(
                view,
                previous,
                selection,
                force=force,
                retained_target=int(available * TARGET_RATIO / 2),
            )
            if fits and not force and self.enqueue is not None:
                queued = self.enqueue(material, current.model_context)
                return self._ready(
                    current,
                    prepared,
                    {
                        "applied": False,
                        "background": dict(queued),
                        "after": budget.evidence(),
                    },
                )
            content, requests = self._summarize(
                material, current.model_context, context_window=budget.context_window
            )
            text = content.render()
            prior_text = previous.text if previous else ""
            if estimate_tokens(text) >= estimate_tokens(
                material.text
            ) + estimate_tokens(prior_text):
                raise ValueError("semantic summary did not reduce the covered context")
            record = self.store.publish(
                material.source,
                text,
                request_ids=requests,
                content=content,
                cancelled=self.cancelled,
            )
            trigger = "manual_or_provider_overflow" if force else "window_pressure"
            compacted, force = True, False
            refreshed = rebuild()
            current = replace(
                refreshed,
                model_context={
                    **refreshed.model_context,
                    "extension_context": bundle.model_context.get(
                        "extension_context", ()
                    ),
                    "compaction_evidence": {
                        "summary_id": record.summary_id,
                        "before": budget.evidence(),
                        "pressure_ratio": PRESSURE_RATIO,
                        "target_ratio": TARGET_RATIO,
                        "request_count": len(requests),
                        "covered_count": len(record.message_ids),
                        "new_message_count": len(record.new_message_ids or ()),
                        "messages_since_previous_compaction": len(view.messages)
                        - begin,
                        "trigger": trigger,
                    },
                },
            )

    def _ready(
        self,
        current: ProductionContextBundle,
        prepared: ComposedRequest,
        evidence: Mapping[str, object],
    ) -> ProductionContextBundle:
        """保存本轮实际采用及排队事实，不把发布当采用；参数：材料、请求和整理证据；返回：可发送请求。"""
        prepared = replace(
            prepared,
            trim_delta={
                **(prepared.trim_delta or {}),
                "compaction": dict(evidence),
                "history_selection": current.model_context.get(
                    "history_selection", {"line_limit_applied": False}
                ),
            },
        )
        return replace(
            current,
            model_context={**current.model_context, "prepared_request": prepared},
        )

    def _summarize(
        self,
        material: CompactionMaterial,
        context: Mapping[str, object],
        *,
        context_window: int,
        verify_originals: bool = True,
    ) -> tuple[SummaryContent, tuple[str, ...]]:
        """按完整交互生成差量并核对原文，未变条目不重写；传参：冻结材料和模型范围；返回：候选及真实请求身份。"""
        previous = material.source.previous
        published = (
            previous.content if previous and previous.content else SummaryContent()
        )
        content = (
            SummaryContent(
                entries=tuple(
                    entry for entry in previous.content.entries if entry.active
                )
            )
            if previous and previous.content
            else SummaryContent()
        )
        base = previous.summary_id if previous else None
        sources = original_sources(material.source)
        pending = source_groups(material)
        if not pending:
            raise ValueError("no new originals to compact")
        model_context = {
            key: context[key]
            for key in ("session_id", "run_id", "segment_id", "tool_registry")
        }
        model_context["context_purpose"] = "compaction"
        requests: list[str] = []
        limit_groups: int | None = None
        maximum_required: int | None = None
        while pending:
            preceding = auxiliary_window(
                self.prepare,
                content,
                model_context,
                _preceding_originals(material, pending[0]),
                base_summary_id=base,
            )
            fragment, remaining, prepared = fit_fragment(
                self.prepare,
                pending,
                content,
                model_context,
                base_summary_id=base,
                limit_groups=limit_groups,
                auxiliary=preceding,
                maximum_required=maximum_required,
            )
            try:
                delta = self._invoke(prepared, model_context, requests)
            except SummaryProviderOverflow:
                maximum_required = (
                    request_budget(
                        prepared.request, prepared.context_window
                    ).required_total
                    - 1
                )
                pending, limit_groups = _smaller_fragment(fragment, remaining)
                continue
            ids = tuple(
                dict.fromkeys(
                    str(message["message_id"])
                    for group in fragment
                    for message in group
                )
            )
            candidate = apply_delta(
                content,
                delta,
                base_summary_id=base,
                sources=sources,
                required_messages=ids,
                published_content=published,
            )
            if verify_originals:
                candidate = self._audit(
                    candidate,
                    fragment,
                    model_context,
                    requests=requests,
                    base=base,
                    sources=sources,
                    auxiliary=preceding,
                    related=_changed_originals(content, delta, material),
                    published_content=published,
                )
            content, pending, limit_groups, maximum_required = (
                candidate,
                remaining,
                None,
                None,
            )
        if context.get("history_representation_version") == 1:
            begin = len(previous.message_ids) if previous and previous.content else 0
            validate_segment_coverage(
                content.segments,
                material.source.view.messages[begin : material.source.covered_count],
            )
        return content, tuple(requests)

    def _audit(
        self,
        candidate: SummaryContent,
        fragment: list[list[dict[str, object]]],
        context: Mapping[str, object],
        *,
        requests: list[str],
        base: str | None,
        sources: Mapping[str, Mapping[str, str]],
        auxiliary: list[dict[str, object]],
        published_content: SummaryContent,
        related: list[list[dict[str, object]]],
    ) -> SummaryContent:
        """再次直接读取原文，记录发现和局部修订；传参：候选、原文和来源；返回：已核对但不保证无遗漏的候选。"""
        pending = [*fragment, *related]
        new_ids = frozenset(
            str(message["message_id"]) for group in fragment for message in group
        )
        current = candidate
        limit_groups: int | None = None
        maximum_required: int | None = None
        while pending:
            preceding = auxiliary_window(
                self.prepare, current, context, auxiliary, base_summary_id=base
            )
            groups, remaining, prepared = fit_fragment(
                self.prepare,
                pending,
                current,
                context,
                base_summary_id=base,
                auxiliary=preceding,
                audit=True,
                new_message_ids=new_ids,
                limit_groups=limit_groups,
                maximum_required=maximum_required,
            )
            started = time.perf_counter()
            try:
                delta = self._invoke(prepared, context, requests)
            except SummaryProviderOverflow:
                maximum_required = (
                    request_budget(
                        prepared.request, prepared.context_window
                    ).required_total
                    - 1
                )
                pending, limit_groups = _smaller_fragment(groups, remaining)
                continue
            ids = tuple(
                dict.fromkeys(
                    str(message["message_id"]) for group in groups for message in group
                )
            )
            revised = apply_delta(
                current,
                delta,
                base_summary_id=base,
                sources=sources,
                required_messages=tuple(
                    identity for identity in ids if identity in new_ids
                ),
                published_content=published_content,
                new_evidence=tuple(new_ids),
            )
            findings = delta.get("findings", [])
            if any(
                item.get("message_id") not in sources
                or not isinstance(item.get("detail"), str)
                for item in findings
            ):
                raise ValueError("summary finding must reference a frozen original")
            check = {
                "request_id": requests[-1],
                "message_ids": ids,
                "findings": findings,
                "segment_ids": tuple(
                    segment.segment_id
                    for segment in revised.segments
                    if set(segment.message_ids).intersection(ids)
                ),
                "segment_versions": {
                    segment.segment_id: segment_digest(segment)
                    for segment in revised.segments
                    if set(segment.message_ids).intersection(ids)
                },
                "revised_ids": tuple(
                    item["entry_id"] for item in delta.get("revise", [])
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "source_spans": _source_spans(groups),
                "auxiliary_message_ids": tuple(
                    item["message_id"] for item in preceding
                ),
                "auxiliary_source_spans": _source_spans([preceding]),
            }
            current = replace(revised, checks=(*current.checks, check))
            pending, limit_groups, maximum_required = remaining, None, None
        return current

    def _invoke(
        self,
        prepared: ComposedRequest,
        context: Mapping[str, object],
        requests: list[str],
    ) -> dict[str, Any]:
        """调用同一生产模型边界并保留失败尝试；传参：完整请求及来源上下文；返回：已解析的差量。"""
        if self.cancelled is not None and self.cancelled():
            raise ExecutionCancelled("context compaction cancelled")
        plan = self.summarize(
            ProductionContextBundle("", {**context, "prepared_request": prepared}, ())
        )
        requests.append(plan.request_id)
        if plan.model_error is not None:
            error = plan.model_error
            if error.category in {"context_overflow", "payload_too_large"}:
                raise SummaryProviderOverflow(error.summary)
            if error.category == "cancelled":
                raise ExecutionCancelled(error.summary)
            if error.category == "budget_exhausted":
                raise RunBudgetExceeded(error.summary)
            raise ValueError(
                f"semantic summary model failed: {error.category}: {error.summary}"
            )
        if self.cancelled is not None and self.cancelled():
            raise ExecutionCancelled(
                "context compaction cancelled before accepting model result"
            )
        if plan.final_output is None or plan.run_tools_request is not None:
            raise ValueError(
                "semantic summary must return a delta, not an executable action"
            )
        return parse_delta(plan.final_output)


def _smaller_fragment(
    fragment: list[list[dict[str, object]]],
    remaining: list[list[dict[str, object]]],
) -> tuple[list[list[dict[str, object]]], int]:
    """供应商拒绝后按完整交互或显式正文分页缩小；传参：被拒片段与余量；返回：可继续的材料和组数。"""
    if len(fragment) > 1:
        return [*fragment, *remaining], max(1, len(fragment) // 2)
    left, right = split_source_group(fragment[0])
    return [left, right, *remaining], 1


def _source_spans(groups: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """记录核对模型实际读取的原文范围，不把分页当完整读取；传参：实际片段；返回：每段位置。"""
    return [
        {
            "message_id": message["message_id"],
            "part_index": index,
            **part.get(
                "source_span",
                {"start": 0, "end": len(part["text"]), "total": len(part["text"])},
            ),
        }
        for group in groups
        for message in group
        for index, part in enumerate(cast(list[dict[str, Any]], message["content"]))
        if isinstance(part.get("text"), str)
    ]


def _changed_originals(
    previous: SummaryContent,
    delta: Mapping[str, Any],
    material: CompactionMaterial,
) -> list[list[dict[str, object]]]:
    """修订条目或已分页片段时重读完整依据；传参：前版、差量和冻结会话；返回：辅助原文。"""
    changed = {
        item["entry_id"]
        for action in ("revise", "retire")
        for item in delta.get(action, [])
    }
    ids = {
        source.message_id
        for entry in previous.entries
        if entry.entry_id in changed
        for source in entry.sources
    }
    changed_segments = {item["segment_id"] for item in delta.get("segments", [])}
    ids.update(
        identity
        for segment in previous.segments
        if segment.segment_id in changed_segments
        for identity in segment.message_ids
    )
    if not ids:
        return []
    all_material = replace(material, source=replace(material.source, previous=None))
    return [
        group
        for group in source_groups(all_material)
        if any(str(message["message_id"]) in ids for message in group)
    ]


def _preceding_originals(
    material: CompactionMaterial, first: list[dict[str, object]]
) -> list[dict[str, object]]:
    """保留短句更正所指的紧邻完整交互；传参：冻结材料和本片首组；返回：不计新增覆盖的前文。"""
    all_material = replace(material, source=replace(material.source, previous=None))
    groups = source_groups(all_material)
    first_id = first[0]["message_id"]
    for index, group in enumerate(groups):
        if any(message["message_id"] == first_id for message in group):
            return groups[index - 1] if index else []
    raise ValueError("summary fragment is not part of its frozen source")


@dataclass(frozen=True, slots=True)
class PreparedModelTurn:
    """本次请求材料和来源重建能力；压缩发布后仍从原事实所有者重读。"""

    bundle: ProductionContextBundle
    rebuild: Callable[[], ProductionContextBundle]
    compactor: ContextCompactor | None

    def fit(self, *, force: bool = False) -> ProductionContextBundle:
        """按现有窗口策略拟合本轮材料；传参：是否强制压缩；返回：可派发请求。"""
        if self.compactor is None:
            return self.bundle
        return self.compactor.fit(self.bundle, rebuild=self.rebuild, force=force)


def prepare_model_turn(
    task: str,
    context: RunContext,
    *,
    builder: ProductionContextBuilder,
    registry: ToolRegistry,
    policy: RuntimeToolPolicy,
    runner: ModelRequestRunner,
    extensions: ExtensionExecution,
    cancellation: CancellationToken,
    watchdog: Watchdog,
    data_root: Path,
    progress: ProgressGuard,
    collaboration: CollaborationRuntime | None,
    last_tool_result: RunToolsResult | None = None,
    system_reminder: str | None = None,
) -> PreparedModelTurn | LLMPlan:
    """组合既有材料、结果视图和摘要能力；传参：当前运行及明确依赖；返回：准备结果或策略错误。"""
    toolset_policy = policy.resolve(context)
    if isinstance(toolset_policy, LLMPlan):
        return toolset_policy
    registry.refresh_sources(context.capability_lease)
    rebuild = partial(
        builder.build,
        task=task,
        context=context,
        toolset_policy=policy_to_mapping(toolset_policy),
        tool_registry=registry.snapshot(),
        system_reminder=system_reminder,
        recoverable_error_notice=_recoverable_error_notice(last_tool_result),
        no_progress_observation=progress.model_notice(),
    )
    bundle = rebuild()
    contributions = extensions.context_contributions(context, watchdog)
    if collaboration is not None:
        collaboration_context = collaboration.context_view(context)
        if collaboration_context:
            contributions.append(collaboration_context)
    bundle = replace(
        bundle,
        model_context={**bundle.model_context, "extension_context": contributions},
    )
    prepare = getattr(runner.llm_client, "prepare_request", None)
    if not callable(prepare):
        if task == "/compact":
            raise ContextReadError(
                "context_compaction_unavailable",
                context.storage_task_id,
                ValueError("model client cannot prepare summary requests"),
            )
        return PreparedModelTurn(bundle, rebuild, None)
    views = ToolResultViews(
        data_root,
        task_id=context.material_task_id,
        prepare=partial(prepare, stage="continue" if last_tool_result else "plan"),
    )
    compactor = ContextCompactor(
        SessionCompactionStore(SessionMessageStore(data_root)),
        views.prepare,
        partial(_summary_plan, runner=runner, context=context, watchdog=watchdog),
        cancelled=lambda: cancellation.cancelled,
        proactive=policy.config().get("auto_context_compaction", True) is True,
    )
    if policy.config().get(
        "auto_context_compaction", True
    ) is True and context.trigger.value in {"user", "resume"}:
        from runtime.context_compaction_jobs import ContextCompactionJobs

        compactor.enqueue = partial(
            ContextCompactionJobs(data_root).accept,
            run=context,
            client=runner.llm_client,
        )
    return PreparedModelTurn(bundle, rebuild, compactor)


def _recoverable_error_notice(last_tool_result: RunToolsResult | None) -> str:
    """从上一轮的可恢复错误观测里取出交给模型的原因说明。

    作者：xxx
    时间：2026-09-01 00:00:00
    传参：last_tool_result 为上一轮结果；首轮或普通工具结果时为 None
    返回：形如 "分类: 原码" 的错误说明；非可恢复错误观测返回空串
    """
    # 只认循环自己造的可恢复错误观测，普通工具失败已经作为 tool 结果回灌过了
    if last_tool_result is None or last_tool_result.status != "error":
        return ""
    meta = last_tool_result.meta or {}
    if not meta.get("recoverable_observation"):
        return ""
    notice = last_tool_result.error or ""
    return notice.strip()


def _summary_plan(
    bundle: ProductionContextBundle,
    *,
    runner: ModelRequestRunner,
    context: RunContext,
    watchdog: Watchdog,
) -> LLMPlan:
    """把带身份的辅助调用结果交给现有摘要策略；传参：请求和执行依赖；返回：策略消费的plan。"""
    return runner.invoke_auxiliary(bundle, context=context, watchdog=watchdog).plan


def system_prompt_estimate() -> str:
    from llm.prompt_composer import (
        ARTIFACT_FAST_PATH_INSTRUCTIONS,
        CONVERGENCE_STRATEGY_INSTRUCTIONS,
    )

    return (
        "protocol + native tool schema discipline: "
        f"{ARTIFACT_FAST_PATH_INSTRUCTIONS} "
        f"{CONVERGENCE_STRATEGY_INSTRUCTIONS} "
        "Use the provided native tool schemas as the authority for "
        "tool names and arguments."
    )
