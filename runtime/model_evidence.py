"""按逻辑请求和真实尝试保存模型证据。

作者：xxx
时间：2026-09-13 20:00:00
"""

from __future__ import annotations

from llm.types import LLMPlan, ModelAttemptEvent, usage_to_mapping
from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore
from runtime.trace_level import get_trace_level, should_write_parsed_plan
from runtime.types import RunContext

MODEL_EVIDENCE_VERSION = 2


class ModelEvidenceWriter:
    """复用运行事实和证据存储，独占模型请求、尝试与解析记录的写入。

    传参：evidence/facts 为调用方注入的现有存储；返回：证据写入器
    """

    def __init__(self, evidence: RunEvidenceStore, facts: RunFactStore) -> None:
        """接收既有存储实例；传参：证据与事实存储；返回：无。"""
        self._evidence = evidence
        self._facts = facts

    def record_attempt(self, context: RunContext, attempt: ModelAttemptEvent) -> None:
        """发送前保存请求，结束后保存真实结果，持久化失败直接向调用方暴露。

        传参：context 为当前运行；attempt 为一次尝试边界；返回：无
        """
        identity = _identity(context, attempt.request_id)
        row: dict[str, object] = {
            **identity,
            "attempt_id": attempt.attempt_id,
            "attempt_index": attempt.attempt_index,
            "provider": attempt.provider,
            "model": attempt.model,
            "started_at": attempt.started_at,
            "input_ids": list(attempt.input_ids),
            "api_family": attempt.api_family,
        }
        if attempt.phase == "started":
            row["event"] = "llm:attempt_started"
            row["request_path"] = self._write_attempt(
                context,
                attempt.attempt_id,
                "attempt_request",
                {
                    **row,
                    "request": dict(attempt.request),
                    "sources": dict(attempt.sources),
                },
            )
        else:
            row.update(
                {
                    "event": "llm:attempt",
                    "elapsed_ms": attempt.elapsed_ms,
                    "success": attempt.error is None,
                    "error_category": attempt.error.category if attempt.error else None,
                    "usage": usage_to_mapping(attempt.usage),
                }
            )
            row["response_path"] = self._write_attempt(
                context,
                attempt.attempt_id,
                "attempt_response",
                {
                    **row,
                    "response": dict(attempt.response),
                },
            )
        self._facts.append(row)

    def record_request(
        self, context: RunContext, *, request_id: str, request_index: int
    ) -> None:
        """在发出逻辑请求前提交关联身份；传参：运行、请求身份和序号；返回：无。"""
        self._evidence.record_request(
            session_id=context.session_id,
            run_id=context.run_id,
            request_id=request_id,
            request_index=request_index,
        )
        self._facts.append(
            {
                **_identity(context, request_id),
                "event": "llm:request",
                "request_index": request_index,
            }
        )

    def write_plan(
        self,
        context: RunContext,
        plan: LLMPlan,
        *,
        request_index: int,
    ) -> dict[str, object]:
        """保存逻辑请求的最终视图，并引用它包含的每次真实尝试。

        传参：context/plan 为运行与模型结果；request_index 为独立请求序号；返回：证据路径
        """
        identity = {
            **_identity(context, plan.request_id),
            "request_index": request_index,
            "protocol_mode": plan.protocol_mode,
            "attempt_ids": [attempt.attempt_id for attempt in plan.model_attempts],
        }
        # 【模型证据】【逻辑请求】逻辑记录只引用真实尝试，三个诊断级别均可查看已捕获正文
        records = self._evidence.attempt_references(plan.request_id)
        request_refs = [row["request_ref"] for row in records]
        response_refs = [row["response_ref"] for row in records if row["response_ref"]]
        request_ref = request_refs[-1] if request_refs else "(no recorded attempt)"
        response_ref = response_refs[-1] if response_refs else "(no recorded attempt)"
        parsed_ref = "(not written, trace_level == off)"
        if should_write_parsed_plan(get_trace_level()):
            parsed_ref = self._evidence.write_record(
                session_id=context.session_id,
                run_id=context.run_id,
                kind="model_plan",
                source_id=plan.request_id or str(request_index),
                payload={
                    **identity,
                    "attempt_request_refs": request_refs,
                    "attempt_response_refs": response_refs,
                    **_parsed_plan(plan, response_ref),
                },
            )
        return {
            "model_request": request_ref,
            "model_response": response_ref,
            "parsed_plan": parsed_ref,
        }

    def record_usage(self, context: RunContext, plan: LLMPlan) -> None:
        """保存逻辑请求的缓存和裁剪指标，用请求身份关联而不借用工具次数。

        传参：context/plan 为当前运行和模型结果；返回：无
        """
        identity = _identity(context, plan.request_id)
        obs = plan.observation
        if obs is not None and (
            obs.cache_read_input_tokens is not None
            or obs.cache_creation_input_tokens is not None
        ):
            self._facts.append(
                {
                    **identity,
                    "event": "llm:cache_usage",
                    "model_call_id": plan.request_id,
                    "model_call_id_source": "request_id",
                    "cache_read_input_tokens": obs.cache_read_input_tokens,
                    "cache_creation_input_tokens": obs.cache_creation_input_tokens,
                }
            )
        if plan.trim_delta is not None:
            self._facts.append({**plan.trim_delta, **identity, "event": "trim:delta"})

    def _write_attempt(
        self,
        context: RunContext,
        attempt_id: str,
        kind: str,
        payload: dict[str, object],
    ) -> str:
        """保存一次真实尝试，不重复逻辑正文；传参：运行、尝试身份、阶段及诊断；返回：证据引用。"""
        return self._evidence.write_record(
            session_id=context.session_id,
            run_id=context.run_id,
            kind=kind,
            source_id=attempt_id,
            payload=payload,
        )


def _identity(context: RunContext, request_id: str) -> dict[str, object]:
    """记录请求发起时的归属；传参：运行与请求身份；返回：可持久化身份字段。"""
    return {
        "schema_version": MODEL_EVIDENCE_VERSION,
        "session_id": context.session_id,
        "run_id": context.run_id,
        "segment_id": context.segment_id,
        "task_id": context.task_id,
        "focus_task_id": context.focus_task_id,
        "compatibility_task_id": context.compatibility_task_id,
        "request_id": request_id,
        "input_message_id": context.payload.get("input_message_id"),
        "parent_session_id": context.parent_session_id,
        "parent_run_id": context.parent_run_id,
        "budget_run_id": context.budget_run_id,
    }


def _parsed_plan(plan: LLMPlan, response_path: str) -> dict[str, object]:
    """生成解析结果视图，协议错误保留其来源；传参：计划和响应引用；返回：新映射。"""
    payload: dict[str, object] = {
        "success": plan.model_error is None,
        "has_final": plan.final_output is not None,
        "has_run_tools": plan.run_tools_request is not None,
        "raw_response_path": response_path,
    }
    if plan.model_error is not None:
        error = plan.model_error
        payload["error"] = {
            "category": error.category,
            "message": error.summary,
            "stage": error.stage,
            "raw_summary": error.raw_summary,
            "raw_response_path": response_path,
        }
    return payload
