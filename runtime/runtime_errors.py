"""既有运行故障证据的共同序列化；作者：xxx；时间：2026-09-28 18:00:00。"""

from runtime.run_evidence import RunEvidenceStore
from runtime.types import RunContext


def record_runtime_error(
    evidence: RunEvidenceStore,
    context: RunContext,
    *,
    category: str,
    message: str,
    stage: str,
    evidence_path: str = "",
    raw_response_path: str = "",
    meta: dict[str, object] | None = None,
    recovery_action: str = "none",
    attempt_count: int = 0,
    budget_total: int = 0,
    budget_remaining: int = 0,
    wait_seconds: float = 0.0,
    final_outcome: str = "",
    observation_fields: dict[str, object] | None = None,
) -> str:
    """保存实际运行失败及其请求证据引用；传参：证据写者、运行与故障字段；返回：错误记录路径。"""
    error_payload: dict[str, object] = {
        "category": category,
        "message": message,
        "stage": stage,
        "segment_id": context.segment_id,
        "task_id": context.task_id,
        "focus_task_id": context.focus_task_id,
        "compatibility_task_id": context.compatibility_task_id,
        "evidence_path": evidence_path,
        "raw_response_path": raw_response_path,
        "meta": meta or {},
    }
    if recovery_action != "none":
        error_payload["recovery_action"] = recovery_action
    if attempt_count > 0:
        error_payload["attempt_count"] = attempt_count
    if budget_total > 0:
        error_payload["budget_total"] = budget_total
        error_payload["budget_remaining"] = budget_remaining
    if wait_seconds > 0:
        error_payload["wait_seconds"] = round(wait_seconds, 2)
    if final_outcome:
        error_payload["final_outcome"] = final_outcome
    if observation_fields:
        error_payload.update(observation_fields)
    return evidence.append_error(
        session_id=context.session_id,
        run_id=context.run_id,
        error=error_payload,
    )
