from __future__ import annotations

import hashlib
from typing import Mapping

from context.token_estimate import estimate_tokens
from llm.model_request import PromptSection, StablePromptSnapshot
from runtime.persistence import RuntimeStore
from tasks.ids import utc_now


def record_stable_prompt_snapshot(
    sections: tuple[PromptSection, ...],
    *,
    model_context: Mapping[str, object],
    policy_version: str,
    tool_strategy_hash: str,
) -> StablePromptSnapshot:
    """记录稳定提示的复用版本；参数：提示分段、会话与工具策略；返回：本次冻结摘要，不重复保存提示全文。"""
    stable_text = _stable_text(sections)
    digest = _prompt_hash(stable_text)
    base = _snapshot(
        digest=digest,
        path="",
        reused=False,
        invalidation_reason=None,
        policy_version=policy_version,
        tool_strategy_hash=tool_strategy_hash,
        stable_text=stable_text,
        sections=sections,
    )
    data_root = str(model_context.get("data_root", "")).strip()
    session_id = str(model_context.get("session_id", "")).strip()
    if not data_root or not session_id:
        return base
    database = RuntimeStore(data_root)
    # 【模型证据】【提示复用】1. 准备只核对已采用版本，实际发送后才推进复用元数据
    with database.snapshot() as source:
        row = source.get("prompt_snapshot", session_id)
        previous = row or {}
        previous_hash, previous_policy, previous_strategy = (
            str(previous.get(key, ""))
            for key in ("hash", "policy_version", "tool_strategy_hash")
        )
        previous_status = "ok" if row else "missing"
    reused = (
        previous_status == "ok"
        and previous_hash == digest
        and previous_policy == policy_version
        and previous_strategy == tool_strategy_hash
    )
    reason = _invalidation_reason(
        status=previous_status,
        previous_hash=previous_hash,
        previous_policy=previous_policy,
        previous_strategy=previous_strategy,
        digest=digest,
        policy_version=policy_version,
        tool_strategy_hash=tool_strategy_hash,
    )
    return _snapshot(
        digest=digest,
        path="",
        reused=reused,
        invalidation_reason=reason,
        policy_version=policy_version,
        tool_strategy_hash=tool_strategy_hash,
        stable_text=stable_text,
        sections=sections,
    )


def adopt_prompt_snapshot(
    snapshot: StablePromptSnapshot, *, data_root: str, session_id: str
) -> None:
    """发送时推进本地提示版本；参数：实际采用快照和归属；返回：无。"""
    metadata = {
        "hash": snapshot.hash,
        "policy_version": snapshot.policy_version,
        "tool_strategy_hash": snapshot.tool_strategy_hash,
        "updated_at": utc_now(),
    }
    with RuntimeStore(data_root).transaction() as batch:
        batch.put("prompt_snapshot", session_id, metadata, session_id=session_id)


def _stable_text(sections: tuple[PromptSection, ...]) -> str:
    """提取本次稳定提示正文用于摘要；参数：分段；返回：有序全文。"""
    return "\n\n".join(item.content for item in sections if item.layer == "stable")


def _prompt_hash(stable_text: str) -> str:
    """计算稳定内容身份；参数：稳定正文；返回：带算法前缀的摘要。"""
    digest = hashlib.sha256(stable_text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _invalidation_reason(
    *,
    status: str,
    previous_hash: str,
    previous_policy: str,
    previous_strategy: str,
    digest: str,
    policy_version: str,
    tool_strategy_hash: str,
) -> str | None:
    """解释当前版本为何失效；参数：新旧摘要与策略；返回：真实失效原因或无变化。"""
    if (
        status == "ok"
        and previous_hash == digest
        and previous_policy == policy_version
        and previous_strategy == tool_strategy_hash
    ):
        return None
    if status == "missing":
        return "missing_snapshot"
    if previous_policy != policy_version:
        return "policy_version_changed"
    if previous_strategy != tool_strategy_hash:
        return "tool_strategy_changed"
    return "hash_changed"


def _snapshot(
    *,
    digest: str,
    path: str,
    reused: bool,
    invalidation_reason: str | None,
    policy_version: str,
    tool_strategy_hash: str,
    stable_text: str,
    sections: tuple[PromptSection, ...],
) -> StablePromptSnapshot:
    """构造本次只读版本摘要；参数：身份、策略及内容度量；返回：冻结提示版本。"""
    return StablePromptSnapshot(
        hash=digest,
        path=path,
        reused=reused,
        invalidation_reason=invalidation_reason,
        policy_version=policy_version,
        tool_strategy_hash=tool_strategy_hash,
        tokens_est=estimate_tokens(stable_text),
        section_count=sum(1 for item in sections if item.layer == "stable"),
        chars=len(stable_text),
    )
