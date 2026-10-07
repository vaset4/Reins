from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from memory.safety_scan import SafetyMatch, scan as safety_scan
from memory.store import MEMORY_STATE_ACTIVE, MemoryDetails, MemoryStore

_LOG = logging.getLogger(__name__)


REVIEW_AUTO = "auto"
REVIEW_AUTO_WITH_NOTIFY = "auto_with_notify"
REVIEW_MODES = {REVIEW_AUTO, REVIEW_AUTO_WITH_NOTIFY}
DEFAULT_REVIEW_MODE = REVIEW_AUTO_WITH_NOTIFY


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    memory_id: str | None
    state: str | None
    review_mode: str
    skipped: bool = False
    notice: str = ""
    blocked_by_safety_scan: bool = False
    safety_matches: tuple[SafetyMatch, ...] = ()
    blocked_by_conflict: bool = False
    conflict_with: str | None = None


class MemoryWriter:
    def __init__(
        self,
        data_root: Path | str,
        *,
        config_path: Path | str | None = None,
    ) -> None:
        self._store = MemoryStore(data_root)
        self._config_path = Path(config_path) if config_path is not None else None

    def write_memory(
        self,
        type: str,
        content: str,
        tags: list[str],
        *,
        applicable_task_tags: list[str] | None = None,
        memory_id: str | None = None,
        details: MemoryDetails | None = None,
        change_id: str | None = None,
    ) -> MemoryWriteResult:
        """扫描安全性并保存明确范围的知识；传参：内容、标签与来源；返回：保存或拒写事实。"""
        review_mode = self.review_mode()

        rejected = _safety_rejection(content, review_mode=review_mode)
        if rejected is not None:
            return rejected

        # 【记忆】【重复提交】只拦同范围同主体的完全相同正文，文字近似不能证明事实相同
        conflict_id = self._detect_conflict(
            type, content, details=details, memory_id=memory_id
        )
        if conflict_id is not None:
            _LOG.info(
                "【记忆】【写入冲突】同范围正文与已有记忆 %s 相同，未重复保存；内容摘要=%s",
                conflict_id,
                content[:60],
            )
            return MemoryWriteResult(
                memory_id=None,
                state=None,
                review_mode=review_mode,
                skipped=True,
                blocked_by_conflict=True,
                conflict_with=conflict_id,
                notice=f"memory skipped; duplicates existing memory {conflict_id}",
            )

        state = MEMORY_STATE_ACTIVE
        memory_id = self._store.create_memory(
            type,
            content,
            tags,
            applicable_task_tags=applicable_task_tags,
            memory_id=memory_id,
            state=state,
            details=details,
            change_id=change_id,
        )
        notice = (
            "memory saved automatically; notify user"
            if review_mode == REVIEW_AUTO_WITH_NOTIFY
            else "memory saved"
        )
        return MemoryWriteResult(memory_id, state, review_mode, notice=notice)

    def review_mode(self) -> str:
        if self._config_path is None or not self._config_path.is_file():
            return DEFAULT_REVIEW_MODE
        data = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
        mode = ""
        if isinstance(data, dict):
            memory_config = data.get("memory", {})
            if isinstance(memory_config, dict):
                mode = str(memory_config.get("review_mode", ""))
        # manual 依赖的人工审核面从未实现过，已随本档退役；若在此静默回落默认值，
        # 配了它的用户会以为记忆在等自己批准、实际早已全自动落库——
        # 「以为有闸门其实没有」是最坏的一种失败，必须响
        if mode == "manual":
            raise ValueError(
                f"memory.review_mode: 'manual' has been retired; "
                f"remove it from {self._config_path} "
                f"(valid: {', '.join(sorted(REVIEW_MODES))})"
            )
        return mode if mode in REVIEW_MODES else DEFAULT_REVIEW_MODE

    def _detect_conflict(
        self,
        type: str,
        content: str,
        *,
        details: MemoryDetails | None,
        memory_id: str | None,
    ) -> str | None:
        """检测同一事实的原样重复，不承担语义合并；传参：正文和事实身份；返回：重复 ID 或空。"""
        context = details or MemoryDetails()
        fields = (
            "kind",
            "scope",
            "subject",
            "fact_key",
            "archive_ref",
            "exact_fields",
            "observed_at",
            "expires_at",
        )
        replaced = {item.memory_id for item in context.supersedes}
        for existing in self._store.list_memories(state=MEMORY_STATE_ACTIVE, type=type):
            if existing.memory_id == memory_id or existing.memory_id in replaced:
                continue
            same_fact = all(
                getattr(existing.details, key) == getattr(context, key)
                for key in fields
            )
            if same_fact and existing.content == content.strip():
                return existing.memory_id
        return None

    def close(self) -> None:
        """释放本写者的索引连接；传参：无；返回：无。"""
        self._store.close()


def _safety_rejection(content: str, *, review_mode: str) -> MemoryWriteResult | None:
    """在任何原件写入前暴露敏感内容拒写原因；传参：正文与记录策略；返回：拒写事实或空。"""
    result = safety_scan(content)
    if result.is_safe:
        return None
    return MemoryWriteResult(
        memory_id=None,
        state=None,
        review_mode=review_mode,
        skipped=True,
        blocked_by_safety_scan=True,
        safety_matches=result.matches,
        notice="memory blocked by safety scan",
    )


__all__ = [
    "DEFAULT_REVIEW_MODE",
    "MemoryWriteResult",
    "MemoryWriter",
    "REVIEW_AUTO",
    "REVIEW_AUTO_WITH_NOTIFY",
]
