"""Session语义摘要的来源、分支可见性及原子发布。

作者：xxx
时间：2026-09-14 20:00:00
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Sequence

from llm.messages import (
    AgentMessage,
    agent_message_to_mapping,
    validate_message_sequence,
)
from context.summary_entries import (
    SummaryContent,
    SummaryEntry,
    content_from_mapping,
    validate_content,
)
from context.history_segments import validate_segment_audits, validate_segment_coverage
from runtime.cancellation import ExecutionCancelled
from runtime.session_message_store import MaterializedSession, SessionMessageStore
from tasks.ids import new_ulid

SUMMARY_VERSION = 3


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """已发布的摘要；消息ID与摘要分开存放，覆盖原文始终保留在Session中。"""

    summary_id: str
    session_id: str
    branch_entry_id: str
    message_ids: tuple[str, ...]
    first_kept_message_id: str
    source_sha256: str
    text: str
    request_ids: tuple[str, ...]
    previous_summary_id: str | None = None
    new_message_ids: tuple[str, ...] | None = None
    content: SummaryContent | None = None
    compaction_job_id: str | None = None

    def model_view(self) -> dict[str, object]:
        """导出模型所需的摘要及紧凑来源范围；传参：无；返回：引用视图。"""
        return {
            "summary_id": self.summary_id,
            "text": self.text,
            "source": {
                "session_id": self.session_id,
                "branch_entry_id": self.branch_entry_id,
                "first_message_id": self.message_ids[0],
                "last_message_id": self.message_ids[-1],
                "message_count": len(self.message_ids),
                "sha256": self.source_sha256,
            },
            "first_kept_message_id": self.first_kept_message_id,
            "new_message_count": len(self.new_message_ids)
            if self.new_message_ids is not None
            else None,
            "original_check": "recorded"
            if self.content is not None and self.content.checks
            else "not_provided",
            "topic_directory": "available"
            if self.content is not None
            else "not_provided",
            "history_segments": len(self.content.segments)
            if self.content is not None
            else 0,
            "read_action": {
                "tool": "read_history",
                "arguments": {"summary_id": self.summary_id},
            },
            "topic_action": {"tool": "read_history", "arguments": {"view": "topics"}},
            "load_read_action": {
                "tool": "capabilities",
                "arguments": {"action": "load", "name": "read_history"},
            },
        }


@dataclass(frozen=True, slots=True)
class SummarySource:
    """摘要开始前取得的分支和覆盖范围；previous指向此前生效的摘要。"""

    view: MaterializedSession
    covered_count: int
    previous: SessionSummary | None = None


class SessionCompactionStore:
    """通过Session共享写锁发布摘要，不另存或改写原消息。"""

    def __init__(self, messages: SessionMessageStore) -> None:
        """注入原消息owner；传参：会话存储；返回：无。"""
        self.messages = messages
        self.database = messages.database

    def current(self, view: MaterializedSession) -> SessionSummary | None:
        """选择当前分支可用的最新摘要；传参：会话快照；返回：摘要或尚无摘要。"""
        ancestors = {entry.entry_id for entry in view.entries}
        with self.database.snapshot() as source:
            rows = source.list("session_compaction", session_id=view.session_id)
            records = tuple(
                _parse_record(row)
                for row in sorted(rows, key=lambda row: row["summary_id"], reverse=True)
            )
        for record in records:
            if record.session_id != view.session_id:
                raise ValueError("context summary belongs to another session")
            if record.branch_entry_id not in ancestors:
                continue
            count = len(record.message_ids)
            if (
                count >= len(view.messages)
                or view.messages[count].message_id != record.first_kept_message_id
            ):
                continue
            covered = view.messages[:count]
            if tuple(message.message_id for message in covered) != record.message_ids:
                continue
            if source_digest(covered) != record.source_sha256:
                raise ValueError("context summary source content changed")
            return record
        return None

    def publish(
        self,
        source: SummarySource,
        text: str,
        *,
        request_ids: tuple[str, ...],
        content: SummaryContent | None = None,
        cancelled: Callable[[], bool] | None = None,
        compaction_job_id: str | None = None,
    ) -> SessionSummary:
        """只在来源仍可核对时原子发布摘要；传参：来源、摘要、真实模型请求ID；返回：已保存记录。"""
        if (
            (not text.strip() and content is None)
            or not request_ids
            or any(not isinstance(value, str) or not value for value in request_ids)
        ):
            raise ValueError("context summary requires text and model request evidence")
        view, count = source.view, source.covered_count
        if not view.leaf_id or not 0 < count < len(view.messages):
            raise ValueError(
                "context summary must cover a prefix and retain a continuation"
            )
        if source.previous is not None and count < len(source.previous.message_ids):
            raise ValueError("summary coverage cannot move backwards")
        covered = view.messages[:count]
        validate_message_sequence(covered)
        if content is not None:
            _validate_content_source(source, content, request_ids=request_ids)
            if content.render() != text:
                raise ValueError("summary text differs from its structured entries")
        record = SessionSummary(
            summary_id=f"summary-{new_ulid()}",
            session_id=view.session_id,
            branch_entry_id=view.leaf_id,
            message_ids=tuple(message.message_id for message in covered),
            first_kept_message_id=view.messages[count].message_id,
            source_sha256=source_digest(covered),
            text=text,
            request_ids=request_ids,
            previous_summary_id=source.previous.summary_id if source.previous else None,
            new_message_ids=tuple(
                message.message_id
                for message in covered[
                    len(source.previous.message_ids) if source.previous else 0 :
                ]
            ),
            content=content,
            compaction_job_id=compaction_job_id,
        )
        with self.messages.write_lock(view.session_id):
            current = self.messages.materialize(view.session_id)
            if view.leaf_id not in {entry.entry_id for entry in current.entries}:
                raise ValueError(
                    "context summary source branch changed before publication"
                )
            if source_digest(current.messages[:count]) != record.source_sha256:
                raise ValueError("context summary source changed before publication")
            if current.messages[count].message_id != record.first_kept_message_id:
                raise ValueError(
                    "context summary continuation changed before publication"
                )
            active = self.current(current)
            if (active.summary_id if active else None) != record.previous_summary_id:
                raise ValueError("context summary was superseded during generation")
            if cancelled is not None and cancelled():
                raise ExecutionCancelled(
                    "context compaction cancelled before publication"
                )
            with self.database.transaction() as batch:
                batch.put(
                    "session_compaction",
                    record.summary_id,
                    {"schema_version": SUMMARY_VERSION, **asdict(record)},
                    session_id=record.session_id,
                )
        return record

    def read(self, summary_id: str, view: MaterializedSession) -> SessionSummary:
        """按身份读取当前分支上的摘要并核对原文；传参：摘要身份和分支；返回：可验证记录。"""
        if not re.fullmatch(r"summary-[A-Za-z0-9_-]+", summary_id):
            raise ValueError("invalid summary identity")
        with self.database.snapshot() as source:
            row = source.get("session_compaction", summary_id)
            if row is None:
                raise FileNotFoundError(summary_id)
            record = _parse_record(row)
        count = len(record.message_ids)
        if record.summary_id != summary_id or record.session_id != view.session_id:
            raise ValueError("summary identity does not match its source")
        if record.branch_entry_id not in {entry.entry_id for entry in view.entries}:
            raise ValueError("summary does not belong to the current branch")
        if (
            tuple(message.message_id for message in view.messages[:count])
            != record.message_ids
        ):
            raise ValueError("summary source range changed")
        if source_digest(view.messages[:count]) != record.source_sha256:
            raise ValueError("context summary source content changed")
        return record

    def chain(self, view: MaterializedSession) -> tuple[SessionSummary, ...]:
        """沿当前已发布快照回查增量目录；传参：分支；返回：从新到旧的摘要链。"""
        records = []
        current = self.current(view)
        seen: set[str] = set()
        while current is not None:
            if current.summary_id in seen:
                raise ValueError("summary chain contains a cycle")
            seen.add(current.summary_id)
            records.append(current)
            current = (
                self.read(current.previous_summary_id, view)
                if current.previous_summary_id
                else None
            )
        return tuple(records)

    def effective_requirements(
        self, view: MaterializedSession
    ) -> tuple[SummaryEntry, ...]:
        """会话更正不随聊天分支回退；参数：会话身份；返回：最新真实用户证据对应的有效要求。"""
        with self.database.snapshot() as source:
            rows = source.list("session_compaction", session_id=view.session_id)
        positions = {
            entry.message.message_id: index
            for index, entry in enumerate(self.messages.read_entries(view.session_id))
            if entry.message is not None
        }
        latest: dict[str, SummaryEntry] = {}
        evidence_order: dict[str, tuple[int, int]] = {}
        for row in sorted(rows, key=lambda item: item["summary_id"]):
            record = _parse_record(row)
            for entry in record.content.entries if record.content is not None else ():
                if entry.kind != "requirement":
                    continue
                user_positions = [
                    positions[citation.message_id]
                    for citation in entry.sources
                    if citation.source_kind == "user_input"
                ]
                order = (max(user_positions), entry.revision)
                if (
                    entry.entry_id not in latest
                    or order > evidence_order[entry.entry_id]
                ):
                    latest[entry.entry_id] = entry
                    evidence_order[entry.entry_id] = order
        return tuple(entry for entry in latest.values() if entry.active)


def source_digest(messages: Sequence[AgentMessage]) -> str:
    """计算覆盖原文的内容版本；传参：按顺序的消息；返回：SHA256。"""
    digest = hashlib.sha256(b"[")
    for index, message in enumerate(messages):
        if index:
            digest.update(b",")
        text = json.dumps(
            agent_message_to_mapping(message),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(text.encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def _validate_content_source(
    source: SummarySource, content: SummaryContent, *, request_ids: tuple[str, ...]
) -> None:
    """发布前核对条目和目录都来自本次冻结原文；传参：来源、候选和真实请求；返回：无。"""
    from context.compaction import original_sources

    validate_content(content)
    originals = original_sources(source)
    for entry in content.entries:
        for citation in entry.sources:
            original = originals.get(citation.message_id)
            if (
                original is None
                or citation.quote not in original["text"]
                or citation.source_kind != original["source_kind"]
            ):
                raise ValueError("summary entry citation differs from its original")
            if citation.result_status != original["result_status"]:
                raise ValueError("summary entry changed its source result status")
    if any(
        identity not in originals
        for topic in content.topics
        for identity in topic.message_ids
    ):
        raise ValueError("summary topic points outside its covered original messages")
    if any(check.get("request_id") not in request_ids for check in content.checks):
        raise ValueError("summary check has no matching model request")
    if content.segments:
        begin = (
            len(source.previous.message_ids)
            if source.previous and source.previous.content
            else 0
        )
        validate_segment_coverage(
            content.segments, source.view.messages[begin : source.covered_count]
        )
        validate_segment_audits(
            content.segments,
            content.checks,
            source.view.messages[begin : source.covered_count],
        )


def _parse_record(value: object) -> SessionSummary:
    """严格解析已保存摘要，损坏记录不能被静默跳过；传参：JSON值；返回：摘要。"""
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] not in {2, SUMMARY_VERSION}
    ):
        raise ValueError("unsupported context summary format")
    for key in (
        "summary_id",
        "session_id",
        "branch_entry_id",
        "first_kept_message_id",
        "source_sha256",
    ):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"context summary {key} must be non-empty text")
    for key in ("message_ids", "request_ids"):
        if not isinstance(value.get(key), list) or not value[key]:
            raise ValueError(f"context summary {key} must be a non-empty list")
        if any(not isinstance(item, str) or not item.strip() for item in value[key]):
            raise ValueError(f"context summary {key} contains an invalid identity")
        if len(set(value[key])) != len(value[key]):
            raise ValueError(f"context summary {key} contains duplicate identities")
    previous = value.get("previous_summary_id")
    if previous is not None and (not isinstance(previous, str) or not previous.strip()):
        raise ValueError("invalid previous summary identity")
    content = (
        content_from_mapping(value["content"])
        if value.get("content") is not None
        else None
    )
    text = value.get("text")
    if not isinstance(text, str) or (content is None and not text.strip()):
        raise ValueError("context summary text must be present")
    if content is not None and content.render() != text:
        raise ValueError("summary rendered text differs from its structured entries")
    added = value.get("new_message_ids")
    if added is not None and (
        not isinstance(added, list)
        or any(
            not isinstance(item, str) or item not in value["message_ids"]
            for item in added
        )
    ):
        raise ValueError(
            "summary new message identities are outside its covered source"
        )
    return SessionSummary(
        summary_id=value["summary_id"],
        session_id=value["session_id"],
        branch_entry_id=value["branch_entry_id"],
        message_ids=tuple(value["message_ids"]),
        first_kept_message_id=value["first_kept_message_id"],
        source_sha256=value["source_sha256"],
        text=text,
        request_ids=tuple(value["request_ids"]),
        previous_summary_id=previous,
        new_message_ids=tuple(added) if added is not None else None,
        content=content,
        compaction_job_id=value.get("compaction_job_id"),
    )
