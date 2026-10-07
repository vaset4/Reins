"""【存储】【文件记录】固定领域类型、不可变来源与显式正文引用。

作者：xxx
时间：2026-09-30 15:00:00
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, BinaryIO
from collections.abc import Mapping
from pathlib import Path

RECORD_KINDS = frozenset(
    {
        "workspace",
        "session_workspace",
        "session",
        "session_entry",
        "session_state",
        "run_fact",
        "tool_operation",
        "ledger",
        "collaboration",
        "background_session",
        "background_input",
        "task",
        "task_summary",
        "task_journal",
        "task_reflection",
        "schedule",
        "schedule_occurrence",
        "notification",
        "run_evidence",
        "model_requests",
        "artifact_records",
        "skill_usage",
        "skill_record",
        "skill_revision",
        "skill_outcome",
        "memory_usage",
        "memory_identity",
        "session_compaction",
        "task_todo",
        "prompt_snapshot",
        "file_restore_point",
        "file_restore_plan",
        "file_restore_operation",
        "context_baseline",
        "context_material_selection",
        "knowledge_maintenance",
        "context_compaction_job",
    }
)
SCHEDULE_KINDS = frozenset({"schedule", "schedule_occurrence", "notification"})
FORMAT_VERSION = 5
READABLE_FORMAT_VERSIONS = frozenset({3, 4, FORMAT_VERSION})
INDEX_VERSION = 2
PAYLOAD_MIN_BYTES = 256


class SourceCorruptionError(ValueError):
    """已经发布的原件丢失、截断或身份损坏。"""


class RecordConflictError(ValueError):
    """领域期望修订已经改变。"""


@dataclass(frozen=True, slots=True)
class PreparedPayload:
    """锁外已冻结的正文与归属，提交时只补充轻量业务身份。"""

    payload: dict[str, Any]
    references: tuple[dict[str, Any], ...]
    owner: str

    def with_fields(self, fields: Mapping[str, Any]) -> PreparedPayload:
        """补充不会覆盖正文的身份字段；参数：新增字段；返回：独立候选载荷。"""
        if set(fields) & set(self.payload):
            raise ValueError("metadata cannot replace prepared payload fields")
        return PreparedPayload({**fields, **self.payload}, self.references, self.owner)


@dataclass(frozen=True, slots=True)
class PreparedContent:
    """锁外完整验证且持有固定只读句柄的原件，只在准备上下文存活期间可提交。"""

    path: Path
    size: int
    sha256: str
    revision: tuple[int, int]
    identity: tuple[int, int]
    handle: BinaryIO = field(compare=False, hash=False, repr=False)


def json_bytes(value: object) -> bytes:
    """稳定编码文件内容；参数：JSON值；返回：UTF-8字节，不接受NaN。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    """计算原件摘要；参数：文件字节；返回：SHA256。"""
    return hashlib.sha256(value).hexdigest()


def record_key(*parts: str) -> str:
    """编码领域复合身份；参数：原始身份段；返回：无拼接歧义的JSON编号。"""
    return json_bytes(parts).decode("utf-8")


@dataclass(frozen=True, slots=True)
class ContentReference:
    """正文文件的显式身份，不根据用户文本猜测附件。"""

    path: str
    sha256: str
    size: int
    media_type: str = "application/octet-stream"

    def to_mapping(self) -> dict[str, Any]:
        """生成引用元数据；参数：无；返回：新的JSON对象。"""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "media_type": self.media_type,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ContentReference:
        """解析并核对引用；参数：文件元数据；返回：类型引用，坏字段明确失败。"""
        if (
            not isinstance(value.get("path"), str)
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64
            or type(value.get("size")) is not int
            or value["size"] < 0
        ):
            raise SourceCorruptionError("invalid content reference")
        return cls(
            value["path"],
            value["sha256"],
            value["size"],
            value.get("media_type", "application/octet-stream"),
        )


@dataclass(frozen=True, slots=True)
class RecordLocation:
    """一条已提交领域事件的位置和完整性证据。"""

    path: str
    offset: int
    length: int
    sha256: str
    sequence: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class StoredRecord:
    """尚未展开大正文的领域记录；payload中的引用位置由独立references表指定。"""

    kind: str
    record_id: str
    revision: int
    payload: dict[str, Any]
    references: tuple[dict[str, Any], ...]
    workspace_id: str | None = None
    session_id: str | None = None
    deleted: bool = False
    location: RecordLocation | None = None

    def to_mapping(self) -> dict[str, Any]:
        """生成普通可读领域事件；参数：无；返回：不含正文副本的映射。"""
        return {
            "kind": self.kind,
            "record_id": self.record_id,
            "revision": self.revision,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "deleted": self.deleted,
            "payload": self.payload,
            "references": list(self.references),
        }

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any], location: RecordLocation
    ) -> StoredRecord:
        """解析提交中的记录；参数：JSON及来源；返回：校验后的领域事件。"""
        if (
            value.get("kind") not in RECORD_KINDS
            or not isinstance(value.get("record_id"), str)
            or type(value.get("revision")) is not int
            or value["revision"] < 1
            or not isinstance(value.get("payload"), dict)
            or not isinstance(value.get("references"), list)
            or type(value.get("deleted")) is not bool
        ):
            raise SourceCorruptionError(
                f"invalid record: {location.path}@{location.offset}"
            )
        return cls(
            value["kind"],
            value["record_id"],
            value["revision"],
            value["payload"],
            tuple(value["references"]),
            value.get("workspace_id"),
            value.get("session_id"),
            value["deleted"],
            location,
        )
