"""记忆原文的版本、适用范围与来源合同。

作者：xxx
时间：2026-09-15 05:40:00
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping

import yaml

MEMORY_FORMAT = 2
MEMORY_STATE_DRAFT = "draft"
MEMORY_STATE_ACTIVE = "active"
MEMORY_STATE_ARCHIVED = "archived"
MEMORY_STATE_WITHDRAWN = "withdrawn"
MEMORY_TYPES = {"rule", "fact", "preference", "experience", "lesson"}
MEMORY_STATES = {
    MEMORY_STATE_DRAFT,
    MEMORY_STATE_ACTIVE,
    MEMORY_STATE_ARCHIVED,
    MEMORY_STATE_WITHDRAWN,
}
MEMORY_KINDS = {"fact", "note", "archive"}
SOURCE_KINDS = {
    "user_input",
    "tool_result",
    "model_inference",
    "import",
    "unspecified",
    "external_edit",
}


@dataclass(frozen=True, slots=True)
class MemorySource:
    """记录可回到原输入或操作的引用，kind 描述证据性质，不代表内容已被核验。"""

    kind: str
    reference: str = ""
    session_id: str = ""
    run_id: str = ""
    observed_at: str | None = None
    message_id: str = ""
    content_sha256: str = ""
    result_status: str = ""


@dataclass(frozen=True, slots=True)
class MemoryReplacement:
    """保存一次跨记录替代的冻结目标；传参：旧记录、版本与理由；返回：不可变提交关系。"""

    memory_id: str
    version: str
    reason: str


@dataclass(frozen=True, slots=True)
class MemoryDetails:
    """表达结论的主体、字段和范围；archive_ref 与 exact_fields 保存原件引用和精确字段。"""

    kind: str = "fact"
    scope: str = "global"
    subject: str = ""
    fact_key: str = ""
    sources: tuple[MemorySource, ...] = ()
    observed_at: str | None = None
    expires_at: str | None = None
    archive_ref: str | None = None
    exact_fields: dict[str, str] = field(default_factory=dict)
    supersedes: tuple[MemoryReplacement, ...] = ()


@dataclass(frozen=True, slots=True)
class Memory:
    """保存一个内容版本；使用统计独立存放，不能改变内容版本或核验事实。"""

    memory_id: str
    type: str
    state: str
    content: str
    tags: list[str] = field(default_factory=list)
    applicable_task_tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str | None = None
    last_verified_at: str | None = None
    format_version: int = MEMORY_FORMAT
    version: str = ""
    revision: int = 1
    previous_version: str | None = None
    change_id: str | None = None
    reason: str = "created"
    details: MemoryDetails = field(default_factory=MemoryDetails)
    verification: tuple[MemorySource, ...] = ()
    legacy_last_verified_at: str | None = None


def validate_details(details: MemoryDetails) -> None:
    """校验知识边界而不猜测事实含义；传参：内容范围与来源；返回：无，非法内容明确报错。"""
    if details.kind not in MEMORY_KINDS:
        raise ValueError(f"invalid memory kind: {details.kind}")
    parts = details.scope.split(":", 1)
    if details.scope not in {"global", "unspecified"} and (
        len(parts) != 2
        or parts[0] not in {"session", "goal", "project"}
        or not parts[1].strip()
    ):
        raise ValueError(
            "memory scope must be global, session:<id>, goal:<id>, or project:<name>"
        )
    if details.kind == "note" and not details.scope.startswith(("session:", "goal:")):
        raise ValueError("working notes must belong to a session or goal")
    if details.kind == "archive" and not details.archive_ref:
        raise ValueError("archive memory requires an original reference")
    validate_sources(details.sources)
    for value in (details.observed_at, details.expires_at):
        validate_time(value)
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in details.exact_fields.items()
    ):
        raise ValueError("memory exact fields must be strings")
    identities = set()
    for target in details.supersedes:
        if (
            any(
                not isinstance(value, str) or not value.strip()
                for value in (target.memory_id, target.version, target.reason)
            )
            or target.memory_id in identities
        ):
            raise ValueError(
                "memory replacement requires unique identities, versions and reasons"
            )
        identities.add(target.memory_id)


def validate_sources(
    sources: tuple[MemorySource, ...], *, verification: bool = False
) -> None:
    """核验必须有可定位证据，不能只传一个时间；传参：来源及核验标识；返回：无。"""
    if verification and not sources:
        raise ValueError("memory verification requires evidence")
    for source in sources:
        if source.kind not in SOURCE_KINDS:
            raise ValueError(f"invalid memory source kind: {source.kind}")
        if source.kind != "unspecified" and not source.reference.strip():
            raise ValueError("memory source requires a reference")
        if verification and source.kind not in {"user_input", "tool_result"}:
            raise ValueError("memory verification requires user input or tool evidence")
        if bool(source.message_id) != bool(source.content_sha256):
            raise ValueError(
                "memory receipt requires both message identity and content hash"
            )
        if source.result_status and (
            source.kind != "tool_result"
            or source.result_status not in {"success", "error"}
        ):
            raise ValueError("invalid memory receipt status")
        validate_time(source.observed_at)


def validate_time(value: str | None) -> None:
    """拒绝没有时区的观察或失效时间；传参：ISO 时间或空；返回：无。"""
    if value is not None and datetime.fromisoformat(value).tzinfo is None:
        raise ValueError("memory time requires a timezone")


def seal_memory(memory: Memory) -> Memory:
    """固定内容与来源的摘要版本；传参：待提交内容；返回：带版本的独立记录。"""
    payload = _memory_payload(replace(memory, version="", last_used_at=None))
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return replace(memory, version=digest)


def format_memory(memory: Memory) -> str:
    """把完整版本写为可读 Markdown；传参：记录；返回：含前言的原文。"""
    payload = _memory_payload(replace(memory, last_used_at=None))
    content = str(payload.pop("content"))
    header = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{header}\n---\n{content.rstrip()}\n"


def parse_memory(text: str, *, editable: bool = False) -> Memory:
    """解析当前编辑稿或核验不可变修订；传参：原文及编辑面标志；返回：不含统计的记录。"""
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise ValueError("memory is missing frontmatter")
    _, header, body = text.split("---\n", 2)
    try:
        payload = yaml.safe_load(header)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid memory frontmatter: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("memory frontmatter must be an object")
    if (
        payload.get("type") not in MEMORY_TYPES
        or payload.get("state", "active") not in MEMORY_STATES
    ):
        raise ValueError("invalid memory type or state")
    if "format_version" not in payload:
        return _legacy_memory(payload, body, text=text)
    if payload["format_version"] != MEMORY_FORMAT:
        raise ValueError("unsupported memory format")
    details = dict(payload["details"])
    details["sources"] = tuple(MemorySource(**item) for item in details["sources"])
    details["supersedes"] = tuple(
        MemoryReplacement(**item) for item in details.get("supersedes", ())
    )
    payload["details"] = MemoryDetails(**details)
    payload["verification"] = tuple(
        MemorySource(**item) for item in payload["verification"]
    )
    memory = Memory(content=body.strip(), **payload)
    validate_details(memory.details)
    validate_time(memory.created_at)
    validate_time(memory.updated_at)
    if memory.last_verified_at is not None:
        validate_sources(memory.verification, verification=True)
    if (
        not memory.content
        or not isinstance(memory.memory_id, str)
        or not memory.memory_id
    ):
        raise ValueError("memory requires identity and nonempty content")
    if (
        not isinstance(memory.version, str)
        or not isinstance(memory.revision, int)
        or memory.revision < 1
    ):
        raise ValueError("invalid memory revision metadata")
    if any(
        not isinstance(values, list)
        or any(not isinstance(item, str) for item in values)
        for values in (memory.tags, memory.applicable_task_tags)
    ):
        raise ValueError("memory tags must be string lists")
    if not editable and seal_memory(memory).version != memory.version:
        raise ValueError(
            f"memory content does not match its version: {memory.memory_id}"
        )
    return memory


def _legacy_memory(payload: Mapping[str, Any], body: str, *, text: str) -> Memory:
    """旧记录仅做有损信息的显式解释，不改原件；传参：前言和正文；返回：历史版本。"""
    return Memory(
        memory_id=str(payload["memory_id"]),
        type=str(payload["type"]),
        state=str(payload.get("state", "active")),
        content=body.strip(),
        tags=list(payload.get("tags", [])),
        applicable_task_tags=list(payload.get("applicable_task_tags", [])),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
        last_used_at=payload.get("last_used_at"),
        legacy_last_verified_at=payload.get("last_verified_at"),
        format_version=1,
        version=hashlib.sha256(text.encode()).hexdigest(),
        reason="legacy_import",
        details=MemoryDetails(
            scope="unspecified", sources=(MemorySource("unspecified"),)
        ),
    )


def memory_view(memory: Memory) -> dict[str, object]:
    """输出含原文与追溯信息的工具视图；传参：记忆；返回：可序列化对象。"""
    return asdict(memory)


def _memory_payload(memory: Memory) -> dict[str, Any]:
    """保持旧版本哈希，同时让新来源/替代关系参与封存；传参：记录；返回：独立序列化内容。"""
    payload = asdict(memory)
    if not memory.details.supersedes:
        payload["details"].pop("supersedes")
    for source in (*payload["details"]["sources"], *payload["verification"]):
        for key in ("message_id", "content_sha256", "result_status"):
            if not source[key]:
                source.pop(key)
    return payload


def effective_memory_states(records: list[Memory]) -> dict[str, dict[str, object]]:
    """从已提交关系解释有效状态，撤销替代者不复活旧记录；传参：原文快照；返回：共享状态视图。"""
    replaced: dict[str, list[str]] = {}
    for memory in records:
        for target in memory.details.supersedes:
            replaced.setdefault(target.memory_id, []).append(memory.memory_id)
    return {
        memory.memory_id: {
            "effective_state": "superseded"
            if memory.memory_id in replaced
            else memory.state,
            "superseded_by": sorted(replaced.get(memory.memory_id, [])),
        }
        for memory in records
    }
