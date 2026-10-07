from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, cast

from tasks.ids import utc_now

TriggerType = Literal["user", "cron", "idle", "delegate", "resume"]
CapabilityMap = dict[str, object]
LEASE_SEGMENT_END = "lease_segment_end"
_SEGMENT_END_TRIGGERS: frozenset[str] = frozenset({"cron", "delegate", "resume"})
_VALID_TRIGGERS: frozenset[str] = frozenset(_SEGMENT_END_TRIGGERS | {"user", "idle"})
_DEFAULT_CAPABILITIES: CapabilityMap = {
    "fs": {"read": [], "write": []},
    "terminal": {"enabled": True, "allow_commands": []},
    "browser": {
        "enabled": True,
        "profile": "default",
        "deny_domains": [],
        "headless": True,
    },
    "mouse_keyboard": {"enabled": False},
    "network": {"enabled": True, "deny_domains": []},
    "background_run": {"enabled": False},
    "mcp": {"enabled": True, "allow_servers": []},
    "code_execution": {"enabled": True},
}

# --- list 字段合并分类（lease-merge-semantics, design §1/§4.3）---
# 按消费方真实判定语义分类，非按字段名猜。未分类 list 默认 union（deny 语义的
# 保守安全默认：union 永不扩权）。
# allow-list（empty = unrestricted / 默认更宽）：child 只能收窄；两非空且交集为空
# = 越权（无法编码为更宽的 []）→ raise（用户裁定 2026-06-01，design §2）。
_ALLOW_LIST_PATHS: frozenset[str] = frozenset(
    {"fs.read", "fs.write", "terminal.allow_commands"}
)
# allow-list（empty = nothing / 全拒，语义与上相反）：纯集合交，空交集 = [] 全拒
# （合法，最严，不 raise）。
_MCP_ALLOW_PATHS: frozenset[str] = frozenset({"mcp.allow_servers"})
# metadata（合并豁免）：既不交也不并，原样深拷贝保留。schedule.required_permanent_grants
# 由 cron 注入、read-only 消费；对其做 allow/deny 合并会损坏授权语义（C5 边界）。
_MERGE_EXEMPT_PATHS: frozenset[str] = frozenset({"schedule"})


class LeaseEscalationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Lease:
    # lease 是一次 run/segment 的能力快照。它保持 frozen，避免模型循环在
    # segment 启动后静默扩大权限。
    type: Literal["lease_snapshot"] = "lease_snapshot"
    ts: str = field(default_factory=utc_now)
    trigger: TriggerType = "user"
    task_id: str = ""
    capabilities: CapabilityMap = field(
        default_factory=lambda: _copy_mapping(_DEFAULT_CAPABILITIES)
    )
    expires_at: str = LEASE_SEGMENT_END
    max_steps: int = 30
    max_tokens: int = 200000


def from_trigger(
    trigger: TriggerType | str,
    *,
    task_id: str = "",
    capabilities: Mapping[str, object] | None = None,
    max_steps: int = 30,
    max_tokens: int = 200000,
    now: datetime | None = None,
) -> Lease:
    return _build_lease(
        _normalize_trigger(trigger),
        task_id=task_id,
        capabilities=capabilities,
        max_steps=max_steps,
        max_tokens=max_tokens,
        now=now,
    )


def from_template(
    template: Mapping[str, object],
    *,
    trigger: TriggerType | str,
    task_id: str = "",
    now: datetime | None = None,
) -> Lease:
    return _build_lease(
        _normalize_trigger(trigger),
        task_id=str(template.get("task_id", task_id)),
        capabilities=_mapping_or_none(template.get("capabilities")),
        max_steps=_int_value(template.get("max_steps"), 30),
        max_tokens=_int_value(template.get("max_tokens"), 200000),
        now=now,
    )


def load_snapshot(snapshot: Mapping[str, object]) -> Lease:
    payload = snapshot.get("lease_snapshot", snapshot)
    if not isinstance(payload, Mapping):
        raise ValueError("lease snapshot must be an object")
    return Lease(
        type="lease_snapshot",
        ts=str(payload.get("ts", utc_now())),
        trigger=_normalize_trigger(str(payload.get("trigger", "user"))),
        task_id=str(payload.get("task_id", "")),
        capabilities=_copy_mapping(
            _mapping_or_none(payload.get("capabilities")) or _DEFAULT_CAPABILITIES
        ),
        expires_at=str(payload.get("expires_at", LEASE_SEGMENT_END)),
        max_steps=_int_value(payload.get("max_steps"), 30),
        max_tokens=_int_value(payload.get("max_tokens"), 200000),
    )


def is_expired(lease: Lease, now: datetime | None = None) -> bool:
    if lease.expires_at == LEASE_SEGMENT_END:
        return False
    current = now or datetime.now(timezone.utc)
    return current >= datetime.fromisoformat(lease.expires_at)


def merge_lease(parent: Lease, child: Lease) -> Lease:
    # 子 lease 只能收窄能力。扩大作用域必须发生在显式审批或入口边界，
    # 不能藏在模型驱动的代码路径里。
    capabilities = _merge_capabilities(parent.capabilities, child.capabilities)
    return Lease(
        ts=child.ts,
        trigger=child.trigger,
        task_id=child.task_id or parent.task_id,
        capabilities=capabilities,
        expires_at=_earlier_expiry(parent.expires_at, child.expires_at),
        max_steps=min(parent.max_steps, child.max_steps),
        max_tokens=min(parent.max_tokens, child.max_tokens),
    )


def _build_lease(
    trigger: TriggerType,
    *,
    task_id: str,
    capabilities: Mapping[str, object] | None,
    max_steps: int,
    max_tokens: int,
    now: datetime | None,
) -> Lease:
    current = now or datetime.now(timezone.utc)
    return Lease(
        ts=current.isoformat(timespec="seconds"),
        trigger=trigger,
        task_id=task_id,
        capabilities=_copy_mapping(capabilities or _DEFAULT_CAPABILITIES),
        expires_at=_expires_at(trigger, current),
        max_steps=max_steps,
        max_tokens=max_tokens,
    )


def _expires_at(trigger: TriggerType, now: datetime) -> str:
    if trigger in _SEGMENT_END_TRIGGERS:
        return LEASE_SEGMENT_END
    delta = timedelta(minutes=30) if trigger == "idle" else timedelta(hours=1)
    return (now + delta).isoformat(timespec="seconds")


def _merge_capabilities(
    parent: Mapping[str, object], child: Mapping[str, object], prefix: str = ""
) -> CapabilityMap:
    for key in child:
        if key not in parent:
            raise LeaseEscalationError(f"capability not in parent lease: {key}")
    result: CapabilityMap = {}
    for key, parent_value in parent.items():
        if key not in child:
            result[key] = _copy_value(parent_value)
            continue
        path = f"{prefix}.{key}" if prefix else key
        result[key] = _merge_value(parent_value, child[key], path)
    return result


def _merge_value(parent: object, child: object, path: str) -> object:
    if path in _MERGE_EXEMPT_PATHS:
        return _copy_value(child)
    if isinstance(parent, bool) and isinstance(child, bool):
        if child and not parent:
            raise LeaseEscalationError(f"capability expands at {path}")
        return parent and child
    if isinstance(parent, list) and isinstance(child, list):
        return _merge_list(parent, child, path)
    if isinstance(parent, Mapping) and isinstance(child, Mapping):
        return _merge_capabilities(parent, child, prefix=path)
    return _copy_value(child if child == parent else parent)


def _merge_list(parent: list[object], child: list[object], path: str) -> list[str]:
    if path in _ALLOW_LIST_PATHS:
        return _intersect_allow(parent, child, path)
    if path in _MCP_ALLOW_PATHS:
        return _intersect_mcp(parent, child)
    return _union_deny(parent, child)


def _intersect_allow(parent: list[object], child: list[object], path: str) -> list[str]:
    # empty = unrestricted（单位元 ⊤）：空侧不施加限制；两非空且交集为空 = 越权 → raise。
    parent_set = {str(item) for item in parent}
    child_set = {str(item) for item in child}
    if not parent_set:
        return sorted(child_set)
    if not child_set:
        return sorted(parent_set)
    intersection = parent_set & child_set
    if not intersection:
        raise LeaseEscalationError(
            f"allow-list narrows to empty (escalation) at {path}"
        )
    return sorted(intersection)


def _intersect_mcp(parent: list[object], child: list[object]) -> list[str]:
    # empty = nothing（全拒）：纯集合交；空交集 = [] 全拒（合法，最严），不 raise。
    return sorted({str(item) for item in parent} & {str(item) for item in child})


def _union_deny(parent: list[object], child: list[object]) -> list[str]:
    # deny-list 与未分类 list 的保守默认：union 永不扩权（更严或相等，恒合法）。
    return sorted({str(item) for item in parent + child})


def _earlier_expiry(parent: str, child: str) -> str:
    if parent == LEASE_SEGMENT_END:
        return child
    if child == LEASE_SEGMENT_END:
        return parent
    return min(datetime.fromisoformat(parent), datetime.fromisoformat(child)).isoformat(
        timespec="seconds"
    )


def _normalize_trigger(trigger: TriggerType | str) -> TriggerType:
    if trigger not in _VALID_TRIGGERS:
        raise ValueError(f"invalid trigger: {trigger}")
    return cast(TriggerType, trigger)


def _mapping_or_none(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("capabilities must be an object")
    return value


def _int_value(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError("lease numeric fields must be integers")


def _copy_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _copy_mapping(value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _copy_mapping(value: Mapping[str, object]) -> CapabilityMap:
    return {str(key): _copy_value(item) for key, item in value.items()}
