from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone

import pytest

from runtime.lease import (
    LEASE_SEGMENT_END,
    Lease,
    LeaseEscalationError,
    from_trigger,
    is_expired,
    merge_lease,
)


def test_lease_fields_are_frozen_and_match_snapshot_contract() -> None:
    lease = from_trigger("user", task_id="task-1")

    assert [field.name for field in fields(Lease)] == [
        "type",
        "ts",
        "trigger",
        "task_id",
        "capabilities",
        "expires_at",
        "max_steps",
        "max_tokens",
    ]
    with pytest.raises(FrozenInstanceError):
        lease.max_steps = 50  # type: ignore[misc]
    assert set(lease.capabilities) == {
        "fs",
        "terminal",
        "browser",
        "mouse_keyboard",
        "network",
        "background_run",
        "mcp",
        "code_execution",
    }


def test_default_code_execution_capability_contract() -> None:
    lease = from_trigger("user", task_id="task-1")

    assert lease.capabilities["code_execution"] == {"enabled": True}


def test_default_browser_capability_contract() -> None:
    lease = from_trigger("user", task_id="task-1")

    assert lease.capabilities["browser"] == {
        "enabled": True,
        "profile": "default",
        "deny_domains": [],
        "headless": True,
    }


def test_trigger_expiry_classes() -> None:
    now = datetime(2026, 5, 5, 8, 0, tzinfo=timezone.utc)

    user_lease = from_trigger("user", task_id="task-1", now=now)
    cron_lease = from_trigger("cron", task_id="task-1", now=now)
    idle_lease = from_trigger("idle", task_id="task-1", now=now)

    assert datetime.fromisoformat(user_lease.expires_at) == now + timedelta(hours=1)
    assert cron_lease.expires_at == LEASE_SEGMENT_END
    assert not is_expired(cron_lease, now + timedelta(days=1))
    assert datetime.fromisoformat(idle_lease.expires_at) == now + timedelta(minutes=30)
    assert is_expired(user_lease, now + timedelta(hours=1, seconds=1))


def test_merge_lease_only_tightens_scope() -> None:
    # allow_commands 有交集 → 合法收窄到交集；deny_domains union；数值取 min。
    # （disjoint allow_commands = 越权拒绝由 test_merge_lease_disjoint_allow_commands_raises 覆盖）
    parent = from_trigger(
        "user",
        task_id="task-1",
        capabilities={
            "terminal": {"enabled": True, "allow_commands": ["git status", "git log"]},
            "browser": {"enabled": True, "deny_domains": ["bank*"]},
        },
        max_steps=30,
        max_tokens=200,
    )
    child = from_trigger(
        "delegate",
        task_id="task-1",
        capabilities={
            "terminal": {"enabled": False, "allow_commands": ["git log"]},
            "browser": {"enabled": True, "deny_domains": ["*.gov.cn"]},
        },
        max_steps=10,
        max_tokens=100,
    )

    merged = merge_lease(parent, child)

    assert merged.max_steps == 10
    assert merged.max_tokens == 100
    assert merged.capabilities["terminal"] == {
        "enabled": False,
        "allow_commands": ["git log"],
    }
    assert merged.capabilities["browser"] == {
        "enabled": True,
        "deny_domains": ["*.gov.cn", "bank*"],
    }


def test_merge_lease_browser_headless_contract() -> None:
    parent = from_trigger("user", task_id="task-1")
    child = from_trigger(
        "delegate",
        task_id="task-1",
        capabilities={
            "browser": {
                "enabled": True,
                "profile": "default",
                "deny_domains": [],
                "headless": False,
            }
        },
    )

    merged = merge_lease(parent, child)

    assert merged.capabilities["browser"] == {
        "enabled": True,
        "profile": "default",
        "deny_domains": [],
        "headless": False,
    }


def test_merge_lease_rejects_new_browser_subkey() -> None:
    parent = from_trigger(
        "user",
        task_id="task-1",
        capabilities={"browser": {"enabled": True}},
    )
    child = from_trigger(
        "delegate",
        task_id="task-1",
        capabilities={"browser": {"enabled": True, "headless": False}},
    )

    with pytest.raises(LeaseEscalationError):
        merge_lease(parent, child)


def test_merge_lease_rejects_capability_expansion() -> None:
    parent = from_trigger(
        "user",
        task_id="task-1",
        capabilities={"terminal": {"enabled": False}},
    )
    child = from_trigger(
        "delegate",
        task_id="task-1",
        capabilities={"terminal": {"enabled": True}},
    )

    with pytest.raises(LeaseEscalationError):
        merge_lease(parent, child)


# --- Batch 1 / lease-merge-semantics: list 字段按 allow/deny/metadata 分类合并 ---
# 缺陷：_merge_value 旧逻辑对所有 list 无差别 union，子 lease 可向 allow-list
# (terminal.allow_commands / fs.read / fs.write / mcp.allow_servers) 加项扩权。
# 详见 design §1 字段分类表、§2 空列表语义陷阱（用户 2026-06-01 裁定空交集→raise）。


def _lease(caps: dict[str, object], *, trigger: str = "user") -> Lease:
    return from_trigger(trigger, task_id="t", capabilities=caps)


# 阶段 A — allow-list 越权护栏（PRD R2 / AC1）


def test_merge_allow_commands_intersect_not_union() -> None:
    parent = _lease(
        {"terminal": {"enabled": True, "allow_commands": ["git status", "git log"]}}
    )
    child = _lease(
        {"terminal": {"enabled": True, "allow_commands": ["git log", "rm -rf"]}},
        trigger="delegate",
    )

    merged = merge_lease(parent, child)

    # 交集收窄：不含 child 想新增的 rm -rf，也不含 parent 独有的 git status。
    assert merged.capabilities["terminal"] == {
        "enabled": True,
        "allow_commands": ["git log"],
    }


def test_merge_allow_commands_disjoint_raises() -> None:
    # 两非空 allow-list 交集为空 = 越权（无法编码为更宽的 []）→ raise（用户裁定）。
    parent = _lease({"terminal": {"enabled": True, "allow_commands": ["git status"]}})
    child = _lease(
        {"terminal": {"enabled": True, "allow_commands": ["git log"]}},
        trigger="delegate",
    )

    with pytest.raises(LeaseEscalationError):
        merge_lease(parent, child)


def test_merge_mcp_allow_servers_intersect() -> None:
    parent = _lease({"mcp": {"enabled": True, "allow_servers": ["a", "b"]}})
    child = _lease(
        {"mcp": {"enabled": True, "allow_servers": ["b", "c"]}}, trigger="delegate"
    )

    merged = merge_lease(parent, child)

    assert merged.capabilities["mcp"] == {"enabled": True, "allow_servers": ["b"]}


def test_merge_mcp_allow_servers_empty_child_denies_all() -> None:
    # mcp 的 empty = nothing（全拒）：纯集合交，child=[] → [] 合法（更严），不 raise。
    parent = _lease({"mcp": {"enabled": True, "allow_servers": ["a", "b"]}})
    child = _lease({"mcp": {"enabled": True, "allow_servers": []}}, trigger="delegate")

    merged = merge_lease(parent, child)

    assert merged.capabilities["mcp"] == {"enabled": True, "allow_servers": []}


def test_merge_mcp_allow_servers_disjoint_denies_all_no_raise() -> None:
    # 与 fs/terminal 相反：mcp 两非空不相交 → [] 全拒（最严，合法），绝不 raise。
    parent = _lease({"mcp": {"enabled": True, "allow_servers": ["a"]}})
    child = _lease(
        {"mcp": {"enabled": True, "allow_servers": ["b"]}}, trigger="delegate"
    )

    merged = merge_lease(parent, child)

    assert merged.capabilities["mcp"] == {"enabled": True, "allow_servers": []}


def test_merge_fs_read_empty_child_keeps_parent_restriction() -> None:
    # fs.read 的 empty = unrestricted（单位元）：child 空 ≠ 想全开，而是不进一步限制。
    parent = _lease({"fs": {"read": ["/proj", "/data"], "write": []}})
    child = _lease({"fs": {"read": [], "write": []}}, trigger="delegate")

    merged = merge_lease(parent, child)

    assert merged.capabilities["fs"] == {"read": ["/data", "/proj"], "write": []}


def test_merge_fs_read_both_nonempty_intersect() -> None:
    parent = _lease({"fs": {"read": ["/proj", "/data"], "write": []}})
    child = _lease({"fs": {"read": ["/data", "/tmp"], "write": []}}, trigger="delegate")

    merged = merge_lease(parent, child)

    # 丢掉 child 想新增的 /tmp（越权项）。
    assert merged.capabilities["fs"] == {"read": ["/data"], "write": []}


def test_merge_fs_read_disjoint_raises() -> None:
    parent = _lease({"fs": {"read": ["/proj"], "write": []}})
    child = _lease({"fs": {"read": ["/tmp"], "write": []}}, trigger="delegate")

    with pytest.raises(LeaseEscalationError):
        merge_lease(parent, child)


def test_merge_fs_write_intersect() -> None:
    parent = _lease({"fs": {"read": [], "write": ["/proj", "/data"]}})
    child = _lease({"fs": {"read": [], "write": ["/data"]}}, trigger="delegate")

    assert merge_lease(parent, child).capabilities["fs"] == {
        "read": [],
        "write": ["/data"],
    }


def test_merge_fs_write_empty_child_keeps_parent() -> None:
    parent = _lease({"fs": {"read": [], "write": ["/proj", "/data"]}})
    child = _lease({"fs": {"read": [], "write": []}}, trigger="delegate")

    assert merge_lease(parent, child).capabilities["fs"] == {
        "read": [],
        "write": ["/data", "/proj"],
    }


def test_merge_fs_write_disjoint_raises() -> None:
    parent = _lease({"fs": {"read": [], "write": ["/proj"]}})
    child = _lease({"fs": {"read": [], "write": ["/tmp"]}}, trigger="delegate")

    with pytest.raises(LeaseEscalationError):
        merge_lease(parent, child)


# 阶段 B — deny-list 保守 union 保持（PRD R3 / AC2）


def test_merge_browser_deny_domains_union() -> None:
    parent = _lease({"browser": {"enabled": True, "deny_domains": ["bank*"]}})
    child = _lease(
        {"browser": {"enabled": True, "deny_domains": ["*.gov.cn"]}}, trigger="delegate"
    )

    merged = merge_lease(parent, child)

    assert merged.capabilities["browser"] == {
        "enabled": True,
        "deny_domains": ["*.gov.cn", "bank*"],
    }


def test_merge_network_deny_domains_union() -> None:
    parent = _lease({"network": {"enabled": True, "deny_domains": ["a.com"]}})
    child = _lease(
        {"network": {"enabled": True, "deny_domains": ["b.com"]}}, trigger="delegate"
    )

    merged = merge_lease(parent, child)

    assert merged.capabilities["network"] == {
        "enabled": True,
        "deny_domains": ["a.com", "b.com"],
    }


def test_merge_terminal_deny_commands_union() -> None:
    parent = _lease({"terminal": {"enabled": True, "deny_commands": ["rm"]}})
    child = _lease(
        {"terminal": {"enabled": True, "deny_commands": ["dd"]}}, trigger="delegate"
    )

    merged = merge_lease(parent, child)

    assert merged.capabilities["terminal"] == {
        "enabled": True,
        "deny_commands": ["dd", "rm"],
    }


# 阶段 C — schedule metadata 合并豁免（PRD R4 / AC3，C5 边界）
# 注：当前生产中 merge 链上的 lease 均不含 schedule（cron 走 Lease() 默认能力，
# triggers/cron.py:40），故本组测试为下游 cron-lease-closure 落地铺路并钉死语义。


def test_merge_schedule_grants_exempt_preserved() -> None:
    base = {"terminal": {"enabled": True, "allow_commands": []}}
    parent = _lease(
        {
            **base,
            "schedule": {
                "required_permanent_grants": [{"tool": "file_write", "args": {}}]
            },
        }
    )
    child = _lease(
        {
            **base,
            "schedule": {
                "required_permanent_grants": [{"tool": "terminal", "args": {}}]
            },
        },
        trigger="delegate",
    )

    merged = merge_lease(parent, child)

    # 既不 union 也不 intersect，更不能 str() 化损坏 dict grants。
    assert merged.capabilities["schedule"] == {
        "required_permanent_grants": [{"tool": "terminal", "args": {}}]
    }


def test_merge_parent_only_schedule_kept() -> None:
    base = {"terminal": {"enabled": True, "allow_commands": []}}
    parent = _lease(
        {
            **base,
            "schedule": {
                "required_permanent_grants": [{"tool": "file_write", "args": {}}]
            },
        }
    )
    child = _lease(dict(base), trigger="delegate")

    merged = merge_lease(parent, child)

    assert merged.capabilities["schedule"] == {
        "required_permanent_grants": [{"tool": "file_write", "args": {}}]
    }


def test_merge_child_adds_schedule_raises() -> None:
    # 豁免合并算法，但不豁免 key 存在性检查（design §4.5）。
    base = {"terminal": {"enabled": True, "allow_commands": []}}
    parent = _lease(dict(base))
    child = _lease(
        {
            **base,
            "schedule": {
                "required_permanent_grants": [{"tool": "terminal", "args": {}}]
            },
        },
        trigger="delegate",
    )

    with pytest.raises(LeaseEscalationError):
        merge_lease(parent, child)


# 阶段 D — 现有 test_merge_lease_only_tightens_scope 旧数据(disjoint)在集成层钉死越权拒绝


def test_merge_lease_disjoint_allow_commands_raises() -> None:
    # 旧 test_merge_lease_only_tightens_scope 的数据：parent ["git status"] ‖ child ["git log"]
    # 两非空、交集空 → 整个 merge_lease raise（design §2，禁塌成 [] = 全开）。
    parent = from_trigger(
        "user",
        task_id="task-1",
        capabilities={
            "terminal": {"enabled": True, "allow_commands": ["git status"]},
            "browser": {"enabled": True, "deny_domains": ["bank*"]},
        },
        max_steps=30,
        max_tokens=200,
    )
    child = from_trigger(
        "delegate",
        task_id="task-1",
        capabilities={
            "terminal": {"enabled": False, "allow_commands": ["git log"]},
            "browser": {"enabled": True, "deny_domains": ["*.gov.cn"]},
        },
        max_steps=10,
        max_tokens=100,
    )

    with pytest.raises(LeaseEscalationError):
        merge_lease(parent, child)
