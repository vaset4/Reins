from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

from llm.model_request import ToolAuditRecord, ToolExclusion, ToolSelectionResult
from llm.toolset_policy import (
    PRESET_VERSION,
    CONSOLIDATED_PRESET_ACTIONS,
    ToolsetPolicy,
    expand_toolset_names,
    allowed_tool_actions,
)
from tools.tool_registry import ToolDefinition, ToolRegistry, ToolRisk
from tools.types import VisibilityReportSummary


def select_tools(
    registry: ToolRegistry,
    *,
    protocol_mode: str = "native_tool_calls",
    policy: ToolsetPolicy | None = None,
    lease: object | None = None,
    allowed_actions: Collection[str] | None = None,
    loaded_tools: Collection[str] = (),
    include_deferred: bool = False,
) -> ToolSelectionResult:
    """从请求快照选择当前授权且已加载的定义；传参：目录、策略与加载状态；返回：定义及排除证据。"""
    del protocol_mode
    registry = registry.snapshot()
    reports, report_summary = registry.get_visibility_report()
    definitions = {item.name: item for item in registry.list_definitions()}
    enabled_names = (
        expand_toolset_names(policy.enabled_toolsets, registry) if policy else None
    )
    disabled_names = (
        expand_toolset_names(policy.disabled_toolsets, registry) if policy else None
    )
    explicit_allowed = _canonical_allowed_names(registry, allowed_actions)
    selected_defs: list[ToolDefinition] = []
    selected: list[ToolAuditRecord] = []
    excluded: list[ToolExclusion] = []

    for report in sorted(reports, key=lambda item: item.name):
        definition = definitions.get(report.name)
        reason = _exclusion_reason(
            report=report,
            definition=definition,
            policy=policy,
            enabled_names=enabled_names,
            disabled_names=disabled_names,
            lease=lease,
            explicit_allowed=explicit_allowed,
        )
        if (
            reason is None
            and definition is not None
            and definition.deferred
            and not include_deferred
        ):
            if definition.name not in loaded_tools and enabled_names is None:
                reason = (
                    "deferred_not_loaded",
                    "discover and load this tool through capabilities",
                )
        if reason is None and definition is not None:
            definition = policy_definition(definition, policy)
            selected_defs.append(definition)
            selected.append(
                _audit_record(
                    definition,
                    definition_version=registry.definition_version(definition.name),
                )
            )
            continue
        excluded.append(_exclusion_record(report, definition, reason))

    return _selection_result(
        registry,
        selected_defs,
        selected=selected,
        excluded=excluded,
        report_summary=report_summary,
        policy=policy,
        lease=lease,
        enabled_names=enabled_names,
        disabled_names=disabled_names,
        explicit_allowed=explicit_allowed,
    )


def _selection_result(
    registry: ToolRegistry,
    selected_defs: list[ToolDefinition],
    *,
    selected: list[ToolAuditRecord],
    excluded: list[ToolExclusion],
    report_summary: VisibilityReportSummary,
    policy: ToolsetPolicy | None,
    lease: object | None,
    enabled_names: frozenset[str] | None,
    disabled_names: frozenset[str] | None,
    explicit_allowed: frozenset[str] | None,
) -> ToolSelectionResult:
    """把选择结果与策略来源投影为同一份请求证据；传参：快照及已判定目录；返回：不可变选择结果。"""
    summary = {
        "selected": len(selected),
        "registered": report_summary.total_tools,
        "candidates": report_summary.visible_tools,
        "actual_sent": len(selected),
        "deferred_not_loaded": sum(
            item.reason == "deferred_not_loaded" for item in excluded
        ),
        "domains": {
            domain: sum(
                (item.domain or item.toolset) == domain for item in selected_defs
            )
            for domain in sorted(
                {item.domain or item.toolset for item in selected_defs}
            )
        },
        "selected_with_approval_required": sum(
            1 for item in selected if item.status == "selected_with_approval_required"
        ),
        "excluded": len(excluded),
        "visibility_report": {
            "total_tools": report_summary.total_tools,
            "visible_tools": report_summary.visible_tools,
            "hidden_tools": report_summary.hidden_tools,
            "unavailable_tools": report_summary.unavailable_tools,
            "warnings": list(report_summary.warnings),
        },
    }
    if policy is not None or lease is not None:
        summary["policy"] = _policy_summary(policy)
        summary["expanded_tool_names"] = {
            "enabled": sorted(enabled_names) if enabled_names is not None else None,
            "disabled": sorted(disabled_names) if disabled_names is not None else None,
        }
    if explicit_allowed is not None:
        summary["allowed_actions"] = sorted(explicit_allowed)
    allowed_tool_names = frozenset(item.name for item in selected_defs)
    return ToolSelectionResult(
        mode="hybrid" if policy is not None or lease is not None else "legacy_default",
        selected_definitions=tuple(sorted(selected_defs, key=lambda item: item.name)),
        selected=tuple(selected),
        excluded=tuple(excluded),
        summary=summary,
        allowed_tool_names=allowed_tool_names,
        policy_source=policy.source if policy is not None else "legacy_default",
        enabled_toolsets=policy.enabled_toolsets if policy is not None else None,
        disabled_toolsets=policy.disabled_toolsets if policy is not None else None,
        registry_version=registry.version,
    )


def canonical_tool_schema_hash(definition: ToolDefinition) -> str:
    payload = {
        "name": definition.name,
        "description": definition.description,
        "parameters": definition.parameters,
        "toolset": definition.toolset,
        "risk": definition.risk.value,
        "readonly": definition.readonly,
        "target_scope_rule": definition.target_scope_rule,
        "source": definition.source,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _audit_record(
    definition: ToolDefinition, *, definition_version: str
) -> ToolAuditRecord:
    approval_required = definition.risk.value == "confirm"
    return ToolAuditRecord(
        name=definition.name,
        toolset=definition.toolset,
        risk=definition.risk.value,
        readonly=definition.readonly,
        source=definition.source,
        status="selected_with_approval_required" if approval_required else "selected",
        approval_required=approval_required,
        schema_hash=canonical_tool_schema_hash(definition),
        definition_version=definition_version,
    )


def _exclusion_record(
    report: object,
    definition: ToolDefinition | None,
    reason: tuple[str, str] | None,
) -> ToolExclusion:
    reason_name, detail = reason or _visibility_reason(report)
    return ToolExclusion(
        name=str(getattr(report, "name")),
        toolset=getattr(report, "toolset", None),
        source=getattr(report, "source", None),
        reason=reason_name,  # type: ignore[arg-type]
        detail=detail,
        schema_hash=canonical_tool_schema_hash(definition)
        if definition is not None
        else None,
    )


def _exclusion_reason(
    *,
    report: object,
    definition: ToolDefinition | None,
    policy: ToolsetPolicy | None,
    enabled_names: frozenset[str] | None,
    disabled_names: frozenset[str] | None,
    lease: object | None,
    explicit_allowed: frozenset[str] | None,
) -> tuple[str, str] | None:
    if definition is None or not getattr(report, "model_visible"):
        return _visibility_reason(report)
    if not getattr(report, "available"):
        return _visibility_reason(report)
    hard = _hard_boundary_reason(definition, lease)
    if hard is not None:
        return hard
    if explicit_allowed is not None and not _name_allowed(
        definition.name, explicit_allowed
    ):
        return ("action_not_allowed", "tool not present in allowed_actions")
    if policy is None:
        return None
    if (
        policy.read_only
        and not definition.readonly
        and not definition.readonly_actions
        and definition.name != "terminal_tool"
    ):
        return ("action_not_allowed", "current approval session is read-only")
    actions = allowed_tool_actions(policy, definition)
    if actions is not None and not actions:
        return (
            "toolset_disabled",
            "no actions are allowed under the current toolset policy",
        )
    if enabled_names is not None and definition.name not in enabled_names:
        return ("toolset_not_enabled", _policy_detail("enabled", policy))
    if (
        disabled_names is not None
        and definition.name in disabled_names
        and (actions is None or definition.name not in CONSOLIDATED_PRESET_ACTIONS)
    ):
        return ("toolset_disabled", _policy_detail("disabled", policy))
    return None


def _visibility_reason(report: object) -> tuple[str, str]:
    if not getattr(report, "model_visible"):
        return ("hidden", str(getattr(report, "reason", "") or "model_visible=False"))
    return ("unavailable", str(getattr(report, "reason", "") or "unavailable"))


def _hard_boundary_reason(
    definition: ToolDefinition,
    lease: object | None,
) -> tuple[str, str] | None:
    if definition.risk is ToolRisk.DENY:
        return ("risk_denied", "tool risk is deny")
    capabilities = getattr(lease, "capabilities", None)
    if not isinstance(capabilities, dict):
        return None
    if _terminal_blocked(definition, capabilities):
        return ("lease_disallowed", "terminal capability is disabled")
    if _network_blocked(definition, capabilities):
        return ("lease_disallowed", "network capability is disabled")
    if _browser_blocked(definition, capabilities):
        return ("lease_disallowed", "browser capability is disabled")
    if _mcp_blocked(definition, capabilities):
        return ("lease_disallowed", "mcp capability disallows this tool")
    if _fs_blocked(definition, capabilities):
        return ("lease_disallowed", "filesystem capability is disabled")
    return None


def _terminal_blocked(
    definition: ToolDefinition,
    capabilities: dict[str, object],
) -> bool:
    terminal = capabilities.get("terminal")
    return definition.name in {"terminal_tool", "code_execution_tool"} and (
        not isinstance(terminal, dict) or terminal.get("enabled") is False
    )


def _network_blocked(
    definition: ToolDefinition,
    capabilities: dict[str, object],
) -> bool:
    network = capabilities.get("network")
    is_web_tool = definition.toolset == "web" or definition.name.startswith("web_")
    return is_web_tool and (
        not isinstance(network, dict) or network.get("enabled") is False
    )


def _browser_blocked(
    definition: ToolDefinition,
    capabilities: dict[str, object],
) -> bool:
    browser = capabilities.get("browser")
    return definition.name.startswith("browser_") and (
        not isinstance(browser, dict) or browser.get("enabled") is False
    )


def _mcp_blocked(
    definition: ToolDefinition,
    capabilities: dict[str, object],
) -> bool:
    if not definition.name.startswith("mcp_"):
        return False
    mcp = capabilities.get("mcp")
    if not isinstance(mcp, dict) or mcp.get("enabled") is False:
        return True
    allowed = [str(item) for item in mcp.get("allow_servers", []) if item]
    return (
        not allowed
        or (definition.mcp_server or _mcp_server_name(definition.name)) not in allowed
    )


def _fs_blocked(
    definition: ToolDefinition,
    capabilities: dict[str, object],
) -> bool:
    fs = capabilities.get("fs")
    return definition.toolset == "file" and (
        not isinstance(fs, dict) or fs.get("enabled") is False
    )


def _mcp_server_name(tool_name: str) -> str:
    parts = tool_name.split("_", maxsplit=2)
    return parts[1] if len(parts) >= 3 and parts[0] == "mcp" else ""


def _canonical_allowed_names(
    registry: ToolRegistry,
    allowed_actions: Collection[str] | None,
) -> frozenset[str] | None:
    if allowed_actions is None:
        return None
    names: set[str] = set()
    for item in allowed_actions:
        raw = str(item).strip()
        if not raw:
            continue
        definition = registry.get(raw)
        names.add(raw if raw.endswith("*") or definition is None else definition.name)
    return frozenset(names)


def _name_allowed(tool_name: str, allowed: frozenset[str]) -> bool:
    return tool_name in allowed or any(
        item.endswith("*") and tool_name.startswith(item[:-1]) for item in allowed
    )


def _policy_detail(kind: str, policy: ToolsetPolicy) -> str:
    names = policy.enabled_toolsets if kind == "enabled" else policy.disabled_toolsets
    return f"{kind}_toolsets={', '.join(names or ())}"


def _policy_summary(policy: ToolsetPolicy | None) -> dict[str, object]:
    if policy is None:
        return {"source": "lease_only", "preset_version": PRESET_VERSION}
    return {
        "source": policy.source,
        "matched_rule": policy.matched_rule,
        "matched_terms": list(policy.matched_terms),
        "preset_version": PRESET_VERSION,
        "read_only": policy.read_only,
    }


def policy_definition(
    definition: ToolDefinition, policy: ToolsetPolicy | None
) -> ToolDefinition:
    """模型只看当前预设允许的动作，执行仍重核同一策略；参数：注册定义、策略；返回：请求投影。"""
    actions = allowed_tool_actions(policy, definition)
    if actions is None:
        return definition
    schema = deepcopy(definition.parameters)
    cast(dict[str, Any], schema["properties"])["action"]["enum"] = sorted(actions)
    readonly = definition.readonly or actions.issubset(definition.readonly_actions)
    return replace(
        definition,
        parameters=schema,
        readonly=readonly,
        risk_level=ToolRisk.SAFE
        if readonly and definition.risk is not ToolRisk.DENY
        else definition.risk,
    )
