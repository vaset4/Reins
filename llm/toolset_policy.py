from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, cast

import yaml

from tools.tool_registry import ToolRegistry

PRESET_VERSION = "phase-b-v1"

# 三级 override 都没给时的 policy 来源标记。值本身不参与过滤，只写进 evidence
# 供审计辨认权限未按用户文字预选；实际发送还取决于常用集合和已加载工具
DEFAULT_POLICY_SOURCE = "default_all"

# 默认不召、但用户可显式启用的那组工具的预设名。它同时是两件事：默认排除项，
# 以及 /toolsets enable opt_in_only 的入口
OPT_IN_PRESET = "opt_in_only"


@dataclass(frozen=True, slots=True)
class ToolsetPolicy:
    enabled_toolsets: tuple[str, ...] | None = None
    disabled_toolsets: tuple[str, ...] | None = None
    source: str = DEFAULT_POLICY_SOURCE
    matched_rule: str = ""
    matched_terms: tuple[str, ...] = ()
    read_only: bool = False


class ToolsetPolicyError(ValueError):
    pass


PRESET_TOOL_NAMES: Mapping[str, tuple[str, ...]] = {
    "safe_read": (
        "file_read",
        "list",
        "grep",
        "find_path",
        "web_search",
        "web_fetch",
        "web_scan",
        "memory_query",
        "memory_manage",
        "skill_search",
        "ask_user",
        "read_artifact",
        "todo",
    ),
    "research": (
        "file_read",
        "list",
        "grep",
        "find_path",
        "web_search",
        "web_fetch",
        "web_scan",
        "memory_query",
        "memory_manage",
        "skill_search",
        "ask_user",
        "read_artifact",
        "todo",
    ),
    "browser_research": (
        "file_read",
        "list",
        "grep",
        "find_path",
        "web_search",
        "web_fetch",
        "web_scan",
        "memory_query",
        "memory_manage",
        "skill_search",
        "read_artifact",
        "todo",
        "browser_navigate",
        "browser_extract",
        "browser_screenshot",
    ),
    "browser_interact": (
        "browser_navigate",
        "browser_extract",
        "browser_screenshot",
        "browser_click",
        "browser_type",
    ),
    # 改代码的意图必须自带验证手段：没有 terminal/code_execution 就改完跑不了测试，
    # 没有 memory_query 就召不回项目约定。两个执行工具是 risk=confirm，仍走审批闸门。
    # memory_manage 的归档动作与 skill_search 随 memory_query 同进退，避免"搜得到改不掉"的断链。
    "workspace_edit": (
        "file_read",
        "list",
        "grep",
        "find_path",
        "file_write",
        "file_patch",
        "read_artifact",
        "todo",
        "memory_query",
        "memory_manage",
        "skill_search",
        "terminal_tool",
        "code_execution_tool",
        "skill_run",
    ),
    "terminal": ("terminal_tool", "code_execution_tool", "skill_run"),
    "automation": (
        "todo",
        "memory_query",
        "memory_manage",
        "memory_note",
        "skill_search",
        "read_artifact",
        "notification_send",
        "notification_status",
        "schedule",
    ),
    # 密钥、剪贴板、截图脱敏三类默认不进模型目录：前三个 risk=safe，调用时不弹审批，
    # 一旦进了目录就没有第二道人工确认了。用户要用时 /toolsets enable opt_in_only，
    # 或按物理 toolset 名 enable secret / vision。
    # 这份清单被 _with_opt_in_excluded 当默认排除项用，名字拼错会被 expand_toolset_names
    # 静默丢掉并直接导致密钥工具泄进默认目录，故有 test_opt_in_preset_has_no_silent_typos 守着
    OPT_IN_PRESET: (
        "secret_use",
        "secret_list_names",
        "screenshot",
        "ocr",
        "redact",
        "clipboard_read",
        "clipboard_write",
    ),
    # full 的空元组是占位，值从不被读取——expand_toolset_names 里 name == "full"
    # 先一步特判成"展开为全部注册工具"。别照抄这个空值：真留空的预设意味着一个工具都选不出来
    "full": (),
}


CONSOLIDATED_PRESET_ACTIONS = {
    "memory_query": frozenset({"search"}),
    "memory_manage": frozenset({"archive", "restore"}),
    "todo": frozenset({"list"}),
}


def allowed_tool_actions(
    policy: ToolsetPolicy | None, definition: object
) -> frozenset[str] | None:
    """整合入口后保留旧预设逐动作权限，多个预设取并集后减去禁用动作；参数：策略、定义；返回：允许动作或无限定。"""
    from tools.tool_registry import ToolDefinition

    if not isinstance(definition, ToolDefinition) or policy is None:
        return None
    action = cast(dict[str, Any], definition.parameters["properties"]).get("action", {})
    if "enum" not in action:
        return None
    all_actions = frozenset(action["enum"])
    result = all_actions
    if definition.name in CONSOLIDATED_PRESET_ACTIONS:
        enabled = (
            all_actions
            if policy.enabled_toolsets is None
            else preset_actions(policy.enabled_toolsets, definition, all_actions)
        )
        disabled = preset_actions(
            policy.disabled_toolsets or (), definition, all_actions
        )
        result = enabled - disabled
    if policy.read_only and not definition.readonly:
        result = result.intersection(definition.readonly_actions)
    return None if result == all_actions else result


def preset_actions(
    names: tuple[str, ...], definition: object, all_actions: frozenset[str]
) -> frozenset[str]:
    """将既有工具预设映射为同一复合入口的原动作集合；参数：预设、定义、全部动作；返回：对应动作。"""
    from tools.tool_registry import ToolDefinition

    assert isinstance(definition, ToolDefinition)
    result: set[str] = set()
    for name in names:
        if name == "full" or (
            name not in PRESET_TOOL_NAMES and name == definition.toolset
        ):
            result.update(all_actions)
        elif definition.name in PRESET_TOOL_NAMES.get(name, ()):
            actions = (
                all_actions
                if name == "automation" and definition.name == "todo"
                else CONSOLIDATED_PRESET_ACTIONS[definition.name]
            )
            result.update(actions)
    return frozenset(result)


def resolve_toolset_policy(
    *,
    payload: Mapping[str, object],
    session_state: object | None,
    config: Mapping[str, object] | None,
    registry: ToolRegistry,
) -> ToolsetPolicy:
    """按 payload > session > config 取显式策略；缺省不额外收窄，实际发送仍受按需加载约束。

    payload: 本轮请求体，可携带 toolset_policy / enabled_toolsets / disabled_toolsets
    session_state: REPL 会话状态，读 toolsets_enabled / toolsets_disabled 两个字段
    config: 运行时配置（~/.reins/config.yaml 解析结果）
    registry: 工具注册表，用于校验 toolset 名是否存在
    返回: 已补上 opt-in 排除并校验通过的 ToolsetPolicy
    """
    # 1. 三级来源逐级取，任一级给了就整体采用（不做逐字段合并）
    explicit = (
        _policy_from_payload(payload)
        or _policy_from_session(session_state)
        or _policy_from_config(config or {})
    )
    # 2. 都没给时不做任何收窄，发送集合仍由常用/按需加载选择
    if explicit is None:
        explicit = ToolsetPolicy(source=DEFAULT_POLICY_SOURCE)
    # 3. 补上 opt-in 兜底排除，再校验 toolset 名
    return validate_toolset_policy(_with_opt_in_excluded(explicit), registry)


def _with_opt_in_excluded(policy: ToolsetPolicy) -> ToolsetPolicy:
    """给未显式收窄的 policy 补上 opt-in 组排除，防止密钥/剪贴板漏进模型目录。

    policy: 三级来源解析出的（或默认的）policy
    返回: 需要兜底时返回补过 disabled_toolsets 的新实例，否则原样返回

    三级来源是整体替换而非逐字段合并，所以用户敲一句 /toolsets disable web 就会把默认
    那条 opt-in 排除整个冲掉。这层兜底放在解析出口，四级来源共用。
    """
    # 用户点名要哪些工具时完全按他说的来：此时再叠加排除会让 enabled 与 disabled
    # 两道检查互相抵消，/toolsets enable secret 的结果会是一个工具都选不出来
    if policy.enabled_toolsets is not None:
        return policy
    disabled = policy.disabled_toolsets or ()
    if OPT_IN_PRESET in disabled:
        return policy
    return replace(policy, disabled_toolsets=disabled + (OPT_IN_PRESET,))


def load_toolset_runtime_config(path: Path | None = None) -> dict[str, object]:
    config_path = path or Path.home() / ".reins" / "config.yaml"
    if not config_path.is_file():
        return {}
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ToolsetPolicyError(f"invalid toolset runtime config: {exc}") from exc
    if not isinstance(raw, Mapping):
        return {}
    return _toolset_config_from_mapping(raw)


def validate_toolset_policy(
    policy: ToolsetPolicy,
    registry: ToolRegistry,
) -> ToolsetPolicy:
    known = known_toolset_names(registry)
    unknown = [name for name in _flatten_policy_names(policy) if name not in known]
    if unknown:
        raise ToolsetPolicyError(f"unknown toolset(s): {', '.join(sorted(unknown))}")
    return policy


def known_toolset_names(registry: ToolRegistry) -> frozenset[str]:
    physical = {definition.toolset for definition in registry.list_definitions()}
    return frozenset(physical | set(PRESET_TOOL_NAMES))


def expand_toolset_names(
    names: tuple[str, ...] | None,
    registry: ToolRegistry,
) -> frozenset[str] | None:
    if names is None:
        return None
    all_tools = {definition.name for definition in registry.list_definitions()}
    expanded: set[str] = set()
    for name in names:
        if name == "full":
            expanded.update(all_tools)
        elif name in PRESET_TOOL_NAMES:
            expanded.update(PRESET_TOOL_NAMES[name])
        else:
            expanded.update(
                definition.name
                for definition in registry.list_definitions()
                if definition.toolset == name
            )
    return frozenset(tool for tool in expanded if tool in all_tools)


def policy_from_mapping(
    value: object,
    *,
    source: str,
) -> ToolsetPolicy | None:
    if not isinstance(value, Mapping):
        return None
    resolved_source = str(value.get("source", source)) or source
    if type(value.get("read_only", False)) is not bool:
        raise ToolsetPolicyError("read_only must be a boolean")
    return ToolsetPolicy(
        enabled_toolsets=_optional_tuple(value.get("enabled_toolsets")),
        disabled_toolsets=_optional_tuple(value.get("disabled_toolsets")),
        source=resolved_source,
        matched_rule=str(value.get("matched_rule", "")),
        matched_terms=_tuple_value(value.get("matched_terms")),
        read_only=value.get("read_only", False),
    )


def policy_to_mapping(policy: ToolsetPolicy) -> dict[str, object]:
    return {
        "enabled_toolsets": list(policy.enabled_toolsets)
        if policy.enabled_toolsets is not None
        else None,
        "disabled_toolsets": list(policy.disabled_toolsets)
        if policy.disabled_toolsets is not None
        else None,
        "source": policy.source,
        "matched_rule": policy.matched_rule,
        "matched_terms": list(policy.matched_terms),
        "read_only": policy.read_only,
    }


def _policy_from_payload(payload: Mapping[str, object]) -> ToolsetPolicy | None:
    nested = policy_from_mapping(payload.get("toolset_policy"), source="payload")
    if nested is not None:
        return nested
    if "enabled_toolsets" not in payload and "disabled_toolsets" not in payload:
        return None
    return ToolsetPolicy(
        enabled_toolsets=_optional_tuple(payload.get("enabled_toolsets")),
        disabled_toolsets=_optional_tuple(payload.get("disabled_toolsets")),
        source="payload",
    )


def _policy_from_session(session_state: object | None) -> ToolsetPolicy | None:
    if session_state is None:
        return None
    enabled = getattr(session_state, "toolsets_enabled", None)
    disabled = getattr(session_state, "toolsets_disabled", None)
    if enabled is None and disabled is None:
        return None
    return ToolsetPolicy(
        enabled_toolsets=_optional_tuple(enabled),
        disabled_toolsets=_optional_tuple(disabled),
        source="session",
    )


def _policy_from_config(config: Mapping[str, object]) -> ToolsetPolicy | None:
    nested = policy_from_mapping(config.get("toolset_policy"), source="config")
    if nested is not None:
        return nested
    if "toolsets_enabled" not in config and "toolsets_disabled" not in config:
        return None
    return ToolsetPolicy(
        enabled_toolsets=_optional_tuple(config.get("toolsets_enabled")),
        disabled_toolsets=_optional_tuple(config.get("toolsets_disabled")),
        source="config",
    )


def _toolset_config_from_mapping(raw: Mapping[object, object]) -> dict[str, object]:
    config: dict[str, object] = {}
    nested = raw.get("toolset_policy")
    if isinstance(nested, Mapping):
        config["toolset_policy"] = nested
    for key in ("toolsets_enabled", "toolsets_disabled"):
        if key in raw:
            config[key] = raw[key]
    toolsets = raw.get("toolsets")
    if isinstance(toolsets, Mapping):
        if "enabled" in toolsets and "toolsets_enabled" not in config:
            config["toolsets_enabled"] = toolsets["enabled"]
        if "disabled" in toolsets and "toolsets_disabled" not in config:
            config["toolsets_disabled"] = toolsets["disabled"]
    return config


def _flatten_policy_names(policy: ToolsetPolicy) -> tuple[str, ...]:
    return tuple(policy.enabled_toolsets or ()) + tuple(policy.disabled_toolsets or ())


def _optional_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _tuple_value(value)


def _tuple_value(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()
