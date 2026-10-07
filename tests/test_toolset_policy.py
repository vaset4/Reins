from __future__ import annotations

from pathlib import Path

import pytest

from llm.tool_selection import select_tools
from llm.toolset_policy import (
    DEFAULT_POLICY_SOURCE,
    OPT_IN_PRESET,
    PRESET_TOOL_NAMES,
    PRESET_VERSION,
    ToolsetPolicy,
    ToolsetPolicyError,
    expand_toolset_names,
    load_toolset_runtime_config,
    policy_to_mapping,
    resolve_toolset_policy,
    validate_toolset_policy,
)
from runtime.session_state import SessionState
from tools.tool_registry import (
    IDEMPOTENT_YES,
    TARGET_SCOPE_LOGICAL,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_FILE,
    TOOLSET_WEB,
    ToolDefinition,
    ToolRegistry,
    ToolRisk,
    get_default_tool_registry,
)


def test_resolve_toolset_policy_uses_payload_before_session_and_config() -> None:
    registry = _registry()
    session = SessionState(
        session_id="session-1",
        toolsets_enabled=["web"],
    )

    policy = resolve_toolset_policy(
        payload={"enabled_toolsets": ["file"]},
        session_state=session,
        config={"toolsets_enabled": ["web"]},
        registry=registry,
    )

    assert policy.source == "payload"
    assert policy.enabled_toolsets == ("file",)


def test_resolve_toolset_policy_uses_session_before_config() -> None:
    registry = _registry()
    session = SessionState(
        session_id="session-1",
        toolsets_enabled=["web"],
    )

    policy = resolve_toolset_policy(
        payload={},
        session_state=session,
        config={"toolsets_enabled": ["file"]},
        registry=registry,
    )

    assert policy.source == "session"
    assert policy.enabled_toolsets == ("web",)


def test_default_policy_does_not_narrow_and_excludes_opt_in() -> None:
    # 三级 override 都没给时的默认契约：不收窄（enabled_toolsets 为 None，
    # select_tools 直接跳过过滤），只减掉 opt-in 那组。disabled 里那条是
    # _with_opt_in_excluded 兜底补的，不是默认 policy 自带的
    policy = resolve_toolset_policy(
        payload={"message": "please edit this file"},
        session_state=None,
        config={},
        registry=get_default_tool_registry(),
    )

    assert policy.source == DEFAULT_POLICY_SOURCE
    assert policy.enabled_toolsets is None
    assert policy.disabled_toolsets == (OPT_IN_PRESET,)


def test_validate_rejects_unknown_toolset_or_preset() -> None:
    registry = _registry()
    policy = resolve_toolset_policy(
        payload={"enabled_toolsets": ["file"]},
        session_state=None,
        config={},
        registry=registry,
    )

    assert validate_toolset_policy(policy, registry) is policy
    with pytest.raises(ToolsetPolicyError, match="unknown toolset"):
        resolve_toolset_policy(
            payload={"enabled_toolsets": ["missing"]},
            session_state=None,
            config={},
            registry=registry,
        )


def test_expand_preset_records_exact_registered_tools() -> None:
    registry = _registry()

    expanded = expand_toolset_names(("workspace_edit",), registry)

    assert PRESET_VERSION == "phase-b-v1"
    assert expanded == frozenset({"file_read", "file_patch"})


def test_load_toolset_runtime_config_reads_explicit_toolsets(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "toolsets:\n  enabled:\n    - web\n  disabled:\n    - terminal\n",
        encoding="utf-8",
    )

    loaded = load_toolset_runtime_config(config)

    assert loaded == {
        "toolsets_enabled": ["web"],
        "toolsets_disabled": ["terminal"],
    }


def _preset_covered_tool_names() -> set[str]:
    covered: set[str] = set()
    for preset, names in PRESET_TOOL_NAMES.items():
        # full 是"放行全部"的特例（空元组由 expand_toolset_names 特判），
        # 拿它算覆盖率会让任何漏配都自动变成已覆盖
        if preset != "full":
            covered |= set(names)
    return covered


def _expanded(preset: str) -> frozenset[str]:
    expanded = expand_toolset_names((preset,), get_default_tool_registry())
    assert expanded is not None
    return expanded


def _selected_names(message: str) -> set[str]:
    registry = get_default_tool_registry()
    policy = resolve_toolset_policy(
        payload={"trigger": "chat", "message": message, "task": message},
        session_state=None,
        config={},
        registry=registry,
    )
    selection = select_tools(registry, policy=policy)
    return {item.name for item in selection.selected_definitions}


def _default_policy(registry: ToolRegistry) -> ToolsetPolicy:
    # 三级 override 都不提供时走到的默认 policy
    return resolve_toolset_policy(
        payload={},
        session_state=None,
        config={},
        registry=registry,
    )


def _names(registry: ToolRegistry, policy: ToolsetPolicy) -> set[str]:
    selection = select_tools(registry, policy=policy)
    return {item.name for item in selection.selected_definitions}


def _selected_for(payload: dict[str, object]) -> set[str]:
    # 走完整的 resolve → select 链路，验的是默认路径的最终产出而不是中间的 policy 对象
    registry = get_default_tool_registry()
    policy = resolve_toolset_policy(
        payload=payload,
        session_state=None,
        config={},
        registry=registry,
    )
    return _names(registry, policy)


def _selected_with(policy: ToolsetPolicy) -> set[str]:
    registry = get_default_tool_registry()
    resolved = resolve_toolset_policy(
        payload={"toolset_policy": policy_to_mapping(policy)},
        session_state=None,
        config={},
        registry=registry,
    )
    return _names(registry, resolved)


def _selected_enabling(*names: str) -> set[str]:
    # 模拟 /toolsets enable <names>：显式收窄，兜底层不再介入
    return _selected_with(ToolsetPolicy(enabled_toolsets=names, source="test"))


# opt-in 组里本机 model_visible 为真的那几个。screenshot / ocr 缺依赖时按 hidden 排除，
# 那是可用性边界不是 policy 层的排除，不能拿来验 toolset_disabled
_OPT_IN_VISIBLE = (
    "secret_use",
    "secret_list_names",
    "clipboard_read",
    "clipboard_write",
    "redact",
)


# memory_query 现身的全部预设。同类只读检索工具要跟它同进退，逐个预设对齐时复用这份清单
_MEMORY_SEARCH_PRESETS = (
    "safe_read",
    "research",
    "browser_research",
    "automation",
    "workspace_edit",
)


def test_opt_in_preset_has_no_silent_typos() -> None:
    # opt-in 清单从常量升格成预设后，没有任何东西再核对它的名字了。而
    # expand_toolset_names 结尾的 `if tool in all_tools` 会把拼错的名字静默丢掉：
    # "secret_use" 打成 "secret_used" → 展开时消失 → 它不再被排除 → 密钥工具进默认
    # 目录，而 secret_use 是 risk=safe 不弹审批。测试全绿而密钥已在模型手里。
    # 只查"名字在 registry 里"不够，数量对比才直接盯住那句静默过滤
    registry = get_default_tool_registry()
    declared = PRESET_TOOL_NAMES[OPT_IN_PRESET]
    registered = {item.name for item in registry.list_definitions()}

    assert set(declared) <= registered, sorted(set(declared) - registered)
    assert len(_expanded(OPT_IN_PRESET)) == len(declared)


def test_retired_and_bridge_tools_stay_invisible_to_the_model() -> None:
    # 内部兼容桥仍不可见；已实现的协作通过能力目录发现
    registry = get_default_tool_registry()

    for name in ("echo", "inspect"):
        definition = registry.get(name)
        assert definition is not None, name
        assert definition.model_visible is False, name


def test_no_preset_claims_a_statically_invisible_tool() -> None:
    # 给 model_visible 恒为 False 的工具配预设是死配置：select_tools 在 policy
    # 之前就按 hidden 丢掉它，配了永远选不中，只会让读表的人误以为该能力可用。
    # notification_send 和 vision 三件套是 model_visible=available，装了依赖就该可见，
    # 它们在预设里是正确配置，不在本条管辖范围
    assert not (_preset_covered_tool_names() & {"echo", "inspect", "delegate"})


def test_opt_in_only_tools_are_claimed_by_their_physical_toolset() -> None:
    # opt-in 组"默认不召但可显式启用"的逃生口有两条：预设名 opt_in_only，以及
    # expand_toolset_names 对非预设名回落物理 toolset 匹配。这里验后一条——
    # 真到不到得了模型还受 model_visible=available 约束（vision 三件套缺依赖时
    # 召不出来），那是可用性边界不是本清单的定性问题
    assert {"ocr", "redact", "screenshot"} <= _expanded("vision")
    assert {"secret_use", "secret_list_names"} <= _expanded("secret")


def test_memory_manage_travels_with_memory_query() -> None:
    # "搜到错的记忆→归档它"是 memory_manage 的主用法。search 在场而 archive 缺席
    # 就是在预设层重造一次断链，正是 ⑥ 卡补 search 回执 id 时要修的那条链
    for preset in _MEMORY_SEARCH_PRESETS:
        expanded = _expanded(preset)
        assert ("memory_query" in expanded) == ("memory_manage" in expanded), preset


def test_skill_search_travels_with_memory_query() -> None:
    # 同为只读检索工具，⑧乙 立卡时的定性就是"照 memory_query 办"
    for preset in _MEMORY_SEARCH_PRESETS:
        expanded = _expanded(preset)
        assert ("memory_query" in expanded) == ("skill_search" in expanded), preset


def test_skill_run_travels_with_code_execution() -> None:
    # ⑧甲 已定 skill_run 是受 CONFIRM 闸门的执行工具，与 code_execution_tool 同性质
    for preset in ("terminal", "workspace_edit"):
        expanded = _expanded(preset)
        assert ("code_execution_tool" in expanded) == ("skill_run" in expanded), preset


def test_inspect_stays_out_of_every_preset() -> None:
    # inspect 靠 model_visible=False 挡住不是漏配：tests/test_readonly_tool_registry.py
    # 有 "inspect not in model_visible" 的既定契约。它的只读兄弟 file_read/list/grep
    # 在预设里，看着像该跟进，但跟进就是死配置——本条钉住别再犯。
    registry = get_default_tool_registry()

    assert registry.get("inspect") is not None
    assert registry.get("inspect").model_visible is False
    assert "inspect" not in _preset_covered_tool_names()


def test_workspace_edit_can_verify_its_own_edits() -> None:
    # 改代码却拿不到执行能力，等于改完验证不了；拿不到记忆检索，等于召不回项目约定
    expanded = _expanded("workspace_edit")

    assert {"terminal_tool", "code_execution_tool"} <= expanded
    assert {"memory_query", "todo"} <= expanded


def test_archive_memory_request_can_discover_and_load_the_archive_tool() -> None:
    """归档能力可被发现和加载，不依赖输入关键词预选；传参：无；返回：无。"""
    from runtime.capability_catalog import browse_tools
    from runtime.lease import from_trigger

    registry = get_default_tool_registry()
    assert "capabilities" in _selected_names("把刚才记错的那条记忆归档掉")
    result = browse_tools(
        registry,
        {"action": "load", "name": "memory_manage"},
        lease=from_trigger("user"),
        policy=_default_policy(registry),
    )
    loaded = select_tools(
        registry,
        policy=_default_policy(registry),
        loaded_tools=result["loaded_tool_names"],
    )
    assert "memory_manage" in {item.name for item in loaded.selected_definitions}


def test_edit_code_request_actually_gets_execution_tools() -> None:
    # 端到端钉死"改完跑不了测试"
    assert "terminal_tool" in _selected_names("帮我改一下这个文件里的函数")


def test_opt_in_exclusion_survives_session_disable_override() -> None:
    # 四级来源是整体替换不是逐字段合并：用户敲 /toolsets disable web 只想"这轮别上网"，
    # 但 session policy 会把默认那条 disabled=(opt_in_only,) 整个冲掉，于是密钥和剪贴板
    # 反而被送进模型目录。这三个工具 risk=safe，调用不弹审批，进了目录就没有第二道确认
    registry = get_default_tool_registry()

    policy = resolve_toolset_policy(
        payload={},
        session_state=SessionState(session_id="session-1", toolsets_disabled=["web"]),
        config={},
        registry=registry,
    )

    assert policy.disabled_toolsets is not None
    assert "web" in policy.disabled_toolsets
    assert "opt_in_only" in policy.disabled_toolsets
    assert not ({"secret_use", "clipboard_read", "redact"} & _names(registry, policy))


def test_opt_in_exclusion_survives_config_disable_override() -> None:
    # ~/.reins/config.yaml 里写一行 toolsets_disabled 走的是同一条整体替换路径
    registry = get_default_tool_registry()

    policy = resolve_toolset_policy(
        payload={},
        session_state=None,
        config={"toolsets_disabled": ["web"]},
        registry=registry,
    )

    assert policy.disabled_toolsets is not None
    assert "opt_in_only" in policy.disabled_toolsets
    assert not ({"secret_use", "clipboard_read", "redact"} & _names(registry, policy))


def test_every_model_visible_non_opt_in_tool_is_discoverable() -> None:
    # 能力目录覆盖所有已授权可用工具；是否已加载只影响当前请求
    registry = get_default_tool_registry()
    policy = _default_policy(registry)
    opt_in = expand_toolset_names(("opt_in_only",), registry) or frozenset()

    selection = select_tools(registry, policy=policy, lease=None, include_deferred=True)
    selected = {item.name for item in selection.selected_definitions}
    reports, _summary = registry.get_visibility_report()
    expected = {
        report.name
        for report in reports
        if report.model_visible
        and report.available
        and report.name not in opt_in
        and report.risk_level is not ToolRisk.DENY
    }

    assert expected - selected == set(), (
        f"这些工具可用、非 opt-in，却无法通过能力目录发现: {sorted(expected - selected)}。"
    )


def test_selection_is_identical_regardless_of_task_text() -> None:
    # 旧的关键词 if 链有序互斥单选，一句话里同时出现多个词时只命中最靠前那条。
    # 第五条把六条规则的关键词全塞进同一句：旧代码必然选出 workspace_edit，
    # 新代码必须无视全部。trigger=cron 覆盖旧链唯一不看文本的分支，
    # 也是最容易被人以"定时任务是特殊情况"为由加回来的一条
    payloads = (
        {},
        {"message": "帮我查一下为什么这里要这么改"},
        {"message": "跑一下测试看看过没过"},
        {"trigger": "cron", "message": "例行巡检"},
        {"message": "写 改 browser 点击 截图 网页 查 解释 research edit patch"},
    )

    results = [_selected_for(payload) for payload in payloads]

    assert all(item == results[0] for item in results), [sorted(x) for x in results]


def test_lookup_intent_can_discover_web_search() -> None:
    """查询与修改混合表达不能撤销网络发现能力；参数：无；返回：无。"""
    from runtime.capability_catalog import browse_tools

    registry = get_default_tool_registry()
    policy = resolve_toolset_policy(
        payload={"message": "帮我查一下为什么这里要这么改"},
        session_state=None,
        config={},
        registry=registry,
    )
    found = browse_tools(
        registry, {"action": "search", "query": "网页"}, lease=None, policy=policy
    )
    assert "web_search" in {item["name"] for item in found["items"]}


def test_run_tests_intent_keeps_terminal() -> None:
    # 立卡动机第二条：一个关键词都不命中，落到最保守那档，连终端都没有，跑不了测试
    assert "terminal_tool" in _selected_for({"message": "跑一下测试看看过没过"})


def test_previously_orphaned_tools_remain_discoverable_without_preset_membership() -> (
    None
):
    """没有被预设枚举的能力仍可按需发现；传参：无；返回：无。"""
    registry = get_default_tool_registry()
    catalog = select_tools(
        registry, policy=_default_policy(registry), include_deferred=True
    )
    assert {"memory_manage", "skill_run", "skill_search", "schedule"} <= {
        item.name for item in catalog.selected_definitions
    }


def test_opt_in_tools_are_excluded_as_toolset_disabled() -> None:
    # 密钥、剪贴板、脱敏默认不进目录，且排除原因必须是 policy 层的 toolset_disabled，
    # 不能是 hidden / unavailable——后两者意味着它们根本召不回来，那是可用性问题
    registry = get_default_tool_registry()
    selection = select_tools(registry, policy=_default_policy(registry))
    reasons = {item.name: item.reason for item in selection.excluded}

    for name in _OPT_IN_VISIBLE:
        assert reasons.get(name) == "toolset_disabled", (name, reasons.get(name))


def test_enable_opt_in_preset_brings_them_back() -> None:
    # R5 的逃生口：默认不召不等于召不回来
    selected = _selected_enabling(OPT_IN_PRESET)

    assert {"secret_use", "secret_list_names", "clipboard_read"} <= selected


def test_explicit_enable_is_not_backfired_by_the_opt_in_floor() -> None:
    # 兜底层若在用户已显式收窄时仍叠加排除，enabled 与 disabled 两道检查会互相抵消，
    # /toolsets enable secret 的结果是一个工具都选不出来
    secret_only = _selected_enabling("secret")

    assert {"secret_use", "secret_list_names"} <= secret_only

    # full 的字面意思就是全部，显式要了就连 opt-in 一起给
    assert {"secret_use", "clipboard_read"} <= _selected_enabling("full")


def test_explicit_narrowing_still_wins() -> None:
    # 默认全发只换掉默认那一档，显式收窄通道原样保留
    assert _selected_enabling("research") == set(PRESET_TOOL_NAMES["research"])


def test_delegate_requires_loading_but_remains_discoverable() -> None:
    """协作不占每轮基础定义，但可以通过目录加载；传参：无；返回：无。"""
    assert "delegate" not in _selected_for({})
    registry = get_default_tool_registry()
    loaded = select_tools(
        registry, policy=_default_policy(registry), loaded_tools={"delegate"}
    )
    assert "delegate" in {item.name for item in loaded.selected_definitions}


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_tool("file_read", TOOLSET_FILE))
    registry.register(_tool("file_patch", TOOLSET_FILE))
    registry.register(_tool("web_search", TOOLSET_WEB))
    return registry


def _tool(name: str, toolset: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} description",
        parameters={},
        toolset=toolset,
        risk_level=TOOL_RISK_SAFE,
        readonly=True,
        target_scope_rule=TARGET_SCOPE_LOGICAL,
        source=TOOL_SOURCE_BUILTIN,
        model_visible=True,
        idempotent=IDEMPOTENT_YES,
        executor=lambda _args: {"ok": True},
    )
