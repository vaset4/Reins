"""能力发现和加载结果连接到下一次模型请求，不另建会话事实源。

作者：xxx
时间：2026-09-14 21:00:00
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from llm.messages import AgentMessage, AssistantMessage, ToolCallPart, ToolResultMessage
from llm.tool_selection import canonical_tool_schema_hash, select_tools
from llm.toolset_policy import ToolsetPolicy
from runtime.lease import Lease
from runtime.persistence import RuntimeStore
from tools.catalog import DEFAULT_CATALOG_PAGE_SIZE, catalog_page, matches_query
from tools.tool_registry import ToolDefinition, ToolRegistry


def browse_tools(
    registry: ToolRegistry,
    args: Mapping[str, object],
    *,
    lease: Lease | None,
    policy: ToolsetPolicy | None,
) -> dict[str, object]:
    """从同一快照发现和成组加载已获授权能力；参数：目录、查询、权限及策略；返回：真实工具页或加载结果。"""
    snapshot = registry.snapshot()
    selection = select_tools(
        snapshot, lease=lease, policy=policy, include_deferred=True
    )
    definitions = {item.name: item for item in selection.selected_definitions}
    all_definitions = {item.name: item for item in snapshot.list_definitions()}
    reports, _summary = snapshot.get_visibility_report()
    visible = {row.name for row in reports if row.model_visible}
    domain, query = str(args.get("domain", "")), str(args.get("query", ""))
    unavailable: list[dict[str, object]] = [
        {
            "name": row.name,
            "domain": all_definitions[row.name].domain
            or all_definitions[row.name].toolset,
            "reason": row.reason,
            "detail": row.detail,
        }
        for row in selection.excluded
        if row.name in visible
        and (
            not domain
            or (all_definitions[row.name].domain or all_definitions[row.name].toolset)
            == domain
        )
    ]
    domains = sorted(
        {
            item.domain or item.toolset
            for item in all_definitions.values()
            if item.name in visible
        }
    )
    groups = [
        {
            "domain": name,
            "available_count": sum(
                (item.domain or item.toolset) == name for item in definitions.values()
            ),
            "load_action": {"action": "load", "domain": name},
        }
        for name in domains
        if not domain or name == domain
    ]
    if args["action"] == "load":
        return load_tools(snapshot, definitions, args, unavailable=unavailable)
    matched = [
        item
        for item in definitions.values()
        if (not domain or (item.domain or item.toolset) == domain)
        and matches_tool_query(query, item)
    ]
    matched.sort(key=lambda item: (item.name.casefold() != query.casefold(), item.name))
    items = [
        {
            "name": item.name,
            "description": item.description,
            "toolset": item.toolset,
            "domain": item.domain or item.toolset,
            "search_terms": list(item.search_terms),
            "semantics": list(item.effective_semantics),
            "readonly_actions": list(item.readonly_actions),
            "definition_version": snapshot.definition_version(item.name),
            "load_action": {"action": "load", "name": item.name},
        }
        for item in matched
    ]
    page = catalog_page(
        items,
        kind="tools",
        query=query,
        cursor=cast(str | None, args.get("cursor")),
        limit=cast(int, args.get("limit", DEFAULT_CATALOG_PAGE_SIZE)),
        source_version=snapshot.version + ":" + domain,
    )
    return {
        **page,
        "registry_version": snapshot.version,
        "sources": snapshot.source_status(),
        "domains": groups,
        "unavailable": unavailable,
        "available_count": len(definitions),
    }


def matches_tool_query(query: str, definition: ToolDefinition) -> bool:
    """名称精确/字面匹配与声明的中文业务词共用，不调用额外模型；参数：查询、工具；返回：是否相关。"""
    text = " ".join(
        (
            definition.name,
            definition.description,
            definition.toolset,
            definition.domain,
            *definition.search_terms,
        )
    )
    return matches_query(query, text) or any(
        term.casefold() in query.casefold() for term in definition.search_terms
    )


def load_tools(
    snapshot: ToolRegistry,
    definitions: dict[str, ToolDefinition],
    args: Mapping[str, object],
    *,
    unavailable: list[dict[str, object]],
) -> dict[str, object]:
    """将单工具或同领域当前可用定义一起返回，加载不执行也不授权；参数：快照、允许集合、选择、不可用说明；返回：加载回执。"""
    name, domain = args.get("name"), args.get("domain")
    if (name is None) == (domain is None):
        raise ValueError("load requires exactly one tool name or domain")
    if args.get("registry_version") not in (None, snapshot.version):
        raise ValueError(
            "registry changed; inspect the current tool definition before loading"
        )
    selected = (
        ([definitions[str(name)]] if name in definitions else [])
        if name is not None
        else [
            item
            for item in definitions.values()
            if (item.domain or item.toolset) == domain
        ]
    )
    if not selected:
        raise ValueError(
            f"tool/domain is not available under current authorization: {name or domain}"
        )
    selected.sort(key=lambda item: item.name)
    result = {
        "registry_version": snapshot.version,
        "loaded_tool_names": [item.name for item in selected],
        "tools": [item.format_for_openai_tool() for item in selected],
        "unavailable": unavailable,
        "domain": domain,
        "next_step": "Loaded native schemas are available in the next request; current permissions still apply.",
    }
    if name is not None:
        item = selected[0]
        result.update(
            definition_version=snapshot.definition_version(item.name),
            schema_hash=canonical_tool_schema_hash(item),
            tool=item.format_for_openai_tool(),
            semantics=list(item.effective_semantics),
        )
    return result


def loaded_tools_from_operations(
    data_root: Path,
    session_id: str,
    messages: Sequence[AgentMessage],
) -> frozenset[str]:
    """从当前分支原件派生已加载及待处理操作入口；传参：存储、会话及未压缩消息；返回：所需工具名，不授予权限。"""
    calls = {
        item.call_id
        for item in messages
        if isinstance(item, ToolResultMessage)
        and item.tool_name == "capabilities"
        and item.status == "success"
    }
    branch_calls = {
        part.call_id
        for item in messages
        if isinstance(item, AssistantMessage)
        for part in item.content
        if isinstance(part, ToolCallPart)
    }
    if not branch_calls:
        return frozenset()
    loaded: set[str] = set()
    with RuntimeStore(data_root).snapshot() as source:
        # 1. 【工具目录】【加载恢复】先用原件元信息选分支和状态，不展开已完成工具的大正文
        for record in source.list_raw("tool_operation", session_id=session_id):
            row = record.payload
            call_id = row["call"]["call_id"]
            if call_id not in branch_calls:
                continue
            if row["state"] in {"started", "unknown", "not_started", "waiting_user"}:
                loaded.update(("operation_status", "resume_operation"))
            if call_id in calls:
                result = source.get("tool_operation", record.record_id)
                assert result is not None
                loaded.update(
                    str(name)
                    for name in result.get("result", {})
                    .get("meta", {})
                    .get("loaded_tool_names", [])
                )
    return frozenset(loaded)
