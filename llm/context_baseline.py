"""【上下文】【稳定请求】保存已采用的历史/知识表示，当前状态始终重新读取。

作者：xxx
时间：2026-10-01 14:00:00
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from typing import TYPE_CHECKING

from context.materials import ContextMaterial, render_materials
from context.selection_store import content_digest as digest, selection_scope
from runtime.persistence import RuntimeStore

if TYPE_CHECKING:
    from llm.model_request import ComposedRequest

RENDER_VERSION = 1


def prepare_baseline(
    context: Mapping[str, object], *, contract: Mapping[str, object]
) -> dict[str, Any]:
    """按当前有效材料选择基线/增量；参数：上下文与请求合同；返回：候选，不写采用事实。"""
    materials = context.get("context_materials", ())
    history = context.get("history_materials", ())
    selection = context.get("material_selection", {})
    if (
        not isinstance(materials, (list, tuple))
        or not isinstance(history, (list, tuple))
        or not isinstance(selection, Mapping)
    ):
        raise ValueError("baseline materials and selection must be structured")
    unique = {
        item.identity: item
        for item in (*history, *materials)
        if isinstance(item, ContextMaterial)
    }
    rows = [
        _material_row(item, selection) for item in unique.values() if not item.protected
    ]
    scope = selection_scope(context)
    root = str(context.get("data_root", ""))
    key = digest(scope)
    previous = None
    if root and scope["session_id"]:
        with RuntimeStore(root).snapshot() as source:
            previous = source.get("context_baseline", key)
    boundary = digest(
        {"scope": scope, "contract": contract, "render_version": RENDER_VERSION}
    )
    reason = _rebuild_reason(previous, boundary, rows)
    if reason:
        baseline, delta = rows, []
    else:
        assert previous is not None
        baseline, delta = previous["baseline"], list(previous["delta"])
        known = {row["identity"] for row in (*baseline, *delta)}
        delta.extend(row for row in rows if row["identity"] not in known)
    return {
        "schema_version": 1,
        "data_root": root,
        "session_id": scope["session_id"],
        "record_id": key,
        "scope": scope,
        "contract": boundary,
        "baseline": baseline,
        "delta": delta,
        "model_target_identity": contract.get("model"),
        "effective_requirements_digest": digest(context.get("effective_requirements")),
        "baseline_id": digest([boundary, baseline]),
        "delta_id": digest(delta),
        "local_reused": reason is None,
        "invalidation_reason": reason,
        "baseline_text": _render(baseline),
        "delta_text": _render(delta),
    }


def _material_row(
    item: ContextMaterial, selection: Mapping[str, object]
) -> dict[str, object]:
    """冻结一种带来源的表示；参数：材料和选档；返回：可重放的正文及来源。"""
    representation = str(selection.get(item.identity, "full"))
    return {
        "identity": item.identity,
        "version": item.version,
        "scope": item.scope,
        "source": item.source,
        "representation": representation,
        "source_digest": digest(item.text),
        "read_reference": item.reference,
        "text": render_materials((item,), {item.identity: representation}),
    }


def _rebuild_reason(
    previous: Mapping[str, Any] | None,
    contract: str,
    rows: Sequence[Mapping[str, object]],
) -> str | None:
    """对已选表示逐个核对版本/权限，新增只入增量；参数：前版和当前材料；返回：重建原因。"""
    if previous is None:
        return "first_adoption"
    if previous.get("schema_version") != 1:
        raise ValueError("unsupported context baseline version")
    if previous["contract"] != contract:
        return "scope_or_request_contract_changed"
    current = {row["identity"]: row for row in rows}
    for row in (*previous["baseline"], *previous["delta"]):
        if current.get(row["identity"]) != row:
            return "material_revised_removed_or_reselected"
    return None


def _render(rows: Sequence[Mapping[str, object]]) -> str:
    """保持已选顺序渲染，不能混入本轮预算；参数：冻结材料；返回：稳定正文。"""
    return "\n\n".join(str(row["text"]) for row in rows)


def baseline_evidence(state: Mapping[str, Any]) -> dict[str, object]:
    """提供可定位版本，正文由实际请求原件持有；参数：候选；返回：采用清单。"""
    return {
        key: state[key]
        for key in (
            "baseline_id",
            "delta_id",
            "scope",
            "local_reused",
            "invalidation_reason",
        )
    } | {
        "materials": [
            {key: value for key, value in row.items() if key != "text"}
            for row in (*state["baseline"], *state["delta"])
        ],
        "adoption_boundary": "local_dispatch",
        "provider_delivery": "unknown_until_response",
        "provider_cache": "unknown_until_provider_report",
    }


def commit_baseline(state: Mapping[str, Any], *, request_id: str) -> None:
    """在已留存请求的本地派发边界提交采用；参数：候选和请求身份；返回：无，不证明远端收到。"""
    if not state.get("data_root") or not state.get("session_id"):
        return
    saved = {key: value for key, value in state.items() if key != "data_root"}
    saved["adopted_request_id"] = request_id
    saved["adoption_boundary"] = "local_dispatch"
    saved["provider_delivery"] = "unknown_until_response"
    with RuntimeStore(str(state["data_root"])).transaction() as batch:
        batch.put(
            "context_baseline",
            str(state["record_id"]),
            saved,
            session_id=str(state["session_id"]),
        )


def adopt_request_context(composed: ComposedRequest, *, request_id: str) -> None:
    """实际发送边界采用准备结果；参数：最终请求和逻辑身份；返回：无。"""
    from context.selection_store import MaterialSelectionStore
    from llm.prompt_snapshot import adopt_prompt_snapshot

    state = composed.context_baseline
    selection = composed.material_selection
    root = str(state.get("data_root", "")) if state else ""
    if not root or not state or not state.get("session_id"):
        return
    # 1. 【上下文】【请求采用】基线与选档在同一发布批次生效，预览不会推进这两个记录
    with RuntimeStore(root).transaction():
        commit_baseline(state, request_id=request_id)
        adopt_prompt_snapshot(
            composed.stable_prompt_snapshot,
            data_root=root,
            session_id=str(state["session_id"]),
        )
        if selection is not None:
            MaterialSelectionStore(root).adopt(selection, request_id=request_id)
