"""在同一个实际请求窗口选择可补读材料，原件由现有存储持有。

作者：xxx
时间：2026-09-25 20:00:00
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from context.token_estimate import estimate_tokens
from context.window import request_budget
from context.selection_store import (
    material_key,
    content_digest,
    prepared_selection,
    selection_candidate,
)
from llm.messages import ToolResultMessage, model_visible_text
from runtime.secret_redaction import redact_value

if TYPE_CHECKING:
    from llm.model_request import ComposedRequest


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextMaterial:
    """表示本次可选的正文和读取入口；传参：来源身份、版本、范围与表示；返回：不可变请求材料。"""

    identity: str
    source: str
    version: str
    scope: str
    text: str
    reference: str = ""
    pinned: bool = False
    protected: bool = False
    representations: tuple[tuple[str, str], ...] = ()


def render_materials(
    materials: tuple[ContextMaterial, ...], selected: Mapping[str, str]
) -> str:
    """按来源渲染已选择的材料，未知状态不当正文；传参：本轮材料与表示选择；返回：模型可读内容。"""
    groups: dict[str, list[str]] = {}
    for item in materials:
        representation = selected.get(item.identity, "full")
        if representation not in {
            "full",
            "reference",
            "in_history",
            *dict(item.representations),
        }:
            raise ValueError("unknown context material representation")
        text = dict(item.representations).get(
            representation, item.text if representation == "full" else item.reference
        )
        groups.setdefault(item.source, []).append(text)
    return "\n\n".join(
        f"recall_{source}=\n" + "\n\n".join(rows) for source, rows in groups.items()
    )


def _collect_materials(context: Mapping[str, object]) -> dict[str, ContextMaterial]:
    """核对材料身份并保护扩展内容，重复来源只保留一份；传参：组装上下文；返回：同一版本的候选。"""
    raw = context.get("context_materials", ())
    if not isinstance(raw, (tuple, list)) or any(
        not isinstance(item, ContextMaterial) for item in raw
    ):
        raise ValueError(
            "context materials must have explicit identities and representations"
        )
    history = context.get("history_materials", ())
    levels = context.get("history_material_levels", {})
    if not isinstance(history, (tuple, list)) or any(
        not isinstance(item, ContextMaterial) for item in history
    ):
        raise ValueError(
            "history materials must have explicit identities and representations"
        )
    if not isinstance(levels, Mapping):
        raise ValueError("history levels must be a mapping")
    materials = [
        replace(item, representations=tuple(levels[item.identity].items()))
        if item.identity in levels
        else item
        for item in (*history, *raw)
    ]
    extensions = context.get("extension_context", ())
    if not isinstance(extensions, (tuple, list)):
        raise ValueError("extension context must be a material sequence")
    for value in extensions:
        text = str(redact_value(value))
        version = hashlib.sha256(text.encode("utf-8")).hexdigest()
        materials.append(
            ContextMaterial(
                identity=f"extension:{version}",
                source="extension",
                version=version,
                scope=str(context.get("session_id", "")),
                text=text,
            )
        )
    unique: dict[str, ContextMaterial] = {}
    for item in materials:
        if item.identity in unique and unique[item.identity] != item:
            raise ValueError("one context material identity names conflicting content")
        unique[item.identity] = item
    return unique


def fit_materials(
    task: str,
    context: Mapping[str, object],
    prepare: Callable[[str, Mapping[str, object]], ComposedRequest],
    *,
    retain: Callable[[ContextMaterial], str],
) -> ComposedRequest:
    """先缩减可补读材料再交给对话压缩，同一预算保护要求；传参：组装器、上下文和原件写者；返回：请求与证据。"""
    unique = _collect_materials(context)
    materials = list(unique.values())
    saved = prepared_selection(context)
    loaded = _materials_in_history(context.get("conversation_history", ()))
    selected = {
        item.identity: "in_history"
        if item.identity in loaded and item.reference
        else "full"
        for item in materials
    }
    for item in materials:
        previous = saved["decisions"].get(
            material_key(_material_evidence(item, "full"))
        )
        if (
            previous
            and not item.protected
            and item.identity not in loaded
            and previous["representation"] != "in_history"
        ):
            selected[item.identity] = previous["representation"]
            if previous.get("read_reference") and not item.reference:
                unique[item.identity] = replace(
                    item, reference=previous["read_reference"]
                )
    notices = (
        context.get("recall_notices", "")
        if "context_materials" in context
        else context.get("recall_context", "")
    )
    base = {
        **context,
        "context_materials": tuple(unique.values()),
        "material_selection": selected,
        "recall_notices": notices,
        "extension_context": (),
    }
    if "history_materials" in context:
        base["history_materials"] = ()
    composed = prepare(task, base)
    original = request_budget(composed.request, composed.context_window)
    # 1. 【上下文】【材料分配】先处理未显式选择的大材料，当前有效要求不参与竞争
    candidates = sorted(
        (
            item
            for item in unique.values()
            if not item.protected
            and selected[item.identity] not in {"reference", "in_history"}
        ),
        key=lambda item: (item.pinned, -estimate_tokens(item.text), item.identity),
    )
    for item in candidates:
        if (
            request_budget(composed.request, composed.context_window).required_total
            <= composed.context_window
        ):
            break
        # 2. 【上下文】【历史降档】同源档位已生成，容量变化只选档，不另起模型调用
        for level, _ in item.representations:
            if level not in {"P2", "P3", "P4"}:
                continue
            if estimate_tokens(dict(item.representations)[level]) >= estimate_tokens(
                dict(item.representations).get(selected[item.identity], item.text)
            ):
                continue
            selected = {**selected, item.identity: level}
            composed = prepare(task, {**base, "material_selection": selected})
            if (
                request_budget(composed.request, composed.context_window).required_total
                <= composed.context_window
            ):
                break
        if (
            request_budget(composed.request, composed.context_window).required_total
            <= composed.context_window
        ):
            break
        if not item.reference:
            item = replace(item, reference=retain(item))
            unique[item.identity] = item
        if estimate_tokens(item.reference) >= estimate_tokens(item.text):
            continue
        selected = {**selected, item.identity: "reference"}
        base = {
            **base,
            "context_materials": tuple(unique.values()),
            "material_selection": selected,
        }
        composed = prepare(task, base)
    evidence = [
        _material_evidence(item, selected[item.identity]) for item in unique.values()
    ]
    rows = [
        {**row, **({"retained_text": item.text} if not item.reference else {})}
        for row, item in zip(evidence, unique.values(), strict=True)
    ]
    tool_rows = cast(
        list[dict[str, Any]], (composed.material_selection or {}).get("catalog", [])
    )
    return replace(
        composed,
        material_selection=selection_candidate(saved, [*rows, *tool_rows]),
        trim_delta={
            **(composed.trim_delta or {}),
            "materials": evidence,
            "before_material_selection": original.evidence(),
        },
    )


def _material_evidence(item: ContextMaterial, representation: str) -> dict[str, object]:
    """按实际表示记录成本和选择原因；传参：材料及最终表示；返回：不复制正文的证据。"""
    reason = "current_requirement" if item.protected else "fits_window"
    if representation == "in_history":
        reason = "already_in_request"
    elif representation == "reference":
        reason = "window_pressure"
    return {
        "identity": item.identity,
        "source": item.source,
        "version": item.version,
        "scope": item.scope,
        "source_digest": content_digest(item.text),
        "representation": representation,
        "protected": item.protected,
        "explicit": item.pinned,
        "original_tokens": estimate_tokens(item.text),
        "final_tokens": estimate_tokens(
            dict(item.representations).get(
                representation,
                item.text if representation == "full" else item.reference,
            )
        ),
        "reason": reason,
        "read_reference": item.reference,
    }


def _materials_in_history(history: object) -> set[str]:
    """只按本轮实际消息中的成功读取身份去重；传参：本轮消息；返回：仍在请求中的来源，不读取使用次数。"""
    if not isinstance(history, (tuple, list)):
        raise ValueError("material history must be a message sequence")
    identities: set[str] = set()
    for message in history:
        if (
            not isinstance(message, ToolResultMessage)
            or message.tool_name != "skill_read"
            or message.status != "success"
        ):
            continue
        try:
            payload = json.loads(model_visible_text(message))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        candidates = [payload, payload.get("meta")]
        if isinstance(payload.get("output"), str):
            try:
                candidates.append(json.loads(payload["output"]))
            except json.JSONDecodeError:
                pass
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(
                candidate.get("resource_ref"), str
            ):
                identities.add(candidate["resource_ref"])
    return identities


def artifact_reference(material: ContextMaterial, artifact_id: str) -> str:
    """渲染已成功保存的临时材料入口；传参：来源及产物身份；返回：含版本的真实读取方法。"""
    return json.dumps(
        {
            "source": material.source,
            "identity": material.identity,
            "version": material.version,
            "representation": "reference",
            "read_action": {
                "tool": "read_artifact",
                "arguments": {"artifact_id": artifact_id, "mode": "full"},
            },
        },
        ensure_ascii=False,
    )
