"""【上下文】【材料选档】持久保存已发送表示与主动释放决定。

作者：xxx
时间：2026-10-01 14:00:00
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from runtime.persistence import RuntimeStore


def content_digest(value: object) -> str:
    """计算版本化表示摘要；参数：JSON兼容内容；返回：SHA256。"""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def selection_scope(context: Mapping[str, object]) -> dict[str, str]:
    """固定会话/分支/权限范围，追加消息不换范围；参数：本轮上下文；返回：范围身份。"""
    session_id = str(context.get("session_id", ""))
    branch = str(context.get("context_branch_id", ""))
    lease = context.get("capability_lease")
    capabilities = getattr(lease, "capabilities", {})
    return {
        "session_id": session_id,
        "branch_id": branch,
        "workspace": str(context.get("tool_working_directory", "")),
        "authority": content_digest(capabilities),
    }


def material_key(row: Mapping[str, Any]) -> str:
    """同名异版或异范围材料不能共享释放决定；参数：来源行；返回：精确选档键。"""
    return content_digest(
        [row[key] for key in ("identity", "version", "scope", "source_digest")]
    )


class MaterialSelectionStore:
    """沿用 RuntimeStore 保存表示，不拥有消息、记忆或文件原件。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定独立数据空间；参数：数据根；返回：无。"""
        self.db = RuntimeStore(data_root)

    def read(self, scope: Mapping[str, str]) -> dict[str, Any]:
        """读取本范围已采用目录与决定；参数：范围；返回：已保存状态或空。"""
        with self.db.snapshot() as source:
            state = source.get(
                "context_material_selection", content_digest(dict(scope))
            )
        if state is not None and state.get("schema_version") != 1:
            raise ValueError("unsupported context material selection version")
        return state or {}

    def adopt(self, state: Mapping[str, Any], *, request_id: str) -> None:
        """请求发送时更新目录，保留并发释放；参数：准备结果和请求ID；返回：无。"""
        scope = state["scope"]
        identity = content_digest(scope)
        with self.db.transaction() as batch:
            previous = batch.get("context_material_selection", identity) or {}
            decisions = dict(previous.get("decisions", {}))
            for row in state["catalog"]:
                key = material_key(row)
                if (
                    key not in decisions
                    or decisions[key].get("reason") != "explicit_release"
                ):
                    decisions[key] = dict(row)
            batch.put(
                "context_material_selection",
                identity,
                {
                    "schema_version": 1,
                    "scope": scope,
                    "catalog": state["catalog"],
                    "decisions": decisions,
                    "adopted_request_id": request_id,
                },
                session_id=scope["session_id"],
            )

    def release(
        self, scope: Mapping[str, str], row: Mapping[str, Any], *, operation_id: str
    ) -> None:
        """短提交再次核对已采用版本；参数：范围、引用表示及操作ID；返回：无，过期报错。"""
        identity, key = content_digest(dict(scope)), material_key(row)
        with self.db.transaction() as batch:
            current = batch.get("context_material_selection", identity)
            latest = (
                next(
                    (item for item in current["catalog"] if material_key(item) == key),
                    None,
                )
                if current
                else None
            )
            if current is None or latest is None:
                raise ValueError("material is no longer in the adopted request")
            # 1. 【上下文】【释放发布】原件读取期间的新请求可能已将同版材料用于当前要求或补读
            if latest["protected"]:
                raise ValueError("material is protected in the current adopted request")
            decisions = {
                **current["decisions"],
                key: {
                    **row,
                    "reason": "explicit_release",
                    "operation_id": operation_id,
                },
            }
            batch.put(
                "context_material_selection",
                identity,
                {**current, "decisions": decisions},
                session_id=scope["session_id"],
            )


def prepared_selection(context: Mapping[str, object]) -> dict[str, Any]:
    """准备只读取，不发布采用记录；参数：上下文；返回：范围及已有选档。"""
    scope = selection_scope(context)
    root = str(context.get("data_root", ""))
    previous = (
        MaterialSelectionStore(root).read(scope)
        if root and scope["session_id"] and scope["branch_id"]
        else {}
    )
    return {
        "data_root": root,
        "scope": scope,
        "decisions": previous.get("decisions", {}),
    }


def selection_candidate(
    prepared: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """把最终材料目录交给发送边界；参数：读取状态及实际表示；返回：待采用载荷。"""
    return {
        "data_root": prepared["data_root"],
        "scope": prepared["scope"],
        "catalog": list(rows),
    }
