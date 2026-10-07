"""【知识维护】【相关来源核验】使用记忆前检查其已引用文件，不扫描空闲工作区。

作者：xxx
时间：2026-10-01 14:20:00
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from memory.records import Memory
from runtime.persistence import RuntimeStore
from runtime.tool_operations import ToolOperationStore
from runtime.workspaces import WorkspaceStore
from tasks.ids import utc_now


def memory_source_validity(
    data_root: Path | str, memory: Memory, *, session_id: str | None
) -> dict[str, Any] | None:
    """解释相关来源变化，不用文件观察废除用户要求；参数：记忆/查询会话；返回：变化状态或空。"""
    if memory.state != "active":
        return None
    changes = changed_memory_sources(data_root, memory, session_id=session_id)
    if not changes:
        return None
    required = memory.type in {"rule", "preference"} and any(
        source.kind == "user_input" for source in memory.details.sources
    )
    meaning = (
        "用户要求仍有效；文件现状变化需核对，不能自行取消该要求"
        if required
        else "以下为历史知识，不能作为已确认的当前事实"
    )
    paths = ", ".join(change["path"] for change in changes)
    return {
        "state": "needs_verification",
        "changes": changes,
        "requirement_retained": required,
        "notice": f"[来源已变化，待核验；{meaning}。相关文件：{paths}]",
    }


def changed_memory_sources(
    data_root: Path | str, memory: Memory, *, session_id: str | None
) -> list[dict[str, Any]]:
    """只核对本次入选记忆已引用的文件字节；参数：记忆/查询会话；返回：变化证据，不调用模型。"""
    observations: dict[str, dict[str, Any]] = {}
    operations = ToolOperationStore(data_root)
    workspaces = WorkspaceStore(data_root)
    for source in (*memory.details.sources, *memory.verification):
        if source.kind != "tool_result" or not source.session_id or not source.run_id:
            continue
        row = operations.load(
            {
                "session_id": source.session_id,
                "run_id": source.run_id,
                "operation_id": source.reference,
            }
        )
        if row is None:
            raise FileNotFoundError(
                f"memory source operation is missing: {source.reference}"
            )
        meta = (row.get("result") or {}).get("meta") or {}
        if not meta.get("resolved_path") or not meta.get("content_sha256"):
            continue
        path = Path(meta["resolved_path"]).resolve()
        workspace = workspaces.for_session(source.session_id)
        if not path.is_relative_to(workspace.project_root):
            continue
        observations[str(path)] = {
            "path": str(path),
            "observed_sha256": meta["content_sha256"],
            "operation_id": source.reference,
        }
    changes = []
    for path_text, observation in observations.items():
        path = Path(path_text)
        try:
            with path.open("rb") as stream:
                current = hashlib.file_digest(stream, "sha256").hexdigest()
        except FileNotFoundError:
            current = "deleted"
        if current != observation["observed_sha256"]:
            changes.append({**observation, "current_sha256": current})
    if changes and session_id is not None:
        digest = hashlib.sha256(
            json.dumps(changes, sort_keys=True).encode()
        ).hexdigest()
        identity = f"evidence-{session_id}-{memory.memory_id}-{memory.version}-{digest}"
        database = RuntimeStore(data_root)
        with database.transaction() as batch:
            if batch.get("knowledge_maintenance", identity) is None:
                batch.put(
                    "knowledge_maintenance",
                    identity,
                    {
                        "record_type": "evidence",
                        "evidence_id": identity,
                        "memory_id": memory.memory_id,
                        "version": memory.version,
                        "changes": changes,
                        "session_id": session_id,
                        "observed_at": utc_now(),
                        "state": "needs_verification",
                    },
                    session_id=session_id,
                )
    return changes
