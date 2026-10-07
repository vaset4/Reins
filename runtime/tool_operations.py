"""工具操作的持久身份与结果，由会话和运行事实引用。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runtime.persistence import RuntimeStore
from runtime.types import RunToolsRequest, RunToolsResult
from tasks.ids import new_ulid, utc_now
from tools.file_resources import FILE_WRITERS


@dataclass(frozen=True, slots=True)
class ToolOperation:
    """冻结工具发起时的归属，恢复和迟到结果使用同一操作编号。"""

    request: RunToolsRequest
    call_id: str
    tool_name: str
    args: dict[str, object]
    task_id: str | None = None
    request_id: str = ""
    operation_id: str = ""
    execution_request: dict[str, object] | None = None
    origin: str = "model"
    registry_version: str = ""
    definition_version: str = ""
    resource: dict[str, object] | None = None


class ToolOperationStore:
    """原子提交操作记录；记录缺失或只有started时均不能推断现实执行成功。"""

    def __init__(self, data_root: Path | str) -> None:
        """指定既有Session存储根；传参：数据根；返回：无。"""
        self.root = Path(data_root)
        self._db = RuntimeStore(self.root)

    def write(self, identity: dict[str, str], payload: dict[str, object]) -> None:
        """保存操作最新事实，供恢复重建消息投影；传参：身份、事实；返回：无，IO失败直抛。"""
        with self._db.transaction() as batch:
            existing = self.load(identity)
            if (
                payload.get("state") == "unknown"
                and existing
                and existing.get("state") == "late_completed"
            ):
                return
            value = {**(existing or {}), **identity, **payload, "updated_at": utc_now()}
            result = value.get("result")
            if isinstance(result, dict) and result.get("content") == result.get(
                "output"
            ):
                value["result"] = {
                    key: item for key, item in result.items() if key != "content"
                }
            batch.put(
                "tool_operation",
                identity["operation_id"],
                value,
                session_id=identity["session_id"],
            )

    def create(
        self, identity: dict[str, str], payload: dict[str, object]
    ) -> dict[str, Any]:
        """按身份原子接纳动作意图，重复提交不改既有执行状态；传参：身份/初始记录；返回：持久记录。"""
        with self._db.transaction():
            row = self.load(identity)
            if row is not None:
                return row
            self.write(identity, payload)
            return {**identity, **payload}

    def load(self, identity: dict[str, str]) -> dict[str, Any] | None:
        """读取指定操作的已提交结果；传参：身份；返回：记录，缺失为None。"""
        with self._db.snapshot() as source:
            row = source.get("tool_operation", identity["operation_id"])
            if row is None:
                return None
            if (
                row["session_id"] != identity["session_id"]
                or row["run_id"] != identity["run_id"]
            ):
                raise ValueError("tool operation identity belongs to another run")
            return self._decode(row)

    def claim_retry(
        self,
        identity: dict[str, str],
        retry_id: str,
        *,
        expected_source: str | None = None,
    ) -> str:
        """副作用前认领一次恢复尝试，重复调用取得已有身份；传参：原操作/恢复编号；返回：唯一尝试身份。"""
        with self._db.transaction():
            row = self.load(identity)
            if row is None:
                raise ValueError("operation no longer exists")
            existing = row.get("retry_operation_id")
            if existing is not None and existing != retry_id:
                return str(existing)
            if expected_source is not None and retry_source(row) != expected_source:
                raise ValueError(
                    "operation changed after recovery authorization; inspect current effects"
                )
            if existing is not None:
                return str(existing)
            self.write(identity, {**row, "retry_operation_id": retry_id})
            return retry_id

    def for_session(self, session_id: str) -> list[dict[str, Any]]:
        """定位当前会话的操作，用于补回已发生结果；传参：会话；返回：记录列表。"""
        with self._db.snapshot() as source:
            rows = sorted(
                source.list("tool_operation", session_id=session_id),
                key=lambda row: (row["updated_at"], row["operation_id"]),
            )
            return [self._decode(row) for row in rows]

    def _decode(self, value: dict[str, Any]) -> dict[str, Any]:
        """把规范结果还原为既有运行接口；传参：文件原件记录；返回：独立操作视图。"""
        result = value.get("result")
        if isinstance(result, dict) and "output" in result and "content" not in result:
            value = {**value, "result": {**result, "content": result["output"]}}
        return value


def operation_payload(
    call: ToolOperation, *, state: str, result: RunToolsResult | None = None
) -> dict[str, object]:
    """构造操作事实，不重复保存超长原文；传参：调用、状态与结果；返回：可持久化对象。"""
    payload: dict[str, object] = {"call": asdict(call), "state": state}
    if result is not None:
        payload["result"] = asdict(result)
    return payload


def new_operation_id() -> str:
    """生成真实操作身份；传参：无；返回：稳定存储编号。"""
    return f"op-{new_ulid()}"


def file_changes(records: list[dict[str, Any]]) -> dict[str, object]:
    """从已存在操作投影文件变化，不读取磁盘或产生新账本；传参：当前可见记录；返回：变化与覆盖说明。"""
    changes: list[dict[str, object]] = []
    for row in records:
        call = row["call"]
        execution = call.get("execution_request") or {
            "tool": call["tool_name"],
            "arguments": call["args"],
        }
        result = row.get("result") or {}
        meta = result.get("meta") or {}
        point_ids = meta.get("restore_point_ids", [])
        if not point_ids and (
            execution["tool"] not in FILE_WRITERS
            or execution.get("source", "builtin") != "builtin"
        ):
            continue
        resource = call.get("resource") or {}
        state = row["state"]
        if state in {"started", "unknown"} or (state != "not_started" and not result):
            outcome = "unknown"
        elif state == "not_started" or meta.get("execution_state") == "not_started":
            outcome = "not_started"
        else:
            outcome = "succeeded" if result.get("status") == "ok" else "failed"
        change: dict[str, object] = {
            "operation_id": row["operation_id"],
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "tool": execution["tool"],
            "state": outcome,
            "requested_path": execution["arguments"].get("path"),
            "resolved_path": meta.get("resolved_path") or resource.get("path"),
            "restore_point_ids": point_ids,
            "restore_status": "snapshot_available_for_verification"
            if point_ids
            else "legacy_change_only_unrestorable",
        }
        if outcome == "succeeded":
            change.update(
                {
                    key: meta[key]
                    for key in (
                        "previous_sha256",
                        "content_sha256",
                        "changed",
                        "created",
                        "byte_count",
                    )
                    if key in meta
                }
            )
        changes.append(change)
    return {
        "changes": changes,
        "scope": "captured_operations_and_legacy_file_changes",
        "disk_state": "historical_not_rechecked",
        "untracked_effects": "only verified restore point roots are covered; remote and outside-workspace effects remain untracked",
    }


def retry_source(row: dict[str, Any]) -> str:
    """固定恢复授权对应的请求、效果和结果版本；传参：原操作记录；返回：重核指纹。"""
    source = {key: row.get(key) for key in ("call", "state", "result")}
    return hashlib.sha256(
        json.dumps(source, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
