"""【请求查看】【领域查询】按已提交文件快照读取目录和原件。

作者：xxx
时间：2026-09-30 15:00:00
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Mapping, cast

from runtime.evidence_catalog import EvidenceCatalog
from runtime.evidence_content import DEFAULT_DETAIL_CHARS, content_page, evidence_node
from runtime.persistence import RuntimeStore, SourceSnapshot
from tools.read_artifact import read_artifact_page

PAGE_SIZE = 30
MAX_PAGE_SIZE = 100
_IDENTITY_FIELDS = ("session_id", "run_id", "request_id", "attempt_id")


class RequestInspection:
    """读取已提交的源记录，查询不装配执行器或改变当前分支。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定文件服务；参数：数据根；返回：无持久读取锁的查询器。"""
        self.db = RuntimeStore(data_root)

    def query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """核验归属并读取一页；参数：RPC查询；返回：绑定空间和提交位置的结果。"""
        action = str(payload.get("action", "runs"))
        scope = {key: str(payload.get(key) or "") for key in _IDENTITY_FIELDS}
        if not scope["session_id"]:
            raise ValueError("session_id is required")
        space = self.db.data_space_id
        boundary, after = _page_boundary(payload, scope, space)
        sequence = boundary["commit"] if boundary is not None else None
        with self.db.snapshot(sequence=sequence) as source:
            catalog = EvidenceCatalog(source, scope["session_id"])
            catalog.validate(scope)
            snapshot = {"commit": source.sequence}
            context = {**scope, "data_space_id": space, "snapshot": snapshot}
            if action == "detail":
                detail = self._detail(catalog, payload, scope)
            elif action == "locate":
                return {
                    **context,
                    "items": catalog.locate(payload, scope),
                    "next_cursor": None,
                }
            else:
                limit = _bounded_int(payload.get("limit", PAGE_SIZE), 1, MAX_PAGE_SIZE)
                rows = self._rows(catalog, action, scope)
                rows = [row for row in rows if row["sequence"] > after][: limit + 1]
                more = len(rows) > limit
                rows = rows[:limit]
                cursor = (
                    _cursor(
                        {
                            "scope": scope,
                            "action": action,
                            "space": space,
                            "snapshot": snapshot,
                            "after": rows[-1]["sequence"],
                        }
                    )
                    if more
                    else None
                )
                return {**context, "items": rows, "next_cursor": cursor}
        artifact = detail.pop("_artifact", None)
        if artifact is not None:
            page = read_artifact_page(
                self.db.data_root,
                artifact["artifact_id"],
                lease=None,
                offset=detail["offset"],
                limit=detail["limit"],
                expected_sha256=artifact["content_sha256"],
            )
            meta = cast(dict[str, Any], page["meta"])
            detail.update(
                {
                    "text": page["content"],
                    "total_chars": meta["total_count"],
                    "has_more": meta["truncated"],
                    "next_offset": meta["next_offset"],
                }
            )
        return {**context, **detail}

    def validate_scope(self, source: SourceSnapshot, scope: Mapping[str, str]) -> None:
        """验证导出和查询的归属；参数：文件快照与选择；返回：无，跨会话引用抛错。"""
        EvidenceCatalog(source, scope["session_id"]).validate(scope)

    def _rows(
        self, catalog: EvidenceCatalog, action: str, scope: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        """构造选中目录的轻量记录；参数：目录、动作与选择；返回：未展开正文的项目。"""
        if action == "runs":
            return catalog.runs()
        if action == "requests":
            _require(scope, "run_id")
            return catalog.request_rows(scope["run_id"])
        if action == "attempts":
            _require(scope, "request_id")
            return [
                {
                    key: row[key]
                    for key in (
                        "sequence",
                        "attempt_id",
                        "attempt_index",
                        "provider",
                        "model",
                        "started_at",
                        "status",
                        "error_category",
                    )
                }
                for row in catalog.attempts
                if row["request_id"] == scope["request_id"]
            ]
        if action == "tools":
            _require(scope, "run_id")
            return [
                {
                    "sequence": index + 1,
                    "operation_id": row.record_id,
                    "status": row.payload["state"],
                    "call_id": row.payload["call"]["call_id"],
                    "tool_name": row.payload["call"]["tool_name"],
                }
                for index, row in enumerate(catalog.tools)
                if row.payload["run_id"] == scope["run_id"]
            ]
        raise ValueError("unknown evidence query action")

    def _detail(
        self,
        catalog: EvidenceCatalog,
        payload: Mapping[str, Any],
        scope: Mapping[str, str],
    ) -> dict[str, Any]:
        """读取一次尝试或工具的单个区块；参数：目录、请求和归属；返回：真实正文分页。"""
        section = str(payload.get("section", "request"))
        offset = _bounded_int(payload.get("offset", 0), 0, None)
        limit = _bounded_int(
            payload.get("limit", DEFAULT_DETAIL_CHARS), 1, DEFAULT_DETAIL_CHARS
        )
        if payload.get("operation_id"):
            identity = str(payload["operation_id"])
            record = next(
                (
                    row
                    for row in catalog.tools
                    if row.record_id == identity
                    and row.payload["run_id"] == scope["run_id"]
                ),
                None,
            )
            if record is None:
                raise ValueError("tool operation does not belong to run")
            if section == "tool_result":
                return self._original(
                    catalog.source, evidence_node(record), offset=offset, limit=limit
                )
            if section not in {"tool_feedback", "sources"}:
                raise ValueError("unknown tool evidence section")
            feedback = catalog.feedback(identity, content=section == "tool_feedback")
            return {
                "section": section,
                "operation_id": identity,
                "retention": "captured",
                **content_page(self.db, feedback, offset=offset, limit=limit),
            }
        if section not in {"request", "response", "sources", "tool_feedback"}:
            raise ValueError("unknown evidence section")
        _require(scope, "attempt_id")
        attempt = next(
            row for row in catalog.attempts if row["attempt_id"] == scope["attempt_id"]
        )
        reference = (
            attempt["response_ref"] if section == "response" else attempt["request_ref"]
        )
        if reference is None:
            text = "尚未收到已提交的返回；供应商结果可能未知"
            return {
                "section": section,
                "status": attempt["status"],
                "retention": "not_returned",
                "text": text,
                "total_chars": len(text),
                "offset": 0,
                "has_more": False,
                "next_offset": None,
            }
        body = catalog.body(reference)
        node = body.get("sources" if section == "tool_feedback" else section)
        if section == "tool_feedback":
            position = _bounded_int(payload.get("position", 0), 0, None)
            candidates = [
                item for item in node["messages"] if item["position"] == position
            ]
            if not candidates:
                raise ValueError("tool feedback position not found")
            node = candidates[0]["fed_back"]
        return {
            "section": section,
            "status": attempt["status"],
            "retention": body["retention"],
            **content_page(self.db, node, offset=offset, limit=limit),
        }

    def _original(
        self,
        source: SourceSnapshot,
        node: Mapping[str, Any],
        *,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        """核验原工具结果及冻结产物；参数：快照、操作和分页；返回：原件页或待读取产物。"""
        result = node.get("result")
        meta = result.get("meta", {}) if isinstance(result, dict) else {}
        identity = meta.get("result_artifact_id")
        if identity:
            artifact = source.get("artifact_records", identity)
            if artifact is None:
                raise ValueError("retained tool artifact is missing")
            task = node["call"].get("task_id")
            if task is not None and artifact["task_id"] != task:
                raise ValueError("tool artifact belongs to another task")
            if not artifact.get("content_sha256") or not artifact.get("retained_path"):
                raise ValueError("tool artifact has no frozen content version")
            return {
                "section": "tool_result",
                "offset": offset,
                "limit": limit,
                "retention": "captured",
                "artifact_id": identity,
                "_artifact": artifact,
            }
        return {
            "section": "tool_result",
            "retention": "captured" if result is not None else "not_returned",
            **content_page(self.db, result, offset=offset, limit=limit),
        }


def _bounded_int(value: object, minimum: int, maximum: int | None) -> int:
    """核验RPC分页范围；参数：输入值及范围；返回：有效整数。"""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError("invalid evidence page range")
    return value


def _require(scope: Mapping[str, str], name: str) -> None:
    """要求下钻身份；参数：选择与字段；返回：无，缺失时抛错。"""
    if not scope.get(name):
        raise ValueError(f"{name} is required")


def _cursor(value: Mapping[str, Any]) -> str:
    """编码绑定空间和选择的游标；参数：位置元数据；返回：不透明字符串。"""
    return base64.urlsafe_b64encode(
        json.dumps(value, separators=(",", ":")).encode()
    ).decode()


def _page_boundary(
    payload: Mapping[str, Any], scope: Mapping[str, str], space: str
) -> tuple[dict[str, int] | None, int]:
    """恢复明确提交截止点；参数：查询、选择和空间；返回：源版本及目录偏移。"""
    cursor = payload.get("cursor")
    if not cursor:
        return None, 0
    decoded = json.loads(base64.urlsafe_b64decode(str(cursor)))
    if (
        decoded.get("scope") != dict(scope)
        or decoded.get("space") != space
        or decoded.get("action") != payload.get("action", "runs")
    ):
        raise ValueError("evidence cursor belongs to another selection or data space")
    boundary = decoded["snapshot"]
    if not isinstance(boundary, dict) or type(boundary.get("commit")) is not int:
        raise ValueError("invalid evidence snapshot boundary")
    return boundary, _bounded_int(decoded["after"], 0, None)
