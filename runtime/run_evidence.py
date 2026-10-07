"""【运行证据】【文件原件】冻结实际请求与返回，SQLite不持有唯一证据。

作者：xxx
时间：2026-09-30 15:00:00
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, cast

from llm.messages import JsonValue, thaw_json_value
from runtime.file_records import record_key
from runtime.persistence import RuntimeStore
from runtime.secret_redaction import redact_value
from tasks.ids import new_ulid, utc_now

_ATTEMPT_KINDS = frozenset({"attempt_request", "attempt_response"})


def evidence_identity(session: str, run: str, kind: str, source: str) -> str:
    """从完整业务归属生成唯一引用；参数：会话、运行、种类、来源；返回：稳定证据编号。"""
    return hashlib.sha256(
        record_key(session, run, kind, source).encode("utf-8")
    ).hexdigest()


class RunEvidenceStore:
    """领域事件保存证据身份，实际input/output文件和正文引用保持不可变。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定文件原件服务；参数：数据根；返回：无长期句柄的实例。"""
        self.store = RuntimeStore(data_root)

    def write_record(
        self,
        *,
        session_id: str,
        run_id: str,
        kind: str,
        source_id: str,
        payload: Mapping[str, object],
    ) -> str:
        """原子发布证据与请求材料；参数：业务身份及正文；返回：可精确回查的证据引用。"""
        if not all((session_id, run_id, kind, source_id)):
            raise ValueError("evidence requires session, run, kind and source identity")
        identity = evidence_identity(session_id, run_id, kind, source_id)
        normalized = thaw_json_value(cast(JsonValue, payload))
        protected = redact_value(normalized)
        if not isinstance(protected, dict):
            raise ValueError("evidence payload must be an object")
        if kind in _ATTEMPT_KINDS:
            protected["retention"] = (
                "protected" if protected != normalized else "captured"
            )
        # 1. 【运行证据】【准备正文】大对象在全空间提交锁外冻结，后续批次只发布引用和元信息
        prepared = self.store.prepare_payload(
            {"body": protected}, session_id=session_id
        )
        with self.store.transaction() as batch:
            previous = batch.get("run_evidence", identity)
            if previous and kind in _ATTEMPT_KINDS:
                if previous["body"] != protected:
                    raise ValueError("committed model attempt evidence is immutable")
                return f"evidence:{identity}"
            record = {
                "evidence_id": identity,
                "session_id": session_id,
                "run_id": run_id,
                "kind": kind,
                "source_id": source_id,
                "created_at": previous["created_at"] if previous else utc_now(),
            }
            if kind in _ATTEMPT_KINDS:
                self._validate_attempt(protected, kind)
                record["attempt"] = _attempt_fields(protected)
                if kind == "attempt_request":
                    record["materials"] = _material_fields(protected)
            # 2. 【运行证据】【唯一原件】实际尝试直接路由input/output文件，不在事件流中另存一份
            batch.put(
                "run_evidence",
                identity,
                prepared.with_fields(record),
                session_id=session_id,
            )
        return f"evidence:{identity}"

    def record_request(
        self, *, session_id: str, run_id: str, request_id: str, request_index: int
    ) -> None:
        """保存尚未派发的逻辑请求身份；参数：运行归属、请求及序号；返回：无。"""
        with self.store.transaction() as batch:
            previous = batch.get("model_requests", request_id)
            if previous is not None:
                if (previous["session_id"], previous["run_id"]) != (session_id, run_id):
                    raise ValueError("model request belongs to another session or run")
                return
            batch.put(
                "model_requests",
                request_id,
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "request_index": request_index,
                    "created_at": utc_now(),
                },
                session_id=session_id,
            )

    def _validate_attempt(self, payload: Mapping[str, Any], kind: str) -> None:
        """核验发送和返回关系；参数：证据及种类；返回：无，孤立返回不发布。"""
        session, run, request = (
            str(payload[name]) for name in ("session_id", "run_id", "request_id")
        )
        self.record_request(
            session_id=session,
            run_id=run,
            request_id=request,
            request_index=int(payload.get("request_index", 0)),
        )
        if kind == "attempt_response":
            identity = evidence_identity(
                session, run, "attempt_request", str(payload["attempt_id"])
            )
            with self.store.snapshot() as source:
                prior = source.raw("run_evidence", identity)
            if prior is None or prior.payload["attempt"]["request_id"] != request:
                raise ValueError("model attempt response has no matching start")

    def attempt_references(self, request_id: str) -> list[dict[str, Any]]:
        """列出尝试的原件引用；参数：逻辑请求身份；返回：有序引用，不展开正文。"""
        with self.store.snapshot() as source:
            request = source.get("model_requests", request_id)
            if request is None:
                return []
            records = source.list_raw("run_evidence", session_id=request["session_id"])
        responses = {
            row.payload["source_id"]: f"evidence:{row.record_id}"
            for row in records
            if row.payload["kind"] == "attempt_response"
            and row.payload["attempt"]["request_id"] == request_id
        }
        starts = [
            row
            for row in records
            if row.payload["kind"] == "attempt_request"
            and row.payload["attempt"]["request_id"] == request_id
        ]
        starts.sort(key=lambda row: row.payload["attempt"]["attempt_index"])
        return [
            {
                "attempt_id": row.payload["source_id"],
                "request_ref": f"evidence:{row.record_id}",
                "response_ref": responses.get(row.payload["source_id"]),
            }
            for row in starts
        ]

    def append_error(
        self, *, session_id: str, run_id: str, error: Mapping[str, object]
    ) -> str:
        """保存真实运行错误；参数：运行身份与错误；返回：证据引用。"""
        return self.write_record(
            session_id=session_id,
            run_id=run_id,
            kind="error",
            source_id=new_ulid(),
            payload={
                "type": "run_error",
                "ts": utc_now(),
                "session_id": session_id,
                "run_id": run_id,
                **dict(error),
            },
        )

    def read_reference(self, reference: str) -> dict[str, Any] | None:
        """完整读取明确证据；参数：evidence引用；返回：保存的正文或不存在。"""
        if not reference.startswith("evidence:") or not reference[9:]:
            raise ValueError("invalid evidence reference")
        with self.store.snapshot() as source:
            row = source.get("run_evidence", reference[9:])
        return None if row is None else cast(dict[str, Any], row["body"])

    def list_records(
        self, *, session_id: str, run_id: str, kind: str | None = None
    ) -> list[dict[str, Any]]:
        """列出选定运行的完整诊断；参数：运行与可选类别；返回：按实际提交顺序展开的证据集合。"""
        with self.store.snapshot() as source:
            records = source.list_raw("run_evidence", session_id=session_id)
            result = []
            for raw in records:
                if raw.payload["run_id"] != run_id or (
                    kind is not None and raw.payload["kind"] != kind
                ):
                    continue
                value = source.get("run_evidence", raw.record_id)
                assert value is not None
                # 1. 【运行证据】【近期诊断】同秒记录和后续修订按提交及批内位置排序，时间与哈希不能代表先后
                assert raw.location is not None, (
                    "diagnostic evidence requires a committed source position"
                )
                position = (raw.location.sequence, raw.location.ordinal)
                result.append(
                    (
                        position,
                        {
                            "reference": f"evidence:{raw.record_id}",
                            "kind": value["kind"],
                            "source_id": value["source_id"],
                            "created_at": value["created_at"],
                            "payload": value["body"],
                        },
                    )
                )
        return [row for _, row in sorted(result, key=lambda item: item[0])]


def _attempt_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """提取不会物化长正文的尝试元信息；参数：真实捕获；返回：可重建目录所需字段。"""
    fields = {
        key: payload.get(key)
        for key in (
            "request_id",
            "attempt_id",
            "attempt_index",
            "provider",
            "model",
            "started_at",
            "error_category",
        )
    }
    response = payload.get("response")
    fields["response_message_id"] = (
        response.get("message_id") if isinstance(response, Mapping) else None
    )
    fields["status"] = (
        (
            "completed"
            if payload["success"]
            else str(payload.get("error_category") or "failed")
        )
        if "success" in payload
        else "started"
    )
    return fields


def _material_fields(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """提取实际回喂的身份关系；参数：请求原件；返回：不复制正文的材料目录。"""
    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        return []
    result = []
    for item in sources.get("messages", ()):
        operation_id = None
        for part in item.get("fed_back", {}).get("content", ()):
            if part.get("kind") != "text":
                continue
            try:
                value = json.loads(part["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("meta"), dict):
                operation_id = value["meta"].get("operation_id")
        result.append(
            {
                **{
                    key: item.get(key)
                    for key in ("position", "message_id", "kind", "call_id")
                },
                "operation_id": operation_id,
            }
        )
    return result
