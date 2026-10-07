"""【请求查看】【来源目录】从同一提交快照提取轻量请求关系。

作者：xxx
时间：2026-09-30 15:00:00
"""

from __future__ import annotations

from typing import Any, Mapping

from runtime.evidence_content import evidence_node
from runtime.file_records import StoredRecord, record_key
from runtime.persistence import SourceSnapshot


def record_position(record: StoredRecord) -> tuple[int, int, str]:
    """按实际提交位置排序；参数：源记录；返回：批次、批内序号和身份。"""
    location = record.location
    return (
        (location.sequence, location.ordinal, record.record_id)
        if location
        else (0, 0, record.record_id)
    )


class EvidenceCatalog:
    """只解读元信息，不展开请求、工具或消息正文。"""

    def __init__(self, source: SourceSnapshot, session_id: str) -> None:
        """固定会话及源快照；参数：已提交快照和会话；返回：轻量目录。"""
        self.source = source
        self.session_id = session_id
        self.evidence = sorted(
            source.list_raw("run_evidence", session_id=session_id), key=record_position
        )
        self.requests = [
            {**row.payload, "sequence": index + 1}
            for index, row in enumerate(
                sorted(
                    source.list_raw("model_requests", session_id=session_id),
                    key=record_position,
                )
            )
        ]
        self.attempts = self._attempts()
        self.tools = sorted(
            source.list_raw("tool_operation", session_id=session_id),
            key=record_position,
        )
        self.facts = sorted(
            source.list_raw("run_fact", session_id=session_id), key=record_position
        )

    def _attempts(self) -> list[dict[str, Any]]:
        """组合启动和返回的同源关系；参数：无；返回：按启动顺序排列的尝试元数据。"""
        replies = {
            row.payload["source_id"]: row
            for row in self.evidence
            if row.payload["kind"] == "attempt_response"
        }
        result: list[dict[str, Any]] = []
        for row in self.evidence:
            if row.payload["kind"] != "attempt_request":
                continue
            fields = row.payload["attempt"]
            reply = replies.get(fields["attempt_id"])
            response_fields = reply.payload["attempt"] if reply else {}
            result.append(
                {
                    **fields,
                    "sequence": len(result) + 1,
                    "status": response_fields.get("status", "started"),
                    "error_category": response_fields.get("error_category"),
                    "response_message_id": response_fields.get("response_message_id"),
                    "request_ref": f"evidence:{row.record_id}",
                    "response_ref": f"evidence:{reply.record_id}" if reply else None,
                    "materials": row.payload.get("materials", []),
                }
            )
        return result

    def validate(self, scope: Mapping[str, str]) -> None:
        """校验层级归属；参数：运行/请求/尝试选择；返回：无，跨归属抛错。"""
        run, request, attempt = (
            scope.get(key, "") for key in ("run_id", "request_id", "attempt_id")
        )
        if run and not (
            any(row.payload.get("run_id") == run for row in self.facts)
            or any(row["run_id"] == run for row in self.requests)
        ):
            raise ValueError("run does not belong to session")
        if request and not any(
            row["request_id"] == request and row["run_id"] == run
            for row in self.requests
        ):
            raise ValueError("request does not belong to run")
        if attempt and not any(
            row["attempt_id"] == attempt and row["request_id"] == request
            for row in self.attempts
        ):
            raise ValueError("attempt does not belong to request")

    def body(self, reference: str | None) -> Any:
        """取得带文件引用的证据正文；参数：证据引用；返回：读取树，未返回为None。"""
        if reference is None:
            return None
        if not reference.startswith("evidence:"):
            raise ValueError("invalid evidence reference")
        record = self.source.raw("run_evidence", reference[9:])
        if record is None or record.session_id != self.session_id:
            raise ValueError("committed model attempt evidence is missing")
        return evidence_node(record)["body"]

    def runs(self) -> list[dict[str, Any]]:
        """按首次事实排列运行及截止状态；参数：无；返回：有限元信息构成的运行目录。"""
        result: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(self.facts, 1):
            row = raw.payload
            run = row["run_id"]
            if run not in result:
                result[run] = {
                    "run_id": run,
                    "sequence": index,
                    "started_at": row.get("ts", ""),
                    "status": "running_or_unknown",
                    "last_fact": index,
                }
            result[run]["last_fact"] = index
            if row.get("event") == "run:lifecycle":
                result[run]["status"] = row["lifecycle"]
        return list(result.values())

    def request_rows(self, run_id: str) -> list[dict[str, Any]]:
        """返回逻辑请求及最后尝试状态；参数：运行；返回：不受详情页长影响的请求目录。"""
        result = []
        for request in self.requests:
            if request["run_id"] != run_id:
                continue
            attempts = [
                row
                for row in self.attempts
                if row["request_id"] == request["request_id"]
            ]
            result.append(
                {
                    **request,
                    "attempt_count": len(attempts),
                    "status": attempts[-1]["status"] if attempts else "prepared",
                }
            )
        return result

    def feedback(self, operation_id: str, *, content: bool) -> list[dict[str, Any]]:
        """定位工具实际进入模型的各版本；参数：操作及是否读取内容；返回：有序引用树。"""
        result = []
        for attempt in self.attempts:
            for material in attempt["materials"]:
                if material.get("operation_id") != operation_id:
                    continue
                fields = {key: attempt[key] for key in ("request_id", "attempt_id")}
                fields.update(
                    {key: material[key] for key in ("position", "message_id")}
                )
                if content:
                    body = self.body(attempt["request_ref"])
                    messages = body["sources"]["messages"]
                    message = next(
                        item
                        for item in messages
                        if item["position"] == material["position"]
                    )
                    fields["content"] = message["fed_back"]
                result.append(fields)
        return result

    def locate(
        self, payload: Mapping[str, Any], scope: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        """按持久卡片身份定位尝试；参数：消息/entry/call及归属；返回：实际引用关系。"""
        message, entry, call = (
            str(payload.get(key, "")) for key in ("message_id", "entry_id", "call_id")
        )
        if entry:
            row = self.source.get("session_entry", record_key(self.session_id, entry))
            if row is None or (
                scope["run_id"] and row.get("run_id") != scope["run_id"]
            ):
                raise ValueError("entry does not belong to selected session or run")
            message = str(
                row.get("message", {}).get("id") or row.get("message_id") or ""
            )
        if not message and not call:
            raise ValueError("entry_id, message_id or call_id is required")
        originals = {
            row.payload["call"]["request_id"]
            for row in self.tools
            if call and row.payload["call"]["call_id"] == call
        }
        requests = {row["request_id"]: row for row in self.requests}
        result = []
        for attempt in self.attempts:
            request = requests[attempt["request_id"]]
            if scope["run_id"] and request["run_id"] != scope["run_id"]:
                continue
            found = message and (
                message
                in {
                    request["request_id"],
                    request["request_id"] + ":tool-calls",
                    attempt["response_message_id"],
                }
                or any(item["message_id"] == message for item in attempt["materials"])
            )
            found = found or (
                call
                and any(item.get("call_id") == call for item in attempt["materials"])
            )
            if found or request["request_id"] in originals:
                item = {
                    key: attempt[key]
                    for key in ("request_id", "attempt_id", "attempt_index", "sequence")
                }
                item["run_id"] = request["run_id"]
                if request["request_id"] in originals:
                    item["relation"] = "tool_origin"
                result.append(item)
        return result
