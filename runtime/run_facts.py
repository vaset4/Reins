from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from runtime.persistence import RuntimeStore
from runtime.secret_redaction import looks_secret_key, redact_secret_text
from tasks.ids import new_ulid, utc_now

if TYPE_CHECKING:
    from runtime.checkpoint import Checkpoint

MAX_TEXT_CHARS = 512
MAX_ITEMS = 20
INVALID_JSONL_WARNING_CODE = "invalid_jsonl"
RUN_LIFECYCLES = frozenset(
    {"waiting_user", "waiting_approval", "paused", "done", "failed"}
)


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    session_id: str
    task_id: str | None = None
    focus_task_id: str | None = None
    compatibility_task_id: str | None = None
    started_at: str = ""
    updated_at: str = ""
    status: str = ""
    last_event: str = ""


@dataclass(frozen=True, slots=True)
class FactReadWarning:
    """表示 tolerant facts 读取跳过的单个损坏行"""

    code: str
    session_id: str
    run_id: str
    line_number: int
    message: str


@dataclass(frozen=True, slots=True)
class TolerantFactRead:
    """表示可信 facts 与结构化损坏 warning 的不可变载体"""

    facts: tuple[dict[str, Any], ...]
    warnings: tuple[FactReadWarning, ...]


@dataclass(frozen=True, slots=True)
class TolerantRunList:
    """表示 session 内可信 run 摘要与损坏 warning"""

    runs: tuple[RunSummary, ...]
    warnings: tuple[FactReadWarning, ...]


class RunFactStore:
    """运行事实按提交顺序保存在所属会话的事件原件中。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定持久事实表；传参：数据根；返回：无。"""
        self._data_root = Path(data_root)
        self._db = RuntimeStore(data_root)

    def append(self, fact: Mapping[str, object]) -> Path:
        """原子追加真实运行事件；传参：含会话与运行身份的事实；返回：所属会话日志位置。"""
        session_id = str(fact.get("session_id", "")).strip()
        run_id = str(fact.get("run_id", "")).strip()
        if not session_id or not run_id:
            raise ValueError("run fact requires session_id and run_id")
        payload = self._normalize_fact(fact)
        identity = new_ulid()
        with self._db.transaction() as batch:
            batch.put(
                "run_fact",
                identity,
                payload,
                session_id=session_id,
                expected_revision=0,
            )
        return self._db.source_path("run_fact", identity)

    def append_from_trajectory(
        self,
        row: Mapping[str, object],
    ) -> Path | None:
        session_id = str(row.get("session_id", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if not session_id or not run_id:
            return None
        fact = self._fact_from_trajectory(row)
        return self.append(fact)

    def append_checkpoint_ref(
        self,
        *,
        session_id: str,
        run_id: str,
        task_id: str | None,
        focus_task_id: str | None,
        compatibility_task_id: str | None,
        segment_id: str,
        checkpoint: "Checkpoint",
    ) -> Path:
        return self.append(
            {
                "type": "run_fact",
                "event": "checkpoint:saved",
                "ts": utc_now(),
                "session_id": session_id,
                "run_id": run_id,
                "task_id": task_id,
                "focus_task_id": focus_task_id,
                "compatibility_task_id": compatibility_task_id,
                "segment_id": segment_id,
                "checkpoint": {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "state": checkpoint.state,
                    "reason": checkpoint.reason,
                    "pending_tool_call": checkpoint.pending_tool_call,
                    "lease_snapshot": dict(checkpoint.lease_snapshot or {}),
                    "working_memory_snapshot_kind": "diagnostic",
                    "working_memory_snapshot": dict(
                        checkpoint.working_memory_snapshot or {}
                    ),
                },
            }
        )

    def append_lifecycle(
        self,
        *,
        lifecycle: str,
        reason: str,
        session_id: str,
        run_id: str,
        segment_id: str,
        task_id: str | None = None,
        focus_task_id: str | None = None,
        compatibility_task_id: str | None = None,
        checkpoint_id: str | None = None,
        resumable: bool | None = None,
    ) -> Path:
        payload: dict[str, object] = {
            "type": "run_fact",
            "event": "run:lifecycle",
            "ts": utc_now(),
            "session_id": session_id,
            "run_id": run_id,
            "task_id": task_id,
            "focus_task_id": focus_task_id,
            "compatibility_task_id": compatibility_task_id,
            "segment_id": segment_id,
            "lifecycle": _normalize_lifecycle(lifecycle),
            "reason": reason,
        }
        if checkpoint_id is not None:
            payload["checkpoint_id"] = checkpoint_id
        if resumable is not None:
            payload["resumable"] = resumable
        return self.append(payload)

    def read_run(self, run_id: str) -> list[dict[str, Any]]:
        """按运行身份读取提交顺序；传参：运行编号；返回：完整事实。"""
        return self._read(run_id=run_id)

    def read_session_run(self, session_id: str, run_id: str) -> list[dict[str, Any]]:
        """按会话和运行读取事实；传参：两个身份；返回：已提交记录。"""
        return self._read(session_id=session_id, run_id=run_id)

    def read_run_tolerant(self, run_id: str) -> TolerantFactRead:
        """提供观察界面结果载体，原件损坏明确抛错；传参：运行身份；返回：事实与空警告。"""
        return TolerantFactRead(tuple(self.read_run(run_id)), ())

    def read_latest_lifecycle(self, run_id: str) -> dict[str, Any]:
        """读取最近持久生命周期；传参：运行身份；返回：状态事实。"""
        return latest_lifecycle_from_facts(self.read_run(run_id))

    def handled_input_ids(self, session_id: str) -> frozenset[str]:
        """按已提交原件读取已经处理的输入；传参：会话身份；返回：去重输入身份。"""
        rows = self._read(session_id=session_id, event="input:handled")
        return frozenset(identity for row in rows for identity in row["input_ids"])

    def find_run(self, run_id: str) -> RunSummary | None:
        """定位运行归属，无目录扫描；传参：运行身份；返回：摘要或不存在。"""
        return self._summarize_facts(self.read_run(run_id))

    def list_runs_for_session(
        self, session_id: str, *, limit: int | None = None
    ) -> list[RunSummary]:
        """列出会话真实运行；传参：会话及上限；返回：最近更新优先的摘要。"""
        return self._summaries(self._read(session_id=session_id), limit)

    def list_runs_for_session_tolerant(
        self, session_id: str, *, limit: int | None = None
    ) -> TolerantRunList:
        """提供观察界面会话摘要；传参：会话及上限；返回：摘要，损坏原件抛错。"""
        return TolerantRunList(
            tuple(self.list_runs_for_session(session_id, limit=limit)), ()
        )

    def list_recent_runs(self, *, limit: int | None = None) -> list[RunSummary]:
        """列出全部真实运行；传参：上限；返回：最近更新优先的摘要。"""
        return self._summaries(self._read(), limit)

    def list_runs_for_task(
        self, task_id: str, *, limit: int | None = None
    ) -> list[RunSummary]:
        """查询曾触及目标的运行；传参：目标及上限；返回：完整运行摘要。"""
        with self._db.snapshot() as source:
            run_ids = {
                row.payload["run_id"]
                for row in source.list_raw("run_fact")
                if task_id
                in (
                    row.payload.get("task_id"),
                    row.payload.get("focus_task_id"),
                    row.payload.get("compatibility_task_id"),
                    row.payload.get("target_task_id"),
                )
            }
            rows = [row for run_id in run_ids for row in self._read(run_id=run_id)]
        return self._summaries(rows, limit)

    def read_task_facts(self, task_id: str) -> list[dict[str, Any]]:
        """聚合目标相关事实；传参：目标身份；返回：按时刻排序的事实。"""
        rows = [
            row
            for run in self.list_runs_for_task(task_id)
            for row in self.read_run(run.run_id)
        ]
        return sorted(rows, key=lambda row: str(row.get("ts", "")))

    def _read(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        event: str | None = None,
    ) -> list[dict[str, Any]]:
        """从提交快照读取领域事实；传参：会话、运行和事件筛选；返回：保持提交顺序的独立记录。"""
        filters = {
            key: value
            for key, value in (("run_id", run_id), ("event", event))
            if value is not None
        }
        with self._db.snapshot() as source:
            return list(source.list("run_fact", session_id=session_id, filters=filters))

    def _summaries(
        self, rows: list[dict[str, Any]], limit: int | None
    ) -> list[RunSummary]:
        """按运行分组生成摘要；传参：有序事实和上限；返回：最近运行。"""
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row["run_id"], []).append(row)
        summaries = [
            item
            for group in groups.values()
            if (item := self._summarize_facts(group)) is not None
        ]
        summaries.sort(key=lambda item: (item.updated_at, item.run_id), reverse=True)
        return summaries[:limit] if limit is not None else summaries

    def _normalize_fact(self, fact: Mapping[str, object]) -> dict[str, Any]:
        """保留事实字段与证据引用，只压缩字段内材料；传参：运行事实；返回：脱敏后的记录。"""
        # 【运行证据】【事实保存】完整身份字段不能挤掉末尾的请求或响应路径
        payload = {
            str(key): "<redacted>"
            if looks_secret_key(str(key))
            else _compact_value(value, depth=1)
            for key, value in fact.items()
        }
        if "input_ids" in fact:
            payload["input_ids"] = list(cast(Sequence[str], fact["input_ids"]))
        payload["type"] = "run_fact"
        payload.setdefault("ts", utc_now())
        payload.setdefault("schema_version", 1)
        return payload

    def _fact_from_trajectory(self, row: Mapping[str, object]) -> dict[str, object]:
        """保留运行和操作身份，再投影事件材料；传参：轨迹行；返回：持久事实。"""
        event = _trajectory_event(row)
        fact: dict[str, object] = {
            "type": "run_fact",
            "event": event,
            "ts": str(row.get("ts", utc_now())),
            "session_id": str(row.get("session_id", "")),
            "run_id": str(row.get("run_id", "")),
            "task_id": _optional_str(row.get("task_id")),
            "focus_task_id": _optional_str(row.get("focus_task_id")),
            "compatibility_task_id": _optional_str(row.get("compatibility_task_id")),
            "segment_id": str(row.get("segment_id", "")),
        }
        for key in (
            "schema_version",
            "request_id",
            "request_index",
            "operation_task_id",
            "operation_id",
        ):
            if key in row:
                fact[key] = row[key]
        return {**fact, **self._trajectory_payload(event, row)}

    def _trajectory_payload(
        self, event: str, row: Mapping[str, object]
    ) -> dict[str, object]:
        """按事件类型提取有用内容，工具结果保留产物和截断信息。

        传参：event/row 为事件及原轨迹；返回：不含身份的事实材料
        """
        if event == "run:start":
            return {
                "trigger": row.get("trigger"),
                "parent_segment_id": row.get("parent_segment_id"),
                "focus_task": row.get("focus_task", {}),
                "lease_summary": _lease_summary(row.get("lease_snapshot")),
            }
        if event == "state:transition":
            return {
                key: row.get(key)
                for key in (
                    "from_state",
                    "to_state",
                    "close_reason",
                    "terminal_reason",
                    "checkpoint_id",
                    "approval_decision",
                )
            }
        if event == "context:built":
            return {"summary": {"tool_history_count": row.get("tool_history_count", 0)}}
        if event == "llm:response":
            return {
                "summary": {
                    "has_final": bool(row.get("has_final")),
                    "has_run_tools": bool(row.get("has_run_tools")),
                    "error": row.get("error"),
                    "observation": row.get("observation"),
                    "evidence": row.get("evidence", {}),
                }
            }
        if event == "tool:request":
            return {
                "tool": {
                    "call_id": row.get("tool_call_id"),
                    "name": row.get("tool_name"),
                    "risk": row.get("risk", "unknown"),
                    "args_summary": _compact_value(row.get("args", {})),
                }
            }
        if event in {"tool:response", "tool:late_response"}:
            tool_meta = row.get("meta", {})
            return {
                "tool": {
                    "call_id": row.get("tool_call_id"),
                    "name": row.get("tool_name"),
                    "status": row.get("status"),
                    "error_category": row.get("error_category"),
                    "error": row.get("error"),
                    "summary": row.get("summary", ""),
                    "output_summary": _tool_output_summary(row),
                    "artifact_refs": _artifact_refs(row.get("output"), tool_meta),
                    **_tool_result_meta_summary(row, tool_meta),
                }
            }
        return {"detail": _compact_value(dict(row))}

    def _summarize_facts(self, facts: list[dict[str, Any]]) -> RunSummary | None:
        if not facts:
            return None
        first = facts[0]
        last = facts[-1]
        run_id = str(first.get("run_id", ""))
        session_id = str(first.get("session_id", ""))
        if not run_id or not session_id:
            return None
        return RunSummary(
            run_id=run_id,
            session_id=session_id,
            task_id=_first_optional(facts, "task_id"),
            focus_task_id=_first_optional(facts, "focus_task_id"),
            compatibility_task_id=_first_optional(facts, "compatibility_task_id"),
            started_at=str(first.get("ts", "")),
            updated_at=str(last.get("ts", "")),
            status=_status_from_facts(facts),
            last_event=str(last.get("event", "")),
        )


def _trajectory_event(row: Mapping[str, object]) -> str:
    if row.get("type") == "lease_snapshot":
        return "run:start"
    if row.get("type") == "state_transition":
        return "state:transition"
    event = str(row.get("event", "")).strip()
    return event or str(row.get("type", "event"))


def latest_lifecycle_from_facts(
    facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    for fact in reversed(facts):
        lifecycle = _lifecycle_from_fact(fact)
        if lifecycle:
            return lifecycle
    return {}


def _lifecycle_from_fact(fact: Mapping[str, Any]) -> dict[str, Any]:
    if fact.get("event") != "run:lifecycle":
        return {}
    lifecycle = str(fact.get("lifecycle", "")).strip().lower()
    if lifecycle not in RUN_LIFECYCLES:
        return {}
    result = dict(fact)
    result["lifecycle"] = lifecycle
    return result


def _normalize_lifecycle(value: str) -> str:
    lifecycle = value.strip().lower()
    if lifecycle not in RUN_LIFECYCLES:
        raise ValueError(f"invalid run lifecycle: {value}")
    return lifecycle


def _lease_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "trigger": value.get("trigger"),
        "max_steps": value.get("max_steps"),
        "max_tokens": value.get("max_tokens"),
        "expires_at": value.get("expires_at"),
        "capabilities": _capability_summary(value.get("capabilities")),
    }


def _capability_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            if "enabled" in item:
                result[str(key)] = bool(item.get("enabled"))
            elif "read" in item or "write" in item:
                result[str(key)] = {
                    "read_paths": len(item.get("read", []))
                    if isinstance(item.get("read"), list)
                    else 0,
                    "write_paths": len(item.get("write", []))
                    if isinstance(item.get("write"), list)
                    else 0,
                }
            else:
                result[str(key)] = "configured"
        else:
            result[str(key)] = item
    return result


def _compact_value(value: object, *, depth: int = 0) -> Any:
    if depth > 4:
        return "<truncated>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= MAX_ITEMS:
                result["<truncated>"] = f"{len(value) - MAX_ITEMS} more keys"
                break
            text_key = str(key)
            if looks_secret_key(text_key):
                result[text_key] = "<redacted>"
            else:
                result[text_key] = _compact_value(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        items = [_compact_value(item, depth=depth + 1) for item in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"<truncated {len(value) - MAX_ITEMS} more items>")
        return items
    if isinstance(value, str):
        return _compact_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _compact_text(str(value))


def _compact_text(value: str) -> str:
    # Value-pattern secret protection comes from the single owner; facts keep
    # only their own length budget (MAX_TEXT_CHARS) on top of it.
    redacted = redact_secret_text(value)
    if len(redacted) <= MAX_TEXT_CHARS:
        return redacted
    return f"{redacted[:MAX_TEXT_CHARS]}...<truncated {len(redacted) - MAX_TEXT_CHARS} chars>"


def _tool_output_summary(row: Mapping[str, object]) -> str:
    output = str(row.get("output", ""))
    refs = _artifact_refs(output, row.get("meta", {}))
    if not refs:
        return _compact_text(output)
    artifact_ids = ", ".join(str(ref["artifact_id"]) for ref in refs)
    summary = str(row.get("summary", "")).strip() or "tool artifact"
    return _compact_text(f"{summary}; artifacts: {artifact_ids}")


def _artifact_refs(output: object, meta: object) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    if isinstance(meta, Mapping):
        _append_artifact_ref(refs, meta)
    if isinstance(output, str) and output.strip().startswith("{"):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            _append_artifact_ref(refs, parsed)
            parsed_meta = parsed.get("meta")
            if isinstance(parsed_meta, Mapping):
                _append_artifact_ref(refs, parsed_meta)
    return refs


def _tool_result_meta_summary(
    row: Mapping[str, object], meta: object
) -> dict[str, Any]:
    meta_map = meta if isinstance(meta, Mapping) else {}
    source_truncated = _truthy_meta(meta_map.get("truncated"))
    prompt_truncated = _truthy_meta(row.get("prompt_truncated"))
    layers = _truncation_layers(source_truncated, prompt_truncated)
    result: dict[str, Any] = {
        "meta": _compact_value(meta_map),
        "output_complete": not layers,
        "source_truncated": source_truncated,
        "prompt_truncated": prompt_truncated,
        "truncation_layers": layers,
    }
    for key in ("total_count", "returned_count", "offset", "next_offset"):
        if key in meta_map:
            result[key] = _compact_value(meta_map[key])
    return result


def _truncation_layers(source_truncated: bool, prompt_truncated: bool) -> list[str]:
    layers: list[str] = []
    if source_truncated:
        layers.append("source")
    if prompt_truncated:
        layers.append("prompt")
    return layers


def _truthy_meta(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def _append_artifact_ref(
    refs: list[dict[str, str]], value: Mapping[str, object]
) -> None:
    artifact_id = str(value.get("artifact_id", "")).strip()
    if not artifact_id:
        return
    artifact_type = str(value.get("artifact_type", "")).strip() or "artifact"
    ref = {"artifact_id": artifact_id, "artifact_type": artifact_type}
    for index, existing in enumerate(refs):
        if existing["artifact_id"] != artifact_id:
            continue
        if existing["artifact_type"] == "artifact" and artifact_type != "artifact":
            refs[index] = ref
        return
    refs.append(ref)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _first_optional(facts: list[dict[str, Any]], key: str) -> str | None:
    for fact in facts:
        value = _optional_str(fact.get(key))
        if value is not None:
            return value
    return None


def _status_from_facts(facts: list[dict[str, Any]]) -> str:
    lifecycle = latest_lifecycle_from_facts(facts)
    if lifecycle:
        return str(lifecycle["lifecycle"])
    for fact in reversed(facts):
        if fact.get("event") == "state:transition":
            to_state = str(fact.get("to_state", ""))
            if to_state in {"DONE", "PAUSED", "FAILED"}:
                return to_state.lower()
    return ""


def _facts_match_task(facts: list[dict[str, Any]], task_id: str) -> bool:
    for fact in facts:
        if fact.get("task_id") == task_id or fact.get("focus_task_id") == task_id:
            return True
    return False


__all__ = [
    "FactReadWarning",
    "INVALID_JSONL_WARNING_CODE",
    "RUN_LIFECYCLES",
    "RunFactStore",
    "RunSummary",
    "TolerantFactRead",
    "TolerantRunList",
    "latest_lifecycle_from_facts",
]
