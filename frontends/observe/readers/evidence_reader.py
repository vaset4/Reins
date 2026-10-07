from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore
from runtime.session_state import SessionStateStore
from runtime.request_inspection import RequestInspection


class EvidenceReader:
    """Read-only access to raw evidence files under a run directory."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._store = RunEvidenceStore(data_root)
        self._inspection = RequestInspection(data_root)

    def query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """复用TUI的按页请求查询；参数：选择与分页；返回：相同的领域结果。"""
        return self._inspection.query(payload)

    def read_latest(
        self, facts: list[dict[str, Any]], key: str
    ) -> Mapping[str, Any] | None:
        for row in reversed(facts):
            attempt_key = {
                "model_request": "request_path",
                "model_response": "response_path",
            }.get(key)
            if attempt_key and isinstance(row.get(attempt_key), str):
                return self._read_json(row[attempt_key])
            if row.get("event") != "llm:response":
                continue
            summary = row.get("summary")
            if not isinstance(summary, Mapping):
                continue
            evidence = summary.get("evidence")
            if not isinstance(evidence, Mapping):
                continue
            raw_path = str(evidence.get(key, ""))
            if not raw_path or raw_path.startswith("("):
                continue
            return self._read_json(raw_path)
        return None

    def read_errors(self, session_id: str, run_id: str) -> list[dict[str, Any]]:
        """读取真实运行错误；传参：运行归属；返回：错误列表。"""
        return [
            row["payload"]
            for row in self._store.list_records(
                session_id=session_id, run_id=run_id, kind="error"
            )
        ]

    def read_raw_file(self, relative_path: str) -> dict[str, Any]:
        """按领域引用读取观察材料；传参：证据/运行/会话引用；返回：明确查询状态。"""
        value: Any
        if relative_path.startswith("evidence:"):
            value = self._store.read_reference(relative_path)
        elif relative_path.startswith("run:"):
            value = RunFactStore(self._data_root).read_run(relative_path[4:])
        elif relative_path.startswith("session:"):
            state = SessionStateStore(self._data_root).load(relative_path[8:])
            value = asdict(state) if state else None
        else:
            return {"status": "invalid_reference", "content_type": "text", "data": ""}
        return {
            "status": "ok" if value is not None else "missing",
            "content_type": "json",
            "data": value,
        }

    def _read_json(self, reference: str) -> Mapping[str, Any] | None:
        """读取模型证据引用；传参：稳定引用；返回：诊断或不存在。"""
        return self._store.read_reference(reference)


__all__ = ["EvidenceReader"]
