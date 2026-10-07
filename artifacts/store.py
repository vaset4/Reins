from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tasks.ids import new_ulid, utc_now
from runtime.persistence import ContentReference, RuntimeStore
from runtime.workspaces import WorkspaceStore

RETENTION_DAYS = {
    "screenshot": 7,
    "html_dump": 7,
    "file_dump": 14,
    "download": 30,
    "output": -1,
}


@dataclass(slots=True)
class ArtifactRecord:
    artifact_id: str
    task_id: str
    type: str
    path: str
    summary: str
    bytes: int
    created_at: str
    retention_until: str | None = None
    content_sha256: str | None = None
    retained_path: str | None = None
    workspace_id: str | None = None
    session_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        """恢复产物身份、冻结内容和作用域；传参：原件映射；返回：完整记录。"""
        return cls(
            artifact_id=str(data["artifact_id"]),
            task_id=str(data["task_id"]),
            type=str(data.get("type", "")),
            path=str(data.get("path", "")),
            summary=str(data.get("summary", "")),
            bytes=int(data.get("bytes", 0)),
            created_at=str(data["created_at"]),
            retention_until=(
                str(data["retention_until"])
                if data.get("retention_until") is not None
                else None
            ),
            content_sha256=data.get("content_sha256"),
            retained_path=data.get("retained_path"),
            workspace_id=data.get("workspace_id"),
            session_id=data.get("session_id"),
        )


class ArtifactStore:
    def __init__(self, data_root: Path | str) -> None:
        """绑定产物原件并核对空间；传参：数据根；返回：无。"""
        self._data_root = Path(data_root)
        self._db = RuntimeStore(data_root)
        self._db.ensure_space()

    def close(self) -> None:
        """短连接自动释放；传参：无；返回：无。"""

    def create_artifact(
        self,
        task_id: str,
        type: str,
        path: str,
        summary: str,
        bytes: int,
        *,
        artifact_id: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> ArtifactRecord:
        """发布产物时冻结实际字节；参数：目标、类型、原路径及摘要；返回：带内容版本的记录。"""
        _validate_artifact_id(artifact_id or "new-artifact")
        workspace_id = self._resolve_workspace(workspace_id, session_id)
        source = (self._data_root / path).resolve()
        allowed = [self._data_root.resolve()]
        if workspace_id is not None:
            allowed.append(
                WorkspaceStore(self._data_root).get(workspace_id).project_root
            )
        if not any(source.is_relative_to(root) for root in allowed):
            raise ValueError("artifact path is outside its data storage and workspace")
        reference = self._db.prepare_file(
            source, workspace_id=workspace_id, session_id=session_id
        )
        created_at = utc_now()
        record = ArtifactRecord(
            artifact_id=artifact_id or f"art-{new_ulid()}",
            task_id=task_id,
            type=type,
            path=path,
            summary=summary,
            bytes=reference.size,
            created_at=created_at,
            retention_until=self._retention_until(type, created_at),
            content_sha256=reference.sha256,
            retained_path=reference.path,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        self._write(record)
        return record

    def load_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        """读取产物原件；传参：身份；返回：完整记录或不存在。"""
        _validate_artifact_id(artifact_id)
        with self._db.snapshot() as source:
            row = source.get("artifact_records", artifact_id)
            return ArtifactRecord.from_dict(row) if row is not None else None

    def list_artifacts(self, task_id: str | None = None) -> list[ArtifactRecord]:
        """列出实际产物元数据；传参：可选目标；返回：产物记录。"""
        with self._db.snapshot() as source:
            rows = source.list(
                "artifact_records",
                filters=None if task_id is None else {"task_id": task_id},
            )
            return [
                ArtifactRecord.from_dict(row)
                for row in sorted(rows, key=lambda row: row["artifact_id"])
            ]

    def read_path(self, artifact_id: str) -> Path:
        """核对冻结字节后提供只读来源；传参：产物身份；返回：原件路径，损坏明确报错。"""
        record = self.load_artifact(artifact_id)
        if record is None:
            raise FileNotFoundError(artifact_id)
        reference = self._content_reference(record)
        self._db.read_content(reference, limit=0)
        return self._data_root / reference.path

    def prune_expired_artifacts(self, now: str) -> list[str]:
        """列出已到保留期的产物，不删除原件；传参：当前时间；返回：到期身份。"""
        rows = [
            row
            for row in self.list_artifacts()
            if row.retention_until is not None and row.retention_until <= now
        ]
        return [
            row.artifact_id
            for row in sorted(rows, key=lambda row: row.retention_until or "")
        ]

    def _write(self, record: ArtifactRecord) -> None:
        """产物元数据与冻结字节在同一提交中发布；传参：完整产物；返回：无。"""
        _validate_artifact_id(record.artifact_id)
        payload = asdict(record)
        with self._db.transaction() as batch:
            batch.reference_content(self._content_reference(record))
            batch.put(
                "artifact_records",
                record.artifact_id,
                payload,
                workspace_id=record.workspace_id,
                session_id=record.session_id,
            )

    def _content_reference(self, record: ArtifactRecord) -> ContentReference:
        """要求产物有真实冻结来源；传参：记录；返回：不可变内容引用。"""
        if record.retained_path is None or record.content_sha256 is None:
            raise ValueError("artifact retained content is missing")
        return ContentReference(
            record.retained_path, record.content_sha256, record.bytes
        )

    def _resolve_workspace(
        self, workspace_id: str | None, session_id: str | None
    ) -> str | None:
        """按显式身份核对归属，不从全局目标猜测；传参：工作区与会话；返回：原工作区或全局。"""
        with self._db.snapshot() as source:
            binding = (
                source.get("session_workspace", session_id) if session_id else None
            )
        if binding is not None:
            if workspace_id is not None and workspace_id != binding["workspace_id"]:
                raise ValueError("artifact workspace differs from its session")
            return str(binding["workspace_id"])
        return workspace_id

    def _retention_until(self, type: str, created_at: str) -> str | None:
        """保留既有产物到期规则；传参：类别与时间；返回：到期时间或长期保留。"""
        days = RETENTION_DAYS.get(type)
        if days is None:
            days = 30
        if days < 0:
            return None
        dt = datetime.fromisoformat(created_at) + timedelta(days=days)
        return dt.isoformat(timespec="seconds")


def _validate_artifact_id(artifact_id: str) -> None:
    """写盘和读取前校验外部产物身份；传参：产物编号；返回：无，越界名称明确拒绝。"""
    if (
        not artifact_id
        or any(char in artifact_id for char in "/\\:")
        or artifact_id in {".", ".."}
    ):
        raise ValueError("artifact id must be a single storage name")
