from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import closing
import hashlib
import os
from pathlib import Path

from artifacts.store import ArtifactStore
from tasks.ids import new_ulid
from tasks.persistence import write_task_yaml


@dataclass(frozen=True, slots=True)
class ArtifactPromptRef:
    artifact_id: str
    type: str
    summary: str
    key_excerpt: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def store_large_output(
    data_root: Path | str,
    task_id: str,
    content: str,
    *,
    type: str = "output",
    threshold: int = 4096,
    ext: str = "txt",
    summary: str | None = None,
    source_id: str | None = None,
) -> ArtifactPromptRef | None:
    """把超限原文完整保存为现有产物，模型视图可以只保留引用。

    传参：数据根、目标和原文；source_id让同一来源的重复投影复用原件；返回：引用，未超限为None
    """
    encoded = content.encode("utf-8")
    if len(encoded) <= threshold:
        return None
    identity = (
        hashlib.sha256(f"{task_id}:{source_id}:".encode() + encoded).hexdigest()
        if source_id
        else new_ulid()
    )
    artifact_id = f"art-{identity}"
    root = Path(data_root).resolve()
    ArtifactStore(data_root)
    path = root / "assets" / f"{artifact_id}.{ext}"
    relative_path = path.relative_to(root).as_posix()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 【产物】【保存原文】正文完整写入后才发布元数据，保存失败不生成可读成功引用
    if source_id:
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError(
                "retained artifact content does not match its source digest"
            )
        if not path.exists():
            write_task_yaml(path, content)
    else:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    artifact_summary = summary or content[:160].strip()
    with closing(ArtifactStore(data_root)) as store:
        if store.load_artifact(artifact_id) is None:
            store.create_artifact(
                task_id,
                type,
                relative_path,
                artifact_summary,
                len(encoded),
                artifact_id=artifact_id,
            )
    return ArtifactPromptRef(
        artifact_id=artifact_id,
        type=type,
        summary=artifact_summary,
        key_excerpt=content[:500],
    )


__all__ = ["ArtifactPromptRef", "store_large_output"]
