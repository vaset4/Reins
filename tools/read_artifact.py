from __future__ import annotations

from pathlib import Path
from contextlib import closing
import hashlib
import os
import re
from typing import Literal, cast

from artifacts.store import ArtifactStore
from context.cursors import decode_cursor, encode_cursor
import path_security
from runtime.lease import Lease

DEFAULT_ARTIFACT_PAGE_CHARS = 40000
DEFAULT_ARTIFACT_HEAD_CHARS = 4096
_ARTIFACT_SCAN_CHARS = 64 * 1024
ArtifactReadMode = Literal["summary", "head", "full"]


def read_artifact(
    data_root: Path | str,
    artifact_id: str,
    *,
    mode: ArtifactReadMode = "summary",
    head_chars: int = DEFAULT_ARTIFACT_HEAD_CHARS,
    lease: Lease | None = None,
) -> str:
    """读取已保存的产物正文；传参：根、身份及读取模式；返回：原文或摘要，越权明确失败。"""
    if (
        mode not in {"summary", "head", "full"}
        or type(head_chars) is not int
        or head_chars < 0
    ):
        raise ValueError("artifact read mode or head size is invalid")
    path, summary = _artifact_source(data_root, artifact_id, lease=lease)
    if mode == "summary":
        return summary
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read(head_chars) if mode == "head" else stream.read()


def read_artifact_page(
    data_root: Path,
    artifact_id: str,
    *,
    lease: Lease | None,
    mode: ArtifactReadMode | None = None,
    offset: int | None = None,
    limit: int = DEFAULT_ARTIFACT_PAGE_CHARS,
    expected_sha256: str | None = None,
    cursor: str | None = None,
) -> dict[str, object]:
    """按字符分页读取产物并核对内容版本，返回可继续的位置。

    传参：产物与授权、模式、位置或续读游标、页长及可选版本；返回：正文片段及读取元数据
    """
    mode, offset, expected_sha256 = _page_position(
        artifact_id,
        cursor,
        mode=mode,
        offset=offset,
        expected_sha256=expected_sha256,
    )
    if type(limit) is not int or not 0 < limit <= DEFAULT_ARTIFACT_PAGE_CHARS:
        raise ValueError("artifact limit is invalid")
    path, summary = _artifact_source(data_root, artifact_id, lease=lease)
    if mode == "summary":
        selected, total = summary[offset : offset + limit], len(summary)
        revision = hashlib.sha256(summary.encode("utf-8")).hexdigest()
    else:
        page_size = min(limit, DEFAULT_ARTIFACT_HEAD_CHARS) if mode == "head" else limit
        selected, total, revision = _read_text_page(path, offset, page_size)
    if expected_sha256 is not None and revision != expected_sha256:
        raise ValueError(
            "artifact content version changed; prior offsets are no longer valid"
        )
    if offset > total:
        raise ValueError("artifact offset is beyond the saved content")
    end = offset + len(selected)
    next_cursor = (
        encode_cursor(
            "artifact",
            {
                "artifact_id": artifact_id,
                "mode": mode,
                "offset": end,
                "sha256": revision,
            },
        )
        if end < total
        else None
    )
    return {
        "content": selected,
        "meta": {
            "artifact_id": artifact_id,
            "mode": mode,
            "content_sha256": revision,
            "offset": offset,
            "next_offset": end if end < total else None,
            "next_cursor": next_cursor,
            "returned_count": len(selected),
            "total_count": total,
            "truncated": end < total,
        },
    }


def _page_position(
    artifact_id: str,
    cursor: str | None,
    *,
    mode: ArtifactReadMode | None,
    offset: int | None,
    expected_sha256: str | None,
) -> tuple[ArtifactReadMode, int, str | None]:
    """合并显式读取位置和游标，拒绝相互矛盾的来源；传参：读取参数；返回：确定的位置与版本。"""
    if cursor is not None:
        saved = decode_cursor(
            cursor,
            "artifact",
            fields={"artifact_id": str, "mode": str, "offset": int, "sha256": str},
        )
        if saved["artifact_id"] != artifact_id:
            raise ValueError("paging cursor belongs to another artifact")
        supplied = {"mode": mode, "offset": offset, "sha256": expected_sha256}
        for key, value in supplied.items():
            if value is not None and value != saved[key]:
                raise ValueError(f"artifact {key} conflicts with paging cursor")
        mode, offset = cast(ArtifactReadMode, saved["mode"]), cast(int, saved["offset"])
        expected_sha256 = cast(str, saved["sha256"])
    mode = "full" if mode is None else mode
    offset = 0 if offset is None else offset
    if mode not in {"summary", "head", "full"} or type(offset) is not int or offset < 0:
        raise ValueError("artifact mode or offset is invalid")
    if (
        expected_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("artifact content version is invalid")
    return mode, offset, expected_sha256


def _read_text_page(path: Path, offset: int, limit: int) -> tuple[str, int, str]:
    """流式核对原文版本，仅保留所需字符页；传参：文件与范围；返回：正文、总字符数及SHA256。"""
    digest, total = hashlib.sha256(), 0
    pieces: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        before = os.fstat(stream.fileno())
        # 1. 【产物】【连续读取】保留CRLF和Unicode字符位置，每页扫描版本但不加载全文
        while chunk := stream.read(_ARTIFACT_SCAN_CHARS):
            digest.update(chunk.encode("utf-8"))
            start, end = max(0, offset - total), min(len(chunk), offset + limit - total)
            if start < end:
                pieces.append(chunk[start:end])
            total += len(chunk)
        statuses = (os.fstat(stream.fileno()), path.stat())
    # 2. 【产物】【连续读取】并发覆盖或替换不能作为同一个稳定页面返回
    original = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if any(
        (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        != original
        for row in statuses
    ):
        raise ValueError("artifact content version changed while reading")
    return "".join(pieces), total, digest.hexdigest()


def _artifact_source(
    data_root: Path | str, artifact_id: str, *, lease: Lease | None
) -> tuple[Path, str]:
    """每次读取都重新核对保存位置及授权；传参：根、产物和当前租约；返回：路径与摘要。"""
    with closing(ArtifactStore(data_root)) as store:
        record = store.load_artifact(artifact_id)
    if record is None:
        raise FileNotFoundError(artifact_id)
    path = _artifact_content_path(
        data_root, record.task_id, record.retained_path or record.path
    )
    if (
        lease is not None
        and path_security.check_read(path, lease) is not path_security.Decision.ALLOWED
    ):
        raise PermissionError("artifact content is outside the authorized read scope")
    return path, record.summary


def _artifact_content_path(data_root: Path | str, task_id: str, path: str) -> Path:
    """解析新旧产物位置并拒绝越出数据根；传参：根、目标与保存路径；返回：真实文件路径。"""
    root = Path(data_root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact path is outside data storage")
    if not resolved.is_file():
        raise FileNotFoundError(path)
    return resolved


__all__ = ["read_artifact", "read_artifact_page"]
