"""【文件恢复】【原件捕获】保存真实文件版本与逐次执行的覆盖证据。

作者：xxx
时间：2026-09-30 21:00:00
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from datetime import datetime, timezone
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

import path_security
from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.file_content import CONTENT_CHUNK_BYTES
from runtime.file_records import ContentReference
from runtime.lease import Lease
from runtime.persistence import RuntimeStore
from tools.config_syntax import UnsupportedConfig, is_private_key, parse_config
from tools.file_persistence import FileEditConflict, file_change_time
from tools.restore_protection import (
    ProtectedContentStore,
    read_security,
    stable_file_read,
)

EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
_PRIVATE_KEY_OVERLAP_BYTES = 256


class SnapshotStore:
    """恢复点与原件的单一读取入口，工作文件IO始终在提交锁外进行。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定已选择的数据空间；传参：数据根；返回：无。"""
        self.data_root = Path(data_root).resolve()
        self.database = RuntimeStore(self.data_root)
        self.protected = ProtectedContentStore(self.data_root)
        self._last_states: dict[str, dict[str, Any]] = {}
        self._reused_files = 0
        self._read_boundary: path_security.ReadBoundary | None = None
        self._capture_reads: ExitStack | None = None
        self._regular_contents: dict[tuple[str, int], ContentReference] = {}

    def capture_file(
        self,
        path: Path,
        lease: Lease,
        workspace_id: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """冻结一份当前文件及权限；传参：路径、权限、工作区、取消信号；返回：可核验状态，IO失败直抛。"""
        _check_capture_cancelled(cancellation)
        path = path.absolute()
        if self._read_boundary is None:
            resolved = path.resolve()
            sensitive = path_security.uses_redacted_files(path, lease)
            permission = path_security.check_read(path, lease, filtered=sensitive)
        else:
            resolved, sensitive, permission = self._read_boundary.inspect(path)
        state: dict[str, Any] = {
            "path": str(path),
            "captured_at": datetime.now(timezone.utc).isoformat(
                timespec="microseconds"
            ),
            "kind": "missing",
            "content": None,
            "protected_content": None,
            "identity": None,
            "version": None,
            "metadata": {},
            "sensitive": sensitive,
            "restorable": True,
        }
        if resolved.is_relative_to(self.data_root):
            return {
                **state,
                "kind": "excluded",
                "restorable": False,
                "error": "runtime_data_root",
            }
        if permission is not path_security.Decision.ALLOWED:
            return {
                **state,
                "kind": "excluded",
                "restorable": False,
                "error": "read_permission_denied",
            }
        if sensitive and not _supported_sensitive_path(path):
            return {
                **state,
                "kind": "excluded",
                "restorable": False,
                "error": "unsupported_sensitive_path",
            }
        try:
            status = path.lstat()
        except FileNotFoundError:
            return state
        state.update(
            identity=f"{status.st_dev}:{status.st_ino}",
            metadata={
                "mode": status.st_mode,
                "mtime_ns": status.st_mtime_ns,
                "ctime_ns": status.st_ctime_ns,
                "nlink": status.st_nlink,
                "size": status.st_size,
            },
        )
        if path.is_symlink() or bool(getattr(status, "st_reparse_tag", False)):
            return {
                **state,
                "kind": "link",
                "restorable": False,
                "error": "link_semantics_not_supported",
            }
        if stat.S_ISDIR(status.st_mode):
            return {
                **state,
                "kind": "directory",
                "restorable": False,
                "error": "directory_inventory_only",
            }
        if not stat.S_ISREG(status.st_mode):
            return {
                **state,
                "kind": "unsupported",
                "restorable": False,
                "error": "not_regular_file",
            }
        if self._capture_reads is not None:
            source = self._capture_reads.enter_context(stable_file_read(path))
            return self._capture_regular(
                path, state, workspace_id, source, cancellation=cancellation
            )
        with stable_file_read(path) as source:
            return self._capture_regular(
                path, state, workspace_id, source, cancellation=cancellation
            )

    def _capture_regular(
        self,
        path: Path,
        state: dict[str, Any],
        workspace_id: str,
        source: BinaryIO,
        *,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """冻结核验后的内容；传参：路径、元信息、工作区、文件流、取消信号；返回：完整文件状态。"""
        state = dict(state)
        # 1. 【文件恢复】【内容核验】先流式检查私钥，禁止先写普通objects再决定是否敏感
        digest, signature, private_key = _inspect_file(
            path, source, cancellation=cancellation
        )
        if signature[:2] != (state["identity"], state["metadata"]["size"]):
            raise FileEditConflict(f"file identity changed before capture: {path}")
        if private_key:
            return {
                **state,
                "kind": "excluded",
                "restorable": False,
                "error": "private_key",
            }
        metadata = {
            **state["metadata"],
            "security": read_security(path),
            "sha256": digest,
            "change_time": signature[2],
        }
        reused = self._reuse_original(state, metadata, cancellation=cancellation)
        if reused is not None:
            return reused
        if state["sensitive"]:
            source.seek(0)
            chunks = []
            while chunk := source.read(CONTENT_CHUNK_BYTES):
                _check_capture_cancelled(cancellation)
                chunks.append(chunk)
            raw = b"".join(chunks)
            _check_capture_cancelled(cancellation)
            try:
                parse_config(Path(state["path"]), raw)
            except UnsupportedConfig as exc:
                return {
                    **state,
                    "kind": "excluded",
                    "restorable": False,
                    "error": str(exc),
                }
            if hashlib.sha256(raw).hexdigest() != digest:
                raise FileEditConflict(f"file changed during protected capture: {path}")
            reference = self.protected.freeze(
                raw, workspace_id=workspace_id, metadata=metadata
            )
            state.update(
                protected_content=reference.to_mapping(),
                version=reference.sha256,
                metadata={"size": metadata["size"]},
            )
        else:
            reference = self._freeze_regular(
                source,
                digest,
                workspace_id,
                size=metadata["size"],
                cancellation=cancellation,
            )
            state.update(
                content=reference.to_mapping(), version=digest, metadata=metadata
            )
        if (
            _file_signature(path) != signature
            or read_security(path) != metadata["security"]
        ):
            raise FileEditConflict(f"file identity changed during capture: {path}")
        _check_capture_cancelled(cancellation)
        return {
            **state,
            "kind": "file",
            "restorable": metadata["nlink"] == 1,
            **(
                {"error": "hard_link_semantics_not_supported"}
                if metadata["nlink"] != 1
                else {}
            ),
        }

    def _freeze_regular(
        self,
        source: BinaryIO,
        digest: str,
        workspace_id: str,
        *,
        size: int,
        cancellation: CancellationToken | None,
    ) -> ContentReference:
        """冻结普通内容或核验同内容原件；参数：固定流/已核验摘要/归属/大小/取消；返回：真实引用。"""
        key = (digest, size)
        known = self._regular_contents.get(key)
        if known is not None:
            # 1. 【文件恢复】【相同内容】当前文件已完整哈希，只复用实际存在且核验通过的冻结原件
            with self.database.prepare_reference(known, cancellation=cancellation):
                pass
            return known
        reference = self.database.prepare_stream(
            source, workspace_id=workspace_id, cancellation=cancellation
        )
        if reference.sha256 != digest:
            raise FileEditConflict("file changed while freezing captured content")
        self._regular_contents[key] = reference
        return reference

    def capture_displaced(
        self,
        path: Path,
        original_state: Mapping[str, Any],
        lease: Lease,
        *,
        workspace_id: str,
    ) -> dict[str, Any]:
        """归档实际被移走的原件；参数：私有备份、原状态、权限、工作区；返回：仍按原路径保护的准确版本。"""
        original = Path(original_state["path"])
        sensitive = bool(original_state.get("sensitive"))
        if (
            path_security.check_read(original, lease, filtered=sensitive)
            is not path_security.Decision.ALLOWED
        ):
            raise PermissionError("displaced original read permission denied")
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode) or getattr(status, "st_reparse_tag", False):
            raise ValueError("displaced original is not a regular file")
        state = {
            **missing_state(str(original)),
            "sensitive": sensitive,
            "identity": f"{status.st_dev}:{status.st_ino}",
            "metadata": {
                "mode": status.st_mode,
                "mtime_ns": status.st_mtime_ns,
                "ctime_ns": status.st_ctime_ns,
                "nlink": status.st_nlink,
                "size": status.st_size,
            },
        }
        with stable_file_read(path) as source:
            return self._capture_regular(path, state, workspace_id, source)

    def _reuse_original(
        self,
        current: dict[str, Any],
        metadata: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any] | None:
        """完整校验后复用冻结对象；传参：本次状态、内部元信息、取消信号；返回：可复用状态或无。"""
        previous = self._last_states.get(current["path"])
        if (
            previous is None
            or previous["kind"] != "file"
            or previous["identity"] != current["identity"]
            or previous["sensitive"] != current["sensitive"]
            or self.state_metadata(previous) != metadata
        ):
            return None
        for _ in self.iter_state(previous, cancellation=cancellation):
            pass
        self._reused_files += 1
        return {**previous, "captured_at": current["captured_at"]}

    def _load_previous_inventory(self, workspace_id: str) -> None:
        """重启后只加载最近一次范围清单作为复用候选；传参：工作区；返回：无，候选仍需当前完整内容校验。"""
        if self._last_states:
            return
        with self.database.snapshot() as source:
            points = source.list_raw(
                "file_restore_point",
                workspace_id=workspace_id,
                filters={"scope": "workspace"},
            )
            if not points:
                return
            latest = max(
                points, key=lambda point: str(point.payload.get("started_at", ""))
            )
            point = source.get("file_restore_point", latest.record_id)
        if point is not None:
            self._last_states = {
                entry["path"]: entry.get("after") or entry["before"]
                for entry in point["entries"]
            }

    def state_metadata(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """读取恢复所需权限信息；传参：文件状态；返回：内部元信息，敏感信息不得直接给模型。"""
        if state.get("protected_content"):
            return dict(
                self.protected.metadata(
                    ContentReference.from_mapping(state["protected_content"])
                )
            )
        return dict(state.get("metadata", {}))

    def read_state(self, state: Mapping[str, Any]) -> bytes:
        """读取准确历史字节；传参：已保存文件状态；返回：原件，缺失或损坏明确失败。"""
        if state.get("kind") != "file":
            raise ValueError("state does not contain a regular file")
        if state.get("protected_content"):
            return self.protected.read(
                ContentReference.from_mapping(state["protected_content"])
            )
        return self.database.read_content(
            ContentReference.from_mapping(state["content"])
        )

    def iter_state(
        self, state: Mapping[str, Any], *, cancellation: CancellationToken | None = None
    ) -> Iterator[bytes]:
        """流式读取普通原件；传参：文件状态和取消信号；返回：完整块，受保护配置由DPAPI整体解密。"""
        _check_capture_cancelled(cancellation)
        if state.get("kind") != "file":
            raise ValueError("state does not contain a regular file")
        if state.get("protected_content"):
            yield self.read_state(state)
        else:
            yield from self.database.iter_content(
                ContentReference.from_mapping(state["content"]),
                cancellation=cancellation,
            )

    def validate_state(self, state: Mapping[str, Any]) -> None:
        """验证历史目标是否真的可用；传参：文件状态；返回：无，损坏或不可恢复直抛。"""
        if not state.get("restorable"):
            raise ValueError(str(state.get("error", "file state is not restorable")))
        if state.get("kind") == "missing":
            return
        for _ in self.iter_state(state):
            pass

    def same_state(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        *,
        identity: bool = False,
    ) -> bool:
        """比较内容或预览版本；传参：两份状态、是否同时核对身份权限；返回：是否相同。"""
        if left.get("kind") != right.get("kind"):
            return False
        if left.get("kind") == "missing":
            return True
        first, second = self.state_metadata(left), self.state_metadata(right)
        same = first.get("sha256") == second.get("sha256")
        if identity:
            same = (
                same
                and left.get("identity") == right.get("identity")
                and first == second
            )
        return same

    def publish_point(
        self, point: dict[str, Any], *, cancellation: CancellationToken | None = None
    ) -> None:
        """以共享短提交保存恢复点；传参：完整点和取消信号；返回：无，已发布前态不能被改写。"""
        _check_capture_cancelled(cancellation)
        prepared = self.database.prepare_payload(
            point, workspace_id=point["workspace_id"]
        )
        references = []
        for entry in point["entries"]:
            _check_capture_cancelled(cancellation)
            for field in ("before", "after", "displaced_before"):
                state = entry.get(field) or {}
                reference = state.get("protected_content") or state.get("content")
                if reference:
                    references.append(ContentReference.from_mapping(reference))
        _check_capture_cancelled(cancellation)
        with self.database.prepare_references(
            references, cancellation=cancellation
        ) as originals:
            with self.database.transaction() as batch:
                previous = batch.get("file_restore_point", point["point_id"])
                if previous is not None:
                    frozen = {
                        entry["path"]: entry["before"] for entry in previous["entries"]
                    }
                    current = {
                        entry["path"]: entry["before"] for entry in point["entries"]
                    }
                    if any(
                        current.get(path) != state for path, state in frozen.items()
                    ):
                        raise ValueError(
                            "published restore originals cannot be replaced"
                        )
                batch.put(
                    "file_restore_point",
                    point["point_id"],
                    prepared,
                    workspace_id=point["workspace_id"],
                    session_id=point.get("session_id") or None,
                )
                for original in originals:
                    batch.reference_prepared_content(original)

    def get_point(self, point_id: str) -> dict[str, Any]:
        """读取一个恢复点；传参：点编号；返回：不可变原件的独立视图。"""
        with self.database.snapshot() as source:
            point = source.get("file_restore_point", point_id)
        if point is None:
            raise ValueError("file restore point does not exist")
        return point

    def list_points(
        self, workspace_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """按原工作区读取恢复点；传参：工作区和可选会话；返回：新点优先的列表。"""
        with self.database.snapshot() as source:
            points = source.list(
                "file_restore_point", workspace_id=workspace_id, session_id=session_id
            )
        return sorted(points, key=lambda point: point["started_at"], reverse=True)

    def capture_workspace(
        self,
        root: Path,
        lease: Lease,
        workspace_id: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """捕获声明根的完整清单；传参：根、权限、工作区、取消信号；返回：逐文件状态、排除项和真实错误。"""
        boundary = path_security.ReadBoundary.from_lease(lease)
        with ExitStack() as readers:
            self._read_boundary, self._capture_reads = boundary, readers
            try:
                result = self._capture_workspace(
                    root, lease, workspace_id, cancellation=cancellation
                )
                if path_security.ReadBoundary.from_lease(lease) != boundary:
                    raise FileEditConflict(
                        "read permission roots changed during workspace capture"
                    )
            finally:
                self._read_boundary, self._capture_reads = None, None
        return result

    def _capture_workspace(
        self,
        root: Path,
        lease: Lease,
        workspace_id: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """在固定读取句柄下完成清单与状态核对；参数：根/租约/工作区/取消；返回：本次完整观察。"""
        from runtime.file_upgrade import retained_staging_paths

        _check_capture_cancelled(cancellation)
        started = time.monotonic()
        self._load_previous_inventory(workspace_id)
        first_capture, reused_before = not self._last_states, self._reused_files
        retained = retained_staging_paths(self.database)
        staging_roots = {path if path.is_dir() else path.parent for path in retained}
        paths, exclusions = _inventory(
            root, self.data_root, staging_roots, cancellation=cancellation
        )
        states, errors = {}, []
        for path in paths:
            _check_capture_cancelled(cancellation)
            try:
                state = self.capture_file(
                    path, lease, workspace_id, cancellation=cancellation
                )
                states[str(path)] = state
                if state["kind"] == "excluded":
                    exclusions.append({"path": str(path), "reason": state["error"]})
            except (OSError, ValueError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
        # 2. 【文件恢复】【范围核验】逐文件清单不是原子卷快照；扫描期间变化必须暴露
        final_paths, final_exclusions = _inventory(
            root, self.data_root, staging_roots, cancellation=cancellation
        )
        if (
            paths != final_paths
            or exclusions[: len(final_exclusions)] != final_exclusions
        ):
            errors.append(
                {
                    "path": str(root),
                    "error": "workspace inventory changed during capture",
                }
            )
        errors.extend(self._verify_captured_states(states, cancellation=cancellation))
        self._last_states = dict(states)
        return {
            "states": states,
            "exclusions": exclusions,
            "errors": errors,
            "metrics": {
                "duration_seconds": time.monotonic() - started,
                "files": len(states),
                "first_capture": first_capture,
                "reused_files": self._reused_files - reused_before,
                "bytes": sum(
                    state["metadata"].get("size", 0) for state in states.values()
                ),
            },
        }

    def _verify_captured_states(
        self,
        states: Mapping[str, dict[str, Any]],
        *,
        cancellation: CancellationToken | None,
    ) -> list[dict[str, str]]:
        """核对固定句柄期间的路径身份与权限；参数：已捕获状态/取消；返回：实际变化或读取错误。"""
        errors = []
        for recorded_path, state in states.items():
            _check_capture_cancelled(cancellation)
            if state["kind"] not in {"file", "directory"}:
                continue
            try:
                metadata = self.state_metadata(state)
                current_path = Path(recorded_path)
                current = current_path.stat()
                changed = (
                    f"{current.st_dev}:{current.st_ino}",
                    current.st_mtime_ns,
                    current.st_ctime_ns,
                ) != (state["identity"], metadata["mtime_ns"], metadata["ctime_ns"])
                if state["kind"] == "file":
                    # 3. 【文件恢复】【范围核验】已核验句柄一直禁止写入和删除，只重核路径身份与权限
                    with stable_file_read(current_path) as source:
                        status = os.fstat(source.fileno())
                        signature = (
                            f"{status.st_dev}:{status.st_ino}",
                            status.st_size,
                            file_change_time(source.fileno()),
                        )
                        changed = (
                            changed
                            or signature
                            != (
                                state["identity"],
                                metadata["size"],
                                metadata["change_time"],
                            )
                            or read_security(current_path) != metadata["security"]
                        )
                if changed:
                    errors.append(
                        {
                            "path": recorded_path,
                            "error": "file changed during workspace capture",
                        }
                    )
            except (OSError, ValueError) as exc:
                errors.append({"path": recorded_path, "error": str(exc)})
        return errors


def new_point(
    identity: Mapping[str, str], *, scope: str, attribution: str
) -> dict[str, Any]:
    """创建一次真实执行的捕获身份；传参：归属、范围、归因；返回：尚未发布的恢复点。"""
    return {
        **identity,
        "point_id": f"point-{uuid4().hex}",
        "scope": scope,
        "attribution": attribution,
        "status": "before_saved",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "completed_at": None,
        "entries": [],
        "exclusions": [],
        "errors": [],
        "metrics": {},
    }


def missing_state(path: str) -> dict[str, Any]:
    """表达完整清单确认的不存在状态；传参：绝对路径；返回：可用于撤销新增的前态。"""
    return {
        "path": path,
        "kind": "missing",
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "content": None,
        "protected_content": None,
        "identity": None,
        "version": None,
        "metadata": {},
        "sensitive": False,
        "restorable": True,
    }


def _file_signature(path: Path) -> tuple[str, int, int]:
    """取得真实身份及Windows变化时间；传参：路径；返回：不依赖mtime的核验签名。"""
    with path.open("rb") as handle:
        status = os.fstat(handle.fileno())
        return (
            f"{status.st_dev}:{status.st_ino}",
            status.st_size,
            file_change_time(handle.fileno()),
        )


def _inspect_file(
    path: Path, source: BinaryIO, *, cancellation: CancellationToken | None = None
) -> tuple[str, tuple[str, int, int], bool]:
    """完整流式核对并筛除私钥；传参：路径、文件流、取消信号；返回：摘要、稳定身份、是否含私钥。"""
    digest, private_key, tail = hashlib.sha256(), False, b""
    source.seek(0)
    status = os.fstat(source.fileno())
    signature = (
        f"{status.st_dev}:{status.st_ino}",
        status.st_size,
        file_change_time(source.fileno()),
    )
    while True:
        _check_capture_cancelled(cancellation)
        chunk = source.read(CONTENT_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        private_key = private_key or is_private_key(tail + chunk)
        tail = chunk[-_PRIVATE_KEY_OVERLAP_BYTES:]
    if file_change_time(source.fileno()) != signature[2]:
        raise FileEditConflict(f"file changed during capture: {path}")
    return digest.hexdigest(), signature, private_key


def _inventory(
    root: Path,
    data_root: Path,
    staging_roots: set[Path],
    *,
    cancellation: CancellationToken | None = None,
) -> tuple[list[Path], list[dict[str, str]]]:
    """枚举用户资料并记录精确排除原因；传参：工作区、数据根、暂存根、取消信号；返回：路径与排除项。"""
    pending, paths, exclusions = [root.resolve()], [], []
    while pending:
        _check_capture_cancelled(cancellation)
        parent = pending.pop()
        with os.scandir(parent) as iterator:
            children = []
            for child in iterator:
                _check_capture_cancelled(cancellation)
                children.append(child)
            children.sort(key=lambda entry: entry.name.casefold())
        for child in children:
            _check_capture_cancelled(cancellation)
            path = Path(child.path)
            reparse = child.is_symlink() or bool(
                getattr(child.stat(follow_symlinks=False), "st_reparse_tag", False)
            )
            resolved = path.resolve() if reparse else path
            reason = "runtime_data_root" if resolved.is_relative_to(data_root) else ""
            if (
                child.is_dir(follow_symlinks=False)
                and child.name in EXCLUDED_DIRECTORIES
            ):
                reason = "dependency_cache_or_git_metadata"
            if resolved in staging_roots:
                reason = "registered_restore_staging"
            if reason:
                exclusions.append({"path": str(path), "reason": reason})
                continue
            paths.append(path)
            if child.is_dir(follow_symlinks=False) and not reparse:
                pending.append(path)
    _check_capture_cancelled(cancellation)
    return sorted(paths), sorted(exclusions, key=lambda row: row["path"])


def _check_capture_cancelled(cancellation: CancellationToken | None) -> None:
    """在目录及内容边界响应停止；传参：当前执行取消信号；返回：无，停止时抛出真实取消。"""
    if cancellation is not None and cancellation.cancelled:
        raise ExecutionCancelled("文件恢复前态捕获已取消，未启动文件动作")


def _supported_sensitive_path(path: Path) -> bool:
    """敏感路径只读取已有解析器支持的配置；传参：路径；返回：是否可走密文恢复通道。"""
    return (
        path.name in {".env", "credentials", "config"}
        or path.name.startswith(".env.")
        or path.suffix.casefold() in {".ini", ".cfg", ".conf", ".env"}
    )
