"""【存储】【正文原件】按工作区去重并校验不可变内容。

作者：xxx
时间：2026-09-30 15:00:00
"""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO

from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.file_records import (
    ContentReference,
    PAYLOAD_MIN_BYTES,
    SourceCorruptionError,
    digest_bytes,
)

CONTENT_CHUNK_BYTES = 64 * 1024
SOURCE_PATH_CACHE_ENTRIES = 4096


@dataclass
class _VerifiedContentScope:
    """一次实际工具执行中持续固定的原件，退出后不再复用完整性证据。"""

    root: Path
    readers: ExitStack
    originals: dict[ContentReference, tuple[int, int]] = field(default_factory=dict)


_VERIFIED_CONTENT: ContextVar[_VerifiedContentScope | None] = ContextVar(
    "verified_content", default=None
)


@contextmanager
def verified_content_scope(root: Path) -> Iterator[None]:
    """同次捕获和发布共用已固定原件，结束必释放；参数：数据根；返回：仅本上下文有效的核验范围。"""
    active = _VERIFIED_CONTENT.get()
    if active is not None and active.root == root:
        yield
        return
    with ExitStack() as readers:
        token = _VERIFIED_CONTENT.set(_VerifiedContentScope(root, readers))
        try:
            yield
        finally:
            _VERIFIED_CONTENT.reset(token)


@lru_cache(maxsize=SOURCE_PATH_CACHE_ENTRIES)
def confined_path(root: Path, relative: str) -> Path:
    """约束原件路径在空间内；参数：根和相对定位；返回：安全绝对路径。"""
    candidate = root / relative
    resolved = candidate.resolve()
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not resolved.is_relative_to(root)
    ):
        raise SourceCorruptionError(f"source path escapes data space: {relative}")
    return resolved


class ContentFiles:
    """只管理内容字节，不判断附件、消息或知识业务类型。"""

    def __init__(self, root: Path) -> None:
        """绑定空间；参数：资料根；返回：无长期文件句柄的内容服务。"""
        self.root = root

    def prepare(
        self, content: bytes, directory: Path, media_type: str
    ) -> ContentReference:
        """在发布事务外准备去重内容；参数：字节、所属目录、媒体类型；返回：完整引用。"""
        from tools.file_persistence import FileEditConflict, publish_file

        digest = digest_bytes(content)
        suffix = ".txt" if media_type.startswith("text/") else ".bin"
        path = directory / "objects" / digest[:2] / (digest + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != content:
                raise SourceCorruptionError(
                    f"content digest collision or corruption: {path}"
                )
        else:
            try:
                publish_file(path, None, content)
            except FileEditConflict:
                if path.read_bytes() != content:
                    raise SourceCorruptionError(
                        f"content digest collision or corruption: {path}"
                    ) from None
        return ContentReference(
            path.relative_to(self.root).as_posix(), digest, len(content), media_type
        )

    def prepare_path(
        self, source: Path, directory: Path, media_type: str
    ) -> ContentReference:
        """流式冻结外部文件；参数：源文件、所属目录、类型；返回：完整且不可变的内容引用。"""
        from tools.restore_protection import stable_file_read

        with stable_file_read(source) as original:
            return self.prepare_stream(original, directory, media_type)

    def prepare_stream(
        self,
        original: BinaryIO,
        directory: Path,
        media_type: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ContentReference:
        """冻结固定句柄并响应停止；参数：完整流、归属、类型、取消信号；返回：完整引用，取消清理临时文件。"""
        from tools.file_persistence import (
            FileEditConflict,
            file_change_time,
            publish_prepared_file,
        )

        object_root = directory / "objects"
        object_root.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile("wb", dir=object_root, delete=False) as target:
                original.seek(0)
                temporary = Path(target.name)
                before = os.fstat(original.fileno())
                original_change = file_change_time(original.fileno())
                digest = hashlib.sha256()
                count = 0
                while True:
                    if cancellation is not None and cancellation.cancelled:
                        raise ExecutionCancelled("文件原件冻结已取消")
                    chunk = original.read(CONTENT_CHUNK_BYTES)
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                after = os.fstat(original.fileno())
                if (before.st_size, original_change) != (
                    after.st_size,
                    file_change_time(original.fileno()),
                ):
                    raise FileEditConflict("source file changed while freezing content")
                target.flush()
                os.fsync(target.fileno())
            identity = digest.hexdigest()
            suffix = ".txt" if media_type.startswith("text/") else ".bin"
            path = object_root / identity[:2] / (identity + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            reference = ContentReference(
                path.relative_to(self.root).as_posix(), identity, count, media_type
            )
            if not path.exists():
                try:
                    publish_prepared_file(path, temporary)
                except FileEditConflict:
                    with self._open_verified(reference, cancellation=cancellation):
                        pass
            else:
                with self._open_verified(reference, cancellation=cancellation):
                    pass
            return reference
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def read(
        self, reference: ContentReference, *, offset: int = 0, limit: int | None = None
    ) -> bytes:
        """读取且核对正文原件；参数：引用及字节分页；返回：准确字节，坏源绝不返回前缀。"""
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError("content offset and limit cannot be negative")
        with self._open_verified(reference) as handle:
            handle.seek(offset)
            return handle.read() if limit is None else handle.read(limit)

    @contextmanager
    def _open_verified(
        self,
        reference: ContentReference,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[BinaryIO]:
        """完整校验后固定只读句柄；参数：引用与取消信号；返回：退出前禁止写入和删除的原件。"""
        from tools.file_persistence import file_change_time
        from tools.restore_protection import stable_file_read

        path = confined_path(self.root, reference.path)
        if cancellation is not None and cancellation.cancelled:
            raise ExecutionCancelled("文件原件核验已取消")
        with ExitStack() as access:
            try:
                handle = access.enter_context(stable_file_read(path))
            except FileNotFoundError as exc:
                raise SourceCorruptionError(
                    f"content missing or corrupt: {reference.path}"
                ) from exc
            stat = os.fstat(handle.fileno())
            signature = (
                stat.st_size,
                stat.st_ino,
                file_change_time(handle.fileno()),
                reference.sha256,
            )
            if stat.st_size != reference.size:
                raise SourceCorruptionError(
                    f"content missing or corrupt: {reference.path}"
                )
            self._verify_open_content(
                path, reference, handle, cancellation=cancellation
            )
            handle.seek(0)
            yield handle
            final = os.fstat(handle.fileno())
            if (
                final.st_size,
                final.st_ino,
                file_change_time(handle.fileno()),
            ) != signature[:3]:
                raise SourceCorruptionError(
                    f"content changed while reading: {reference.path}"
                )

    def _verify_open_content(
        self,
        path: Path,
        reference: ContentReference,
        handle: BinaryIO,
        *,
        cancellation: CancellationToken | None,
    ) -> None:
        """完整核验或复用持续固定的同一原件；参数：路径/引用/实际句柄/取消；返回：无，坏源抛错。"""
        from tools.restore_protection import stable_file_read

        scope = _VERIFIED_CONTENT.get()
        scope = scope if scope is not None and scope.root == self.root else None
        status = os.fstat(handle.fileno())
        identity = (status.st_dev, status.st_ino)
        if scope is not None and scope.originals.get(reference) == identity:
            return
        # 1. 【存储】【原件核验】仅持续持有禁止写删的句柄才复用事实，不靠时间戳缓存证明完整性
        digest = hashlib.sha256()
        while chunk := handle.read(CONTENT_CHUNK_BYTES):
            if cancellation is not None and cancellation.cancelled:
                raise ExecutionCancelled("文件原件核验已取消")
            digest.update(chunk)
        if digest.hexdigest() != reference.sha256:
            raise SourceCorruptionError(f"content missing or corrupt: {reference.path}")
        if scope is not None:
            guard = scope.readers.enter_context(stable_file_read(path))
            guarded = os.fstat(guard.fileno())
            if (guarded.st_dev, guarded.st_ino) != identity:
                raise SourceCorruptionError(
                    f"content changed while pinning: {reference.path}"
                )
            scope.originals[reference] = identity

    def iterate(
        self,
        reference: ContentReference,
        *,
        chunk_size: int = CONTENT_CHUNK_BYTES,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[bytes]:
        """常量内存读取已冻结内容；参数：引用、分块大小、取消信号；返回：字节块迭代器。"""
        if chunk_size <= 0:
            raise ValueError("content chunk size must be positive")
        with self._open_verified(reference, cancellation=cancellation) as handle:
            while chunk := handle.read(chunk_size):
                if cancellation is not None and cancellation.cancelled:
                    raise ExecutionCancelled("文件原件读取已取消")
                yield chunk

    def pack(
        self, value: Mapping[str, Any], directory: Path
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        """把大正文移入明确引用表；参数：领域载荷与目录；返回：普通JSON和外置位置表。"""
        references: list[dict[str, Any]] = []
        scope = _VERIFIED_CONTENT.get()
        prepared: dict[str, ContentReference] | None = (
            {} if scope is not None and scope.root == self.root else None
        )
        payload = self._pack_value(value, directory, (), references, prepared=prepared)
        if not isinstance(payload, dict):
            raise TypeError("record payload must be an object")
        return payload, tuple(references)

    def _pack_value(
        self,
        value: Any,
        directory: Path,
        pointer: tuple[str | int, ...],
        references: list[dict[str, Any]],
        *,
        prepared: dict[str, ContentReference] | None,
    ) -> Any:
        """递归保留JSON形状；参数：节点、目录、位置、引用输出；返回：不混淆用户标记的节点。"""
        if isinstance(value, str) and len(value.encode("utf-8")) >= PAYLOAD_MIN_BYTES:
            reference = self._prepare_payload_text(value, directory, prepared)
            references.append(
                {"pointer": list(pointer), "content": reference.to_mapping()}
            )
            return None
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("persistent JSON object keys must be strings")
            return {
                key: self._pack_value(
                    item, directory, (*pointer, key), references, prepared=prepared
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self._pack_value(
                    item, directory, (*pointer, index), references, prepared=prepared
                )
                for index, item in enumerate(value)
            ]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"unsupported persistent JSON value: {type(value).__name__}")

    def _prepare_payload_text(
        self, value: str, directory: Path, prepared: dict[str, ContentReference] | None
    ) -> ContentReference:
        """准备一份大正文并复用本批固定原件；参数：原文/目录/本批引用；返回：真实内容引用。"""
        reference = prepared.get(value) if prepared is not None else None
        if reference is not None:
            return reference
        reference = self.prepare(
            value.encode("utf-8"), directory, "text/plain; charset=utf-8"
        )
        if prepared is not None:
            # 1. 【存储】【重复元信息】相同权限等正文只冻结一次，复用前固定原件直到本次执行结束
            with self._open_verified(reference):
                pass
            prepared[value] = reference
        return reference

    def unpack(
        self, payload: dict[str, Any], references: tuple[dict[str, Any], ...]
    ) -> dict[str, Any]:
        """展开明确登记的字符串引用；参数：载荷与位置表；返回：独立领域值。"""
        result = copy.deepcopy(payload)
        try:
            for item in references:
                pointer = item["pointer"]
                target: Any = result
                for part in pointer[:-1]:
                    target = target[part]
                if not pointer or target[pointer[-1]] is not None:
                    raise SourceCorruptionError(
                        "content reference does not point to an empty payload slot"
                    )
                target[pointer[-1]] = self.read(
                    ContentReference.from_mapping(item["content"])
                ).decode("utf-8")
        except (KeyError, IndexError, TypeError, UnicodeError) as exc:
            raise SourceCorruptionError(
                "invalid content reference location or UTF-8"
            ) from exc
        return result
