"""【记忆】【文件原件】管理可编辑Markdown、不可变修订与已发布身份。

作者：xxx
时间：2026-09-30 16:00:00
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from memory.records import (
    MEMORY_FORMAT,
    Memory,
    MemorySource,
    format_memory,
    parse_memory,
    seal_memory,
)
from runtime.file_content import confined_path
from runtime.persistence import RuntimeStore
from schedules.persistence import immutable_bytes, record_path
from tasks.ids import new_ulid, utc_now
from tools.file_persistence import FileEditConflict, publish_file, read_file_bytes
from tools.file_persistence import content_sha256

ValidateRevision = Callable[[Memory, Memory | None, list[Memory]], None]


class MemoryPublicationError(RuntimeError):
    """当前Markdown已保存，但后续清理或登记失败，不能把操作当成未保存重做。"""

    def __init__(
        self, memory: Memory, error: Exception, *, phase: str = "registration"
    ) -> None:
        """保留已提交身份和失败步骤；参数：记忆、异常和阶段；返回：无。"""
        self.memory_id, self.version = memory.memory_id, memory.version
        self.phase = phase
        super().__init__(
            f"memory committed: {memory.memory_id}@{memory.version}; {phase} failed: {error}"
        )


class MemoryFiles:
    """当前稿决定发布版本；目录登记只负责发现缺失，不保存第二份正文。"""

    def __init__(self, root: Path, database: RuntimeStore) -> None:
        """绑定同一数据空间；参数：数据根及共享存储；返回：无。"""
        self.root, self.database = root, database

    def directory(self, memory: Memory) -> Path:
        """保持逻辑作用域并定位物理目录；参数：记忆；返回：所属知识目录。"""
        scope = memory.details.scope
        if scope.startswith("project:"):
            return self.database.workspace_directory(scope.split(":", 1)[1]) / "memory"
        if scope.startswith("session:"):
            with self.database.snapshot() as source:
                identity = source.get("memory_identity", memory.memory_id)
                # 1. 【会话记忆】【目录归属】首次绑定工作区不搬迁已发布记忆；改作用域时才重新选择目录
                if identity is not None and identity["scope"] == scope:
                    return confined_path(self.root, identity["path"]).parent
                binding = source.get("session_workspace", scope.split(":", 1)[1])
            if binding is not None:
                return (
                    self.database.workspace_directory(str(binding["workspace_id"]))
                    / "memory"
                )
        return self.root / "global" / "memory"

    def scan(self) -> dict[str, tuple[Path, bytes, Memory]]:
        """枚举当前稿并拒绝重复身份、坏前言和已发布文件丢失；参数：无；返回：当前文件快照。"""
        with self.database.snapshot() as source:
            identities = source.list("memory_identity")
        moves = {
            item["relocation"]["previous_path"]: item["relocation"]
            for item in identities
            if "relocation" in item
        }
        roots = [
            self.root / "global" / "memory",
            *(self.root / "workspaces").glob("*/memory"),
        ]
        found: dict[str, tuple[Path, bytes, Memory]] = {}
        for root in roots:
            for path in sorted(root.glob("*.md")):
                raw = path.read_bytes()
                move = moves.get(path.relative_to(self.root).as_posix())
                if move is not None and (self.root / move["path"]).is_file():
                    if content_sha256(raw) != move["previous_sha256"]:
                        raise FileEditConflict(
                            f"memory source changed during scope relocation: {path}"
                        )
                    continue
                try:
                    memory = parse_memory(raw.decode("utf-8"), editable=True)
                    if memory.format_version != MEMORY_FORMAT:
                        raise ValueError("unsupported editable memory format")
                    record_path(root, memory.memory_id)
                    if self.directory(memory).resolve() != root.resolve():
                        raise ValueError("memory directory does not match its scope")
                    if memory.memory_id in found:
                        raise ValueError(
                            f"duplicate memory identity: {memory.memory_id}"
                        )
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise ValueError(f"invalid memory original: {path}: {exc}") from exc
                found[memory.memory_id] = path, raw, memory
        for item in identities:
            if item["memory_id"] not in found:
                raise FileNotFoundError(
                    f"published memory original missing: {self.root / item['path']}"
                )
        return found

    def synchronize(
        self, validate: ValidateRevision
    ) -> tuple[list[Memory], tuple[Memory, ...]]:
        """识别编辑、重命名及新稿，提交可追溯修订；参数：既有业务校验；返回：有效原件集合。"""
        files = self.scan()
        records = [item[2] for item in files.values()]
        result: list[Memory] = []
        committed: list[Memory] = []
        for path, raw, candidate in files.values():
            revision_path = (
                self.revision_path(path, candidate.memory_id, candidate.version)
                if candidate.version
                else None
            )
            if revision_path is None or not revision_path.exists():
                # 1. 【记忆】【新建编辑稿】新身份仍需完整前言，系统计算初始版本
                with self.database.snapshot() as source:
                    known = source.get("memory_identity", candidate.memory_id)
                if known is not None:
                    raise FileNotFoundError(
                        f"published memory revision missing: {revision_path}"
                    )
                observation = MemorySource(
                    "external_edit", reference=str(path), observed_at=utc_now()
                )
                candidate = seal_memory(
                    replace(
                        candidate,
                        revision=1,
                        previous_version=None,
                        last_verified_at=None,
                        verification=(),
                        reason="external Markdown creation",
                        details=replace(
                            candidate.details,
                            sources=(*candidate.details.sources, observation),
                        ),
                    )
                )
                validate(
                    candidate,
                    None,
                    [item for item in records if item.memory_id != candidate.memory_id],
                )
                self.publish(candidate, path=path, original=raw)
                committed.append(candidate)
            else:
                original = self.read_revision(
                    path, candidate.memory_id, candidate.version
                )
                self.validate_chain(path, original)
                self._require_published_ancestor(path, original)
                candidate = self._accept_edit(
                    path,
                    raw,
                    candidate,
                    original=original,
                    records=records,
                    validate=validate,
                )
                if candidate.version != original.version:
                    committed.append(candidate)
            self.register(candidate, path)
            result.append(candidate)
        return result, tuple(committed)

    def _accept_edit(
        self,
        path: Path,
        raw: bytes,
        candidate: Memory,
        *,
        original: Memory,
        records: list[Memory],
        validate: ValidateRevision,
    ) -> Memory:
        """接纳真实外部编辑并清除旧核验；参数：文件快照、编辑稿、原版及业务校验；返回：当前版本。"""
        candidate = replace(
            candidate,
            updated_at=original.updated_at,
            revision=original.revision,
            previous_version=original.previous_version,
            reason=original.reason,
            change_id=original.change_id,
        )
        if seal_memory(candidate).version == original.version:
            return original
        # 2. 【记忆】【外部编辑】身份、范围、来源、发布关系和核验不由手改前言伪造
        protected = (
            "memory_id",
            "type",
            "created_at",
            "format_version",
            "last_verified_at",
            "verification",
        )
        if any(getattr(candidate, key) != getattr(original, key) for key in protected):
            raise ValueError(f"protected memory metadata changed: {path}")
        protected_details = (
            "kind",
            "scope",
            "subject",
            "fact_key",
            "sources",
            "supersedes",
        )
        if any(
            getattr(candidate.details, key) != getattr(original.details, key)
            for key in protected_details
        ):
            raise ValueError(
                f"protected memory scope/source/replacement changed: {path}"
            )
        observed = utc_now()
        source = MemorySource(
            "external_edit", reference=str(path), observed_at=observed
        )
        edited = seal_memory(
            replace(
                candidate,
                revision=original.revision + 1,
                previous_version=original.version,
                updated_at=observed,
                created_at=original.created_at,
                last_verified_at=None,
                verification=(),
                details=replace(
                    candidate.details, sources=(*original.details.sources, source)
                ),
                reason="external Markdown edit",
                change_id=None,
                last_used_at=None,
            )
        )
        validate(edited, original, records)
        self.publish(edited, path=path, original=raw)
        return edited

    def publish(
        self, memory: Memory, *, path: Path | None = None, original: bytes | None = None
    ) -> Path:
        """先同步修订再比较当前字节并原子发布；参数：新版与期望原件；返回：当前稿路径。"""
        target = path or record_path(
            self.directory(memory), memory.memory_id, suffix=".md"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = format_memory(memory).encode("utf-8")
        immutable_bytes(
            self.revision_path(target, memory.memory_id, memory.version), raw
        )
        backup = (
            target.parent / "edit-conflicts" / f"{memory.memory_id}-{new_ulid()}.md"
            if original is not None
            else None
        )
        if backup is not None:
            backup.parent.mkdir(parents=True, exist_ok=True)
        # 2. 【记忆】【并发编辑】最终替换保留实际旧字节，竞争时留下两版供用户核对
        publish_file(target, original, raw, backup_path=backup)
        phase = "backup_cleanup"
        try:
            if backup is not None:
                backup.unlink()
            phase = "registration"
            self.register(memory, target)
        except Exception as exc:
            raise MemoryPublicationError(memory, exc, phase=phase) from exc
        return target

    def register(self, memory: Memory, path: Path) -> None:
        """登记已发布身份与当前位置，重命名不改变身份；参数：记忆及当前路径；返回：无。"""
        payload = {
            "memory_id": memory.memory_id,
            "path": path.relative_to(self.root).as_posix(),
            "scope": memory.details.scope,
            "version": memory.version,
        }
        with self.database.transaction() as batch:
            previous = batch.get("memory_identity", memory.memory_id)
            if previous is not None and "relocation" in previous:
                payload["relocation"] = previous["relocation"]
            if previous != payload:
                batch.put("memory_identity", memory.memory_id, payload)

    def relocate(
        self, memory: Memory, previous: Memory, path: Path, original: bytes
    ) -> None:
        """跨作用域修订先登记搬迁意图，再发布新目录原件；参数：新版、旧版及当前字节；返回：无。"""
        target = record_path(self.directory(memory), memory.memory_id, suffix=".md")
        # 1. 【记忆】【作用域更正】复制可达历史修订，新目录单独保留完整谱系
        revision = previous
        while True:
            raw = self.revision_path(
                path, memory.memory_id, revision.version
            ).read_bytes()
            immutable_bytes(
                self.revision_path(target, memory.memory_id, revision.version), raw
            )
            if revision.previous_version is None:
                break
            revision = self.read_revision(
                path, memory.memory_id, revision.previous_version
            )
        self.require_unchanged(previous)
        intent = {
            "path": target.relative_to(self.root).as_posix(),
            "version": memory.version,
            "previous_path": path.relative_to(self.root).as_posix(),
            "previous_sha256": content_sha256(original),
        }
        with self.database.transaction() as batch:
            identity = batch.get("memory_identity", memory.memory_id)
            if identity is None:
                raise FileNotFoundError(
                    f"memory publication identity missing: {memory.memory_id}"
                )
            batch.put(
                "memory_identity", memory.memory_id, {**identity, "relocation": intent}
            )
        # 2. 【记忆】【作用域更正】新当前稿发布后即为新版；中断时同步入口按意图辨认唯一当前稿
        self.publish(memory, path=target)
        backup = path.parent / "edit-conflicts" / f"{memory.memory_id}-{new_ulid()}.md"
        try:
            backup.parent.mkdir(parents=True, exist_ok=True)
            path.rename(backup)
            if backup.read_bytes() != original:
                raise FileEditConflict(
                    f"memory changed during scope relocation; user bytes preserved at {backup}",
                    backup_path=backup,
                )
            backup.unlink()
        except OSError as exc:
            raise MemoryPublicationError(
                memory, exc, phase="relocation_cleanup"
            ) from exc

    def _require_published_ancestor(self, path: Path, memory: Memory) -> None:
        """禁止手动退回旧版复活已替代知识；参数：当前路径和版本；返回：无。"""
        with self.database.snapshot() as source:
            identity = source.get("memory_identity", memory.memory_id)
        if identity is None:
            return
        while memory.version != identity["version"]:
            if memory.previous_version is None:
                raise ValueError(
                    f"memory publication rollback or unrelated revision: {path}"
                )
            memory = self.read_revision(path, memory.memory_id, memory.previous_version)

    def revision_path(self, current: Path, memory_id: str, version: str) -> Path:
        """定位不可变修订，拒绝越界版本；参数：当前路径、身份、版本；返回：修订路径。"""
        root = record_path(current.parent / "revisions", memory_id, suffix="")
        return record_path(root, version, suffix=".md")

    def read_revision(self, path: Path, memory_id: str, version: str) -> Memory:
        """核验历史身份和内容摘要；参数：当前路径、身份、版本；返回：不可变记忆。"""
        revision = self.revision_path(path, memory_id, version)
        memory = parse_memory(revision.read_text(encoding="utf-8"))
        if memory.memory_id != memory_id or memory.version != version:
            raise ValueError(f"memory revision identity mismatch: {revision}")
        return memory

    def validate_chain(self, path: Path, memory: Memory) -> None:
        """只接受沿当前版可达的完整修订链；参数：当前路径与发布版本；返回：无。"""
        seen = {memory.version}
        while memory.previous_version is not None:
            if memory.previous_version in seen:
                raise ValueError(f"memory revision chain contains a cycle: {path}")
            seen.add(memory.previous_version)
            memory = self.read_revision(path, memory.memory_id, memory.previous_version)

    def require_unchanged(self, memory: Memory) -> None:
        """模型取用前验证已选择的原件版本；参数：冻结候选；返回：无，变化明确报冲突。"""
        files = self.scan()
        if memory.memory_id not in files:
            raise FileNotFoundError(memory.memory_id)
        path, raw, current = files[memory.memory_id]
        if (
            seal_memory(current).version != memory.version
            or read_file_bytes(path) != raw
        ):
            raise FileEditConflict(f"memory changed before use: {memory.memory_id}")
