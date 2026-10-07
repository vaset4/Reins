"""【文件恢复】【基础验收】真实 Windows 锁、密文和捕获证据。

作者：xxx
时间：2026-09-30 22:00:00
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from runtime.file_content import CONTENT_CHUNK_BYTES
from runtime.file_snapshots import SnapshotStore
from runtime.lease import from_trigger
from runtime.workspaces import WorkspaceStore
from tools.file_persistence import FileEditConflict, replace_prepared_file
from tools.restore_protection import (
    ProtectedContentStore,
    create_private_staging,
    handle_security,
    move_handle_file,
    private_handle_security,
    protected_replace_prepared_file,
    read_security,
    security_handle,
    verify_private_acl,
    write_private_file,
)
from tools.workspace_coordination import WorkspaceBusyError, workspace_write_window


def snapshot_environment(tmp_path: Path) -> tuple[Path, SnapshotStore, object, str]:
    """建立真实隔离权限与工作区；参数：临时根；返回：工作根、捕获器、权限和工作区身份。"""
    root = tmp_path / "work"
    root.mkdir()
    store = SnapshotStore(tmp_path / "data")
    workspace = WorkspaceStore(store.data_root).bind_session("foundation-session", root)
    lease = from_trigger(
        "user",
        task_id="foundation",
        capabilities={
            "fs": {
                "project_root": str(root),
                "read": [str(root)],
                "write": [str(root)],
                "default_sensitive_policy": "redacted",
            }
        },
    )
    return root, store, lease, workspace.workspace_id


@pytest.mark.parametrize(
    "held,requested,conflict",
    [
        ("parent", "parent/child/file.txt", True),
        ("parent/child", "parent", True),
        ("parent/file.txt", "parent/file.txt", True),
        ("parent/a.txt", "parent/b.txt", False),
        ("left", "right/file.txt", False),
    ],
)
def test_real_process_physical_scope_admission(
    tmp_path: Path, held: str, requested: str, conflict: bool
) -> None:
    """不同数据根仍协调物理父子目录而允许独立文件；参数：路径关系；返回：无。"""
    root = tmp_path / "physical"
    root.mkdir()
    code = """
import sys
from pathlib import Path
from runtime.persistence import RuntimeStore
from tools.workspace_coordination import workspace_write_window
RuntimeStore(sys.argv[2]).ensure_space()
with workspace_write_window(Path(sys.argv[1]), subtree=not sys.argv[1].endswith('.txt'), owner='child-run'):
    print('locked', flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(root / held), str(tmp_path / "child-data")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "locked"
        SnapshotStore(tmp_path / "other-data").database.ensure_space()
        if conflict:
            with pytest.raises(
                WorkspaceBusyError, match="occupying process is not identified"
            ):
                with workspace_write_window(
                    root / requested,
                    subtree=not requested.endswith(".txt"),
                    owner="other-run",
                ):
                    pytest.fail("overlapping writer entered")
        else:
            with workspace_write_window(
                root / requested, subtree=not requested.endswith(".txt")
            ):
                pass
    finally:
        process.communicate("exit\n", timeout=10)
    assert process.returncode == 0
    with workspace_write_window(root / held, subtree=not held.endswith(".txt")):
        pass


def test_hard_link_aliases_conflict(tmp_path: Path) -> None:
    """不同路径指向同一文件对象仍互斥；参数：临时目录；返回：无。"""
    first, alias = tmp_path / "first", tmp_path / "alias"
    first.write_bytes(b"same object")
    os.link(first, alias)
    with workspace_write_window(first):
        with pytest.raises(WorkspaceBusyError):
            with workspace_write_window(alias):
                pytest.fail("hard link alias entered")


def test_dpapi_roundtrip_integrity_and_private_acl(tmp_path: Path) -> None:
    """秘密只持久化密文且篡改明确失败；参数：隔离根；返回：无。"""
    protected = ProtectedContentStore(tmp_path / "data")
    secret = b"SYNTHETIC_TOKEN=foundation-secret-123"
    reference = protected.freeze(
        secret, workspace_id=None, metadata={"security": "private-evidence"}
    )
    assert protected.read(reference) == secret
    assert protected.metadata(reference) == {"security": "private-evidence"}
    path = protected.data_root / reference.path
    verify_private_acl(path)
    assert secret not in path.read_bytes()
    encrypted = bytearray(path.read_bytes())
    encrypted[-1] ^= 1
    path.write_bytes(encrypted)
    with pytest.raises(ValueError, match="corrupt"):
        protected.read(reference)


def test_sensitive_fixed_handle_rename_preserves_actual_object(tmp_path: Path) -> None:
    """路径被外部换名后仍只移动已收紧权限的原对象；参数：隔离根；返回：无。"""
    target, external = tmp_path / "config", tmp_path / "external"
    target.write_bytes(b"original")
    staging = create_private_staging(tmp_path)
    with security_handle(target, rename=True) as handle:
        original_acl = handle_security(handle)
        private_handle_security(handle)
        target.rename(external)
        target.write_bytes(b"new external file")
        move_handle_file(handle, staging / "before")
    assert target.read_bytes() == b"new external file"
    assert (staging / "before").read_bytes() == b"original"
    verify_private_acl(staging / "before")
    assert original_acl


def test_protected_replace_preserves_acl_and_does_not_overwrite_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """敏感发布竞争创建时保留双方准确字节与私有备份；参数：根和最后窗口注入；返回：无。"""
    import tools.file_persistence as publication

    target = tmp_path / ".env"
    target.write_bytes(b"TOKEN=before")
    original_acl = read_security(target)
    staging = create_private_staging(tmp_path)
    temporary, backup = staging / "new", staging / "old"
    write_private_file(temporary, b"TOKEN=restored")
    publish = publication.publish_prepared_file

    def competing_create(path: Path, prepared: Path) -> None:
        """在实际移动后创建竞争目标；参数：目标与暂存；返回：真实发布结果。"""
        path.write_bytes(b"TOKEN=external")
        publish(path, prepared)

    monkeypatch.setattr(publication, "publish_prepared_file", competing_create)
    with pytest.raises(FileEditConflict):
        protected_replace_prepared_file(target, temporary, backup_path=backup)
    assert target.read_bytes() == b"TOKEN=external"
    assert backup.read_bytes() == b"TOKEN=before"
    verify_private_acl(backup)
    assert read_security(target) == original_acl


def test_plain_replace_retains_last_moment_external_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实 ReplaceFile 在最终竞争后仍保存实际被替换内容；参数：根和边界注入；返回：无。"""
    import tools.file_persistence as publication

    target = tmp_path / "draft"
    target.write_bytes(b"before")
    staging = create_private_staging(tmp_path)
    temporary, backup = staging / "new", staging / "old"
    write_private_file(temporary, b"restored")
    native = publication._KERNEL.ReplaceFileW

    def competing_replace(*arguments: object) -> object:
        """在真正系统调用前追加外部稿；参数：原生发布参数；返回：Windows结果。"""
        Path(str(arguments[0])).write_bytes(b"last external draft")
        return native(*arguments)

    monkeypatch.setattr(publication._KERNEL, "ReplaceFileW", competing_replace)
    replace_prepared_file(target, temporary, backup_path=backup)
    assert target.read_bytes() == b"restored"
    assert backup.read_bytes() == b"last external draft"


def test_snapshot_stream_reuse_private_key_and_user_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """大原件流式去重，私钥不落普通对象，用户同名前缀文件仍覆盖；参数：根和读限制；返回：无。"""
    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    target = root / "large.bin"
    block = b"\x00\xff" * CONTENT_CHUNK_BYTES
    with target.open("wb") as stream:
        for _ in range(32):
            stream.write(block)
    (root / ".reins-restore-user-notes").write_text("my notes", encoding="utf-8")
    key = root / "notes.txt"
    key.write_bytes(
        b"-----BEGIN PRIVATE KEY-----\nSYNTHETIC-KEY\n-----END PRIVATE KEY-----"
    )
    actual_read = Path.read_bytes

    def reject_large_read(path: Path) -> bytes:
        """阻止大文件全量载入内存；参数：路径；返回：小文件内容。"""
        assert path != target
        return actual_read(path)

    monkeypatch.setattr(Path, "read_bytes", reject_large_read)
    first = store.capture_workspace(root, lease, workspace_id)
    second = store.capture_workspace(root, lease, workspace_id)
    assert not first["errors"] and not second["errors"]
    assert second["metrics"]["reused_files"] >= 2
    assert str(root / ".reins-restore-user-notes") in first["states"]
    assert first["states"][str(key)]["kind"] == "excluded"
    assert all(
        b"SYNTHETIC-KEY" not in actual_read(path)
        for path in store.data_root.rglob("*.bin")
    )
    captured = first["states"][str(target)]
    assert (
        hashlib.sha256(b"".join(store.iter_state(captured))).hexdigest()
        == hashlib.sha256(block * 32).hexdigest()
    )


def test_protected_snapshot_never_exposes_secret_or_plain_digest(
    tmp_path: Path,
) -> None:
    """敏感状态与持久事件仅含密文引用，普通对象没有秘密；参数：根；返回：无。"""
    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    secret = b"TOKEN=synthetic-secret-value\n"
    path = root / ".env"
    path.write_bytes(secret)
    state = store.capture_file(path, lease, workspace_id)
    assert state["sensitive"] and store.read_state(state) == secret
    assert hashlib.sha256(secret).hexdigest() not in json.dumps(state)
    assert secret not in json.dumps(state).encode()
    assert store.state_metadata(state)["sha256"] == hashlib.sha256(secret).hexdigest()


def test_restore_blob_commit_uses_verified_generation_outside_global_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """大原件登记不在提交锁重扫正文且变化代次失效明确失败；参数：根和IO观察；返回：无。"""
    import runtime.file_journal as journal
    from runtime.file_snapshots import new_point

    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    path = root / "large.bin"
    path.write_bytes(b"stream" * CONTENT_CHUNK_BYTES)
    state = store.capture_file(path, lease, workspace_id)
    blob = store.data_root / state["content"]["path"]
    digest = journal.file_digest

    def reject_commit_rescan(candidate: Path) -> tuple[int, str]:
        """观察真实提交阶段的大对象读取；参数：候选原件；返回：小文件校验。"""
        assert candidate != blob, "large original was rescanned inside commit"
        return digest(candidate)

    monkeypatch.setattr(journal, "file_digest", reject_commit_rescan)
    point = new_point(
        {"workspace_id": workspace_id, "session_id": "foundation-session"},
        scope="file",
        attribution="exact",
    )
    point["entries"] = [{"path": str(path), "before": state, "after": None}]
    store.publish_point(point)
    assert (
        store.get_point(point["point_id"])["entries"][0]["before"]["version"]
        == state["version"]
    )


def test_killed_sensitive_publication_leaves_registered_private_original(
    tmp_path: Path,
) -> None:
    """在真实移走后强杀保留意图、私有原件并释放范围锁；参数：隔离根；返回：无。"""
    from runtime.file_upgrade import retained_staging_paths
    from runtime.persistence import RuntimeStore

    target, data = tmp_path / ".env", tmp_path / "data"
    target.write_bytes(b"TOKEN=actual-before-crash")
    code = """
import sys,time
from pathlib import Path
import tools.file_persistence as publication
from runtime.persistence import RuntimeStore
from tools.restore_protection import create_private_staging,write_private_file,protected_replace_prepared_file
from tools.workspace_coordination import workspace_write_window
path,data=Path(sys.argv[1]),Path(sys.argv[2])
with workspace_write_window(path.parent,subtree=True):
    staging=create_private_staging(path.parent)
    temporary,backup=staging/'new',staging/'before'
    with RuntimeStore(data).transaction() as batch:
        batch.put('file_restore_operation','crash',{'entries':[{'backup_path':str(backup),'temporary_path':str(temporary)}]})
    write_private_file(temporary,b'TOKEN=new')
    def interrupt(path,prepared):
        print('moved',flush=True)
        time.sleep(120)
    publication.publish_prepared_file=interrupt
    protected_replace_prepared_file(path,temporary,backup_path=backup)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(target), str(data)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "moved"
    finally:
        process.kill()
        process.wait(timeout=10)
    assert not target.exists()
    preserved = retained_staging_paths(RuntimeStore(data))
    backup = next(path for path in preserved if path.name == "before")
    assert backup.read_bytes() == b"TOKEN=actual-before-crash"
    verify_private_acl(backup)
    with workspace_write_window(target.parent, subtree=True):
        pass


def test_only_registered_staging_is_excluded_from_workspace_capture(
    tmp_path: Path,
) -> None:
    """暂存排除依据已登记意图，用户同名前缀资料保留；参数：根；返回：无。"""
    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    stage = create_private_staging(root)
    backup = stage / "before"
    write_private_file(backup, b"retained sensitive original")
    user = root / ".reins-restore-user-notes"
    user.write_text("user document", encoding="utf-8")
    with store.database.transaction() as batch:
        batch.put(
            "file_restore_operation",
            "pending",
            {"entries": [{"backup_path": str(backup)}]},
        )
    captured = store.capture_workspace(root, lease, workspace_id)
    assert not captured["errors"]
    assert str(backup) not in captured["states"]
    assert str(user) in captured["states"]
    assert {"path": str(stage), "reason": "registered_restore_staging"} in captured[
        "exclusions"
    ]


def test_content_cannot_change_to_private_key_between_inspection_and_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """检查和冻结共用系统固定句柄，竞争写不能把私钥带入普通objects；参数：根和实际写竞争；返回：无。"""
    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    path = root / "ordinary.txt"
    path.write_bytes(b"ordinary original")
    prepare = store.database.prepare_stream

    def competing_write(
        source: object, *, workspace_id: str, cancellation: object = None
    ) -> object:
        """在冻结前尝试真实外部写入；参数：捕获流和工作区；返回：准确冻结引用。"""
        with pytest.raises(PermissionError):
            path.write_bytes(b"-----BEGIN PRIVATE KEY-----\nSYNTHETIC-KEY")
        return prepare(source, workspace_id=workspace_id, cancellation=cancellation)

    monkeypatch.setattr(store.database, "prepare_stream", competing_write)
    state = store.capture_file(path, lease, workspace_id)
    assert store.read_state(state) == b"ordinary original"
    assert path.read_bytes() == b"ordinary original"


def test_protection_import_preserves_win32_gui_startup() -> None:
    """恢复模块先导入后通知仍可启动，避免错载pywintypes；参数：无；返回：无。"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tools.restore_protection; import pywintypes; import win32gui; print('win32-ready')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "win32-ready"


def test_prepared_original_detects_same_size_change_with_restored_mtime(
    tmp_path: Path,
) -> None:
    """锁外核验到提交间固定原件，释放后同长篡改仍检出；参数：根；返回：无。"""
    from runtime.file_records import ContentReference, SourceCorruptionError

    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    path = root / "draft.txt"
    path.write_bytes(b"before")
    state = store.capture_file(path, lease, workspace_id)
    reference = ContentReference.from_mapping(state["content"])
    blob = store.data_root / reference.path
    original = blob.stat()
    with store.database.prepare_reference(reference) as prepared:
        with pytest.raises(PermissionError):
            blob.write_bytes(b"change")
        with store.database.transaction() as batch:
            batch.reference_prepared_content(prepared)
    assert prepared.handle.closed
    blob.write_bytes(b"change")
    os.utime(blob, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(SourceCorruptionError, match="missing or corrupt"):
        with store.database.prepare_reference(reference):
            pass


def test_original_cache_cannot_hide_same_time_same_length_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实际改写即使时间签名相同也必须重核内容；参数：隔离根、碰撞时钟；返回：无。"""
    import tools.file_persistence as publication
    from runtime.file_records import SourceCorruptionError

    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    path = root / "draft.txt"
    path.write_bytes(b"before")
    state = store.capture_file(path, lease, workspace_id)
    blob = store.data_root / state["content"]["path"]
    monkeypatch.setattr(publication, "file_change_time", lambda descriptor: 100)
    assert store.read_state(state) == b"before"
    original = blob.stat()
    blob.write_bytes(b"change")
    os.utime(blob, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(SourceCorruptionError, match="missing or corrupt"):
        store.read_state(state)


def test_prepared_batch_deduplicates_and_releases_on_failure(tmp_path: Path) -> None:
    """同批引用只固定一次且异常释放全部句柄；参数：隔离根；返回：无。"""
    from runtime.file_records import ContentReference, SourceCorruptionError

    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    path = root / "draft.txt"
    path.write_bytes(b"before")
    state = store.capture_file(path, lease, workspace_id)
    reference = ContentReference.from_mapping(state["content"])
    blob = store.data_root / reference.path
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        with store.database.prepare_references([reference, reference]) as originals:
            assert len(originals) == 1
            with store.database.transaction() as batch:
                batch.reference_prepared_content(originals[0])
                raise RuntimeError("simulated commit failure")
    assert originals[0].handle.closed
    blob.write_bytes(b"before")
    with pytest.raises(SourceCorruptionError, match="released"):
        with store.database.transaction() as batch:
            batch.reference_prepared_content(originals[0])


def test_prepared_batch_partial_failure_releases_earlier_handles(
    tmp_path: Path,
) -> None:
    """后续原件坏掉时也释放已核验的前项；参数：隔离根；返回：无。"""
    from runtime.file_records import ContentReference, SourceCorruptionError

    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    path = root / "draft.txt"
    path.write_bytes(b"before")
    state = store.capture_file(path, lease, workspace_id)
    reference = ContentReference.from_mapping(state["content"])
    missing = ContentReference("global/objects/missing.bin", "0" * 64, 1)
    with pytest.raises(SourceCorruptionError, match="missing or corrupt"):
        with store.database.prepare_references([reference, missing]):
            pass
    (store.data_root / reference.path).write_bytes(b"before")


def test_workspace_pins_content_when_change_time_collides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """时间碰撞也不能改写已冻结的捕获内容，结束后释放；参数：隔离根、枚举注入；返回：无。"""
    import runtime.file_snapshots as snapshots
    import tools.file_persistence as publication

    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    target = root / "draft.txt"
    target.write_bytes(b"before")
    inventory = snapshots._inventory
    scans = 0
    monkeypatch.setattr(snapshots, "file_change_time", lambda descriptor: 100)
    monkeypatch.setattr(publication, "file_change_time", lambda descriptor: 100)

    def competing_inventory(*args: object, **kwargs: object) -> object:
        """第二次枚举前尝试同长度替换，实际句柄应阻止改写；参数：范围；返回：原目录清单。"""
        nonlocal scans
        scans += 1
        if scans == 2:
            with pytest.raises(PermissionError):
                target.write_bytes(b"change")
        return inventory(*args, **kwargs)

    monkeypatch.setattr(snapshots, "_inventory", competing_inventory)
    captured = store.capture_workspace(root, lease, workspace_id)
    assert store.read_state(captured["states"][str(target)]) == b"before"
    assert captured["errors"] == [] and target.read_bytes() == b"before"
    target.write_bytes(b"change")
