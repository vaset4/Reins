"""【审查修复】【捕获完整性】复用只在真实句柄有效期间成立。

作者：xxx
时间：2026-10-03 11:00:00
"""

from pathlib import Path
from time import sleep

import pytest

import path_security
from runtime.cancellation import ExecutionCancelled
from runtime.file_content import verified_content_scope
from runtime.file_records import SourceCorruptionError
from runtime.file_snapshots import SnapshotStore
from runtime.lease import from_trigger
from runtime.persistence import RuntimeStore
from tests.test_stage8_foundation import snapshot_environment
from tests.test_stage8_file_capture import capture_env as _capture_env, execute

capture_env = _capture_env


@pytest.mark.parametrize("redacted", [False, True])
@pytest.mark.parametrize(
    "filename", ["draft.txt", ".env", "secret.pem", "denied.txt", "nested/private.txt"]
)
def test_compiled_read_boundary_preserves_permissions(
    tmp_path: Path, redacted: bool, filename: str
) -> None:
    """批量读取仍保持普通和脱敏路径的原权限；参数：根/模式/目标；返回：无。"""
    fs = {
        "project_root": str(tmp_path),
        "read": [str(tmp_path)],
        "deny_read": ["denied.txt", "*/private.txt"],
    }
    if redacted:
        fs["default_sensitive_policy"] = "redacted"
    lease = from_trigger("user", task_id="permissions", capabilities={"fs": fs})
    path = tmp_path / filename
    resolved, sensitive, decision = path_security.ReadBoundary.from_lease(
        lease
    ).inspect(path)
    assert resolved == path.resolve()
    assert sensitive == path_security.uses_redacted_files(path, lease)
    assert decision == path_security.check_read(path, lease, filtered=sensitive)


def test_original_reuse_releases_handles_and_rechecks_next_scope(
    tmp_path: Path, monkeypatch
) -> None:
    """同次执行固定原件，结束后同时间同长度损坏仍拒绝；参数：根/碰撞时钟；返回：无。"""
    store = RuntimeStore(tmp_path)
    reference = store.prepare_content(b"original")
    path = tmp_path / reference.path
    monkeypatch.setattr(
        "tools.file_persistence.file_change_time", lambda descriptor: 100
    )
    with verified_content_scope(store.data_root):
        assert store.read_content(reference) == b"original"
        assert RuntimeStore(tmp_path).read_content(reference) == b"original"
        with pytest.raises(PermissionError):
            path.write_bytes(b"changed!")
    path.write_bytes(b"changed!")
    with verified_content_scope(store.data_root), pytest.raises(SourceCorruptionError):
        store.read_content(reference)


def test_capture_time_does_not_consume_subprocess_deadline(
    capture_env, monkeypatch
) -> None:
    """前后捕获各自在时限内时，合计耗时不把真实成功命令改成超时；参数：环境/慢捕获；返回：无。"""
    capture = SnapshotStore.capture_workspace

    def slow_capture(self, *args, **kwargs):
        """模拟有界文件IO延迟并执行真实捕获；参数：原参数；返回：真实清单。"""
        sleep(0.65)
        return capture(self, *args, **kwargs)

    monkeypatch.setattr(SnapshotStore, "capture_workspace", slow_capture)
    result = execute(
        capture_env,
        "terminal_tool",
        {"command": "echo completed", "cwd": str(capture_env["root"])},
        timeout=1,
    )
    assert isinstance(result, dict) and result["exit_code"] == 0, result
    assert result["stdout"].strip() == "completed"


@pytest.mark.parametrize("cancel", [False, True])
def test_capture_holds_verified_working_bytes_then_releases(
    tmp_path: Path, monkeypatch, cancel: bool
) -> None:
    """全清单核对期间原字节不能被替换，成功或取消都释放；参数：根/注入器/取消；返回：无。"""
    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    path = root / "draft.txt"
    path.write_bytes(b"original")
    capture_file = SnapshotStore.capture_file

    def observe(self, target, active_lease, identity, **options):
        """在单文件冻结后尝试外部写入；参数：原捕获参数；返回：真实状态或取消。"""
        state = capture_file(self, target, active_lease, identity, **options)
        if target == path:
            with pytest.raises(PermissionError):
                path.write_bytes(b"changed!")
            if cancel:
                raise ExecutionCancelled("cancel capture")
        return state

    monkeypatch.setattr(SnapshotStore, "capture_file", observe)
    if cancel:
        with pytest.raises(ExecutionCancelled):
            store.capture_workspace(root, lease, workspace_id)
    else:
        capture = store.capture_workspace(root, lease, workspace_id)
        assert capture["errors"] == []
        assert store.read_state(capture["states"][str(path)]) == b"original"
    path.write_bytes(b"changed!")
