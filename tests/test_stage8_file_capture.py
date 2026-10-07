"""【文件恢复】【捕获验收】真实文件、子进程及停止未确认边界。

作者：xxx
时间：2026-09-30 21:00:00
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event, Timer
from typing import Any

import pytest

from approval import ApprovalDecision
from runtime.file_snapshots import SnapshotStore
from runtime.cancellation import CancellationToken
from runtime.lease import from_trigger
from runtime.shared_budget import BudgetOwner
from runtime.watchdog import Watchdog
from runtime.workspaces import WorkspaceStore
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import (
    Idempotent,
    PreparedToolExecution,
    ToolDefinition,
    ToolRisk,
)
from tools.types import ToolError, ToolErrorCategory
from tools.workspace_coordination import WorkspaceBusyError, workspace_write_window


@pytest.fixture
def capture_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """构造隔离工作区与持久会话；传参：临时根；返回：实际服务依赖。"""
    root, data = tmp_path / "work", tmp_path / "data"
    root.mkdir()
    workspace = WorkspaceStore(data).bind_session("session-capture", root)
    lease = from_trigger(
        "user",
        task_id="task-capture",
        capabilities={
            "fs": {
                "project_root": str(root),
                "read": [str(root)],
                "write": [str(root)],
                "default_sensitive_policy": "redacted",
            }
        },
    )
    registry = build_tool_registry(repo_root=root, data_root=data)
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda request: ApprovalDecision.ONCE,
    )
    yield {
        "root": root,
        "data": data,
        "workspace": workspace,
        "lease": lease,
        "registry": registry,
        "store": SnapshotStore(data),
    }
    registry.close()


def execute(
    env: dict[str, Any], name: str, args: dict[str, object], **options: Any
) -> object:
    """经真实registry和watchdog派发；传参：环境、工具、参数与超时选项；返回：实际工具回执。"""
    watchdog = Watchdog(
        env["lease"],
        data_root=env["data"],
        tool_timeout_seconds=options.get("timeout", 20),
        budget_owner=BudgetOwner("session-capture", "run-capture", env["lease"]),
    )
    prepared = env["registry"].prepare_tool_execution(
        name,
        args,
        env["lease"],
        watchdog=watchdog,
        operation_id=options.get("operation_id", "op-capture"),
        on_late=options.get("on_late"),
        cancellation=options.get("cancellation"),
    )
    assert isinstance(prepared, PreparedToolExecution), prepared
    prepared.file_session_id = "session-capture"
    prepared.capture_identity = {
        "session_id": "session-capture",
        "run_id": "run-capture",
        "input_id": "input-capture",
    }
    return env["registry"].execute_prepared_tool(prepared)


def points(env: dict[str, Any]) -> list[dict[str, Any]]:
    """读取当前工作区原件；传参：环境；返回：已提交恢复点。"""
    return env["store"].list_points(env["workspace"].workspace_id)


def test_exact_write_freezes_user_edit_before_replacement(
    capture_env: dict[str, Any],
) -> None:
    """未提交用户稿真实冻结，重启可读且新增有明确不存在状态；传参：环境；返回：无。"""
    env = capture_env
    target = env["root"] / "draft.txt"
    original = b"user draft\r\n"
    target.write_bytes(original)
    result = execute(
        env,
        "file_write",
        {
            "path": str(target),
            "content": "new text",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    assert isinstance(result, dict) and result["meta"]["restore_point_ids"]
    point = points(env)[0]
    assert point["status"] == "complete" and point["attribution"] == "exact"
    assert (
        point["input_id"] == "input-capture" and point["operation_id"] == "op-capture"
    )
    restarted = SnapshotStore(env["data"])
    assert restarted.read_state(point["entries"][0]["before"]) == original
    assert restarted.read_state(point["entries"][0]["after"]) == b"new text"
    result = execute(
        env, "file_write", {"path": str(env["root"] / "new.txt"), "content": "created"}
    )
    assert isinstance(result, dict)
    assert points(env)[0]["entries"][0]["before"]["kind"] == "missing"


def test_capture_failure_prevents_write_and_after_failure_preserves_result(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """前态IO失败不写入，后态登记失败仍保持真实成功结果；传参：环境与故障注入；返回：无。"""
    env = capture_env
    target = env["root"] / "draft.txt"
    target.write_bytes(b"old")
    arguments = {
        "path": str(target),
        "content": "updated",
        "expected_sha256": hashlib.sha256(b"old").hexdigest(),
    }
    capture_file = SnapshotStore.capture_file

    def fail_read(
        self: SnapshotStore,
        path: Path,
        lease: Any,
        workspace_id: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """模拟真实文件IO失败；传参：捕获参数；返回：不返回。"""
        raise OSError("disk read failed")

    monkeypatch.setattr(SnapshotStore, "capture_file", fail_read)
    failed = execute(env, "file_write", arguments)
    assert isinstance(failed, ToolError) and target.read_bytes() == b"old"
    monkeypatch.setattr(SnapshotStore, "capture_file", capture_file)
    publish = SnapshotStore.publish_point

    def fail_after(
        self: SnapshotStore,
        point: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> None:
        """只中断后态登记；传参：点；返回：无或真实登记错误。"""
        if point["status"] == "complete":
            raise OSError("after-state commit failed")
        publish(self, point)

    monkeypatch.setattr(SnapshotStore, "publish_point", fail_after)
    succeeded = execute(env, "file_write", arguments)
    assert isinstance(succeeded, dict) and target.read_bytes() == b"updated"
    assert succeeded["meta"]["restore_record_errors"]
    assert env["store"].read_state(points(env)[0]["entries"][0]["before"]) == b"old"


def test_unknown_code_captures_binary_rename_delete_and_ignored_user_files(
    capture_env: dict[str, Any],
) -> None:
    """真实Python进程的前后差异覆盖用户资料和产物；传参：环境；返回：无。"""
    env = capture_env
    root = env["root"]
    (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (root / "ignored.txt").write_bytes(b"ignored user file")
    (root / "removed.bin").write_bytes(b"\x00\xffbinary")
    (root / "renamed.txt").write_bytes(b"rename me")
    artifact = root / ".reins" / "workspace" / "task-capture" / "user-result.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"controlled result")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "cache.js").write_bytes(b"dependency")
    code = f"from pathlib import Path\np=Path({str(root)!r})\n" + "\n".join(
        [
            "(p/'ignored.txt').write_bytes(b'changed')",
            "(p/'removed.bin').unlink()",
            "(p/'renamed.txt').rename(p/'renamed-new.txt')",
            "(p/'new.bin').write_bytes(bytes([0,255,1]))",
            "(p/'.reins/workspace/task-capture/user-result.txt').write_bytes(b'new result')",
        ]
    )
    result = execute(env, "code_execution_tool", {"code": code})
    assert isinstance(result, dict), result
    point = points(env)[0]
    entries = {Path(row["path"]).name: row for row in point["entries"]}
    assert point["attribution"] == "observed" and point["status"] == "complete"
    assert entries["removed.bin"]["after"]["kind"] == "missing"
    assert entries["new.bin"]["before"]["kind"] == "missing"
    assert (
        env["store"].read_state(entries["removed.bin"]["before"]) == b"\x00\xffbinary"
    )
    assert (
        env["store"].read_state(entries["user-result.txt"]["before"])
        == b"controlled result"
    )
    assert entries["renamed.txt"]["after"]["kind"] == "missing"
    assert entries["renamed-new.txt"]["before"]["kind"] == "missing"
    assert "cache.js" not in entries and any(
        row["path"].endswith("node_modules") for row in point["exclusions"]
    )


def test_sensitive_originals_are_encrypted_and_private_keys_excluded(
    capture_env: dict[str, Any],
) -> None:
    """恢复敏感配置密文保存，不读取私钥或把敏感正文放普通对象；传参：环境；返回：无。"""
    env = capture_env
    secret = b"TOKEN=stage8-capture-private-value\r\nPORT=3000\r\n"
    target = env["root"] / ".env"
    target.write_bytes(secret)
    key = env["root"] / "private.key"
    key.write_bytes(b"-----BEGIN PRIVATE KEY-----\nnot-a-real-key")
    state = env["store"].capture_file(
        target, env["lease"], env["workspace"].workspace_id
    )
    assert (
        state["sensitive"] and state["content"] is None and state["protected_content"]
    )
    assert "sha256" not in state["metadata"]
    assert env["store"].read_state(state) == secret
    assert env["store"].same_state(
        state,
        env["store"].capture_file(target, env["lease"], env["workspace"].workspace_id),
        identity=True,
    )
    snapshot = env["store"].capture_workspace(
        env["root"], env["lease"], env["workspace"].workspace_id
    )
    assert snapshot["states"][str(key)]["restorable"] is False
    assert any(row["path"] == str(key) for row in snapshot["exclusions"])
    for path in env["data"].rglob("*"):
        if path.is_file():
            assert b"stage8-capture-private-value" not in path.read_bytes()


def test_unconfirmed_stop_keeps_write_window_until_late_process_exit(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """超时不能提前释放仍写文件的进程窗口；传参：环境与停止失败注入；返回：无。"""
    env = capture_env
    finished, late = Event(), []

    def reject_stop(process: Any, *, tree: Any = None) -> None:
        """暴露停止失败而让真实子进程继续；传参：进程；返回：不返回。"""
        raise OSError("test cannot confirm stop")

    def receive(result: object) -> None:
        """记录实际退出后的结果；传参：迟到结果；返回：无。"""
        late.append(result)
        finished.set()

    monkeypatch.setattr("tools.exec_channel._kill_process_tree", reject_stop)
    path = env["root"] / "late.txt"
    code = f"from pathlib import Path\nimport time\np=Path({str(path)!r})\np.write_text('started')\ntime.sleep(3)\np.write_text('late')"
    result = execute(
        env, "code_execution_tool", {"code": code}, timeout=0.4, on_late=receive
    )
    assert (
        isinstance(result, ToolError) and result.details["execution_state"] == "unknown"
    )
    assert points(env)[0]["status"] == "before_saved"
    with pytest.raises(WorkspaceBusyError):
        with workspace_write_window(path, owner="competing-restore"):
            pytest.fail("active process released the observation window")
    assert finished.wait(8)
    assert path.read_text() == "late" and points(env)[0]["status"] == "complete"
    with workspace_write_window(path, owner="after-exit"):
        assert late


def test_large_binary_deduplicates_and_missing_original_never_uses_current_file(
    capture_env: dict[str, Any],
) -> None:
    """完整大文件不截断，重用对象且损坏无法由当前文件冒充；传参：环境；返回：无。"""
    env = capture_env
    path = env["root"] / "large.bin"
    content = bytes(range(256)) * 16384
    path.write_bytes(content)
    first = env["store"].capture_file(path, env["lease"], env["workspace"].workspace_id)
    second = env["store"].capture_file(
        path, env["lease"], env["workspace"].workspace_id
    )
    assert first["content"] == second["content"] and first["content"]["size"] == len(
        content
    )
    assert b"".join(env["store"].iter_state(first)) == content
    (env["data"] / first["content"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        env["store"].validate_state(first)


def test_exact_capture_does_not_scan_workspace(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """单文件写仅冻结准确目标，普通只读请求不扫描；传参：环境与观察器；返回：无。"""

    def forbidden_scan(*args: Any, **kwargs: Any) -> None:
        """拒绝无关全量扫描；传参：扫描参数；返回：不返回。"""
        pytest.fail("an exact or readonly tool scanned the workspace")

    monkeypatch.setattr(SnapshotStore, "capture_workspace", forbidden_scan)
    assert isinstance(
        execute(capture_env, "file_write", {"path": "new.txt", "content": "new"}), dict
    )
    assert isinstance(execute(capture_env, "file_read", {"path": "new.txt"}), dict)


def test_each_real_retry_retains_its_own_before_and_after(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实重试逐次捕获并归属原操作；参数：环境与退避替换；返回：无。"""
    env = capture_env
    path = env["root"] / "retry.txt"
    path.write_text("initial")
    calls = []

    def retrying_write(arguments: dict[str, object]) -> object:
        """模拟可重试后端的实际写入；参数：宿主参数；返回：第一次错误，第二次真实完成。"""
        calls.append(arguments)
        path.write_text("intermediate" if len(calls) == 1 else "finished")
        if len(calls) == 1:
            return ToolError(
                ToolErrorCategory.TRANSPORT, "transient transport loss", retryable=True
            )
        return {"content": "finished"}

    env["registry"].register(
        ToolDefinition(
            name="retry_writer",
            description="test writer",
            parameters={},
            toolset="file",
            risk_level=ToolRisk.SAFE,
            readonly=False,
            target_scope_rule="logical_scope",
            source="builtin",
            idempotent=Idempotent.YES,
            executor=retrying_write,
        )
    )
    monkeypatch.setattr("tools.tool_registry._backoff_sleep", lambda seconds: None)
    result = execute(env, "retry_writer", {})
    assert isinstance(result, dict) and len(result["meta"]["restore_point_ids"]) == 2
    saved = points(env)
    assert len(saved) == 2 and len({point["execution_id"] for point in saved}) == 2
    assert {point["operation_id"] for point in saved} == {"op-capture"}
    versions = {
        (
            env["store"].read_state(point["entries"][0]["before"]),
            env["store"].read_state(point["entries"][0]["after"]),
        )
        for point in saved
    }
    assert versions == {(b"initial", b"intermediate"), (b"intermediate", b"finished")}


def test_final_replacement_preserves_external_edit(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """最终发布竞争保存真正被替换的用户稿；参数：环境和外部竞争注入；返回：无。"""
    from tools.file_persistence import replace_prepared_file

    env = capture_env
    path = env["root"] / "draft.txt"
    path.write_bytes(b"initial")

    def competing_replace(target: Path, temporary: Path, *, backup_path: Path) -> None:
        """模拟外部编辑器在最后发布前写入；参数：发布路径；返回：无。"""
        target.write_bytes(b"last user edit")
        replace_prepared_file(target, temporary, backup_path=backup_path)

    monkeypatch.setattr(
        "tools.file_persistence.replace_prepared_file", competing_replace
    )
    result = execute(
        env,
        "file_write",
        {
            "path": str(path),
            "content": "agent edit",
            "expected_sha256": hashlib.sha256(b"initial").hexdigest(),
        },
    )
    assert isinstance(result, ToolError)
    assert result.details["execution_state"] == "unknown"
    saved = points(env)[0]
    assert saved["status"] == "incomplete" and saved["attribution"] == "observed"
    assert env["store"].read_state(saved["entries"][0]["before"]) == b"initial"
    assert (
        env["store"].read_state(saved["entries"][0]["displaced_before"])
        == b"last user edit"
    )
    assert path.read_bytes() == b"agent edit"


def test_after_inventory_error_never_invents_deleted_children(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """后态目录无法读取时子文件状态未知，不能伪造删除；参数：环境和IO注入；返回：无。"""
    env = capture_env
    path = env["root"] / "kept.txt"
    path.write_bytes(b"unchanged")
    capture_workspace = SnapshotStore.capture_workspace
    calls = []

    def fail_after(
        self: SnapshotStore,
        root: Path,
        lease: Any,
        workspace_id: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """第二次目录清单返回真实覆盖错误；参数：扫描依赖；返回：完整前态或不可读后态。"""
        calls.append(root)
        captured = capture_workspace(self, root, lease, workspace_id)
        if len(calls) == 2:
            return {
                **captured,
                "states": {},
                "errors": [{"path": str(root), "error": "directory unreadable"}],
            }
        return captured

    monkeypatch.setattr(SnapshotStore, "capture_workspace", fail_after)
    result = execute(env, "code_execution_tool", {"code": "print('executed')"})
    assert isinstance(result, dict)
    saved = points(env)[0]
    assert saved["status"] == "incomplete" and saved["entries"][0]["after"] is None
    assert path.read_bytes() == b"unchanged"


def test_detached_descendant_keeps_window_after_root_exits(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """根进程退出后仍核验脱离标准输出的后代；参数：环境和停止失败注入；返回：无。"""
    env = capture_env
    finished = Event()
    path = env["root"] / "descendant.txt"
    child = f"from pathlib import Path; import time; time.sleep(2); Path({str(path)!r}).write_text('child done')"
    code = f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', {child!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"

    def reject_stop(process: Any, *, tree: Any = None) -> None:
        """保留真实后代进程用于停止未确认验收；参数：进程和Job；返回：不返回。"""
        raise OSError("cannot confirm descendant stopped")

    monkeypatch.setattr("tools.exec_channel._kill_process_tree", reject_stop)
    result = execute(
        env,
        "code_execution_tool",
        {"code": code},
        timeout=0.4,
        on_late=lambda value: finished.set(),
    )
    assert (
        isinstance(result, ToolError) and result.details["execution_state"] == "unknown"
    )
    with pytest.raises(WorkspaceBusyError):
        with workspace_write_window(path, owner="competing-restore"):
            pytest.fail("live descendant lost its workspace window")
    assert finished.wait(8) and path.read_text() == "child done"
    assert points(env)[0]["status"] == "complete"


def test_sensitive_patch_captures_real_bytes_without_plaintext_staging(
    capture_env: dict[str, Any],
) -> None:
    """脱敏读写真实接线保留原始字节与受保护前后态；参数：环境；返回：无。"""
    env = capture_env
    path = env["root"] / ".env"
    original = b"TOKEN=private-stage8-original\r\nPORT=3000\r\n"
    path.write_bytes(original)
    view = execute(env, "file_read", {"path": str(path)})
    assert isinstance(view, dict) and "private-stage8-original" not in str(view)
    result = execute(
        env,
        "file_patch",
        {
            "path": str(path),
            "view_id": view["meta"]["view_id"],
            "expected_sha256": view["meta"]["content_sha256"],
            "old_text": "PORT=3000",
            "new_text": "PORT=4000",
        },
    )
    assert isinstance(result, dict), result
    point = points(env)[0]
    assert point["status"] == "complete" and not Path(point["staging_path"]).exists()
    assert env["store"].read_state(point["entries"][0]["before"]) == original
    assert env["store"].read_state(point["entries"][0]["after"]) == original.replace(
        b"3000", b"4000"
    )
    for stored in env["data"].rglob("*"):
        if stored.is_file():
            assert b"private-stage8-original" not in stored.read_bytes()


def test_user_cancellation_keeps_window_until_actual_exit(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """用户取消和超时遵守相同实际停止边界；参数：环境、停止失败注入；返回：无。"""
    env = capture_env
    token, finished = CancellationToken(), Event()
    path = env["root"] / "cancelled.txt"
    code = f"from pathlib import Path; import time; time.sleep(3); Path({str(path)!r}).write_text('actual completion')"

    def reject_stop(process: Any, *, tree: Any = None) -> None:
        """模拟停止请求无法完成；参数：真实进程树；返回：不返回。"""
        raise OSError("cannot confirm cancelled process stopped")

    monkeypatch.setattr("tools.exec_channel._kill_process_tree", reject_stop)
    timer = Timer(0.5, token.cancel)
    timer.start()
    try:
        result = execute(
            env,
            "code_execution_tool",
            {"code": code},
            cancellation=token,
            on_late=lambda value: finished.set(),
        )
        assert (
            isinstance(result, ToolError)
            and result.category == ToolErrorCategory.CANCELLED
        )
        assert result.details["execution_state"] == "unknown"
        with pytest.raises(WorkspaceBusyError):
            with workspace_write_window(path, owner="restore-during-cancel"):
                pytest.fail("cancelled but live backend released the window")
        assert finished.wait(8) and path.read_text() == "actual completion"
        assert points(env)[0]["status"] == "complete"
    finally:
        timer.cancel()


@pytest.mark.parametrize("effect", ["file", "workspace"])
def test_cancellation_during_before_capture_never_starts_effect(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, effect: str
) -> None:
    """前态捕获结束前取消不能在随后写文件或启动命令；参数：环境、注入器、效果类型；返回：无。"""
    env, token = capture_env, CancellationToken()
    path = env["root"] / "cancel-before.txt"
    method = "capture_file" if effect == "file" else "capture_workspace"
    capture = getattr(SnapshotStore, method)

    def cancel_capture(
        self: SnapshotStore,
        target: Path,
        lease: Any,
        workspace_id: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """在真实前态读取后、返回实际派发前取消；参数：原捕获依赖；返回：真实捕获结果。"""
        result = capture(self, target, lease, workspace_id, cancellation=cancellation)
        token.cancel()
        return result

    monkeypatch.setattr(SnapshotStore, method, cancel_capture)
    name = "file_write" if effect == "file" else "code_execution_tool"
    arguments = (
        {"path": str(path), "content": "must not publish"}
        if effect == "file"
        else {
            "code": f"from pathlib import Path; Path({str(path)!r}).write_text('must not execute')"
        }
    )
    result = execute(env, name, arguments, cancellation=token)
    assert (
        isinstance(result, ToolError) and result.category == ToolErrorCategory.CANCELLED
    )
    assert result.details["execution_state"] == "not_started" and not path.exists()
    with workspace_write_window(path, owner="after-cancel-before"):
        assert not path.exists()


def test_staging_cleanup_failure_preserves_successful_write(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """发布成功后的暂存清理失败只影响恢复记录；参数：环境和目录清理注入；返回：无。"""
    env = capture_env
    path = env["root"] / "cleanup.txt"
    path.write_bytes(b"original")
    remove_directory = Path.rmdir

    def reject_staging_cleanup(directory: Path) -> None:
        """仅阻止目标卷暂存目录回收；参数：目录；返回：无，目标文件已成功发布。"""
        if directory.parent == env["root"] and directory.name.startswith(
            ".reins-restore-"
        ):
            raise PermissionError("staging cleanup denied")
        remove_directory(directory)

    monkeypatch.setattr(Path, "rmdir", reject_staging_cleanup)
    result = execute(
        env,
        "file_write",
        {
            "path": str(path),
            "content": "published",
            "expected_sha256": hashlib.sha256(b"original").hexdigest(),
        },
    )
    assert isinstance(result, dict) and path.read_bytes() == b"published"
    assert any(
        "staging cleanup denied" in error
        for error in result["meta"]["restore_record_errors"]
    )
    assert points(env)[0]["status"] == "incomplete"
    assert (
        env["store"].read_state(points(env)[0]["entries"][0]["before"]) == b"original"
    )


def test_cancellation_before_resuming_process_never_runs_user_code(
    capture_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """挂起进程绑定Job期间取消不得恢复用户代码；参数：环境和取消注入；返回：无。"""
    from tools.exec_process_tree import ProcessTree

    env, token = capture_env, CancellationToken()
    path = env["root"] / "must-not-start.txt"
    attach = ProcessTree.attach
    completed, late = Event(), []

    def cancel_after_attachment(tree: ProcessTree, process_handle: int) -> None:
        """将真实挂起进程纳入Job后触发取消；参数：Job和进程句柄；返回：无。"""
        attach(tree, process_handle)
        token.cancel()

    monkeypatch.setattr(ProcessTree, "attach", cancel_after_attachment)

    def receive(result: object) -> None:
        """记录后态捕获完成后的真实取消结果；参数：迟到回执；返回：无。"""
        late.append(result)
        completed.set()

    result = execute(
        env,
        "code_execution_tool",
        {
            "code": f"from pathlib import Path; Path({str(path)!r}).write_text('must not run')"
        },
        cancellation=token,
        on_late=receive,
    )
    if isinstance(result, ToolError) and result.details["execution_state"] == "unknown":
        assert completed.wait(8)
        result = late[0]
    assert (
        isinstance(result, ToolError) and result.category == ToolErrorCategory.CANCELLED
    )
    assert result.details["execution_state"] == "not_started" and not path.exists()
    assert token.backend_evidence()["process_tree_stopped"] is True
    with workspace_write_window(path, owner="after-suspended-cancel"):
        assert not path.exists()
