"""从实际文件执行边界验证授权范围和交互失效。

作者：xxx
时间：2026-09-14 15:00:00
"""

from __future__ import annotations

import hashlib
from contextlib import closing
from dataclasses import replace
from threading import Event, Thread

import pytest

from approval import ApprovalDecision, ApprovalRequest, request_approval
from approval.channel import ApprovalChannel
from runtime.cancellation import CancellationToken
from runtime.lease import from_trigger
from runtime.watchdog import Watchdog
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.types import ToolError


def test_task_grant_reuses_exact_file_and_action_after_content_change(
    tmp_path, monkeypatch
):
    """修改内容可复用授权，邻接文件、其他动作和任务需新批准；传参：目录/替换器；返回：无。"""
    project, data = tmp_path / "project", tmp_path / "data"
    project.mkdir()
    with closing(TaskStore(data)) as store:
        task = store.create_task("编辑文件")
        other = store.create_task("另一事项")
    monkeypatch.setattr("approval._config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.chdir(tmp_path)
    seen = []

    def backend(request):
        """首文件批准任务范围，其他申请拒绝；传参：明确请求；返回：用户决定。"""
        seen.append(request)
        return ApprovalDecision.TASK if len(seen) == 1 else ApprovalDecision.DENY

    monkeypatch.setattr("approval._backend", backend)
    lease = from_trigger(
        "user",
        task_id=task.task_id,
        capabilities={
            "fs": {"project_root": str(project), "read": [str(project)], "write": []}
        },
    )
    registry = build_tool_registry(repo_root=project, data_root=data)
    watchdog = Watchdog(lease, data_root=data)
    result = registry.execute_tool(
        "file_write", {"path": "a.txt", "content": "one"}, lease, watchdog=watchdog
    )
    assert not isinstance(result, ToolError)
    assert seen[0].resource.target == str((project / "a.txt").resolve()).casefold()
    digest = hashlib.sha256(b"one").hexdigest()
    result = registry.execute_tool(
        "file_write",
        {"path": "a.txt", "content": "two", "expected_sha256": digest},
        lease,
        watchdog=watchdog,
    )
    assert not isinstance(result, ToolError)
    assert len(seen) == 1
    assert (project / "a.txt").read_text() == "two"
    assert isinstance(
        registry.execute_tool(
            "file_write",
            {"path": "b.txt", "content": "blocked"},
            lease,
            watchdog=watchdog,
        ),
        ToolError,
    )
    patch = {
        "path": "a.txt",
        "old_text": "two",
        "new_text": "blocked",
        "expected_sha256": hashlib.sha256(b"two").hexdigest(),
    }
    assert isinstance(
        registry.execute_tool("file_patch", patch, lease, watchdog=watchdog), ToolError
    )
    assert isinstance(
        registry.execute_tool(
            "file_write",
            {"path": "a.txt", "content": "blocked"},
            replace(lease, task_id=other.task_id),
            watchdog=watchdog,
        ),
        ToolError,
    )
    assert not (project / "b.txt").exists()
    assert (project / "a.txt").read_text() == "two"
    assert len(seen) == 4


def test_correction_before_approval_commit_does_not_save_stale_grant(
    tmp_path, monkeypatch
):
    """批准回传之前到达的纠正会使原申请失效；传参：目录/替换器；返回：无。"""
    with closing(TaskStore(tmp_path)) as store:
        task = store.create_task("等待授权")
    token = CancellationToken()

    def backend(_request):
        """制造审批提交边界的停止；传参：请求；返回：已过时的决定。"""
        token.cancel()
        return ApprovalDecision.TASK

    monkeypatch.setattr("approval._backend", backend)
    monkeypatch.setattr("approval._config_path", lambda: tmp_path / "config.yaml")
    request = ApprovalRequest(
        "action",
        {},
        "confirm",
        from_trigger("user", task_id=task.task_id),
        tmp_path,
        "授权",
        cancellation=token,
    )
    assert request_approval(request) == ApprovalDecision.CANCELLED
    with closing(TaskStore(tmp_path)) as store:
        assert store.require_task(task.task_id).grants == []


def test_old_approval_number_cannot_approve_new_request(tmp_path):
    """旧决定编号在下一待批动作中失效；传参：临时根；返回：无。"""
    presented, seen, decisions = Event(), [], []

    def present(identity, _request):
        """记录用户看到的请求编号；传参：编号/请求；返回：无。"""
        seen.append(identity)
        presented.set()

    channel = ApprovalChannel(present)
    request = ApprovalRequest(
        "action", {}, "confirm", from_trigger("user", task_id="scope"), tmp_path, "授权"
    )

    def ask():
        """等待当前请求的明确决定；传参：无；返回：无。"""
        decisions.append(channel.request(request))

    first = Thread(target=ask, daemon=True)
    first.start()
    assert presented.wait(2)
    channel.interrupt()
    first.join(2)
    presented.clear()
    second = Thread(target=ask, daemon=True)
    second.start()
    assert presented.wait(2)
    with pytest.raises(ValueError, match="编号"):
        channel.answer(f"/approve {seen[0]} task")
    channel.answer(f"/approve {seen[1]} deny")
    second.join(2)
    assert decisions == [ApprovalDecision.CANCELLED, ApprovalDecision.DENY]
