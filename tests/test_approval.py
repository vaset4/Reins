from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalUnavailable,
    register_approval_backend,
    request_approval,
)
from approval.cli import cli_request_approval
from runtime.persistence import RuntimeStore
from runtime.lease import from_trigger
from tasks.store import TaskStore


def test_once_grant_appends_task_record(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path)
    req = _request()
    _seed_task(req.data_root)
    register_approval_backend(lambda _req: ApprovalDecision.ONCE)

    try:
        assert request_approval(req) is ApprovalDecision.ONCE
    finally:
        register_approval_backend(None)

    data = TaskStore(req.data_root).load_task_payload("task-1")
    assert data["grants"][-1]["scope"] == "once"
    assert data["grants"][-1]["tool"] == "file_write"


def test_task_grant_updates_task_record_without_dropping_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path)
    _seed_task(tmp_path / ".reins" / "data")
    data_root = tmp_path / ".reins" / "data"
    original = TaskStore(data_root).load_task_payload("task-1")
    with RuntimeStore(data_root).transaction() as source:
        source.put(
            "task",
            "task-1",
            {**original, "goal": "keep me", "tags": ["a"], "custom": "keep unknown"},
        )
    register_approval_backend(lambda _req: ApprovalDecision.TASK)

    try:
        assert request_approval(_request()) is ApprovalDecision.TASK
    finally:
        register_approval_backend(None)

    data = TaskStore(data_root).load_task_payload("task-1")
    assert data["goal"] == "keep me"
    assert data["tags"] == ["a"]
    assert data["custom"] == "keep unknown"
    assert data["revision"] == original["revision"] + 1
    assert data["grants"][-1]["scope"] == "task"


def test_permanent_grant_updates_user_config(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path)
    config = tmp_path / ".reins" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("trace_level: basic\n", encoding="utf-8")
    register_approval_backend(lambda _req: ApprovalDecision.PERMANENT)

    try:
        assert request_approval(_request()) is ApprovalDecision.PERMANENT
    finally:
        register_approval_backend(None)

    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert data["trace_level"] == "basic"
    assert data["permanent_grants"][-1]["scope"] == "permanent"


def test_existing_task_grant_returns_once_without_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path)
    data_root = tmp_path / ".reins" / "data"
    _seed_task(data_root)
    TaskStore(data_root).append_grant(
        "task-1",
        {
            "tool": "file_write",
            "args": {"path": "out.txt"},
            "scope": "task",
            "ts": "2026-05-05T00:00:00+00:00",
        },
    )
    calls = 0

    def backend(_req: ApprovalRequest) -> ApprovalDecision:
        nonlocal calls
        calls += 1
        return ApprovalDecision.DENY

    register_approval_backend(backend)
    try:
        assert request_approval(_request()) is ApprovalDecision.ONCE
    finally:
        register_approval_backend(None)
    assert calls == 0


def test_existing_permanent_grant_returns_once_without_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path)
    config = tmp_path / ".reins" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "permanent_grants": [
                    {
                        "tool": "file_write",
                        "args": {"path": "out.txt"},
                        "scope": "permanent",
                        "ts": "2026-05-05T00:00:00+00:00",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    calls = 0

    def backend(_req: ApprovalRequest) -> ApprovalDecision:
        nonlocal calls
        calls += 1
        return ApprovalDecision.DENY

    register_approval_backend(backend)
    try:
        assert request_approval(_request()) is ApprovalDecision.ONCE
    finally:
        register_approval_backend(None)
    assert calls == 0


@pytest.mark.parametrize(
    "decision",
    [
        ApprovalDecision.ONCE,
        ApprovalDecision.TASK,
        ApprovalDecision.PERMANENT,
        ApprovalDecision.DENY,
    ],
)
def test_backend_decisions_are_returned(
    tmp_path: Path,
    monkeypatch,
    decision: ApprovalDecision,
) -> None:
    _set_home(monkeypatch, tmp_path)
    _seed_task(tmp_path / ".reins" / "data")
    register_approval_backend(lambda _req: decision)
    try:
        assert request_approval(_request()) is decision
    finally:
        register_approval_backend(None)


def test_backend_exception_is_distinct_from_user_denial(
    tmp_path: Path, monkeypatch
) -> None:
    _set_home(monkeypatch, tmp_path)

    def backend(_req: ApprovalRequest) -> ApprovalDecision:
        raise RuntimeError("backend down")

    register_approval_backend(backend)
    try:
        with pytest.raises(ApprovalUnavailable, match="backend down"):
            request_approval(_request())
    finally:
        register_approval_backend(None)


def test_tui_fallback_parses_menu_choice(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "3")

    assert cli_request_approval(_request()) is ApprovalDecision.PERMANENT


def _request(data_root: Path | None = None) -> ApprovalRequest:
    return ApprovalRequest(
        tool="file_write",
        args={"path": "out.txt"},
        risk="confirm",
        lease=from_trigger("user", task_id="task-1"),
        data_root=data_root or (Path.home() / ".reins" / "data"),
        message="write file",
    )


def _set_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


def _seed_task(data_root: Path) -> None:
    """准备授权归属的真实任务；传参：隔离数据根；返回：无。"""
    store = TaskStore(data_root)
    store.create_task("待授权工作", task_id="task-1")
    store.close()


def test_task_grant_isolated_to_data_root_and_does_not_touch_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    data_root = tmp_path / "isolated-data"
    other_root = tmp_path / "other-data"
    _set_home(monkeypatch, home)
    _seed_task(data_root)
    home_before = list(home.rglob("*")) if home.exists() else []
    register_approval_backend(lambda _req: ApprovalDecision.TASK)

    try:
        assert request_approval(_request(data_root)) is ApprovalDecision.TASK
    finally:
        register_approval_backend(None)

    task = TaskStore(data_root).load_task("task-1")
    assert task is not None and task.grants[-1]["scope"] == "task"
    assert not (home / ".reins" / "data" / "tasks" / "task-1").exists()

    register_approval_backend(lambda _req: ApprovalDecision.DENY)
    try:
        assert request_approval(_request(data_root)) is ApprovalDecision.ONCE
        assert request_approval(_request(other_root)) is ApprovalDecision.DENY
    finally:
        register_approval_backend(None)

    home_after = list(home.rglob("*")) if home.exists() else []
    assert home_after == home_before
