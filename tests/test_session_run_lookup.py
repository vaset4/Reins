"""验证已知会话的运行事实读取及后台回执恢复不遍历其他会话。

作者：xxx
时间：2026-09-28 23:00:00
"""

from pathlib import Path
from typing import NoReturn
from collections.abc import Iterator

import pytest

from app.background.sessions import BackgroundSession, SessionRecord, SessionServices
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from schedules.notifications import NotificationStore
from tools.tool_registry import ToolRegistry


def test_session_run_read_keeps_identity_and_observes_new_facts(tmp_path: Path) -> None:
    """读取指定会话，追加事实立即可见，不借用其他会话记录；传参：隔离目录；返回：无。"""
    facts = RunFactStore(tmp_path)
    facts.append(
        {
            "session_id": "session-a",
            "run_id": "run-shared",
            "event": "run:lifecycle",
            "lifecycle": "done",
        }
    )
    facts.append(
        {
            "session_id": "session-b",
            "run_id": "run-shared",
            "event": "run:lifecycle",
            "lifecycle": "waiting_user",
        }
    )
    assert [
        row["lifecycle"] for row in facts.read_session_run("session-a", "run-shared")
    ] == ["done"]
    assert facts.read_session_run("session-missing", "run-shared") == []
    facts.append(
        {
            "session_id": "session-a",
            "run_id": "run-shared",
            "event": "input:handled",
            "input_ids": ["new"],
        }
    )
    restarted = RunFactStore(tmp_path)
    assert restarted.read_session_run("session-a", "run-shared")[-1]["input_ids"] == [
        "new"
    ]
    assert len(restarted.read_session_run("session-b", "run-shared")) == 1


def test_session_run_read_exposes_corruption(tmp_path: Path) -> None:
    """已知路径中的损坏事实不能被空结果掩盖；传参：隔离目录；返回：无。"""
    facts = RunFactStore(tmp_path)
    path = facts.append(
        {"session_id": "session-a", "run_id": "run-a", "event": "run:start"}
    )
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="committed source corrupt"):
        facts.read_session_run("session-a", "run-a")


def test_completed_background_recovery_avoids_global_session_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """终态回执恢复无需全会话定位，重复恢复只产生一份通知；传参：隔离目录和读量计量；返回：无。"""

    def forbidden_model(_options: dict[str, object]) -> NoReturn:
        """终态恢复不能重新装配模型；传参：模型选项；返回：无，误调用直接失败。"""
        raise AssertionError("completed recovery attempted a new model run")

    session_id, run_id = "session-background", "run-completed"
    SessionMessageStore(tmp_path).create_session(session_id)
    RunFactStore(tmp_path).append(
        {
            "session_id": session_id,
            "run_id": run_id,
            "event": "run:lifecycle",
            "lifecycle": "done",
        }
    )
    services = SessionServices(tmp_path, tmp_path, forbidden_model, ToolRegistry)
    session = BackgroundSession(
        SessionRecord(
            session_id,
            status="running",
            intent={"run_id": run_id, "input_id": "accepted-input"},
        ),
        services,
    )
    traversals: list[str] = []
    original_glob = Path.glob

    def counted_glob(path: Path, pattern: str) -> Iterator[Path]:
        """计量运行定位的真实目录遍历调用；传参：路径/模式；返回：原迭代器。"""
        if path == tmp_path / "sessions" and pattern.startswith("*/runs/"):
            traversals.append(pattern)
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", counted_glob)
    try:
        session.recover()
        session.recover()
        assert session.record.status == "done"
        assert not session.runtime.active
        notices = NotificationStore(tmp_path).list_all()
        assert len(notices) == 1 and notices[0].source["run_id"] == run_id
        assert traversals == []
    finally:
        session.close()
