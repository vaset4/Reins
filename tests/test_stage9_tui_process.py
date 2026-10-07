"""【上下文】【正式入口联验】真实后台进程、界面与重启读取同一持久事实。

作者：xxx
时间：2026-10-01 15:00:00
"""

import json
import subprocess
import sys
from pathlib import Path

from app.background.client import ensure_running
from app.background.context_management import ContextManagement
from app.background.sessions import BackgroundSessionRecords, SessionRecord
from runtime.lease import from_trigger
from runtime.session_message_store import SessionMessageStore
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from scripts.testing.llm import from_test_turns
from tests.test_background_process import REPOSITORY, process_setup

__all__ = ["process_setup"]
DRIVER_TIMEOUT_SECONDS = 35


def test_formal_context_panel_preserves_sources_and_controls_after_host_restart(
    process_setup,
):
    """正式入口查看及暂停不调用模型，后台重启保持设置和来源；参数：隔离后台；返回：无。"""
    project, model = process_setup
    root = Path.home() / ".reins/data"
    session = "stage9-context-session"
    WorkspaceStore(root).bind_session(session, project)
    messages = SessionMessageStore(root)
    identity = messages.accept_input(
        session, "请记住这个项目只在本地运行", run_id="source-run"
    ).entry_id
    messages.deliver_inputs(session, run_id="source-run", task_id=None)
    context = RunContext(
        trigger=Trigger.USER,
        session_id=session,
        run_id="source-run",
        payload={"input_message_id": identity},
        capability_lease=from_trigger(
            "user",
            capabilities={
                "background_run": {"enabled": True},
                "fs": {"read": [str(project)], "project_root": str(project)},
            },
        ),
    )
    manager = ContextManagement(root)
    row = manager.knowledge.observe(context, client=from_test_turns([]))
    assert row is not None
    manager.knowledge.cancel(row["work_id"])
    BackgroundSessionRecords(root).save(SessionRecord(session))
    originals = SessionMessageStore(root).read_entries(session)
    client = ensure_running(project_root=project, data_root=root)
    client.call("attach", session_id=session, project_root=str(project))
    script = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')"
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            script,
            str(REPOSITORY / "tests/scripts/tui_context_driver.py"),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=DRIVER_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["session_id"] == session and session in receipt["detail"]
    assert client.stop()
    restarted = ensure_running(project_root=project, data_root=root)
    assert restarted.data_space_id == client.data_space_id
    view = restarted.call(
        "context_management",
        payload={
            "data_space_id": client.data_space_id,
            "session_id": session,
            "action": "overview",
        },
    )
    assert not view["history"]["enabled"] and not view["knowledge"]["enabled"]
    assert SessionMessageStore(root).read_entries(session) == originals
    assert not model.packets and not (project / "background-result.txt").exists()
