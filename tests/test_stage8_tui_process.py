"""文件真实捕获到正式 TUI 预览恢复的完整使用链。

作者：xxx
时间：2026-09-30 20:40:00
"""

import json
import subprocess
import sys
from pathlib import Path

from app.background.client import ensure_running
from runtime.session_message_store import SessionMessageStore
from tests.test_background_process import REPOSITORY, process_setup, wait_until

__all__ = ["process_setup"]
DRIVER_TIMEOUT_SECONDS = 35


def test_formal_tui_restores_captured_tool_file_without_model_call(process_setup):
    """真实SDK工具写入后经正式恢复入口撤销，历史及模型调用保持不变；参数：隔离后台；返回：无。"""
    project, model = process_setup
    root = Path.home() / ".reins/data"
    client = ensure_running(project_root=project, data_root=root)
    session = client.call("attach", project_root=str(project))["session_id"]
    model.first_release.set()
    model.final_release.set()
    client.call(
        "submit",
        session_id=session,
        input_id="stage8-real-input",
        text="写出文件供预览恢复",
        model_config={
            "provider": "openai_compatible",
            "model": "background-test-model",
            "base_url": f"http://127.0.0.1:{model.server_port}/v1",
            "api_mode": "chat_completions",
        },
        api_key="isolated-test-key",
    )
    wait_until(lambda: client.call("poll", session_id=session)["status"] == "done")
    target = project / "background-result.txt"
    assert target.read_text(encoding="utf-8") == "只写一次"
    entries = SessionMessageStore(root).read_entries(session)
    count = len(model.packets)
    script = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')"
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            script,
            str(REPOSITORY / "tests/scripts/tui_restore_driver.py"),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=DRIVER_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["operation"]["status"] == "completed"
    assert "只写一次" in receipt["difference"]
    assert not target.exists()
    assert SessionMessageStore(root).read_entries(session) == entries
    assert len(model.packets) == count
    same = client.call(
        "file_restore_execute",
        payload={
            "plan_id": receipt["operation"]["plan_id"],
            "confirmation": {"accepted": True},
        },
    )
    assert same["operation_id"] == receipt["operation"]["operation_id"]
    assert not target.exists()
