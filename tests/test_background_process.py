"""用真实 Windows 前后台进程验证关闭界面、重连和崩溃恢复。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import json
import importlib
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread

import pytest

from app.background.client import connect, ensure_running, is_running
from app.background.sessions import BackgroundSessionRecords
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.tool_operations import ToolOperationStore
from schedules.notifications import NotificationStore
from schedules.store import ScheduleStore
from runtime.workspaces import WorkspaceStore

REPOSITORY = Path(__file__).resolve().parents[1]
WAIT_SECONDS = 15


class ModelBoundary(ThreadingHTTPServer):
    """仅替换模型的网络边界，实际 SDK、工具、会话与后台均使用生产路径。"""

    daemon_threads = True

    def __init__(self):
        """创建可控制回复时机的模型服务；传参：无；返回：无。"""
        self.first_release, self.final_release, self.received_final = (
            Event(),
            Event(),
            Event(),
        )
        self.packets = []
        super().__init__(("127.0.0.1", 0), ModelRequest)


class ModelRequest(BaseHTTPRequestHandler):
    """返回标准 Chat Completions 事件，不伪造工具执行结果。"""

    def do_POST(self):
        """首轮要求真实写文件，后续只在已有结果后回复；传参：HTTP 请求；返回：标准事件流。"""
        packet = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.packets.append(packet)
        has_result = any(message["role"] == "tool" for message in packet["messages"])
        if has_result:
            self.server.received_final.set()
            assert self.server.final_release.wait(WAIT_SECONDS * 2)
            delta, finish = {"content": "后台文件已完成"}, "stop"
        else:
            assert self.server.first_release.wait(WAIT_SECONDS * 2)
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "write-once",
                        "type": "function",
                        "function": {
                            "name": "file_write",
                            "arguments": json.dumps(
                                {"path": "background-result.txt", "content": "只写一次"}
                            ),
                        },
                    }
                ]
            }
            finish = "tool_calls"
        template = {
            "id": "model-boundary",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": packet["model"],
        }
        rows = [
            {
                **template,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                **template,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        ]
        body = (
            "".join("data: " + json.dumps(row) + "\n\n" for row in rows)
            + "data: [DONE]\n\n"
        ).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, _format, *args):
        """不输出模型请求或认证头；传参：HTTP 日志；返回：无。"""
        return


@pytest.fixture
def process_setup(tmp_path, monkeypatch):
    """隔离用户配置和实际前后台进程，结束时核实后台退出；传参：临时目录；返回：项目与模型服务。"""
    project, home = tmp_path / "project", tmp_path / "profile"
    project.mkdir()
    home.mkdir()
    model = ModelBoundary()
    thread = Thread(target=model.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(project))
    monkeypatch.setenv(
        "XIANGMU_LLM_BASE_URL", f"http://127.0.0.1:{model.server_port}/v1"
    )
    monkeypatch.setenv("XIANGMU_LLM_MODEL", "background-test-model")
    monkeypatch.setenv("XIANGMU_LLM_API_KEY", "isolated-test-key")
    monkeypatch.setenv("XIANGMU_LLM_API_MODE", "chat_completions")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    # 1. 【后台会话】【重连验证】本组隔离自动知识维护，模型边界只模拟当前文件任务
    from runtime.knowledge_maintenance import KnowledgeMaintenance

    KnowledgeMaintenance(Path.home() / ".reins" / "data").configure(enabled=False)
    try:
        yield project, model
    finally:
        model.first_release.set()
        model.final_release.set()
        root = Path.home() / ".reins" / "data"
        if is_running(root):
            assert connect(root).stop(timeout=WAIT_SECONDS), (
                root / "background" / "host.log"
            ).read_text(errors="replace")
        model.shutdown()
        model.server_close()
        thread.join()


def frontend(entry, *, message=True, reconnect=False):
    """从公开入口输入或重连后断开；传参：入口、是否发送和是否等历史；返回：真实前台退出结果。"""
    if entry == "repl":
        command = [sys.executable, "-m", "app.cli", "chat"]
        input_text = "写出本地结果\n/exit\n" if message else "/exit\n"
        return subprocess.run(
            command,
            cwd=REPOSITORY,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=WAIT_SECONDS,
        )
    script = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')"
    mode = "send" if message else ("reconnect" if reconnect else "smoke")
    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(REPOSITORY / "tests/scripts/tui_process_driver.py"),
            mode,
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=WAIT_SECONDS,
    )


def wait_until(predicate, *, timeout=WAIT_SECONDS):
    """等待实际文件或执行回执，超时暴露失败；传参：可观察条件；返回：最后观测结果。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("expected durable result did not arrive")


@pytest.mark.parametrize("entry", ["repl", "tui"])
def test_frontend_exit_keeps_work_running_and_reconnects_without_replay(
    process_setup, entry
):
    """REPL退出与TUI断开均不等待后台；重连不重放；传参：隔离进程与入口；返回：无。"""
    project, model = process_setup
    result = frontend(entry)
    assert result.returncode == 0, result.stderr
    root = Path.home() / ".reins" / "data"
    client = connect(root)
    assert is_running(root) and not (project / "background-result.txt").exists()
    model.first_release.set()
    model.final_release.set()
    session = client.call("attach")["session_id"]
    wait_until(lambda: client.call("poll", session_id=session)["status"] == "done")
    output = project / "background-result.txt"
    assert output.read_text(encoding="utf-8") == "只写一次"
    before = output.stat().st_mtime_ns
    reopened = frontend(entry, message=False, reconnect=True)
    assert reopened.returncode == 0, reopened.stderr
    assert "后台文件已完成" in reopened.stdout
    if entry == "tui":
        cards = json.loads(reopened.stdout.splitlines()[-1])
        assert len({card["key"] for card in cards}) == len(cards)
        assert sum(card["text"] == "后台文件已完成" for card in cards) == 1
        assert sum(card["role"] == "tool" for card in cards) == 1
    assert output.stat().st_mtime_ns == before and len(model.packets) == 2
    assert (
        len(
            [
                row
                for row in SessionMessageStore(root).read_entries(session)
                if row.type == "inbound"
            ]
        )
        == 1
    )


def test_real_host_restart_preserves_finished_tool_and_budget_root(process_setup):
    """文件已写而模型仍等待时杀掉宿主，重启不能重做副作用或刷新额度；传参：隔离进程；返回：无。"""
    importlib.import_module("pywintypes")
    import win32api
    import win32con
    import win32event

    project, model = process_setup
    model.first_release.set()
    assert frontend("repl").returncode == 0
    root = Path.home() / ".reins" / "data"
    client = connect(root)
    assert model.received_final.wait(WAIT_SECONDS)
    output = project / "background-result.txt"
    before = output.stat().st_mtime_ns
    session = client.call("attach")["session_id"]
    original = BackgroundSessionRecords(root).load(session)
    assert original is not None
    handle = win32api.OpenProcess(
        win32con.PROCESS_TERMINATE | win32con.SYNCHRONIZE, False, client.pid
    )
    try:
        win32api.TerminateProcess(handle, 23)
        assert (
            win32event.WaitForSingleObject(handle, WAIT_SECONDS * 1000)
            == win32event.WAIT_OBJECT_0
        )
    finally:
        handle.Close()
    model.final_release.set()
    restarted = ensure_running(project_root=project, data_root=root)
    wait_until(lambda: restarted.call("poll", session_id=session)["status"] == "done")
    current = BackgroundSessionRecords(root).load(session)
    assert current is not None
    assert current.intent["root_run_id"] == original.intent["root_run_id"]
    assert current.intent["run_id"] != original.intent["run_id"]
    assert output.stat().st_mtime_ns == before
    operations = ToolOperationStore(root).for_session(session)
    assert (
        len([row for row in operations if row["call"]["tool_name"] == "file_write"])
        == 1
    )
    assert len(RunFactStore(root).list_runs_for_session(session)) == 2


def test_real_background_delivers_due_reminder_after_frontend_exit(process_setup):
    """关闭真实界面后到期提醒进入 Windows 渠道，提交和已读状态分开；传参：隔离进程；返回：无。"""
    project, _model = process_setup
    assert frontend("tui", message=False).returncode == 0
    root = Path.home() / ".reins" / "data"
    with closing(ScheduleStore(root)) as store:
        due = datetime.now(timezone.utc) + timedelta(seconds=1)
        store.create_schedule(
            "reminder-after-close",
            "at:" + due.isoformat(),
            name="Reins 后台验证",
            kind="reminder",
            workspace_id=WorkspaceStore(root).register(project).workspace_id,
            timezone_name="Asia/Shanghai",
            prompt="关闭聊天窗口后到期的隔离测试提醒",
        )

    def submitted():
        """读取真实渠道提交，不用入队状态代替；传参：无；返回：已提交通知或无。"""
        return next(
            (
                row
                for row in NotificationStore(root).list_all()
                if row.delivery_status == "submitted"
            ),
            None,
        )

    notice = wait_until(submitted)
    assert notice.read_at is None and notice.attempts[-1]["channel"] == "windows_shell"
    assert notice.source["scheduled_at"] == due.isoformat()


def test_explicit_host_stop_cancels_active_work_and_restart_keeps_it_stopped(
    process_setup,
):
    """明确停止整个后台后，重新启动不能执行旧的待发工具；传参：隔离进程；返回：无。"""
    project, model = process_setup
    assert frontend("repl").returncode == 0
    root = Path.home() / ".reins" / "data"
    wait_until(lambda: model.packets)
    client = connect(root)
    session = client.call("attach")["session_id"]
    assert client.stop(timeout=WAIT_SECONDS)
    model.first_release.set()
    model.final_release.set()
    count = len(model.packets)
    restarted = ensure_running(project_root=project, data_root=root)
    state = restarted.call("poll", session_id=session)
    assert state["stopped"] and not state["active"]
    assert (
        not (project / "background-result.txt").exists() and len(model.packets) == count
    )
