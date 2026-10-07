"""停止真实进程、未知效果与迟到结果验证。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

import ctypes
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from runtime.cancellation import CancellationToken
from runtime.lease import from_trigger
from runtime.watchdog import Watchdog
from tools.exec_channel import ExecResult, ExecSpec, LocalSubprocessBackend
from tools.types import ToolError


def test_uncancellable_thread_returns_unknown_and_delivers_late_result() -> None:
    """超时后不声称线程已杀死，真实结果仍能交回；传参：无；返回：无。"""
    release, delivered = Event(), Event()
    late: list[object] = []
    watchdog = Watchdog(from_trigger("user"), tool_timeout_seconds=0.02)

    def operation() -> str:
        """模拟已在执行但不能中断的后端；传参：无；返回：实际晚到结果。"""
        release.wait(5)
        return "late real result"

    def receive(result: object) -> None:
        """保存原操作结果；传参：后端回执；返回：无。"""
        late.append(result)
        delivered.set()

    try:
        result = watchdog.run_tool_with_timeout(operation, on_late=receive)
        assert isinstance(result, ToolError)
        assert result.details["execution_state"] == "unknown"
        assert result.retryable is False
        assert "killed" not in result.partial_state
    finally:
        release.set()
    assert delivered.wait(3)
    assert late == ["late real result"]


def _wait_file(path: Path) -> None:
    """等待真实子进程的就绪文件，有明确测试截止；传参：文件；返回：无。"""
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows process tree contract")
def test_stop_reaches_actual_windows_process_tree(tmp_path: Path) -> None:
    """根进程和实际子进程都终止，已写文件保留；传参：临时目录；返回：无。"""
    token = CancellationToken()
    code = (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path('child.pid').write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    spec = ExecSpec(code, "py", tmp_path, timeout_seconds=10, cancellation=token)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(LocalSubprocessBackend().run, spec)
        _wait_file(tmp_path / "child.pid")
        child_pid = int((tmp_path / "child.pid").read_text())
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, child_pid)
        assert handle
        try:
            token.cancel()
            result = future.result(timeout=5)
            assert result.cancelled is True
            assert result.execution_state == "stopped"
            assert kernel.WaitForSingleObject(handle, 1000) == 0
        finally:
            kernel.CloseHandle(handle)


def test_failed_backend_stop_is_unknown_until_real_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """停止失败不伪造成功，稍后实际结束仍回收结果；传参：临时目录与替换器；返回：无。"""
    token, delivered = CancellationToken(), Event()
    late: list[object] = []
    code = "from pathlib import Path\nimport time\nPath('ready').touch()\nwhile not Path('release').exists(): time.sleep(0.01)\nprint('real completion')"
    spec = ExecSpec(code, "py", tmp_path, timeout_seconds=0.1, cancellation=token)

    def fail_stop(_process: object, *, tree: object = None) -> None:
        """模拟系统拒绝终止进程；传参：真实子进程；返回：无，抛出停止失败。"""
        raise RuntimeError("stop backend rejected termination")

    def receive(result: object) -> None:
        """接收实际完成后的结果；传参：迟到回执；返回：无。"""
        late.append(result)
        delivered.set()

    monkeypatch.setattr("tools.exec_channel._kill_process_tree", fail_stop)
    watchdog = Watchdog(from_trigger("user"), tool_timeout_seconds=0.2)
    try:
        result = watchdog.run_tool_with_timeout(
            lambda: LocalSubprocessBackend().run(spec),
            cancellation=token,
            on_late=receive,
        )
        assert isinstance(result, ToolError)
        assert result.details["execution_state"] == "unknown"
        assert result.details["stop_confirmed"] is False
        assert "rejected" in result.details["stop_error"]
    finally:
        (tmp_path / "release").touch()
    assert delivered.wait(3)
    assert isinstance(late[0], ExecResult)
    assert late[0].stdout.strip() == "real completion"
    assert late[0].execution_state == "completed"
