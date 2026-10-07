"""通过原生 Windows ConPTY 驱动正式终端，不替换 Textual 事件循环。

作者：xxx
时间：2026-09-29 19:00:00
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import pywintypes  # noqa: F401
import win32api
import win32con
import win32event
from winpty import Backend, PTY, WinptyError

WAIT_SECONDS = 15
POLL_SECONDS = 0.05
MILLISECONDS_PER_SECOND = 1000
FORCED_EXIT_CODE = 27


class NativeTuiTerminal:
    """持有真实伪控制台和进程句柄，读写 UTF-8 终端流并保存证据。"""

    def __init__(self, repository: Path, evidence: Path) -> None:
        """启动正式入口；参数：源码工作目录、输出证据文件；返回：无。"""
        self.evidence = evidence
        self.output = ""
        self.steps = evidence.with_suffix(".steps")
        self._cursor_queries = 0
        self._eof = False
        # 1. 显式选择原生后端，不允许自动切换到旧 WinPTY
        self.pty = PTY(120, 40, backend=Backend.ConPTY)
        command = subprocess.list2cmdline(
            [
                "-X",
                "utf8",
                "-c",
                "from frontends.tui.main import main; raise SystemExit(main([]))",
            ]
        )
        assert self.pty.spawn(
            sys.executable, cmdline=" " + command, cwd=str(repository)
        )
        self.handle = win32api.OpenProcess(
            win32con.SYNCHRONIZE | win32con.PROCESS_TERMINATE,
            False,
            self.pty.pid,
        )

    def __enter__(self) -> NativeTuiTerminal:
        """进入资源作用域；参数：无；返回：终端本身。"""
        return self

    def __exit__(self, *_exc) -> None:
        """结束时释放进程与伪控制台；参数：异常信息；返回：无。"""
        self.record("cleanup start")
        try:
            if self.alive():
                win32api.TerminateProcess(self.handle, FORCED_EXIT_CODE)
                assert (
                    win32event.WaitForSingleObject(
                        self.handle,
                        WAIT_SECONDS * MILLISECONDS_PER_SECOND,
                    )
                    == win32event.WAIT_OBJECT_0
                )
            self.record("cleanup process stopped")
            self.pump()
        finally:
            self.evidence.write_text(self.output, encoding="utf-8")
            self.handle.Close()
            self.record("cleanup handle closed")
            # 2. pywinpty 的原生对象析构关闭 HPCON 与所属管道
            self.pty = None
            self.record("cleanup pseudoconsole closed")

    def record(self, step: str) -> None:
        """保存原生终端交互步骤以定位阻塞；参数：步骤；返回：无。"""
        with self.steps.open("a", encoding="utf-8") as stream:
            stream.write(f"{time.monotonic():.3f} {step}\n")

    def pump(self) -> None:
        """非阻塞读取真实终端输出；参数：无；返回：无。"""
        if self._eof:
            return
        try:
            self.output += self.pty.read(blocking=False)
        except WinptyError as exc:
            # 标准输出可先于进程句柄结束，EOF 不等同于进程已成功退出
            if str(exc) != "Standard out reached EOF":
                raise
            self._eof = True
            self.record("terminal output EOF; awaiting process exit")
            return
        queries = self.output.count("\x1b[6n")
        if queries > self._cursor_queries:
            self.pty.write("\x1b[1;1R" * (queries - self._cursor_queries))
            self._cursor_queries = queries

    def send(self, text: str) -> None:
        """将按键或括号粘贴写入终端设备；参数：终端字符；返回：无。"""
        self.record(f"send {text!r}")
        self.pty.write(text)
        self.record("sent")

    def resize(self, columns: int, rows: int) -> None:
        """调用原生终端尺寸更新；参数：列、行；返回：无。"""
        self.pty.set_size(columns, rows)

    def alive(self) -> bool:
        """从进程句柄查询存活状态；参数：无；返回：是否尚未退出。"""
        return win32event.WaitForSingleObject(self.handle, 0) == win32event.WAIT_TIMEOUT

    def wait(self, condition: Callable[[], object]) -> object:
        """持续读取输出直到可观察条件成立；参数：条件；返回：条件结果。"""
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            self.pump()
            result = condition()
            if result:
                return result
            if not self.alive():
                raise AssertionError(
                    f"TUI exited early: {self.pty.get_exitstatus()}\n{self.output[-3000:]!r}"
                )
            time.sleep(POLL_SECONDS)
        raise AssertionError(
            f"native terminal condition timed out\n{self.output[-3000:]!r}"
        )

    def disconnect(self) -> None:
        """发送 Ctrl+Q 并核对正常退出；参数：无；返回：无。"""
        self.record("disconnect start")
        self.send("\x11")
        self.wait(lambda: not self.alive())
        self.record("disconnect process stopped")
        assert self.pty.get_exitstatus() == 0
