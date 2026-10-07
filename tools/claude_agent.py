"""Claude Code双向SDK协议适配，执行能力由Reins工具桥提供。

作者：xxx
时间：2026-09-14 15:20:00
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, TextIO
from uuid import uuid4

from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.secret_redaction import redact_secret_text
from tools.exec_channel import _kill_process_tree

_READ_POLL_SECONDS = 0.05
_INITIALIZE_SECONDS = 30.0
_CLOSE_SECONDS = 2.0
_STDERR_TAIL_CHARS = 4096


@dataclass(frozen=True, slots=True)
class ClaudeAgentOptions:
    """外部会话启动信息；cwd为实际工作目录，session_id可恢复，模型与输出限额由调用方注入。"""

    cwd: Path
    session_id: str
    output_tokens: int
    max_turns: int
    model: str | None = None
    resume: bool = False
    executable: Path | None = None
    instructions: str = ""


def find_claude_executable() -> Path:
    """定位已安装的原生CLI，Windows批处理包装不经shell执行；传参：无；返回：真实exe路径。"""
    native = shutil.which("claude.exe")
    candidates = [Path(native)] if native else []
    shim = shutil.which("claude.cmd")
    if shim:
        candidates.append(
            Path(shim).parent / "node_modules/@anthropic-ai/claude-code/bin/claude.exe"
        )
    candidates.append(Path.home() / ".local/bin/claude.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Claude Code native executable not found; install or configure an authenticated local Claude Code"
    )


class ClaudeAgentProcess:
    """保持stdin开放的外部会话；消息、SDK MCP请求、停止和真实结果可以双向交错。"""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        cancellation: CancellationToken,
        control: Callable[[dict[str, Any]], dict[str, Any]],
        evidence: Callable[[dict[str, Any]], None],
    ) -> None:
        """注入控制请求执行者和证据写者；传参：启动选项、停止信号与回调；返回：无。"""
        self.options, self.cancellation = options, cancellation
        self._control, self._evidence = control, evidence
        self._process: subprocess.Popen[str] | None = None
        self._queue: Queue[dict[str, Any] | BaseException | None] = Queue()
        self._backlog: deque[dict[str, Any]] = deque()
        self._write_lock = Lock()
        self._stderr = ""

    def start(self) -> None:
        """创建真实进程并完成SDK握手；传参：无；返回：无，配置或握手失败明确暴露。"""
        if self.cancellation.cancelled:
            raise ExecutionCancelled("external agent cancelled before start")
        if self._process is not None:
            raise RuntimeError("external agent process already started")
        options = self.options
        command = [
            str(options.executable or find_claude_executable()),
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--tools",
            "",
            "--disable-slash-commands",
            "--permission-prompt-tool",
            "stdio",
            "--strict-mcp-config",
            "--mcp-config",
            json.dumps({"mcpServers": {"reins": {"type": "sdk", "name": "reins"}}}),
            "--max-turns",
            str(options.max_turns),
            "--setting-sources",
            "user",
            "--settings",
            json.dumps({"disableAllHooks": True}),
        ]
        command.append(
            f"--resume={options.session_id}"
            if options.resume
            else f"--session-id={options.session_id}"
        )
        if options.model:
            command.append(f"--model={options.model}")
        if options.instructions:
            command.extend(["--append-system-prompt", options.instructions])
        env = {
            **os.environ,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(options.output_tokens),
        }
        self._process = subprocess.Popen(
            command,
            cwd=options.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        assert self._process.stdout is not None and self._process.stderr is not None
        Thread(
            target=self._read_stdout, args=(self._process.stdout,), daemon=True
        ).start()
        Thread(
            target=self._read_stderr, args=(self._process.stderr,), daemon=True
        ).start()
        self.cancellation.report_backend(
            backend="claude_code", pid=self._process.pid, supports_stop=True
        )
        self.cancellation.register_closer(self.interrupt)
        self._initialize()

    def send(self, text: str, *, input_id: str) -> None:
        """运行中送入有稳定身份的用户/协作输入；传参：正文与输入ID；返回：无，后续结果另行读取。"""
        self._write(
            {
                "type": "user",
                "uuid": input_id,
                "session_id": self.options.session_id,
                "message": {"role": "user", "content": text},
                "parent_tool_use_id": None,
            }
        )

    def next_event(self, timeout: float = _READ_POLL_SECONDS) -> dict[str, Any] | None:
        """保持控制通道可用并读取一个真实事件；传参：等待秒数；返回：事件，暂时无数据为空。"""
        if self._backlog:
            return self._backlog.popleft()
        return self._receive(timeout)

    def interrupt(self) -> None:
        """把取消送至仍开放的SDK控制通道；传参：无；返回：无，发送本身不是停止证明。"""
        process = self._process
        if process is not None and process.poll() is None:
            self._write(
                {
                    "type": "control_request",
                    "request_id": f"interrupt-{uuid4().hex}",
                    "request": {"subtype": "interrupt"},
                }
            )

    def close(self) -> dict[str, object]:
        """关闭会话进程并核对退出，必要时复用现有进程树停止边界；传参：无；返回：实际本机停止证据。"""
        process = self._process
        if process is None:
            return {"execution_state": "not_started"}
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=_CLOSE_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
        result = {
            "execution_state": "stopped",
            "pid": process.pid,
            "exit_code": process.returncode,
            "external_session_id": self.options.session_id,
            "remote_model_stop": "unknown",
        }
        self.cancellation.report_backend(**result)
        return result

    def _initialize(self) -> None:
        """初始化SDK MCP连接，保留握手期间的真实事件；传参：无；返回：无。"""
        identity = f"initialize-{uuid4().hex}"
        self._write(
            {
                "type": "control_request",
                "request_id": identity,
                "request": {"subtype": "initialize", "hooks": None, "skills": []},
            }
        )
        deadline = time.monotonic() + _INITIALIZE_SECONDS
        while time.monotonic() < deadline:
            if self.cancellation.cancelled:
                raise ExecutionCancelled("external initialization cancelled")
            event = self._receive(_READ_POLL_SECONDS)
            if event is None:
                continue
            if (
                event.get("type") == "control_response"
                and event.get("response", {}).get("request_id") == identity
            ):
                if event["response"].get("subtype") != "success":
                    raise RuntimeError(
                        f"Claude SDK initialization failed: {event['response']}"
                    )
                return
            self._backlog.append(event)
        raise TimeoutError("Claude SDK initialization timed out")

    def _receive(self, timeout: float) -> dict[str, Any] | None:
        """在业务事件之前处理真实控制请求，错误不伪装成成功；传参：超时；返回：事件或暂时空值。"""
        try:
            event = self._queue.get(timeout=timeout)
        except Empty:
            return None
        if event is None:
            raise RuntimeError(
                f"Claude Code connection closed before result: {self._stderr}"
            )
        if isinstance(event, BaseException):
            raise event
        self._evidence({"direction": "received", "message": event})
        if event.get("type") != "control_request":
            return event
        identity = event["request_id"]
        try:
            response = self._control(event["request"])
        except Exception as exc:
            self._write(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "error",
                        "request_id": identity,
                        "error": str(exc),
                    },
                }
            )
            raise
        self._write(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": identity,
                    "response": response,
                },
            }
        )
        return None

    def _write(self, message: dict[str, Any]) -> None:
        """按行串行提交控制或输入消息，先保存发送证据；传参：消息；返回：无。"""
        process = self._process
        if process is None or process.stdin is None or process.stdin.closed:
            raise RuntimeError("Claude Code input channel is not open")
        with self._write_lock:
            self._evidence({"direction": "sent", "message": message})
            process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            process.stdin.flush()

    def _read_stdout(self, stream: TextIO) -> None:
        """读取完整JSON行，不丢弃未知协议错误；传参：stdout；返回：无，错误交给运行所有者。"""
        try:
            for line in stream:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Claude Code event must be a JSON object")
                self._queue.put(value)
        except Exception as exc:
            self._queue.put(exc)
        finally:
            self._queue.put(None)

    def _read_stderr(self, stream: TextIO) -> None:
        """保留有界脱敏诊断，正文不会进入模型历史；传参：stderr；返回：无。"""
        for line in stream:
            self._stderr = (self._stderr + redact_secret_text(line))[
                -_STDERR_TAIL_CHARS:
            ]
