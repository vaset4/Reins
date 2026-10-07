from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TextIO


class ReinsSubprocess:
    def __init__(
        self,
        args: list[str] | None = None,
        cwd: Path | str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.args = args or [sys.executable, "-m", "app.cli"]
        self.cwd = Path(cwd) if cwd is not None else None
        self.env = {**os.environ, **(env or {})}
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def start(self) -> int:
        if self.process is not None and self.process.poll() is None:
            return self.process.pid
        self.process = subprocess.Popen(
            self.args,
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self.process.pid

    def send(self, msg: str) -> str:
        process = self._require_process()
        stdin = self._require_pipe(process.stdin, "stdin")
        stdout = self._require_pipe(process.stdout, "stdout")
        stdin.write(msg)
        if not msg.endswith("\n"):
            stdin.write("\n")
        stdin.flush()
        return stdout.readline().rstrip("\n")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=self.timeout_seconds)

    def __enter__(self) -> "ReinsSubprocess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop()

    @classmethod
    def resume_last_task(cls, *args, **kwargs) -> "ReinsSubprocess":  # noqa: ANN002, ANN003
        return cls(*args, **kwargs)

    def assert_resumed_correctly(self) -> None:
        if self.process is None:
            raise AssertionError("process was not started")
        if self.process.poll() is not None:
            raise AssertionError("process exited before resume assertion")

    def _require_process(self) -> subprocess.Popen[str]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("process is not running")
        return self.process

    @staticmethod
    def _require_pipe(pipe: TextIO | None, name: str) -> TextIO:
        if pipe is None:
            raise RuntimeError(f"process {name} pipe is unavailable")
        return pipe
