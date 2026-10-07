"""执行后端可替换性验证，仅证明接口传递。

作者：xxx
时间：2026-09-22 10:30:00
"""

from pathlib import Path

from runtime.cancellation import CancellationToken
from tools.exec_channel import ExecBackend, ExecChannel, ExecResult, ExecSpec


class _FakeBackend(ExecBackend):
    def __init__(self) -> None:
        """保存收到的执行规格；传参：无；返回：无。"""
        self.specs: list[ExecSpec] = []

    def run(self, spec: ExecSpec) -> ExecResult:
        """返回预设的后端结果以核对通道透传；传参：执行规格；返回：对应结果。"""
        self.specs.append(spec)
        if spec.cancellation is not None and spec.cancellation.cancelled:
            return ExecResult("", "", -1, cancelled=True, execution_state="not_started")
        if spec.code == "stderr":
            return ExecResult("", "diagnostic", 3)
        if spec.code == "timeout":
            return ExecResult(
                "partial", "", -1, timed_out=True, execution_state="stopped"
            )
        return ExecResult(spec.code, "", 0)


def test_exec_backend_interface_covers_success_error_stderr_timeout_and_cancel(
    tmp_path: Path,
) -> None:
    """通道保留成功、失败与停止的不同事实；传参：临时目录；返回：无。"""
    channel = ExecChannel(_FakeBackend())
    assert channel.run(ExecSpec("echo", "py", tmp_path)).stdout == "echo"
    failed = channel.run(ExecSpec("stderr", "py", tmp_path))
    assert failed.exit_code == 3 and failed.stderr == "diagnostic"
    timed_out = channel.run(ExecSpec("timeout", "py", tmp_path))
    assert timed_out.timed_out and timed_out.execution_state == "stopped"
    token = CancellationToken()
    token.cancel("user_stop")
    cancelled = channel.run(ExecSpec("echo", "py", tmp_path, cancellation=token))
    assert cancelled.cancelled and cancelled.execution_state == "not_started"


def test_exec_spec_exposes_backend_mapping_fields(tmp_path: Path) -> None:
    """后端收到完整目录、环境、脱敏与取消参数；传参：临时目录；返回：无。"""
    token = CancellationToken()
    spec = ExecSpec(
        "echo",
        "shell",
        tmp_path,
        env_passthrough={"LANG": "C"},
        cancellation=token,
        timeout_seconds=7,
        redact_values=("synthetic-secret",),
    )
    backend = _FakeBackend()

    ExecChannel(backend).run(spec)

    assert backend.specs == [spec]
    assert backend.specs[0].cancellation is token
