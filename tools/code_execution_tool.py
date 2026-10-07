from __future__ import annotations

from pathlib import Path

import path_security
from context.artifact_ref import store_large_output
from runtime.lease import Lease
from runtime.cancellation import CancellationToken
from typing import cast
from tools.exec_channel import ExecChannel, ExecResult, ExecSpec, LocalSubprocessBackend
from tools.types import ToolError, ToolErrorCategory

_CHANNEL = ExecChannel(LocalSubprocessBackend())


def code_execute(
    code: str,
    cwd: Path,
    timeout_seconds: float = 30.0,
    *,
    cancellation: CancellationToken | None = None,
) -> dict[str, object]:
    # 受控执行核：经 ExecChannel 子进程跑 Python（落盘临时脚本 + 受控 header +
    # env 剥机密 + 输出脱敏 + 超时杀进程组）。cwd 由调用方固定到 lease workspace。
    result = _CHANNEL.run(
        ExecSpec(
            code=code,
            kind="py",
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )
    )
    return _to_payload(result)


def executor(args: dict[str, object]) -> object:
    code = str(args.get("code", ""))
    if not code.strip():
        return ToolError(ToolErrorCategory.INVALID_INPUT, "missing code")
    lease = args.get("__lease__")
    if not isinstance(lease, Lease):
        # fail-closed：无 lease 不在受控通道外跑代码。
        return ToolError(
            ToolErrorCategory.PERMISSION, "code_execution_no_lease", retryable=False
        )
    if not _code_execution_enabled(lease):
        return ToolError(
            ToolErrorCategory.PERMISSION, "code_execution_disabled", retryable=False
        )
    cwd = _resolve_cwd(lease)
    if cwd is None:
        return ToolError(
            ToolErrorCategory.PERMISSION, "code_execution_no_workspace", retryable=False
        )
    timeout_seconds = _float_arg(args.get("__timeout_seconds__"), 30.0)
    result = code_execute(
        code,
        cwd,
        timeout_seconds=timeout_seconds,
        cancellation=cast(CancellationToken | None, args.get("__cancellation__")),
    )
    if result.get("timed_out") or result.get("cancelled"):
        return ToolError(
            ToolErrorCategory.TIMEOUT
            if result.get("timed_out")
            else ToolErrorCategory.CANCELLED,
            "code_execution_interrupted",
            retryable=False,
            partial_state="previous side effects are not rolled back",
            details=result,
        )
    return _artifact_large_stdout(result, args)


def _code_execution_enabled(lease: Lease) -> bool:
    capability = lease.capabilities.get("code_execution")
    if isinstance(capability, dict):
        return capability.get("enabled") is not False
    return True


def _resolve_cwd(lease: Lease) -> Path | None:
    workspace = path_security.task_workspace(lease)
    if workspace is None:
        return None
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _to_payload(result: ExecResult) -> dict[str, object]:
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "cancelled": result.cancelled,
        "execution_state": result.execution_state,
        "pid": result.pid,
        "stop_error": result.stop_error,
    }


def _artifact_large_stdout(
    result: dict[str, object], args: dict[str, object]
) -> dict[str, object]:
    stdout = str(result.get("stdout", ""))
    data_root = args.get("__data_root__")
    task_id = str(args.get("__task_id__", "")).strip()
    if data_root is None or not task_id:
        return result
    if not isinstance(data_root, str | Path):
        return result
    ref = store_large_output(
        Path(data_root),
        task_id,
        stdout,
        summary="code execution stdout",
    )
    if ref is None:
        return result
    updated = dict(result)
    updated["stdout"] = ""
    updated["stdout_artifact"] = ref.to_dict()
    return updated


def _float_arg(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return default
