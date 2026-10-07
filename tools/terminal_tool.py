from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from pathlib import Path

import path_security
from runtime.lease import Lease
from runtime.cancellation import CancellationToken
from typing import cast
from tools.exec_channel import ExecChannel, ExecResult, ExecSpec, LocalSubprocessBackend
from tools.types import ToolError, ToolErrorCategory

_CHANNEL = ExecChannel(LocalSubprocessBackend())

DEFAULT_DENY_COMMANDS = (
    "del /f /s /q*",
    "erase /f /s /q*",
    "rd /s /q*",
    "rmdir /s /q*",
    "format*",
    "shutdown*",
    # 跨 shell 破坏性删除（git-bash / WSL / PowerShell）：递归/强制删除即拦。
    "rm -rf*",
    "rm -fr*",
    "rm -r -f*",
    "remove-item*-recurse*",
)

_COMMAND_SEGMENT_SEPARATORS = frozenset({"&", "|", ";"})
_READONLY_COMMANDS = frozenset(
    {
        "pwd",
        "get-location",
        "git status",
        "git status --short",
        "git status --porcelain",
        "git status --branch",
        "git diff",
        "git diff --stat",
        "git diff --name-only",
        "git diff --check",
        "git log --oneline",
    }
)


def is_readonly_command(command: str) -> bool:
    """只声明明确无写入参数的单个命令；传参：实际命令文本；返回：是否可按只读策略运行。"""
    if any(character in command for character in "\r\n;&|><`$"):
        return False
    return " ".join(command.casefold().split()) in _READONLY_COMMANDS


def executor(args: dict[str, object]) -> object:
    """执行已通过注册表审批的终端命令；参数：模型参数及注入租约；返回：退出码、输出或明确错误。"""
    command = str(args.get("command", "")).strip()
    if not command:
        return ToolError(ToolErrorCategory.INVALID_INPUT, "missing command")
    lease = args.get("__lease__")
    if not isinstance(lease, Lease):
        # fail-closed：受控通道不在无能力快照下跑命令。
        return ToolError(
            ToolErrorCategory.PERMISSION, "terminal_no_lease", retryable=False
        )
    permission = _check_terminal_permission(command, lease)
    if permission is not None:
        return permission
    cwd = _resolve_cwd(lease, cast(str | None, args.get("cwd")))
    if cwd is None:
        return ToolError(
            ToolErrorCategory.PERMISSION, "terminal_no_workspace", retryable=False
        )
    if not cwd.is_dir():
        return ToolError(
            ToolErrorCategory.INVALID_INPUT,
            f"terminal cwd is not an existing directory: {cwd}",
        )
    timeout_seconds = _float_arg(args.get("__timeout_seconds__"), 30.0)
    result = _CHANNEL.run(
        ExecSpec(
            code=command,
            kind="shell",
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            cancellation=cast(CancellationToken | None, args.get("__cancellation__")),
        )
    )
    if result.timed_out or result.cancelled:
        return ToolError(
            ToolErrorCategory.TIMEOUT
            if result.timed_out
            else ToolErrorCategory.CANCELLED,
            "terminal_command_interrupted",
            retryable=False,
            partial_state="previous side effects are not rolled back",
            details={
                "execution_state": result.execution_state,
                "pid": result.pid,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "stop_error": result.stop_error,
            },
        )
    return _to_payload(result)


def _resolve_cwd(lease: Lease, requested: str | None = None) -> Path | None:
    """按注册表已校验的路径选择目录，未指定时沿用任务目录；参数：租约与cwd；返回：实际目录。"""
    # 1. 【终端】【工作目录】相对路径与审批统一以原项目为根，不继承宿主当前目录
    if requested is not None and requested.strip():
        return path_security.resolve_target(Path(requested.strip()), lease)
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
    }


def _check_terminal_permission(command: str, lease: Lease) -> ToolError | None:
    terminal = lease.capabilities.get("terminal")
    if not isinstance(terminal, Mapping):
        return None
    if terminal.get("enabled") is False:
        return ToolError(
            ToolErrorCategory.PERMISSION, "terminal_disabled", retryable=False
        )
    deny = [str(item) for item in terminal.get("deny_commands", []) if item]
    allow = [str(item) for item in terminal.get("allow_commands", []) if item]
    segments = _command_segments(command)
    for segment in segments:
        if _matches(segment, [*DEFAULT_DENY_COMMANDS, *deny]):
            return ToolError(
                ToolErrorCategory.PERMISSION,
                "terminal_command_denied",
                retryable=False,
                partial_state=f"denied segment: {segment}",
            )
    if allow:
        for segment in segments:
            if not _matches(segment, allow):
                return ToolError(
                    ToolErrorCategory.PERMISSION,
                    "terminal_command_not_allowed",
                    retryable=False,
                    partial_state=f"not allowed segment: {segment}",
                )
    return None


def _command_segments(command: str) -> list[str]:
    segments: list[str] = []
    quote = ""
    start = 0
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            if char == quote:
                quote = ""
        elif char in {"'", '"'}:
            quote = char
        elif char in _COMMAND_SEGMENT_SEPARATORS:
            segment = command[start:index].strip()
            if segment:
                segments.append(segment)
            if char in {"&", "|"} and index + 1 < len(command):
                if command[index + 1] == char:
                    index += 1
            start = index + 1
        index += 1
    tail = command[start:].strip()
    if tail:
        segments.append(tail)
    return segments or [command.strip()]


def _matches(command: str, patterns: list[str]) -> bool:
    candidates = _match_candidates(command)
    return any(
        fnmatch.fnmatch(first, pattern.lower())
        or fnmatch.fnmatch(candidate, pattern.lower())
        for pattern in patterns
        for candidate, first in candidates
    )


def _match_candidates(command: str) -> list[tuple[str, str]]:
    normalized = command.strip().lower()
    grouped = _strip_grouping_edges(normalized)
    unique = [normalized]
    if grouped and grouped != normalized:
        unique.append(grouped)
    return [
        (candidate, candidate.split(maxsplit=1)[0] if candidate else "")
        for candidate in unique
        if candidate
    ]


def _strip_grouping_edges(command: str) -> str:
    stripped = command.strip()
    while stripped.startswith("("):
        stripped = stripped[1:].lstrip()
    while stripped.endswith(")"):
        stripped = stripped[:-1].rstrip()
    return stripped


def _float_arg(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return default
