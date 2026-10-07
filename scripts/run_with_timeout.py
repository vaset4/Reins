"""为 CI 命令提供可复现的硬超时和进程树清理。"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence

TIMEOUT_EXIT_CODE = 124


@dataclass(frozen=True, slots=True)
class CommandConfig:
    """保存待执行命令及其超时秒数。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：seconds 为正数超时秒数；command 为原始命令参数
    返回：不可变命令配置对象
    """

    seconds: float
    command: tuple[str, ...]


def _positive_seconds(value: str) -> float:
    """解析正数超时参数。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：value 为命令行传入的秒数字符串
    返回：大于零且有限的浮点秒数；非法值抛 argparse.ArgumentTypeError
    """
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("--seconds must be a finite positive number")
    return seconds


def _parse_args(argv: Sequence[str]) -> CommandConfig:
    """解析 wrapper 参数并保留分隔符后的原始命令参数。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：argv 为不包含脚本名的命令行参数
    返回：包含超时和原始命令参数的不可变配置
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", required=True, type=_positive_seconds)
    raw_args = list(argv)
    if "--" not in raw_args:
        parser.error("command must follow the -- separator")
    separator_index = raw_args.index("--")
    options = parser.parse_args(raw_args[:separator_index])
    command = tuple(raw_args[separator_index + 1 :])
    if not command:
        parser.error("command must not be empty")
    return CommandConfig(seconds=options.seconds, command=command)


def _start_command(config: CommandConfig) -> subprocess.Popen[bytes]:
    """在独立的 Windows 进程组或 POSIX session 中启动命令。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：config 为待执行命令及超时配置
    返回：已启动且继承当前标准流的子进程
    """
    if sys.platform == "win32":
        return subprocess.Popen(
            config.command,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(config.command, start_new_session=True)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """终止超时命令及其创建的所有后代进程。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：process 为已超时的命令进程
    返回：无；Windows 无法终止仍在运行的进程树时抛 RuntimeError
    """
    if sys.platform == "win32":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0 and process.poll() is None:
            raise RuntimeError(
                f"failed to terminate process tree for pid {process.pid}"
            )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run(config: CommandConfig) -> int:
    """执行命令并透传退出码，超时时清理进程树。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：config 为待执行命令及超时配置
    返回：命令退出码，或稳定的超时退出码 124
    """
    process = _start_command(config)
    try:
        return process.wait(timeout=config.seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        print("COMMAND_TIMEOUT", file=sys.stderr, flush=True)
        return TIMEOUT_EXIT_CODE


def main(argv: Sequence[str] | None = None) -> int:
    """运行命令行 wrapper。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：argv 为可选命令行参数；为空时读取当前进程参数
    返回：被执行命令或 wrapper 的退出码
    """
    return run(_parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
