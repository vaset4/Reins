from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.cli import build_llm_client
from app.startup import resolve_startup_identity
from frontends.tui.interactive import run_interactive_tui
from runtime.schema_meta import UnsupportedSchemaError, ensure_current_schema


def main(argv: Sequence[str] | None = None) -> int:
    """解析公开脚本参数并启动终端 TUI

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：argv 为可选命令行参数序列；None 时由 argparse 读取进程参数
    返回：共享 TUI 启动函数返回的整数退出码
    """
    parser = argparse.ArgumentParser(description="Launch the Reins terminal TUI.")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Path to the Reins runtime data directory.",
    )
    args = parser.parse_args(argv)
    return run(data_root=args.data_root)


def run(*, data_root: Path | str | None = None) -> int:
    """使用共享启动身份启动终端 TUI

    参数：data_root 为可选显式运行数据根
    返回：终端 TUI 退出码
    """
    identity = resolve_startup_identity(data_root=data_root)
    # 【Schema Startup】【TUI 入口】拒绝未知旧数据后才允许装配模型与交互循环
    try:
        ensure_current_schema(identity.data_root)
    except UnsupportedSchemaError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    startup_error = ""
    try:
        llm_client = build_llm_client({}, project_root=identity.project_root)
    except Exception as exc:
        llm_client = None
        startup_error = f"所属工作区模型配置不可用：{exc}"
    return run_interactive_tui(
        project_root=identity.project_root,
        data_root=identity.data_root,
        llm_client=llm_client,
        startup_error=startup_error,
    )


__all__ = ["main", "run"]
