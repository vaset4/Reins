from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_with_timeout.py"
TIMEOUT_EXIT_CODE = 124


def _run_wrapper(
    *command: str,
    seconds: float = 5,
) -> subprocess.CompletedProcess[str]:
    """通过公开 CLI 运行 timeout wrapper。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：command 为分隔符后原样传递的命令参数；seconds 为允许运行的秒数
    返回：包含退出码和标准流的已完成进程结果
    """
    return subprocess.run(
        [
            sys.executable,
            str(TIMEOUT_SCRIPT),
            "--seconds",
            str(seconds),
            "--",
            sys.executable,
            *command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_wrapper_preserves_command_argv_and_success_exit_code() -> None:
    """正常命令应收到原始 argv，并向调用方返回退出码零。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：无
    返回：无
    """
    expected_args = ["value with spaces", "--literal-option", "a&b"]
    result = _run_wrapper(
        "-c",
        "import json, sys; print(json.dumps(sys.argv[1:]))",
        *expected_args,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == expected_args


def test_wrapper_preserves_nonzero_exit_code() -> None:
    """失败命令的非零退出码应原样返回。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：无
    返回：无
    """
    result = _run_wrapper("-c", "import sys; sys.exit(7)")

    assert result.returncode == 7
    assert "COMMAND_TIMEOUT" not in result.stderr


def test_wrapper_times_out_with_stable_error() -> None:
    """超过硬超时的命令应被终止并返回稳定错误。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：无
    返回：无
    """
    started_at = time.monotonic()
    result = _run_wrapper("-c", "import time; time.sleep(10)", seconds=0.2)

    assert result.returncode == TIMEOUT_EXIT_CODE
    assert "COMMAND_TIMEOUT" in result.stderr
    assert time.monotonic() - started_at < 5


def test_wrapper_terminates_grandchild_process(tmp_path: Path) -> None:
    """超时时孙进程不得存活到写出延迟哨兵文件。

    作者：xxx
    时间：2026-08-18 00:00:00
    传参：tmp_path 为 pytest 提供的隔离临时目录
    返回：无
    """
    sentinel = tmp_path / "grandchild-survived.txt"
    grandchild_code = (
        "import pathlib, sys, time; "
        "time.sleep(1); "
        "pathlib.Path(sys.argv[1]).write_text('survived', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
        "time.sleep(10)"
    )

    result = _run_wrapper(
        "-c",
        parent_code,
        grandchild_code,
        str(sentinel),
        seconds=0.2,
    )
    time.sleep(1.2)

    assert result.returncode == TIMEOUT_EXIT_CODE
    assert "COMMAND_TIMEOUT" in result.stderr
    assert not sentinel.exists()
