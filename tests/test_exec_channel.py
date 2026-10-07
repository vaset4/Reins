from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tools.exec_channel import (
    ExecChannel,
    ExecResult,
    ExecSpec,
    LocalSubprocessBackend,
    redact_secrets,
    strip_secret_env,
)


def _channel() -> ExecChannel:
    return ExecChannel(LocalSubprocessBackend())


def test_run_python_returns_stdout_and_exit_code(tmp_path: Path) -> None:
    result = _channel().run(
        ExecSpec(code='print("hi")', kind="py", cwd=tmp_path, timeout_seconds=10)
    )

    assert isinstance(result, ExecResult)
    assert result.stdout.strip() == "hi"
    assert result.exit_code == 0
    assert result.timed_out is False


def test_run_python_nonzero_exit_on_exception(tmp_path: Path) -> None:
    result = _channel().run(
        ExecSpec(
            code='raise RuntimeError("bad")',
            kind="py",
            cwd=tmp_path,
            timeout_seconds=10,
        )
    )

    assert result.exit_code != 0
    assert "RuntimeError" in result.stderr


def test_run_shell_branch_executes_command(tmp_path: Path) -> None:
    result = _channel().run(
        ExecSpec(code="echo shell-ok", kind="shell", cwd=tmp_path, timeout_seconds=10)
    )

    assert result.exit_code == 0
    assert "shell-ok" in result.stdout


def test_shell_preserves_quotes_in_python_command(tmp_path: Path) -> None:
    """Windows命令内的引号原样进入解释器；参数：隔离目录；返回：无。"""
    command = f'"{sys.executable}" -c "print(12 * 3)"'
    result = _channel().run(
        ExecSpec(code=command, kind="shell", cwd=tmp_path, timeout_seconds=10)
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "36"


def test_shell_enters_quoted_directory_and_reads_relative_file(tmp_path: Path) -> None:
    """含空格的绝对目录可切换并读取相对文件；参数：隔离目录；返回：无。"""
    directory = tmp_path / "quoted workspace"
    directory.mkdir()
    (directory / "result.txt").write_text("verified-result", encoding="utf-8")
    command = f'cd /d "{directory}" && type result.txt'
    result = _channel().run(
        ExecSpec(code=command, kind="shell", cwd=tmp_path, timeout_seconds=10)
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "verified-result"


def test_run_python_cwd_is_the_spec_cwd(tmp_path: Path) -> None:
    result = _channel().run(
        ExecSpec(
            code="import os; print(os.getcwd())",
            kind="py",
            cwd=tmp_path,
            timeout_seconds=10,
        )
    )

    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


@pytest.mark.skipif(
    os.name != "nt", reason="Windows CreateProcess working-directory boundary"
)
def test_python_executes_in_long_directory_with_relative_files_and_imports(
    tmp_path: Path,
) -> None:
    """深层版本目录保持真实工作位置、相对资源和同目录导入；传参：隔离目录；返回：无。"""
    windows_path_limit = 260
    directory = tmp_path
    while len(str(directory)) <= windows_path_limit:
        directory /= "嵌套 工作目录"
    directory.mkdir(parents=True)
    (directory / "sibling.py").write_text("VALUE = 'sibling-ok'\n", encoding="utf-8")
    (directory / "input.txt").write_text("relative-ok", encoding="utf-8")
    result = _channel().run(
        ExecSpec(
            code="import json, sibling\nfrom pathlib import Path\n"
            "print(json.dumps({'cwd': str(Path.cwd()), 'resource': Path('input.txt').read_text(), 'module': sibling.VALUE}))",
            kind="py",
            cwd=directory,
            timeout_seconds=10,
        )
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["cwd"]).resolve() == directory.resolve()
    assert payload["resource"] == "relative-ok" and payload["module"] == "sibling-ok"
    assert sorted(path.name for path in directory.glob("*.py")) == ["sibling.py"]


def test_run_python_does_not_leave_temp_script(tmp_path: Path) -> None:
    _channel().run(
        ExecSpec(code='print("x")', kind="py", cwd=tmp_path, timeout_seconds=10)
    )

    leftover = list(tmp_path.glob("*.py"))
    assert leftover == []


def test_timeout_kills_and_flags(tmp_path: Path) -> None:
    result = _channel().run(
        ExecSpec(
            code="import time; time.sleep(30)",
            kind="py",
            cwd=tmp_path,
            timeout_seconds=1,
        )
    )

    assert result.timed_out is True
    assert result.exit_code != 0


def test_strip_secret_env_removes_blacklisted_names() -> None:
    source = {
        "PATH": "/bin",
        "MY_API_KEY": "secret",
        "AUTH_TOKEN": "t",
        "DB_PASSWORD": "p",
        "SOME_SECRET": "s",
        "X_CREDENTIAL": "c",
        "HOME": "/home",
    }

    cleaned = strip_secret_env(source)

    assert "PATH" in cleaned
    assert "HOME" in cleaned
    assert "MY_API_KEY" not in cleaned
    assert "AUTH_TOKEN" not in cleaned
    assert "DB_PASSWORD" not in cleaned
    assert "SOME_SECRET" not in cleaned
    assert "X_CREDENTIAL" not in cleaned


def test_redact_secrets_masks_known_values() -> None:
    text = "the token is supersecretvalue123 here"

    redacted = redact_secrets(text, ["supersecretvalue123"])

    assert "supersecretvalue123" not in redacted
    assert "the token is" in redacted


def test_subprocess_does_not_receive_secret_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LEAKY_API_KEY", "should-not-be-visible")
    result = _channel().run(
        ExecSpec(
            code="import os; print(os.environ.get('LEAKY_API_KEY', 'ABSENT'))",
            kind="py",
            cwd=tmp_path,
            timeout_seconds=10,
        )
    )

    assert result.stdout.strip() == "ABSENT"


def test_output_with_secret_value_is_redacted(tmp_path: Path) -> None:
    result = _channel().run(
        ExecSpec(
            code="print('leak: topsecret99')",
            kind="py",
            cwd=tmp_path,
            timeout_seconds=10,
            redact_values=["topsecret99"],
        )
    )

    assert "topsecret99" not in result.stdout


def test_output_auto_redacts_secret_env_values(tmp_path: Path, monkeypatch) -> None:
    # 输出脱敏在生产生效（无需调用方显式喂值，AC4 / R5 / C16）：backend 自动脱敏它
    # 剥离的 secret 命名 env 变量的值——脚本即便经其他渠道拿到该值并打印，回传前也打码。
    monkeypatch.setenv("LEAKY_API_KEY", "topsecretenvvalue123")
    result = _channel().run(
        ExecSpec(
            code="print('leaked: topsecretenvvalue123')",
            kind="py",
            cwd=tmp_path,
            timeout_seconds=10,
        )
    )

    assert "topsecretenvvalue123" not in result.stdout
    assert "[REDACTED]" in result.stdout


def test_output_auto_redaction_skips_short_secret_values(
    tmp_path: Path, monkeypatch
) -> None:
    # 长度护栏：短的 secret 命名配置值（如 AUTH_FLAG=ok）不自动脱敏，避免误打码污染
    # 正常输出；显式 redact_values 不受此限。
    monkeypatch.setenv("AUTH_FLAG", "ok")
    result = _channel().run(
        ExecSpec(
            code="print('status ok done')", kind="py", cwd=tmp_path, timeout_seconds=10
        )
    )

    assert "ok" in result.stdout


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        ExecSpec(code="x", kind="ruby", cwd=Path("."), timeout_seconds=1)  # type: ignore[arg-type]
