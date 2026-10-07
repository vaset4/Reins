from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv
from collections.abc import Mapping, Sequence
from pathlib import Path
from zipfile import ZipFile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_TIMEOUT_SECONDS = 60
PDF_READING_PROBE = """
import sys
from pathlib import Path
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from runtime.types import ReadOnlyInspectionRequest
from tools.readonly_inspection import ReadOnlyInspectionExecutor

PAGE_WIDTH_POINTS, PAGE_HEIGHT_POINTS = 200, 200
PROBE_MAX_ENTRIES, PROBE_MAX_CHARS, PROBE_MAX_MATCHES = 10, 4000, 10
root = Path(sys.argv[1])
pdf_path = root / "packaging-text.pdf"
marker = "Reins PDF packaging contract"
writer = PdfWriter()
page = writer.add_blank_page(width=PAGE_WIDTH_POINTS, height=PAGE_HEIGHT_POINTS)
font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                         NameObject("/Subtype"): NameObject("/Type1"),
                         NameObject("/BaseFont"): NameObject("/Helvetica")})
page[NameObject("/Resources")] = DictionaryObject({
    NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
contents = DecodedStreamObject()
contents.set_data(f"BT /F1 12 Tf 20 150 Td ({marker}) Tj ET".encode("ascii"))
page.replace_contents(contents)
writer.write(pdf_path)
writer.close()
reader = ReadOnlyInspectionExecutor(root, PROBE_MAX_ENTRIES, PROBE_MAX_CHARS, PROBE_MAX_MATCHES)
result = reader.execute(ReadOnlyInspectionRequest("read_file", pdf_path.name))
assert result.status == "ok", result
assert marker in result.output, result
assert result.meta["page_count"] == 1, result
assert result.meta["file_type"] == "pdf", result
"""


def test_tui_extra_and_script_metadata() -> None:
    """验证发布元数据声明 TUI 的完整直接依赖和稳定入口

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：无
    返回：无；断言 TUI extra 和 reins-tui script 符合发布合同
    """
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))

    assert metadata["project"]["optional-dependencies"]["tui"] == [
        "Pillow>=10.0",
        "prompt-toolkit>=3.0",
        "pypdf>=6.9.1",
        "textual>=8.2.8,<9",
    ]
    assert metadata["project"]["scripts"]["reins-tui"] == "frontends.tui.main:main"
    package_patterns = metadata["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "reins_secrets*" in package_patterns
    assert "secrets*" not in package_patterns
    assert "*.tcss" in metadata["tool"]["setuptools"]["package-data"]["frontends.tui"]


@pytest.fixture
def outside_checkout():
    """建立仓库外工作目录；参数：无；返回：自动清理的系统临时目录。"""
    with tempfile.TemporaryDirectory(prefix="reins-wheel-outside-") as directory:
        yield Path(directory)


def test_pdf_reading_probe_extracts_text(tmp_path: Path) -> None:
    """单独验证安装探针确实生成并读取文字 PDF；参数：隔离目录；返回：无。"""
    _run_command(
        [sys.executable, "-c", PDF_READING_PROBE, str(tmp_path)],
        cwd=PROJECT_ROOT,
        env=_clean_subprocess_environment(),
    )


def test_tui_wheel_runs_outside_source_checkout(
    tmp_path: Path, outside_checkout: Path
) -> None:
    """验证隔离环境只安装 wheel 的 TUI extra 后可运行公开探针

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：tmp_path 提供 wheel 与 venv；outside_checkout 提供仓库外工作目录
    返回：无；断言公开帮助、PDF实际读取、全屏启动和后台断开均正常
    """
    test_tui_extra_and_script_metadata()
    prepared_wheel = os.environ.get("REINS_PACKAGING_WHEEL")
    wheel_path = (
        Path(prepared_wheel).resolve()
        if prepared_wheel
        else _build_project_wheel(tmp_path)
    )
    with ZipFile(wheel_path) as wheel:
        assert "frontends/tui/interactive.tcss" in wheel.namelist()
    venv_python, reins_tui, outside_root, clean_env = _install_tui_wheel(
        wheel_path, tmp_path, outside_checkout
    )
    help_result = _run_command(
        [str(reins_tui), "--help"], cwd=outside_root, env=clean_env
    )
    _run_command(
        [
            str(venv_python),
            "-c",
            "import approval.tui; import frontends.tui.interactive",
        ],
        cwd=outside_root,
        env=clean_env,
    )
    _run_command(
        [str(venv_python), "-c", PDF_READING_PROBE, str(outside_root)],
        cwd=outside_root,
        env=clean_env,
    )
    _run_command(
        [
            str(venv_python),
            "-c",
            (
                "import secrets; import reins_secrets.store; "
                "assert not hasattr(secrets, '__path__')"
            ),
        ],
        cwd=outside_root,
        env=clean_env,
    )

    assert "--data-root" in help_result.stdout
    _run_command(
        [
            str(venv_python),
            "-c",
            "import pathlib, frontends.tui.interactive as tui; "
            "assert pathlib.Path(tui.__file__).is_relative_to(pathlib.Path(__import__('sys').prefix)); "
            "assert pathlib.Path(tui.__file__).with_name('interactive.tcss').is_file()",
        ],
        cwd=outside_root,
        env=clean_env,
    )
    project = outside_root / "project"
    project.mkdir()
    profile = outside_root / "profile"
    profile.mkdir()
    clean_env.update(
        USERPROFILE=str(profile),
        REINS_PROJECT_ROOT=str(project),
        XIANGMU_LLM_API_KEY="isolated-test-key",
        XIANGMU_LLM_MODEL="packaging-test",
        XIANGMU_LLM_API_MODE="chat_completions",
        XIANGMU_LLM_BASE_URL="http://127.0.0.1:1/v1",
    )
    try:
        _run_command(
            [
                str(venv_python),
                "-c",
                "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')",
                str(PROJECT_ROOT / "tests/scripts/tui_process_driver.py"),
                "smoke",
            ],
            cwd=outside_root,
            env=clean_env,
        )
    finally:
        _run_command(
            [
                str(venv_python),
                "-c",
                "import sys; from pathlib import Path; from app.background.client import connect, is_running; "
                "root = Path(sys.argv[1]); assert not is_running(root) or connect(root).stop(timeout=15)",
                str(profile / ".reins/data"),
            ],
            cwd=outside_root,
            env=clean_env,
        )


def _build_project_wheel(tmp_path: Path) -> Path:
    """从隔离临时源码构建真实项目 wheel

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：tmp_path 提供临时源码和 wheel 输出目录
    返回：构建完成的 wheel 文件路径
    """
    source_root = tmp_path / "source"
    _stage_packaging_source(source_root)
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    _run_command(
        [
            sys.executable,
            "-c",
            (
                "import sys; from setuptools.build_meta import build_wheel; "
                "build_wheel(sys.argv[1])"
            ),
            str(wheel_dir),
        ],
        cwd=source_root,
    )
    return next(wheel_dir.glob("reins_agent-*.whl"))


def _install_tui_wheel(
    wheel_path: Path, tmp_path: Path, outside_root: Path
) -> tuple[Path, Path, Path, dict[str, str]]:
    """在独立 venv 中安装项目 wheel 并核对依赖

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：wheel_path 为发布产物；tmp_path 提供 venv；outside_root 为仓库外目录
    返回：venv Python、console script、仓库外目录和隔离环境变量
    """
    # 1. 依赖可在测试前装入专用隔离环境，避免联网安装占用测试硬超时
    prepared_venv = os.environ.get("REINS_PACKAGING_VENV")
    venv_root = Path(prepared_venv).resolve() if prepared_venv else tmp_path / "venv"
    if prepared_venv:
        assert (venv_root / "pyvenv.cfg").is_file(), (
            "REINS_PACKAGING_VENV must be a dedicated venv"
        )
        assert (
            "include-system-site-packages = false"
            in (venv_root / "pyvenv.cfg").read_text()
        )
    else:
        venv.EnvBuilder(with_pip=False).create(venv_root)
    venv_python = venv_root / "Scripts" / "python.exe"
    reins_tui = venv_root / "Scripts" / "reins-tui.exe"
    clean_env = _clean_subprocess_environment()

    _run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(venv_python),
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            "--prefer-binary",
            "--retries",
            "1",
            "--timeout",
            "5",
            *(["--no-deps", "--force-reinstall"] if prepared_venv else []),
            f"{wheel_path}[tui]",
        ],
        cwd=outside_root,
        env=clean_env,
    )
    # 2. 新 wheel 仍须满足完整依赖合同，预装环境不允许掩盖缺失依赖
    _run_command(
        [sys.executable, "-m", "pip", "--python", str(venv_python), "check"],
        cwd=outside_root,
        env=clean_env,
    )
    return venv_python, reins_tui, outside_root, clean_env


def _stage_packaging_source(destination: Path) -> None:
    """把 setuptools 声明的产品源码复制到隔离构建目录

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：destination 为临时源码根目录
    返回：无；目录只含 pyproject、顶层模块和显式产品包
    """
    destination.mkdir()
    shutil.copy2(PROJECT_ROOT / "pyproject.toml", destination / "pyproject.toml")
    shutil.copy2(PROJECT_ROOT / "path_security.py", destination / "path_security.py")
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    package_patterns = metadata["tool"]["setuptools"]["packages"]["find"]["include"]
    for pattern in package_patterns:
        package_name = str(pattern).removesuffix("*")
        shutil.copytree(
            PROJECT_ROOT / package_name,
            destination / package_name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """运行隔离发布探针并在失败时保留标准输出和错误输出

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：command 为命令参数；cwd 为仓库外目录；env 为可选环境变量
    返回：成功命令的 CompletedProcess，失败时由 subprocess 显式抛错
    """
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        capture_output=True,
        check=False,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        rendered_command = subprocess.list2cmdline(list(command))
        raise AssertionError(
            f"command failed ({result.returncode}): {rendered_command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _clean_subprocess_environment() -> dict[str, str]:
    """构造不继承源码搜索路径和用户 site-packages 的探针环境

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：无
    返回：移除 PYTHONPATH 并禁用用户 site-packages 的新环境映射
    """
    clean_env = dict(os.environ)
    clean_env.pop("PYTHONPATH", None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    return clean_env
