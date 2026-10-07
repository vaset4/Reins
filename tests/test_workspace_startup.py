"""独立工作区启动与真实文件工具范围回归。"""

import os
import shutil
import subprocess
import sys


def test_tui_uses_current_workspace_and_separate_data_root(tmp_path, monkeypatch):
    """TUI将当前目录传给模型与界面，单独data-root不改变工作区；参数：隔离目录和替身；返回：无。"""
    from frontends.tui import main as tui

    workspace = tmp_path / "papers"
    workspace.mkdir()
    data = tmp_path / "runtime"
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("REINS_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("XIANGMU_PROJECT_ROOT", raising=False)
    roots = []

    def model(_options, *, project_root):
        """记录模型配置的真实工作区；参数：选项和项目根；返回：隔离客户端。"""
        roots.append(project_root)
        return object()

    def run(**options):
        """核对正式TUI接收到的启动身份；参数：入口参数；返回：成功退出码。"""
        assert options["project_root"] == workspace
        assert options["data_root"] == data
        return 0

    monkeypatch.setattr(tui, "build_llm_client", model)
    monkeypatch.setattr(tui, "run_interactive_tui", run)
    assert tui.main(["--data-root", str(data)]) == 0
    assert roots == [workspace]


def test_cli_background_and_tools_use_launch_directory(tmp_path):
    """临时论文目录启动后台，读取本地文件而非安装源码；参数：隔离目录；返回：无。"""
    workspace = tmp_path / "papers"
    workspace.mkdir()
    (workspace / "paper.txt").write_text("workspace-paper", encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"REINS_PROJECT_ROOT", "XIANGMU_PROJECT_ROOT", "PYTHONPATH"}
    }
    env["USERPROFILE"] = str(tmp_path / "profile")
    env["PYTHONUTF8"] = "1"
    tui_command = shutil.which("reins-tui")
    assert tui_command, "请先安装项目的tui extra"
    result = subprocess.run(
        [tui_command, "--help"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    script = """
from app.startup import resolve_startup_identity
from tools.builtin_tools import build_tool_registry
from pathlib import Path
identity = resolve_startup_identity()
assert identity.project_root == Path.cwd()
registry = build_tool_registry(repo_root=identity.project_root, data_root=identity.data_root)
listing = registry.get('list').executor({'path': '.'})
assert 'paper.txt' in str(listing)
assert 'AGENTS.md' not in str(listing)
reading = registry.get('file_read').executor({'path': 'paper.txt'})
assert 'workspace-paper' in str(reading)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    try:
        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "background", "start"],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "profile/.reins/data/space.json").is_file()
        assert (tmp_path / "profile/.reins/data/index.sqlite").is_file()
        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "background", "status"],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    finally:
        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "background", "stop"],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
