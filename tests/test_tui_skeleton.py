from __future__ import annotations

from pathlib import Path

import pytest

from app.startup import StartupIdentity
import frontends.tui.main as tui_main


def test_run_delegates_to_interactive_tui(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "custom-data"
    llm_client = object()
    calls: dict[str, object] = {}

    def fake_build_llm_client(
        overrides: dict[str, object],
        *,
        project_root: Path | None = None,
    ) -> object:
        calls["llm_overrides"] = overrides
        calls["llm_project_root"] = project_root
        return llm_client

    def fake_run_interactive_tui(
        *,
        project_root: Path,
        data_root: Path,
        llm_client: object,
        startup_error: str,
    ) -> int:
        calls["startup_error"] = startup_error
        calls["repl_project_root"] = project_root
        calls["repl_data_root"] = data_root
        calls["repl_client"] = llm_client
        return 0

    monkeypatch.setattr(
        tui_main,
        "resolve_startup_identity",
        lambda **_kwargs: StartupIdentity(project_root, data_root),
    )
    monkeypatch.setattr(tui_main, "build_llm_client", fake_build_llm_client)
    monkeypatch.setattr(
        tui_main,
        "run_interactive_tui",
        fake_run_interactive_tui,
    )

    assert tui_main.run(data_root=data_root) == 0
    assert calls == {
        "startup_error": "",
        "llm_overrides": {},
        "llm_project_root": project_root,
        "repl_project_root": project_root,
        "repl_data_root": data_root,
        "repl_client": llm_client,
    }


def test_run_uses_default_reins_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    expected_data_root = project_root / ".reins" / "data"
    calls: dict[str, object] = {}

    def fake_run_interactive_tui(
        *,
        project_root: Path,
        data_root: Path,
        llm_client: object,
        startup_error: str,
    ) -> int:
        del project_root, llm_client
        assert startup_error == ""
        calls["data_root"] = data_root
        return 0

    monkeypatch.setattr(
        tui_main,
        "resolve_startup_identity",
        lambda **_kwargs: StartupIdentity(project_root, expected_data_root),
    )
    monkeypatch.setattr(tui_main, "build_llm_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(tui_main, "run_interactive_tui", fake_run_interactive_tui)

    assert tui_main.run() == 0
    assert calls == {"data_root": expected_data_root}


def test_main_passes_data_root_to_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证脚本参数只转发给共享 TUI 启动函数

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：monkeypatch 用于替换启动函数；tmp_path 提供显式 data root
    返回：无；断言 main 返回 run 的退出码并原样转发路径
    """
    data_root = tmp_path / "data"
    calls: list[Path | str | None] = []

    def fake_run(*, data_root: Path | str | None = None) -> int:
        calls.append(data_root)
        return 7

    monkeypatch.setattr(tui_main, "run", fake_run)

    assert tui_main.main(["--data-root", str(data_root)]) == 7
    assert calls == [data_root]


def test_main_help_exits_without_starting_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证帮助参数不创建运行时身份或进入交互循环

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：monkeypatch 用于监控共享启动函数
    返回：无；断言 argparse 以 0 退出且 run 未被调用
    """
    calls: list[None] = []

    def fake_run(*, data_root: Path | str | None = None) -> int:
        del data_root
        calls.append(None)
        return 0

    monkeypatch.setattr(tui_main, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        tui_main.main(["--help"])

    assert exc_info.value.code == 0
    assert calls == []
