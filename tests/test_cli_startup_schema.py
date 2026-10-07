from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from app.startup import StartupIdentity
from llm.client import MissingConfigurationLLMClient
from runtime.schema_meta import ensure_current_schema


@pytest.fixture(autouse=True)
def isolated_user_home(tmp_path, monkeypatch):
    """隔离默认用户资料库；参数：临时目录与环境替换器；返回：无。"""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


def test_cli_initializes_schema_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 CLI 在构造模型和运行任务前初始化 schema

    参数：monkeypatch 隔离 CLI 依赖；tmp_path 提供项目根
    返回：无；断言运行调用看到 current meta
    """
    from app import cli
    from app.run_task import RunTaskResponse

    data_root = tmp_path / ".reins" / "data"

    def fake_run_task(*_args: object, **_kwargs: object) -> RunTaskResponse:
        assert ensure_current_schema(data_root).status == "current"
        return RunTaskResponse(
            task_id="task",
            run_id="run",
            segment_id="segment",
            status="done",
            output="ok",
        )

    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "build_llm_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "run_task", fake_run_task)
    monkeypatch.setattr(sys, "argv", ["reins", "--task", "hello"])

    assert cli.main() == 0


def test_cli_rejects_nonempty_root_before_model_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 CLI 对非空缺 meta 数据在模型装配前失败

    参数：monkeypatch 设置入口；tmp_path 为项目根；capsys 捕获错误
    返回：无；断言退出 1、稳定 code 与零 meta 写入
    """
    from app import cli

    data_root = tmp_path / ".reins" / "data"
    (data_root / "tasks").mkdir(parents=True)
    (data_root / "tasks" / "old.yaml").write_text("old", encoding="utf-8")
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        cli,
        "build_llm_client",
        lambda *_args, **_kwargs: pytest.fail("schema gate must run first"),
    )
    monkeypatch.setattr(sys, "argv", ["reins", "--task", "hello"])

    assert cli.main() == 1
    error = capsys.readouterr().err
    assert "SCHEMA_UNSUPPORTED[DATABASE_UNSUPPORTED]" in error
    assert str(data_root) not in error
    assert not (data_root / "reins.db").exists()


def test_tui_rejects_nonempty_root_before_model_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 TUI 使用同一 schema gate 且不进入模型装配

    参数：monkeypatch 注入启动身份；tmp_path 提供 data root；capsys 捕获错误
    返回：无；断言退出 1 且不写 meta
    """
    import frontends.tui.main as tui_main

    data_root = tmp_path / "tui-data"
    (data_root / "sessions").mkdir(parents=True)
    (data_root / "sessions" / "old.jsonl").write_text("old", encoding="utf-8")
    identity = StartupIdentity(tmp_path, data_root)
    monkeypatch.setattr(
        tui_main, "resolve_startup_identity", lambda **_kwargs: identity
    )
    monkeypatch.setattr(
        tui_main,
        "build_llm_client",
        lambda *_args, **_kwargs: pytest.fail("schema gate must run first"),
    )

    assert tui_main.run(data_root=data_root) == 1
    assert "SCHEMA_UNSUPPORTED[DATABASE_UNSUPPORTED]" in capsys.readouterr().err
    assert not (data_root / "reins.db").exists()


def test_gateway_reports_nonempty_root_as_structured_startup_error(
    tmp_path: Path,
) -> None:
    """验证 Gateway 将 schema 拒绝呈现为结构化启动错误

    参数：tmp_path 提供隔离 data root
    返回：无；断言退出 1、错误 code 和原数据不变
    """
    from app.gateway import run_gateway_stdio

    data_root = tmp_path / "gateway-data"
    (data_root / "tasks").mkdir(parents=True)
    (data_root / "tasks" / "old.yaml").write_text("old", encoding="utf-8")
    output = StringIO()

    code = run_gateway_stdio(
        identity=StartupIdentity(tmp_path, data_root),
        llm_client=MissingConfigurationLLMClient(),
        input_stream=StringIO(""),
        output_stream=output,
    )

    assert code == 1
    payload = json.loads(output.getvalue())
    assert payload["status"] == "error"
    assert payload["data"] == {
        "error_type": "unsupported_schema",
        "schema_code": "DATABASE_UNSUPPORTED",
    }
    assert "SCHEMA_UNSUPPORTED[DATABASE_UNSUPPORTED]" in payload["output"]
    assert not (data_root / "reins.db").exists()
