"""CLI resume inspect/execute 公开合同测试

作者：xxx
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final, from_test_sequence

import sys
from pathlib import Path

import pytest

from llm.client import MissingConfigurationLLMClient
from llm.messages import ToolCallPart
from runtime.checkpoint import Checkpoint, checkpoint_to_ledger_state, save_checkpoint
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_facts import RunFactStore
from runtime.schema_meta import ensure_current_schema
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tests.test_session_runtime import capture_requests


@pytest.fixture(autouse=True)
def isolated_user_home(tmp_path, monkeypatch):
    """隔离默认用户数据根；参数：临时目录及环境替换器；返回：无，不访问真实资料库。"""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


def test_inspect_resume_is_read_only(tmp_path: Path) -> None:
    """验证 inspect 只读取恢复点

    参数：tmp_path 为隔离的数据与项目根目录
    返回：无；断言运行事实和 Ledger 均未新增
    """
    from app.run_task import inspect_resume

    data_root = tmp_path / ".reins" / "data"
    checkpoint = _save_checkpoint(data_root)
    before_runs = _run_fact_snapshot(data_root)
    before_events = LedgerStore(data_root).read_events()

    response = inspect_resume(
        f"{checkpoint.task_id}::{checkpoint.checkpoint_id}",
        tmp_path,
    )

    assert response.run_id == ""
    assert response.segment_id == ""
    assert response.output.startswith("RESUME_READY")
    assert _run_fact_snapshot(data_root) == before_runs
    assert LedgerStore(data_root).read_events() == before_events


def test_execute_resume_uses_exact_requested_checkpoint(tmp_path: Path) -> None:
    """验证 execute 不会把指定 checkpoint 漂移为 latest

    参数：tmp_path 为隔离的数据与项目根目录
    返回：无；断言新 run 的 parent segment 指向首个 checkpoint
    """
    from app.run_task import execute_resume

    data_root = tmp_path / ".reins" / "data"
    first = _save_checkpoint(data_root, segment_id="source-first")
    _save_checkpoint(data_root, segment_id="source-latest")
    client = from_test_sequence(['{"type":"final","content":"resumed"}'])

    response = execute_resume(
        f"{first.task_id}::{first.checkpoint_id}",
        tmp_path,
        llm_client=client,
        tool_registry=build_tool_registry(repo_root=tmp_path, data_root=data_root),
    )

    facts = RunFactStore(data_root).read_run(response.run_id)
    start = next(row for row in facts if row.get("event") == "run:start")
    assert response.status == "done"
    assert response.output == "resumed"
    assert response.run_id.startswith("run-")
    assert response.segment_id.startswith("resume-")
    assert start["trigger"] == "resume"
    assert start["parent_segment_id"] == "source-first"


@pytest.mark.parametrize("decision", ["skip", "replay"])
def test_execute_resume_preference_reaches_model_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    """恢复偏好进入真实请求，旧写操作不会自动重做；传参：目录、替换器、偏好；返回：无。"""
    from app.run_task import execute_resume

    data_root = tmp_path / ".reins" / "data"
    checkpoint = _save_checkpoint(
        data_root,
        pending={"tool_name": "file_write", "args": {"path": "x"}, "call_id": "c1"},
    )
    client = from_test_sequence(['{"type":"final","content":"skipped"}'])
    requests = capture_requests(client, monkeypatch)

    response = execute_resume(
        checkpoint.checkpoint_id,
        tmp_path,
        decision=decision,
        llm_client=client,
        tool_registry=build_tool_registry(repo_root=tmp_path, data_root=data_root),
    )

    facts = RunFactStore(data_root).read_run(response.run_id)
    assert not any(row.get("event") == "tool:request" for row in facts)
    assert f"requested_resolution={decision}" in str(requests[0])
    assert "file_write" in str(requests[0])
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize(
    "case",
    [
        ("skip", None, "requires a pending tool"),
    ],
)
def test_execute_resume_rejects_inapplicable_decision_before_run(
    tmp_path: Path,
    case: tuple[str, dict[str, object] | None, str],
) -> None:
    """验证不适用 decision 在创建新 run 前失败

    参数：tmp_path 为隔离根；decision/pending/message 描述错误矩阵
    返回：无；断言错误稳定且无新 run facts
    """
    from app.run_task import execute_resume

    decision, pending, message = case
    data_root = tmp_path / ".reins" / "data"
    checkpoint = _save_checkpoint(data_root, pending=pending)
    before = _run_fact_snapshot(data_root)

    with pytest.raises(ValueError, match=message):
        execute_resume(
            checkpoint.checkpoint_id,
            tmp_path,
            decision=decision,
            llm_client=from_test_sequence([]),
            tool_registry=build_tool_registry(repo_root=tmp_path, data_root=data_root),
        )

    assert _run_fact_snapshot(data_root) == before


@pytest.mark.parametrize("decision", ["skip", "replay"])
def test_cli_rejects_decision_without_execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    decision: str,
) -> None:
    """验证 inspect 模式不能提交 pending decision

    参数：monkeypatch 替换 argv；tmp_path 为项目根；decision 为合法值
    返回：无；argparse 必须以退出码 2 拒绝
    """
    from app import cli

    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["reins", "resume", "--checkpoint", "ck", "--decision", decision],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_cli_rejects_retired_decision_value() -> None:
    """验证旧 approval scope 值不能进入 resume 合同

    参数：无
    返回：无；argparse 必须以退出码 2 拒绝 once
    """
    from app.cli import build_run_parser

    with pytest.raises(SystemExit) as exc_info:
        build_run_parser().parse_args(
            ["resume", "--checkpoint", "ck", "--execute", "--decision", "once"]
        )

    assert exc_info.value.code == 2


def test_cli_inspect_starts_with_resume_ready_and_skips_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 CLI inspect 不装配模型且首行就是恢复摘要

    参数：monkeypatch 注入 inspect；tmp_path 为项目根；capsys 捕获输出
    返回：无；断言退出 0、无 TASK 头且未创建 LLM client
    """
    from app import cli
    from app.run_task import RunTaskResponse

    response = RunTaskResponse(
        task_id="task",
        run_id="",
        segment_id="",
        status="paused",
        output="RESUME_READY\ncheckpoint: ck",
    )
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "inspect_resume", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(
        cli,
        "build_llm_client",
        lambda *_args, **_kwargs: pytest.fail("inspect must not build an LLM client"),
    )
    monkeypatch.setattr(sys, "argv", ["reins", "resume", "--checkpoint", "ck"])

    assert cli.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "RESUME_READY"
    assert all(not line.startswith("TASK ") for line in lines)


@pytest.mark.parametrize("case", [("done", 0), ("paused", 3), ("failed", 1)])
def test_cli_execute_maps_runtime_status_to_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: tuple[str, int],
) -> None:
    """验证 execute 的运行终态映射为稳定退出码

    参数：monkeypatch 注入 service；tmp_path 为项目根；status/expected 为映射
    返回：无；断言 done/paused/failed 对应 0/3/1
    """
    from app import cli
    from app.run_task import RunTaskResponse

    status, expected = case
    response = RunTaskResponse(
        task_id="task",
        run_id="run-new",
        segment_id="resume-new",
        status=status,
        output=status,
    )
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "build_llm_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "execute_resume", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(cli, "resolve_resume_identity", lambda *_args: ("ck", tmp_path))
    monkeypatch.setattr(
        sys, "argv", ["reins", "resume", "--checkpoint", "ck", "--execute"]
    )

    assert cli.main() == expected


def test_cli_execute_real_pause_returns_exit_3_without_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证真实 execute 遇到用户输入边界时暂停

    参数：monkeypatch 注入模型响应；tmp_path 为项目根；capsys 捕获输出
    返回：无；断言退出码 3、真实身份与 waiting_user 事实
    """
    from app import cli

    data_root = tmp_path / ".reins" / "data"
    checkpoint = _save_checkpoint(data_root)
    client = from_test_native_tool_then_final(
        [
            ToolCallPart("question-resume", "ask_user", {"question": "which file?"}),
        ],
        "等待回答后再继续",
    )
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "build_llm_client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(
        sys,
        "argv",
        ["reins", "resume", "--checkpoint", checkpoint.checkpoint_id, "--execute"],
    )

    assert cli.main() == 3
    output = capsys.readouterr()
    assert "status=paused" in output.out
    assert "RESUME_READY" not in output.out
    runs = RunFactStore(data_root).list_runs_for_task(checkpoint.task_id)
    facts = RunFactStore(data_root).read_run(runs[0].run_id)
    lifecycle = [row for row in facts if row.get("event") == "run:lifecycle"][-1]
    assert lifecycle["lifecycle"] == "waiting_user"
    assert not any(row.get("event") == "approval:required" for row in facts)
    calls = [row["tool"]["name"] for row in facts if row.get("event") == "tool:request"]
    assert calls == ["ask_user"]


def test_cli_execute_provider_missing_fails_without_success_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 provider 配置缺失保留失败 run 但不输出成功头

    参数：monkeypatch 注入缺失配置的 client；tmp_path 为项目根；capsys 捕获输出
    返回：无；断言退出码 1、stderr 错误与 failed lifecycle
    """
    from app import cli

    data_root = tmp_path / ".reins" / "data"
    checkpoint = _save_checkpoint(data_root)
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        cli,
        "build_llm_client",
        lambda *_args, **_kwargs: MissingConfigurationLLMClient(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["reins", "resume", "--checkpoint", checkpoint.checkpoint_id, "--execute"],
    )

    assert cli.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "RESUME_ERROR" in output.err
    assert "MODEL_PROVIDER_ERROR: missing provider configuration" in output.err
    runs = RunFactStore(data_root).list_runs_for_task(checkpoint.task_id)
    assert runs[0].status == "failed"


def test_cli_resume_service_error_has_no_success_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 service 错误只输出稳定错误前缀

    参数：monkeypatch 注入缺失 checkpoint；tmp_path 为项目根；capsys 捕获输出
    返回：无；断言退出 1 且 stdout 没有成功头
    """
    from app import cli

    def missing_checkpoint(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("ck")

    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "inspect_resume", missing_checkpoint)
    monkeypatch.setattr(sys, "argv", ["reins", "resume", "--checkpoint", "ck"])

    assert cli.main() == 1
    output = capsys.readouterr()
    assert "RESUME_ERROR" in output.err
    assert "TASK " not in output.out
    assert "RESUME_READY" not in output.out


def _save_checkpoint(
    data_root: Path,
    *,
    segment_id: str = "source-segment",
    pending: dict[str, object] | None = None,
) -> Checkpoint:
    """创建可被生产 resolver 精确读取的 checkpoint

    参数：data_root 为数据根；segment_id 为来源段；pending 为待决工具
    返回：已写入 Ledger 的 checkpoint
    """
    ensure_current_schema(data_root)
    WorkspaceStore(data_root).bind_session("session-source", data_root.parent.parent)
    task_id = "task-resume-contract"
    store = TaskStore(data_root)
    if store.load_task(task_id) is None:
        store.create_task("resume contract", task_id=task_id)
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id=task_id,
            session_id="session-source",
            run_id="run-source",
            segment_id=segment_id,
            state="paused",
            pending_tool_call=pending,
            reason="test pause",
        )
    )
    LedgerWriter(
        LedgerStore(data_root), source="tests.cli_resume"
    ).record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        checkpoint.reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )
    return checkpoint


def _run_fact_snapshot(data_root: Path) -> dict[str, bytes]:
    """读取当前全部 run facts 作为副作用快照

    参数：data_root 为数据根目录
    返回：相对路径到原始内容的不可变比较映射
    """
    sessions = data_root / "sessions"
    if not sessions.exists():
        return {}
    return {
        path.relative_to(data_root).as_posix(): path.read_bytes()
        for path in sessions.glob("*/runs/*/facts.jsonl")
    }
