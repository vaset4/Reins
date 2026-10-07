from __future__ import annotations
from scripts.testing.llm import from_test_sequence

import sys
from pathlib import Path

import pytest

from app.run_task import RunTaskResponse
from app.run_task import run_task
from app.repl import run_repl
from app.startup import resolve_startup_identity
from app.repl.console import reset_console_for_tests
from llm.messages import TextPart
from runtime.lease import Lease
from runtime.run_facts import RunFactStore
from runtime.session_state import SessionStateStore
from runtime.session_messages import materialize_messages
from runtime.types import Trigger
from tasks.store import TaskStore
from triggers.user import make_run_context


def test_run_task_rejects_unknown_session_semantics(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        run_task("hello", tmp_path, source="chat")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        run_task("hello", tmp_path, keep_session_open=True)  # type: ignore[call-arg]


def test_run_task_uses_user_context_formal_task_identity(tmp_path: Path) -> None:
    response = run_task(
        "inspect workspace",
        tmp_path,
        session_id="session-test",
        run_id="run-test",
        data_root=tmp_path / ".reins" / "data",
    )

    facts = RunFactStore(tmp_path / ".reins" / "data").read_run("run-test")
    start = next(row for row in facts if row.get("event") == "run:start")
    assert response.task_id == start["task_id"]
    assert start["focus_task_id"] == response.task_id
    assert start["compatibility_task_id"] is None


def test_user_context_maps_existing_inbox_task_as_compatibility(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path)
    inbox = store.create_task("plain chat", task_id="chat-compat", is_inbox=True)

    context = make_run_context(
        "hello",
        data_root=tmp_path,
        task_id=inbox.task_id,
        formal_task=False,
        session_id="session-chat",
        run_id="run-chat",
        lease=Lease(),
    )

    assert context.trigger is Trigger.USER
    assert context.task_id is None
    assert context.focus_task_id is None
    assert context.compatibility_task_id == inbox.task_id
    assert context.storage_task_id == inbox.task_id
    messages = materialize_messages(tmp_path, "session-chat")
    assert messages[-1].kind == "user"
    assert (
        "".join(
            part.text for part in messages[-1].content if isinstance(part, TextPart)
        )
        == "hello"
    )


def test_repl_plain_prompt_uses_user_context_compatibility_identity(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    data_root = project / ".reins" / "data"
    data_root.mkdir(parents=True)
    client = from_test_sequence(['{"type":"final","content":"ok"}'])
    reset_console_for_tests()

    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt(["hello", "/exit"]),
    )

    inbox = TaskStore(data_root).get_inbox_tasks()
    assert len(inbox) == 1
    start = _first_run_start(data_root)
    assert start["task_id"] is None
    assert start["focus_task_id"] is None
    assert start["compatibility_task_id"] == inbox[0].task_id


def test_runtime_event_type_is_retired() -> None:
    import runtime.types as runtime_types

    assert not hasattr(runtime_types, "RuntimeEvent")


def test_cli_chat_passes_only_consumed_run_task_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app import cli

    captured: dict[str, object] = {}

    def fake_run_task(
        task: str, project_root: Path, **kwargs: object
    ) -> RunTaskResponse:
        captured["project_root"] = project_root
        captured.update(kwargs)
        return RunTaskResponse(
            task_id="task",
            run_id="run",
            segment_id="seg",
            status="done",
            output=task,
        )

    monkeypatch.setattr(cli, "build_llm_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "run_task", fake_run_task)
    monkeypatch.setattr(
        sys, "argv", ["reins", "chat", "--task", "hello", "--session", "s1"]
    )
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))

    assert cli.main() == 0
    assert captured["project_root"] == tmp_path.resolve()
    assert captured["session_id"] == "s1"
    assert captured["data_root"] == resolve_startup_identity().data_root
    assert "source" not in captured
    assert "keep_session_open" not in captured
    assert "run_tools" not in captured


def test_cli_resume_rejects_inspect_decision_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app import cli

    monkeypatch.setattr(cli, "build_llm_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        ["reins", "resume", "--checkpoint", "ck", "--decision", "skip"],
    )
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "resume --decision requires --execute" in capsys.readouterr().err


def test_project_root_env_prefers_reins_over_legacy_xiangmu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "reins-root"
    legacy = tmp_path / "xiangmu-root"
    identity = resolve_startup_identity(
        environ={
            "REINS_PROJECT_ROOT": str(canonical),
            "XIANGMU_PROJECT_ROOT": str(legacy),
        }
    )

    assert identity.project_root == canonical.resolve()


def _scripted_prompt(lines: list[str]):
    iterator = iter(lines)

    def fn(_label: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError()

    return fn


def _first_run_start(data_root: Path) -> dict[str, object]:
    """按正式会话和运行目录查询首条事实；参数：数据根；返回：真实run:start事实。"""
    session = SessionStateStore(data_root).list_recent()[0]
    rows = RunFactStore(data_root).read_session_run(
        session.session_id, session.last_run_id
    )
    return next(row for row in rows if row.get("event") == "run:start")
