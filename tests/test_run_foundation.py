from __future__ import annotations

from pathlib import Path


def test_run_task_persists_task_and_records_facts(tmp_path: Path) -> None:
    from app.run_task import run_task
    from runtime.run_facts import RunFactStore
    from tasks.store import TaskStore

    response = run_task(
        task="inspect workspace",
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
    )

    assert response.status == "failed"
    assert response.output.startswith(
        "MODEL_PROVIDER_ERROR: missing provider configuration"
    )

    data_root = tmp_path / ".reins" / "data"
    task = TaskStore(data_root).load_task(response.task_id)
    assert task is not None
    assert task.goal == "inspect workspace"
    # 一次运行的终态不再改写正式目标的状态：目标只由目标操作推进，
    # 失败的运行留下的是一个仍然打开、可以重试的目标
    assert task.status == "active"

    runs = RunFactStore(data_root).list_runs_for_task(response.task_id)
    assert runs
    facts = RunFactStore(data_root).read_run(runs[0].run_id)
    assert any(fact.get("event") == "run:start" for fact in facts)
    # 运行自己的失败仍然要留痕，只是痕迹落在运行事实而不是目标状态上
    lifecycle = [fact for fact in facts if fact.get("event") == "run:lifecycle"][-1]
    assert lifecycle["lifecycle"] == "failed"
    # checkpoints/ are no longer written; trajectory.jsonl still emitted by
    # watchdog/lease writers — those move to facts in commit (b).
    assert not (data_root / "tasks" / response.task_id / "checkpoints").exists()


def test_inspect_resume_uses_task_checkpoint(tmp_path: Path) -> None:
    from app.run_task import inspect_resume, run_task
    from runtime.checkpoint import list_checkpoints

    initial = run_task(
        task="inspect workspace",
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
    )
    checkpoints = list_checkpoints(
        initial.task_id,
        data_root=tmp_path / ".reins" / "data",
    )
    assert checkpoints

    resumed = inspect_resume(
        checkpoint_id=f"{initial.task_id}::{checkpoints[-1].checkpoint_id}",
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
    )

    assert resumed.status == "failed"
    assert "RESUME_READY" in resumed.output
    assert "source: task checkpoint" in resumed.output
    assert f"storage_task: {initial.task_id}" in resumed.output


def test_inspect_resume_uses_requested_legacy_task_checkpoint_state(
    tmp_path: Path,
) -> None:
    from app.run_task import inspect_resume
    from runtime.checkpoint import Checkpoint, save_checkpoint
    from runtime.run_facts import RunFactStore

    data_root = tmp_path / ".reins" / "data"
    store = RunFactStore(data_root)
    first = save_checkpoint(
        Checkpoint(
            task_id="legacy-task",
            session_id="session-legacy",
            run_id="run-paused",
            segment_id="user-1",
            state="PAUSED",
        )
    )
    store.append_checkpoint_ref(
        session_id="session-legacy",
        run_id="run-paused",
        task_id="legacy-task",
        focus_task_id="legacy-task",
        compatibility_task_id=None,
        segment_id="user-1",
        checkpoint=first,
    )
    _record_ledger_checkpoint(data_root, first)
    second = save_checkpoint(
        Checkpoint(
            task_id="legacy-task",
            session_id="session-legacy",
            run_id="run-done",
            segment_id="user-2",
            state="DONE",
        )
    )
    store.append_checkpoint_ref(
        session_id="session-legacy",
        run_id="run-done",
        task_id="legacy-task",
        focus_task_id="legacy-task",
        compatibility_task_id=None,
        segment_id="user-2",
        checkpoint=second,
    )
    _record_ledger_checkpoint(data_root, second)

    resumed = inspect_resume(
        checkpoint_id=f"legacy-task::{first.checkpoint_id}",
        project_root=tmp_path,
        data_root=data_root,
    )

    assert resumed.status == "paused"
    assert f"checkpoint: {first.checkpoint_id} (PAUSED)" in resumed.output


def test_inspect_resume_accepts_session_or_run_checkpoint_identity(
    tmp_path: Path,
) -> None:
    from app.run_task import inspect_resume
    from runtime.checkpoint import Checkpoint, save_checkpoint
    from runtime.run_facts import RunFactStore

    data_root = tmp_path / ".reins" / "data"
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id="chat-compat",
            session_id="session-abc",
            run_id="run-old",
            compatibility_task_id="chat-compat",
            segment_id="user-1",
            state="PAUSED",
        )
    )
    RunFactStore(data_root).append_checkpoint_ref(
        session_id="session-abc",
        run_id="run-old",
        task_id=None,
        focus_task_id=None,
        compatibility_task_id="chat-compat",
        segment_id="user-1",
        checkpoint=checkpoint,
    )
    _record_ledger_checkpoint(data_root, checkpoint)

    by_session = inspect_resume(
        checkpoint_id="session-abc",
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
    )
    by_run = inspect_resume(
        checkpoint_id="run-old",
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
    )

    assert "source: session" in by_session.output
    assert "session: session-abc" in by_session.output
    assert "run: run-old" in by_session.output
    assert "compatibility_task: chat-compat" in by_session.output
    assert "source: run" in by_run.output
    assert "session: session-abc" in by_run.output
    assert "run: run-old" in by_run.output


def test_inspect_resume_maps_checkpoint_states_from_ledger(tmp_path: Path) -> None:
    from app.run_task import inspect_resume
    from runtime.checkpoint import Checkpoint, save_checkpoint

    data_root = tmp_path / ".reins" / "data"
    cases = [
        ("session-wait-user", "WAITING_USER", "paused"),
        ("session-wait-tool", "pre_tool", "paused"),
        ("session-done", "DONE", "done"),
        ("session-failed", "FAILED", "failed"),
    ]
    for session_id, state, expected_status in cases:
        checkpoint = save_checkpoint(
            Checkpoint(
                task_id=f"task-{session_id}",
                session_id=session_id,
                run_id=f"run-{session_id}",
                segment_id=f"segment-{session_id}",
                state=state,
                reason=state.lower(),
                pending_tool_call={"tool_name": "file_write"}
                if state == "pre_tool"
                else None,
            )
        )
        _record_ledger_checkpoint(data_root, checkpoint)

        resumed = inspect_resume(
            checkpoint_id=session_id,
            project_root=tmp_path,
            data_root=tmp_path / ".reins" / "data",
        )

        assert resumed.status == expected_status
        assert "RESUME_READY" in resumed.output
        assert f"reason: {state.lower()}" in resumed.output


def _record_ledger_checkpoint(data_root: Path, checkpoint: object) -> None:
    from runtime.checkpoint import Checkpoint, checkpoint_to_ledger_state
    from runtime.ledger import LedgerStore
    from runtime.ledger_writer import LedgerWriter

    assert isinstance(checkpoint, Checkpoint)
    reason = checkpoint.reason or checkpoint.state.lower()
    LedgerWriter(
        LedgerStore(data_root),
        source="tests.run_foundation",
    ).record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )
