from __future__ import annotations

from pathlib import Path

from context.relevance import decide_task_relevance
from runtime.run_facts import RunFactStore
from runtime.types import Trigger


def test_task_relevance_honors_explicit_override(tmp_path: Path) -> None:
    decision = decide_task_relevance(
        effective_task_id="task-1",
        trigger=Trigger.USER,
        payload={},
        task_relevant=True,
        data_root=tmp_path,
    )

    assert decision.include_task_context
    assert decision.reason == "explicit_task_relevance_true"


def test_task_relevance_continue_task_id_is_explicit_continue(
    tmp_path: Path,
) -> None:
    decision = decide_task_relevance(
        effective_task_id="task-1",
        trigger=Trigger.USER,
        payload={"continue_task_id": "task-1"},
        data_root=tmp_path,
    )

    assert decision.include_task_context
    assert decision.reason == "explicit_continue"


def test_task_relevance_resume_and_checkpoint_payload_are_relevant(
    tmp_path: Path,
) -> None:
    resume = decide_task_relevance(
        effective_task_id="task-1",
        trigger=Trigger.RESUME,
        payload={},
        data_root=tmp_path,
    )
    checkpoint = decide_task_relevance(
        effective_task_id="task-1",
        trigger=Trigger.USER,
        payload={"checkpoint_id": "checkpoint-1"},
        data_root=tmp_path,
    )

    assert resume.include_task_context
    assert resume.reason == "resume_trigger"
    assert checkpoint.include_task_context
    assert checkpoint.reason == "checkpoint_payload"


def test_task_relevance_current_run_unfinished_checkpoint(
    tmp_path: Path,
) -> None:
    store = RunFactStore(tmp_path)
    store.append(
        {
            "event": "checkpoint:saved",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "checkpoint": {"checkpoint_id": "checkpoint-1", "state": "PAUSED"},
        }
    )

    decision = decide_task_relevance(
        effective_task_id="task-1",
        trigger=Trigger.USER,
        payload={},
        data_root=tmp_path,
        run_id="run-1",
    )

    assert decision.include_task_context
    assert decision.reason == "unfinished_checkpoint_in_current_run"
    assert decision.run_fact_sources == ("run_facts:run:run-1",)


def test_task_relevance_infers_focus_task_from_current_run_checkpoint(
    tmp_path: Path,
) -> None:
    RunFactStore(tmp_path).append(
        {
            "event": "checkpoint:saved",
            "session_id": "session-1",
            "run_id": "run-1",
            "focus_task_id": "task-1",
            "checkpoint": {"checkpoint_id": "checkpoint-1", "state": "PAUSED"},
        }
    )

    decision = decide_task_relevance(
        effective_task_id=None,
        trigger=Trigger.USER,
        payload={},
        data_root=tmp_path,
        run_id="run-1",
    )

    assert decision.include_task_context
    assert decision.focus_task_id == "task-1"
    assert decision.reason == "unfinished_checkpoint_in_current_run"


def test_task_relevance_continues_recent_same_focus_task(
    tmp_path: Path,
) -> None:
    store = RunFactStore(tmp_path)
    for index in range(2):
        store.append(
            {
                "event": "run:start",
                "ts": f"2026-05-09T00:00:0{index}Z",
                "session_id": "session-1",
                "run_id": f"run-{index}",
                "task_id": "task-1",
                "focus_task_id": "task-1",
            }
        )

    decision = decide_task_relevance(
        effective_task_id="task-1",
        trigger=Trigger.USER,
        payload={"message": "continue pytest work"},
        data_root=tmp_path,
        session_id="session-1",
        task_goal="pytest work",
        task_tags=["pytest"],
    )

    assert decision.include_task_context
    assert decision.reason == "recent_focus_task_continuity"


def test_task_relevance_topic_switch_stays_minimal(tmp_path: Path) -> None:
    decision = decide_task_relevance(
        effective_task_id="task-1",
        trigger=Trigger.USER,
        payload={"message": "what is the weather today"},
        data_root=tmp_path,
        session_id="session-1",
        task_goal="pytest work",
        task_tags=["pytest"],
    )

    assert not decision.include_task_context
    assert decision.reason == "topic_switch"


def test_task_relevance_recent_terminal_run_loses_focus(tmp_path: Path) -> None:
    store = RunFactStore(tmp_path)
    store.append(
        {
            "event": "run:start",
            "ts": "2026-05-09T00:00:00Z",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "focus_task_id": "task-1",
        }
    )
    store.append(
        {
            "event": "state:transition",
            "ts": "2026-05-09T00:00:01Z",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "focus_task_id": "task-1",
            "to_state": "DONE",
        }
    )

    decision = decide_task_relevance(
        effective_task_id="task-1",
        trigger=Trigger.USER,
        payload={"message": "continue pytest work"},
        data_root=tmp_path,
        session_id="session-1",
        task_goal="pytest work",
        task_tags=["pytest"],
    )

    assert not decision.include_task_context
    assert decision.reason == "recent_focus_task_terminal"
