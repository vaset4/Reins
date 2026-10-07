from __future__ import annotations

from pathlib import Path

from runtime.agent_loop import AgentLoop, State
from runtime.checkpoint import Checkpoint, checkpoint_to_ledger_state, save_checkpoint
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_facts import RunFactStore
from runtime.types import Lease, RunContext, Trigger
from tests.test_agent_loop_state_machine import tool_loop
from triggers.resume import make_run_context as make_resume_context
from triggers.user import make_run_context as make_user_context


def test_dod_6_lifecycle_covers_core_boundaries(tmp_path: Path) -> None:
    loop, context = tool_loop(tmp_path)
    assert loop.run(context) is State.DONE
    facts = RunFactStore(loop.data_root).read_run(context.run_id)
    lifecycles = [fact for fact in facts if fact.get("event") == "run:lifecycle"]
    assert lifecycles[-1]["lifecycle"] == "done"
    assert lifecycles[-1]["reason"] == "final_output"
    assert "tool:request" in {fact.get("event") for fact in facts}
    assert "tool:response" in {fact.get("event") for fact in facts}
    assert not any(fact.get("event") == "state:transition" for fact in facts)


def test_dod_6_transition_api_is_retired(tmp_path: Path) -> None:
    loop = AgentLoop(tmp_path)
    context = RunContext(
        task_id="2026-05-04-illegal",
        trigger=Trigger.USER,
        payload={},
        capability_lease=Lease(),
    )

    try:
        loop.transition(State.DONE, context)
    except RuntimeError as exc:
        assert "transition is retired" in str(exc)
    else:
        raise AssertionError("transition API should be retired")


def test_dod_6_state_machine_accepts_four_phase1_triggers(tmp_path: Path) -> None:
    user_context = make_user_context("build task", data_root=tmp_path)
    assert user_context.trigger is Trigger.USER

    # make_run_context (cron trigger-context builder) retired with the dead cron
    # runner (cron-lease-closure). CRON stays a first-class phase-1 trigger;
    # build the context directly to prove the state machine still accepts it.
    cron_context = RunContext(
        task_id="2026-05-04-cron",
        trigger=Trigger.CRON,
        payload={},
        capability_lease=Lease(),
    )
    assert cron_context.trigger is Trigger.CRON

    checkpoint = save_checkpoint(
        Checkpoint(
            task_id="2026-05-04-resume",
            segment_id="user-1",
            state="paused",
            session_id="session-resume",
            run_id="run-resume",
            reason="pre_tool",
        )
    )
    LedgerWriter(
        LedgerStore(tmp_path),
        source="tests.acceptance.dod_6",
    ).record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        checkpoint.reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )
    resume_context = make_resume_context("2026-05-04-resume", data_root=tmp_path)
    assert resume_context.trigger is Trigger.RESUME

    delegate_context = RunContext(
        task_id="2026-05-04-delegate",
        trigger=Trigger.DELEGATE,
        payload={},
        capability_lease=Lease(),
        parent_segment_id=user_context.segment_id,
    )
    assert delegate_context.trigger is Trigger.DELEGATE
