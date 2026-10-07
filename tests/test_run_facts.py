from __future__ import annotations
from scripts.testing.llm import from_test_stub

from pathlib import Path

import pytest

from runtime.agent_loop import AgentLoop, State
from runtime.checkpoint import Checkpoint, checkpoint_to_ledger_state, save_checkpoint
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_facts import RunFactStore
from tests.test_agent_loop_state_machine import tool_loop
from triggers.resume import make_run_context as make_resume_context
from triggers.user import make_run_context as make_user_context


def test_run_fact_store_roundtrip_and_query_indexes(tmp_path: Path) -> None:
    store = RunFactStore(tmp_path)

    store.append(
        {
            "event": "run:start",
            "ts": "2026-05-09T00:00:00Z",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "args": {
                "api_token": "secret",
                "body": "x" * 600,
            },
        }
    )
    store.append(
        {
            "event": "state:transition",
            "ts": "2026-05-09T00:00:01Z",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "to_state": "DONE",
        }
    )

    facts = store.read_run("run-1")

    assert facts[0]["args"]["api_token"] == "<redacted>"
    assert "<truncated" in facts[0]["args"]["body"]
    assert "compatibility_path" not in facts[0]
    assert store.list_runs_for_session("session-1")[0].run_id == "run-1"
    by_task = store.list_runs_for_task("task-1")
    assert by_task[0].run_id == "run-1"
    assert by_task[0].status == "done"


def test_run_lifecycle_fact_is_latest_status(tmp_path: Path) -> None:
    store = RunFactStore(tmp_path)
    store.append(
        {
            "event": "run:start",
            "ts": "2026-05-09T00:00:00Z",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": "task-1",
        }
    )
    store.append(
        {
            "event": "state:transition",
            "ts": "2026-05-09T00:00:01Z",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "to_state": "DONE",
        }
    )
    store.append_lifecycle(
        lifecycle="paused",
        reason="lease step limit reached",
        session_id="session-1",
        run_id="run-1",
        segment_id="segment-1",
        task_id="task-1",
        checkpoint_id="ck-1",
        resumable=True,
    )

    lifecycle = store.read_latest_lifecycle("run-1")
    by_task = store.list_runs_for_task("task-1")

    assert lifecycle["event"] == "run:lifecycle"
    assert lifecycle["lifecycle"] == "paused"
    assert lifecycle["reason"] == "lease step limit reached"
    assert lifecycle["checkpoint_id"] == "ck-1"
    assert lifecycle["resumable"] is True
    assert by_task[0].status == "paused"


def test_run_lifecycle_rejects_running_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid run lifecycle"):
        RunFactStore(tmp_path).append_lifecycle(
            lifecycle="running",
            reason="not durable",
            session_id="session-1",
            run_id="run-1",
            segment_id="segment-1",
        )


def test_plain_session_run_writes_run_facts_with_inbox_compatibility(
    tmp_path: Path,
) -> None:
    context = make_user_context("plain chat", data_root=tmp_path, formal_task=False)

    assert (
        AgentLoop(tmp_path, llm_client=from_test_stub("会话答复")).run(context)
        is State.DONE
    )

    facts = RunFactStore(tmp_path).read_run(context.run_id)
    events = [fact["event"] for fact in facts]

    assert "run:start" in events
    assert "run:lifecycle" in events
    assert "checkpoint:saved" in events
    assert facts[-1]["lifecycle"] == "done"
    assert facts[0]["task_id"] is None
    assert facts[0]["compatibility_task_id"] == context.compatibility_task_id
    assert all("compatibility_path" not in fact for fact in facts)
    checkpoint_refs = [
        fact["checkpoint"] for fact in facts if fact.get("event") == "checkpoint:saved"
    ]
    assert checkpoint_refs
    for ref in checkpoint_refs:
        assert "compatibility_path" not in ref
        assert "pending_tool_call" in ref
        assert "working_memory_snapshot" in ref
        assert ref["working_memory_snapshot_kind"] == "diagnostic"


def test_formal_task_run_facts_record_tool_and_task_listing(tmp_path: Path) -> None:
    loop, context = tool_loop(tmp_path)
    assert loop.run(context) is State.DONE
    store = RunFactStore(loop.data_root)
    facts = store.read_run(context.run_id)
    tool_request = next(fact for fact in facts if fact.get("event") == "tool:request")
    tool_response = next(fact for fact in facts if fact.get("event") == "tool:response")
    pre_tool_ref = next(
        fact["checkpoint"]
        for fact in facts
        if fact.get("event") == "checkpoint:saved"
        and fact["checkpoint"].get("reason") == "pre_tool"
    )

    assert tool_request["tool"]["name"] == "file_read"
    assert tool_request["tool"]["args_summary"] == {
        "path": str(tmp_path / "evidence.txt")
    }
    assert tool_response["tool"]["status"] == "ok"
    assert pre_tool_ref["pending_tool_call"]["tool_name"] == "file_read"
    assert store.list_runs_for_task(context.storage_task_id)[0].run_id == context.run_id


def test_resume_trigger_can_find_checkpoint_through_run_facts(
    tmp_path: Path,
) -> None:
    checkpoint = save_checkpoint(
        Checkpoint(
            task_id="chat-compat",
            session_id="session-abc",
            run_id="run-from-facts",
            compatibility_task_id="chat-compat",
            segment_id="user-1",
            state="PAUSED",
            reason="paused",
        )
    )
    RunFactStore(tmp_path).append_checkpoint_ref(
        session_id="session-abc",
        run_id="run-from-facts",
        task_id=None,
        focus_task_id=None,
        compatibility_task_id="chat-compat",
        segment_id="user-1",
        checkpoint=checkpoint,
    )
    LedgerWriter(
        LedgerStore(tmp_path), source="tests.run_facts"
    ).record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        checkpoint.reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )

    context = make_resume_context(data_root=tmp_path, source_run_id="run-from-facts")

    assert context.task_id is None
    assert context.compatibility_task_id == "chat-compat"
    assert context.parent_segment_id == "user-1"


def test_read_task_facts_aggregates_runs_by_ts(tmp_path: Path) -> None:
    store = RunFactStore(tmp_path)
    for run_id, ts in (
        ("run-a", "2026-05-10T00:00:00Z"),
        ("run-b", "2026-05-10T00:00:01Z"),
    ):
        store.append(
            {
                "event": "run:start",
                "ts": ts,
                "session_id": "session-shared",
                "run_id": run_id,
                "task_id": "task-shared",
            }
        )

    facts = store.read_task_facts("task-shared")
    assert [fact["run_id"] for fact in facts] == ["run-a", "run-b"]


def test_run_fact_store_rejects_corrupt_jsonl_line(tmp_path: Path) -> None:
    """已提交事实坏字节不能被当作有效运行；参数：隔离根；返回：无。"""
    facts_path = RunFactStore(tmp_path).append(
        {"event": "run:start", "session_id": "session-1", "run_id": "run-1"}
    )
    facts_path.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        RunFactStore(tmp_path).read_run("run-1")
