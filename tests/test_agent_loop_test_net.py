from __future__ import annotations
from scripts.testing.llm import from_test_error, from_test_sequence

from pathlib import Path

import pytest

from llm.client import RealLLMClient
from llm.retry_utils import MAX_TOTAL_ATTEMPTS
from runtime.agent_loop import AgentLoop, State
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.run_evidence import RunEvidenceStore
from runtime.stream_events import LifecycleChanged
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry


def test_run_context_storage_identity_priority() -> None:
    cases = [
        ("formal-task", "compat-task", "run-formal", "formal-task", True),
        (None, "compat-task", "run-session", "compat-task", False),
        (None, None, "run-fallback", "run-fallback", False),
    ]

    for task_id, compat_id, run_id, expected, has_task in cases:
        context = RunContext(
            task_id=task_id,
            compatibility_task_id=compat_id,
            run_id=run_id,
            trigger=Trigger.USER,
            payload={"message": "hello"},
            capability_lease=from_trigger("user", task_id=task_id or ""),
        )

        assert context.storage_task_id == expected
        assert context.has_formal_task is has_task


def test_sync_invalid_model_protocol_budget_records_persistent_failure(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(['{"type":"unsupported"}', '{"type":"unsupported"}'])
    loop, context = _build_loop(project, client)

    assert loop.run(context) is State.FAILED

    errors = _errors(project, context)
    budget_error = _budget_exhausted_error(errors)
    budget_fact = _budget_exhausted_fact(project, context)
    terminal = _terminal_lifecycle(project, context)
    observations = [
        item["observation"]
        for item in loop.tool_history
        if isinstance(item.get("observation"), dict)
    ]

    assert budget_error["category"] == "invalid_model_protocol"
    assert budget_error["recovery_action"] == "fail"
    assert budget_error["attempt_count"] == 2
    assert budget_error["budget_total"] == 1
    assert budget_error["budget_remaining"] == 0
    assert budget_error["final_outcome"] == "exhausted"
    assert budget_fact["recovery_action"] == "fail"
    assert budget_fact["budget_exhausted"] is True
    assert observations[-1]["budget_exhausted"] is True
    assert terminal["lifecycle"] == "failed"


def test_stream_invalid_model_protocol_budget_records_persistent_failure(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(['{"type":"unsupported"}', '{"type":"unsupported"}'])
    loop, context = _build_loop(project, client)

    events = list(loop.run_stream(context))

    assert loop.state is State.FAILED
    assert isinstance(events[-1], LifecycleChanged)
    assert events[-1].lifecycle == "failed"
    errors = _errors(project, context)
    budget_error = _budget_exhausted_error(errors)
    budget_fact = _budget_exhausted_fact(project, context)
    assert budget_error["category"] == "invalid_model_protocol"
    assert budget_error["final_outcome"] == "exhausted"
    assert budget_fact["budget_exhausted"] is True


def test_rate_limited_retry_exhaustion_fails_once_at_loop_level(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    # from_test_error 是 repeat 语义：每次调用都撞同一个可重试错，重试预算才会真正烧完
    client = from_test_error(
        "provider rate limited test request",
        category="rate_limited",
        retryable=True,
    )
    loop, context = _build_loop(project, client)

    assert loop.run(context) is State.FAILED

    errors = _errors(project, context)
    terminal = _terminal_lifecycle(project, context)
    llm_fact = _last_llm_fact(project, context)
    observation = llm_fact["summary"]["observation"]

    rate_limited = [row for row in errors if row["category"] == "rate_limited"]

    # 客户端内部烧掉整轮重试预算，但对 loop 只暴露一次失败，不是每次重试各记一条
    assert len(rate_limited) == 1
    assert errors[-1]["category"] == "rate_limited"
    assert "recovery_action" not in errors[-1]
    assert "budget_total" not in errors[-1]
    assert observation["attempt_count"] == MAX_TOTAL_ATTEMPTS
    assert observation["was_retried"] is True
    assert observation["error_category"] == "rate_limited"
    assert terminal["lifecycle"] == "failed"


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    (project / "tools" / "alpha.py").write_text("print('hi')\n", encoding="utf-8")
    (project / ".reins" / "data").mkdir(parents=True)
    return project


def _build_loop(
    project: Path,
    client: RealLLMClient,
) -> tuple[AgentLoop, RunContext]:
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("test goal")
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": record.goal},
        capability_lease=from_trigger("user", task_id=record.task_id),
        segment_id=f"user-{record.task_id}",
    )
    loop = AgentLoop(data_root, llm_client=client, tool_registry=registry)
    return loop, context


def _errors(project: Path, context: RunContext) -> list[dict[str, object]]:
    """读取本运行实际提交的错误；参数：项目与运行；返回：按提交顺序的错误正文。"""
    return [
        row["payload"]
        for row in RunEvidenceStore(project / ".reins" / "data").list_records(
            session_id=context.session_id, run_id=context.run_id, kind="error"
        )
    ]


def _budget_exhausted_error(
    errors: list[dict[str, object]],
) -> dict[str, object]:
    return next(item for item in errors if item.get("final_outcome") == "exhausted")


def _budget_exhausted_fact(
    project: Path,
    context: RunContext,
) -> dict[str, object]:
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    detail = next(
        item["detail"]
        for item in facts
        if item.get("event") == "tool:error_observation"
        and isinstance(item.get("detail"), dict)
        and item["detail"].get("budget_exhausted") is True
    )
    return detail


def _terminal_lifecycle(project: Path, context: RunContext) -> dict[str, object]:
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    lifecycles = [item for item in facts if item.get("event") == "run:lifecycle"]
    return lifecycles[-1]


def _last_llm_fact(project: Path, context: RunContext) -> dict[str, object]:
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    llm_facts = [item for item in facts if item.get("event") == "llm:response"]
    return llm_facts[-1]
