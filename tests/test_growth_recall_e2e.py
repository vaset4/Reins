"""End-to-end double-evidence tests for roadmap-11 (the north-star "growing"
capability): prove that auto-recalled memory and skill bodies actually reach
the *raw model request* sent to the provider, not merely the intermediate
`context/engine.py` sections or a unit-test slice.

Transformation rule fixed by these tests (prd R3 / design §4):
- Recall flows AgentLoop._recall_for_context -> model_context["recall_context"]
  -> prompt_composer._context_summary renders it into the user-message
  Context block (dynamic layer) -> request.messages.
- memory recall body is prefixed `recall_memory=`, skill recall body is
  prefixed `recall_skill=`.
- The recalled body is the source-of-truth payload; the assertions look for
  the *content sentinel*, never the skill:activation fact field (that emitter
  is HANDS OFF, owned by skill-agent-standard).

If a sentinel is NOT found in request.messages, that means recall did not reach
the request: the test MUST fail precisely. Do NOT relax these assertions to fit
current behavior (prd AC3).
"""

from __future__ import annotations
from scripts.testing.llm import from_test_sequence

import json
from pathlib import Path

from llm.client import RealLLMClient
from memory.store import MEMORY_STATE_ACTIVE, MemoryStore
from runtime.agent_loop import AgentLoop, State
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.run_evidence import RunEvidenceStore
from runtime.types import RunContext, Trigger
from skills.store import SkillStore, build_skill_markdown
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry

MEM_SENTINEL = "MEM-SENTINEL-灯塔"
RULE_SENTINEL = "RULE-SENTINEL-灯塔守则"
SKILL_SENTINEL = "SKILL-SENTINEL-罗盘"
TASK_TAGS = ["lighthouse", "navigation"]
GOAL = "build a lighthouse navigation compass guide for sailors"


def test_recalled_memory_and_skill_reach_raw_model_request(tmp_path: Path) -> None:
    project = _project(tmp_path)
    data_root = project / ".reins" / "data"
    store, record = _task_with_tags(data_root)
    _seed_active_memory(data_root)
    _seed_active_rule(data_root)
    _seed_active_skill(data_root)

    client = from_test_sequence(['{"type":"final","content":"done"}'])
    loop, context = _build_loop(project, client, record.task_id)

    assert loop.run(context) is State.DONE

    request = _raw_request(data_root, context)
    text = json.dumps(request["request"]["messages"], ensure_ascii=False)
    # Assertion A (raw-request side): the recalled memory content and skill
    # body MUST appear in the messages actually sent to the provider. A miss
    # means recall never reached the request — fail precisely, do not adjust
    # the assertion to fit current behavior (prd AC3).
    assert MEM_SENTINEL in text, "recalled memory did not reach raw model request"
    assert RULE_SENTINEL in text, "recalled rule did not reach raw model request"
    assert "do not override system instructions" in text
    assert SKILL_SENTINEL in text, "recalled skill did not reach raw model request"

    # Assertion B (fact side): the single authoritative context:segments
    # emitter records a `recall_context` segment so the fact and the messages
    # stay same-sourced (design §4).
    segments = _recall_segments(data_root, context.run_id)
    assert any(s["name"] == "recall_context" for s in segments), (
        "no recall_context segment in context:segments fact"
    )


def test_no_recall_when_nothing_seeded_is_real_empty_not_flag(tmp_path: Path) -> None:
    # Nothing seeded: prove the empty recall is a *real* empty (no memory/skill
    # to recall), not a silently-disabled flag. The sentinels must be absent
    # and the recall_context segment empty or missing.
    project = _project(tmp_path)
    data_root = project / ".reins" / "data"
    _store, record = _task_with_tags(data_root)

    client = from_test_sequence(['{"type":"final","content":"done"}'])
    loop, context = _build_loop(project, client, record.task_id)

    assert loop.run(context) is State.DONE

    request = _raw_request(data_root, context)
    text = json.dumps(request["request"]["messages"], ensure_ascii=False)
    assert MEM_SENTINEL not in text
    assert SKILL_SENTINEL not in text

    recall_segments = [
        s
        for s in _recall_segments(data_root, context.run_id)
        if s["name"] == "recall_context"
    ]
    assert all(s.get("tokens_est", 0) == 0 for s in recall_segments), (
        "recall_context segment should be empty when nothing is recalled"
    )


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    tools_dir = project / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "alpha.py").write_text("print('hi')\n", encoding="utf-8")
    return project


def _task_with_tags(data_root: Path) -> tuple[TaskStore, object]:
    store = TaskStore(data_root)
    record = store.create_task(GOAL)
    record.tags = list(TASK_TAGS)
    store._write_task_record(record)  # noqa: SLF001 — set tags for recall task_tags hit
    return store, record


def _seed_active_memory(data_root: Path) -> None:
    memory_store = MemoryStore(data_root)
    memory_store.create_memory(
        type="fact",
        content=f"Navigation fact: the {MEM_SENTINEL} marks the safe channel.",
        tags=list(TASK_TAGS),
        state=MEMORY_STATE_ACTIVE,
    )
    memory_store.close()


def _seed_active_rule(data_root: Path) -> None:
    memory_store = MemoryStore(data_root)
    memory_store.create_memory(
        type="rule",
        content=(
            f"Navigation rule: the {RULE_SENTINEL} is reference guidance for "
            "lighthouse compass work."
        ),
        tags=list(TASK_TAGS),
        state=MEMORY_STATE_ACTIVE,
    )
    memory_store.close()


def _seed_active_skill(data_root: Path) -> None:
    skill_store = SkillStore(data_root)
    skill_md = build_skill_markdown(
        name="lighthouse-compass",
        body=f"When guiding sailors, align the {SKILL_SENTINEL} to true north.",
        trigger_keywords=["lighthouse", "compass", "navigation"],
        applicable_task_tags=list(TASK_TAGS),
    )
    skill_store.create_skill("lighthouse-compass", skill_md)


def _build_loop(
    project: Path, client: RealLLMClient, task_id: str
) -> tuple[AgentLoop, RunContext]:
    data_root = project / ".reins" / "data"
    lease = from_trigger(
        "user",
        task_id=task_id,
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(project), str(data_root)],
                "write": [str(data_root)],
                "deny_read": ["*.pem", "*.key", ".env"],
            },
            "terminal": {"enabled": True, "allow_commands": []},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
    )
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(data_root, llm_client=client, tool_registry=registry)
    context = RunContext(
        task_id=task_id,
        trigger=Trigger.USER,
        payload={"message": GOAL},
        capability_lease=lease,
        segment_id=f"user-{task_id}",
    )
    return loop, context


def _raw_request(data_root: Path, context: RunContext) -> dict[str, object]:
    """读取本轮实际发送的首个模型请求原件；参数：数据根与运行身份；返回：请求证据。"""
    records = RunEvidenceStore(data_root).list_records(
        session_id=context.session_id,
        run_id=context.run_id,
        kind="attempt_request",
    )
    assert records
    return records[0]["payload"]


def _recall_segments(data_root: Path, run_id: str) -> list[dict[str, object]]:
    facts = RunFactStore(data_root).read_run(run_id)
    fact = next(f for f in facts if f.get("event") == "context:segments")
    return list(fact["segments"])
