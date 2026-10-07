from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from _pytest.logging import LogCaptureFixture

from context.cache import cache_tier_counts, to_prompt_sections
from context.engine import (
    _recalled_memory_body,
    build_context_sections_for_tests,
    recall_context_body,
    token_stats,
)
from context.memory_recall import RecallOutcome
from memory.index import MemoryIndexState
from context.memory_recall import TYPE_LIMITS, MemoryRecallResult
from llm.types import CacheTier
from memory.store import MEMORY_STATE_ACTIVE, Memory, MemoryDetails, MemoryStore
from runtime.run_facts import RunFactStore
from runtime.session_messages import append_assistant_message, append_user_message
from runtime.types import Trigger
from skills.store import SkillStore, build_skill_markdown
from tasks.store import TaskStore

SESSION_ID = "session-context-engine"


def test_context_engine_builds_11_sections_with_recall_and_cache_tiers(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-04-test"
    store = TaskStore(tmp_path)
    store.create_task("run pytest and update docs", task_id=task_id)
    store._update_metadata(
        task_id,
        {
            "tags": ["pytest"],
            "spec_refs": ["spec.md"],
            "skill_refs": ["exact-skill"],
        },
    )
    (tmp_path / "spec.md").write_text("Always keep tests focused.", encoding="utf-8")
    from tools.todo_tool import add_todo, update_todo

    for text, status in (
        ("pending item", "pending"),
        ("done item", "done"),
        ("active item", "in_progress"),
    ):
        item = add_todo(task_id, text, data_root=tmp_path)
        update_todo(task_id, item.idx, status, data_root=tmp_path)
    store.update_summary(task_id, "Current summary.")
    for index in range(21):
        append_user_message(tmp_path, SESSION_ID, f"message {index} pytest")
    for index in range(3):
        RunFactStore(tmp_path).append(
            {
                "event": "tool:response",
                "session_id": SESSION_ID,
                "run_id": "run-test",
                "task_id": task_id,
                "tool": {
                    "call_id": str(index),
                    "name": "pytest",
                    "status": "failed",
                    "args_summary": {"k": "unit"},
                    "error": "tests failed",
                },
            }
        )

    memories = MemoryStore(tmp_path)
    memories.create_memory(
        "rule", "Run pytest before committing.", ["pytest"], memory_id="rule-1"
    )
    memories.create_memory(
        "lesson",
        "Repeated pytest failure needs a new approach.",
        ["pytest"],
        memory_id="lesson-1",
    )
    memories.close()
    skills = SkillStore(tmp_path)
    skills.create_skill(
        "exact-skill",
        build_skill_markdown(name="Exact", body="Exact referenced skill."),
        meta={},
    )
    skills.create_skill(
        "pytest-runner",
        build_skill_markdown(
            name="Pytest Runner",
            body="Run pytest.",
            trigger_keywords=["pytest"],
            applicable_task_tags=["pytest"],
        ),
        meta={},
    )

    sections = build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "please run pytest"},
        lease=SimpleNamespace(max_tokens=1000),
        session_id=SESSION_ID,
        task_relevant=True,
        data_root=tmp_path,
        project_root=tmp_path,
    )

    assert [section.name for section in sections] == [
        "identity",
        "task",
        "todo",
        "intent",
        "resume_hint",
        "progress",
        "summary",
        "specs",
        "skills",
        "facts_preferences",
        "conversation",
        "tool_results",
        "trigger_payload",
        "tools_output",
    ]
    assert cache_tier_counts(sections) == {
        CacheTier.STABLE: 2,
        CacheTier.SEMI_STABLE: 4,
        CacheTier.DYNAMIC: 8,
    }
    assert "📌 spec.md" in sections[7].content
    assert "🔍 rule-1" in sections[7].content
    assert sections[7].content.count("rule-1") == 1
    assert "📌 exact-skill" in sections[8].content
    assert "🔍 pytest-runner" in sections[8].content
    assert "lesson:lesson-1" in sections[9].content
    assert "message 0" not in sections[10].content
    assert "message 20" in sections[10].content
    assert "consecutive_failure_pattern" in sections[11].content
    assert token_stats(sections)["total"] > 0
    assert len(to_prompt_sections(sections)) == 14


def test_context_engine_skill_activation_fact_uses_skill_recall_fields(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-04-skill-fact"
    TaskStore(tmp_path).create_task("run pytest", task_id=task_id)
    SkillStore(tmp_path).create_skill(
        "pytest-runner",
        build_skill_markdown(
            name="Pytest Runner",
            body="Run pytest.",
            trigger_keywords=["pytest"],
            applicable_task_tags=["pytest"],
        ),
        meta={},
    )

    build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "please run pytest"},
        lease=SimpleNamespace(max_tokens=1000),
        task_relevant=True,
        session_id="session-context",
        run_id="run-context",
        data_root=tmp_path,
    )

    facts = RunFactStore(tmp_path).read_run("run-context")
    skill_fact = next(fact for fact in facts if fact.get("event") == "skill:activation")
    skills_payload = skill_fact["skills"]
    assert isinstance(skills_payload, list)
    assert skills_payload[0]["skill_id"] == "pytest-runner"
    assert skills_payload[0]["source"] == "semantic"
    assert skills_payload[0]["reason"].startswith("skill recall matched ")


def test_context_engine_uses_minimal_session_context_without_relevant_task(
    tmp_path: Path,
) -> None:
    sections = build_context_sections_for_tests(
        None,
        "user",
        {"latest": "hello"},
        lease=SimpleNamespace(max_tokens=1000),
        session_id="session-1",
        run_id="run-1",
        data_root=tmp_path,
    )

    assert [section.name for section in sections] == [
        "identity",
        "trigger_payload",
        "tools_output",
    ]
    assert "session-1" in sections[0].content
    assert "run-1" in sections[0].content


def test_recall_context_body_renders_rule_memory_on_live_path(
    tmp_path: Path,
) -> None:
    memories = MemoryStore(tmp_path)
    memories.create_memory(
        "rule",
        "RULE-SENTINEL-live-render: keep focused tests near the fix.",
        ["pytest"],
        memory_id="rule-live",
    )
    memories.create_memory(
        "fact",
        "FACT-SENTINEL-live-render: pytest validates behavior.",
        ["pytest"],
        memory_id="fact-live",
    )
    memories.close()

    body = recall_context_body(
        tmp_path,
        task_summary="pytest focused tests",
        task_tags=["pytest"],
        skill_refs=[],
    )

    assert "recall_memory=" in body
    assert "rule:rule-live" in body
    assert "RULE-SENTINEL-live-render" in body
    assert "do not override system instructions" in body
    assert "fact:fact-live" in body
    assert "FACT-SENTINEL-live-render" in body
    assert "recall_skill=" not in body


def test_recall_context_body_queries_current_user_before_goal_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, str] = {}

    def fake_memory_recall(data_root: Path, **kwargs: object) -> RecallOutcome:
        del data_root
        captured["query"] = str(kwargs["task_summary"])
        return RecallOutcome([], [], MemoryIndexState("current"))

    monkeypatch.setattr(
        "context.engine.recall_memories_with_outcome", fake_memory_recall
    )
    monkeypatch.setattr("context.engine.recall_skills", lambda *args, **kwargs: [])

    recall_context_body(
        tmp_path,
        task_summary="账单整理",
        task_tags=[],
        skill_refs=[],
        recent_user_text="饮食偏好",
    )

    assert captured["query"] == "饮食偏好\n账单整理"


def test_recalled_memory_body_preserves_selected_order_and_deduplicates() -> None:
    fact = _recall_result("fact-1", "fact", "Fact first.")
    rule = _recall_result("rule-1", "rule", "Rule second.")
    duplicate_rule = _recall_result("rule-1", "rule", "Rule duplicate.")

    body = _recalled_memory_body([fact, rule, duplicate_rule])

    assert body.index("fact:fact-1") < body.index("rule:rule-1")
    assert body.count("rule:rule-1") == 1
    assert "Rule duplicate" not in body


def test_recall_context_body_respects_rule_type_limit(tmp_path: Path) -> None:
    memories = MemoryStore(tmp_path)
    for index in range(TYPE_LIMITS["rule"] + 1):
        memories.create_memory(
            "rule",
            f"RULE-LIMIT-SENTINEL-{index}: pytest rule limit.",
            ["pytest"],
            memory_id=f"rule-limit-{index}",
        )
    memories.close()

    body = recall_context_body(
        tmp_path,
        task_summary="pytest rule limit",
        task_tags=["pytest"],
        skill_refs=[],
    )

    assert body.count("rule:rule-limit-") == TYPE_LIMITS["rule"]


def _read_run_facts(data_root: Path, run_id: str) -> list[dict[str, object]]:
    return RunFactStore(data_root).read_run(run_id)


def test_recall_context_body_writes_injection_explain_observation(
    tmp_path: Path,
) -> None:
    # 1. 同一主体字段的不同结论须进入运行观测，供追溯冲突
    memories = MemoryStore(tmp_path)
    memories.create_memory(
        "fact",
        "OBS-SENTINEL: pytest fixtures live in conftest.",
        ["pytest"],
        memory_id="obs-a",
        details=MemoryDetails(subject="repository", fact_key="fixture_location"),
    )
    memories.create_memory(
        "fact",
        "obs-sentinel: pytest fixtures live in tests/helpers.py.",
        ["testing"],
        memory_id="obs-b",
        details=MemoryDetails(subject="repository", fact_key="fixture_location"),
    )
    memories.close()

    recall_context_body(
        tmp_path,
        task_summary="pytest observation",
        task_tags=["pytest", "testing"],
        skill_refs=[],
        session_id="session-obs",
        run_id="run-obs",
        task_id="task-obs",
    )

    facts = _read_run_facts(tmp_path, "run-obs")
    explains = [f for f in facts if f.get("event") == "memory:injection_explain"]
    assert len(explains) == 1
    explain = explains[0]["explain"]
    assert isinstance(explain, dict)
    warnings = explain["warnings"]
    assert any("conflict_detected" in str(w) for w in warnings)
    conflicted = [row for row in explain["injected"] if row["has_conflict"] is True]
    assert {row["memory_id"] for row in conflicted} == {"obs-a", "obs-b"}


def test_recall_context_body_observation_records_skipped(tmp_path: Path) -> None:
    # AC5：被 type 上限筛掉的记忆如实进 explain.skipped（含 reason）
    memories = MemoryStore(tmp_path)
    for index in range(TYPE_LIMITS["rule"] + 2):
        memories.create_memory(
            "rule",
            f"SKIP-SENTINEL-{index}: pytest rule beyond limit.",
            ["pytest"],
            memory_id=f"skip-rule-{index}",
        )
    memories.close()

    recall_context_body(
        tmp_path,
        task_summary="pytest skip",
        task_tags=["pytest"],
        skill_refs=[],
        session_id="session-skip",
        run_id="run-skip",
        task_id="task-skip",
    )

    facts = _read_run_facts(tmp_path, "run-skip")
    explain = next(f for f in facts if f.get("event") == "memory:injection_explain")[
        "explain"
    ]
    assert len(explain["skipped"]) >= 1
    assert all(str(row["reason"]) for row in explain["skipped"])


def test_recall_context_body_without_ids_writes_no_observation(
    tmp_path: Path,
) -> None:
    # AC4：不传 id（tests/scripts 现状调用）→ 不 append 任何事件、正常返回正文
    memories = MemoryStore(tmp_path)
    memories.create_memory(
        "fact",
        "NOOBS-SENTINEL: pytest run without ids.",
        ["pytest"],
        memory_id="noobs-1",
    )
    memories.close()

    body = recall_context_body(
        tmp_path,
        task_summary="pytest no ids",
        task_tags=["pytest"],
        skill_refs=[],
    )

    assert "NOOBS-SENTINEL" in body
    # 无 id 时 sessions 目录根本不应被写入
    assert not (tmp_path / "sessions").exists()


def test_recall_context_body_observation_does_not_change_prompt_body(
    tmp_path: Path,
) -> None:
    # AC3：喂模型正文不变——传 id（写观测）与不传 id（不写观测）返回串逐字节相等
    memories = MemoryStore(tmp_path)
    memories.create_memory(
        "fact",
        "GOLDEN-SENTINEL: pytest golden body invariance.",
        ["pytest"],
        memory_id="golden-1",
    )
    memories.close()

    without_ids = recall_context_body(
        tmp_path,
        task_summary="pytest golden",
        task_tags=["pytest"],
        skill_refs=[],
    )
    with_ids = recall_context_body(
        tmp_path,
        task_summary="pytest golden",
        task_tags=["pytest"],
        skill_refs=[],
        session_id="session-golden",
        run_id="run-golden",
        task_id="task-golden",
    )

    assert with_ids == without_ids


def test_recall_context_body_includes_conflict_notice_for_same_scoped_field(
    tmp_path: Path,
) -> None:
    """有冲突时 recall_context_body 应包含 recall_conflict= 段"""
    memories = MemoryStore(tmp_path)
    # 同一字段存在不同约定，正文与来源一起交给模型判断
    memories.create_memory(
        "rule",
        "Always use black",
        ["formatting"],
        memory_id="dup-1",
        details=MemoryDetails(subject="repository", fact_key="formatter"),
    )
    memories.create_memory(
        "rule",
        "Always use ruff",
        ["style"],
        memory_id="dup-2",
        details=MemoryDetails(subject="repository", fact_key="formatter"),
    )
    memories.close()

    body = recall_context_body(
        tmp_path,
        task_summary="code formatting",
        task_tags=["formatting", "style"],
        skill_refs=[],
    )

    # 应包含 recall_conflict= 段
    assert "recall_conflict=" in body
    assert "different claims about the same scoped subject" in body
    assert "dup-1" in body
    assert "dup-2" in body


def test_recall_context_body_no_conflict_notice_when_no_duplicates(
    tmp_path: Path,
) -> None:
    """无冲突时 recall_context_body 不应包含 recall_conflict= 段（正文逐字节不变）"""
    memories = MemoryStore(tmp_path)
    # 两条内容不同的记忆
    memories.create_memory("rule", "use black", ["formatting"], memory_id="distinct-1")
    memories.create_memory("rule", "use pytest", ["testing"], memory_id="distinct-2")
    memories.close()

    body = recall_context_body(
        tmp_path,
        task_summary="coding rules",
        task_tags=["formatting", "testing"],
        skill_refs=[],
    )

    # 不应包含 recall_conflict= 段
    assert "recall_conflict=" not in body


def test_recall_context_body_observation_logs_prefixed_line(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    # AC6：写观测路径打【记忆】【召回观测】info 日志；无 log.warn
    memories = MemoryStore(tmp_path)
    memories.create_memory(
        "fact",
        "LOG-SENTINEL: pytest recall observation log.",
        ["pytest"],
        memory_id="log-1",
    )
    memories.close()

    with caplog.at_level(logging.INFO):
        recall_context_body(
            tmp_path,
            task_summary="pytest log",
            task_tags=["pytest"],
            skill_refs=[],
            session_id="session-log",
            run_id="run-log",
            task_id="task-log",
        )

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "【记忆】【召回观测】" in logged
    assert not any(record.levelno == logging.WARNING for record in caplog.records)


def _recall_result(
    memory_id: str, memory_type: str, content: str
) -> MemoryRecallResult:
    return MemoryRecallResult(
        memory=Memory(
            memory_id=memory_id,
            type=memory_type,
            state=MEMORY_STATE_ACTIVE,
            content=content,
        ),
        score=1.0,
        bm25=1.0,
        tag_jaccard=1.0,
        recency_bonus=0.0,
        stale_penalty=0.0,
    )


def test_context_engine_defaults_to_minimal_context_even_with_focus_task(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-09-focus"
    TaskStore(tmp_path).create_task("focused but not relevant", task_id=task_id)

    sections = build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "hello"},
        lease=SimpleNamespace(max_tokens=1000),
        session_id="session-1",
        run_id="run-1",
        focus_task_id=task_id,
        data_root=tmp_path,
    )

    assert [section.name for section in sections] == [
        "identity",
        "trigger_payload",
        "tools_output",
    ]


def test_context_engine_auto_includes_task_on_resume_trigger(tmp_path: Path) -> None:
    task_id = "2026-05-09-resume"
    TaskStore(tmp_path).create_task("resume paused pytest work", task_id=task_id)

    sections = build_context_sections_for_tests(
        task_id,
        Trigger.RESUME,
        {},
        lease=SimpleNamespace(max_tokens=1000),
        session_id="session-1",
        run_id="run-1",
        data_root=tmp_path,
    )

    assert "task" in [section.name for section in sections]
    assert "resume_trigger" in sections[0].content


def test_context_engine_infers_focus_from_unfinished_checkpoint_run_fact(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-09-checkpoint"
    TaskStore(tmp_path).create_task("resume checkpoint work", task_id=task_id)
    RunFactStore(tmp_path).append(
        {
            "event": "checkpoint:saved",
            "session_id": "session-1",
            "run_id": "run-1",
            "focus_task_id": task_id,
            "checkpoint": {"checkpoint_id": "checkpoint-1", "state": "PAUSED"},
        }
    )

    sections = build_context_sections_for_tests(
        None,
        "user",
        {},
        lease=SimpleNamespace(max_tokens=1000),
        session_id="session-1",
        run_id="run-1",
        data_root=tmp_path,
    )

    assert "task" in [section.name for section in sections]
    assert "unfinished_checkpoint_in_current_run" in sections[0].content
    assert task_id in sections[0].content


def test_context_engine_uses_recent_run_facts_to_keep_focus(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-09-continuity"
    TaskStore(tmp_path).create_task("continue pytest work", task_id=task_id)
    store = RunFactStore(tmp_path)
    for index in range(2):
        store.append(
            {
                "event": "run:start",
                "ts": f"2026-05-09T00:00:0{index}Z",
                "session_id": "session-1",
                "run_id": f"run-{index}",
                "task_id": task_id,
                "focus_task_id": task_id,
            }
        )

    sections = build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "please continue pytest"},
        lease=SimpleNamespace(max_tokens=1000),
        session_id="session-1",
        run_id="run-next",
        focus_task_id=task_id,
        data_root=tmp_path,
    )

    assert "task" in [section.name for section in sections]
    assert "recent_focus_task_continuity" in sections[0].content


def test_context_engine_topic_switch_with_old_focus_stays_minimal(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-09-topic-switch"
    TaskStore(tmp_path).create_task("continue pytest work", task_id=task_id)

    sections = build_context_sections_for_tests(
        None,
        "user",
        {"latest": "what is the weather today"},
        lease=SimpleNamespace(max_tokens=1000),
        session_id="session-1",
        run_id="run-1",
        focus_task_id=task_id,
        data_root=tmp_path,
    )

    assert [section.name for section in sections] == [
        "identity",
        "trigger_payload",
        "tools_output",
    ]
    assert "topic_switch" in sections[0].content


def test_context_engine_prefers_run_facts_for_tool_results(tmp_path: Path) -> None:
    task_id = "2026-05-09-tool-facts"
    TaskStore(tmp_path).create_task("inspect pytest output", task_id=task_id)
    task_dir = tmp_path / "tasks" / task_id
    _append_jsonl(
        task_dir / "trajectory.jsonl",
        {"tool": "legacy", "status": "ok", "output": "from legacy trajectory"},
    )
    RunFactStore(tmp_path).append(
        {
            "event": "tool:response",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": task_id,
            "tool": {
                "call_id": "call-1",
                "name": "pytest",
                "status": "ok",
                "output_summary": "from run facts",
            },
        }
    )

    sections = build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "pytest output"},
        lease=SimpleNamespace(max_tokens=1000),
        task_relevant=True,
        session_id="session-1",
        run_id="run-1",
        data_root=tmp_path,
    )

    tool_results = sections[
        [section.name for section in sections].index("tool_results")
    ].content
    assert "from run facts" in tool_results
    assert "from legacy trajectory" not in tool_results
    assert "run_facts:run:run-1" in sections[0].content


def test_context_engine_preserves_run_fact_tool_args_for_failure_pattern(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-09-tool-facts-failure"
    TaskStore(tmp_path).create_task("inspect repeated file failures", task_id=task_id)
    store = RunFactStore(tmp_path)
    for index in range(3):
        call_id = f"call-{index}"
        store.append(
            {
                "event": "tool:request",
                "session_id": "session-1",
                "run_id": "run-1",
                "task_id": task_id,
                "tool": {
                    "call_id": call_id,
                    "name": "file_read",
                    "risk": "safe",
                    "args_summary": {"path": "/tmp/missing"},
                },
            }
        )
        store.append(
            {
                "event": "tool:response",
                "session_id": "session-1",
                "run_id": "run-1",
                "task_id": task_id,
                "tool": {
                    "call_id": call_id,
                    "name": "file_read",
                    "status": "failed",
                    "error_category": "not_found",
                    "error": "missing",
                    "output_summary": "failed",
                },
            }
        )

    sections = build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "file failure"},
        lease=SimpleNamespace(max_tokens=1000),
        task_relevant=True,
        session_id="session-1",
        run_id="run-1",
        data_root=tmp_path,
    )

    tool_results = sections[
        [section.name for section in sections].index("tool_results")
    ].content
    assert "consecutive_failure_pattern" in tool_results
    assert "/tmp/missing" in tool_results


def test_context_engine_does_not_use_uncommitted_legacy_trajectory_tool_results(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-09-tool-legacy"
    TaskStore(tmp_path).create_task("inspect pytest output", task_id=task_id)
    task_dir = tmp_path / "tasks" / task_id
    _append_jsonl(
        task_dir / "trajectory.jsonl",
        {"tool": "legacy", "status": "ok", "output": "from legacy trajectory"},
    )

    sections = build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "pytest output"},
        lease=SimpleNamespace(max_tokens=1000),
        task_relevant=True,
        data_root=tmp_path,
    )

    tool_results = sections[
        [section.name for section in sections].index("tool_results")
    ].content
    assert "from legacy trajectory" not in tool_results
    assert f"trajectory:{task_id}" not in sections[0].content


def _skill_activation_sources(data_root: Path, run_id: str) -> dict[str, str]:
    """从 run facts 取每个 skill 的激活来源。

    source 为 explicit 表示 trigger_keywords 命中了用户最近那句话，semantic 表示没命中；
    skill 召回无条件返回 top-k，所以"有没有激活"看不出查询词对不对，只有这个字段能看出来。
    """
    facts = RunFactStore(data_root).read_run(run_id)
    fact = next((f for f in facts if f.get("event") == "skill:activation"), None)
    if fact is None:
        return {}
    payload = fact["skills"]
    assert isinstance(payload, list)
    return {str(entry["skill_id"]): str(entry["source"]) for entry in payload}


def _seed_keyword_skill(data_root: Path) -> None:
    """建一个只靠用户发言里的关键词才会激活的 skill。"""
    SkillStore(data_root).create_skill(
        "pytest-runner",
        build_skill_markdown(
            name="Pytest Runner",
            body="Run pytest.",
            trigger_keywords=["zebrafish"],
            applicable_task_tags=[],
        ),
        meta={},
    )


def test_recent_user_text_survives_history_longer_than_conversation_tail(
    tmp_path: Path,
) -> None:
    # 召回查询词改成从 typed 消息序列取，历史超过 conversation 尾窗（20 行）时
    # 仍要认出用户最后那句话；旧的 dict 版读的是截断后的尾窗，用户发言被挤出窗口就丢词
    task_id = "2026-09-01-recent-user-text"
    TaskStore(tmp_path).create_task("run checks", task_id=task_id)
    _seed_keyword_skill(tmp_path)
    session_id = "session-long-history"
    append_user_message(tmp_path, session_id, "please check the zebrafish pipeline")
    # 用户发言之后再堆 25 条助手回复，把它推到 20 行尾窗之外
    for index in range(25):
        append_assistant_message(tmp_path, session_id, f"working step {index}")

    build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "continue"},
        lease=SimpleNamespace(max_tokens=1000),
        task_relevant=True,
        session_id=session_id,
        run_id="run-long-history",
        data_root=tmp_path,
    )

    assert _skill_activation_sources(tmp_path, "run-long-history") == {
        "pytest-runner": "explicit"
    }


def test_recent_user_text_matches_tail_window_result_when_user_turn_is_recent(
    tmp_path: Path,
) -> None:
    # 用户发言仍在尾窗内时，新旧两种取法必须给出同一个查询词，改造不改这条既有路径
    task_id = "2026-09-01-recent-user-text-tail"
    TaskStore(tmp_path).create_task("run checks", task_id=task_id)
    _seed_keyword_skill(tmp_path)
    session_id = "session-recent-turn"
    for index in range(25):
        append_assistant_message(tmp_path, session_id, f"earlier step {index}")
    append_user_message(tmp_path, session_id, "please check the zebrafish pipeline")
    append_assistant_message(tmp_path, session_id, "on it")

    build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "continue"},
        lease=SimpleNamespace(max_tokens=1000),
        task_relevant=True,
        session_id=session_id,
        run_id="run-recent-turn",
        data_root=tmp_path,
    )

    assert _skill_activation_sources(tmp_path, "run-recent-turn") == {
        "pytest-runner": "explicit"
    }


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    """构造已退休路径的未提交残留，验证不会当作新事实；参数：路径和旧行；返回：无。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
