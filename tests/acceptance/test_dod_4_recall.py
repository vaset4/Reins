from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from context.memory_recall import MemoryRecallResult, recall_memories, stale_penalty
from memory.store import MemoryStore
from memory.records import MemorySource
from llm.messages import TextPart, UserMessage
from runtime.session_message_store import SessionMessageStore
from tasks.store import TaskStore
from runtime.persistence import RuntimeStore
from tests.acceptance.helpers.synth_data import (
    SyntheticMemory,
    make_memory,
    make_similar_tasks,
    make_task,
)


def test_dod_4_fourth_similar_task_recalls_rule_fact_and_lesson(
    reins_data_dir: Path,
) -> None:
    for task in make_similar_tasks(3, theme="git-commit"):
        _write_task(reins_data_dir, task.task_id, task.goal, task.tags)
    fourth = make_task(goal="finish git commit after tests", tags=["git-commit"])
    _write_task(reins_data_dir, fourth.task_id, fourth.goal, fourth.tags)
    _write_memories(
        reins_data_dir,
        [
            make_memory("rule", "Run focused tests before git commit.", ["git-commit"]),
            make_memory("fact", "Git commit hooks may run pytest.", ["git-commit"]),
            make_memory(
                "lesson",
                "When git commit fails, change approach after repeated test errors.",
                ["git-commit"],
            ),
        ],
        applicable_task_tags=["git-commit"],
    )

    results = recall_memories(
        reins_data_dir,
        task_summary=fourth.goal,
        task_tags=fourth.tags,
    )

    assert {item.memory.type for item in results[:3]} == {"rule", "fact", "lesson"}
    fact_result = _find_type(results, "fact")
    expected = (
        0.5 * fact_result.bm25
        + 0.3 * fact_result.tag_jaccard
        + 0.2 * fact_result.recency_bonus
        - 0.5 * fact_result.stale_penalty
    )
    assert fact_result.score == pytest.approx(expected)
    assert 0.0 <= fact_result.tag_jaccard <= 1.0


def test_dod_4_partial_tag_overlap_still_recalls_but_changes_order(
    reins_data_dir: Path,
) -> None:
    _write_memories(
        reins_data_dir,
        [
            make_memory(
                "lesson",
                "Git commit failure requires a focused test before retry.",
                ["git-commit"],
            ),
            make_memory(
                "rule",
                "Use a short git commit checklist before finishing.",
                ["git-commit"],
            ),
            make_memory(
                "fact",
                "Git rebase can replay commit history and change conflict order.",
                ["git-rebase"],
            ),
        ],
        applicable_task_tags=["git-commit"],
    )

    exact = recall_memories(
        reins_data_dir,
        task_summary="git commit failure focused test",
        task_tags=["git-commit"],
    )
    partial = recall_memories(
        reins_data_dir,
        task_summary="git rebase commit history conflict",
        task_tags=["git-rebase"],
    )

    assert {"lesson", "rule", "fact"}.issubset(
        {item.memory.type for item in partial[:3]}
    )
    assert [item.memory.memory_id for item in exact[:3]] != [
        item.memory.memory_id for item in partial[:3]
    ]
    commit_lesson = next(item for item in partial if item.memory.type == "lesson")
    assert commit_lesson.tag_jaccard == 0.0


def test_dod_4_lesson_boost_orders_lesson_above_similar_rule(
    reins_data_dir: Path,
) -> None:
    _write_memories(
        reins_data_dir,
        [
            make_memory(
                "rule",
                "If pytest fails during commit, inspect the failing test.",
                ["pytest"],
            ),
            make_memory(
                "lesson",
                "If pytest fails during commit, inspect the failing test.",
                ["pytest"],
            ),
        ],
        applicable_task_tags=["pytest"],
    )

    results = recall_memories(
        reins_data_dir,
        task_summary="pytest fails during commit",
        task_tags=["pytest"],
    )

    assert [item.memory.type for item in results[:2]] == ["lesson", "rule"]
    assert results[0].score > results[1].score


def test_dod_4_stale_memory_falls_out_of_top_k(reins_data_dir: Path) -> None:
    now = datetime(2026, 5, 4, tzinfo=timezone.utc)
    store = MemoryStore(reins_data_dir)
    source = SessionMessageStore(reins_data_dir).append_message(
        "recall-verification",
        UserMessage(
            "verified-input",
            (TextPart("已核验：pytest tmp_path fixture keeps recall local"),),
        ),
    )
    evidence = (
        MemorySource("user_input", source.entry_id, session_id="recall-verification"),
    )
    for memory_id in ("fresh-a", "fresh-b", "fresh-c"):
        store.create_memory(
            "fact",
            "pytest tmp_path fixture keeps recall local",
            ["testing"],
            memory_id=memory_id,
        )
        store.verify_memory(
            memory_id, (now - timedelta(days=1)).isoformat(), evidence=evidence
        )
    stale_id = store.create_memory(
        "fact",
        "pytest tmp_path fixture keeps recall local",
        ["testing"],
        memory_id="stale",
    )
    store.verify_memory(
        stale_id, (now - timedelta(days=60)).isoformat(), evidence=evidence
    )
    store.close()

    results = recall_memories(
        reins_data_dir,
        task_summary="pytest tmp_path fixture recall",
        task_tags=["testing"],
        now=now,
    )

    assert "stale" not in [item.memory.memory_id for item in results[:3]]
    assert stale_penalty((now - timedelta(days=60)).isoformat(), now) > stale_penalty(
        (now - timedelta(days=1)).isoformat(), now
    )


def _write_task(data_root: Path, task_id: str, goal: str, tags: list[str]) -> None:
    """创建包含真实召回标签的目标原件；参数：数据根、目标身份、目标和标签；返回：无。"""
    store = TaskStore(data_root)
    store.create_task(goal, task_id=task_id)
    with RuntimeStore(data_root).transaction() as batch:
        batch.put("task", task_id, {**store.load_task_payload(task_id), "tags": tags})


def _write_memories(
    data_root: Path,
    memories: list[SyntheticMemory],
    *,
    applicable_task_tags: list[str],
) -> None:
    store = MemoryStore(data_root)
    for memory in memories:
        store.create_memory(
            memory.type,
            memory.content,
            memory.tags,
            applicable_task_tags=applicable_task_tags,
            memory_id=memory.memory_id,
        )
    store.close()


def _find_type(
    results: list[MemoryRecallResult], memory_type: str
) -> MemoryRecallResult:
    return next(item for item in results if item.memory.type == memory_type)
