from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from context.engine import build_context_sections_for_tests
from memory.store import MemoryStore
from runtime.run_facts import RunFactStore
from tasks.store import TaskStore


def test_build_context_sections_for_tests_emits_skipped_memory_type(
    tmp_path: Path,
) -> None:
    task_id = "2026-06-03-memory-score"
    TaskStore(tmp_path).create_task("pytest memory score", task_id=task_id)
    store = MemoryStore(tmp_path)
    for index in range(4):
        store.create_memory(
            "fact",
            f"pytest memory fact {index}",
            ["pytest"],
            memory_id=f"fact-{index}",
        )
    store.create_memory(
        "fact",
        "pytest stale memory",
        ["pytest"],
        memory_id="fact-stale",
    )
    store.close()

    build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "pytest memory"},
        lease=SimpleNamespace(max_tokens=1000),
        task_relevant=True,
        session_id="session-memory-score",
        run_id="run-memory-score",
        data_root=tmp_path,
        project_root=tmp_path,
    )

    facts = RunFactStore(tmp_path).read_run("run-memory-score")
    breakdown = next(
        fact for fact in facts if fact["event"] == "memory:score_breakdown"
    )
    skipped = breakdown["skipped"]

    # 断言的是「skipped 条目带得出 reason 与 type 字段」这个结构契约。
    # 原先一并要求 "stale"，那是 fact 超 180 天被硬闸剔除时的 reason；
    # 08-06 卡删掉那道闸后「久未复核」只影响打分、不再产生 skip，
    # 故只留 type_limit 作载体（5 条 fact 撞 limit=3 稳定产出它），结构断言不放宽
    assert {item["reason"] for item in skipped} >= {"type_limit"}
    assert {item["type"] for item in skipped} == {"fact"}
