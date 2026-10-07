"""读取已提交知识时，另一个写者准备新版不能使无关会话失败。

作者：xxx
时间：2026-09-15 03:40:00
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

from context.memory_recall import recall_memories
from context.skill_recall import recall_skills
from memory.store import MemoryStore
from schedules.persistence import claim_file
from skills.store import SkillStore, build_skill_markdown


def test_committed_memory_remains_readable_while_revision_writer_owns_lock(tmp_path):
    """写者尚未发布时仍可读取完整旧版，使用统计不占内容写锁；传参：目录；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory("fact", "budget is 620", ["budget"])
        with ThreadPoolExecutor(max_workers=1) as pool, store.locked():
            result = pool.submit(
                recall_memories, tmp_path, task_summary="budget", task_tags=["budget"]
            ).result(timeout=5)
        assert [row.memory.memory_id for row in result] == [identity]


def test_published_method_remains_readable_while_new_version_is_prepared(tmp_path):
    """方法候选写入不能占住已发布版本的读取和使用统计；传参：目录；返回：无。"""
    store = SkillStore(tmp_path)
    old = store.create_skill(
        "net", build_skill_markdown(name="net", body="sum charges and subtract refunds")
    )
    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        claim_file(tmp_path / "skills" / "net" / ".write.lock") as acquired,
    ):
        assert acquired
        result = pool.submit(
            recall_skills, tmp_path, task_summary="charges refunds", task_tags=[]
        ).result(timeout=5)
    assert [row.skill.version for row in result] == [old.version]
