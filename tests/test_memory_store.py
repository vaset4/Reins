from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory.store import MEMORY_STATE_ACTIVE, MEMORY_STATE_DRAFT, MemoryStore


def test_memory_store_crud_round_trip_and_index(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    memory_id = store.create_memory(
        "rule",
        "Always run the focused test before the full suite.",
        ["testing", "phase2"],
        applicable_task_tags=["llm"],
        memory_id="01HXMEMORY00000000000000001",
    )

    loaded = store.load_memory(memory_id)

    assert loaded.type == "rule"
    assert loaded.state == MEMORY_STATE_ACTIVE
    assert loaded.content == "Always run the focused test before the full suite."
    assert loaded.tags == ["testing", "phase2"]
    assert loaded.applicable_task_tags == ["llm"]
    assert (tmp_path / "index.sqlite").is_file()
    assert store.current_path(memory_id).parent == tmp_path / "global" / "memory"

    conn = sqlite3.connect(tmp_path / "index.sqlite")
    row = conn.execute(
        "SELECT type, state, tags FROM memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    assert row == ("rule", "active", '["testing", "phase2"]')


def test_memory_store_lists_by_state_type_and_tags(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.create_memory("fact", "pytest passed", ["testing"], memory_id="fact-1")
    store.create_memory("rule", "keep changes small", ["testing"], memory_id="rule-1")

    records = store.list_memories(type="rule", tags=["testing"])

    assert [record.memory_id for record in records] == ["rule-1"]


def test_memory_state_machine_rejects_invalid_transition(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    memory_id = store.create_memory(
        "fact", "draft fact", ["draft"], memory_id="draft-1", state=MEMORY_STATE_DRAFT
    )

    active = store.update_memory_state(memory_id, MEMORY_STATE_ACTIVE)

    assert active.state == MEMORY_STATE_ACTIVE
    with pytest.raises(ValueError, match="invalid memory state transition"):
        store.update_memory_state(memory_id, MEMORY_STATE_DRAFT)


def test_archive_and_restore_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    memory_id = store.create_memory("lesson", "avoid stale fixes", ["debug"])

    assert store.archive_memory(memory_id).state == "archived"
    assert store.restore_memory(memory_id).state == "active"


def test_memory_index_can_be_rebuilt_from_records(tmp_path: Path) -> None:
    from memory.store import rebuild_memory_index

    store = MemoryStore(tmp_path)
    memory_id = store.create_memory("fact", "index rebuild works", ["index"])
    store.close()
    with sqlite3.connect(tmp_path / "index.sqlite") as conn:
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM memories_fts")

    assert rebuild_memory_index(tmp_path) == 1
    conn = sqlite3.connect(tmp_path / "index.sqlite")
    row = conn.execute(
        "SELECT state FROM memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    assert row == ("active",)


def test_failed_rebuild_keeps_existing_index(tmp_path: Path, monkeypatch) -> None:
    from memory.index import MemoryIndexState
    from memory.store import rebuild_memory_index

    store = MemoryStore(tmp_path)
    memory_id = store.create_memory("fact", "keep the old index", ["index"])
    store.close()
    index_path = tmp_path / "index.sqlite"
    before = (
        sqlite3.connect(index_path)
        .execute("SELECT memory_id, content FROM memories_fts")
        .fetchall()
    )

    monkeypatch.setattr(
        "memory.store.inspect_index",
        lambda connection, records: MemoryIndexState("stale", stale=(memory_id,)),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        rebuild_memory_index(tmp_path)

    after = (
        sqlite3.connect(index_path)
        .execute("SELECT memory_id, content FROM memories_fts")
        .fetchall()
    )
    assert after == before
    assert not list((tmp_path / "memory").glob("index.rebuild-*.db"))


def test_chinese_index_rebuild_keeps_canonical_memory(tmp_path: Path) -> None:
    """旧分词库显式升级，保留原件与已打开连接；传参：临时目录；返回：无。"""
    from contextlib import closing

    from context.memory_recall import recall_memories_with_outcome
    from memory.index import TOKENIZER_VERSION
    from memory.store import rebuild_memory_index

    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "preference", "用户饮食偏好：不吃辣，晚餐清淡", [], memory_id="meal"
        )
        original = store.load_memory(identity)
        with closing(sqlite3.connect(tmp_path / "index.sqlite")) as conn:
            with conn:
                conn.execute(
                    "UPDATE memories_fts SET content = ?",
                    (store.load_memory(identity).content,),
                )
                conn.execute(
                    "UPDATE runtime_meta SET value='0' WHERE key='memory_tokenizer_version'"
                )
        assert store.index_status().state == "stale"
        assert store.index_status().tokenizer_version == 0

        assert rebuild_memory_index(tmp_path) == 1

        assert store.index_status().state == "current"
        assert store.index_status().tokenizer_version == TOKENIZER_VERSION
        assert store.load_memory(identity) == original
        for query in ("饮食偏好", "晚餐"):
            outcome = recall_memories_with_outcome(
                tmp_path, task_summary=query, task_tags=[]
            )
            assert any(
                item.memory.memory_id == identity and item.bm25 > 0
                for item in outcome.selected
            )


def test_publish_failure_rolls_back_whole_index(tmp_path: Path) -> None:
    """发布中途失败必须恢复旧行与分词版本；传参：临时目录；返回：无。"""
    from contextlib import closing

    from memory.store import rebuild_memory_index

    with closing(MemoryStore(tmp_path)) as store:
        store.create_memory(
            "fact", "preserve existing search rows", [], memory_id="original"
        )
    with closing(sqlite3.connect(tmp_path / "index.sqlite")) as conn:
        with conn:
            conn.execute(
                "UPDATE runtime_meta SET value='0' WHERE key='memory_tokenizer_version'"
            )
            conn.execute(
                "CREATE TRIGGER fail_publish BEFORE INSERT ON memories BEGIN SELECT RAISE(ABORT, 'publication denied'); END"
            )
        before = conn.execute("SELECT * FROM memories_fts").fetchall()

        with pytest.raises(sqlite3.IntegrityError, match="publication denied"):
            rebuild_memory_index(tmp_path)

        assert conn.execute("SELECT * FROM memories_fts").fetchall() == before
        assert conn.execute("SELECT memory_id FROM memories").fetchall() == [
            ("original",)
        ]
        assert conn.execute(
            "SELECT value FROM runtime_meta WHERE key='memory_tokenizer_version'"
        ).fetchone() == ("0",)
    assert not list((tmp_path / "memory").glob("index.rebuild-*.db"))
