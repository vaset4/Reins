from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.repl.slash_commands import (
    ReplState,
    SlashCommandContext,
    create_default_registry,
)
from memory.store import MEMORY_STATE_DRAFT, MemoryStore
from tools.builtin_tools import build_tool_registry


def test_cli_memory_approve_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """草稿转正的老路子必须清晰失败，不能静默成功

    manual 复核退役后没有「批准草稿」这回事。有人照旧习惯或旧文档去跑
    `reins memory approve`，要拿到明确的失败退出，而不是「命令跑了但什么也没发生」。
    断言只钉 approve 这个子动作，不钉整个 memory 前缀（降低将来加别的子命令时假红的概率）。
    """
    from app import cli

    project = tmp_path / "project"
    data_root = project / ".reins" / "data"
    project.mkdir()
    memory_id = MemoryStore(data_root).create_memory(
        "rule", "legacy draft rule", ["review"], state=MEMORY_STATE_DRAFT
    )
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(project))
    monkeypatch.setattr(sys, "argv", ["reins", "memory", "approve", memory_id])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code != 0
    # 草稿原样留着，没有被悄悄转正
    assert MemoryStore(data_root).load_memory(memory_id).state == MEMORY_STATE_DRAFT


def test_cli_memory_migrate_experience_is_retired_without_rewriting_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧 experience 迁移入口必须失败，并保留未知历史数据"""
    from app import cli

    project = tmp_path / "project"
    memory_dir = project / ".reins" / "data" / "memory"
    memory_dir.mkdir(parents=True)
    legacy_file = memory_dir / "legacy.md"
    legacy_file.write_text(
        "---\n"
        "memory_id: legacy\n"
        "type: experience\n"
        "state: active\n"
        "tags: []\n"
        "created_at: '2026-06-02T00:00:00+00:00'\n"
        "updated_at: '2026-06-02T00:00:00+00:00'\n"
        "last_verified_at: '2026-06-02T00:00:00+00:00'\n"
        "applicable_task_tags: []\n"
        "---\n"
        "Legacy memory.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(project))
    monkeypatch.setattr(sys, "argv", ["reins", "memory", "migrate-experience"])

    original_bytes = legacy_file.read_bytes()
    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code != 0
    assert legacy_file.read_bytes() == original_bytes


def test_repl_memory_approve_is_unknown_command(tmp_path: Path) -> None:
    """REPL 侧同理：/memory approve 不该静默成功

    与 CLI 那条各守一条路（斜杠命令 / 命令行），不重复。
    """
    project = tmp_path / "project"
    data_root = project / ".reins" / "data"
    project.mkdir()
    store = MemoryStore(data_root)
    memory_id = store.create_memory(
        "preference", "legacy draft preference", ["review"], state=MEMORY_STATE_DRAFT
    )
    ctx = SlashCommandContext(
        repl_state=ReplState(session_id="session-memory"),
        store=None,
        registry=build_tool_registry(repo_root=project, data_root=data_root),
        llm_client=None,
        project_root=project,
        data_root=data_root,
        prompt_fn=lambda _prompt: "",
    )

    result = create_default_registry().dispatch(f"/memory approve {memory_id}", ctx)

    assert result is not None
    assert "Unknown command" in result.message
    # 草稿原样留着，没有被悄悄转正
    assert MemoryStore(data_root).load_memory(memory_id).state == MEMORY_STATE_DRAFT
