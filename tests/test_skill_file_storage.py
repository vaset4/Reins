"""【技能】【文件原件】验证发布、撤回及统计不依赖SQLite唯一事实。

作者：xxx
时间：2026-09-30 17:00:00
"""

from pathlib import Path
import shutil

import pytest

from memory.records import MemorySource
from skills.store import SkillStore, build_skill_markdown
from runtime.persistence import RuntimeStore, SourceCorruptionError


def test_skill_publication_and_usage_survive_index_removal(tmp_path: Path) -> None:
    """索引重建后保持发布状态、来源、修订与使用统计；参数：隔离根；返回：无。"""
    store = SkillStore(tmp_path)
    first = store.create_skill(
        "method", build_skill_markdown(name="方法", body="原方法"), script="print('ok')"
    )
    second = store.revise_skill(
        "method",
        build_skill_markdown(name="方法", body="改进方法"),
        expected_version=first.version,
        reason="用户更正",
        sources=(MemorySource("user_input", "input-2"),),
        publish=True,
    )
    raw = (second.root / "SKILL.md").read_bytes()
    store.touch_skill("method", version=second.version)
    store.update_skill_stats("method", success=True, version=second.version)
    store.record_outcome(
        "method",
        second.version,
        outcome="achieved",
        case_id="case-1",
        reason="结果核验通过",
        sources=(MemorySource("tool_result", "operation-2"),),
        event_id="outcome-1",
    )
    store.withdraw_version("method", second.version, reason="发现新问题")
    (tmp_path / "index.sqlite").unlink()
    reopened = SkillStore(tmp_path)
    assert reopened.list_skills(active_only=True) == []
    loaded = reopened.load_skill("method", version=second.version)
    assert loaded.frontmatter.state == "withdrawn"
    assert loaded.previous_version == first.version and loaded.sources == second.sources
    assert loaded.evidence_summary["selection_count"] == 1
    assert loaded.evidence_summary["script_run_count"] == 1
    assert loaded.evidence_summary["task_outcomes"][0]["outcome"] == "achieved"
    assert (
        loaded.version == second.version
        and (loaded.root / "SKILL.md").read_bytes() == raw
    )
    assert "withdrawn" in (tmp_path / "skills" / "method" / "events.jsonl").read_text(
        encoding="utf-8"
    )


def test_copy_source_space_without_index_keeps_skill_resources(tmp_path: Path) -> None:
    """停写后复制文件空间可读取技能历史；参数：隔离根；返回：无。"""
    original, copied = tmp_path / "original", tmp_path / "copied"
    store = SkillStore(original)
    skill = store.create_skill(
        "reader",
        build_skill_markdown(name="阅读", body="先读实际内容"),
        resources={"reference.txt": "资料正文".encode()},
    )
    store.archive_skill("reader")
    shutil.copytree(original, copied, ignore=shutil.ignore_patterns("index.sqlite*"))
    reopened = SkillStore(copied).load_skill("reader")
    assert (
        reopened.version == skill.version and reopened.frontmatter.state == "archived"
    )
    assert (reopened.root / "reference.txt").read_text(encoding="utf-8") == "资料正文"


def test_missing_skill_resource_is_not_empty_knowledge(tmp_path: Path) -> None:
    """已发布资源丢失时明确失败；参数：隔离根；返回：无。"""
    store = SkillStore(tmp_path)
    skill = store.create_skill("reader", build_skill_markdown(name="阅读", body="指南"))
    (skill.root / "SKILL.md").unlink()
    (tmp_path / "index.sqlite").unlink()
    with pytest.raises(FileNotFoundError):
        SkillStore(tmp_path).list_skills(active_only=True)


def test_rebuild_rejects_missing_committed_skill_resource(tmp_path: Path) -> None:
    """完整源重建必须验证技能资源，不能丢资源仍报ready；参数：隔离根；返回：无。"""
    store = SkillStore(tmp_path)
    skill = store.create_skill(
        "reader", build_skill_markdown(name="阅读", body="已发布指南")
    )
    (skill.root / "SKILL.md").unlink()
    runtime = RuntimeStore(tmp_path)
    with pytest.raises(SourceCorruptionError, match="missing"):
        runtime.rebuild_index(force=True)
    assert runtime.index_status["state"] == "failed"
