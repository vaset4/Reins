"""验证方法修订、指定版本取用与撤回，不把保存或脚本退出当成任务效果。

作者：xxx
时间：2026-09-15 02:25:05
"""

from __future__ import annotations

import pytest

from memory.records import MemorySource
from skills.store import SkillStore, build_skill_markdown
from tools.skill_tool import read_skill, skill_version


def test_existing_skill_requires_revision_and_keeps_original(tmp_path):
    """同名方法修改必须保留前版和理由；传参：隔离目录；返回：无。"""
    store = SkillStore(tmp_path)
    old = store.create_skill(
        "totals", build_skill_markdown(name="汇总", body="将金额相加")
    )
    old_bytes = (old.root / "SKILL.md").read_bytes()
    with pytest.raises(ValueError, match="revise"):
        store.create_skill(
            "totals", build_skill_markdown(name="汇总", body="扣除退款后相加")
        )
    revised = store.revise_skill(
        "totals",
        build_skill_markdown(name="汇总", body="扣除退款后相加"),
        expected_version=skill_version(old),
        reason="原方法遗漏退款",
        sources=(MemorySource("user_input", "feedback-1"),),
    )
    assert revised.previous_version == skill_version(old)
    assert (old.root / "SKILL.md").read_bytes() == old_bytes
    assert read_skill(tmp_path, "totals")["version"] == skill_version(old)
    assert (
        read_skill(tmp_path, "totals", version=revised.version, trial=True)["body"]
        == "扣除退款后相加"
    )
    store.publish_version("totals", revised.version, reason="显式试用未评估的方法")
    assert read_skill(tmp_path, "totals")["version"] == revised.version
    assert read_skill(tmp_path, "totals")["evaluation_status"] == "not_evaluated"


def test_withdrawal_blocks_new_use_without_rewriting_past_version(tmp_path):
    """撤回影响新请求，不自动切回其他版本，也不抹掉原文；传参：目录；返回：无。"""
    store = SkillStore(tmp_path)
    old = store.create_skill("method", build_skill_markdown(name="方法", body="旧方法"))
    new = store.revise_skill(
        "method",
        build_skill_markdown(name="方法", body="有问题的新方法"),
        expected_version=old.version,
        reason="反馈修订",
        sources=(MemorySource("user_input", "feedback"),),
        publish=True,
    )
    store.withdraw_version("method", new.version, reason="独立案例失败")
    with pytest.raises(ValueError, match="withdrawn"):
        read_skill(tmp_path, "method", version=new.version)
    assert store.load_skill("method", version=new.version).body == "有问题的新方法"
    assert store.list_skills(active_only=True) == []
    assert read_skill(tmp_path, "method", version=old.version)["body"] == "旧方法"


def test_guide_loading_and_script_exit_do_not_validate_task_effect(tmp_path):
    """内容取用和脚本退出分别记账，均不推断用户目标已经达到；传参：目录；返回：无。"""
    store = SkillStore(tmp_path)
    skill = store.create_skill(
        "noop",
        build_skill_markdown(name="空操作", body="尚待验证"),
        script="def main(args):\n    return 'done'\n",
        meta={"script_entry": "script.py:main", "validation": "作者声称验证过"},
    )
    version, raw = skill.version, (skill.root / "SKILL.md").read_bytes()
    store.touch_skill("noop")
    store.update_skill_stats("noop", success=True, version=version)
    loaded = read_skill(tmp_path, "noop", version=version)
    assert loaded["evaluation_status"] == "not_evaluated"
    assert loaded["evidence_summary"]["script_exit_zero_count"] == 1
    resource = next(item for item in loaded["resources"] if item["name"] == "script.py")
    assert resource["read_action"] == {
        "tool": "file_read",
        "path": str(skill.root / "script.py"),
    }
    assert loaded["evidence_summary"]["task_outcomes"] == []
    assert store.load_skill("noop").version == version
    assert (skill.root / "SKILL.md").read_bytes() == raw


def test_duplicate_revision_does_not_publish_twice(tmp_path):
    """同一后处理请求重投返回原修订，不增加发布事实；传参：目录；返回：无。"""
    store = SkillStore(tmp_path)
    old = store.create_skill("method", build_skill_markdown(name="方法", body="原方法"))
    markdown = build_skill_markdown(name="方法", body="加入反例检查")
    options = {
        "expected_version": old.version,
        "reason": "纳入真实反例",
        "sources": (MemorySource("tool_result", "result-1"),),
        "change_id": "postprocess-1",
        "publish": True,
    }
    first = store.revise_skill("method", markdown, **options)
    second = store.revise_skill("method", markdown, **options)
    assert first.version == second.version
    assert len(store.list_versions("method")) == 2


@pytest.mark.parametrize("action", ["create", "revise", "publish"])
def test_invalid_script_cannot_replace_published_method(tmp_path, action):
    """截断脚本必须在发布前明确失败，完整草案可保存但不执行；传参：目录和发布路径；返回：无。"""
    store = SkillStore(tmp_path)
    markdown = build_skill_markdown(name="净额", body="按类别汇总净额")
    bad_script = 'def main(args):\n    if args["kind"] == "refun'
    original = store.create_skill(
        "method",
        markdown,
        script="raise RuntimeError('must not execute during publication')\n",
    )
    if action == "publish":
        draft = store.revise_skill(
            "method",
            markdown,
            expected_version=original.version,
            reason="未完成草案",
            sources=(MemorySource("user_input", "feedback"),),
            script=bad_script,
        )
    with pytest.raises(ValueError, match="script syntax invalid"):
        if action == "create":
            store.create_skill("broken", markdown, script=bad_script)
        elif action == "revise":
            store.revise_skill(
                "method",
                markdown,
                expected_version=original.version,
                reason="截断输出",
                sources=(MemorySource("user_input", "feedback"),),
                script=bad_script,
                publish=True,
            )
        else:
            store.publish_version("method", draft.version, reason="尝试发布草案")
    assert store.load_skill("method").version == original.version
    assert not store.skill_exists("broken")
