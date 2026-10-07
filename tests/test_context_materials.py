"""验证大材料缩成真实读取入口，以及有效要求与原件的保留。

作者：xxx
时间：2026-09-25 20:00:00
"""

from scripts.testing.llm import from_test_turns
import json
from contextlib import closing

import pytest

from context.engine import recall_context_materials
from context.materials import ContextMaterial
from context.window import request_budget
from scripts.testing.llm import ScriptedTurnOptions
from llm.messages import (
    AssistantMessage,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from runtime.tool_result_views import ToolResultViews
from skills.store import SkillStore, build_skill_markdown
from tasks.store import TaskStore
from tools.read_artifact import read_artifact
from tools.tool_registry import ToolRegistry


def prepare_materials(root, *, materials=(), extensions=(), window=6000):
    """通过真实请求组装和现有产物写者选择材料；传参：根、材料及窗口；返回：实际请求。"""
    with closing(TaskStore(root)) as tasks:
        task = tasks.create_task("核对当前资料")
    client = from_test_turns([], options=ScriptedTurnOptions(context_window=window))
    views = ToolResultViews(root, task_id=task.task_id, prepare=client.prepare_request)
    context = {
        "session_id": "material-test",
        "context_materials": materials,
        "extension_context": extensions,
        "conversation_history": (
            UserMessage("current", (TextPart("不要联网，继续本地核对"),)),
        ),
        "input_message_id": "current",
        "tool_registry": ToolRegistry(),
        "task_summary_layers": {"intent": "完成本地工作"},
    }
    return views.prepare("继续", context)


@pytest.mark.parametrize("large", [False, True])
def test_skill_keeps_version_and_usable_read_action_when_projected(tmp_path, large):
    """小技能保留正文，长技能改成固定版本入口且原件完整；传参：隔离根与大小；返回：无。"""
    body = "技能正文唯一标记\n" + "逐项核对本地资料\n" * (2000 if large else 1)
    skill = SkillStore(tmp_path).create_skill(
        "local-review", build_skill_markdown(name="本地核对", body=body), meta={}
    )
    notices, materials = recall_context_materials(
        tmp_path, task_summary="核对资料", task_tags=[], skill_refs=["local-review"]
    )
    assert not notices
    composed = prepare_materials(tmp_path, materials=materials)
    text = composed.render_text_to_model
    evidence = composed.trim_delta["materials"]
    selected = next(row for row in evidence if row["source"] == "skill")
    assert selected["representation"] == ("reference" if large else "full")
    assert ("技能正文唯一标记" in text) is not large
    assert skill.version in text and "skill_read" in text
    assert (
        SkillStore(tmp_path).load_skill("local-review", version=skill.version).body
        == skill.body
    )
    assert (
        request_budget(composed.request, composed.context_window).required_total
        <= composed.context_window
    )


def test_extension_reference_has_real_original_and_does_not_evict_requirements(
    tmp_path,
):
    """唯一扩展材料先保存再引用，有效要求不被压掉；传参：隔离根；返回：无。"""
    requirement = ContextMaterial(
        identity="rule",
        source="memory",
        version="v1",
        scope="global",
        text="必须保留的限制：不得上传",
        protected=True,
    )
    extension = "协作提供的长材料🙂\n" * 3500
    composed = prepare_materials(
        tmp_path, materials=(requirement,), extensions=(extension, extension)
    )
    evidence = composed.trim_delta["materials"]
    assert len(evidence) == 2 and "不得上传" in composed.render_text_to_model
    selected = next(row for row in evidence if row["source"] == "extension")
    reference = json.loads(selected["read_reference"])
    assert selected["representation"] == "reference"
    assert (
        read_artifact(
            tmp_path, reference["read_action"]["arguments"]["artifact_id"], mode="full"
        )
        == extension
    )
    assert (
        request_budget(composed.request, composed.context_window).required_total
        <= composed.context_window
    )


def test_explicit_skill_is_kept_before_unpinned_material(tmp_path):
    """窗口只能容纳一个正文时保留显式技能；传参：隔离根；返回：无。"""
    materials = tuple(
        ContextMaterial(
            identity=name,
            source="skill",
            version="v1",
            scope="task",
            text=name + "核对资料" * 1200,
            reference=f"{name}: use skill_read",
            pinned=pinned,
        )
        for name, pinned in (("chosen", True), ("recalled", False))
    )
    composed = prepare_materials(tmp_path, materials=materials, window=8000)
    selected = {
        row["identity"]: row["representation"]
        for row in composed.trim_delta["materials"]
    }
    assert selected == {"chosen": "full", "recalled": "reference"}


def test_actual_layers_account_for_request_and_protected_task_state(tmp_path):
    """各层按最终请求计量，任务与当前输入不列作可删正文；传参：隔离根；返回：无。"""
    composed = prepare_materials(tmp_path, extensions=("很长的扩展\n" * 4000,))
    evidence = composed.trim_delta
    total = (
        sum(row["tokens_est"] for row in evidence["layers"])
        + evidence["layer_estimate_overhead"]
    )
    assert (
        total
        == request_budget(composed.request, composed.context_window).required_total
    )
    assert evidence["task_state_competes_with_history"] is False
    assert (
        "完成本地工作" in composed.render_text_to_model
        and "不要联网" in composed.render_text_to_model
    )
    assert (
        next(row for row in evidence["layers"] if row["name"] == "current")["trimmable"]
        is False
    )


def test_skill_read_body_is_not_reinjected_when_actual_receipt_is_in_request(tmp_path):
    """只按仍在本轮请求中的版本化回执去重；传参：隔离根；返回：无。"""
    from tools.skill_tool import read_skill

    skill = SkillStore(tmp_path).create_skill(
        "guide", build_skill_markdown(name="指南", body="唯一方法正文"), meta={}
    )
    _, materials = recall_context_materials(
        tmp_path, task_summary="资料", task_tags=[], skill_refs=["guide"]
    )
    receipt = read_skill(tmp_path, "guide", version=skill.version)
    with closing(TaskStore(tmp_path)) as tasks:
        task = tasks.create_task("读取指南")
    client = from_test_turns([])
    views = ToolResultViews(
        tmp_path, task_id=task.task_id, prepare=client.prepare_request
    )
    history = (
        UserMessage("input", (TextPart("读取指南"),)),
        AssistantMessage(
            "call", (ToolCallPart("read", "skill_read", {"skill_id": "guide"}),)
        ),
        ToolResultMessage(
            "result",
            "read",
            "skill_read",
            (TextPart(json.dumps(receipt, ensure_ascii=False)),),
            "success",
        ),
    )
    context = {
        "context_materials": materials,
        "conversation_history": history,
        "input_message_id": "input",
        "tool_registry": ToolRegistry(),
    }
    composed = views.prepare("继续", context)
    assert composed.render_text_to_model.count("唯一方法正文") == 1
    assert composed.trim_delta["materials"][0]["representation"] == "in_history"
    refreshed = views.prepare("继续", {**context, "conversation_history": history[:1]})
    assert refreshed.trim_delta["materials"][0]["representation"] == "full"


@pytest.mark.parametrize("window", [7000, 10000])
def test_tool_preview_releases_space_before_skill_body_is_replaced(tmp_path, window):
    """大回执缩短后能放下技能正文时保留正文和原件入口；传参：隔离根与窗口；返回：无。"""
    with closing(TaskStore(tmp_path)) as tasks:
        task = tasks.create_task("按技能核对工具资料")
    skill = ContextMaterial(
        identity="guide",
        source="skill",
        version="v1",
        scope="task",
        text="技能正文必须保留\n" + "逐项验证条件\n" * 100,
        reference="guide: use skill_read",
        pinned=True,
    )
    receipt = ToolResultMessage(
        "result", "read", "probe", (TextPart("很长的原始资料\n" * 5000),), "success"
    )
    history = (
        UserMessage("input", (TextPart("不要联网，按技能核对资料"),)),
        AssistantMessage("call", (ToolCallPart("read", "probe", {}),)),
        receipt,
    )
    client = from_test_turns([], options=ScriptedTurnOptions(context_window=window))
    views = ToolResultViews(
        tmp_path, task_id=task.task_id, prepare=client.prepare_request
    )
    context = {
        "context_materials": (skill,),
        "conversation_history": history,
        "input_message_id": "input",
        "tool_registry": ToolRegistry(),
    }
    composed = views.prepare("继续", context)
    assert composed.trim_delta["materials"][0]["representation"] == "full"
    assert "技能正文必须保留" in composed.render_text_to_model
    assert request_budget(composed.request, window).required_total <= window
    projected = next(
        message
        for message in composed.messages
        if isinstance(message, ToolResultMessage)
    )
    reference = json.loads(projected.content[0].text)["read_full_message"][
        "artifact_id"
    ]
    original = json.loads(read_artifact(tmp_path, reference, mode="full"))
    assert original["content"][0]["text"] == receipt.content[0].text
    assert context["conversation_history"] == history
