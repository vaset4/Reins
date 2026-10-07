"""验证主动提炼与原运行分离，重启交付不重复发布且来源边界固定。

作者：xxx
时间：2026-09-15 02:25:05
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

from contextlib import closing
from datetime import datetime, timezone

import pytest

from app.run_task import run_task
from app.scheduled_run import create_scheduler
from llm.messages import ToolCallPart, ToolResultMessage, model_visible_text
from runtime.session_messages import append_user_message
from schedules.notifications import NotificationStore
from schedules.store import ScheduleStore
from skills.store import SkillStore, build_skill_markdown
from tests.test_session_runtime import capture_requests
from tools.builtin_tools import build_tool_registry


@pytest.fixture(autouse=True)
def isolate_explicit_reflection(tmp_path):
    """本组只验证显式提炼，自动维护由专门生产链测试覆盖；参数：隔离根；返回：无。"""
    from runtime.knowledge_maintenance import KnowledgeMaintenance

    KnowledgeMaintenance(tmp_path).configure(enabled=False)


def registry_for_jobs(root):
    """显式加载本测试选定的真实工具定义；传参：隔离根；返回：生产注册表。"""
    registry = build_tool_registry(repo_root=root, data_root=root)
    definitions = [
        registry.get(name)
        for name in (
            "schedule",
            "knowledge_reflect",
            "knowledge_read",
            "skill_manage",
            "skill_read",
        )
    ]
    for definition in definitions:
        definition.deferred = False
    registry.publish(
        definitions, replace_names=tuple(item.name for item in definitions)
    )
    return registry


def request_reflection(
    root, monkeypatch, *, publish=True, entrypoint="knowledge_reflect"
):
    """通过生产会话接纳一次提炼，初始方法由明确作者操作发布；传参：目录与发布意图；返回：运行和原版本。"""
    request = ToolCallPart(
        "reflect",
        "knowledge_reflect",
        {
            "objective": "汇总订单时纳入退款负数，完善已有方法",
            "skill_id": "order-totals",
            "publish": publish,
        },
    )
    if entrypoint == "schedule":
        request = ToolCallPart(
            "reflect",
            "schedule",
            {
                "action": "create",
                "name": "后台修订订单方法",
                "kind": "work",
                "timezone": "UTC",
                "time": f"at:{datetime.now(timezone.utc).isoformat()}",
                "prompt": "纳入退款，修订订单汇总方法并发布，保留用户出处",
            },
        )
    parent = from_test_native_tool_then_final([request], "提炼已交给后台")
    requests = capture_requests(parent, monkeypatch)
    # 1. 首次生产启动先初始化空数据根，随后在同一根写入作者方法
    from runtime.schema_meta import ensure_current_schema

    ensure_current_schema(root)
    old = SkillStore(root).create_skill(
        "order-totals", build_skill_markdown(name="订单汇总", body="只把正金额相加")
    )
    response = run_task(
        "订单汇总遗漏退款，应把退款作为负数；请提炼这次经验",
        root,
        data_root=root,
        llm_client=parent,
        tool_registry=registry_for_jobs(root),
    )
    assert response.status == "done" and len(requests) == 2
    return response, old


def test_reflection_uses_frozen_sources_and_resume_does_not_republish(
    tmp_path, monkeypatch
):
    """后处理读取原范围，交付中断只补通知，不再次运行模型或发布版本；传参：目录/替换器；返回：无。"""
    parent, old = request_reflection(tmp_path, monkeypatch)
    with closing(ScheduleStore(tmp_path)) as store:
        jobs = store.list_all_schedules()
    assert len(jobs) == 1 and jobs[0].source_run_id == parent.run_id
    assert jobs[0].target_task_id is None
    source_session = jobs[0].source_session_id
    assert len(SkillStore(tmp_path).list_versions("order-totals")) == 1
    append_user_message(
        tmp_path, source_session, "后加材料：这句话不属于已接纳的提炼范围"
    )
    worker = from_test_native_tool_then_final(
        [
            ToolCallPart("source", "knowledge_read", {"action": "user_inputs"}),
            ToolCallPart(
                "revision",
                "skill_manage",
                {
                    "action": "revise",
                    "skill_id": "order-totals",
                    "expected_version": old.version,
                    "body": "净额等于付款总额减去退款总额；同时核对条数",
                    "reason": "原例遗漏退款",
                    "source_mode": "origin_inputs",
                    "publish": True,
                },
            ),
        ],
        "已保存新版，仍待独立案例评估",
    )
    requests = capture_requests(worker, monkeypatch)
    original = NotificationStore.enqueue

    def lose_ack(store, *args, **kwargs):
        """实际保存通知后丢失回执；传参：通知接纳；返回：抛出明确交付故障。"""
        original(store, *args, **kwargs)
        raise OSError("lost knowledge notification acknowledgement")

    with closing(
        create_scheduler(
            project_root=tmp_path,
            data_root=tmp_path,
            llm_factory=lambda _: worker,
            registry_factory=lambda: registry_for_jobs(tmp_path),
        )
    ) as scheduler:
        with monkeypatch.context() as fault:
            fault.setattr(NotificationStore, "enqueue", lose_ack)
            with pytest.raises(OSError, match="acknowledgement"):
                scheduler.run_due_jobs(now=datetime.now(timezone.utc))
    revised = SkillStore(tmp_path).load_skill("order-totals")
    assert (
        revised.previous_version == old.version
        and revised.evaluation_status == "not_evaluated"
    )
    assert revised.sources[0].session_id == source_session
    assert "后加材料" not in str(requests[-1])
    assert "订单汇总遗漏退款" in str(requests[-1])
    import json

    receipt = next(
        message
        for message in requests[-1].messages
        if isinstance(message, ToolResultMessage)
        and message.tool_name == "knowledge_read"
    )
    original_inputs = json.loads(json.loads(model_visible_text(receipt))["output"])[
        "messages"
    ]
    assert original_inputs and all(
        message["kind"] == "user" for message in original_inputs
    )

    def no_repeat(_options):
        """已保存方法后的交付恢复不再创建模型；传参：配置；返回：若调用即失败。"""
        raise AssertionError("completed reflection must not run twice")

    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=no_repeat
        )
    ) as restarted:
        assert (
            restarted.run_due_jobs(now=datetime.now(timezone.utc))[0].status
            == "succeeded"
        )
    assert SkillStore(tmp_path).load_skill("order-totals").version == revised.version
    assert len(SkillStore(tmp_path).list_versions("order-totals")) == 2


def test_draft_only_reflection_cannot_publish_by_changing_model_arguments(
    tmp_path, monkeypatch
):
    """未授权发布的提炼仍可保存候选，但不能自行扩大为启用；传参：目录/替换器；返回：无。"""
    _, old = request_reflection(tmp_path, monkeypatch, publish=False)
    worker = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "revision",
                "skill_manage",
                {
                    "action": "revise",
                    "skill_id": "order-totals",
                    "expected_version": old.version,
                    "body": "未核验的新方法",
                    "reason": "模型尝试直接发布",
                    "source_mode": "origin_inputs",
                    "publish": True,
                },
            )
        ],
        "发布请求已被拒绝",
    )
    requests = capture_requests(worker, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path,
            data_root=tmp_path,
            llm_factory=lambda _: worker,
            registry_factory=lambda: registry_for_jobs(tmp_path),
        )
    ) as scheduler:
        scheduler.run_due_jobs(now=datetime.now(timezone.utc))
    assert "allows drafts only" in str(requests[-1])
    assert SkillStore(tmp_path).load_skill("order-totals").version == old.version


def test_reflection_rejects_branch_anchor_and_explains_valid_source_selection(
    tmp_path, monkeypatch
):
    """分支锚点不能冒充用户，失败后可按明确指引引用原输入；传参：目录与替换器；返回：无。"""
    _, old = request_reflection(tmp_path, monkeypatch)
    with closing(ScheduleStore(tmp_path)) as store:
        job = store.list_all_schedules()[0]
    anchor = job.knowledge_origin["source_entry_id"]
    assert anchor not in job.prompt
    args = {
        "action": "revise",
        "skill_id": "order-totals",
        "expected_version": old.version,
        "body": "净额为付款减退款",
        "reason": "依据用户更正",
        "source_mode": "origin_inputs",
        "publish": True,
    }
    worker = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "invalid-source", "skill_manage", {**args, "source_input_ids": [anchor]}
            ),
            ToolCallPart("valid-source", "skill_manage", args),
        ],
        "已按真实来源修订",
    )
    requests = capture_requests(worker, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path,
            data_root=tmp_path,
            llm_factory=lambda _: worker,
            registry_factory=lambda: registry_for_jobs(tmp_path),
        )
    ) as scheduler:
        scheduler.run_due_jobs(now=datetime.now(timezone.utc))
    assert "omit source_input_ids" in str(requests[-1])
    revised = SkillStore(tmp_path).load_skill("order-totals")
    assert revised.previous_version == old.version
    assert revised.sources[0].reference != anchor


def test_normal_schedule_can_revise_from_its_frozen_user_source(tmp_path, monkeypatch):
    """普通后台任务也能引用原始用户要求，后加输入不混入来源；传参：目录与替换器；返回：无。"""
    from approval import ApprovalDecision
    from tests.support.approval import install_approval

    install_approval(monkeypatch, lambda _request: ApprovalDecision.ONCE)
    parent, old = request_reflection(tmp_path, monkeypatch, entrypoint="schedule")
    with closing(ScheduleStore(tmp_path)) as store:
        job = store.list_all_schedules()[0]
    assert job.knowledge_origin is None
    append_user_message(
        tmp_path, job.source_session_id, "后加材料：不能混入此前已接纳的范围"
    )
    worker = from_test_native_tool_then_final(
        [
            ToolCallPart("read-origin", "knowledge_read", {"action": "user_inputs"}),
            ToolCallPart(
                "revise",
                "skill_manage",
                {
                    "action": "revise",
                    "skill_id": "order-totals",
                    "expected_version": old.version,
                    "body": "付款减去退款得到净额",
                    "reason": "原用户指出漏算退款",
                    "source_mode": "origin_inputs",
                    "publish": True,
                },
            ),
        ],
        "已修订，尚未独立验证",
    )
    requests = capture_requests(worker, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path,
            data_root=tmp_path,
            llm_factory=lambda _: worker,
            registry_factory=lambda: registry_for_jobs(tmp_path),
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=datetime.now(timezone.utc))[0]
    revised = SkillStore(tmp_path).load_skill("order-totals")
    assert revised.previous_version == old.version
    assert revised.sources[0].session_id == job.source_session_id
    assert revised.sources[0].run_id == parent.run_id
    assert "订单汇总遗漏退款" in str(requests[-1])
    assert "后加材料" not in str(requests[-1])
    assert all(
        message.status == "success"
        for message in requests[-1].messages
        if isinstance(message, ToolResultMessage)
    )
    assert result.status == "succeeded" and revised.evaluation_status == "not_evaluated"


@pytest.mark.parametrize("entrypoint", ["schedule", "knowledge_reflect"])
def test_schedule_update_preserves_sources_of_already_accepted_work(
    tmp_path, monkeypatch, entrypoint
):
    """修改未来工作使用新要求，已接纳发生仍读取原分支；传参：目录、捕获器和入口；返回：无。"""
    from approval import ApprovalDecision
    from tests.support.approval import install_approval

    install_approval(monkeypatch, lambda _request: ApprovalDecision.ONCE)
    parent, _ = request_reflection(tmp_path, monkeypatch, entrypoint=entrypoint)
    with closing(
        create_scheduler(project_root=tmp_path, data_root=tmp_path)
    ) as scheduler:
        accepted = scheduler.accept_due(now=datetime.now(timezone.utc))[0]
    coordinator = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "update",
                "schedule",
                {
                    "action": "update",
                    "schedule_id": accepted.schedule_id,
                    "prompt": "新任务只核对重复订单，不修改方法",
                    "time": f"at:{datetime.now(timezone.utc).isoformat()}",
                },
            )
        ],
        "未来工作已修改",
    )
    changed = run_task(
        "未来改为检查重复订单，刚才已接纳的工作照原要求完成",
        tmp_path,
        data_root=tmp_path,
        llm_client=coordinator,
        tool_registry=registry_for_jobs(tmp_path),
    )
    assert changed.status == "done"
    with closing(ScheduleStore(tmp_path)) as store:
        future = store.load_schedule(accepted.schedule_id)
    assert future.source_run_id == changed.run_id
    assert accepted.schedule_snapshot["source_run_id"] == parent.run_id
    old_reader = from_test_native_tool_then_final(
        [ToolCallPart("original", "knowledge_read", {"action": "user_inputs"})],
        "已读取原要求",
    )
    old_requests = capture_requests(old_reader, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path,
            data_root=tmp_path,
            llm_factory=lambda _: old_reader,
            registry_factory=lambda: registry_for_jobs(tmp_path),
        )
    ) as scheduler:
        assert scheduler.run_occurrence(accepted.occurrence_id).status == "succeeded"
    new_reader = from_test_native_tool_then_final(
        [ToolCallPart("updated", "knowledge_read", {"action": "user_inputs"})],
        "已读取新要求",
    )
    new_requests = capture_requests(new_reader, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path,
            data_root=tmp_path,
            llm_factory=lambda _: new_reader,
            registry_factory=lambda: registry_for_jobs(tmp_path),
        )
    ) as scheduler:
        assert (
            scheduler.run_due_jobs(now=datetime.now(timezone.utc))[0].status
            == "succeeded"
        )
    receipts = [
        message
        for request in (old_requests[-1], new_requests[-1])
        for message in request.messages
        if isinstance(message, ToolResultMessage)
        and message.tool_name == "knowledge_read"
    ]
    assert len(receipts) == 2 and all(
        message.status == "success" for message in receipts
    )
    original, updated = (model_visible_text(message) for message in receipts)
    assert "订单汇总遗漏退款" in original and "未来改为检查重复订单" not in original
    assert "未来改为检查重复订单" in updated and "订单汇总遗漏退款" not in updated
