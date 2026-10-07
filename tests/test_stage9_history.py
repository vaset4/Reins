"""【上下文】【阶段九验证】四级历史、当前要求、后台来源与恢复的行为证据。

作者：xxx
时间：2026-10-01 12:00:00
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone

import pytest

from context.compaction import compaction_material
from context.history_segments import (
    HistorySegment,
    segment_digest,
    validate_segment_coverage,
)
from context.history_view import history_context
from context.summary_entries import SummaryCitation, SummaryContent, SummaryEntry
from context.window import RequestBudget
from llm.messages import (
    AssistantMessage,
    ToolCallPart,
    agent_message_to_mapping,
    model_visible_text,
)
from runtime.cancellation import CancellationToken
from runtime.context_compaction_jobs import ContextCompactionJobs
from runtime.context_compaction_worker import execute_compaction
from runtime.context_preparation import ContextCompactor
from runtime.cron import CronExecution
from runtime.history_reader import read_history_page
from runtime.lease import from_trigger
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import MaterializedSession, SessionMessageStore
from runtime.session_messages import (
    ToolExchange,
    append_assistant_message,
    append_tool_exchange,
    append_user_message,
)
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from schedules.occurrences import OccurrenceStore
from schedules.store import ScheduleStore
from scripts.testing.llm import from_test_turns
from tests.test_semantic_compaction import _response, _summary_delta
from tools.tool_registry import ToolRegistry
from context.production_builder import ProductionContextBuilder, ProductionContextBundle


def source_history(root, session="long"):
    """构造有条件原话与当前尾部；参数：临时空间、会话；返回：消息和冻结范围。"""
    messages = SessionMessageStore(root)
    first = append_user_message(root, session, "费用不得超过500，禁止上传")
    append_assistant_message(root, session, "已检查本地材料，方案未确定")
    append_user_message(root, session, "继续当前任务")
    return messages, SummarySource(messages.materialize(session), 2), first


def segment_content(source, first):
    """提供已审计的同源档位与独立要求；参数：原件、用户身份；返回：发布候选。"""
    ids = tuple(
        message.message_id for message in source.view.messages[: source.covered_count]
    )
    segment = HistorySegment(
        "local-study",
        "本地方案选择",
        ids,
        "本地材料已检查，费用不得超过500，禁止上传；方案未确定",
        "费用不得超过500且禁止上传，方案未确定",
        "本地方案尚未确定；费用不得超过500，禁止上传",
        "本地方案未定/限500/禁止上传",
    )
    entry = SummaryEntry(
        "cost",
        "requirement",
        "费用不得超过500，禁止上传",
        (SummaryCitation(first, "费用不得超过500，禁止上传", "user_input"),),
    )
    return SummaryContent(
        entries=(entry,),
        segments=(segment,),
        checks=(
            {
                "request_id": "audit",
                "message_ids": ids,
                "segment_ids": (segment.segment_id,),
                "segment_versions": {segment.segment_id: segment_digest(segment)},
                "source_spans": [
                    {
                        "message_id": message.message_id,
                        "part_index": index,
                        "start": 0,
                        "end": len(part["text"]),
                        "total": len(part["text"]),
                    }
                    for message in source.view.messages[: source.covered_count]
                    for index, part in enumerate(
                        agent_message_to_mapping(message)["content"]
                    )
                    if isinstance(part.get("text"), str)
                ],
            },
        ),
    )


def test_empty_history_view_preserves_an_uncreated_session_without_writing(tmp_path):
    """合法空会话只提供空上下文，不读取不存在原件或创建分支；参数：隔离空间；返回：无。"""
    root = tmp_path / "uncreated-space"
    store = SessionCompactionStore(SessionMessageStore(root))
    view = MaterializedSession("future-session", None, (), ())
    context = history_context(view, store)
    assert (
        context["history_materials"] == () and context["history_material_levels"] == {}
    )
    assert (
        context["context_branch_id"] == "" and context["effective_requirements"] == ""
    )
    assert not root.exists()
    selection = ProductionContextBuilder(
        root, system_prompt_provider=lambda: "测试指令"
    ).read_conversation_history("future-session")
    assert selection.messages == () and selection.summary is None
    assert selection.history_context == context
    assert not store.messages.exists("future-session")


def test_legacy_constraints_stay_visible_until_originals_are_rebuilt(tmp_path):
    """未分离要求的旧摘要必须整体保留，原文重建后只采用新表示；参数：隔离空间；返回：无。"""
    messages, source, first = source_history(tmp_path)
    store = SessionCompactionStore(messages)
    legacy = store.publish(
        source, "费用不得超过500，禁止上传；方案未定", request_ids=("old-generation",)
    )
    before = history_context(messages.materialize("long"), store)
    assert before["history_materials"][0].protected is True
    rebuilt = replace(source, previous=legacy)
    content = segment_content(rebuilt, first)
    saved = store.publish(
        rebuilt, content.render(), content=content, request_ids=("generate", "audit")
    )
    after = history_context(messages.materialize("long"), store)
    assert len(after["history_materials"]) == 1
    assert after["history_materials"][0].identity.startswith(
        f"segment:{saved.summary_id}:"
    )
    assert "费用不得超过500" in after["effective_requirements"]
    assert (
        store.read(legacy.summary_id, messages.materialize("long")).text == legacy.text
    )


def test_every_level_keeps_business_scope_and_title_only_locator(tmp_path):
    """降档仍说明适用范围，定位档允许仅用足够明确的标题；参数：隔离来源；返回：无。"""
    _, source, first = source_history(tmp_path)
    segment = replace(
        segment_content(source, first).segments[0], scope="仅本地测试环境", p4=""
    )
    for level in ("P1", "P2", "P3", "P4"):
        text = segment.render(level)
        assert "仅本地测试环境" in text and segment.title in text
    assert segment.render("P4").endswith(segment.title)


def test_four_levels_recover_exact_originals_and_keep_requirements_separate(tmp_path):
    """降档不改原件/当前要求，目录可回同一消息范围；参数：临时空间；返回：无。"""
    messages, source, first = source_history(tmp_path)
    content = segment_content(source, first)
    store = SessionCompactionStore(messages)
    saved = store.publish(
        source, content.render(), content=content, request_ids=("generate", "audit")
    )
    view = messages.materialize("long")
    before = messages.read_entries("long")
    for level in ("P1", "P2", "P3", "P4"):
        page = read_history_page(
            view, call_id="read", summaries=store, view_kind="segments", level=level
        )
        row = page["segments"][0]
        assert row["summary_id"] == saved.summary_id and row["level"] == level
        original = read_history_page(view, call_id="read", source_ref=row["source_ref"])
        assert {item["message_id"] for item in original["messages"]} == set(
            saved.message_ids
        )
    projected = history_context(view, store)
    assert "不得超过500" in projected["effective_requirements"]
    assert len(projected["history_materials"]) == 1
    append_user_message(tmp_path, "long", "追加输入")
    assert (
        history_context(messages.materialize("long"), store)["context_branch_id"]
        == projected["context_branch_id"]
    )
    assert messages.read_entries("long")[: len(before)] == before


def test_segments_reject_source_gaps_split_tools_and_unaudited_levels(tmp_path):
    """同范围与请求身份之外还需完整交互及审计；参数：临时空间；返回：无。"""
    messages, source, first = source_history(tmp_path)
    content = segment_content(source, first)
    with pytest.raises(ValueError, match="exactly"):
        validate_segment_coverage(
            (replace(content.segments[0], message_ids=(first,)),),
            source.view.messages[:2],
        )
    with pytest.raises(ValueError, match="audit"):
        SessionCompactionStore(messages).publish(
            source,
            content.render(),
            content=replace(content, checks=()),
            request_ids=("generate",),
        )
    append_tool_exchange(
        tmp_path, "tools", ToolExchange("call", "file_read", rendered="原件结果")
    )
    group = messages.materialize("tools").messages
    split = tuple(
        replace(
            content.segments[0],
            segment_id=f"s-{index}",
            message_ids=(message.message_id,),
        )
        for index, message in enumerate(group)
    )
    with pytest.raises(ValueError, match="splits"):
        validate_segment_coverage(split, group)


def test_pending_tail_does_not_block_older_complete_prefix(tmp_path):
    """未完成工具留在尾部，旧完整交互仍可整理；参数：临时空间；返回：无。"""
    messages, _, _ = source_history(tmp_path)
    messages.append_message(
        "long",
        AssistantMessage(
            "pending", (ToolCallPart("still-running", "file_read", {"path": "x"}),)
        ),
    )
    view = messages.materialize("long")
    budget = RequestBudget(
        context_window=1000,
        instructions=100,
        tools=0,
        protocol=0,
        messages=200,
        output_reserved=50,
    )
    material = compaction_material(view, None, budget, force=True, retained_target=1)
    assert material.source.covered_count > 0
    assert "pending" not in {
        message.message_id for message in view.messages[: material.source.covered_count]
    }


def test_accepted_background_work_does_not_wait_in_frontend(tmp_path, monkeypatch):
    """请求仍容纳时接纳整理并立即返回旧材料；参数：隔离空间；返回：无。"""
    messages, _, _ = source_history(tmp_path)
    client = from_test_turns([])
    context = {
        "session_id": "long",
        "run_id": "r",
        "segment_id": "s",
        "tool_registry": ToolRegistry(),
        "conversation_history": messages.materialize("long").messages,
    }
    bundle = ProductionContextBundle("继续", context, ())
    accepted = []

    def enqueue(material, _context):
        """记录后台接纳范围；参数：冻结材料；返回：已排队证据。"""
        accepted.append(material)
        return {"status": "queued", "job_id": "pending-history"}

    def no_model(_bundle):
        """前台不能等待摘要模型；参数：请求；返回：不返回。"""
        raise AssertionError("front-end waited on summary generation")

    monkeypatch.setattr("runtime.context_preparation.PRESSURE_RATIO", 0.0)
    monkeypatch.setattr("runtime.context_preparation.TARGET_RATIO", 0.001)
    compactor = ContextCompactor(
        SessionCompactionStore(messages),
        client.prepare_request,
        no_model,
        enqueue=enqueue,
    )
    ready = compactor.fit(bundle, rebuild=lambda: bundle)
    assert len(accepted) == 1
    assert (
        ready.model_context["prepared_request"].trim_delta["compaction"]["background"][
            "status"
        ]
        == "queued"
    )
    assert (
        SessionCompactionStore(messages).current(messages.materialize("long")) is None
    )


def background_request(root, monkeypatch):
    """经正式持久接纳构造后台发生；参数：隔离根与模型替换；返回：工作与脚本客户端。"""
    messages, source, _ = source_history(root)
    WorkspaceStore(root).bind_session("long", root)
    run = RunContext(
        trigger=Trigger.USER,
        payload={},
        session_id="long",
        run_id="source-run",
        capability_lease=from_trigger(
            "user", capabilities={"background_run": {"enabled": True}}
        ),
    )
    client = from_test_turns([])
    jobs = ContextCompactionJobs(root)
    from context.compaction import CompactionMaterial

    accepted = jobs.accept(CompactionMaterial(source, ""), {}, run=run, client=client)
    schedule = ScheduleStore(root).load_schedule(f"schedule-{accepted['job_id']}")
    occurrence = OccurrenceStore(root).accept(
        schedule.schedule_id,
        schedule.next_run_at,
        next_run_at=None,
        schedule_snapshot=asdict(schedule),
    )
    calls = []

    def stream(request, **_options):
        """后台调用期间继续追加真实输入；参数：实际请求；返回：有来源的测试响应。"""
        calls.append(request)
        if len(calls) == 1:
            append_user_message(root, "long", "新增尾部不得丢失")
        body = "\n".join(model_visible_text(message) for message in request.messages)
        return _response(_summary_delta(body), len(calls))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    request = CronExecution(
        schedule, occurrence, datetime.now(timezone.utc), CancellationToken()
    )
    return jobs, accepted, request, client, calls, messages


def test_background_uses_frozen_sources_preserves_tail_and_recovers_commit(
    tmp_path, monkeypatch
):
    """真实辅助边界不伪造输入，提交后重启不重做模型；参数：隔离根；返回：无。"""
    jobs, accepted, request, client, calls, messages = background_request(
        tmp_path, monkeypatch
    )
    assert not calls and accepted["status"] == "queued"
    result = execute_compaction(
        request,
        data_root=tmp_path,
        llm_factory=lambda _: client,
        registry_factory=ToolRegistry,
    )
    assert result.status == "done", result.error
    assert len(calls) >= 2 and jobs.load(accepted["job_id"])["status"] == "published"
    assert "新增尾部不得丢失" in str(messages.materialize("long").messages)
    assert messages.materialize(request.occurrence.session_id).messages == ()
    count = len(calls)
    jobs.update(accepted["job_id"], status="running", summary_id=None)
    recovered = execute_compaction(
        request,
        data_root=tmp_path,
        llm_factory=lambda _: client,
        registry_factory=ToolRegistry,
    )
    assert recovered.status == "done" and len(calls) == count


def test_background_cancel_and_branch_change_do_not_publish(tmp_path, monkeypatch):
    """取消或来源分支切换不能被记录成成功；参数：隔离根；返回：无。"""
    jobs, accepted, request, client, calls, messages = background_request(
        tmp_path, monkeypatch
    )
    messages.branch("long", messages.materialize("long").entries[0].entry_id)
    result = execute_compaction(
        request,
        data_root=tmp_path,
        llm_factory=lambda _: client,
        registry_factory=ToolRegistry,
    )
    assert result.status == "failed" and "branch" in result.error and not calls
    assert jobs.load(accepted["job_id"])["summary_id"] is None
    request.cancellation.cancel("user cancelled maintenance")
    result = execute_compaction(
        request,
        data_root=tmp_path,
        llm_factory=lambda _: client,
        registry_factory=ToolRegistry,
    )
    assert (
        result.status == "paused"
        and jobs.load(accepted["job_id"])["status"] == "cancelled"
    )


def test_pause_affects_new_admission_and_running_cancel_waits_for_execution(
    tmp_path, monkeypatch
):
    """只读状态无写，暂停保留已接纳工作，取消不伪造停止；参数：隔离空间；返回：无。"""
    empty = tmp_path / "untouched"
    assert ContextCompactionJobs(empty).status() == {"enabled": True, "jobs": []}
    assert not empty.exists()
    root = tmp_path / "data"
    jobs, accepted, request, client, calls, _ = background_request(root, monkeypatch)
    jobs.configure(enabled=False)
    assert jobs.status(session_id="long")["enabled"] is False
    assert jobs.load(accepted["job_id"])["status"] == "queued"
    jobs.update(accepted["job_id"], status="running")
    pending = jobs.cancel(accepted["job_id"])
    assert pending["status"] == "running" and pending["cancel_requested"]
    outcome = execute_compaction(
        request,
        data_root=root,
        llm_factory=lambda _: client,
        registry_factory=ToolRegistry,
    )
    assert outcome.status == "paused" and not calls
    assert jobs.load(accepted["job_id"])["status"] == "cancelled"
    assert len(jobs.status()["jobs"]) == 1
