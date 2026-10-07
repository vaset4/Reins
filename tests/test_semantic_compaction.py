"""经真实模型装配验证语义摘要、来源、分支和预算。

作者：xxx
时间：2026-09-14 20:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_turns

import json
from contextlib import closing

import pytest

from context.production_builder import ProductionContextBuilder
from context.compaction import (
    CompactionMaterial,
    original_sources,
    source_groups,
    summary_task,
)
from context.summary_entries import SummaryCitation, SummaryContent, SummaryEntry
from context.window import request_budget
from scripts.testing.llm import ScriptedTurnOptions
from llm.messages import model_visible_text
from llm.provider_result import ProviderError, reported
from llm.types import ModelUsage
from runtime.agent_loop import AgentLoop, State
from runtime.context_preparation import ContextCompactor
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.run_evidence import RunEvidenceStore
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_assistant_message, append_user_message
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore
from tests.test_model_attempts import _event
from tools.tool_registry import ToolRegistry

TEST_WINDOW = 5000
AUDIT_CANDIDATE_REPEATS = 120
OVERSIZED_AUDIT_CANDIDATE_REPEATS = 400
AUDIT_SOURCE_PARAGRAPHS = 700
SUMMARY = """## 目标
选择可行的本地研究方案
## 约束
总费用不能超过500，原文不能上传
## 决定
只使用本地资料进行比较
## 进展
已取得材料，但方案选择尚未完成
## 失败
旧的线上方案因上传要求被排除
## 待办
比较剩余方案，核对费用和材料来源
## 来源
依据保存的用户约束及研究资料；完整消息范围见摘要记录
"""


def _response(text, index, *, stop_reason="end_turn"):
    """通过Provider事件返回可计量响应；传参：正文、请求序号和结束原因；返回：事件流。"""
    yield _event("response_start", 0, message_id=f"summary-response-{index}")
    yield _event("content_start", 1, block_id="text", content_kind="text")
    yield _event("content_delta", 2, block_id="text", delta=text)
    yield _event("content_end", 3, block_id="text")
    yield _event(
        "usage_update",
        4,
        usage=ModelUsage(
            input_tokens=reported(30),
            output_tokens=reported(10),
            total_tokens=reported(40),
        ),
    )
    yield _event("response_done", 5, stop_reason=stop_reason)


def _summary_delta(body):
    """用固定场景的有来源回答驱动模型边界；传参：实际摘要请求；返回：测试差量，不作语义评测。"""
    material = json.loads(body.rsplit("\n", 1)[-1])
    messages = [message for group in material["original_groups"] for message in group]
    entries = material["current_entries"]
    additions = []
    for message in messages:
        text = "\n".join(part.get("text", "") for part in message.get("content", []))
        span = message["content"][0].get("source_span", {})
        identity = f"limits-{message['message_id']}-{span.get('start', 0)}"
        if message["source_kind"] == "user_input" and not any(
            entry["entry_id"] == identity for entry in entries
        ):
            additions.append(
                {
                    "entry_id": identity,
                    "kind": "requirement",
                    "text": text,
                    "sources": [{"message_id": message["message_id"], "quote": text}],
                }
            )
    segments = []
    existing = {
        segment["segment_id"] for segment in material.get("current_segments", [])
    }
    for group in material["original_groups"]:
        segment_id = f"research-{group[0]['message_id']}"
        if segment_id not in existing:
            segments.append(
                {
                    "segment_id": segment_id,
                    "title": "本地研究方案与原件核对",
                    "message_ids": [message["message_id"] for message in group],
                    "p1": "核对研究资料；费用不能超过500，原文不能上传",
                    "p2": "本地研究；费用不能超过500，原文不能上传",
                    "p3": "方案未定，保留原件",
                    "p4": "本地方案",
                }
            )
    return json.dumps(
        {
            "base_summary_id": material["base_summary_id"],
            "add": additions,
            "segments": segments,
            "dispositions": [
                {
                    "message_id": identity,
                    "destinations": ["original"],
                    "reason": "过程资料可回查",
                }
                for identity in dict.fromkeys(
                    message["message_id"] for message in messages
                )
            ],
        },
        ensure_ascii=False,
    )


def _runtime(
    tmp_path, monkeypatch, *, mode="ok", max_steps=20, context_window=TEST_WINDOW
):
    """建立长会话并在Adapter边界观察真实请求；传参：临时根及故障模式；返回：循环、上下文和请求。"""
    with closing(TaskStore(tmp_path)) as tasks:
        task = tasks.create_task("选择研究方案")
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        session_id="semantic-context",
        payload={"message": "继续选择方案"},
        capability_lease=from_trigger(
            "user", task_id=task.task_id, max_steps=max_steps
        ),
    )
    append_user_message(
        tmp_path, context.session_id, "总费用不能超过500，原文不能上传。"
    )
    for index in range(8):
        append_assistant_message(
            tmp_path,
            context.session_id,
            f"研究资料{index}：" + "可复核的原始材料 " * 400,
        )
    context.payload["input_message_id"] = append_user_message(
        tmp_path, context.session_id, "继续选择方案"
    )
    client = from_test_turns(
        ["unused"], options=ScriptedTurnOptions(context_window=context_window)
    )
    loop = AgentLoop(tmp_path, llm_client=client, tool_registry=ToolRegistry())
    requests = []

    def stream(request, **_kwargs):
        """摘要只整理资料，主请求按实际拿到的约束选择结果；传参：请求；返回：模型事件。"""
        requests.append(request)
        body = "\n".join(model_visible_text(message) for message in request.messages)
        if "整理下面的会话资料" in body:
            assert not request.tools
            if mode == "cancel":
                loop.cancellation.cancel("stop during context summary")
            return _response(
                "缺少来源的摘要" if mode == "invalid" else _summary_delta(body),
                len(requests),
                stop_reason="max_output_tokens" if mode == "truncated" else "end_turn",
            )
        instructions = "\n".join(part.text for part in request.instructions)
        assert "不能超过500" in instructions and "原文不能上传" in instructions
        return _response("选择预算500以内的本地方案", len(requests))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    return loop, context, requests


def test_semantic_summary_preserves_constraints_sources_and_actual_attempts(
    tmp_path, monkeypatch
):
    """摘要后真实请求仍带关键约束且原文可回取；传参：临时根和替换器；返回：无。"""
    loop, context, requests = _runtime(tmp_path, monkeypatch)
    owner = SessionMessageStore(tmp_path)
    original = owner.read_entries(context.session_id)
    assert loop.run(context) is State.DONE
    assert loop.last_output == "选择预算500以内的本地方案"
    assert owner.read_entries(context.session_id)[: len(original)] == original
    record = SessionCompactionStore(owner).current(
        owner.materialize(context.session_id)
    )
    assert record is not None and record.source_sha256
    assert len(record.request_ids) > 1
    assert len(requests) == len(record.request_ids) + 1
    assert all(
        request_budget(request, TEST_WINDOW).required_total <= TEST_WINDOW
        for request in requests
    )
    assert (
        sum(
            message.message_id == context.payload["input_message_id"]
            for message in requests[-1].messages
        )
        == 1
    )
    facts = RunFactStore(tmp_path).read_run(context.run_id)
    attempts = [fact for fact in facts if fact.get("event") == "llm:attempt"]
    assert len(attempts) == len(requests)
    assert set(record.request_ids) < {fact["request_id"] for fact in attempts}
    handled = [fact for fact in facts if fact.get("event") == "input:handled"]
    main_ids = {fact["request_id"] for fact in attempts} - set(record.request_ids)
    assert {fact["request_id"] for fact in handled} == main_ids
    assert context.payload["input_message_id"] in {
        identity for fact in handled for identity in fact["input_ids"]
    }
    evidence = [
        segment
        for fact in facts
        if fact.get("event") == "context:segments"
        for segment in fact["segments"]
        if segment["name"] == "session_summary"
    ][-1]
    assert (
        evidence["source"] == "SessionCompactionStore"
        and evidence["authority"] == "projection"
    )
    assert evidence["summary_id"] == record.summary_id
    assert evidence["summary_source"]["sha256"] == record.source_sha256


@pytest.mark.parametrize(
    "mode, expected",
    [("invalid", State.FAILED), ("cancel", State.PAUSED), ("truncated", State.FAILED)],
)
def test_failed_or_cancelled_summary_does_not_replace_original_history(
    tmp_path, monkeypatch, mode, expected
):
    """摘要失败或中断不能先删除旧材料；传参：故障与期望状态；返回：无。"""
    monkeypatch.setenv("REINS_TRACE_LEVEL", "debug")
    loop, context, requests = _runtime(tmp_path, monkeypatch, mode=mode)
    owner = SessionMessageStore(tmp_path)
    before = owner.materialize(context.session_id).messages
    assert loop.run(context) is expected
    after = owner.materialize(context.session_id)
    assert after.messages[: len(before)] == before
    assert SessionCompactionStore(owner).current(after) is None
    assert len(requests) == 1
    if mode == "truncated":
        facts = RunFactStore(tmp_path).read_run(context.run_id)
        attempt = next(fact for fact in facts if fact.get("event") == "llm:attempt")
        response = RunEvidenceStore(tmp_path).read_reference(attempt["response_path"])
        assert response is not None
        assert response["response"]["stop_reason"] == "max_output_tokens"
        assert attempt["usage"]["output_tokens"]["value"] == 10


def test_summary_usage_is_recorded_without_blocking_following_dispatch(
    tmp_path, monkeypatch
):
    """辅助模型的消耗同样计入运行账目，但不阻塞后续派发；传参：临时根和替换器；返回：无。"""
    loop, context, requests = _runtime(tmp_path, monkeypatch, max_steps=1)

    assert loop.run(context) is State.DONE
    bodies = [
        "\n".join(model_visible_text(message) for message in request.messages)
        for request in requests
    ]
    # 摘要请求与主请求都真实发出，步数上限不再截断任何一次
    assert any("整理下面的会话资料" in body for body in bodies)
    assert any("整理下面的会话资料" not in body for body in bodies)
    evidence = context.payload["_runtime_budget_evidence"]
    assert evidence["steps_used"] >= 2
    assert evidence["tokens_used"] > 0


@pytest.mark.parametrize("oversized", [False, True])
def test_audit_uses_actual_candidate_budget_and_preserves_all_source_pages(
    tmp_path, monkeypatch, oversized
):
    """候选占满旧预留空间仍可逐页核对，物理超窗明确失败；传参：目录、替换器和超窗状态；返回：无。"""
    original = "原始核对材料。" + "可复核的原文记录。" * AUDIT_SOURCE_PARAGRAPHS
    append_assistant_message(tmp_path, "candidate-audit", original)
    append_user_message(tmp_path, "candidate-audit", "继续核对")
    owner = SessionMessageStore(tmp_path)
    source = SummarySource(owner.materialize("candidate-audit"), 1)
    source_id = source.view.messages[0].message_id
    store = SessionCompactionStore(owner)
    prior = store.publish(source, "保留已发布摘要", request_ids=("prior-request",))
    repetitions = (
        OVERSIZED_AUDIT_CANDIDATE_REPEATS if oversized else AUDIT_CANDIDATE_REPEATS
    )
    candidate = SummaryContent(
        entries=(
            SummaryEntry(
                "progress",
                "observation",
                "材料仍需核对，不能提前报完成。" * repetitions,
                (SummaryCitation(source_id, "原始核对材料。", "assistant"),),
            ),
        )
    )
    client = from_test_turns(
        [], options=ScriptedTurnOptions(context_window=TEST_WINDOW)
    )
    context = {"context_purpose": "compaction"}
    requests = []

    def stream(request, **_options):
        """按真实输入记录已核对原文，不修订候选；传参：实际请求；返回：供应商事件。"""
        requests.append(request)
        body = "\n".join(model_visible_text(message) for message in request.messages)
        payload = json.loads(body.rsplit("\n", 1)[-1])
        delta = {
            "base_summary_id": None,
            "dispositions": [
                {
                    "message_id": identity,
                    "destinations": ["original"],
                    "reason": "逐页核对后保留原文",
                }
                for identity in payload["required_disposition_message_ids"]
            ],
        }
        return _response(json.dumps(delta), len(requests))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    compactor = ContextCompactor(
        store,
        client.prepare_request,
        lambda bundle: client.plan(
            bundle.model_task, context=dict(bundle.model_context)
        ),
    )
    arguments = dict(
        requests=[],
        base=None,
        sources=original_sources(source),
        auxiliary=[],
        related=[],
        published_content=SummaryContent(),
    )
    groups = source_groups(CompactionMaterial(source, ""))
    if oversized:
        with pytest.raises(
            ValueError, match="fixed metadata or required entries exceed"
        ):
            compactor._audit(candidate, groups, context, **arguments)
        assert requests == []
        assert store.current(owner.materialize("candidate-audit")) == prior
        return
    # 【上下文】【候选容量】固定候选本身放得下，但不能再为另一请求重复扣除最大输出
    prepared = client.prepare_request(
        summary_task([], candidate, base_summary_id=None), context
    )
    budget = request_budget(prepared.request, TEST_WINDOW)
    assert (
        budget.required_total
        < TEST_WINDOW
        < budget.required_total + budget.output_reserved
    )
    checked = compactor._audit(candidate, groups, context, **arguments)
    assert checked.entries == candidate.entries and len(requests) > 1
    assert all(
        request_budget(request, TEST_WINDOW).required_total <= TEST_WINDOW
        for request in requests
    )
    spans = sorted(
        (span["start"], span["end"])
        for check in checked.checks
        for span in check["source_spans"]
        if span["message_id"] == source_id
    )
    assert spans[0][0] == 0 and spans[-1][1] == len(original)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert store.current(owner.materialize("candidate-audit")) == prior


def test_summary_is_branch_bound_and_publication_detects_rewind(tmp_path):
    """回退分支后不能发布或使用另一分支摘要；传参：临时存储；返回：无。"""
    owner = SessionMessageStore(tmp_path)
    append_user_message(tmp_path, "summary-branch", "原约束")
    first_entry = owner.materialize("summary-branch").leaf_id
    append_assistant_message(tmp_path, "summary-branch", "已调研")
    append_user_message(tmp_path, "summary-branch", "继续")
    source = SummarySource(owner.materialize("summary-branch"), 2)
    store = SessionCompactionStore(owner)
    record = store.publish(source, SUMMARY, request_ids=("request-actual",))
    builder = ProductionContextBuilder(tmp_path, system_prompt_provider=lambda: "")
    assert builder.read_conversation_history("summary-branch").summary == record
    owner.branch("summary-branch", first_entry)
    assert store.current(owner.materialize("summary-branch")) is None
    with pytest.raises(ValueError, match="branch changed"):
        store.publish(source, SUMMARY, request_ids=("request-late",))


def test_provider_overflow_compacts_before_retrying_with_actual_window(
    tmp_path, monkeypatch
):
    """本地估算可放入但供应商拒绝时，先生成摘要再重试；传参：临时根；返回：无。"""
    loop, context, requests = _runtime(tmp_path, monkeypatch, context_window=80000)
    adapter = loop.llm_client._adapter_registry.require("scripted_test")
    accepted_stream = adapter.stream

    def reject_first(request, **kwargs):
        """模拟供应商分词器报告超窗；传参：实际请求；返回：错误或后续真实事件流。"""
        if not requests:
            requests.append(request)
            error = ProviderError(
                category="context_overflow",
                stage="transport",
                retryable=False,
                summary="input too large",
                provider="scripted",
                model="stub",
                api_family="scripted",
            )
            return iter(
                [
                    _event("response_start", 0, message_id="rejected"),
                    _event("response_error", 1, error=error),
                ]
            )
        return accepted_stream(request, **kwargs)

    monkeypatch.setattr(adapter, "stream", reject_first)
    assert loop.run(context) is State.DONE
    assert "不能超过500" in loop.last_output or "预算500以内" in loop.last_output
    assert len(requests) >= 3
    owner = SessionMessageStore(tmp_path)
    summary = SessionCompactionStore(owner).current(
        owner.materialize(context.session_id)
    )
    assert summary is not None
    assert (
        request_budget(requests[-1], 80000).required_total
        < request_budget(requests[0], 80000).required_total
    )
    facts = RunFactStore(tmp_path).read_run(context.run_id)
    assert len([fact for fact in facts if fact.get("event") == "llm:attempt"]) == len(
        requests
    )


def test_large_existing_summary_can_move_to_a_smaller_model_window(
    tmp_path, monkeypatch
):
    """已有摘要超出新窗口时分块整理，关键约束与前版关系保留；传参：临时根；返回：无。"""
    loop, context, requests = _runtime(tmp_path, monkeypatch)
    owner = SessionMessageStore(tmp_path)
    store = SessionCompactionStore(owner)
    view = owner.materialize(context.session_id)
    large = SUMMARY.replace("已取得材料，但方案选择尚未完成", "已有材料摘录 " * 3000)
    prior = store.publish(
        SummarySource(view, len(view.messages) - 1),
        large,
        request_ids=("request-prior-model",),
    )
    assert loop.run(context) is State.DONE
    current = store.current(owner.materialize(context.session_id))
    assert current.previous_summary_id == prior.summary_id
    assert "原文不能上传" in current.text and len(current.request_ids) > 1
    assert all(
        request_budget(request, TEST_WINDOW).required_total <= TEST_WINDOW
        for request in requests
    )


def test_summary_provider_overflow_reduces_fragment_before_retry(tmp_path, monkeypatch):
    """摘要请求也处理供应商分词差异，失败尝试仍有证据；传参：临时根；返回：无。"""
    window = 16000
    loop, context, requests = _runtime(tmp_path, monkeypatch, context_window=window)
    adapter = loop.llm_client._adapter_registry.require("scripted_test")
    stream = adapter.stream

    def reject_first_fragment(request, **kwargs):
        """仅拒绝首次摘要片段；传参：实际请求；返回：失败或正常供应商事件。"""
        if not requests:
            requests.append(request)
            error = ProviderError(
                category="context_overflow",
                stage="transport",
                retryable=False,
                summary="summary input too large",
                provider="scripted",
                model="stub",
                api_family="scripted",
            )
            return iter(
                [
                    _event("response_start", 0, message_id="summary-rejected"),
                    _event("response_error", 1, error=error),
                ]
            )
        return stream(request, **kwargs)

    monkeypatch.setattr(adapter, "stream", reject_first_fragment)
    assert loop.run(context) is State.DONE
    assert (
        request_budget(requests[1], window).required_total
        < request_budget(requests[0], window).required_total
    )
    owner = SessionMessageStore(tmp_path)
    summary = SessionCompactionStore(owner).current(
        owner.materialize(context.session_id)
    )
    assert len(summary.request_ids) + 1 == len(requests)
