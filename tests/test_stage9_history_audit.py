"""【上下文】【历史核对】最终四档正文必须具有对应版本和完整原文的核对证据。

作者：xxx
时间：2026-10-01 14:30:06
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from context.compaction import CompactionMaterial, source_groups, summary_task
from context.production_builder import ProductionContextBundle
from context.window import request_budget
from llm.messages import model_visible_text
from runtime.context_preparation import ContextCompactor
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_assistant_message, append_user_message
from scripts.testing.llm import ScriptedTurnOptions, from_test_turns
from tests.test_semantic_compaction import _response
from tests.test_stage9_history import segment_content, source_history

PAGED_CONTEXT_WINDOW = 7000
LONG_SOURCE_REPETITIONS = 400


def test_changed_level_cannot_reuse_an_earlier_audit(tmp_path):
    """后续把否定条件反转不能借旧核对发布；参数：隔离空间；返回：无。"""
    messages, source, first = source_history(tmp_path)
    content = segment_content(source, first)
    altered = replace(content.segments[0], p2="允许上传，费用可以超过500，方案已确定")
    candidate = replace(content, segments=(altered,))
    with pytest.raises(ValueError, match="audit"):
        SessionCompactionStore(messages).publish(
            source,
            candidate.render(),
            content=candidate,
            request_ids=("generate", "audit"),
        )
    assert (
        SessionCompactionStore(messages).current(messages.materialize("long")) is None
    )


def test_paged_segment_revision_rechecks_all_original_pages(tmp_path, monkeypatch):
    """后页完善同一段时重新核对完整依据；参数：隔离空间和模型替换；返回：无。"""
    text = "只使用本地材料，费用上限500，方案尚未确定。" * LONG_SOURCE_REPETITIONS
    append_assistant_message(tmp_path, "pages", text)
    append_user_message(tmp_path, "pages", "继续")
    messages = SessionMessageStore(tmp_path)
    material = CompactionMaterial(SummarySource(messages.materialize("pages"), 1), "")
    client = from_test_turns(
        [], options=ScriptedTurnOptions(context_window=PAGED_CONTEXT_WINDOW)
    )
    requests = []
    generated_pages = []

    def stream(request, **_options):
        """逐页完善同一目标段，独立核对保持正文；参数：生产请求；返回：脚本响应。"""
        requests.append(request)
        body = "\n".join(model_visible_text(message) for message in request.messages)
        payload = json.loads(body.rsplit("\n", 1)[-1])
        delta = {
            "base_summary_id": None,
            "dispositions": [
                {
                    "message_id": identity,
                    "destinations": ["original"],
                    "reason": "保留来源",
                }
                for identity in payload["required_disposition_message_ids"]
            ],
        }
        if "独立原文核对：" not in body:
            first = payload["original_groups"][0][0]
            span = first["content"][0].get("source_span")
            generated_pages.append(span)
            delta["segments"] = [
                {
                    "segment_id": "local-plan",
                    "title": "本地方案评估",
                    "message_ids": [first["message_id"]],
                    "p1": f"已综合第{len(generated_pages)}页：费用上限500，方案未定",
                    "p2": "本地方案评估；费用上限500，方案未定",
                    "p3": "本地方案未定；上限500",
                    "p4": "本地/500/未定",
                }
            ]
        return _response(json.dumps(delta, ensure_ascii=False), len(requests))

    def summarize(bundle: ProductionContextBundle):
        """通过客户端执行已准备的核对请求；参数：材料包；返回：模型计划。"""
        return client.plan(bundle.model_task, context=dict(bundle.model_context))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    store = SessionCompactionStore(messages)
    compactor = ContextCompactor(store, client.prepare_request, summarize)
    content, request_ids = compactor._summarize(
        material,
        {
            "session_id": "pages",
            "run_id": "r",
            "segment_id": "s",
            "tool_registry": None,
            "history_representation_version": 1,
        },
        context_window=PAGED_CONTEXT_WINDOW,
    )
    saved = store.publish(
        material.source, content.render(), content=content, request_ids=request_ids
    )
    assert len(generated_pages) > 1 and all(generated_pages)
    assert saved.content == content
    assert all(
        request_budget(request, PAGED_CONTEXT_WINDOW).required_total
        <= PAGED_CONTEXT_WINDOW
        for request in requests
    )


def test_partial_original_page_cannot_claim_complete_level_audit(tmp_path):
    """只核对一句原话的前半不能声称完整核对；参数：隔离空间；返回：无。"""
    messages, source, first = source_history(tmp_path)
    content = segment_content(source, first)
    check = {
        **content.checks[0],
        "source_spans": [
            {
                "message_id": first,
                "part_index": 0,
                "start": 0,
                "end": 2,
                "total": len(source.view.messages[0].content[0].text),
            }
        ],
    }
    candidate = replace(content, checks=(check,))
    with pytest.raises(ValueError, match="audit"):
        SessionCompactionStore(messages).publish(
            source,
            candidate.render(),
            content=candidate,
            request_ids=("generate", "audit"),
        )


def test_fragment_keeps_related_levels_without_repeating_unrelated_segments(tmp_path):
    """本页核对只携带同源片段，其他片段由候选继承；参数：隔离原件；返回：无。"""
    _, source, first = source_history(tmp_path)
    content = segment_content(source, first)
    unrelated = replace(
        content.segments[0],
        segment_id="other-topic",
        message_ids=("unrelated-original",),
        p1="其他目标的历史经过，无需本页重复核对",
    )
    candidate = replace(content, segments=(*content.segments, unrelated))
    groups = source_groups(CompactionMaterial(source, ""))
    payload = json.loads(
        summary_task(groups, candidate, base_summary_id=None, audit=True).rsplit(
            "\n", 1
        )[-1]
    )
    assert [segment["segment_id"] for segment in payload["current_segments"]] == [
        content.segments[0].segment_id
    ]
    assert payload["current_entries"][0]["text"] == "费用不得超过500，禁止上传"
    assert candidate.segments[-1] == unrelated


def test_independent_original_check_replaces_wrong_conditions_in_every_level(
    tmp_path, monkeypatch
):
    """核对请求包含四档与原文，修订结果成为最终版本；参数：隔离空间和脚本模型；返回：无。"""
    messages, source, _ = source_history(tmp_path)
    client = from_test_turns([])
    observed = []
    wrong = "允许上传，上限5000，方案已经完成"
    correct = {
        "p1": "本地材料已检查；费用不得超过500，禁止上传；方案未定",
        "p2": "本地方案评估：费用限500，禁止上传，方案未定",
        "p3": "本地方案未定；限500且禁止上传",
        "p4": "本地/限500/禁止上传/未定",
    }

    def stream(request, **_options):
        """生成错误候选再按原文提供修订；参数：真实请求；返回：脚本模型事件。"""
        body = "\n".join(model_visible_text(message) for message in request.messages)
        payload = json.loads(body.rsplit("\n", 1)[-1])
        observed.append(payload)
        auditing = "独立原文核对：" in body
        segment = {
            "segment_id": "local",
            "title": "本地方案评估",
            "message_ids": [
                message.message_id
                for message in source.view.messages[: source.covered_count]
            ],
        }
        if auditing:
            assert all(
                payload["current_segments"][0][level] == wrong for level in correct
            )
            assert "费用不得超过500，禁止上传" in json.dumps(
                payload["original_groups"], ensure_ascii=False
            )
        segment.update(correct if auditing else dict.fromkeys(correct, wrong))
        delta = {
            "base_summary_id": None,
            "segments": [segment],
            "dispositions": [
                {
                    "message_id": identity,
                    "destinations": ["original"],
                    "reason": "可回查",
                }
                for identity in payload["required_disposition_message_ids"]
            ],
        }
        return _response(json.dumps(delta, ensure_ascii=False), len(observed))

    def summarize(bundle: ProductionContextBundle):
        """执行实际准备的辅助请求；参数：上下文包；返回：模型计划。"""
        return client.plan(bundle.model_task, context=dict(bundle.model_context))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    store = SessionCompactionStore(messages)
    compactor = ContextCompactor(store, client.prepare_request, summarize)
    content, request_ids = compactor._summarize(
        CompactionMaterial(source, ""),
        {
            "session_id": "long",
            "run_id": "r",
            "segment_id": "s",
            "tool_registry": None,
            "history_representation_version": 1,
        },
        context_window=0,
    )
    saved = store.publish(
        source, content.render(), content=content, request_ids=request_ids
    )
    assert len(observed) == 2
    for level, text in correct.items():
        assert getattr(saved.content.segments[0], level) == text
