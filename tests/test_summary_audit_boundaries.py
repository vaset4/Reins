"""验证超大旧依据逐片核对与提交后的接续边界。

作者：xxx
时间：2026-09-26 12:00:00
"""

from scripts.testing.llm import from_test_turns
import json

import pytest

from context.compaction import CompactionMaterial, original_sources, source_groups
from context.summary_entries import SummaryContent
from context.production_builder import ProductionContextBundle
from context.window import request_budget
from scripts.testing.llm import ScriptedTurnOptions
from llm.messages import model_visible_text
from runtime.context_preparation import ContextCompactor
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_assistant_message, append_user_message
from tests.test_semantic_compaction import _response, _summary_delta


def test_audit_reads_all_large_old_source_pages_within_window(tmp_path, monkeypatch):
    """旧依据不能反复整段回填造成无限缩片；传参：隔离目录与替换器；返回：无。"""
    old_text = (
        "旧依据开始。" + "已完成的过程说明，没有新增决定。\n" * 700 + "旧依据结束。"
    )
    append_assistant_message(tmp_path, "audit", old_text)
    append_user_message(tmp_path, "audit", "本轮仅核对实际变化")
    append_user_message(tmp_path, "audit", "继续")
    owner = SessionMessageStore(tmp_path)
    source = SummarySource(owner.materialize("audit"), 2)
    old_id = source.view.messages[0].message_id
    groups = source_groups(CompactionMaterial(source, ""))
    client = from_test_turns([], options=ScriptedTurnOptions(context_window=6500))
    observed = []

    def stream(request, **_options):
        """原样记录实际核对范围；传参：生产请求；返回：无修订测试差量。"""
        observed.append(request)
        body = "\n".join(model_visible_text(message) for message in request.messages)
        payload = json.loads(body.rsplit("\n", 1)[-1])
        delta = {
            "base_summary_id": None,
            "dispositions": [
                {
                    "message_id": identity,
                    "destinations": ["original"],
                    "reason": "原文可追溯",
                }
                for identity in payload["required_disposition_message_ids"]
            ],
        }
        return _response(json.dumps(delta), len(observed))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    compactor = ContextCompactor(
        SessionCompactionStore(owner),
        client.prepare_request,
        lambda bundle: client.plan(
            bundle.model_task, context=dict(bundle.model_context)
        ),
    )
    checked = compactor._audit(
        SummaryContent(),
        groups[1:],
        {"context_purpose": "compaction"},
        requests=[],
        base=None,
        sources=original_sources(source),
        auxiliary=[],
        related=groups[:1],
        published_content=SummaryContent(),
    )
    assert len(observed) > 1
    assert all(
        request_budget(request, 6500).required_total <= 6500 for request in observed
    )
    spans = sorted(
        (span["start"], span["end"])
        for check in checked.checks
        for span in check["source_spans"]
        if span["message_id"] == old_id
    )
    assert spans[0][0] == 0 and spans[-1][1] == len(old_text)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))


def test_rebuild_failure_preserves_published_summary_and_concurrent_input(
    tmp_path, monkeypatch
):
    """摘要发布后重建失败，重启仍使用同一摘要和真实并发输入；传参：隔离根与替换器；返回：无。"""
    append_user_message(tmp_path, "rebuild", "不要联网")
    append_assistant_message(
        tmp_path, "rebuild", "已完成的目录核对过程，无新增决定。" * 200
    )
    append_user_message(tmp_path, "rebuild", "继续排查")
    owner = SessionMessageStore(tmp_path)
    original = owner.read_entries("rebuild")
    store = SessionCompactionStore(owner)
    client = from_test_turns([])
    calls = []

    def stream(request, **_options):
        """在生成期间接纳新要求，回复有来源的测试摘要；传参：请求；返回：供应商事件。"""
        calls.append(request)
        if len(calls) == 1:
            append_user_message(tmp_path, "rebuild", "现在不要重启")
        body = "\n".join(model_visible_text(message) for message in request.messages)
        return _response(_summary_delta(body), len(calls))

    def bundle():
        """从现有消息和摘要所有者重建输入；传参：无；返回：生产上下文包。"""
        view = owner.materialize("rebuild")
        summary = store.current(view)
        messages = (
            view.messages[len(summary.message_ids) :] if summary else view.messages
        )
        return ProductionContextBundle(
            "继续",
            {
                "session_id": "rebuild",
                "run_id": "run",
                "segment_id": "segment",
                "tool_registry": None,
                "conversation_history": messages,
                "input_message_id": messages[-1].message_id,
                "session_summary": summary.model_view() if summary else None,
            },
            (),
        )

    def failed_rebuild():
        """模拟发布后的上下文读取故障；传参：无；返回：不返回。"""
        raise OSError("rebuild failed after committed summary")

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    compactor = ContextCompactor(
        store,
        client.prepare_request,
        lambda value: client.plan(value.model_task, context=dict(value.model_context)),
    )
    with pytest.raises(OSError, match="committed summary"):
        compactor.fit(bundle(), rebuild=failed_rebuild, force=True)
    saved = store.current(owner.materialize("rebuild"))
    assert saved is not None
    call_count = len(calls)
    restored = ContextCompactor(
        SessionCompactionStore(SessionMessageStore(tmp_path)),
        client.prepare_request,
        lambda value: client.plan(value.model_task, context=dict(value.model_context)),
    )
    prepared = restored.fit(bundle(), rebuild=bundle).model_context["prepared_request"]
    assert len(calls) == call_count
    assert store.current(owner.materialize("rebuild")) == saved
    assert (
        "不要联网" in prepared.render_text_to_model
        and "现在不要重启" in prepared.render_text_to_model
    )
    assert owner.read_entries("rebuild")[: len(original)] == original
