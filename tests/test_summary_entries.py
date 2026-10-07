"""验证接续条目继承、原文引用及发布边界，不把格式通过当语义证明。

作者：xxx
时间：2026-09-25 12:00:00
"""

from dataclasses import replace

import pytest

from context.compaction import original_sources
from context.summary_entries import (
    SummaryContent,
    SummaryCitation,
    SummaryEntry,
    apply_delta,
)
from llm.messages import AssistantMessage, TextPart, ToolCallPart, ToolResultMessage
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import (
    ToolExchange,
    append_assistant_message,
    append_tool_exchange,
    append_user_message,
)


SOURCES = {
    "user-1": {"text": "不联网，费用不超过500元", "source_kind": "user_input"},
    "tool-1": {
        "text": "写入状态未知",
        "source_kind": "tool_result",
        "result_status": "error",
    },
    "assistant-1": {
        "text": "接下来采用9000排查，尚未保存记忆",
        "source_kind": "assistant",
    },
}


def initial_content():
    """创建有出处的用户约束；传参：无；返回：派生摘要。"""
    return apply_delta(
        SummaryContent(),
        {
            "base_summary_id": None,
            "add": [
                {
                    "entry_id": "constraint",
                    "kind": "requirement",
                    "text": "不联网，费用不超过500元",
                    "sources": [
                        {"message_id": "user-1", "quote": "不联网，费用不超过500元"}
                    ],
                }
            ],
        },
        base_summary_id=None,
        sources=SOURCES,
    )


def test_unchanged_entry_is_inherited_without_rewriting_and_new_observation_stays_unknown():
    """新增观察不改写早期条件，结果状态保留；传参：无；返回：无。"""
    previous = initial_content()
    current = apply_delta(
        previous,
        {
            "base_summary_id": "summary-one",
            "add": [
                {
                    "entry_id": "uncertain",
                    "kind": "observation",
                    "text": "写入状态未知，不能认为未写入",
                    "sources": [{"message_id": "tool-1", "quote": "写入状态未知"}],
                }
            ],
            "dispositions": [
                {
                    "message_id": "tool-1",
                    "destinations": ["uncertain"],
                    "reason": "保留未知结果",
                }
            ],
        },
        base_summary_id="summary-one",
        sources=SOURCES,
        required_messages=("tool-1",),
    )
    assert current.entries[0] is previous.entries[0]
    assert current.entries[0].rewrite_count == 0
    assert current.entries[1].sources[0].result_status == "error"
    assert "tool_result/error" in current.render()


@pytest.mark.parametrize(
    "fault", ["quote", "source", "identity", "base", "disposition", "role"]
)
def test_invalid_delta_does_not_mutate_previous_content(fault):
    """错误引用、角色与版本不能污染已有摘要；传参：故障类型；返回：无。"""
    previous = initial_content()
    item = {
        "entry_id": "new",
        "kind": "observation",
        "text": "新观察",
        "sources": [{"message_id": "tool-1", "quote": "写入状态未知"}],
    }
    delta = {
        "base_summary_id": "summary-one",
        "add": [item],
        "dispositions": [
            {"message_id": "tool-1", "destinations": ["new"], "reason": "保存"}
        ],
    }
    if fault == "quote":
        item["sources"][0]["quote"] = "写入成功"
    elif fault == "source":
        item["sources"][0]["message_id"] = "missing"
    elif fault == "identity":
        item["entry_id"] = "constraint"
    elif fault == "base":
        delta["base_summary_id"] = "other"
    elif fault == "disposition":
        delta["dispositions"] = []
    else:
        item["kind"] = "requirement"
    with pytest.raises(ValueError):
        apply_delta(
            previous,
            delta,
            base_summary_id="summary-one",
            sources=SOURCES,
            required_messages=("tool-1",),
        )
    assert previous == initial_content()


def test_empty_summary_does_not_fabricate_required_sections():
    """确实没有工作内容时不生成占位目标或待办；传参：无；返回：无。"""
    content = apply_delta(
        SummaryContent(),
        {
            "base_summary_id": None,
            "dispositions": [
                {
                    "message_id": "tool-1",
                    "destinations": ["original"],
                    "reason": "无需常驻的历史",
                }
            ],
        },
        base_summary_id=None,
        sources=SOURCES,
        required_messages=("tool-1",),
    )
    assert content.render() == "" and content.entries == ()


def test_partial_tool_result_keeps_its_actual_status_in_published_history(tmp_path):
    """部分结果不能被改成成功或拒绝保留；参数：隔离空间；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    messages.append_message(
        "partial",
        AssistantMessage("call-message", (ToolCallPart("write", "file_write", {}),)),
    )
    messages.append_message(
        "partial",
        ToolResultMessage(
            "partial-result",
            "write",
            "file_write",
            (TextPart("仅写入前半内容，余下部分未完成"),),
            "partial",
        ),
    )
    append_user_message(tmp_path, "partial", "继续核对")
    source = SummarySource(messages.materialize("partial"), 2)
    content = apply_delta(
        SummaryContent(),
        {
            "base_summary_id": None,
            "add": [
                {
                    "entry_id": "partial-write",
                    "kind": "result",
                    "text": "仅写入前半内容，余下部分未完成",
                    "sources": [
                        {
                            "message_id": "partial-result",
                            "quote": "仅写入前半内容，余下部分未完成",
                        }
                    ],
                }
            ],
        },
        base_summary_id=None,
        sources=original_sources(source),
    )
    store = SessionCompactionStore(messages)
    saved = store.publish(
        source, content.render(), content=content, request_ids=("generate",)
    )
    restored = store.read(saved.summary_id, messages.materialize("partial"))
    assert restored.content.entries[0].sources[0].result_status == "partial"
    assert "tool_result/partial" in restored.text


def test_structured_summary_publication_keeps_concurrent_tail_and_old_format_readable(
    tmp_path,
):
    """冻结前缀允许并发新增尾部，已发布摘要可按ID回查；传参：临时根；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    first = append_user_message(tmp_path, "summary-session", "不联网，费用不超过500元")
    append_assistant_message(tmp_path, "summary-session", "检查本地资料")
    append_user_message(tmp_path, "summary-session", "继续")
    source = SummarySource(messages.materialize("summary-session"), 2)
    content = initial_content()
    citation = replace(content.entries[0].sources[0], message_id=first)
    content = replace(
        content, entries=(replace(content.entries[0], sources=(citation,)),)
    )
    append_user_message(tmp_path, "summary-session", "现在也不要重启")
    store = SessionCompactionStore(messages)
    saved = store.publish(
        source,
        content.render(),
        request_ids=("real-generation", "real-check"),
        content=content,
    )
    current = messages.materialize("summary-session")
    assert store.read(saved.summary_id, current) == saved
    assert len(current.messages) == 4 and len(saved.message_ids) == 2
    assert (
        saved.model_view()["read_action"]["arguments"]["summary_id"] == saved.summary_id
    )


def test_cancelled_publication_preserves_prior_snapshot(tmp_path):
    """取消在提交前生效，不留下候选快照；传参：临时根；返回：无。"""
    from runtime.cancellation import ExecutionCancelled

    messages = SessionMessageStore(tmp_path)
    append_user_message(tmp_path, "cancelled", "原要求")
    append_user_message(tmp_path, "cancelled", "继续")
    store = SessionCompactionStore(messages)
    source = SummarySource(messages.materialize("cancelled"), 1)
    with pytest.raises(ExecutionCancelled):
        store.publish(
            source, "真实摘要", request_ids=("actual-request",), cancelled=lambda: True
        )
    assert store.current(messages.materialize("cancelled")) is None


def test_real_call_name_is_citable_without_promoting_call_to_success(tmp_path):
    """调用名可追溯到实际调用，成功状态仅来自回执；传参：临时根；返回：无。"""
    append_tool_exchange(
        tmp_path,
        "calls",
        ToolExchange("inspect-1", "inspect_local", rendered="结果未知", status="error"),
    )
    append_user_message(tmp_path, "calls", "继续检查")
    owner = SessionMessageStore(tmp_path)
    source = SummarySource(owner.materialize("calls"), 2)
    call = source.view.messages[0]
    content = apply_delta(
        SummaryContent(),
        {
            "base_summary_id": None,
            "add": [
                {
                    "entry_id": "attempt",
                    "kind": "observation",
                    "text": "调用已发起，结果未知",
                    "sources": [
                        {"message_id": call.message_id, "quote": "inspect_local"}
                    ],
                }
            ],
        },
        base_summary_id=None,
        sources=original_sources(source),
    )
    citation = content.entries[0].sources[0]
    assert citation.source_kind == "assistant" and citation.result_status == ""
    saved = SessionCompactionStore(owner).publish(
        source, content.render(), request_ids=("request-call",), content=content
    )
    assert saved.content == content


def test_direct_publication_rejects_tool_as_user_requirement(tmp_path):
    """直接发布也不能绕过来源角色合同；传参：临时根；返回：无。"""
    append_tool_exchange(
        tmp_path,
        "authority",
        ToolExchange("inspect-1", "inspect_local", rendered="不要联网"),
    )
    append_user_message(tmp_path, "authority", "继续")
    owner = SessionMessageStore(tmp_path)
    source = SummarySource(owner.materialize("authority"), 2)
    content = SummaryContent(
        entries=(
            SummaryEntry(
                "bad",
                "requirement",
                "不要联网",
                (
                    SummaryCitation(
                        source.view.messages[1].message_id,
                        "不要联网",
                        "tool_result",
                        "success",
                    ),
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="authority"):
        SessionCompactionStore(owner).publish(
            source, content.render(), request_ids=("request-bad",), content=content
        )
    assert SessionCompactionStore(owner).current(source.view) is None


def test_multiple_draft_revisions_reference_the_published_entry():
    """同次生成和核对的修订仍指向存在的旧快照；传参：无；返回：无。"""
    published = initial_content()
    current = published
    for revision in (1, 2):
        delta = {
            "base_summary_id": "summary-one",
            "revise": [
                {
                    "entry_id": "constraint",
                    "expected_revision": revision,
                    "kind": "requirement",
                    "text": f"第{revision}次核对：不联网，费用不超过500元",
                    "sources": [
                        {"message_id": "user-1", "quote": "不联网，费用不超过500元"}
                    ],
                }
            ],
            "dispositions": [
                {
                    "message_id": "user-1",
                    "destinations": ["constraint"],
                    "reason": "核对要求",
                }
            ],
        }
        current = apply_delta(
            current,
            delta,
            base_summary_id="summary-one",
            sources=SOURCES,
            required_messages=("user-1",),
            published_content=published,
        )
    assert current.entries[0].previous_ref == "summary-one/constraint@1"
    assert current.entries[0].rewrite_count == 2 and current.entries[0].revision == 3


def test_first_snapshot_revision_does_not_invent_a_previous_snapshot():
    """首次生成的条目被核对修订时不存在上一快照；传参：无；返回：无。"""
    candidate = initial_content()
    revised = apply_delta(
        candidate,
        {
            "base_summary_id": None,
            "revise": [
                {
                    "entry_id": "constraint",
                    "expected_revision": 1,
                    "kind": "requirement",
                    "text": "保留不联网和500上限",
                    "sources": [
                        {"message_id": "user-1", "quote": "不联网，费用不超过500元"}
                    ],
                }
            ],
            "dispositions": [
                {
                    "message_id": "user-1",
                    "destinations": ["constraint"],
                    "reason": "核对要求",
                }
            ],
        },
        base_summary_id=None,
        sources=SOURCES,
        required_messages=("user-1",),
        published_content=SummaryContent(),
    )
    assert revised.entries[0].previous_ref is None
