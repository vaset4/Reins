"""从实际聊天入口验证纠正接纳、审批与扩展装配。

作者：xxx
时间：2026-09-14 15:00:00
"""

from __future__ import annotations
from runtime.session_state import SessionStateStore
from scripts.testing.llm import (
    from_test_native_tool_then_final,
    from_test_sequence,
    from_test_stub,
)

from threading import Event

import pytest
from rich.console import Console

from app.repl import run_repl
from app.repl.console import reset_console_for_tests
from llm.messages import (
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    model_visible_text,
)
from runtime.extensions import RuntimeExtensions
from runtime.session_message_store import SessionMessageStore
from runtime.session_compaction import SessionCompactionStore
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
from tests.test_session_runtime import capture_requests


@pytest.fixture(params=[run_repl], ids=["repl"])
def entrypoint(request):
    """使用生产入口并隔离终端输出；传参：测试参数；返回：实际入口函数。"""
    reset_console_for_tests(Console(record=True, force_terminal=False))
    yield request.param
    reset_console_for_tests(None)


def test_entrypoint_accepts_correction_during_model_generation(
    tmp_path, monkeypatch, entrypoint
):
    """输入线程在模型生成时接纳纠正，下一实际请求使用一次；传参：目录/替换器/入口；返回：无。"""
    started, release = Event(), Event()

    def first_request():
        """在供应商请求内建立确定时机；传参：无；返回：无。"""
        started.set()
        assert release.wait(5)

    client = from_test_sequence(["旧结论", "采用新的中文约束"])
    requests = capture_requests(client, monkeypatch, before_first=first_request)
    answers = iter(["改成中文，并保留来源", "/exit"])

    def prompt(_label):
        """输入纠正后才释放第一请求；传参：提示；返回：用户输入。"""
        assert started.wait(5)
        answer = next(answers)
        if answer == "/exit":
            release.set()
        return answer

    try:
        assert (
            entrypoint(
                project_root=tmp_path,
                data_root=tmp_path,
                llm_client=client,
                tool_registry=ToolRegistry(),
                initial_message="原始要求",
                prompt_fn=prompt,
            )
            == 0
        )
    finally:
        release.set()
    assert len(requests) == 2
    messages = requests[-1].messages
    assert (
        sum(
            isinstance(item, UserMessage)
            and model_visible_text(item) == "改成中文，并保留来源"
            for item in messages
        )
        == 1
    )
    session = SessionStateStore(tmp_path).list_recent()[0].session_id
    store = SessionMessageStore(tmp_path)
    assert (
        len([row for row in store.read_entries(session) if row.type == "inbound"]) == 2
    )
    assert "旧结论" not in str(store.materialize(session).messages)


def test_entrypoint_ordinary_input_cancels_pending_approval(
    tmp_path, monkeypatch, entrypoint
):
    """普通纠正不会被当审批答案，原待批动作不执行；传参：目录/替换器/入口；返回：无。"""
    presented, decision_received = Event(), Event()
    effects = []
    monkeypatch.setattr(
        "app.repl.session_host._present_approval", lambda *_args: presented.set()
    )
    monkeypatch.setattr(
        "app.repl.session_host._present_batch", lambda *_args: presented.set()
    )
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "action",
            "需要批准的动作",
            {},
            "agent",
            ToolRisk.CONFIRM,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            executor=lambda _args: effects.append(True) or "实际执行",
        )
    )
    client = from_test_native_tool_then_final(
        [ToolCallPart("needs-approval", "action", {})], "按新要求继续"
    )
    requests = capture_requests(client, monkeypatch)

    def finished(_event):
        """以真实后续请求完成为退出屏障；传参：运行事实；返回：无后续动作。"""
        decision_received.set()
        return ()

    commands = iter(["不用执行，先解释", "/exit"])

    def prompt(_label):
        """只有原动作正在审批时才输入普通纠正；传参：提示；返回：输入。"""
        assert presented.wait(5)
        answer = next(commands)
        if answer == "/exit":
            assert decision_received.wait(5)
        return answer

    entrypoint(
        project_root=tmp_path,
        data_root=tmp_path,
        llm_client=client,
        tool_registry=registry,
        initial_message="执行动作",
        prompt_fn=prompt,
        extensions=RuntimeExtensions(after_run=(finished,)),
    )
    assert effects == []
    assert len(requests) == 2
    assert "不用执行，先解释" in str(requests[-1])
    result = next(
        item for item in requests[-1].messages if isinstance(item, ToolResultMessage)
    )
    assert result.status != "success"
    assert "not_started" in str(result)


def test_entrypoint_extension_material_reaches_adapter(
    tmp_path, monkeypatch, entrypoint
):
    """扩展材料经过实际入口进入Adapter请求；传参：目录/替换器/入口；返回：无。"""
    client = from_test_stub("已采用来源")
    requests = capture_requests(client, monkeypatch)
    extensions = RuntimeExtensions(
        context_sources=(lambda _event: "来源：资料A，约束：总额不超过100元",)
    )
    entrypoint(
        project_root=tmp_path,
        data_root=tmp_path,
        llm_client=client,
        tool_registry=ToolRegistry(),
        initial_message="整理计划",
        max_turns=1,
        extensions=extensions,
    )
    assert len(requests) == 1
    assert "资料A" in str(requests[0].instructions)
    assert "总额不超过100元" in str(requests[0].instructions)


def test_manual_compaction_uses_semantic_model_and_keeps_original(
    tmp_path, monkeypatch, entrypoint
):
    """REPL 聊天入口能显式生成有来源摘要；传参：临时根和入口；返回：无。"""
    import json
    from tests.test_semantic_compaction import _response, _summary_delta

    client = from_test_sequence([])
    requests = []

    def stream(request, **_options):
        """按真实生成/核对请求给出有来源差量；传参：请求；返回：供应商事件。"""
        requests.append(request)
        body = "\n".join(model_visible_text(message) for message in request.messages)
        if "整理下面的会话资料" in body or "核对" in body and "original_groups" in body:
            delta = json.loads(_summary_delta(body))
            for entry in delta["add"]:
                entry["text"] = "总费用不能超过500，原文不能上传。"
                entry["sources"][0]["quote"] = entry["text"]
            return _response(json.dumps(delta, ensure_ascii=False), len(requests))
        return _response(
            "已接收研究材料" if len(requests) == 1 else "会话已整理，原文可以继续回读",
            len(requests),
        )

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    commands = iter(["/compact", "/exit"])
    original = (
        "总费用不能超过500，原文不能上传。" + "有来源的研究材料 " * 900 + "资料结束"
    )
    assert (
        entrypoint(
            project_root=tmp_path,
            data_root=tmp_path,
            llm_client=client,
            tool_registry=ToolRegistry(),
            initial_message=original,
            prompt_fn=lambda _prompt: next(commands),
        )
        == 0
    )
    assert len(requests) == 4
    assert "整理下面的会话资料" in str(requests[1].messages)
    assert "核对" in str(requests[2].messages)
    assert not requests[1].tools and not requests[2].tools and not requests[3].tools
    session_id = SessionStateStore(tmp_path).list_recent()[0].session_id
    owner = SessionMessageStore(tmp_path)
    view = owner.materialize(session_id)
    assert original == model_visible_text(view.messages[0])
    summary = SessionCompactionStore(owner).current(view)
    assert summary is not None and "原文不能上传" in summary.text
    assert summary.summary_id in str(requests[3].instructions)
