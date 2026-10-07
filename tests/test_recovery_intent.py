"""验证宿主恢复引用原输入、持久幂等且不污染聊天正文。

作者：xxx
时间：2026-09-25 12:00:00
"""

from scripts.testing.llm import from_test_sequence
from contextlib import closing
from dataclasses import asdict, replace
from threading import Event

import pytest

from app.background.sessions import BackgroundSession, SessionRecord, SessionServices
from llm.messages import UserMessage, model_visible_text
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_runtime import RecoveryIntent, SessionRuntime
from tasks.store import TaskStore
from tests.test_session_runtime import capture_requests
from tools.tool_registry import ToolRegistry


def recovery_source(root):
    """准备已交付但宿主中断的原运行；传参：隔离根；返回：存储与恢复引用。"""
    messages, facts = SessionMessageStore(root), RunFactStore(root)
    messages.accept_input(
        "recovery-session", "只诊断，不联网、不重启", input_id="real-input"
    )
    messages.deliver_inputs("recovery-session", run_id="run-original", task_id=None)
    facts.append(
        {
            "event": "run:start",
            "session_id": "recovery-session",
            "run_id": "run-original",
        }
    )
    facts.append(
        {
            "event": "input:handled",
            "session_id": "recovery-session",
            "run_id": "run-original",
            "input_ids": ["real-input"],
        }
    )
    return (
        messages,
        facts,
        RecoveryIntent(
            "recovery-original", "run-original", "real-input", "host_interrupted"
        ),
    )


def test_recovery_reuses_handled_input_without_appending_body_and_is_idempotent(
    tmp_path,
):
    """已处理输入仍可恢复其未结束运行，重投不重复执行；传参：临时根；返回：无。"""
    messages, facts, intent = recovery_source(tmp_path)
    before = messages.read_entries("recovery-session")
    requests, started, release = [], Event(), Event()

    def run(request):
        """冻结执行窗口以验证并发重投；传参：运行引用；返回：无。"""
        requests.append(request)
        started.set()
        assert release.wait(5)

    runtime = SessionRuntime(
        "recovery-session", messages=messages, facts=facts, run=run
    )
    runtime.resume(intent)
    assert started.wait(5)
    runtime.resume(intent)
    release.set()
    assert runtime.wait_idle(5)
    restarted = SessionRuntime(
        "recovery-session", messages=messages, facts=facts, run=run
    )
    restarted.resume(intent)
    assert restarted.wait_idle(5)
    assert (
        len(requests) == 1
        and requests[0].input_id == "real-input"
        and requests[0].recovery == intent
    )
    assert messages.read_entries("recovery-session") == before
    events = [row["event"] for row in facts.read_run("run-original")]
    assert events.count("recovery:accepted") == events.count("recovery:handled") == 1


@pytest.mark.parametrize("change", ["other_session", "missing_input", "finished"])
def test_recovery_rejects_invalid_source_before_acceptance(tmp_path, change):
    """不存在或已结束的来源不能唤醒运行；传参：临时根与故障；返回：无。"""
    messages, facts, intent = recovery_source(tmp_path)
    if change == "missing_input":
        intent = replace(intent, input_id="missing")
    if change == "finished":
        facts.append(
            {
                "event": "run:lifecycle",
                "session_id": "recovery-session",
                "run_id": "run-original",
                "lifecycle": "waiting_user",
            }
        )
    calls = []
    runtime = SessionRuntime(
        "other" if change == "other_session" else "recovery-session",
        messages=messages,
        facts=facts,
        run=calls.append,
    )
    with pytest.raises(ValueError):
        runtime.resume(intent)
    assert calls == []
    assert not any(
        row["event"] == "recovery:accepted" for row in facts.read_run("run-original")
    )


def test_background_recovery_reaches_model_with_original_user_and_runtime_reference(
    tmp_path, monkeypatch
):
    """真实组装后的恢复请求保留用户要求且无伪用户续跑；传参：临时根与捕获器；返回：无。"""
    messages, facts, _intent = recovery_source(tmp_path)
    with closing(TaskStore(tmp_path)) as tasks:
        task = tasks.create_task("继续诊断", is_inbox=True)
    lease = from_trigger("user", task_id=task.task_id)
    record = SessionRecord(
        "recovery-session",
        compatibility_task_id=task.task_id,
        status="running",
        intent={
            "run_id": "run-original",
            "root_run_id": "run-original",
            "input_id": "real-input",
            "task_id": None,
            "focus_task_id": None,
            "trigger": "user",
            "lease": asdict(lease),
        },
    )
    client = from_test_sequence(["继续本地诊断，不重复未知副作用"])
    requests = capture_requests(client, monkeypatch)
    services = SessionServices(
        tmp_path, tmp_path, lambda _options: client, ToolRegistry
    )
    session = BackgroundSession(record, services)
    session.records.save_input(
        record.session_id, "real-input", {}, needs_ephemeral_key=False
    )
    try:
        session.recover()
        assert session.runtime.wait_idle(10)
        assert session.record.status == "done", session.record.error
        assert len(requests) == 1
        users = [
            model_visible_text(item)
            for item in requests[0].messages
            if isinstance(item, UserMessage)
        ]
        assert users == ["只诊断，不联网、不重启"]
        instructions = "\n".join(part.text for part in requests[0].instructions)
        assert "recovery_intent" in instructions and "host_interrupted" in instructions
        assert (
            len(
                [
                    entry
                    for entry in messages.read_entries("recovery-session")
                    if entry.type == "inbound"
                ]
            )
            == 1
        )
        session.recover()
        assert session.runtime.wait_idle(2) and len(requests) == 1
        assert any(
            row["event"] == "recovery:handled" for row in facts.read_run("run-original")
        )
    finally:
        session.close()
