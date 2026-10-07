"""验证文件更新、两次压缩、未保存事实、长技能和宿主恢复组成同一工作链。

作者：xxx
时间：2026-09-25 20:00:00
"""

from scripts.testing.llm import _from_scripted, from_test_turns
import json
from contextlib import closing
from dataclasses import asdict, replace

from app.background.sessions import BackgroundSession, SessionRecord, SessionServices
from context.compaction import CompactionMaterial
from scripts.testing.llm import ScriptedTurnOptions, _ScriptedTurn
from llm.messages import (
    ToolCallPart,
    UserMessage,
    agent_message_to_mapping,
    model_visible_text,
)
from memory.records import MemoryDetails, MemorySource
from memory.store import MemoryStore
from runtime.agent_loop import State
from runtime.workspaces import WorkspaceStore
from runtime.context_preparation import ContextCompactor
from runtime.history_reader import read_history_page
from runtime.run_facts import RunFactStore
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_assistant_message, append_user_message
from skills.store import SkillStore, build_skill_markdown
from tasks.store import TaskStore
from tests.test_file_batch_tracking import _runtime
from tests.test_semantic_compaction import _response, _summary_delta
from tests.test_session_runtime import capture_requests
from tools.file_persistence import content_sha256
from tools.tool_registry import ToolRegistry


def summarize_working_chain(root, owner, monkeypatch):
    """通过真实请求适配器驱动两次摘要发布；传参：隔离根、消息与替换器；返回：摘要链，不作为模型语义证据。"""
    client = from_test_turns([], options=ScriptedTurnOptions(context_window=16000))

    def stream(request, **_options):
        """固定保留场景中的公开工作事实与原话；传参：实际请求；返回：带真实来源的测试响应。"""
        body = "\n".join(model_visible_text(message) for message in request.messages)
        delta = json.loads(_summary_delta(body))
        material = json.loads(body.rsplit("\n", 1)[-1])
        existing = {entry["entry_id"] for entry in material["current_entries"]}
        for group in material["original_groups"]:
            for message in group:
                text = "\n".join(part.get("text", "") for part in message["content"])
                if text.startswith("实际采用9000") and "working-port" not in existing:
                    delta["add"].append(
                        {
                            "entry_id": "working-port",
                            "kind": "conclusion",
                            "text": text,
                            "sources": [
                                {"message_id": message["message_id"], "quote": text}
                            ],
                        }
                    )
        return _response(json.dumps(delta, ensure_ascii=False), 1)

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    store = SessionCompactionStore(owner)
    context = {
        "session_id": "joint",
        "run_id": "summary-test",
        "segment_id": "summary",
        "tool_registry": ToolRegistry(),
    }
    records = []
    for index in range(2):
        append_assistant_message(root, "joint", "已核对的过程材料\n" * 100)
        append_user_message(root, "joint", f"继续本地诊断，第{index + 1}段")
        view = owner.materialize("joint")
        previous = store.current(view)
        source = SummarySource(view, len(view.messages) - 1, previous)
        text = json.dumps(
            [agent_message_to_mapping(message) for message in view.messages],
            ensure_ascii=False,
        )
        compactor = ContextCompactor(
            store,
            client.prepare_request,
            lambda bundle: client.plan(
                bundle.model_task, context=dict(bundle.model_context)
            ),
        )
        content, requests = compactor._summarize(
            CompactionMaterial(source, text), context, context_window=16000
        )
        records.append(
            store.publish(
                source, content.render(), request_ids=requests, content=content
            )
        )
    return records


def seed_working_chain(root):
    """真实改写本地文件，并保留未入库的公开结论；传参：测试项目；返回：原文、运行及恢复事实。"""
    loop, context, registry = _runtime(root, [])
    data = loop.data_root
    WorkspaceStore(data).bind_session("joint", root)
    owner, facts = SessionMessageStore(data), RunFactStore(data)
    owner.accept_input(
        "joint", "以前端口是8000；只诊断，不联网、不重启", input_id="original-input"
    )
    owner.deliver_inputs("joint", run_id="original-run", task_id=context.task_id)
    facts.append(
        {"event": "run:start", "session_id": "joint", "run_id": "original-run"}
    )
    facts.append(
        {
            "event": "input:handled",
            "session_id": "joint",
            "run_id": "original-run",
            "input_ids": ["original-input"],
        }
    )
    context.session_id = "joint"
    context.payload.update(
        message="以前端口是8000；只诊断，不联网、不重启",
        input_message_id="original-input",
    )
    path = root / "inspection.txt"
    path.write_text("port=8000", encoding="utf-8")
    calls = (
        ToolCallPart("read-old", "file_read", {"path": str(path)}),
        ToolCallPart(
            "write-new",
            "file_write",
            {
                "path": str(path),
                "content": "port=9000",
                "expected_sha256": content_sha256(b"port=8000"),
            },
        ),
        ToolCallPart("read-new", "file_read", {"path": str(path)}),
    )
    loop.llm_client = _from_scripted(
        [
            *(_ScriptedTurn(calls=(call,)) for call in calls),
            _ScriptedTurn(
                text="实际采用9000继续诊断，有当前文件证据；尚未保存长期记忆。"
            ),
        ]
    )
    assert loop.run(context) is State.DONE
    with closing(MemoryStore(data)) as memories:
        memories.create_memory(
            "fact",
            "旧记录：服务端口8000",
            ["端口"],
            memory_id="port",
            details=MemoryDetails(
                subject="本地服务",
                fact_key="端口",
                sources=(
                    MemorySource("user_input", "original-input", session_id="joint"),
                ),
            ),
        )
    SkillStore(data).create_skill(
        "diagnostic-guide",
        build_skill_markdown(
            name="诊断指南", body="长技能正文标记\n" + "核对证据\n" * 15000
        ),
        meta={},
    )
    with closing(TaskStore(data)) as tasks:
        tasks.update_task_refs(context.task_id, skill_refs=["diagnostic-guide"])
    return loop, context, owner, facts, registry


def test_recovery_after_two_compactions_keeps_working_fact_without_saving_memory(
    tmp_path, monkeypatch
):
    """恢复后同一请求同时包含新结论、旧记忆身份和有效要求；传参：临时根及捕获器；返回：无。"""
    loop, context, owner, facts, registry = seed_working_chain(tmp_path)
    records = summarize_working_chain(loop.data_root, owner, monkeypatch)
    before = owner.materialize("joint")
    inbound_count = sum(
        entry.type == "inbound" for entry in owner.read_entries("joint")
    )
    with closing(MemoryStore(loop.data_root)) as memories:
        saved_memory = memories.load_memory("port")
    client = from_test_turns(
        ["继续按9000诊断，保持不联网和不重启"],
        options=ScriptedTurnOptions(context_window=16000),
    )
    requests = capture_requests(client, monkeypatch)
    record = SessionRecord(
        "joint",
        compatibility_task_id=context.task_id,
        status="running",
        intent={
            "run_id": "original-run",
            "root_run_id": "original-run",
            "input_id": "original-input",
            "task_id": context.task_id,
            "focus_task_id": context.task_id,
            "trigger": "user",
            "lease": asdict(context.capability_lease),
        },
    )
    session = BackgroundSession(
        record,
        SessionServices(
            tmp_path, loop.data_root, lambda _options: client, lambda: registry
        ),
    )
    session.records.save_input(
        record.session_id, "original-input", {}, needs_ephemeral_key=False
    )
    try:
        session.recover()
        assert session.runtime.wait_idle(10)
        assert session.record.status == "done", session.record.error
        assert len(requests) == 1
        text = "\n".join(part.text for part in requests[0].instructions)
        assert "实际采用9000" in text and "尚未保存长期记忆" in text
        assert (
            "不联网、不重启" in text
            and "skill_read" in text
            and "长技能正文标记" not in text
        )
        assert "recovery_intent" in text
        users = [
            message
            for message in requests[0].messages
            if isinstance(message, UserMessage)
        ]
        assert sum(message.message_id == "original-input" for message in users) == 1
        assert {message.message_id for message in users} <= {
            message.message_id for message in before.messages
        }
        assert (
            sum(entry.type == "inbound" for entry in owner.read_entries("joint"))
            == inbound_count
        )
        with closing(MemoryStore(loop.data_root)) as memories:
            # 【上下文恢复】【记忆核对】1. 召回可更新使用时间，正文、版本、来源与修订事实不能改变
            assert (
                replace(
                    memories.load_memory("port"), last_used_at=saved_memory.last_used_at
                )
                == saved_memory
            )
        history = read_history_page(
            owner.materialize("joint"),
            call_id="none",
            summaries=SessionCompactionStore(owner),
            summary_id=records[0].summary_id,
        )
        assert "port=9000" in str(history["messages"])
        allocations = [
            fact
            for fact in facts.read_run(requests[0].request_metadata.run_id)
            if fact.get("event") == "context:request_allocation"
        ]
        assert (
            allocations
            and not allocations[0]["selection"]["history_selection"][
                "line_limit_applied"
            ]
        )
        assert any(
            item["identity"].startswith("memory:port@")
            for item in allocations[0]["selection"]["materials"]
        )
    finally:
        session.close()
