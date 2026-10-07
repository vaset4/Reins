from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from context import production_builder
from context.production_builder import ContextReadError, ProductionContextBuilder
from llm.messages import AgentMessage, model_visible_text
from memory.store import MemoryStore
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.lease import from_trigger
from runtime.persistence import RuntimeStore
from runtime.run_facts import RunFactStore
from runtime.session_messages import append_assistant_message, append_user_message
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry


def _builder(data_root: Path) -> ProductionContextBuilder:
    return ProductionContextBuilder(
        data_root,
        system_prompt_provider=lambda: "system prompt",
    )


SESSION_ID = "session-context"


def _context(task_id: str | None, message: str) -> RunContext:
    lease_task = task_id or "compat-task"
    return RunContext(
        task_id=task_id,
        compatibility_task_id=None if task_id else lease_task,
        session_id=SESSION_ID,
        trigger=Trigger.USER,
        payload={"message": message},
        capability_lease=from_trigger("user", task_id=lease_task),
    )


def _writer(data_root: Path) -> LedgerWriter:
    return LedgerWriter(LedgerStore(data_root), source="test")


def _seed_user(data_root: Path, *texts: str) -> None:
    """把用户消息写进唯一消息 owner，供历史读取路径消费。

    参数：data_root 为 data 根目录；texts 为按时间顺序的用户输入
    返回：无
    """
    for text in texts:
        append_user_message(data_root, SESSION_ID, text)


def _seed_assistant(data_root: Path, text: str) -> None:
    """把一条模型回答写进唯一消息 owner。"""
    append_assistant_message(data_root, SESSION_ID, text)


def test_builder_applies_resume_hint_and_reuses_history_for_recall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = TaskStore(tmp_path)
    record = store.create_task("finish the page")
    store.update_summary_layers(record.task_id, resume_hint="Resume from section 2.")
    _seed_user(tmp_path, "older user")
    _seed_assistant(tmp_path, "older assistant")
    _seed_user(tmp_path, "latest user")
    captured: dict[str, object] = {}

    def fake_recall(data_root: Path, **kwargs: Any) -> tuple[str, tuple[()]]:
        captured["data_root"] = data_root
        captured["recent_user_text"] = kwargs["recent_user_text"]
        return "recall_memory=\nremember this", ()

    monkeypatch.setattr(production_builder, "recall_context_materials", fake_recall)

    bundle = _builder(tmp_path).build(
        task="继续",
        context=_context(record.task_id, "继续"),
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )

    assert bundle.model_task == "Resume from section 2.\n\nUser said: 继续"
    assert captured["recent_user_text"] == "latest user"
    conversation_history = cast(
        tuple[AgentMessage, ...],
        bundle.model_context["conversation_history"],
    )
    assert conversation_history[-1].kind == "user"
    assert model_visible_text(conversation_history[-1]) == "latest user"
    assert bundle.model_context["recall_context"] == "recall_memory=\nremember this"


def test_builder_returns_real_empty_recall_when_task_record_is_absent(
    tmp_path: Path,
) -> None:
    context = _context(None, "plain chat")

    bundle = _builder(tmp_path).build(
        task="plain chat",
        context=context,
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )

    assert bundle.model_context["storage_task_id"] == "compat-task"
    assert bundle.model_context["recall_context"] == ""
    recall_segments = [
        item for item in bundle.segments if item.name == "recall_context"
    ]
    assert len(recall_segments) == 1
    assert recall_segments[0].token_est == 0


def test_recall_uses_user_source_even_when_display_history_is_trimmed(
    tmp_path: Path,
) -> None:
    """后台输入与历史裁剪不能替换用户的召回主题；传参：临时目录；返回：无。"""
    from llm.model_request import compose_model_request
    from runtime.session_message_store import SessionMessageStore
    from skills.store import SkillStore, build_skill_markdown

    task = TaskStore(tmp_path).create_task("bill invoice")
    _seed_user(tmp_path, "核对账单", "记住饮食偏好，今晚晚餐清淡")
    owner = SessionMessageStore(tmp_path)
    for index in range(3):
        owner.accept_input(
            SESSION_ID,
            "后台继续检查星际航行",
            input_id=f"auto-{index}",
            input_source="agent",
        )
    owner.deliver_inputs(SESSION_ID, run_id="background", task_id=task.task_id)
    skills = SkillStore(tmp_path)
    skills.create_skill(
        "meal",
        build_skill_markdown(name="meal", body="饮食偏好，安排清淡晚餐"),
        meta={},
    )
    skills.create_skill(
        "space",
        build_skill_markdown(name="space", body="后台继续检查星际航行"),
        meta={},
    )
    builder = _builder(tmp_path)
    bundle = builder.build(
        task="继续",
        context=_context(task.task_id, "继续"),
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )
    request = compose_model_request(
        task=bundle.model_task,
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context=bundle.model_context,
        registry=ToolRegistry(),
        context_window=30000,
    ).request

    instructions = "\n".join(part.text for part in request.instructions)
    assert "安排清淡晚餐" in instructions
    assert "recall_skill" in instructions
    assert "星际航行" not in instructions
    assert (
        builder.read_conversation_history(SESSION_ID, limit=1).recall_user_text
        == "记住饮食偏好，今晚晚餐清淡"
    )


def test_chinese_topic_change_updates_request_recall_without_scope_leak(
    tmp_path: Path,
) -> None:
    """用户换话题后请求召回晚餐记忆，其他目标原件不串入；传参：临时目录；返回：无。"""
    from contextlib import closing

    from llm.model_request import compose_model_request
    from memory.store import MemoryDetails

    task = TaskStore(tmp_path).create_task("bill invoice")
    foreign_workspace = WorkspaceStore(tmp_path).register(tmp_path / "another-project")
    with closing(MemoryStore(tmp_path)) as memories:
        for index in range(3):
            memories.create_memory(
                "preference",
                f"bill invoice 账单 {index}",
                [],
                memory_id=f"bill-{index}",
            )
        memories.create_memory("preference", "晚餐选择清淡蔬菜", [], memory_id="meal")
        memories.create_memory(
            "preference",
            "晚餐跨目标秘密",
            [],
            memory_id="foreign-goal",
            details=MemoryDetails(scope="goal:another-goal"),
        )
        memories.create_memory(
            "preference",
            "晚餐跨项目秘密",
            [],
            memory_id="foreign-project",
            details=MemoryDetails(scope=f"project:{foreign_workspace.workspace_id}"),
        )
    _seed_user(tmp_path, "账单")
    builder = _builder(tmp_path)
    first = builder.build(
        task="账单",
        context=_context(task.task_id, "账单"),
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )
    assert "晚餐选择清淡蔬菜" not in str(first.model_context["recall_context"])
    _seed_user(tmp_path, "晚餐")

    second = builder.build(
        task="晚餐",
        context=_context(task.task_id, "晚餐"),
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )
    request = compose_model_request(
        task=second.model_task,
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context=second.model_context,
        registry=ToolRegistry(),
        context_window=30000,
    ).request

    instructions = "\n".join(part.text for part in request.instructions)
    assert "晚餐选择清淡蔬菜" in instructions
    assert "跨目标秘密" not in instructions
    assert "跨项目秘密" not in instructions


def test_builder_writes_recall_injection_explain_fact(tmp_path: Path) -> None:
    # AC1：生产 live 召回链带 RunContext → recall_context_body 算出 snapshot 后
    # 写回 memory:injection_explain 观测事件（含 injected/skipped/warnings），补回可排查性
    store = TaskStore(tmp_path)
    record = store.create_task("run focused pytest", task_id="2026-07-23-recall")
    memories = MemoryStore(tmp_path)
    memories.create_memory(
        "fact",
        "keep focused pytest tests near the fix",
        ["pytest"],
        memory_id="obs-fact-1",
    )
    memories.close()
    context = _context(record.task_id, "run focused pytest")

    _builder(tmp_path).build(
        task="run focused pytest",
        context=context,
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )

    facts = RunFactStore(tmp_path).read_run(context.run_id)
    explains = [f for f in facts if f.get("event") == "memory:injection_explain"]
    assert len(explains) == 1
    explain = cast(dict[str, Any], explains[0]["explain"])
    assert explain["round_id"] == "run focused pytest"
    injected_ids = [item["memory_id"] for item in explain["injected"]]
    assert "obs-fact-1" in injected_ids
    assert "skipped" in explain
    assert "warnings" in explain


def test_builder_keeps_unsummarized_history(tmp_path: Path) -> None:
    """未发布语义摘要时保留全部原文；传参：临时存储；返回：无。"""
    _seed_user(tmp_path, *[f"{index} " + "word " * 300 for index in range(20)])

    selection = _builder(tmp_path).read_conversation_history(SESSION_ID)

    assert selection.truncated is False
    assert selection.retained_count == 20
    assert selection.retained_count == len(selection.messages)
    assert any(item.kind == "user" for item in selection.messages)
    assert all(item.kind != "system" for item in selection.messages)


def test_builder_segments_keep_external_tokens_est_schema(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    record = store.create_task("goal")
    _seed_user(tmp_path, "hello")

    bundle = _builder(tmp_path).build(
        task="goal",
        context=_context(record.task_id, "goal"),
        toolset_policy={"mode": "hybrid"},
        tool_registry=ToolRegistry(),
        system_reminder="[system_reminder]decide[/system_reminder]",
    )
    facts = [item.to_run_fact_segment() for item in bundle.segments]
    reminder = next(item for item in facts if item["name"] == "system_reminder")
    conversation = next(
        item for item in facts if item["name"] == "conversation_history"
    )
    policy = next(item for item in facts if item["name"] == "toolset_policy")

    assert "tokens_est" in reminder
    assert "token_est" not in reminder
    assert reminder["layer"] == "ephemeral"
    assert conversation["message_count"] == 1
    assert conversation["token_estimate_kind"] == "message_with_overhead"
    assert conversation["authority"] == "fact"
    assert policy["tokens_est"] == 0


def test_builder_read_errors_are_visible(tmp_path: Path) -> None:
    """规范会话记录损坏时明确暴露读取失败；传参：临时存储；返回：无。"""
    _seed_user(tmp_path, "hello")

    database = RuntimeStore(tmp_path)
    with database.snapshot() as source:
        record = source.list_raw("session_entry", session_id=SESSION_ID)[0]
    assert record.location is not None
    path = tmp_path / record.location.path
    raw = path.read_bytes()
    offset = record.location.offset
    path.write_bytes(raw[:offset] + b"!" + raw[offset + 1 :])

    with pytest.raises(ContextReadError) as exc_info:
        _builder(tmp_path).read_conversation_history(SESSION_ID)

    assert exc_info.value.category == "conversation_history_read_failed"


def test_builder_does_not_read_old_taskstore_current_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path)
    record = store.create_task("goal")
    store.update_summary_layers(record.task_id, resume_hint="Resume from Ledger.")
    _seed_user(tmp_path, "canonical turn")

    def fail_summary_layers(self: TaskStore, task_id: str) -> object:
        del self, task_id
        raise AssertionError("old summary source should not be read")

    # 旧对话来源已随消息 owner 切换退休，读不到比"读了就炸"更强
    assert not hasattr(TaskStore, "read_conversation_tail")
    assert not hasattr(TaskStore, "append_conversation")
    monkeypatch.setattr(TaskStore, "read_summary_layers", fail_summary_layers)

    bundle = _builder(tmp_path).build(
        task="继续",
        context=_context(record.task_id, "继续"),
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )

    assert bundle.model_task == "Resume from Ledger.\n\nUser said: 继续"
    history = cast(
        tuple[AgentMessage, ...],
        bundle.model_context["conversation_history"],
    )
    assert [item.kind for item in history] == ["user"]
    assert model_visible_text(history[0]) == "canonical turn"


def test_builder_does_not_swallow_recall_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = TaskStore(tmp_path)
    record = store.create_task("goal")

    def fail_recall(data_root: Path, **kwargs: Any) -> str:
        del data_root, kwargs
        raise RuntimeError("recall failed")

    monkeypatch.setattr(production_builder, "recall_context_materials", fail_recall)

    with pytest.raises(RuntimeError, match="recall failed"):
        _builder(tmp_path).build(
            task="goal",
            context=_context(record.task_id, "goal"),
            toolset_policy={},
            tool_registry=ToolRegistry(),
        )


def test_production_context_does_not_read_run_facts_for_tool_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """守卫不变量：生产 Context 组装路径不经过 context/engine.py 的 run-facts → 模型可见正文链。

    context/engine.py 中 _read_relevant_tool_rows / _tool_rows_from_facts / _resume_priority_context
    三个脚手架函数读取 run facts 用于测试路径的 tool_results segment 组装，生产路径不应调用这些函数。

    此测试通过两个维度确认隔离：
    1. 行为断言：monkeypatch RunFactStore 四个读方法并计数，build 完成后总计数必须为 0
    2. 产物断言：bundle.segments 中不应出现名为 tool_results 的 segment

    参数：monkeypatch 为 pytest fixture；tmp_path 为临时目录
    返回：无
    """
    store = TaskStore(tmp_path)
    record = store.create_task("implement feature", task_id="2026-09-01-guard")
    _seed_user(tmp_path, "start implementing")
    _seed_assistant(tmp_path, "working on it")
    _seed_user(tmp_path, "continue")

    # 1. 对 RunFactStore 四个读方法计数
    read_counts = {"total": 0}

    original_store_init = RunFactStore.__init__

    def patched_init(self: RunFactStore, data_root: Path | str) -> None:
        original_store_init(self, data_root)
        original_read_run = self.read_run
        original_read_run_tolerant = self.read_run_tolerant
        original_read_latest_lifecycle = self.read_latest_lifecycle
        original_read_task_facts = self.read_task_facts

        def counted_read_run(run_id: str) -> list[dict[str, Any]]:
            read_counts["total"] += 1
            return original_read_run(run_id)

        def counted_read_run_tolerant(run_id: str) -> list[dict[str, Any]]:
            read_counts["total"] += 1
            return original_read_run_tolerant(run_id)

        def counted_read_latest_lifecycle(run_id: str) -> dict[str, Any] | None:
            read_counts["total"] += 1
            return original_read_latest_lifecycle(run_id)

        def counted_read_task_facts(task_id: str) -> list[dict[str, Any]]:
            read_counts["total"] += 1
            return original_read_task_facts(task_id)

        self.read_run = counted_read_run  # type: ignore[method-assign]
        self.read_run_tolerant = counted_read_run_tolerant  # type: ignore[method-assign]
        self.read_latest_lifecycle = counted_read_latest_lifecycle  # type: ignore[method-assign]
        self.read_task_facts = counted_read_task_facts  # type: ignore[method-assign]

    monkeypatch.setattr(RunFactStore, "__init__", patched_init)

    # 2. 执行生产路径 build
    bundle = _builder(tmp_path).build(
        task="continue",
        context=_context(record.task_id, "continue"),
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )

    # 3. 断言：RunFactStore 读方法总调用次数为 0
    assert read_counts["total"] == 0, (
        f"生产 Context 组装不应读取 run facts，实际读取 {read_counts['total']} 次"
    )

    # 4. 断言：segments 中无 tool_results
    segment_names = {s.name for s in bundle.segments}
    assert "tool_results" not in segment_names, (
        "生产 Context 不应包含 tool_results segment，这是测试路径专用的脚手架产物"
    )
