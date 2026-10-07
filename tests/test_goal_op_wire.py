"""goal_op 接线 + durable focus 投影端到端行为测试（本任务核心 AC）。

覆盖 PRD R2-R7/R9 与 AC：
- new/switch 两轮真生效：轮1 更新 durable focus，轮2 RunContext 切到新目标
- switch 按名字（子串）命中、含 done 目标
- fail-closed：not_found / ambiguous（列候选 id）/ new 空 body —— 不落决定，保留原目标
- 审计事实保留：本段 done 后 read_run 仍拿到 focus_decision
- 异常路径不丢：goal_op 持久化后同段抛异常，finally 仍投影 durable focus
- USER/CRON 入口统一回喂 durable focus updated
- 非法 goal kind 由 parser 拒绝，不进入 goal handler

作者：LKX
时间：2026-07-16 00:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_sequence

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from app.repl.turn import render_user_turn
from app.repl.slash_commands import ReplState
from llm.client import RealLLMClient
from runtime.agent_loop import AgentLoop
from runtime.run_facts import RunFactStore
from runtime.run_evidence import RunEvidenceStore
from runtime.ledger import LedgerStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import materialize_messages
from runtime.types import RunContext, Trigger
from tasks.records import TASK_STATUS_DONE
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tests.test_session_runtime import capture_requests


# ---------------------------------------------------------------------------
# 测试环境搭建
# ---------------------------------------------------------------------------


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    (project / ".reins" / "data").mkdir(parents=True, exist_ok=True)
    (project / ".reins" / "workspace").mkdir(parents=True, exist_ok=True)
    return project


def _repl_env(tmp_path: Path, *, initial_goal: str = "初始目标"):
    """搭一套脚本化 REPL 环境，返回驱动 render_user_turn 所需的全部句柄。"""
    project = _project(tmp_path)
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task(initial_goal)
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    state = ReplState(current_task_id=record.task_id)
    console = Console(file=io.StringIO(), force_terminal=False)
    return {
        "console": console,
        "state": state,
        "store": store,
        "registry": registry,
        "project_root": project,
        "data_root": data_root,
        "initial_task_id": record.task_id,
    }


def _drive(env, client, message: str) -> None:
    """用给定 stub LLM 驱动一轮 render_user_turn。"""
    render_user_turn(
        console=env["console"],
        state=env["state"],
        registry=env["registry"],
        llm_client=client,
        project_root=env["project_root"],
        data_root=env["data_root"],
        user_message=message,
    )


def _drive_capture(
    env, monkeypatch: "pytest.MonkeyPatch", client, message: str
) -> list[str]:
    """读取本轮已提交、会交付给模型的工具反馈；传参：入口依赖和输入；返回：反馈正文。"""
    _drive(env, client, message)
    entries = (
        SessionMessageStore(env["data_root"])
        .materialize(env["state"].session_id)
        .entries
    )
    return [
        part.text
        for entry in entries
        if entry.run_id == env["state"].current_run_id
        and entry.message
        and entry.message.kind == "tool_result"
        for part in entry.message.content
    ]


def _focus_decisions(data_root: Path, run_id: str) -> list[dict]:
    facts = RunFactStore(data_root).read_run(run_id)
    return [f for f in facts if f.get("event") == "goal_op:focus_decision"]


def _new_client(*responses: str) -> RealLLMClient:
    return from_test_sequence(list(responses), protocol_mode="text_json")


def test_progress_reply_keeps_long_term_goal_and_focus(tmp_path: Path) -> None:
    """中间答复只结束运行，长期事项与下一轮焦点继续保留。

    传参：tmp_path 为隔离数据目录；返回：无，核对任务、摘要与会话入口
    """
    env = _repl_env(tmp_path, initial_goal="核对两份材料")
    task_id = env["initial_task_id"]
    progress = "已核对第一份材料，等待第二份"
    _drive(
        env,
        _new_client(json.dumps({"type": "final", "content": progress})),
        "先核对第一份",
    )

    task = env["store"].require_task(task_id)
    assert task.status == "active"
    assert task.done_at is None
    assert env["state"].current_task_id == task_id
    assert env["store"].read_summary(task_id) == progress
    _drive(
        env, _new_client('{"type":"final","content":"继续核对第二份"}'), "第二份到了"
    )
    assert env["state"].current_task_id == task_id
    assert env["store"].require_task(task_id).status == "active"


def test_shared_chat_entry_saves_user_input_under_its_run(tmp_path: Path) -> None:
    """REPL 与终端界面共用入口保存一次输入，关联实际启动的运行。

    传参：tmp_path 为隔离数据目录；返回：无，核对 canonical 消息与运行关联
    """
    env = _repl_env(tmp_path)
    _drive(env, _new_client('{"type":"final","content":"已收到"}'), "补充一个约束")
    messages = materialize_messages(env["data_root"], env["state"].session_id)
    users = [message for message in messages if message.kind == "user"]
    assert len(users) == 1
    entries = (
        SessionMessageStore(env["data_root"])
        .materialize(env["state"].session_id)
        .entries
    )
    user_entry = next(
        row for row in entries if row.message and row.message.kind == "user"
    )
    assert user_entry.run_id == env["state"].current_run_id


def test_goal_switch_refreshes_materials_without_overwriting_old_progress(
    tmp_path: Path, monkeypatch
) -> None:
    """运行中切换事项后使用新摘要，并把后续答复保存在新事项。

    传参：tmp_path 为隔离目录；返回：无，核对模型请求材料和两个任务摘要
    """
    env = _repl_env(tmp_path)
    old_id = env["initial_task_id"]
    target = env["store"].create_task("核对新报告")
    env["store"].update_summary(old_id, "旧事项进展标记")
    env["store"].update_summary(target.task_id, "新事项进展标记")
    client = _new_client(
        json.dumps(
            {
                "type": "run_tools",
                "tool": "goal",
                "arguments": {"action": "switch", "goal_ref": target.task_id},
            }
        ),
        '{"type":"final","content":"新报告还需第二份材料"}',
    )
    actual_requests = capture_requests(client, monkeypatch)
    _drive(env, client, "继续核对新报告")
    facts = RunFactStore(env["data_root"]).read_run(env["state"].current_run_id)
    responses = [row for row in facts if row["event"] == "llm:response"]
    evidence = RunEvidenceStore(env["data_root"])
    request = evidence.read_reference(
        responses[-1]["summary"]["evidence"]["model_request"]
    )
    assert request is not None and request["run_id"] == env["state"].current_run_id
    assert "新事项进展标记" in str(actual_requests[-1].instructions)
    assert env["store"].read_summary(old_id) == "旧事项进展标记"
    assert env["store"].read_summary(target.task_id) == "新报告还需第二份材料"
    assert (
        len(
            evidence.list_records(
                session_id=env["state"].session_id,
                run_id=env["state"].current_run_id,
                kind="attempt_request",
            )
        )
        == 2
    )


def test_goal_completion_requires_and_preserves_actual_result_evidence(
    tmp_path: Path,
) -> None:
    """模型明确完成事项时引用已提交的工具结果，任务保留依据与来源。

    传参：tmp_path 为隔离目录；返回：无，核对实际入口、状态与证据
    """
    env = _repl_env(tmp_path, initial_goal="读取两份材料并核对")
    task_id = env["initial_task_id"]
    (env["project_root"] / "result.txt").write_text(
        "两份材料已核对一致", encoding="utf-8"
    )
    _drive(
        env,
        _new_client(
            '{"type":"run_tools","tool":"file_read","arguments":{"path":"result.txt"}}',
            '{"type":"final","content":"核对结果已保存"}',
        ),
        "核对结果",
    )
    messages = materialize_messages(env["data_root"], env["state"].session_id)
    result = next(message for message in messages if message.kind == "tool_result")
    completion = {
        "type": "run_tools",
        "tool": "goal",
        "arguments": {
            "action": "complete",
            "goal_ref": task_id,
            "expected_revision": 1,
            "goal_body": "两份材料一致，已完成核对",
            "evidence": [
                {
                    "kind": "tool_result",
                    "reference": result.call_id,
                    "reason": "保存的核对结果",
                }
            ],
        },
    }
    _drive(
        env,
        _new_client(json.dumps(completion), '{"type":"final","content":"已完成核对"}'),
        "确认核对结论",
    )
    task = env["store"].require_task(task_id)
    assert task.status == "done"
    assert task.status_source == "goal_completion"
    assert task.completion["evidence"][0]["call_id"] == result.call_id
    assert task.completion["session_id"] == env["state"].session_id
    assert task.completion["run_id"] == env["state"].current_run_id


def test_goal_completion_with_missing_evidence_keeps_goal_active(
    tmp_path: Path,
) -> None:
    """没有完成依据时拒绝改任务状态，模型仍可解释剩余工作。

    传参：tmp_path 为隔离目录；返回：无，核对目标没有被一句完成声明关闭
    """
    env = _repl_env(tmp_path)
    task_id = env["initial_task_id"]
    _drive(
        env,
        _new_client(
            json.dumps(
                {
                    "type": "run_tools",
                    "tool": "goal",
                    "arguments": {
                        "action": "complete",
                        "goal_ref": task_id,
                        "expected_revision": 1,
                        "goal_body": "完成了",
                        "evidence": [],
                    },
                }
            ),
            '{"type":"final","content":"还需要核验结果"}',
        ),
        "检查完成情况",
    )
    assert env["store"].require_task(task_id).status == "active"
    assert env["state"].current_task_id == task_id


def test_tool_after_goal_switch_has_consistent_operation_ownership(
    tmp_path: Path,
) -> None:
    """切换后工具执行、消息、账本和恢复记录都归发起目标；传参：隔离根；返回：无。"""
    env = _repl_env(tmp_path)
    target = env["store"].create_task("核对第二个事项")
    (env["project_root"] / "report.txt").write_text("核对结果", encoding="utf-8")
    definition = env["registry"].get("file_read")
    execute = definition.executor
    owners: list[str] = []

    def capture_owner(args: dict[str, object]) -> object:
        """观察真实工具收到的归属后继续执行；传参：工具参数；返回：真实读取结果。"""
        owners.append(str(args["__task_id__"]))
        return execute(args)

    definition.executor = capture_owner
    env["registry"].replace(definition)
    _drive(
        env,
        _new_client(
            json.dumps(
                {
                    "type": "run_tools",
                    "tool": "goal",
                    "arguments": {"action": "switch", "goal_ref": target.task_id},
                }
            ),
            '{"type":"run_tools","tool":"file_read","arguments":{"path":"report.txt"}}',
            '{"type":"final","content":"该事项仍有后续工作"}',
        ),
        "核对第二个事项",
    )
    assert owners == [target.task_id]
    session_id, run_id = env["state"].session_id, env["state"].current_run_id
    entries = SessionMessageStore(env["data_root"]).materialize(session_id).entries
    result = next(
        entry
        for entry in entries
        if entry.message
        and entry.message.kind == "tool_result"
        and entry.message.tool_name == "file_read"
    )
    assert result.task_id == target.task_id
    ledger = LedgerStore(env["data_root"]).read_session_events(session_id)
    assert {row.task_id for row in ledger if row.event.startswith("tool.")} == {
        env["initial_task_id"],
        target.task_id,
    }
    facts = RunFactStore(env["data_root"]).read_run(run_id)
    assert (
        next(row for row in facts if row["event"] == "run:start")["task_id"]
        == env["initial_task_id"]
    )
    requests = [row for row in facts if row["event"] == "llm:request"]
    calls = [
        row
        for row in facts
        if row["event"] in {"tool:request", "tool:response"}
        and row["tool"]["name"] == "file_read"
    ]
    assert {row["operation_task_id"] for row in calls} == {target.task_id}
    assert {row["request_id"] for row in calls} == {requests[1]["request_id"]}
    pending = next(
        row["checkpoint"]["pending_tool_call"]
        for row in facts
        if row["event"] == "checkpoint:saved"
        and (row["checkpoint"].get("pending_tool_call") or {}).get("tool_name")
        == "file_read"
    )
    assert pending["operation_task_id"] == target.task_id
    assert pending["request_id"] == requests[1]["request_id"]
    assert any(
        row.run_id == run_id
        for row in RunFactStore(env["data_root"]).list_runs_for_task(target.task_id)
    )
    attempts = RunEvidenceStore(env["data_root"]).list_records(
        session_id=session_id, run_id=run_id, kind="attempt_request"
    )
    request = next(
        row["payload"]
        for row in attempts
        if row["payload"]["request_id"] == requests[1]["request_id"]
    )
    assert f".reins/workspace/{target.task_id}/outputs" in str(request)


# ---------------------------------------------------------------------------
# 核心：两轮真生效
# ---------------------------------------------------------------------------


def test_repl_two_turn_new_takes_effect_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """轮1 模型发 goal_op(new) → 回流后 current_task_id = 新建 id；
    轮2 render_user_turn 构造的 RunContext.task_id/storage_task_id = 新 id（跨轮真切换）。"""
    env = _repl_env(tmp_path)
    initial = env["initial_task_id"]

    # spy 捕获每轮 loop 实际收到的 RunContext
    captured: list[RunContext] = []
    orig = AgentLoop.run_stream

    def spy(self, context):
        captured.append(context)
        yield from orig(self, context)

    monkeypatch.setattr(AgentLoop, "run_stream", spy)

    # 轮1：新建目标"写季度报告"
    _drive(
        env,
        _new_client(
            '{"type":"run_tools","tool":"goal","arguments":{"action":"new","goal_body":"写季度报告"}}',
            '{"type":"final","content":"ok"}',
        ),
        "帮我写季度报告",
    )
    new_id = env["state"].current_task_id
    assert new_id is not None
    assert new_id != initial
    assert env["store"].load_task(new_id).goal == "写季度报告"

    # 轮2：普通 final，断言这轮 RunContext 已切到新目标
    _drive(env, _new_client('{"type":"final","content":"done"}'), "继续")
    turn2_context = captured[-1]
    assert turn2_context.task_id == new_id
    assert turn2_context.storage_task_id == new_id


def test_repl_two_turn_switch_by_id_takes_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """轮1 switch 到既有目标（按 id，瀑布①）→ 轮2 RunContext 切到该目标。"""
    env = _repl_env(tmp_path)
    other = env["store"].create_task("重构登录模块")

    captured: list[RunContext] = []
    orig = AgentLoop.run_stream

    def spy(self, context):
        captured.append(context)
        yield from orig(self, context)

    monkeypatch.setattr(AgentLoop, "run_stream", spy)

    _drive(
        env,
        _new_client(
            json.dumps(
                {
                    "type": "run_tools",
                    "tool": "goal",
                    "arguments": {"action": "switch", "goal_ref": other.task_id},
                }
            ),
            '{"type":"final","content":"ok"}',
        ),
        "切回登录那个",
    )
    assert env["state"].current_task_id == other.task_id

    _drive(env, _new_client('{"type":"final","content":"done"}'), "继续")
    assert captured[-1].task_id == other.task_id


def test_switch_by_name_substring_takes_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """switch 按名字子串（瀑布③）命中唯一 → 下一轮切到该目标；含 done 目标（对齐 slash）。"""
    env = _repl_env(tmp_path)
    target = env["store"].create_task("帮我规划本周买菜清单")
    env["store"].update_task_status(target.task_id, TASK_STATUS_DONE)

    captured: list[RunContext] = []
    orig = AgentLoop.run_stream

    def spy(self, context):
        captured.append(context)
        yield from orig(self, context)

    monkeypatch.setattr(AgentLoop, "run_stream", spy)

    _drive(
        env,
        _new_client(
            '{"type":"run_tools","tool":"goal","arguments":{"action":"switch","goal_ref":"买菜"}}',
            '{"type":"final","content":"ok"}',
        ),
        "切回上周那个买菜的",
    )
    assert env["state"].current_task_id == target.task_id

    _drive(env, _new_client('{"type":"final","content":"done"}'), "继续")
    assert captured[-1].task_id == target.task_id


# ---------------------------------------------------------------------------
# 审计事实保留 + 异常路径 durable focus 不丢
# ---------------------------------------------------------------------------


def test_decision_survives_terminal_cleanup(tmp_path: Path) -> None:
    """本段跑到 done 后，read_run 仍能拿到 goal_op:focus_decision（append-only 未被终态抹掉）。"""
    env = _repl_env(tmp_path)
    _drive(
        env,
        _new_client(
            '{"type":"run_tools","tool":"goal","arguments":{"action":"new","goal_body":"买菜"}}',
            '{"type":"final","content":"ok"}',
        ),
        "顺便记个买菜",
    )
    run_id = env["state"].current_run_id
    decisions = _focus_decisions(env["data_root"], run_id)
    assert len(decisions) == 1
    assert decisions[0]["target_task_id"] == env["state"].current_task_id


def test_decision_applied_on_exception_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """goal_op 持久化后同段抛异常，finally 仍投影 durable focus。"""
    env = _repl_env(tmp_path)
    initial = env["initial_task_id"]

    orig = AgentLoop.run_stream

    def boom(self, context):
        # 先真跑（goal_op 落决定），消费完后抛异常，模拟同段稍后失败
        yield from orig(self, context)
        raise RuntimeError("simulated mid-segment failure")

    monkeypatch.setattr(AgentLoop, "run_stream", boom)

    # render_user_turn 的 except 吞掉异常并打印，finally 回流不受影响
    _drive(
        env,
        _new_client(
            '{"type":"run_tools","tool":"goal","arguments":{"action":"new","goal_body":"意外中断的目标"}}',
            '{"type":"final","content":"ok"}',
        ),
        "新目标",
    )
    assert env["state"].current_task_id is not None
    assert env["state"].current_task_id != initial
    assert env["store"].load_task(env["state"].current_task_id).goal == "意外中断的目标"


# ---------------------------------------------------------------------------
# fail-closed：不落决定，原目标仍可继续
# ---------------------------------------------------------------------------


def test_switch_not_found_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """switch 匹配 0 个 → 回喂错误、无决定，保留原焦点。"""
    env = _repl_env(tmp_path)
    contents = _drive_capture(
        env,
        monkeypatch,
        _new_client(
            '{"type":"run_tools","tool":"goal","arguments":{"action":"switch","goal_ref":"根本不存在的目标xyz"}}',
            '{"type":"final","content":"ok"}',
        ),
        "切到不存在的",
    )
    assert env["state"].current_task_id == env["initial_task_id"]
    run_id = env["state"].current_run_id
    assert _focus_decisions(env["data_root"], run_id) == []
    assert any("no goal matches" in c for c in contents)


def test_switch_ambiguous_fail_closed_lists_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """switch 匹配多个 → 回喂候选、无决定，保留原焦点。"""
    env = _repl_env(tmp_path)
    a = env["store"].create_task("写季度报告")
    b = env["store"].create_task("审阅年度报告")
    contents = _drive_capture(
        env,
        monkeypatch,
        _new_client(
            '{"type":"run_tools","tool":"goal","arguments":{"action":"switch","goal_ref":"报告"}}',
            '{"type":"final","content":"ok"}',
        ),
        "切回报告",
    )
    assert env["state"].current_task_id == env["initial_task_id"]
    run_id = env["state"].current_run_id
    assert _focus_decisions(env["data_root"], run_id) == []
    joined = "\n".join(contents)
    assert "matches 2 goals" in joined
    # 候选 id 端给模型，供下一轮走瀑布①精确命中
    assert a.task_id in joined
    assert b.task_id in joined


def test_new_empty_body_fail_closed_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """new 空 body → 回喂错误、不建任务、不落决定，保留原焦点。"""
    env = _repl_env(tmp_path)
    before = len(env["store"].list_tasks())
    contents = _drive_capture(
        env,
        monkeypatch,
        _new_client(
            '{"type":"run_tools","tool":"goal","arguments":{"action":"new","goal_body":"   "}}',
            '{"type":"final","content":"ok"}',
        ),
        "新建",
    )
    assert env["state"].current_task_id == env["initial_task_id"]
    assert len(env["store"].list_tasks()) == before
    run_id = env["state"].current_run_id
    assert _focus_decisions(env["data_root"], run_id) == []
    assert any("goal body required" in c for c in contents)


# ---------------------------------------------------------------------------
# 入口统一读取 durable focus
# ---------------------------------------------------------------------------


def test_feedback_wording_by_trigger_cron(tmp_path: Path) -> None:
    """CRON 命中 goal_op 后也回喂 durable focus 已更新。"""
    project = _project(tmp_path)
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("cron 任务")
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    from runtime.lease import from_trigger
    from runtime.default_capabilities import build_local_agent_capabilities

    lease = from_trigger(
        "cron",
        task_id=record.task_id,
        capabilities=build_local_agent_capabilities(project, data_root),
    )
    loop = AgentLoop(
        data_root,
        llm_client=_new_client(
            '{"type":"run_tools","tool":"goal","arguments":{"action":"new","goal_body":"cron 新目标"}}',
            '{"type":"final","content":"ok"}',
        ),
        tool_registry=registry,
    )
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.CRON,
        payload={"message": "cron"},
        capability_lease=lease,
        segment_id=f"cron-{record.task_id}",
    )
    # 目标工具的持久结果是后续模型读取的反馈
    list(loop.run_stream(context))
    # CRON 入口也由 durable session state 承接 focus，成功反馈不再承诺入口切换
    facts = RunFactStore(data_root).read_run(context.run_id)
    decisions = [f for f in facts if f.get("event") == "goal_op:focus_decision"]
    assert len(decisions) == 1
    results = [
        message
        for message in materialize_messages(data_root, context.session_id)
        if message.kind == "tool_result"
    ]
    assert context.focus_task_id in str(results)


def test_feedback_wording_user_durable_focus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REPL/USER 入口 → 回喂明确说明 durable focus 已更新。"""
    env = _repl_env(tmp_path)
    contents = _drive_capture(
        env,
        monkeypatch,
        _new_client(
            '{"type":"run_tools","tool":"goal","arguments":{"action":"new","goal_body":"用户新目标"}}',
            '{"type":"final","content":"ok"}',
        ),
        "新目标",
    )
    assert env["state"].current_task_id in "\n".join(contents)
