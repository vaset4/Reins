"""三维进展观察与真实模型反馈验收。

作者：xxx
时间：2026-09-24 23:30:00
"""

from scripts.testing.llm import _from_scripted
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.testing.llm import _ScriptedTurn
from llm.messages import ToolCallPart
from runtime.agent_loop import State
from runtime.progress import ProgressGuard, observe
from runtime.run_facts import RunFactStore
from runtime.stream_events import LifecycleChanged, SegmentPaused
from runtime.tool_operations import ToolOperation
from runtime.types import RunToolsRequest, RunToolsResult
from tests.test_harness_feedback_requests import _run
from tests.test_file_batch_tracking import _runtime
from runtime.session_message_store import SessionMessageStore
from tools.tool_registry import Idempotent, ToolDefinition, ToolRisk


def _observation(
    action="A", output="same", *, paths=None, definition=None, arguments=None, meta=None
):
    """从实际工具结果构造观察；传参：动作、返回、文件范围与声明；返回：生产观察。"""
    definition = definition or ToolDefinition(
        "probe",
        "读取",
        {},
        "agent",
        ToolRisk.SAFE,
        True,
        "logical_scope",
        "builtin",
        idempotent=Idempotent.YES,
    )
    args = arguments or {"value": action}
    call = ToolOperation(
        RunToolsRequest("probe", arguments=args),
        "call",
        "probe",
        args,
        definition_version="v1",
    )
    result = RunToolsResult.ok(action="probe", content=output, meta=meta)
    return observe(call, result, definition, tracked_paths=paths or set())


def test_real_loop_keeps_repeated_evidence_until_model_finishes(tmp_path):
    """重复超过旧阈值后仍由模型结束，原始证据持续可见；传参：目录；返回：无。"""
    turns = tuple(
        _ScriptedTurn(calls=(ToolCallPart(str(index), "probe", {"value": "A"}),))
        for index in range(7)
    )
    loop, context, adapter = _run(
        tmp_path, (*turns, _ScriptedTurn(text="证据已足够，结束核对"))
    )
    assert loop.state is State.DONE and len(adapter.requests) == 8
    assert loop.last_output == "证据已足够，结束核对"
    notices = [
        "NO_PROGRESS_SUSPECTED" in str(request.observations)
        for request in adapter.requests
    ]
    assert notices == [False, False, False, False, True, True, True, True]
    facts = [
        row["evidence"]
        for row in RunFactStore(tmp_path).read_run(context.run_id)
        if row.get("event", "").startswith("progress:")
    ]
    assert [row["repeats"] for row in facts] == list(range(7))
    assert facts[-1]["decision"] == "suspected"


@pytest.mark.parametrize("ending", ["ask_user", "cancel"])
def test_repeated_tools_preserve_user_wait_and_cancellation(tmp_path, ending):
    """跨过旧重复阈值后仍准确等待用户或响应停止；传参：目录和终止方式；返回：无。"""
    loop, context, registry = _runtime(tmp_path, [])
    executions = []

    def probe(_arguments):
        """产生相同证据，并在指定轮接纳用户停止；传参：工具参数；返回：原始结果。"""
        executions.append(True)
        if ending == "cancel" and len(executions) == 7:
            loop.cancellation.cancel("user stop requested")
        return "unchanged-evidence"

    registry.register(
        ToolDefinition(
            "probe",
            "核对证据",
            {},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=probe,
        )
    )
    loop.llm_client = _from_scripted(
        [
            *(
                _ScriptedTurn(calls=(ToolCallPart(str(index), "probe", {}),))
                for index in range(7)
            ),
            _ScriptedTurn(
                calls=(
                    ToolCallPart(
                        "question", "ask_user", {"question": "请提供缺少的资料"}
                    ),
                )
            ),
        ]
    )
    events = list(loop.run_stream(context))
    assert loop.state is State.PAUSED and len(executions) == 7
    terminal = [event for event in events if isinstance(event, LifecycleChanged)][-1]
    expected = "waiting_user" if ending == "ask_user" else "paused"
    reason = "awaiting user input" if ending == "ask_user" else "user stop requested"
    assert terminal.lifecycle == expected and terminal.reason == reason
    assert [event.reason for event in events if isinstance(event, SegmentPaused)] == [
        reason
    ]
    facts = RunFactStore(tmp_path / "data").read_run(context.run_id)
    saved = [row for row in facts if row.get("event") == "run:lifecycle"][-1]
    assert saved["lifecycle"] == expected and saved["reason"] == reason


def test_actual_file_change_and_external_edit_reset_repeated_verification(tmp_path):
    """相同失败输出遇到真实文件改动会重建基线；传参：目录；返回：无。"""
    path = tmp_path / "tested.py"
    path.write_bytes(b"before")
    guard = ProgressGuard({})
    for _ in range(4):
        last = guard.observe_batch(
            (_observation(output="tests failed", paths={str(path)}),)
        )[0]
    assert last.repeats == 3
    path.write_bytes(b"after actual edit")
    new = guard.observe_batch(
        (_observation(output="tests failed", paths={str(path)}),)
    )[0]
    assert new.repeats == 0 and guard.notice is None
    path.unlink()
    missing = guard.observe_batch(
        (_observation(output="tests failed", paths={str(path)}),)
    )[0]
    assert missing.repeats == 0 and missing.files[0]["exists"] is False


def test_stable_alternation_requires_both_outputs_and_no_retroactive_counts():
    """四项稳定交替只建立模式，后续才计数；传参：无；返回：无。"""
    guard = ProgressGuard({})
    counts = [
        guard.observe_batch((_observation(action),))[0].repeats
        for action in "ABABABABA"
    ]
    assert counts == [0, 0, 0, 0, 1, 2, 3, 4, 5]
    changing = ProgressGuard({})
    for index, action in enumerate("ABABABABAB"):
        item = _observation(action, str(index) if action == "B" else "same")
        assert changing.observe_batch((item,))[0].repeats == 0
    assert changing.notice is None


def test_new_parallel_evidence_wins_over_old_results_in_either_order():
    """同批新证据不能被后提交的旧结果掩盖；传参：无；返回：无。"""
    for reverse in (False, True):
        guard = ProgressGuard({})
        old = _observation()
        for _ in range(4):
            guard.observe_batch((old,))
        items = (old, _observation("B", "new evidence"))
        observed = guard.observe_batch(tuple(reversed(items)) if reverse else items)
        assert all(item.repeats == 0 for item in observed) and guard.notice is None


def test_unknown_hash_and_truncation_never_claim_stability(tmp_path, monkeypatch):
    """读哈希失败与无完整哈希的截断结果明确未知；传参：目录和替换器；返回：无。"""
    path = tmp_path / "unavailable"

    def denied(_path):
        """模拟实际IO失败；传参：文件；返回：不返回。"""
        raise PermissionError("hash denied")

    monkeypatch.setattr(Path, "read_bytes", denied)
    unknown = _observation(paths={str(path)})
    assert not unknown.known and unknown.reason == "file_hash_unavailable"
    truncated = _observation(meta={"truncated": True})
    assert not truncated.known
    assert _observation(
        meta={"truncated": True, "content_sha256": "full-version"}
    ).known
    guard = ProgressGuard({})
    assert all(guard.observe_batch((unknown,))[0].repeats == 0 for _ in range(8))


def test_semantic_exemptions_are_trusted_only_for_builtin_declarations():
    """内建轮询与真实可重试失败清episode，MCP标签不获豁免；传参：无；返回：无。"""
    builtin = ToolDefinition(
        "status",
        "状态",
        {},
        "agent",
        ToolRisk.CONFIRM,
        True,
        "logical_scope",
        "builtin",
        idempotent=Idempotent.YES,
        semantics=("polling", "verification"),
    )
    assert _observation(definition=builtin).exempt
    external = replace(builtin, source="mcp")
    assert (
        not _observation(definition=external).exempt
        and external.effective_semantics == ()
    )
    retry = replace(builtin, semantics=("retryable",))
    call = ToolOperation(RunToolsRequest("status"), "call", "status", {})
    failed = RunToolsResult.error_result(
        action="status", error="transport", meta={"retryable": True}
    )
    assert observe(call, failed, retry, tracked_paths=set()).exempt
    assert not observe(
        call, replace(failed, meta={"retryable": False}), retry, tracked_paths=set()
    ).exempt


def test_business_timestamp_nonce_and_full_output_remain_evidence():
    """业务字段和真实输出变化不被计时清理吞掉；传参：无；返回：无。"""
    first = _observation(arguments={"timestamp": 1, "nonce": "a"}, output="one")
    second = _observation(arguments={"timestamp": 2, "nonce": "a"}, output="one")
    assert first.action != second.action
    assert _observation(output="two").result != _observation(output="one").result
    assert (
        _observation(meta={"timestamp": 1}).result
        != _observation(meta={"timestamp": 2}).result
    )
    assert (
        _observation(meta={"elapsed_ms": 1}).result
        == _observation(meta={"elapsed_ms": 2}).result
    )


def test_undispatched_tool_is_not_evidence_of_stable_execution():
    """被拒绝或取消的调用没有实际结果，不能累计卡死；传参：无；返回：无。"""
    guard = ProgressGuard({})
    for _ in range(4):
        guard.observe_batch((_observation(),))
    blocked = _observation(meta={"execution_state": "not_started"})
    assert not blocked.known and blocked.reason == "execution_not_started"
    assert guard.observe_batch((blocked,))[0].repeats == 0 and guard.notice is None


@pytest.mark.parametrize("change", ["external_file", "user", "agent"])
def test_runtime_tracks_file_changes_and_distinguishes_real_user_input(
    tmp_path, change
):
    """真实写入建立跟踪范围，外部修改或用户新要求清计数，自动续跑不清；传参：目录与变化；返回：无。"""
    loop, context, registry = _runtime(tmp_path, [])
    calls = []

    def verify(_arguments):
        """始终返回相同测试结果，在运行中改变真实环境或接纳输入；传参：工具参数；返回：失败正文。"""
        calls.append(True)
        if len(calls) == 5 and change == "external_file":
            (tmp_path / "tested.py").write_bytes(b"externally changed")
        if len(calls) == 4 and change in {"user", "agent"}:
            SessionMessageStore(tmp_path / "data").accept_input(
                context.session_id, "继续核对最新要求", input_source=change
            )
        return "tests still failed"

    registry.register(
        ToolDefinition(
            "verify",
            "验证相同对象",
            {},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=verify,
            semantics=("verification",),
        )
    )
    loop.llm_client = _from_scripted(
        [
            _ScriptedTurn(
                calls=(
                    ToolCallPart(
                        "write",
                        "file_write",
                        {"path": "tested.py", "content": "before"},
                    ),
                )
            ),
            *(
                _ScriptedTurn(calls=(ToolCallPart(str(index), "verify", {}),))
                for index in range(6)
            ),
            _ScriptedTurn(text="已核对"),
        ]
    )
    list(loop.run_stream(context))
    facts = [
        row["evidence"]
        for row in RunFactStore(tmp_path / "data").read_run(context.run_id)
        if row.get("event", "").startswith("progress:")
        and row["evidence"]["tool"] == "verify"
    ]
    assert facts[3]["repeats"] == 3
    assert all(
        item["files"][0]["path"].lower() == str(tmp_path / "tested.py").lower()
        for item in facts
    )
    if change == "agent":
        assert loop.state is State.DONE and facts[-1]["repeats"] == 5
    else:
        assert loop.state is State.DONE and facts[4]["repeats"] == 0
        if change == "external_file":
            assert facts[3]["environment"] != facts[4]["environment"]


@pytest.mark.parametrize(
    "config", [{"remind_after": 0}, {"remind_after": True}, {"remind_after": "3"}]
)
def test_invalid_thresholds_fail_explicitly(config):
    """非法配置不静默恢复默认；传参：配置；返回：无。"""
    with pytest.raises(ValueError, match="positive integer"):
        ProgressGuard({"progress_guard": config})


def test_retired_stop_threshold_fails_explicitly():
    """旧停止配置不被静默采纳或忽略；传参：无；返回：无。"""
    with pytest.raises(ValueError, match="stop_after is retired"):
        ProgressGuard({"progress_guard": {"stop_after": 5}})


def test_disabled_guard_and_new_run_do_not_reuse_old_decisions(monkeypatch):
    """关闭开关不提示，新运行不承接旧隐式次数；传参：替换器；返回：无。"""
    guard = ProgressGuard({})
    guard.start_run("first")
    for _ in range(4):
        guard.observe_batch((_observation(),))
    guard.start_run("second")
    assert guard.observe_batch((_observation(),))[0].repeats == 0
    monkeypatch.setenv("REINS_DISABLE_PROGRESS_GUARD", "1")
    for _ in range(8):
        item = guard.observe_batch((_observation(),))[0]
        assert item.decision == "observed" and guard.model_notice() == ""
