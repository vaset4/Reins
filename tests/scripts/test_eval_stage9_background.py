"""【阶段九评估】【后台入口机制】仅用本地模型替身验证真实调度、持久原件和总计量。

作者：xxx
时间：2026-10-01 20:30:00
"""

import json
from contextlib import closing, nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from context.history_segments import HistorySegment
from llm.messages import ToolCallPart, model_visible_text
from llm.types import ModelAttemptStarted, ModelOutputDelta
from memory.store import MemoryStore
from scripts.eval_stage9_background import (
    EvaluationClients,
    StreamingGate,
    evaluate_history,
    evaluate_knowledge,
    history_adoption,
    main,
)
from scripts.eval_stage9_meter import call_metrics, save
from scripts.long_context_cases import CASES
from scripts.testing.llm import (
    _ScriptedOptions,
    _ScriptedTurn,
    _from_scripted,
    _test_target,
)
from tests.scripts.test_eval_stage9 import budget_at
from tests.test_semantic_compaction import _response, _summary_delta


def configuration(root, scenario, *, attempts=28):
    """创建单独本地实验配置；参数：临时目录/场景/额度；返回：独立预算配置。"""
    budget = budget_at(root, calls=attempts, attempts=attempts, output=attempts * 16384)
    return {
        "sample_output": str(root / scenario),
        "budget": str(budget.path),
        "profile": "local-test",
        "per_call_output_tokens": 16384,
        "window": 65536,
        "scenario": scenario,
    }


def scripted(turns):
    """创建有明确输出上限的本地Provider；参数：脚本轮次；返回：真实生产客户端。"""
    target = replace(_test_target(65536), max_output_tokens=16384)
    return _from_scripted(turns, options=_ScriptedOptions(resolved_target=target))


def test_dry_plan_does_not_create_output_or_dispatch(tmp_path, capsys):
    """默认CLI只披露独立拟议预算和未覆盖范围；参数：临时目录/输出捕获；返回：无。"""
    output = tmp_path / "background-evaluation"
    assert main(["--output", str(output), "--profile", "explicit-profile"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["profile"] == "explicit-profile" and plan["max_attempts"] == 28
    assert plan["max_output_tokens"] == 458752 and plan["scenarios"] == [
        "history",
        "knowledge",
    ]
    assert plan["window"] == 65536 and plan["per_call_output_tokens"] == 16384
    assert "natural window pressure" in plan["not_covered"]
    assert not output.exists()
    with pytest.raises(SystemExit):
        main(["--output", str(output)])
    with pytest.raises(ValueError, match="requires explicit --execute"):
        main(["--worker-config", str(tmp_path / "must-not-read.json")])


def test_history_scheduler_runs_while_foreground_continues_then_adopts(
    tmp_path, monkeypatch
):
    """本地替身只替换Provider，旧源/实际调度/前台新输入/最终采用均走生产；参数：隔离目录；返回：无。"""
    config = configuration(tmp_path, "history")
    requests = []

    def create(role, _options):
        """按角色提供确定性模型响应；参数：角色/固定选型；返回：本地客户端。"""
        client = scripted([])

        def respond(request, **_kwargs):
            """返回机制测试的Provider事件，不作为真实语义证据；参数：实际请求；返回：事件流。"""
            requests.append((role, request))
            body = "\n".join(
                model_visible_text(message) for message in request.messages
            )
            answer = (
                _summary_delta(body)
                if role == "background-summary"
                else json.dumps(dict(CASES[0].expected))
            )
            return _response(answer, len(requests))

        monkeypatch.setattr(
            client._adapter_registry.require("scripted_test"), "stream", respond
        )
        return client

    clients = EvaluationClients(config, create)
    result = evaluate_history(config, clients)
    assert result["passed"], result
    assert result["checks"]["foreground_completed_before_gate_release"]
    assert result["checks"]["published_version_adopted"]
    assert (
        result["controlled_pause_seconds"]
        >= result["during"]["foreground_wait_seconds"]
    )
    assert result["gate_evidence"]["event"] == "ModelOutputDelta"
    assert result["gate_evidence"]["observed_characters"] > 0
    assert result["during"]["input_message_id"] != result["after"]["input_message_id"]
    assert {role for role, _ in requests} == {"foreground", "background-summary"}
    positions = [
        index for index, (role, _) in enumerate(requests) if role == "foreground"
    ]
    first_background = next(
        index
        for index, (role, _) in enumerate(requests)
        if role == "background-summary"
    )
    assert first_background < positions[1]
    metrics = call_metrics(tmp_path / "history/background-summary/calls")
    assert metrics["by_phase"]["summary_generation"]["started_attempts"] >= 1
    assert metrics["by_phase"]["summary_audit"]["started_attempts"] >= 1
    budget = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert budget["attempts"] == len(requests)
    assert all(request.max_output_tokens == 16384 for _role, request in requests)
    assert all(result["adoption"]["checks"].values())
    from runtime.session_compaction import SessionCompactionStore
    from runtime.session_message_store import SessionMessageStore

    root = tmp_path / "history/data"
    owner = SessionMessageStore(root)
    summary = SessionCompactionStore(owner).current(owner.materialize("conditions"))
    call = sorted((tmp_path / "history/foreground/calls").glob("*/call.json"))[-1]
    for attempt_path in call.parent.glob("*-started.json"):
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        original_payload = attempt["request"]
        attempt["request"] = {
            "metadata": {"summary_id": summary.summary_id},
            "messages": [
                {
                    "role": "system",
                    "content": "body omitted",
                    "metadata": original_payload,
                }
            ],
        }
        attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
    assert not history_adoption(root, call, summary)["passed"]


def knowledge_creator(root, *, save=True, finish_correction=True):
    """提供本地模型并由生产工具写记忆；参数：数据根/保存及更正完成开关；返回：工厂。"""

    def create(role, _options):
        """根据测试阶段构造真实工具调用；参数：角色/选型；返回：脚本Provider客户端。"""
        if role.startswith("foreground"):
            return scripted(
                [
                    _ScriptedTurn(
                        text='{"port":9000}'
                        if role.endswith("verification")
                        else "收到"
                    )
                ]
            )
        if not save:
            return scripted([_ScriptedTurn(text="维护完成")])
        calls = [ToolCallPart("read", "knowledge_read", {"action": "messages"})]
        if role == "worker-new":
            calls.append(
                ToolCallPart(
                    "save",
                    "memory_manage",
                    {
                        "action": "create",
                        "type": "rule",
                        "kind": "fact",
                        "memory_scope": "project",
                        "subject": "服务",
                        "fact_key": "端口",
                        "content": "本项目服务端口固定为8000",
                        "source_mode": "origin_inputs",
                    },
                )
            )
        elif role == "worker-correction":
            with closing(MemoryStore(root)) as memories:
                old = memories.list_memories()[0]
            calls.append(
                ToolCallPart(
                    "revise",
                    "memory_manage",
                    {
                        "action": "revise",
                        "memory_id": old.memory_id,
                        "expected_version": old.version,
                        "content": "本项目服务端口固定为9000",
                        "reason": "原用户明确更正",
                        "source_mode": "origin_inputs",
                    },
                )
            )
        if role != "worker-correction" or finish_correction:
            calls.append(
                ToolCallPart(
                    "finish",
                    "knowledge_finish",
                    {
                        "outcome": "no_op" if role == "worker-no_op" else "completed",
                        "reason": "已核对全部原来源",
                    },
                )
            )
        return scripted(
            [_ScriptedTurn(calls=tuple(calls)), _ScriptedTurn(text="处理结束")]
        )

    return create


def test_knowledge_evaluation_uses_real_sources_versions_and_noop(tmp_path):
    """自动提炼/无知识no-op/更正与新会话使用均核对真实原件；参数：隔离目录；返回：无。"""
    config = configuration(tmp_path, "knowledge")
    clients = EvaluationClients(config, knowledge_creator(tmp_path / "knowledge/data"))
    result = evaluate_knowledge(config, clients)
    assert result["passed"], result
    assert [stage["work"]["state"] for stage in result["stages"]] == [
        "completed",
        "no_op",
        "completed",
    ]
    assert result["stages"][-1]["checks"]["old_version_not_active"]
    assert result["stages"][-1]["checks"]["old_originals_readable"]
    assert result["semantic"]["passed"]
    budget = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert budget["attempts"] >= 10 and budget["attempts"] <= 28
    assert (
        len(list((tmp_path / "knowledge").glob("*/calls/*/*-started.json")))
        == budget["attempts"]
    )


def test_knowledge_plain_final_cannot_fake_success(tmp_path):
    """后台只说维护完成而未读写原件时，验收必须失败；参数：隔离目录；返回：无。"""
    config = configuration(tmp_path, "knowledge")
    clients = EvaluationClients(
        config, knowledge_creator(tmp_path / "knowledge/data", save=False)
    )
    result = evaluate_knowledge(config, clients)
    assert not result["passed"] and result["stages"][0]["work"]["state"] == "failed"
    assert not result["stages"][0]["after"]
    assert not result["stages"][0]["checks"]["actual_commits"]


def test_knowledge_failed_correction_retains_actual_commits(tmp_path):
    """更正真实提交后额度耗尽，展示提交但不能通过整体验收；参数：隔离目录；返回：无。"""
    # 1. 【阶段九评估】【失败对账】前两阶段各三次，更正前台和修订各一次，后续模型请求超出八次额度
    allowed_attempts = 8
    config = configuration(tmp_path, "knowledge", attempts=allowed_attempts)
    clients = EvaluationClients(
        config, knowledge_creator(tmp_path / "knowledge/data", finish_correction=False)
    )
    result = evaluate_knowledge(config, clients)
    assert not result["passed"]
    assert [stage["stage"] for stage in result["stages"]] == [
        "new",
        "no_op",
        "correction",
    ]
    assert all(stage["passed"] for stage in result["stages"][:2])
    correction = result["stages"][-1]
    assert correction["work"]["state"] == "failed"
    assert "allowance exhausted before dispatch" in correction["work"]["error"]
    assert not correction["passed"] and not correction["checks"]["completed"]
    assert correction["checks"]["actual_commits"]
    assert correction["checks"]["current_rule_has_real_source"]
    assert (
        correction["checks"]["old_version_not_active"]
        and correction["checks"]["old_originals_readable"]
    )
    revised = correction["after"][0]
    assert len(correction["work"]["commits"]) == 1
    commit = correction["work"]["commits"][0]
    assert commit["operation_id"] == revised["change_id"]
    assert (commit["memory_id"], commit["version"]) == (
        revised["memory_id"],
        revised["version"],
    )
    saved = json.loads(
        (tmp_path / "knowledge/correction-result.json").read_text(encoding="utf-8")
    )
    assert (
        saved["work"]["commits"] == correction["work"]["commits"]
        and not saved["passed"]
    )
    budget = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert budget["attempts"] == allowed_attempts
    assert (
        len(list((tmp_path / "knowledge").glob("*/calls/*/*-started.json")))
        == allowed_attempts
    )


def test_history_budget_exhaustion_stops_auxiliary_before_dispatch(tmp_path):
    """只授权一次尝试时前台完成后不得再发送后台请求；参数：隔离目录；返回：无。"""
    config = configuration(tmp_path, "history", attempts=1)
    clients = EvaluationClients(
        config,
        lambda _role, _options: scripted(
            [_ScriptedTurn(text=json.dumps(dict(CASES[0].expected)))]
        ),
    )
    with pytest.raises(RuntimeError, match="before its first provider output delta"):
        evaluate_history(config, clients)
    budget = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert budget["attempts"] == 1 and budget["logical_calls"] == 1
    assert len(list((tmp_path / "history").glob("*/calls/*/*-started.json"))) == 1


def test_gate_ignores_local_attempt_notice_until_real_output_delta():
    """本地派发通知不会开门，只有实际非空供应商增量触发；参数：无；返回：无。"""

    class Client:
        """为事件边界回归提供最短模型流，不冒充真实语义结果。"""

        def plan_stream(self, _task, _context):
            """先通知派发，再交回空/非空供应商增量；参数：请求；返回：事件流。"""
            yield ModelAttemptStarted(1, 3)
            yield ModelOutputDelta("text", "")
            yield ModelOutputDelta("thinking", "供应商已经返回思考内容")

    gate = StreamingGate(Client())
    stream = gate.plan_stream("check")
    assert isinstance(next(stream), ModelAttemptStarted)
    assert not gate.entered.is_set() and gate.evidence is None
    assert next(stream).text == ""
    assert not gate.entered.is_set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, stream)
        try:
            assert gate.entered.wait(timeout=5)
            assert not future.done()
            assert gate.evidence == {
                "event": "ModelOutputDelta",
                "channel": "thinking",
                "observed_characters": len("供应商已经返回思考内容"),
            }
        finally:
            gate.release.set()
        assert future.result(timeout=5).channel == "thinking"
    stream.close()


def test_gate_timeout_closes_source_and_preserves_independent_timing(
    tmp_path, monkeypatch
):
    """gate超时仍关闭底层流并保存单独时长，不依赖案例成功返回；参数：隔离目录/超时注入；返回：无。"""
    closed = []

    class Client:
        """记录源流资源释放的本地边界。"""

        def plan_stream(self, _task, _context):
            """返回真实增量形状并在关闭时释放资源；参数：请求；返回：事件流。"""
            try:
                yield ModelOutputDelta("text", "provider output")
            finally:
                closed.append(True)

    monkeypatch.setattr("scripts.eval_stage9_background.GATE_TIMEOUT_SECONDS", 0)
    gate = StreamingGate(Client(), evidence_path=tmp_path / "gate.json")
    with pytest.raises(TimeoutError, match="did not release"):
        next(gate.plan_stream("check"))
    assert closed == [True]
    evidence = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert evidence["trigger"]["event"] == "ModelOutputDelta"
    assert not evidence["released"] and evidence["controlled_pause_seconds"] >= 0


@pytest.mark.parametrize("level", ["P1", "P2", "P3", "P4"])
def test_adoption_checks_selected_level_and_persisted_representation(
    tmp_path, monkeypatch, level
):
    """材料身份版本不随选档变，采用必须核对实际档位与正文；参数：隔离原件/档位；返回：无。"""
    segment = HistorySegment(
        "segment", "历史", ("message",), "完整", "中等", "简短", "索引"
    )
    summary = SimpleNamespace(
        summary_id="summary",
        session_id="session",
        content=SimpleNamespace(segments=(segment,)),
    )
    material = {
        "source": "history",
        "identity": "segment:summary:segment",
        "version": "summary/P2",
        "representation": level,
    }
    persisted = {**material, "text": segment.render(level)}
    baseline = {
        "adopted_request_id": "request",
        "baseline_id": "baseline",
        "delta_id": "delta",
        "baseline": [persisted],
        "delta": [],
    }
    snapshot = SimpleNamespace(list=lambda *_args, **_kwargs: [baseline])
    monkeypatch.setattr(
        "runtime.persistence.RuntimeStore",
        lambda _root: SimpleNamespace(snapshot=lambda: nullcontext(snapshot)),
    )
    call = tmp_path / "call.json"
    save(
        call,
        {
            "request_id": "request",
            "composition": {
                "context_baseline": {
                    "baseline_id": "baseline",
                    "delta_id": "delta",
                    "materials": [material],
                }
            },
        },
    )
    save(
        tmp_path / "attempt-started.json",
        {
            "request_id": "request",
            "attempt_id": "attempt",
            "request": {
                "messages": [{"role": "system", "content": segment.render(level)}]
            },
        },
    )
    assert history_adoption(tmp_path, call, summary)["passed"]
    # 1. 【阶段九评估】【实际采用】相同身份和正文不能掩盖已持久采用的档位不一致
    baseline["baseline"] = [
        {**persisted, "representation": "P3" if level != "P3" else "P4"}
    ]
    assert not history_adoption(tmp_path, call, summary)["passed"]
