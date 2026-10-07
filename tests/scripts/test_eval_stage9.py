"""【阶段九评估】【机制验证】验证源码隔离、费用拒绝和正式四级历史采用，无付费请求。

作者：xxx
时间：2026-10-01 14:30:00
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from llm.messages import model_visible_text
from llm.types import ModelAttemptEvent
from scripts.eval_stage9 import (
    BASELINE_ROOT,
    CONTROLLED_WINDOW,
    DEFAULT_OUTPUT_PER_CALL,
    PROJECT_ROOT,
    main,
    parse_args,
    experiment_plan,
)
from scripts.eval_stage9_meter import (
    EvaluationBudget,
    EvaluationLimitExceeded,
    MeteredClient,
    call_metrics,
    save,
)
from scripts.eval_stage9_worker import evaluate
from scripts.long_context_cases import CASES
from scripts.testing.llm import (
    _ScriptedOptions,
    _from_scripted,
    _test_target,
    from_test_turns,
)
from tests.test_semantic_compaction import _response, _summary_delta
from tools.tool_registry import ToolRegistry


def budget_at(root, *, calls=16, attempts=16, output=262144):
    """建立全组共享额度；参数：隔离位置及上限；返回：总账。"""
    path = root / "budget.json"
    save(
        path,
        {
            "logical_calls": 0,
            "attempts": 0,
            "max_attempts": attempts,
            "output_tokens": 0,
            "max_logical_calls": calls,
            "max_output_tokens": output,
            "limit_exceeded": None,
        },
    )
    return EvaluationBudget(path)


def test_default_plan_never_starts_model_or_creates_output(tmp_path, capsys):
    """默认命令只给四样本与明确额度；参数：隔离目录与输出捕获；返回：无。"""
    output = tmp_path / "evaluation"
    assert main(["--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["samples"]) == 4 and plan["max_logical_calls"] == 16
    assert plan["max_attempts"] == 16
    assert plan["max_output_tokens"] == 262144
    assert plan["window"] == 65536 and not output.exists()
    args = parse_args(["--output", str(output), "--mode", "natural"])
    assert experiment_plan(args)["window"] == "profile_unchanged"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output",
                str(output),
                "--window",
                "32000",
                "--per-call-output-tokens",
                "16384",
            ]
        )
    # 1. 修复回测可只派发新版更正案例，避免意外重跑旧版消耗已分配额度
    assert (
        main(
            [
                "--output",
                str(output),
                "--cases",
                "correction",
                "--variants",
                "candidate",
                "--max-logical-calls",
                "4",
                "--max-attempts",
                "4",
                "--max-output-tokens",
                "65536",
            ]
        )
        == 0
    )
    repair_plan = json.loads(capsys.readouterr().out)
    assert repair_plan["samples"] == [
        {"case": "correction", "repeat": 1, "variant": "candidate"}
    ]
    assert (
        repair_plan["max_attempts"] == 4 and repair_plan["max_output_tokens"] == 65536
    )
    assert not output.exists()
    with pytest.raises(SystemExit):
        parse_args(["--output", str(output), "--variants", "candidate", "candidate"])


def test_retry_output_allowance_rejects_before_recording_unsent_attempt(tmp_path):
    """重试也占整次输出额度，unknown不释放额度；参数：隔离目录；返回：无。"""
    budget = budget_at(tmp_path, output=7)
    client = MeteredClient(None, output=tmp_path / "calls", budget=budget)
    recorder = client._recorder(tmp_path / "attempts", None)
    first = ModelAttemptEvent(
        "started",
        "request",
        "first",
        1,
        "fixture",
        "fixture",
        "now",
        request={"max_completion_tokens": 4},
    )
    recorder(first)
    recorder(
        ModelAttemptEvent(
            "finished", "request", "first", 1, "fixture", "fixture", "now"
        )
    )
    with pytest.raises(EvaluationLimitExceeded):
        recorder(
            ModelAttemptEvent(
                "started",
                "request",
                "retry",
                2,
                "fixture",
                "fixture",
                "now",
                request={"max_completion_tokens": 4},
            )
        )
    assert not (tmp_path / "attempts/retry-started.json").exists()
    final = json.loads(
        (tmp_path / "attempts/first-finished.json").read_text(encoding="utf-8")
    )
    assert final["usage"]["input_tokens"]["value"] is None
    assert json.loads(budget.path.read_text())["output_tokens"] == 4


def test_attempt_limit_counts_retries_even_when_output_ceiling_shrinks(tmp_path):
    """输出变小也不能放行第17次真实尝试；参数：隔离目录；返回：无。"""
    budget = budget_at(tmp_path)
    client = MeteredClient(None, output=tmp_path / "calls", budget=budget)
    recorder = client._recorder(tmp_path / "attempts", None)
    for number in range(16):
        recorder(
            ModelAttemptEvent(
                "started",
                "same-logical-request",
                f"retry-{number}",
                number + 1,
                "fixture",
                "fixture",
                "now",
                request={"max_completion_tokens": 16 - number},
            )
        )
    before = json.loads(budget.path.read_text())
    with pytest.raises(EvaluationLimitExceeded, match="attempts"):
        recorder(
            ModelAttemptEvent(
                "started",
                "same-logical-request",
                "retry-16",
                17,
                "fixture",
                "fixture",
                "now",
                request={"max_completion_tokens": 1},
            )
        )
    after = json.loads(budget.path.read_text())
    assert after["attempts"] == 16 and after["output_tokens"] == before["output_tokens"]
    assert len(list((tmp_path / "attempts").glob("*-started.json"))) == 16


def test_logical_limit_stops_real_client_before_second_dispatch(tmp_path):
    """总调用上限在真实客户端之前生效；参数：隔离目录；返回：无。"""
    client = MeteredClient(
        from_test_turns(["one", "two"]),
        output=tmp_path / "calls",
        budget=budget_at(tmp_path, calls=1),
    )
    assert (
        client.plan("continue", {"tool_registry": ToolRegistry()}).final_output == "one"
    )
    with pytest.raises(EvaluationLimitExceeded):
        client.plan("continue", {"tool_registry": ToolRegistry()})
    assert len(list((tmp_path / "calls").glob("*/call.json"))) == 1


def test_candidate_exercises_formal_four_level_compaction_then_main_request(
    tmp_path, monkeypatch
):
    """受控模型经过正式主循环生成/核对/发布再采用；参数：隔离目录；返回：无。"""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    target = replace(
        _test_target(CONTROLLED_WINDOW), max_output_tokens=DEFAULT_OUTPUT_PER_CALL
    )
    client = _from_scripted([], options=_ScriptedOptions(resolved_target=target))
    requests = []

    def respond(request, **kwargs):
        """只替换供应商事件，保留主循环与全部上下文流程；参数：真实请求；返回：合成事件。"""
        requests.append(request)
        body = "\n".join(model_visible_text(message) for message in request.messages)
        answer = (
            _summary_delta(body)
            if "整理下面的会话资料" in body
            else json.dumps(dict(CASES[0].expected))
        )
        return _response(answer, len(requests))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", respond
    )
    output = tmp_path / "sample"
    measured = MeteredClient(
        client, output=output / "calls", budget=budget_at(tmp_path)
    )
    result = evaluate(
        {
            "sample_output": str(output),
            "case": "conditions",
            "rounds": 2,
            "mode": "controlled",
            "variant": "candidate",
        },
        measured,
    )
    assert result["passed"], result
    assert result["rounds"][0]["four_level_summary_count"] == 1
    assert result["rounds"][1]["four_level_summary_count"] == 2
    from runtime.session_message_store import SessionMessageStore

    owner = SessionMessageStore(output / "data")
    inbound = [
        entry for entry in owner.read_entries("conditions") if entry.type == "inbound"
    ]
    ids = {
        row[turn]["input_message_id"]
        for row in result["rounds"]
        for turn in ("turn", "compaction_turn")
    }
    assert len(inbound) == len(ids) == 4
    assert {entry.entry_id for entry in inbound} == ids
    assert all(entry.input_source == "user" for entry in inbound)
    originals = {
        message.message_id: model_visible_text(message)
        for message in owner.materialize("conditions").messages
    }
    assert all(
        originals[entry.entry_id] == model_visible_text(entry.message)
        for entry in inbound
    )
    assert (
        sum(model_visible_text(entry.message) == "/compact" for entry in inbound) == 2
    )
    calls = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (output / "calls").glob("*/call.json")
    ]
    assert {row["purpose"] for row in calls} >= {
        "main",
        "compaction",
        "compaction_confirmation",
    }
    main_calls = [row for row in calls if row["purpose"] == "main"]
    foreground = [
        row for row in calls if row["purpose"] in {"main", "compaction_confirmation"}
    ]
    assert {row["input_message_id"] for row in foreground} == ids
    assert all(
        row["input_message_id"] is None
        for row in calls
        if row["purpose"] == "compaction"
    )
    assert ids <= {
        message.message_id for request in requests for message in request.messages
    }
    assert main_calls[-1]["composition"]["context_baseline"]["materials"]
    metrics = call_metrics(output / "calls")
    assert metrics["cost_amount"] is None
    assert metrics["by_phase"]["summary_generation"]["finished_attempts"] >= 2
    assert metrics["by_phase"]["summary_audit"]["finished_attempts"] >= 2
    assert metrics["by_phase"]["compaction_confirmation"]["finished_attempts"] == 2
    assert (
        metrics["by_phase"]["main"]["usage"]["cache_read_input_tokens"][
            "unknown_attempts"
        ]
        >= 1
    )
    assert all(request.max_output_tokens == 16384 for request in requests)


def test_frozen_baseline_imports_complete_old_production_tree():
    """独立解释器导入整份旧主循环，无两个模块拼接；参数：无；返回：无。"""
    script = "\n".join(
        [
            "import pathlib, sys",
            f"root = pathlib.Path({str(BASELINE_ROOT)!r}).resolve()",
            "sys.path.insert(0, str(root))",
            "import runtime.agent_loop, context.production_builder, llm.client",
            "packages = {'runtime', 'context', 'llm', 'app', 'tools', 'memory'}",
            "paths = [pathlib.Path(mod.__file__).resolve() for name, mod in sys.modules.items() if name.split('.')[0] in packages and getattr(mod, '__file__', None)]",
            "assert len(paths) > 50 and all(path.is_relative_to(root) for path in paths)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=BASELINE_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
