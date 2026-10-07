"""Stable UX deterministic runner 与纯 oracle 的冻结合同测试

作者：xxx
"""

from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from scripts import stable_ux_acceptance as acceptance
from scripts.stable_ux_oracles import OracleAssertion, OracleResult, evaluate

SCENARIO_IDS = tuple(f"scenario_{index}" for index in range(1, 8))
Tamper = Callable[[dict[str, object]], None]


def _terminal_facts() -> list[dict[str, object]]:
    """返回包含 run/tool/done 链的标准 observed facts"""
    return [
        {"event": "run:start"},
        {"event": "llm:response"},
        {"event": "tool:request", "tool": {"name": "file_write"}},
        {"event": "tool:response", "tool": {"name": "file_write"}},
        {"event": "run:lifecycle", "lifecycle": "done"},
    ]


def _common_evidence(scenario_id: str) -> dict[str, object]:
    """返回与场景收集器当前公开键一致的隔离身份和报告字段"""
    return {
        "user_inputs": [f"input-{scenario_id}"],
        "session_id": f"session-{scenario_id}",
        "run_ids": [f"run-{scenario_id}"],
        "actual_output": f"output-{scenario_id}",
        "project_root": "project",
        "data_root": "data",
        "home_root": "home",
        "roots_are_isolated": True,
        "fact_files": ["data/sessions/session/runs/run/facts.jsonl"],
        "artifact_files": [],
        "notes": [],
    }


def _happy_scenario_1() -> dict[str, object]:
    """返回场景 1 的完整成功 observed evidence"""
    return _common_evidence("scenario_1") | {
        "html_exists": True,
        "html_contains_reins": True,
        "html_sha256": "a" * 64,
        "approval_records": [
            {
                "tool": "file_write",
                "risk": "confirm",
                "target": "C:\\isolated\\index.html",
                "target_existed": False,
            }
        ],
        "approval_count": 1,
        "approval_before_exists": True,
        "facts": _terminal_facts(),
        "fact_events": [fact["event"] for fact in _terminal_facts()],
        "session_state": {"last_run_id": "run-scenario_1", "last_run_status": "done"},
        "artifact_files": ["project/index.html"],
    }


def _happy_scenario_2() -> dict[str, object]:
    """返回场景 2 的完整成功 observed evidence"""
    previous = "请写一个个人介绍页面"
    correction = "我的意思是介绍 Reins 当前目录这个项目"
    return _common_evidence("scenario_2") | {
        "run_ids": ["run-2a", "run-2b"],
        "session_continuous": True,
        "page_contains_reins": True,
        "page_personal_primary": False,
        "page_text": "<html><h1>Reins 项目介绍</h1></html>",
        "conversation_rows": [
            {"role": "user", "content": previous},
            {"role": "assistant", "content": "已创建页面"},
            {"role": "user", "content": correction},
        ],
        "conversation_text": f"{previous}\n{correction}",
        "user_inputs": [previous, correction],
        "summary": "页面已纠正为 Reins 项目介绍",
        "html_before_sha256": "a" * 64,
        "html_after_sha256": "b" * 64,
        "approval_count": 1,
        "artifact_files": ["project/index.html"],
    }


def _happy_scenario_3() -> dict[str, object]:
    """返回场景 3 的协议错误恢复 observed evidence"""
    invalid_raw = "这是可读文本，但不是合法协议 JSON"
    return _common_evidence("scenario_3") | {
        "bare_protocol_error": invalid_raw,
        "raw_response_files": ["data/reins.db"],
        "raw_response_refs": ["evidence:response-1", "evidence:response-2"],
        "raw_response_text": f"{invalid_raw}\n合法恢复响应",
        "protocol_error_visible_to_model": True,
        "model_requests": [
            {"messages": [{"content": "初始请求"}]},
            {"messages": [{"content": "invalid_model_protocol，请修正协议"}]},
        ],
        "facts": [
            {
                "event": "llm:response",
                "summary": {
                    "error": {
                        "category": "invalid_model_protocol",
                        "raw_response_path": "evidence:response-1",
                    }
                },
            },
            {"event": "llm:response", "summary": {"has_final": True}},
            {"event": "run:lifecycle", "lifecycle": "done"},
        ],
        "final_lifecycle": "done",
        "summary": "Reins 是一个可恢复、可审计的 Agent harness",
        "error_fact_present": True,
        "artifact_files": ["data/reins.db"],
    }


def _happy_scenario_4() -> dict[str, object]:
    """返回场景 4 的失败回灌与改策 observed evidence"""
    error = "retryable_once"
    calls = [{"strategy": "initial"}, {"strategy": "alternate"}]
    responses = [
        {
            "name": "acceptance_strategy",
            "status": "error",
            "error": error,
            "meta": {"retryable": True},
        },
        {"name": "acceptance_strategy", "status": "success", "error": "", "meta": {}},
    ]
    facts: list[dict[str, object]] = []
    for response in responses:
        facts.extend(
            [
                {"event": "tool:request", "tool": {"name": "acceptance_strategy"}},
                {"event": "tool:response", "tool": response},
            ]
        )
    facts.append({"event": "run:lifecycle", "lifecycle": "done"})
    return _common_evidence("scenario_4") | {
        "tool_calls": calls,
        "tool_responses": responses,
        "strategy_changed": True,
        "attempt_count": 2,
        "model_tool_call_count": 2,
        "max_steps": 6,
        "confirm_side_effect_bypassed": False,
        "final_lifecycle": "done",
        "model_requests": [{"content": "首次调用"}, {"content": f"观察到 {error}"}],
        "facts": facts,
    }


def _happy_scenario_5() -> dict[str, object]:
    """返回与真实两次恢复同形的原件、查询和权限回执；参数：无；返回：完整观测。"""
    pending = {
        "tool_name": "file_write",
        "args": {
            "path": "pending.txt",
            "content": "must not run",
            "expected_sha256": "a" * 64,
        },
        "call_id": "call-source",
    }
    source = {
        "checkpoint_id": "cp-source",
        "session_id": "session-source",
        "run_id": "run-source",
        "pending_tool_call": pending,
    }
    operation = {
        "operation_id": "op-source",
        "session_id": "session-source",
        "run_id": "run-source",
        "call": pending,
        "state": "not_started",
    }
    query = {
        "operation_id": "op-query",
        "run_id": "run-query",
        "session_id": "session-source",
        "state": "completed",
        "call": {
            "tool_name": "operation_status",
            "call_id": "call-query",
            "args": {"operation_id": "op-source"},
        },
        "result": {"status": "ok", "content": json.dumps({"operations": [operation]})},
    }
    retry = {
        "operation_id": "op-retry",
        "run_id": "run-retry",
        "session_id": "session-source",
        "state": "not_started",
        "call": {
            "tool_name": "resume_operation",
            "call_id": "call-retry",
            "args": {"action": "retry", "operation_id": "op-source"},
        },
        "authorization": {
            "decision": "deny",
            "source": "user_action",
            "batch_id": "batch",
        },
        "result": {
            "status": "error",
            "meta": {
                "execution_state": "not_started",
                "tool_error_category": "permission",
                "approval_state": "denied",
            },
        },
    }
    query_facts = _resume_receipt_facts("operation_status", "query", "ok")
    retry_facts = _resume_receipt_facts("resume_operation", "retry", "error")
    feedback = {
        "kind": "tool_result",
        "tool_name": "operation_status",
        "call_id": "call-query",
        "status": "success",
        "content": [
            {
                "kind": "text",
                "text": json.dumps(
                    {
                        "status": "ok",
                        "output": json.dumps({"operations": [operation]}),
                        "meta": {"operation_id": "op-query"},
                    }
                ),
            }
        ],
    }
    return _common_evidence("scenario_5") | {
        "session_id": "session-source",
        "run_ids": ["run-query", "run-retry"],
        "source_checkpoint": source,
        "source_checkpoint_after": deepcopy(source),
        "checkpoint_after_query": {
            "run_id": "run-query",
            "session_id": "session-source",
        },
        "checkpoint_after_retry": {
            "run_id": "run-retry",
            "session_id": "session-source",
        },
        "source_operation": operation,
        "source_operation_after": deepcopy(operation),
        "query_operations": [query],
        "retry_operations": [retry],
        "approval_batches": [
            {
                "batch_id": "batch",
                "requests": [
                    {
                        "tool": "file_write",
                        "args": pending["args"],
                        "operation_id": "op-retry",
                    }
                ],
            }
        ],
        "model_requests": [
            {"content": "查询"},
            {
                "run_id": "run-query",
                "session_id": "session-source",
                "sources": {
                    "messages": [
                        {
                            "kind": "tool_result",
                            "call_id": "call-query",
                            "fed_back": feedback,
                        }
                    ]
                },
            },
        ],
        "query_facts": query_facts,
        "retry_facts": retry_facts,
        "facts": query_facts + retry_facts,
        "before_sha256": "a" * 64,
        "after_query_sha256": "a" * 64,
        "after_retry_sha256": "a" * 64,
    }


def _resume_receipt_facts(
    tool: str, suffix: str, status: str
) -> list[dict[str, object]]:
    """构造带真实身份字段的纯判定器输入；参数：工具、身份后缀及状态；返回：请求回执与结束事实。"""
    return [
        {
            "event": "tool:request",
            "run_id": f"run-{suffix}",
            "session_id": "session-source",
            "tool": {"name": tool, "call_id": f"call-{suffix}"},
        },
        {
            "event": "tool:response",
            "run_id": f"run-{suffix}",
            "session_id": "session-source",
            "operation_id": f"op-{suffix}",
            "tool": {"name": tool, "call_id": f"call-{suffix}", "status": status},
        },
        {"event": "llm:response", "summary": {"has_final": True}},
        {"event": "run:lifecycle", "lifecycle": "done"},
    ]


def _happy_scenario_6() -> dict[str, object]:
    """返回场景 6 的本地 stdio MCP 成功与拒绝 observed evidence"""
    return _common_evidence("scenario_6") | {
        "mcp_allowed_result": {
            "name": "mcp_echo_echo",
            "status": "success",
            "output_summary": "echo: reins-echo",
        },
        "mcp_denied_result": {
            "name": "mcp_echo_missing",
            "status": "error",
            "error": "MCP tool unavailable",
        },
        "mcp_risk": "confirm",
        "mcp_transport_used": True,
        "mcp_process_stopped": True,
        "approval_records": [
            {"tool": "mcp_echo_echo", "risk": "confirm", "target_existed": False}
        ],
        "facts": [
            {"event": "tool:request", "tool": {"name": "mcp_echo_echo"}},
            {
                "event": "tool:response",
                "tool": {"name": "mcp_echo_echo", "status": "success"},
            },
            {"event": "tool:request", "tool": {"name": "mcp_echo_missing"}},
            {
                "event": "tool:response",
                "tool": {"name": "mcp_echo_missing", "status": "error"},
            },
            {"event": "run:lifecycle", "lifecycle": "done"},
        ],
        "final_lifecycle": "done",
        "artifact_files": ["mcp_test.yaml"],
    }


def _happy_scenario_7() -> dict[str, object]:
    """返回场景 7 的 Chromium extract、截图与引用 observed evidence"""
    digest = "c" * 64
    return _common_evidence("scenario_7") | {
        "browser_text": "Hello Email Send Docs",
        "screenshot_exists": True,
        "screenshot_size": 256,
        "screenshot_sha256": digest,
        "screenshot_bytes_sha256": digest,
        "screenshot_path": "data/tasks/task/artifacts/artifact-screen-1.png",
        "screenshot_referenced": True,
        "artifact_refs": [
            {"artifact_id": "artifact-screen-1", "artifact_type": "image"}
        ],
        "artifact_files": [
            "project/sample.html",
            "data/tasks/task/artifacts/artifact-screen-1.png",
        ],
        "facts": [
            {"event": "tool:request", "tool": {"name": "browser_navigate"}},
            {"event": "tool:response", "tool": {"name": "browser_navigate"}},
            {"event": "tool:request", "tool": {"name": "browser_extract"}},
            {"event": "tool:response", "tool": {"name": "browser_extract"}},
            {"event": "tool:request", "tool": {"name": "browser_screenshot"}},
            {
                "event": "tool:response",
                "tool": {
                    "name": "browser_screenshot",
                    "artifact_refs": [
                        {"artifact_id": "artifact-screen-1", "artifact_type": "image"}
                    ],
                },
            },
            {"event": "run:lifecycle", "lifecycle": "done"},
        ],
        "final_lifecycle": "done",
    }


_HAPPY_BUILDERS: Mapping[str, Callable[[], dict[str, object]]] = {
    "scenario_1": _happy_scenario_1,
    "scenario_2": _happy_scenario_2,
    "scenario_3": _happy_scenario_3,
    "scenario_4": _happy_scenario_4,
    "scenario_5": _happy_scenario_5,
    "scenario_6": _happy_scenario_6,
    "scenario_7": _happy_scenario_7,
}


def _tamper_scenario_1(observed: dict[str, object]) -> None:
    """删除场景 1 的 Reins HTML 内容判定证据"""
    observed["html_contains_reins"] = False


def _tamper_scenario_2(observed: dict[str, object]) -> None:
    """把场景 2 的页面内容证据改回个人介绍"""
    observed["page_text"] = "<html><h1>个人介绍</h1></html>"
    observed["page_contains_reins"] = False
    observed["page_personal_primary"] = True


def _tamper_scenario_3(observed: dict[str, object]) -> None:
    """删除场景 3 的首个 raw response evidence"""
    observed["raw_response_files"] = []


def _tamper_scenario_4(observed: dict[str, object]) -> None:
    """删除场景 4 的首次 retryable tool error"""
    responses = observed["tool_responses"]
    assert isinstance(responses, list)
    observed["tool_responses"] = responses[1:]


def _tamper_scenario_5(observed: dict[str, object]) -> None:
    """删除源恢复点原件证据；参数：观测；返回：无。"""
    observed["source_checkpoint_after"] = {}


def _tamper_scenario_6(observed: dict[str, object]) -> None:
    """把场景 6 的 MCP 工具风险篡改为 SAFE"""
    observed["mcp_risk"] = "safe"


def _tamper_scenario_7(observed: dict[str, object]) -> None:
    """清空场景 7 的真实 screenshot bytes 哈希"""
    observed["screenshot_bytes_sha256"] = ""


_TAMPER_CASES: tuple[tuple[str, Tamper, str], ...] = (
    ("scenario_1", _tamper_scenario_1, "S1_HTML_CONTENT_INVALID"),
    ("scenario_2", _tamper_scenario_2, "S2_PAGE_CORRECTION_INVALID"),
    ("scenario_3", _tamper_scenario_3, "S3_RAW_RESPONSE_MISSING"),
    ("scenario_4", _tamper_scenario_4, "S4_RETRYABLE_ERROR_MISSING"),
    ("scenario_5", _tamper_scenario_5, "S5_PENDING_EVIDENCE_CHANGED"),
    ("scenario_6", _tamper_scenario_6, "S6_MCP_RISK_INVALID"),
    ("scenario_7", _tamper_scenario_7, "S7_SCREENSHOT_INVALID"),
)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_each_oracle_accepts_complete_observed_evidence(scenario_id: str) -> None:
    """验证七个 oracle 只在完整 observed evidence 下返回 pass"""
    result = evaluate(scenario_id, _HAPPY_BUILDERS[scenario_id]())

    assert result.status == "pass"
    assert result.error_code == ""
    assert result.assertions
    assert all(assertion.passed for assertion in result.assertions)


@pytest.mark.parametrize(("scenario_id", "tamper", "error_code"), _TAMPER_CASES)
def test_each_oracle_rejects_tampered_observed_evidence(
    scenario_id: str,
    tamper: Tamper,
    error_code: str,
) -> None:
    """验证七组删改 observed evidence 的反证均稳定返回 fail 和机器码"""
    observed = _HAPPY_BUILDERS[scenario_id]()
    tamper(observed)

    result = evaluate(scenario_id, observed)

    assert result.status == "fail"
    assert result.error_code == error_code
    assert any(not assertion.passed for assertion in result.assertions)


def _report_observation(scenario_id: str) -> dict[str, object]:
    """返回 runner 聚合测试所需的最小场景输出字段"""
    return {
        "scenario_id": scenario_id,
        "title": f"title-{scenario_id}",
        "user_inputs": [f"input-{scenario_id}"],
        "session_id": f"session-{scenario_id}",
        "run_ids": [f"run-{scenario_id}"],
        "actual_output": f"output-{scenario_id}",
        "fact_files": [f"{scenario_id}/data/facts.jsonl"],
        "artifact_files": [f"{scenario_id}/artifacts/evidence.txt"],
        "can_continue": True,
        "notes": ["deterministic fixture"],
    }


def _pass_result(scenario_id: str) -> OracleResult:
    """返回 runner 聚合测试使用的成功 oracle 结果"""
    assertion = OracleAssertion(
        id=f"{scenario_id}_observed",
        passed=True,
        actual="observed",
    )
    return OracleResult(status="pass", assertions=(assertion,), error_code="")


def _install_fake_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_scenario: str = "",
    raised_scenario: str = "",
    called: list[str] | None = None,
) -> None:
    """安装无副作用场景边界，专门验证 runner 聚合而非产品执行

    参数：failure_scenario 控制 oracle fail；raised_scenario 控制场景异常；called 记录顺序
    返回：无
    """

    def fake_run_scenario(
        scenario_id: str,
        *,
        evidence_root: Path,
        source_root: Path,
    ) -> Mapping[str, object]:
        """记录场景顺序并返回确定性报告字段"""
        del evidence_root, source_root
        if called is not None:
            called.append(scenario_id)
        if scenario_id == raised_scenario:
            raise RuntimeError("controlled scenario failure")
        return _report_observation(scenario_id)

    def fake_evaluate(
        scenario_id: str,
        observed: Mapping[str, object],
    ) -> OracleResult:
        """按指定场景返回确定性 pass/fail，不读取场景路径"""
        del observed
        if scenario_id != failure_scenario:
            return _pass_result(scenario_id)
        assertion = OracleAssertion("controlled_failure", False, "tampered")
        return OracleResult("fail", (assertion,), "CONTROLLED_FAILURE")

    monkeypatch.setattr(acceptance, "run_scenario", fake_run_scenario, raising=False)
    monkeypatch.setattr(acceptance, "evaluate", fake_evaluate, raising=False)


def test_v2_report_contains_exactly_seven_pass_or_fail_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 v2 report 顶层和值域合同，不再允许 blocked"""
    called: list[str] = []
    _install_fake_harness(monkeypatch, called=called)

    report = acceptance.build_report(tmp_path, evidence_root=tmp_path / "evidence")

    assert report["schema"] == "reins.stable_ux_acceptance.v2"
    assert report["status"] == "pass"
    assert report["status_counts"] == {"pass": 7, "fail": 0}
    assert report["evidence_root"] == "evidence"
    assert called == list(SCENARIO_IDS)
    _assert_report_results(report["results"])


def _assert_report_results(value: object) -> None:
    """校验 ScenarioResult 的固定字段、顺序与相对 evidence path"""
    assert isinstance(value, list)
    assert [item["scenario_id"] for item in value] == list(SCENARIO_IDS)
    required = {
        "scenario_id",
        "title",
        "status",
        "user_inputs",
        "session_id",
        "run_ids",
        "actual_output",
        "assertions",
        "fact_files",
        "artifact_files",
        "error_code",
        "can_continue",
        "notes",
    }
    for item in value:
        assert set(item) == required
        assert item["status"] in {"pass", "fail"}
        assert item["status"] != "blocked"
        for path in [*item["fact_files"], *item["artifact_files"]]:
            assert not Path(path).is_absolute()


def test_report_continues_after_scenario_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证单场景异常记 fail 后仍按 1-7 顺序完成完整报告"""
    called: list[str] = []
    _install_fake_harness(monkeypatch, raised_scenario="scenario_3", called=called)

    report = acceptance.build_report(tmp_path, evidence_root=tmp_path / "evidence")

    assert called == list(SCENARIO_IDS)
    assert report["status"] == "fail"
    assert report["status_counts"] == {"pass": 6, "fail": 1}
    results = report["results"]
    failed = next(item for item in results if item["scenario_id"] == "scenario_3")
    assert failed["status"] == "fail"
    assert failed["error_code"]


def test_cli_exit_code_and_written_report_follow_aggregate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 CLI 全通过退出 0、任一失败退出 1，且两次都先写完整 v2 report"""
    pass_output = tmp_path / "pass-report.json"
    _install_fake_harness(monkeypatch)

    pass_code = acceptance.main(
        ["--project-root", str(tmp_path), "--output", str(pass_output)]
    )

    assert pass_code == 0
    assert json.loads(pass_output.read_text(encoding="utf-8"))["status"] == "pass"

    fail_output = tmp_path / "fail-report.json"
    _install_fake_harness(monkeypatch, failure_scenario="scenario_4")
    fail_code = acceptance.main(
        ["--project-root", str(tmp_path), "--output", str(fail_output)]
    )
    failed_report = json.loads(fail_output.read_text(encoding="utf-8"))

    assert fail_code == 1
    assert failed_report["status"] == "fail"
    assert len(failed_report["results"]) == 7


def test_unknown_scenario_is_a_programming_error() -> None:
    """验证未知场景不会被默认成功或静默映射到其他 oracle"""
    with pytest.raises(ValueError, match="unknown stable UX scenario"):
        evaluate("scenario_8", {})


@pytest.fixture(scope="module")
def actual_resume_observed(tmp_path_factory):
    """运行一次真实恢复场景并验证原件可读取；参数：临时目录工厂；返回：共享的实际证据。"""
    from scripts.stable_ux_scenarios import run_scenario

    root = tmp_path_factory.mktemp("stable-resume")
    observed = run_scenario(
        "scenario_5",
        evidence_root=root,
        source_root=Path(__file__).resolve().parents[2],
    )
    assert evaluate("scenario_5", observed).status == "pass"
    assert all(
        (root / "scenario_5" / name).is_file() for name in observed["fact_files"]
    )
    return observed


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("query_operations",), [], "S5_QUERY_EVIDENCE_INVALID"),
        (
            ("source_checkpoint_after", "pending_tool_call"),
            None,
            "S5_PENDING_EVIDENCE_CHANGED",
        ),
        (
            ("source_operation_after", "call", "args"),
            {"path": "other.txt"},
            "S5_PENDING_EVIDENCE_CHANGED",
        ),
        (
            ("source_operation_after", "call", "call_id"),
            "other-call",
            "S5_PENDING_EVIDENCE_CHANGED",
        ),
        (
            ("source_operation_after", "state"),
            "completed",
            "S5_PENDING_EVIDENCE_CHANGED",
        ),
        (
            ("source_operation_after", "operation_id"),
            "other-op",
            "S5_PENDING_EVIDENCE_CHANGED",
        ),
        (
            ("query_operations", 0, "call", "args"),
            {"operation_id": "other-op"},
            "S5_QUERY_EVIDENCE_INVALID",
        ),
        (("after_query_sha256",), "b" * 64, "S5_SIDE_EFFECT_BEFORE_DECISION"),
        (("after_retry_sha256",), "b" * 64, "S5_SIDE_EFFECT_BEFORE_DECISION"),
        (("query_facts",), [], "S5_QUERY_EVIDENCE_INVALID"),
        (("model_requests",), [], "S5_QUERY_EVIDENCE_INVALID"),
        (("session_id",), "another-session", "S5_RESUME_RUN_MISSING"),
        (
            ("retry_operations", 0, "result", "status"),
            "ok",
            "S5_RETRY_PERMISSION_NOT_ENFORCED",
        ),
        (
            ("approval_batches", 0, "requests", 0, "operation_id"),
            "other-op",
            "S5_RETRY_PERMISSION_NOT_ENFORCED",
        ),
        (
            ("approval_batches", 0, "requests", 0, "args"),
            {"path": "other.txt"},
            "S5_RETRY_PERMISSION_NOT_ENFORCED",
        ),
        (
            ("checkpoint_after_query", "run_id"),
            "other-run",
            "S5_TERMINAL_LIFECYCLE_MISSING",
        ),
    ],
)
def test_real_resume_evidence_rejects_tampering(
    actual_resume_observed, path, value, error
):
    """篡改实际运行证据必须触发具体拒绝；参数：真实观测、字段路径和替换值；返回：无。"""
    observed = deepcopy(actual_resume_observed)
    target = observed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = evaluate("scenario_5", observed)
    assert result.status == "fail" and result.error_code == error


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("source_call", "S5_PENDING_EVIDENCE_CHANGED"),
        ("query_run", "S5_QUERY_EVIDENCE_INVALID"),
        ("retry_run", "S5_RETRY_PERMISSION_NOT_ENFORCED"),
        ("query_feedback", "S5_QUERY_EVIDENCE_INVALID"),
        ("query_final", "S5_TERMINAL_LIFECYCLE_MISSING"),
    ],
)
def test_real_resume_rejects_consistent_but_unrelated_evidence(
    actual_resume_observed, kind, error
):
    """多处一起篡改仍须绑定原恢复身份和实际回灌；参数：真实证据、篡改类型及错误码；返回：无。"""
    observed = deepcopy(actual_resume_observed)
    if kind == "source_call":
        for name in ("source_operation", "source_operation_after"):
            observed[name]["call"]["call_id"] = "another-source-call"
    elif kind in {"query_run", "retry_run"}:
        prefix = kind.removesuffix("_run")
        observed[f"{prefix}_operations"][0]["run_id"] = "another-run"
        for fact in observed[f"{prefix}_facts"]:
            fact["run_id"] = "another-run"
    elif kind == "query_feedback":
        call_id = observed["query_operations"][0]["call"]["call_id"]
        messages = observed["model_requests"][-1]["sources"]["messages"]
        observed["model_requests"][-1]["sources"]["messages"] = [
            row for row in messages if row.get("call_id") != call_id
        ]
        assert any(row.get("call_id") == "call-pending" for row in messages)
    else:
        observed["query_facts"] = [
            row for row in observed["query_facts"] if row["event"] != "run:lifecycle"
        ]
    result = evaluate("scenario_5", observed)
    assert result.status == "fail" and result.error_code == error


def test_selected_scenario_cli_preserves_failure_and_default_coverage(
    tmp_path, monkeypatch
):
    """单场景检查保留失败状态；参数：隔离根和替身；返回：无"""
    called = []
    _install_fake_harness(monkeypatch, called=called, failure_scenario="scenario_3")
    output = tmp_path / "report.json"
    assert (
        acceptance.main(
            [
                "--project-root",
                str(tmp_path),
                "--output",
                str(output),
                "--scenario",
                "scenario_3",
            ]
        )
        == 1
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert called == ["scenario_3"]
    assert report["status_counts"] == {"pass": 0, "fail": 1}


def test_protocol_recovery_report_points_to_real_file_originals(tmp_path):
    """协议恢复证据必须定位文件原件；参数：隔离根；返回：无"""
    from scripts.stable_ux_scenarios import run_scenario

    observed = run_scenario(
        "scenario_3",
        evidence_root=tmp_path,
        source_root=Path(__file__).resolve().parents[2],
    )
    assert evaluate("scenario_3", observed).status == "pass"
    root = tmp_path / "scenario_3"
    responses = [root / name for name in observed["raw_response_files"]]
    assert len(responses) == 2
    assert all(path.name == "output.json" and path.is_file() for path in responses)
    facts = [root / name for name in observed["fact_files"]]
    assert facts and all(
        path.name == "events.jsonl" and path.is_file() for path in facts
    )


def test_cli_resolves_evidence_from_report_directory(tmp_path, monkeypatch):
    """报告只以输出目录定位证据；参数：隔离根和替身；返回：无"""
    project = tmp_path / "source"
    project.mkdir()
    output = tmp_path / "reports" / "report.json"
    _install_fake_harness(monkeypatch)
    relative_path = acceptance._relative_path
    bases = []

    def report_relative(base, target):
        """记录相对路径的基准；参数：基准与目标目录；返回：相对路径"""
        bases.append(base)
        assert base == output.parent
        return relative_path(base, target)

    monkeypatch.setattr(acceptance, "_relative_path", report_relative)
    assert (
        acceptance.main(["--project-root", str(project), "--output", str(output)]) == 0
    )
    assert bases
    assert (
        json.loads(output.read_text(encoding="utf-8"))["evidence_root"]
        == "stable_ux_evidence"
    )
