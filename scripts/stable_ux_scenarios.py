from __future__ import annotations
from scripts.testing.llm import from_test_sequence

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import approval
import yaml
from approval import ApprovalDecision, ApprovalRequest
from runtime.agent_loop import AgentLoop
from runtime.checkpoint import (
    Checkpoint,
    checkpoint_to_ledger_state,
    summarize_checkpoint,
)
from runtime.lease import Lease, from_trigger
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_facts import RunFactStore
from runtime.run_evidence import RunEvidenceStore
from runtime.persistence import RuntimeStore
from runtime.schema_meta import ensure_current_schema
from runtime.session_messages import append_user_message, read_history_rows
from runtime.session_state import SessionStateStore
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore
from tools.browser.playwright_adapter import close_session
from tools.builtin_tools import build_tool_registry
from tools.mcp_client.registry import attach_mcp_registry
from tools.tool_registry import (
    IDEMPOTENT_YES,
    TARGET_SCOPE_LOGICAL,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_AGENT,
    ToolDefinition,
    ToolRegistry,
)
from tools.types import ToolError, ToolErrorCategory

ObservedEvidence = dict[str, object]
_ScenarioRunner = Callable[["_ScenarioEnvironment"], ObservedEvidence]


@dataclass(frozen=True, slots=True)
class _ScenarioEnvironment:
    """保存单场景隔离根。

    作者：xxx
    时间：2026-08-18 00:00:00
    参数：root/source_root 为场景证据根和仓库只读 fixture 根
    返回：不可变的 project/data/home 路径集合
    """

    root: Path
    project_root: Path
    data_root: Path
    home: Path
    source_root: Path


@dataclass(frozen=True, slots=True)
class _RunCapture:
    """保存一次真实 AgentLoop 执行后观察到的证据。"""

    task_id: str
    session_id: str
    run_id: str
    output: str
    facts: tuple[dict[str, object], ...]
    model_requests: tuple[dict[str, object], ...]


class _ApprovalRecorder:
    """记录审批发生时目标文件的真实状态。"""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self.calls: list[dict[str, object]] = []

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        """记录审批请求并允许一次。

        参数：request 为 ToolRegistry 在副作用前生成的生产审批请求
        返回：ONCE，确保授权只覆盖当前精确调用
        """
        target = _request_target(self._project_root, request.args)
        self.calls.append(
            {
                "tool": request.tool,
                "risk": request.risk,
                "target": str(target) if target is not None else "",
                "target_existed": bool(target and target.exists()),
                "data_root": str(request.data_root),
            }
        )
        return ApprovalDecision.ONCE


def run_scenario(
    scenario_id: str,
    *,
    evidence_root: Path,
    source_root: Path,
) -> ObservedEvidence:
    """运行一个隔离的 deterministic 产品场景。

    作者：xxx
    时间：2026-08-18 00:00:00
    参数：scenario_id 为冻结场景编号；evidence_root/source_root 为输出与 fixture 根
    返回：只包含实际观察证据的映射；未知场景抛 ValueError
    """
    runners: dict[str, _ScenarioRunner] = {
        "scenario_1": _scenario_1,
        "scenario_2": _scenario_2,
        "scenario_3": _scenario_3,
        "scenario_4": _scenario_4,
        "scenario_5": _scenario_5,
        "scenario_6": _scenario_6,
        "scenario_7": _scenario_7,
    }
    runner = runners.get(scenario_id)
    if runner is None:
        raise ValueError(f"unknown stable UX scenario: {scenario_id}")
    environment = _create_environment(evidence_root, scenario_id, source_root)
    with _isolated_user_state(environment.home):
        observed = runner(environment)
    observed_path = environment.root / "observed.json"
    observed_path.write_text(_json_text(observed), encoding="utf-8")
    return observed


def _scenario_1(environment: _ScenarioEnvironment) -> ObservedEvidence:
    """场景 1：经真实审批与 file_write 创建 Reins 页面。"""
    message = "读取项目资料并创建介绍 Reins 的 index.html。"
    recorder = _ApprovalRecorder(environment.project_root)
    responses = (
        _tool("file_read", {"path": "README.md"}),
        _tool("file_read", {"path": "Architecture.md"}),
        _tool("file_write", {"path": "index.html", "content": _reins_html()}),
        _final("Reins 项目页面已创建。"),
    )
    with _approval_backend(recorder):
        capture = _run_user_turn(environment, message, responses)
    html = environment.project_root / "index.html"
    facts_path = _facts_path(environment, capture)
    return _common_observed(environment, capture, [message]) | {
        "html_exists": html.is_file(),
        "html_contains_reins": html.is_file() and "Reins" in _read_text(html),
        "html_sha256": _hash_file(html),
        "approval_count": len(recorder.calls),
        "approval_before_exists": _approval_targets_were_absent(recorder.calls),
        "approval_records": recorder.calls,
        "fact_events": _fact_events(capture.facts),
        "facts": list(capture.facts),
        "session_state": _session_state_mapping(environment, capture.session_id),
        "fact_files": [_relative(environment.root, facts_path)],
        "artifact_files": [_relative(environment.root, html)],
    }


def _scenario_2(environment: _ScenarioEnvironment) -> ObservedEvidence:
    """场景 2：在同一 session 中纠正个人介绍目标。"""
    page = environment.project_root / "index.html"
    page.write_text(_personal_html(), encoding="utf-8")
    before_hash = _hash_file(page)
    task_id = _create_task(environment, "把个人介绍纠正为 Reins 项目介绍")
    first = _run_user_turn(
        environment,
        "先保留当前个人介绍页面。",
        (_final("已记录当前个人介绍。"),),
        task_id=task_id,
    )
    recorder = _ApprovalRecorder(environment.project_root)
    correction = "我的意思是介绍当前目录这个项目 Reins，不是个人介绍。"
    responses = (
        _tool("file_read", {"path": "index.html"}),
        _tool("file_read", {"path": "README.md"}),
        _tool("file_write", {"path": "index.html", "content": _reins_html()}),
        _final("页面已纠正为 Reins 项目介绍。"),
    )
    with _approval_backend(recorder):
        second = _run_user_turn(
            environment,
            correction,
            responses,
            task_id=task_id,
            session_id=first.session_id,
        )
    conversation = _conversation_text(environment, second.session_id)
    conversation_rows = _conversation_rows(environment, second.session_id)
    return _common_observed(
        environment, second, ["先保留当前个人介绍页面。", correction]
    ) | {
        "run_ids": [first.run_id, second.run_id],
        "session_continuous": first.session_id == second.session_id,
        "page_contains_reins": "Reins" in _read_text(page),
        "page_personal_primary": "个人介绍" in _read_text(page),
        "conversation_text": conversation,
        "conversation_rows": conversation_rows,
        "page_text": _read_text(page),
        "summary": _task_summary(environment, task_id),
        "facts": [*first.facts, *second.facts],
        "html_before_sha256": before_hash,
        "html_after_sha256": _hash_file(page),
        "approval_count": len(recorder.calls),
        "fact_files": [
            _relative(environment.root, _facts_path(environment, first)),
            _relative(environment.root, _facts_path(environment, second)),
        ],
        "artifact_files": [_relative(environment.root, page)],
    }


def _scenario_3(environment: _ScenarioEnvironment) -> ObservedEvidence:
    """场景 3：协议错误作为 evidence 回灌后恢复到 done。"""
    invalid = "Reins 是一个本地优先的 Agent harness，但这不是 JSON。"
    capture = _run_user_turn(
        environment,
        "读取当前目录并用一句话说明 Reins。",
        (invalid, _final("Reins 是本地优先、可恢复且可审计的 Agent harness。")),
    )
    responses = RunEvidenceStore(environment.data_root).list_records(
        session_id=capture.session_id,
        run_id=capture.run_id,
        kind="attempt_response",
    )
    raw_files = sorted(
        {
            RuntimeStore(environment.data_root).source_path(
                "run_evidence", row["reference"].removeprefix("evidence:")
            )
            for row in responses
        }
    )
    summary = _task_summary(environment, capture.task_id)
    request_text = _json_text(list(capture.model_requests)[1:])
    return _common_observed(environment, capture, ["读取当前目录并说明 Reins。"]) | {
        "raw_response_files": [_relative(environment.root, path) for path in raw_files],
        "raw_response_text": _json_text([row["payload"] for row in responses]),
        "raw_response_refs": [row["reference"] for row in responses],
        "protocol_error_visible_to_model": "invalid_model_protocol" in request_text,
        "final_lifecycle": _final_lifecycle(capture.facts),
        "summary": summary,
        "bare_protocol_error": invalid,
        "error_fact_present": _has_error_fact(capture.facts, "invalid_model_protocol"),
        "model_requests": list(capture.model_requests),
        "facts": list(capture.facts),
        "fact_files": [_relative(environment.root, _facts_path(environment, capture))],
        "artifact_files": [_relative(environment.root, path) for path in raw_files],
    }


def _scenario_4(environment: _ScenarioEnvironment) -> ObservedEvidence:
    """场景 4：可恢复工具错误回灌后模型更换参数。"""
    calls: list[dict[str, object]] = []
    registry = build_tool_registry(
        repo_root=environment.project_root,
        data_root=environment.data_root,
    )
    _register_strategy_tool(registry, calls)
    responses = (
        _tool("acceptance_strategy", {"strategy": "initial"}),
        _tool("acceptance_strategy", {"strategy": "alternate"}),
        _final("已改用 alternate 策略完成。"),
    )
    capture = _run_user_turn(
        environment,
        "工具失败后读取错误并换一个策略。",
        responses,
        registry=registry,
        max_steps=6,
    )
    tool_responses = _tool_facts(capture.facts, event="tool:response")
    return _common_observed(environment, capture, ["工具失败后换策略。"]) | {
        "tool_calls": calls,
        "tool_responses": tool_responses,
        "strategy_changed": len(calls) >= 2 and calls[0] != calls[-1],
        "attempt_count": len(calls),
        "model_tool_call_count": len(_tool_facts(capture.facts, event="tool:request")),
        "max_steps": 6,
        "confirm_side_effect_bypassed": False,
        "final_lifecycle": _final_lifecycle(capture.facts),
        "model_requests": list(capture.model_requests),
        "facts": list(capture.facts),
        "fact_files": [_relative(environment.root, _facts_path(environment, capture))],
    }


def _scenario_5(environment: _ScenarioEnvironment) -> ObservedEvidence:
    """检查恢复查询、合法结束与显式重试权限；参数：隔离环境；返回：真实执行证据。"""
    from scripts.stable_ux_resume_scenario import run_resume_scenario

    return run_resume_scenario(environment)


def _scenario_6(environment: _ScenarioEnvironment) -> ObservedEvidence:
    """场景 6：真实本地 MCP transport 覆盖允许与不可用工具。"""
    config_path = _write_local_mcp_config(environment)
    registry = build_tool_registry(
        repo_root=environment.project_root,
        data_root=environment.data_root,
    )
    recorder = _ApprovalRecorder(environment.project_root)
    responses = (
        _tool("capabilities", {"action": "load", "name": "mcp_echo_echo"}),
        _tool("mcp_echo_echo", {"text": "reins-echo"}),
        _tool("capabilities", {"action": "load", "name": "mcp_echo_missing"}),
        _final("MCP 成功与不可用路径均已记录。"),
    )
    with _approval_backend(recorder), closing(registry):
        capture = _run_user_turn(
            environment,
            "调用本地 echo MCP，再请求一个不存在的 MCP 工具。",
            responses,
            registry=registry,
            mcp_config=config_path,
        )
    definition = registry.get("mcp_echo_echo")
    tool_responses = _tool_facts(capture.facts, event="tool:response")
    return _common_observed(environment, capture, ["调用本地 echo MCP。"]) | {
        "mcp_allowed_result": _find_tool_fact(tool_responses, "mcp_echo_echo"),
        "mcp_denied_result": _find_tool_fact(tool_responses, "capabilities"),
        "mcp_risk": definition.risk.value if definition is not None else "",
        "mcp_transport_used": "reins-echo" in _json_text(tool_responses),
        "mcp_process_stopped": (environment.root / "server_exited").is_file(),
        "approval_records": recorder.calls,
        "final_lifecycle": _final_lifecycle(capture.facts),
        "facts": list(capture.facts),
        "fact_files": [_relative(environment.root, _facts_path(environment, capture))],
        "artifact_files": [_relative(environment.root, config_path)],
    }


def _scenario_7(environment: _ScenarioEnvironment) -> ObservedEvidence:
    """场景 7：真实 Chromium 打开本地页面、提取文本并截图。"""
    sample = environment.project_root / "sample.html"
    shutil.copy2(
        environment.source_root / "tests" / "acceptance" / "fixtures" / "sample.html",
        sample,
    )
    url = sample.resolve().as_uri()
    responses = (
        _tool("browser_navigate", {"url": url}),
        _tool("browser_extract", {}),
        _tool("browser_screenshot", {"full_page": "true"}),
        _final("本地页面已提取并保存截图。"),
    )
    recorder = _ApprovalRecorder(environment.project_root)
    try:
        with _approval_backend(recorder):
            capture = _run_user_turn(
                environment,
                "打开本地 sample.html，提取页面文字并截图。",
                responses,
            )
    finally:
        close_session()
    screenshots = sorted(environment.data_root.glob("tasks/*/artifacts/*.png"))
    screenshot = screenshots[0] if screenshots else None
    screenshot_hash = _hash_file(screenshot)
    facts_text = _json_text(capture.facts)
    return _common_observed(environment, capture, ["打开本地页面并截图。"]) | {
        "browser_text": _browser_extract_text(capture.facts),
        "screenshot_exists": bool(screenshot and screenshot.is_file()),
        "screenshot_size": screenshot.stat().st_size if screenshot else 0,
        "screenshot_sha256": screenshot_hash,
        "screenshot_bytes_sha256": screenshot_hash,
        "screenshot_path": _relative(environment.root, screenshot)
        if screenshot
        else "",
        "screenshot_referenced": bool(screenshot and screenshot.stem in facts_text),
        "artifact_refs": _artifact_refs(capture.facts),
        "final_lifecycle": _final_lifecycle(capture.facts),
        "facts": list(capture.facts),
        "fact_files": [_relative(environment.root, _facts_path(environment, capture))],
        "artifact_files": [
            _relative(environment.root, path) for path in [sample, *screenshots]
        ],
    }


def _create_environment(
    evidence_root: Path, scenario_id: str, source_root: Path
) -> _ScenarioEnvironment:
    """创建场景独占的 project/data/home 根。"""
    root = evidence_root.resolve() / scenario_id
    if root.exists():
        shutil.rmtree(root)
    project_root = root / "project"
    data_root = root / "data"
    home = root / "home"
    for path in (project_root, data_root, home):
        path.mkdir(parents=True, exist_ok=True)
    ensure_current_schema(data_root)
    (project_root / "README.md").write_text(
        "# Reins\nLocal-first agent harness.\n", encoding="utf-8"
    )
    (project_root / "Architecture.md").write_text(
        "# Architecture\nAgentLoop -> tools -> evidence.\n", encoding="utf-8"
    )
    return _ScenarioEnvironment(
        root, project_root, data_root, home, source_root.resolve()
    )


@contextmanager
def _isolated_user_state(home: Path) -> Iterator[None]:
    """把用户级读取重定向到场景模拟 home，并在退出时恢复。"""
    values = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "REINS_HOME": str(home / ".reins"),
    }
    with (
        patch.dict(os.environ, values, clear=False),
        patch.object(Path, "home", classmethod(lambda cls: home)),
    ):
        yield


@contextmanager
def _approval_backend(recorder: _ApprovalRecorder) -> Iterator[None]:
    """在单场景内登记审批 recorder，并在结束后清除全局 backend。"""
    approval.register_approval_backend(recorder)
    try:
        yield
    finally:
        approval.register_approval_backend(None)


def _run_user_turn(
    environment: _ScenarioEnvironment,
    message: str,
    responses: Sequence[str],
    *,
    task_id: str = "",
    session_id: str = "",
    registry: ToolRegistry | None = None,
    max_steps: int = 30,
    mcp_config: Path | None = None,
) -> _RunCapture:
    """用生产 RunContext 和 AgentLoop 执行一轮用户输入。"""
    active_task_id = task_id or _create_task(environment, message)
    context = RunContext(
        task_id=active_task_id,
        trigger=Trigger.USER,
        payload={"message": message},
        capability_lease=_lease(
            environment,
            active_task_id,
            max_steps=max_steps,
            mcp_config=mcp_config,
        ),
        session_id=session_id,
    )
    # 用户输入先落成 canonical 消息，再进 loop；session_id 由 RunContext 定稿
    context.payload["input_message_id"] = append_user_message(
        environment.data_root,
        context.session_id,
        message,
        run_id=context.run_id,
        task_id=active_task_id,
    )
    active_registry = registry or build_tool_registry(
        repo_root=environment.project_root,
        data_root=environment.data_root,
    )
    if mcp_config is not None:
        attach_mcp_registry(context.capability_lease, active_registry)
    return _run_context(environment, context, responses, active_registry)


def _run_context(
    environment: _ScenarioEnvironment,
    context: RunContext,
    responses: Sequence[str],
    registry: ToolRegistry,
) -> _RunCapture:
    """执行已构造的生产上下文并读取落盘 facts 与模型请求证据。

    作者：LKX
    时间：2026-08-31 17:20:00
    传参：environment 为场景隔离根；context 为生产 RunContext；responses 为逐轮受控模型文本；
          registry 为本场景工具注册表
    返回：_RunCapture，含 loop 输出、facts 和模型每轮实际收到的请求

    只有"模型这轮回什么"被换成脚本，装配、发送编排、证据投影都仍是生产实现。
    """
    client = from_test_sequence(responses, protocol_mode="text_json")
    loop = AgentLoop(environment.data_root, llm_client=client, tool_registry=registry)
    # 【验证】【模型证据】本场景明确检查实际请求，局部开启完整诊断
    with patch.dict(os.environ, {"REINS_TRACE_LEVEL": "debug"}):
        list(loop.run_stream(context))
    facts = RunFactStore(environment.data_root).read_run(context.run_id)
    return _RunCapture(
        context.storage_task_id,
        context.session_id,
        context.run_id,
        loop.last_output,
        tuple(facts),
        _model_request_payloads(environment, context),
    )


def _model_request_payloads(
    environment: _ScenarioEnvironment,
    context: RunContext,
) -> tuple[dict[str, object], ...]:
    """【验证】【模型证据】按实际尝试顺序读取本次运行的完整请求。

    参数：environment 为隔离根，context 为运行身份；返回：实际发出的请求列表
    """
    records = RunEvidenceStore(environment.data_root).list_records(
        session_id=context.session_id,
        run_id=context.run_id,
        kind="attempt_request",
    )
    return tuple(row["payload"] for row in records)


def _lease(
    environment: _ScenarioEnvironment,
    task_id: str,
    *,
    trigger: str = "user",
    max_steps: int = 30,
    mcp_config: Path | None = None,
) -> Lease:
    """建立只允许隔离根与本地 fixture 的能力租约。"""
    workspace = environment.project_root / ".reins" / "workspace"
    capabilities: dict[str, object] = {
        "fs": {
            "project_root": str(environment.project_root),
            "read": [
                str(environment.project_root),
                str(environment.data_root),
                str(workspace),
            ],
            "write": [str(environment.data_root), str(workspace)],
            "deny_read": ["*.pem", "*.key", ".env"],
        },
        "terminal": {"enabled": False, "allow_commands": []},
        "browser": {
            "enabled": True,
            "profile": "acceptance",
            "deny_domains": [],
            "headless": True,
        },
        "network": {"enabled": True, "deny_domains": []},
        "mouse_keyboard": {"enabled": False},
        "background_run": {"enabled": False},
        "mcp": {
            "enabled": True,
            "allow_servers": ["echo"] if mcp_config else [],
            **({"config_path": str(mcp_config)} if mcp_config else {}),
        },
    }
    return from_trigger(
        trigger,
        task_id=task_id,
        capabilities=capabilities,
        max_steps=max_steps,
        max_tokens=200_000,
    )


def _create_task(environment: _ScenarioEnvironment, goal: str) -> str:
    """在隔离 data root 建立正式任务并关闭索引连接。"""
    store = TaskStore(environment.data_root)
    record = store.create_task(goal)
    store.close()
    return record.task_id


def _persist_source_checkpoint(
    environment: _ScenarioEnvironment, checkpoint: Checkpoint
) -> None:
    """通过 Ledger writer 持久化场景 5 的源 checkpoint。"""
    writer = LedgerWriter(
        LedgerStore(environment.data_root), source="stable_ux_acceptance"
    )
    writer.record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        checkpoint.reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )
    RunFactStore(environment.data_root).append_checkpoint_ref(
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
        task_id=checkpoint.task_id,
        focus_task_id=checkpoint.focus_task_id,
        compatibility_task_id=checkpoint.compatibility_task_id,
        segment_id=checkpoint.segment_id,
        checkpoint=checkpoint,
    )
    SessionStateStore(environment.data_root).record_checkpoint(
        summarize_checkpoint(checkpoint)
    )


def _register_strategy_tool(
    registry: ToolRegistry, calls: list[dict[str, object]]
) -> None:
    """注册只供场景输入使用、仍穿过 ToolRegistry 的确定性失败工具。"""

    def execute(args: dict[str, object]) -> object:
        public = {key: value for key, value in args.items() if not key.startswith("__")}
        calls.append(public)
        if public.get("strategy") == "initial":
            return ToolError(
                ToolErrorCategory.TRANSPORT, "retryable_once", retryable=True
            )
        return {"status": "ok", "strategy": public.get("strategy")}

    registry.register(
        ToolDefinition(
            name="acceptance_strategy",
            description="Choose a deterministic local strategy.",
            parameters={"strategy": {"type": "string", "required": True}},
            toolset=TOOLSET_AGENT,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            idempotent=IDEMPOTENT_YES,
            executor=execute,
        )
    )


def _write_local_mcp_config(environment: _ScenarioEnvironment) -> Path:
    """派生只引用官方SDK本地服务的隔离配置；传参：场景环境；返回：配置路径。"""
    source = (
        environment.source_root / "tests" / "acceptance" / "fixtures" / "mcp_test.yaml"
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("servers"), dict):
        raise ValueError("invalid MCP acceptance fixture")
    echo = dict(payload["servers"]["echo"])
    wrapper = environment.source_root / "tests" / "fixtures" / "standard_mcp_server.py"
    echo["command"] = [sys.executable, str(wrapper)]
    echo["env"] = {
        "PYTHONPATH": str(environment.source_root),
        "MCP_TEST_TRACE_DIR": str(environment.root),
    }
    local = environment.root / "mcp_test.yaml"
    local.write_text(
        yaml.safe_dump({"servers": {"echo": echo}}, sort_keys=False), encoding="utf-8"
    )
    return local


def _common_observed(
    environment: _ScenarioEnvironment,
    capture: _RunCapture,
    user_inputs: list[str],
) -> ObservedEvidence:
    """组装所有场景共享的身份、输出和隔离证明。"""
    return {
        "user_inputs": user_inputs,
        "session_id": capture.session_id,
        "run_ids": [capture.run_id],
        "actual_output": capture.output,
        "project_root": _relative(environment.root, environment.project_root),
        "data_root": _relative(environment.root, environment.data_root),
        "home_root": _relative(environment.root, environment.home),
        "roots_are_isolated": _roots_are_isolated(environment),
        "fact_files": [],
        "artifact_files": [],
        "notes": [],
    }


def _request_target(project_root: Path, args: Mapping[str, object]) -> Path | None:
    """从审批参数解析场景项目内目标路径。"""
    value = args.get("path")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _approval_targets_were_absent(calls: Sequence[Mapping[str, object]]) -> bool:
    """判断所有写入审批发生时目标是否尚不存在。"""
    return bool(calls) and all(not bool(call.get("target_existed")) for call in calls)


def _tool(name: str, arguments: Mapping[str, object]) -> str:
    """编码 text-json 工具动作。"""
    return json.dumps(
        {"type": "run_tools", "tool": name, "arguments": dict(arguments)},
        ensure_ascii=False,
    )


def _final(content: str) -> str:
    """编码 text-json final 动作。"""
    return json.dumps({"type": "final", "content": content}, ensure_ascii=False)


def _reins_html() -> str:
    """返回场景真实写入的 Reins 项目页。"""
    return "<!doctype html><html><body><h1>Reins</h1><p>Local-first agent harness.</p></body></html>\n"


def _personal_html() -> str:
    """返回场景 2 的纠正前页面。"""
    return "<!doctype html><html><body><h1>个人介绍</h1><p>这是个人主页。</p></body></html>\n"


def _facts_path(environment: _ScenarioEnvironment, capture: _RunCapture) -> Path:
    """定位本次运行的真实事件原件；参数：隔离根与运行身份；返回：唯一原件路径。"""
    store = RuntimeStore(environment.data_root)
    with store.snapshot() as snapshot:
        rows = snapshot.list_raw(
            "run_fact",
            session_id=capture.session_id,
            filters={"run_id": capture.run_id},
        )
    paths = {store.source_path("run_fact", row.record_id) for row in rows}
    if len(paths) != 1:
        raise ValueError(f"expected one run fact source, found {len(paths)}")
    return paths.pop()


def _relative(root: Path, path: Path) -> str:
    """把场景路径转换为 evidence root 相对路径。"""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_text(path: Path) -> str:
    """读取 UTF-8 观察证据。"""
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _hash_file(path: Path | None) -> str:
    """计算实际文件 SHA-256；缺失文件返回空串。"""
    if path is None or not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_text(value: object) -> str:
    """以稳定 UTF-8 JSON 表达观察结构。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _fact_events(facts: Sequence[Mapping[str, object]]) -> list[str]:
    """按落盘顺序返回 fact event。"""
    return [str(fact.get("event", "")) for fact in facts]


def _tool_facts(
    facts: Sequence[Mapping[str, object]], *, event: str
) -> list[dict[str, object]]:
    """读取指定 event 下的 tool payload。"""
    return [
        dict(tool)
        for fact in facts
        if fact.get("event") == event
        and isinstance((tool := fact.get("tool")), Mapping)
    ]


def _find_tool_fact(
    facts: Sequence[Mapping[str, object]], tool_name: str
) -> dict[str, object]:
    """返回指定工具最后一个响应 fact。"""
    matches = [dict(fact) for fact in facts if fact.get("name") == tool_name]
    return matches[-1] if matches else {}


def _final_lifecycle(facts: Sequence[Mapping[str, object]]) -> str:
    """读取最后一个 run lifecycle 值。"""
    values = [
        str(fact.get("lifecycle", ""))
        for fact in facts
        if fact.get("event") == "run:lifecycle"
    ]
    return values[-1] if values else ""


def _has_error_fact(facts: Sequence[Mapping[str, object]], marker: str) -> bool:
    """判断结构化 facts 是否包含指定错误分类。"""
    return marker in _json_text(facts)


def _conversation_text(environment: _ScenarioEnvironment, session_id: str) -> str:
    """读取会话对话作为多轮连续性证据。"""
    return _json_text(_conversation_rows(environment, session_id))


def _conversation_rows(
    environment: _ScenarioEnvironment, session_id: str
) -> list[dict[str, object]]:
    """读取结构化对话行，供 oracle 独立判断多轮连续性。"""
    return read_history_rows(environment.data_root, session_id, limit=100)


def _session_state_mapping(
    environment: _ScenarioEnvironment, session_id: str
) -> dict[str, object]:
    """读取结构化 session state，供 oracle 独立核对 run identity。"""
    state = SessionStateStore(environment.data_root).load(session_id)
    return asdict(state) if state is not None else {}


def _task_summary(environment: _ScenarioEnvironment, task_id: str) -> str:
    """读取任务最终 summary 并关闭索引连接。"""
    store = TaskStore(environment.data_root)
    summary = store.read_summary(task_id)
    store.close()
    return summary


def _browser_extract_text(facts: Sequence[Mapping[str, object]]) -> str:
    """从 browser_extract 工具响应读取实际文本摘要。"""
    tool = _find_tool_fact(_tool_facts(facts, event="tool:response"), "browser_extract")
    return str(tool.get("output_summary", ""))


def _artifact_refs(facts: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """从原始 tool response facts 收集 artifact 引用。"""
    refs: list[dict[str, object]] = []
    for tool in _tool_facts(facts, event="tool:response"):
        value = tool.get("artifact_refs")
        if isinstance(value, list):
            refs.extend(dict(item) for item in value if isinstance(item, Mapping))
    return refs


def _roots_are_isolated(environment: _ScenarioEnvironment) -> bool:
    """确认 project/data/home 都位于当前场景证据根。"""
    root = environment.root.resolve()
    return all(
        path.resolve().is_relative_to(root)
        for path in (environment.project_root, environment.data_root, environment.home)
    )


__all__ = ["ObservedEvidence", "run_scenario"]
