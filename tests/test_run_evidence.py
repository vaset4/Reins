from __future__ import annotations
from scripts.testing.llm import (
    from_test_native_tool_then_final,
    from_test_sequence,
    from_test_turns,
)

import json
from pathlib import Path

import pytest

from context.production_builder import ProductionContextBuilder
from llm.client import RealLLMClient
from scripts.testing.llm import scripted_provider_error
from llm.messages import (
    AgentMessage,
    AssistantMessage,
    TextPart,
    ToolCallPart,
    UserMessage,
)
from llm.types import LLMPlan, ModelError
from runtime.agent_loop import AgentLoop, State
from runtime.cancellation import CancellationToken
from runtime.watchdog import Watchdog
from runtime.ledger import LedgerStore
from runtime.lease import from_trigger
from runtime.run_evidence_view import build_run_evidence_view
from runtime.run_facts import RunFactStore
from runtime.run_evidence import RunEvidenceStore
from runtime.session_messages import append_assistant_message, append_user_message
from runtime.types import RunContext, RunToolsRequest, RunToolsResult, Trigger
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import (
    IDEMPOTENT_YES,
    TARGET_SCOPE_LOGICAL,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    ToolDefinition,
    ToolRegistry,
)
from tools.types import ToolError, ToolErrorCategory


def test_model_raw_evidence_and_parsed_plan_are_written(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(['{"type":"final","content":"hello evidence"}'])
    loop, context, store = _build_loop(project, client)

    assert loop.run(context) is State.DONE

    request = _evidence(project, context, "attempt_request")[0]
    response = _evidence(project, context, "attempt_response")[0]
    parsed = _evidence(project, context, "model_plan")[0]
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    llm_fact = next(fact for fact in facts if fact.get("event") == "llm:response")

    assert request["request"]["messages"][0]["role"] == "system"
    assert request["sources"]["composition"]["tool_selection"]
    assert "tools" in request["request"]
    assert response["response"]["text"] == '{"type":"final","content":"hello evidence"}'
    assert parsed["success"] is True
    assert parsed["protocol_mode"] == "native_tool_calls"
    assert parsed["has_final"] is True
    assert llm_fact["summary"]["evidence"]["model_request"].startswith("evidence:")
    assert not (project / ".reins" / "data" / "sessions").exists()
    assert store.read_summary(context.task_id or "") == "hello evidence"


def test_model_requests_without_tools_keep_distinct_evidence(tmp_path: Path) -> None:
    """连续目标动作没有工具调用时，各轮请求与响应仍独立保存。

    传参：tmp_path 为隔离目录；返回：无，核对事实引用、请求身份和历史响应
    """
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"goal_op","goal_op_kind":"new","goal_body":"核对报告"}',
            '{"type":"final","content":"还需第二份材料"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _ = _build_loop(project, client)
    assert loop.run(context) is State.DONE
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    responses = [row for row in facts if row["event"] == "llm:response"]
    paths = [row["summary"]["evidence"]["model_response"] for row in responses]
    assert len(paths) == len(set(paths)) == 2
    request_ids = [row["request_id"] for row in responses]
    assert len(set(request_ids)) == 2
    first = RunEvidenceStore(project / ".reins" / "data").read_reference(paths[0])
    assert "goal_op" in first["response"]["text"]


def test_provider_retry_preserves_each_attempt_evidence(tmp_path: Path) -> None:
    """一次请求中的失败重试与成功尝试都能独立回取，未知用量不填零。

    传参：tmp_path 为隔离目录；返回：无，核对尝试身份、错误及用量
    """
    project = _project(tmp_path)
    client = from_test_turns(
        [
            scripted_provider_error(
                category="rate_limited", summary="暂时限流", retryable=True
            ),
            "重试后收到结果",
        ]
    )
    loop, context, _ = _build_loop(project, client)
    assert loop.run(context) is State.DONE
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    attempts = [row for row in facts if row["event"] == "llm:attempt"]
    assert len(attempts) == 2
    assert len({row["attempt_id"] for row in attempts}) == 2
    assert len({row["request_id"] for row in attempts}) == 1
    assert [row["success"] for row in attempts] == [False, True]
    assert attempts[0]["error_category"] == "rate_limited"
    assert attempts[0]["usage"]["total_tokens"]["value"] is None
    assert attempts[0]["usage"]["total_tokens"]["status"] == "unknown"


def test_persisted_current_input_enters_model_request_once(tmp_path: Path) -> None:
    """已保存的本轮输入在首轮与工具续轮里均只出现一次。

    传参：tmp_path 为隔离目录；返回：无，核对真正发送的 Provider 请求
    """
    from app.run_task import run_task

    project = _project(tmp_path)
    message = "请读取 tools/alpha.py 并解释"
    client = from_test_native_tool_then_final(
        (ToolCallPart("call-read", "file_read", {"path": "tools/alpha.py"}),),
        "已读取",
    )
    response = run_task(
        message,
        project,
        data_root=project / ".reins" / "data",
        llm_client=client,
        session_id="session-input-once",
    )
    assert response.status == "done"
    requests = RunEvidenceStore(project / ".reins" / "data").list_records(
        session_id="session-input-once", run_id=response.run_id, kind="attempt_request"
    )
    assert len(requests) == 2
    for record in requests:
        body = record["payload"]["request"]
        assert json.dumps(body["messages"], ensure_ascii=False).count(message) == 1


def test_prompt_snapshot_is_reused_without_duplicate_fulltext_files(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"final","content":"first"}',
            '{"type":"final","content":"sediment reflection"}',
            '{"type":"final","content":"second"}',
            '{"type":"final","content":"sediment reflection"}',
        ]
    )
    loop, context, _store = _build_loop(project, client)

    assert loop.run(context) is State.DONE
    second_context = RunContext(
        task_id=context.task_id,
        trigger=Trigger.USER,
        payload={"message": "test goal"},
        capability_lease=context.capability_lease,
        session_id=context.session_id,
        segment_id=f"user-{context.task_id}-2",
    )
    second_loop = AgentLoop(
        project / ".reins" / "data",
        llm_client=client,
        tool_registry=build_tool_registry(
            repo_root=project,
            data_root=project / ".reins" / "data",
        ),
    )

    assert second_loop.run(second_context) is State.DONE

    data_root = project / ".reins" / "data"
    first_request = _evidence(project, context, "attempt_request")[0]["sources"][
        "composition"
    ]
    second_request = _evidence(project, second_context, "attempt_request")[0][
        "sources"
    ]["composition"]
    snapshot = second_request["stable_prompt_snapshot"]
    from runtime.session_state import SessionStateStore

    state = SessionStateStore(data_root).load(context.session_id)

    assert first_request["stable_prompt_snapshot"]["reused"] is False
    assert snapshot["reused"] is True
    assert snapshot["path"] == ""
    assert not (data_root / "sessions").exists()
    assert state is not None
    assert state.stable_prompt_hash == snapshot["hash"]


def test_parse_failure_writes_evidence_and_error_log_without_summary_pollution(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        ["not json at all", "not json at all"], protocol_mode="text_json"
    )
    loop, context, store = _build_loop(project, client)
    store.update_summary(context.task_id or "", "user wanted project summary")

    assert loop.run(context) is State.FAILED

    parsed = _evidence(project, context, "model_plan")[0]
    errors = _evidence(project, context, "error")
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    llm_fact = next(fact for fact in facts if fact.get("event") == "llm:response")

    assert parsed["success"] is False
    assert parsed["error"]["category"] == "invalid_model_protocol"
    assert parsed["error"]["raw_response_path"].startswith("evidence:")
    assert errors[-1]["category"] == "invalid_model_protocol"
    error_messages = " ".join(e.get("message", "") for e in errors)
    assert (
        "invalid json" in error_messages or "invalid_model_protocol" in error_messages
    )
    assert store.read_summary(context.task_id or "") == "user wanted project summary"
    assert llm_fact["summary"]["evidence"]["errors"].startswith("evidence:")


def test_native_tool_call_evidence_preserves_raw_tool_call(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_native_tool_then_final(
        [ToolCallPart("call-native-1", "list", {"path": "tools"})],
        "listed tools",
    )
    loop, context, _store = _build_loop(project, client)

    assert loop.run(context) is State.DONE

    response = _evidence(project, context, "attempt_response")[0]
    parsed = _evidence(project, context, "model_plan")[0]

    assert response["response"]["tool_calls"][0]["id"] == "call-native-1"
    assert response["response"]["tool_calls"][0]["function"]["name"] == "list"
    assert parsed["has_run_tools"] is True


def test_real_run_evidence_is_rebuildable_from_ledger(tmp_path: Path) -> None:
    """真实运行证据可从 Ledger 重建。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无
    """
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"ledger evidence done"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _store = _build_loop(project, client)

    assert loop.run(context) is State.DONE

    data_root = project / ".reins" / "data"
    view = build_run_evidence_view(
        LedgerStore(data_root).read_run_events(context.run_id)
    )

    assert view.model_requests
    assert view.model_requests[0]["payload"]["model"] == "stub-model"
    assert view.model_requests[0]["payload"]["context_segment_names"]
    assert view.context_segments
    assert view.context_segments[0]["payload"]["segments"]
    assert [row["event"] for row in view.tool_events] == [
        "tool.requested",
        "tool.completed",
    ]
    assert view.lifecycle_events[-1]["payload"]["status"] == "done"


def test_empty_provider_response_writes_error_log(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = _EmptyThenFinalClient()
    loop, context, store = _build_loop(project, client)
    store.update_summary(context.task_id or "", "keep this intent")

    assert loop.run(context) is State.DONE

    errors = _evidence(project, context, "error")
    assert any(error["category"] == "empty_response" for error in errors)
    assert store.read_summary(context.task_id or "") == "recovered from empty response"


def test_invalid_input_tool_error_requests_replan(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"reported failure"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, store = _build_loop(project, client)
    store.update_summary(context.task_id or "", "inspect tools")

    def fail_tool(*args: object, **kwargs: object) -> ToolError:
        return ToolError(ToolErrorCategory.INVALID_INPUT, "bad path", retryable=False)

    registry = build_tool_registry(
        repo_root=project,
        data_root=project / ".reins" / "data",
    )
    definition = registry.get("list")
    assert definition is not None
    definition.executor = fail_tool
    registry.replace(definition)
    loop.tool_registry = registry

    assert loop.run(context) is State.DONE

    errors = _evidence(project, context, "error")
    recovery_errors = [row for row in errors if row.get("recovery_action")]
    assert recovery_errors[-1]["category"] == "invalid_tool_arguments"
    assert recovery_errors[-1]["stage"] == "tool"
    assert recovery_errors[-1]["recovery_action"] == "replan"
    assert "Tool likely misused" in recovery_errors[-1]["message"]
    assert store.read_summary(context.task_id or "") == "reported failure"


def test_tool_argument_error_is_returned_to_next_model_turn(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = _ToolRecoveryClient()
    loop, context, _store = _build_loop(project, client)

    assert loop.run(context) is State.DONE

    observation = loop.tool_history[-1]["observation"]
    assert observation["tool_name"] == "list"
    assert observation["error_type"] == "invalid_tool_arguments"
    assert observation["retryable"] is True
    assert observation["args_summary"] == {}


def test_model_tool_argument_error_is_returned_to_next_model_turn(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"file_read","arguments":{}}',
            '{"type":"final","content":"asked for a valid path"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _store = _build_loop(project, client)

    assert loop.run(context) is State.DONE

    observation = loop.tool_history[-1]["observation"]
    assert observation["tool_name"] == "file_read"
    assert observation["error_type"] == "invalid_tool_arguments"
    assert observation["retryable"] is True
    assert "file_read" in str(observation["args_summary"])


def test_missing_file_read_returns_replan_observation(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"file_read","arguments":{"path":"missing.txt"}}',
            '{"type":"final","content":"changed strategy"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, store = _build_loop(project, client)
    store.update_summary(context.task_id or "", "keep original goal")

    assert loop.run(context) is State.DONE

    observations = [
        row["observation"]
        for row in loop.tool_history
        if isinstance(row.get("observation"), dict)
    ]
    assert len(observations) == 1
    assert observations[0]["tool_name"] == "file_read"
    assert observations[0]["error_type"] == "invalid_tool_arguments"
    assert observations[0]["retryable"] is True
    assert observations[0]["replan_required"] is True
    assert "path not found: missing.txt" in str(observations[0]["error_message"])
    errors = _evidence(project, context, "error")
    recovery_actions = [
        row.get("recovery_action") for row in errors if row.get("recovery_action")
    ]
    assert recovery_actions[-1] == "replan"
    assert "retry" not in recovery_actions
    assert store.read_summary(context.task_id or "") == "changed strategy"


def test_tool_execution_error_is_returned_to_next_model_turn(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"changed strategy"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _store = _build_loop(project, client)

    def fail_tool(*args: object, **kwargs: object) -> ToolError:
        return ToolError(ToolErrorCategory.TIMEOUT, "temporary timeout", retryable=True)

    definition = loop.tool_registry.get("list")
    assert definition is not None
    definition.executor = fail_tool
    loop.tool_registry.replace(definition)

    assert loop.run(context) is State.DONE
    assert loop.tool_history[-1]["observation"]["error_type"] == "tool_execution_error"
    assert loop.tool_history[-1]["observation"]["retryable"] is True


def test_same_tool_error_budget_is_fed_back_without_stopping_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重复工具失败及额度耗尽进入实际模型请求，模型仍能完成本轮。

    传参：tmp_path 为隔离目录，monkeypatch 仅替换本例退避等待；返回：无，核对执行与请求证据
    """
    repeats = 8
    expected_retry_delays = [2.0, 4.0, 8.0]
    waits: list[float] = []
    attempts: list[dict[str, object]] = []
    original_wait = CancellationToken.wait

    def wait_without_delay(token: CancellationToken, seconds: float) -> bool:
        """记录真实退避时长并保留取消检测；传参：取消信号、秒数；返回：是否已取消。"""
        waits.append(seconds)
        return original_wait(token, 0)

    # 1. 只替换本例的可取消退避，工具执行、失败和重试次数保持真实
    monkeypatch.setattr(CancellationToken, "wait", wait_without_delay)
    project = _project(tmp_path)
    client = from_test_sequence(
        ['{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}'] * repeats
        + ['{"type":"final","content":"done"}'],
        protocol_mode="text_json",
    )
    loop, context, store = _build_loop(project, client)
    store.update_summary(context.task_id or "", "keep this goal")

    def fail_tool(arguments: dict[str, object]) -> ToolError:
        """记录真实执行并返回可重试超时；传参：工具参数；返回：明确的超时错误。"""
        attempts.append(dict(arguments))
        return ToolError(ToolErrorCategory.TIMEOUT, "temporary timeout", retryable=True)

    definition = loop.tool_registry.get("list")
    assert definition is not None
    definition.executor = fail_tool
    loop.tool_registry.replace(definition)

    assert loop.run(context) is State.DONE
    assert len(attempts) == repeats * (len(expected_retry_delays) + 1)
    assert waits == expected_retry_delays * repeats

    # 2. 原始超时随工具回执进入每次续轮，额度耗尽也必须进入真正发送给 Provider 的请求
    requests = _evidence(project, context, "attempt_request")
    assert len(requests) == repeats + 1
    for request in requests[1:]:
        assert any(
            message["role"] == "tool" and "temporary timeout" in str(message["content"])
            for message in request["request"]["messages"]
        )
    feedback = [
        str(message["content"])
        for message in requests[-1]["request"]["messages"]
        if message["role"] == "user"
        and "runtime_observations" in str(message["content"])
    ]
    assert len(feedback) == 1
    assert "retry budget exhausted" in feedback[0]
    errors = _evidence(project, context, "error")
    assert any(error.get("message") == "timeout: temporary timeout" for error in errors)
    observations = [
        row["observation"] for row in loop.tool_history if "observation" in row
    ]
    assert len(observations) == repeats
    assert any(item.get("budget_exhausted") is True for item in observations)

    # 3. 错误反馈后仍由模型交付最终答案，摘要不被运行时错误替代
    assert store.read_summary(context.task_id or "") == "done"


def test_non_retryable_or_side_effect_tool_error_stops_without_repeat(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"file_write","arguments":{"path":"notes.txt","content":"hello"}}'
        ],
        protocol_mode="text_json",
    )
    loop, context, _store = _build_loop(project, client)
    context.payload["message"] = "edit notes.txt"

    def fail_tool(*args: object, **kwargs: object) -> ToolError:
        return ToolError(
            ToolErrorCategory.PERMISSION, "approval_denied", retryable=False
        )

    definition = loop.tool_registry.get("file_write")
    assert definition is not None
    definition.executor = fail_tool
    loop.tool_registry.replace(definition)

    assert loop.run(context) is State.FAILED
    observations = [
        row["observation"]
        for row in loop.tool_history
        if isinstance(row.get("observation"), dict)
    ]
    assert len(observations) == 1
    assert observations[0]["retryable"] is False
    recovery_actions = [
        row.get("recovery_action")
        for row in _evidence(project, context, "error")
        if row.get("recovery_action")
    ]
    assert recovery_actions[-1] == "fail"
    assert "retry" not in recovery_actions


def test_sync_runtime_ask_user_pauses_segment(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = _AskUserClient()
    loop, context, store = _build_loop(project, client)

    assert loop.run(context) is State.PAUSED
    assert client.continued is False
    assert "Which file should I update?" in store.read_summary(context.task_id or "")

    trajectory = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    terminal = [row for row in trajectory if row.get("event") == "run:lifecycle"][-1]
    assert terminal["lifecycle"] == "waiting_user"
    assert terminal["reason"] == "awaiting user input"


def test_empty_response_recovery_pauses_on_llm_failure_budget(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = _EmptyResponseClient()
    loop, context, store = _build_loop(project, client)
    store.update_summary(context.task_id or "", "keep empty response intent")

    assert loop.run(context) is State.PAUSED

    observations = [
        row["observation"]
        for row in loop.tool_history
        if isinstance(row.get("observation"), dict)
    ]
    assert observations
    assert observations[-1]["error_type"] == "empty_response"
    assert observations[-1]["retryable"] is True
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    lifecycle = [row for row in facts if row.get("event") == "run:lifecycle"][-1]
    assert lifecycle["lifecycle"] == "paused"
    assert lifecycle["reason"] == "segment llm failure budget hit"
    summary = store.read_summary(context.task_id or "")
    assert summary.startswith("SEGMENT_PAUSED: segment llm failure budget hit")


def test_run_evidence_redacts_secret_values(tmp_path: Path) -> None:
    project = _project(tmp_path)
    data_root = project / ".reins" / "data"
    from runtime.run_evidence import RunEvidenceStore

    store = RunEvidenceStore(data_root)
    reference = store.write_record(
        session_id="session-1",
        run_id="run-1",
        kind="probe",
        source_id="request-1",
        payload={
            "api_key": "sk-abc12345678901234567890",
            "message": "Authorization: Bearer abcdefghijklmnopqrstuv",
        },
    )

    payload = store.read_reference(reference)
    assert payload["api_key"] == "<redacted>"
    assert "Bearer <redacted>" in payload["message"]
    assert "abcdefghijklmnopqrstuv" not in payload["message"]


def test_run_evidence_keeps_token_count_fields(tmp_path: Path) -> None:
    project = _project(tmp_path)
    data_root = project / ".reins" / "data"
    from runtime.run_evidence import RunEvidenceStore

    store = RunEvidenceStore(data_root)
    reference = store.write_record(
        session_id="session-1",
        run_id="run-1",
        kind="probe",
        source_id="response-1",
        payload={
            "completion_tokens": 42,
            "prompt_tokens": 100,
            "total_tokens": 142,
            "access_token": "real-access-secret",
            "refresh_token": "real-refresh-secret",
        },
    )

    payload = store.read_reference(reference)
    assert payload["completion_tokens"] == 42
    assert payload["prompt_tokens"] == 100
    assert payload["total_tokens"] == 142
    assert payload["access_token"] == "<redacted>"
    assert payload["refresh_token"] == "<redacted>"


def test_failed_model_error_preserves_session_summary(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        ["not json at all", "not json at all"], protocol_mode="text_json"
    )
    loop, context, _store = _build_loop(project, client)
    from runtime.session_state import SessionState, SessionStateStore

    session_store = SessionStateStore(project / ".reins" / "data")
    session_store.save(
        SessionState(session_id=context.session_id, summary="keep session intent")
    )

    assert loop.run(context) is State.FAILED

    state = session_store.load(context.session_id)
    assert state is not None
    assert state.summary == "keep session intent"


def _history_shape(history: tuple[AgentMessage, ...]) -> list[tuple[str, str]]:
    """把 typed 对话历史抽成 (消息种类, 正文) 列表，绕开随机 message_id 做断言。

    作者：LKX
    时间：2026-08-31 14:09:39
    传参：history 为 model_context 里的 canonical 消息序列，每条只含一个 TextPart
    返回：按原顺序排列的 (kind, text) 列表
    """
    shape: list[tuple[str, str]] = []
    for message in history:
        # 每条消息只有一个文本块，取不到就是历史内容块被改形状了，让 assert 直接指出来
        assert len(message.content) == 1
        part = message.content[0]
        assert isinstance(part, TextPart)
        shape.append((message.kind, part.text))
    return shape


def test_model_context_injects_conversation_history_for_evidence_task(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = _ContextCaptureClient()
    loop, context, _store = _build_loop(project, client)
    # 历史只由唯一消息 owner 按 session 提供，不再经 TaskStore/Ledger 造对话
    data_root = project / ".reins" / "data"
    append_user_message(data_root, context.session_id, "test goal")
    append_assistant_message(data_root, context.session_id, "previous answer")

    assert loop.run(context) is State.DONE

    assert len(client.contexts) == 1
    model_context = client.contexts[0]
    history = model_context["conversation_history"]
    assert isinstance(history, tuple)
    # 历史进 model_context 时已是 canonical typed 消息，不再是 role/content 的 Wire dict
    assert [type(message) for message in history] == [UserMessage, AssistantMessage]
    assert _history_shape(history) == [
        ("user", "test goal"),
        ("assistant", "previous answer"),
    ]
    assert model_context["session_id"] == context.session_id
    assert model_context["run_id"] == context.run_id


def test_context_summary_read_failure_stops_before_model_call(
    monkeypatch, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    client = _ContextCaptureClient()
    loop, context, _store = _build_loop(project, client)  # type: ignore[arg-type]

    def fail_summary_read(self: LedgerStore, task_id: str) -> object:
        del self, task_id
        raise OSError("summary store unavailable")

    monkeypatch.setattr(LedgerStore, "read_task_events", fail_summary_read)

    assert loop.run(context) is State.FAILED
    assert client.contexts == []

    errors = _evidence(project, context, "error")
    assert errors[-1]["category"] == "context_summary_read_failed"
    assert errors[-1]["stage"] == "context"


def test_conversation_history_read_failure_stops_before_model_call(
    monkeypatch, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    client = _ContextCaptureClient()
    loop, context, _store = _build_loop(project, client)  # type: ignore[arg-type]

    def fail_conversation_read(data_root: Path | str, session_id: str) -> object:
        del data_root, session_id
        raise OSError("conversation store unavailable")

    def summary_ok(
        self: ProductionContextBuilder,
        task_id: str,
        *,
        store: TaskStore | None = None,
    ) -> dict[str, str]:
        del self, task_id, store
        return {"intent": "", "resume_hint": "", "progress": "", "summary": ""}

    monkeypatch.setattr(ProductionContextBuilder, "task_summary_context", summary_ok)
    monkeypatch.setattr(
        "runtime.agent_loop.materialize_messages", fail_conversation_read
    )
    monkeypatch.setattr("context.engine.materialize_messages", fail_conversation_read)

    assert loop.run(context) is State.FAILED
    assert client.contexts == []

    errors = _evidence(project, context, "error")
    assert errors[-1]["category"] == "conversation_history_read_failed"
    assert errors[-1]["stage"] == "context"


def test_agent_loop_toolset_policy_uses_runtime_config_and_loop_registry(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("custom tool")
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"custom_tool","arguments":{"value":"x"}}',
            '{"type":"final","content":"custom done"}',
        ],
        protocol_mode="text_json",
    )
    registry = ToolRegistry()
    registry.register(_custom_tool())
    loop = AgentLoop(
        data_root,
        llm_client=client,
        tool_registry=registry,
        runtime_config={"toolsets_enabled": ["custom"]},
    )
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": record.goal},
        capability_lease=from_trigger("user", task_id=record.task_id),
        segment_id=f"user-{record.task_id}",
    )

    assert loop.run(context) is State.DONE
    assert loop.last_output == "custom done"
    request = _evidence(project, context, "attempt_request")[0]["sources"][
        "composition"
    ]
    selected = request["tool_selection"]["selected"]
    assert [item["name"] for item in selected] == ["custom_tool"]
    assert request["tool_selection"]["policy_source"] == "config"


def test_real_llm_client_allowed_actions_limits_current_request(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"file_write","arguments":{"path":"notes.txt","content":"hello"}}',
        ],
        protocol_mode="text_json",
        allowed_actions=["file_read"],
    )
    loop, context, _store = _build_loop(project, client)

    assert loop.run(context) is State.FAILED

    request = _evidence(project, context, "attempt_request")[0]["sources"][
        "composition"
    ]
    parsed = _evidence(project, context, "model_plan")[0]
    selected = request["tool_selection"]["selected"]

    assert "notes.txt" not in {path.name for path in project.iterdir()}
    assert [item["name"] for item in selected] == ["file_read"]
    assert parsed["error"]["category"] == "invalid_tool_arguments"
    assert "not allowed in current request" in parsed["error"]["raw_summary"]


def test_context_segments_marks_system_reminder_ephemeral(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    store = TaskStore(data_root)
    record = store.create_task("goal")
    client = _ContextCaptureClient()
    loop = AgentLoop(data_root, llm_client=client)  # type: ignore[arg-type]
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": "goal"},
        capability_lease=from_trigger("user", task_id=record.task_id),
    )

    # _call_model 现在是生成器（边生成边发增量），不排空则函数体一行都不执行
    loop._prepare_turn_core(context).store.close()
    list(
        loop._call_model(
            "goal",
            context,
            None,
            system_reminder="[system_reminder]decide[/system_reminder]",
            watchdog=Watchdog(context.capability_lease),
        )
    )

    facts = RunFactStore(data_root).read_run(context.run_id)
    segment_fact = next(
        fact for fact in facts if fact.get("event") == "context:segments"
    )
    reminder = next(
        item for item in segment_fact["segments"] if item["name"] == "system_reminder"
    )
    assert reminder["layer"] == "ephemeral"


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    tools_dir = project / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "alpha.py").write_text("print('hi')\n", encoding="utf-8")
    return project


def _custom_tool() -> ToolDefinition:
    return ToolDefinition(
        name="custom_tool",
        description="Custom test tool.",
        parameters={
            "value": {
                "type": "string",
                "description": "Value to echo",
                "required": True,
            }
        },
        toolset="custom",
        risk_level=TOOL_RISK_SAFE,
        readonly=True,
        target_scope_rule=TARGET_SCOPE_LOGICAL,
        source=TOOL_SOURCE_BUILTIN,
        model_visible=True,
        idempotent=IDEMPOTENT_YES,
        executor=lambda args: {"content": f"custom {args.get('value', '')}"},
    )


def _build_loop(
    project: Path, llm_client: RealLLMClient
) -> tuple[AgentLoop, RunContext, TaskStore]:
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("test goal")
    lease = from_trigger(
        "user",
        task_id=record.task_id,
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [
                    str(project),
                    str(data_root),
                    str(project / ".reins" / "workspace"),
                ],
                "write": [str(data_root), str(project / ".reins" / "workspace")],
                "deny_read": ["*.pem", "*.key", ".env"],
            },
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {
                "enabled": True,
                "profile": "default",
                "deny_domains": [],
                "headless": True,
            },
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
    )
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(data_root, llm_client=llm_client, tool_registry=registry)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": record.goal},
        capability_lease=lease,
        segment_id=f"user-{record.task_id}",
    )
    return loop, context, store


def _evidence(project: Path, context: RunContext, kind: str) -> list[dict[str, object]]:
    """按规范身份读取运行证据；参数：项目、运行及类别；返回：持久有序内容。"""
    store = RunEvidenceStore(project / ".reins" / "data")
    return [
        row["payload"]
        for row in store.list_records(
            session_id=context.session_id, run_id=context.run_id, kind=kind
        )
    ]


class _ContextCaptureClient:
    def __init__(self) -> None:
        self.contexts: list[dict[str, object]] = []

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task
        assert isinstance(context, dict)
        self.contexts.append(context)
        return LLMPlan(final_output="done")

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        del task, run_tools_result, context
        raise AssertionError("unexpected tool continuation")


def test_toolset_runtime_config_reaches_model_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    config = home / ".reins" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("toolsets:\n  enabled:\n    - web\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    project = _project(tmp_path)
    client = _ContextCaptureClient()
    loop, context, _store = _build_loop(project, client)  # type: ignore[arg-type]

    assert loop.run(context) is State.DONE

    policy = client.contexts[0]["toolset_policy"]
    assert isinstance(policy, dict)
    assert policy["source"] == "config"
    assert policy["enabled_toolsets"] == ["web"]


class _PolicyBypassClient:
    def __init__(self) -> None:
        self.results: list[RunToolsResult] = []

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task, context
        return LLMPlan(
            run_tools_request=RunToolsRequest(
                action="list",
                tool_name="list",
                arguments={"path": "tools"},
            )
        )

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        del task, context
        self.results.append(run_tools_result)
        return LLMPlan(final_output="stopped after policy denial")


def test_agent_loop_rechecks_tool_policy_before_execution(tmp_path: Path) -> None:
    from runtime.session_state import SessionState, SessionStateStore

    project = _project(tmp_path)
    client = _PolicyBypassClient()
    loop, context, _store = _build_loop(project, client)  # type: ignore[arg-type]
    data_root = project / ".reins" / "data"
    SessionStateStore(data_root).save(
        SessionState(session_id=context.session_id, toolsets_enabled=["web"])
    )

    def must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("策略未放行的工具不得进入执行体")

    definition = loop.tool_registry.get("list")
    assert definition is not None
    definition.executor = must_not_run
    loop.tool_registry.replace(definition)

    assert loop.run(context) is State.DONE
    assert client.results
    denied = client.results[0]
    assert denied.status != "ok"
    assert "not allowed" in str(denied.error)
    # 调用意图在副作用前就先落一张 request 事实，策略核对在派发时执行，
    # 因此判据是这张 request 对应的 response 记为拒绝，而不是整条事实缺席
    facts = RunFactStore(data_root).read_run(context.run_id)
    responses = [row["tool"] for row in facts if row.get("event") == "tool:response"]
    assert responses
    assert responses[0]["status"] != "ok"
    assert "not allowed" in str(responses[0]["error"])


class _ToolRecoveryClient:
    def __init__(self) -> None:
        pass

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task, context
        return LLMPlan(
            run_tools_request=RunToolsRequest(
                action="list",
                tool_name="list",
                arguments={},
            )
        )

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        del task, run_tools_result, context
        return LLMPlan(final_output="fixed after observing tool error")


class _EmptyResponseClient:
    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task, context
        return self._empty()

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        del task, run_tools_result, context
        return self._empty()

    def _empty(self) -> LLMPlan:
        error = ModelError.create(
            category="empty_response",
            summary="provider returned empty response",
            raw_summary="empty text",
            stage="transport",
        )
        return LLMPlan(final_output=error.render_output(), model_error=error)


class _AskUserClient:
    def __init__(self) -> None:
        self.continued = False

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task, context
        return LLMPlan(
            run_tools_request=RunToolsRequest(
                action="ask_user",
                tool_name="ask_user",
                arguments={
                    "question": "Which file should I update?",
                },
            )
        )

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        del task, run_tools_result, context
        self.continued = True
        return LLMPlan(final_output="should not continue")


class _EmptyThenFinalClient(_EmptyResponseClient):
    def __init__(self) -> None:
        self.calls = 0

    def _empty(self) -> LLMPlan:
        self.calls += 1
        if self.calls == 1:
            return super()._empty()
        return LLMPlan(final_output="recovered from empty response")
