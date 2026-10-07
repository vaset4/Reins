from __future__ import annotations
from scripts.testing.llm import from_test_sequence

import json
from difflib import unified_diff
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from runtime.agent_loop import AgentLoop
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_messages import append_user_message, read_history_rows
from runtime.session_state import SessionState, SessionStateStore
from runtime.stream_events import (
    LeaseSnapshot,
    SegmentPaused,
    StreamEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.types import ToolError, ToolErrorCategory

# 两条路径共用同一段名：段名进提示词，参与 token 估算，必须逐字节相同
_PARITY_SEGMENT_ID = "user-parity"


def test_run_and_run_stream_match_durable_tool_then_final(tmp_path: Path) -> None:
    responses = (
        '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
        '{"type":"final","content":"tools/ contains alpha.py"}',
    )

    sync = _run_sync(tmp_path, responses)
    stream = _run_stream(tmp_path, responses)

    _assert_live_only_events(stream.events)
    _assert_snapshot_parity(sync.snapshot, stream.snapshot)


def test_run_and_run_stream_match_durable_convergence_reminder(
    tmp_path: Path,
) -> None:
    responses = ('{"type":"final","content":"ready to continue"}',)

    sync = _run_sync(
        tmp_path, responses, message="继续", setup=_seed_convergence_reminder_state
    )
    stream = _run_stream(
        tmp_path, responses, message="继续", setup=_seed_convergence_reminder_state
    )

    assert not any(isinstance(event, SegmentPaused) for event in stream.events)
    _assert_convergence_reminder_snapshot(sync.snapshot)
    _assert_convergence_reminder_snapshot(stream.snapshot)
    _assert_snapshot_parity(sync.snapshot, stream.snapshot)


def test_run_and_run_stream_match_durable_tool_error_recovery(
    tmp_path: Path,
) -> None:
    responses = (
        '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
        '{"type":"final","content":"changed strategy"}',
    )

    sync = _run_sync(tmp_path, responses, setup=_make_list_timeout)
    stream = _run_stream(tmp_path, responses, setup=_make_list_timeout)

    assert any(
        isinstance(event, ToolExecutionCompleted) and event.is_error
        for event in stream.events
    )
    _assert_snapshot_parity(sync.snapshot, stream.snapshot)


@dataclass(frozen=True, slots=True)
class _RunResult:
    events: list[StreamEvent]
    snapshot: dict[str, object]


SetupFn = Callable[[AgentLoop, RunContext, TaskStore], None]


def _run_sync(
    tmp_path: Path,
    responses: tuple[str, ...],
    *,
    message: str = "test goal",
    setup: SetupFn | None = None,
) -> _RunResult:
    loop, context, store, data_root = _build_loop(
        tmp_path, "sync", responses, message=message
    )
    if setup is not None:
        setup(loop, context, store)

    loop.run(context)
    return _RunResult(events=[], snapshot=_snapshot(loop, context, store, data_root))


def _run_stream(
    tmp_path: Path,
    responses: tuple[str, ...],
    *,
    message: str = "test goal",
    setup: SetupFn | None = None,
) -> _RunResult:
    loop, context, store, data_root = _build_loop(
        tmp_path, "stream", responses, message=message
    )
    if setup is not None:
        setup(loop, context, store)

    events = list(loop.run_stream(context))

    return _RunResult(
        events=events, snapshot=_snapshot(loop, context, store, data_root)
    )


def _build_loop(
    tmp_path: Path,
    label: str,
    responses: tuple[str, ...],
    *,
    message: str,
) -> tuple[AgentLoop, RunContext, TaskStore, Path]:
    project = tmp_path / "repo"
    tools_dir = project / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "alpha.py").write_text("print('hi')\n", encoding="utf-8")
    data_root = tmp_path / label / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("test goal")
    client = from_test_sequence(list(responses), protocol_mode="text_json")
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(data_root, llm_client=client, tool_registry=registry)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": message},
        capability_lease=from_trigger(
            "user",
            task_id=record.task_id,
            capabilities=_lease_capabilities(project, data_root),
        ),
        # 段名会进提示词，两侧长度不同会让 token 估算差出 1，把本该等价的预算字段
        # 变成假漂移；两侧已靠各自的数据目录隔离，段名不必区分
        segment_id=_PARITY_SEGMENT_ID,
    )
    # 用户输入落到唯一消息 owner；session_id 由 RunContext 定稿，故排在其后
    append_user_message(data_root, context.session_id, record.goal)
    return loop, context, store, data_root


def _lease_capabilities(project: Path, data_root: Path) -> dict[str, object]:
    workspace = project / ".reins" / "workspace"
    return {
        "fs": {
            "project_root": str(project),
            "read": [str(project), str(data_root), str(workspace)],
            "write": [str(data_root), str(workspace)],
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
    }


def _seed_convergence_reminder_state(
    _loop: AgentLoop,
    context: RunContext,
    store: TaskStore,
) -> None:
    record = store.require_task(context.storage_task_id)
    SessionStateStore(_loop.data_root).save(
        SessionState(
            session_id=context.session_id,
            consecutive_readonly_count=15,
            original_user_goal=record.goal,
            hint_injection_count=2,
        )
    )


def _assert_convergence_reminder_snapshot(snapshot: dict[str, object]) -> None:
    assert snapshot["last_output"] == "ready to continue"
    facts = snapshot["facts"]
    assert isinstance(facts, list)
    assert not any(
        isinstance(row, dict) and row.get("event") == "convergence_stop_hint_injected"
        for row in facts
    )
    assert not any(
        isinstance(row, dict) and row.get("event") == "convergence_segment_paused"
        for row in facts
    )
    lifecycles = [
        row
        for row in facts
        if isinstance(row, dict) and row.get("event") == "run:lifecycle"
    ]
    assert lifecycles
    assert lifecycles[-1].get("lifecycle") == "done"


def _make_list_timeout(
    loop: AgentLoop,
    _context: RunContext,
    _store: TaskStore,
) -> None:
    definition = loop.tool_registry.get("list")
    assert definition is not None

    def fail_tool(_args: dict[str, object]) -> ToolError:
        return ToolError(ToolErrorCategory.TIMEOUT, "temporary timeout", retryable=True)

    definition.executor = fail_tool
    # get() 返回的是定义副本，改副本不影响目录，必须显式发布回去故障才注入得进去
    loop.tool_registry.replace(definition)


def _snapshot(
    loop: AgentLoop,
    context: RunContext,
    store: TaskStore,
    data_root: Path,
) -> dict[str, object]:
    facts = RunFactStore(data_root).read_run(context.run_id)
    assert loop.state is not None
    return {
        "state": loop.state.value,
        "last_output": loop.last_output,
        "facts": facts,
        "conversation": read_history_rows(data_root, context.session_id, limit=50),
        "summary": asdict(store.read_summary_layers(context.storage_task_id)),
        "tool_history": list(loop.tool_history),
        "session_state": _session_state_payload(data_root, context.session_id),
    }


def _session_state_payload(data_root: Path, session_id: str) -> dict[str, object]:
    state = SessionStateStore(data_root).load(session_id)
    assert state is not None
    payload = asdict(state)
    return payload


def _assert_snapshot_parity(sync: dict[str, object], stream: dict[str, object]) -> None:
    """逐字段比较持久结果并输出可定位差异；参数：两个真实入口快照；返回：无。"""
    left = json.dumps(_normalize(sync), ensure_ascii=False, sort_keys=True, indent=2)
    right = json.dumps(_normalize(stream), ensure_ascii=False, sort_keys=True, indent=2)
    difference = "\n".join(
        unified_diff(
            left.splitlines(), right.splitlines(), fromfile="run", tofile="run_stream"
        )
    )
    assert left == right, difference


def _normalize(value: object, *, key: str = "") -> object:
    if isinstance(value, dict):
        if key == "evidence" and "definition_version" in value and "repeats" in value:
            # 两次独立运行的注册表实例不同，其动作哈希天然不同；保留版本、计数和决策核对
            assert isinstance(value["action"], str) and len(value["action"]) == 64
            value = {**value, "action": "<action-fingerprint>"}
        return {
            item_key: _normalize(item_value, key=item_key)
            for item_key, item_value in value.items()
            if item_key not in _NATURALLY_DIFFERENT_KEYS
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        if key.endswith("_id") or key in _ID_KEYS:
            return "<id>"
        if key.endswith("_at") or key == "ts":
            return "<time>"
        return _normalize_generated_text(value)
    return value


def _normalize_generated_text(value: str) -> str:
    normalized = value
    for prefix in (
        "session-",
        "run-",
        "request-",
        "attempt-",
        "op-",
        "msg-",
        "call-",
        "2026-",
    ):
        if prefix in normalized:
            normalized = _replace_token_prefix(normalized, prefix)
    # 目录实例 id 每次构造都换（tool_registry.py:289 的 uuid4），两侧天然不同。
    # 只掩 uuid 本身，保留后面的修订号和工具名，否则「一侧少注册了一个工具」
    # 这类真漂移会被一起盖掉
    return _REGISTRY_INSTANCE_ID.sub("registry-<id>", normalized)


def _replace_token_prefix(value: str, prefix: str) -> str:
    parts: list[str] = []
    for token in value.split(prefix):
        if not parts:
            parts.append(token)
            continue
        suffix = _trim_identifier_suffix(token)
        parts.append(f"<{prefix.rstrip('-')}>{suffix}")
    return "".join(parts)


def _trim_identifier_suffix(value: str) -> str:
    index = 0
    while index < len(value) and (value[index].isalnum() or value[index] in "-_"):
        index += 1
    return value[index:]


# 形如 registry-<32位hex>，出现在 registry_version/definition_version 字段，
# 也嵌在 conversation 正文的 JSON 串里（字符串内部，结构化归一化进不去）
_REGISTRY_INSTANCE_ID = re.compile(r"registry-[0-9a-f]{32}")


def _assert_live_only_events(events: list[StreamEvent]) -> None:
    assert any(isinstance(event, LeaseSnapshot) for event in events)
    assert any(isinstance(event, ToolExecutionStarted) for event in events)
    assert any(isinstance(event, ToolExecutionCompleted) for event in events)
    serialized = json.dumps(
        _normalize([asdict(event) for event in events]), sort_keys=True
    )
    durable = json.dumps(
        _normalize([event for event in events]), sort_keys=True, default=str
    )
    assert "ToolExecutionStarted" not in serialized
    assert durable


_NATURALLY_DIFFERENT_KEYS = {
    "checkpoint_id",
    # Provider timing is wall-clock telemetry, not durable behavioral output.
    "elapsed_ms",
    "elapsed_seconds",
    "stable_prompt_hash",
}

_ID_KEYS = {
    "run_id",
    "session_id",
    "task_id",
    "focus_task_id",
    "compatibility_task_id",
    "segment_id",
    "call_id",
    "tool_call_id",
    "id",
}
