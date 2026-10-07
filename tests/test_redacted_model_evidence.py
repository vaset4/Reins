"""验证秘密替换经过真实模型请求、整批审批和持久证据链。

作者：xxx
时间：2026-09-24 22:00:00
"""

from contextlib import closing
import json
import re

import pytest

from approval import ApprovalDecision
from approval.batch_types import ApprovalChoice, BatchDecision
from approval.session import ApprovalMode
from llm.client import RealLLMClient
from scripts.testing.llm import (
    _ScriptedAdapter,
    _ScriptedTurn,
    _test_config,
    _test_connection,
    _test_model_registry,
)
from llm.messages import ToolCallPart
from llm.provider_adapter import AdapterRegistry
from runtime.agent_loop import AgentLoop
from runtime.workspaces import WorkspaceStore
from runtime.default_capabilities import build_local_agent_capabilities
from runtime.lease import from_trigger
from runtime.session_messages import append_user_message
from runtime.shared_budget import BudgetOwner
from runtime.types import RunContext, Trigger
from runtime.watchdog import Watchdog
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry


class RecordingAdapter(_ScriptedAdapter):
    """供应商响应可控，请求仍由生产组装器生成。"""

    def __init__(self, turns):
        """保存响应和请求序列；传参：响应；返回：无。"""
        super().__init__(turns)
        self.requests = []

    def stream(self, request, **kwargs):
        """记录实际将发送的请求；传参：请求和模型身份；返回：供应商事件流。"""
        self.requests.append(request)
        return super().stream(request, **kwargs)


def _run(root, monkeypatch, scenario, *, protocol="native_tool_calls"):
    """执行真实主循环及审批；传参：隔离目录、替换器、场景；返回：文件、决定和下一次请求。"""
    data = root / "data"
    with closing(TaskStore(data)) as tasks:
        task = tasks.create_task("按批准的内容更新配置")
    lease = from_trigger(
        "user",
        task_id=task.task_id,
        capabilities=build_local_agent_capabilities(root, data),
    )
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": task.goal},
        capability_lease=lease,
    )
    WorkspaceStore(data).bind_session(context.session_id, root)
    registry = build_tool_registry(repo_root=root, data_root=data)
    path = root / ".env"
    prompts = []
    args = {"path": ".env"}
    if scenario in {"create", "create_alias"}:
        args["content"] = "TOKEN=NEW_SYNTHETIC_PRIVATE\nPORT=4000\n"
        if scenario == "create_alias":
            args = {"filePath": ".env", "newContent": args["content"]}
    else:
        path.write_bytes(b"TOKEN=OLD_SYNTHETIC_PRIVATE\nPORT=3000\n")
        watchdog = Watchdog(
            lease,
            data_root=data,
            budget_owner=BudgetOwner(context.session_id, context.run_id, lease),
        )
        view = registry.execute_tool(
            "file_read", {"path": ".env"}, lease, watchdog=watchdog
        )
        token = re.search(r"<redacted:[0-9a-f]{32}>", view["content"]).group()
        replacement = (
            ["NEW_SYNTHETIC_PRIVATE"]
            if scenario == "invalid"
            else "NEW_SYNTHETIC_PRIVATE"
        )
        args.update(
            content=view["content"],
            view_id=view["meta"]["view_id"],
            expected_sha256=view["meta"]["content_sha256"],
            secret_replacements={token: replacement},
        )
        if scenario == "add_secret":
            args.pop("secret_replacements")
            args["content"] += "PASSWORD=NEW_SYNTHETIC_PRIVATE\n"

    def decide(batch):
        """一次决定全部秘密替换；传参：真实申请；返回：明确决定，必要时模拟外部编辑。"""
        prompts.append(batch)
        if scenario == "external":
            path.write_bytes(b"TOKEN=EXTERNAL_PRIVATE\n")
        if scenario == "cancel":
            return BatchDecision(cancelled=True)
        return BatchDecision(
            tuple(
                ApprovalChoice(
                    item.operation_id,
                    ApprovalDecision.DENY
                    if scenario == "deny"
                    else ApprovalDecision.ONCE,
                )
                for item in batch.requests
            )
        )

    monkeypatch.setattr("approval.batch._batch_backend", decide)
    monkeypatch.setattr("approval._config_path", lambda: data / "approval.yaml")
    monkeypatch.setenv("REINS_TRACE_LEVEL", "debug")
    append_user_message(data, context.session_id, task.goal)
    turns = (
        _ScriptedTurn(
            text="候选新值 NEW_SYNTHETIC_PRIVATE",
            thinking=("更换为 NEW_SYNTHETIC_PRIVATE",),
            calls=(ToolCallPart("write", "file_write", args),),
        ),
        _ScriptedTurn(text="已返回实际结果"),
    )
    if protocol == "text_json":
        turns = (
            _ScriptedTurn(
                text=json.dumps(
                    {"type": "run_tools", "tool": "file_write", "arguments": args}
                )
            ),
            _ScriptedTurn(
                text=json.dumps({"type": "final", "content": "已返回实际结果"})
            ),
        )
    adapter = RecordingAdapter(turns)
    client = RealLLMClient(
        _test_config(),
        adapter_registry=AdapterRegistry([adapter]),
        model_registry=_test_model_registry(),
        connection=_test_connection(),
        protocol_mode=protocol,
    )
    loop = AgentLoop(data, llm_client=client, tool_registry=registry)
    if scenario == "auto":
        loop.approval_session.set_mode(ApprovalMode.AUTO)
    if scenario == "save_failed":
        from runtime.ledger_writer import LedgerWriter

        record = LedgerWriter.record_once

        def fail_decision(writer, event):
            """模拟批准事实无法持久化；传参：写者和事件；返回：原写入结果或明确错误。"""
            if event.event == "approval.decided":
                raise OSError("approval storage interrupted")
            return record(writer, event)

        monkeypatch.setattr(LedgerWriter, "record_once", fail_decision)
    list(loop.run_stream(context))
    return path, prompts, adapter.requests, data


@pytest.mark.parametrize(
    "scenario",
    [
        "allow",
        "deny",
        "cancel",
        "external",
        "invalid",
        "create",
        "create_alias",
        "add_secret",
    ],
)
def test_secret_change_never_enters_model_requests_or_persistent_evidence(
    tmp_path, monkeypatch, scenario
):
    """批准才改文件，旧新秘密都不扩散到证据和下一轮模型；传参：目录、替换器、场景；返回：无。"""
    path, prompts, requests, data = _run(tmp_path, monkeypatch, scenario)
    assert len(requests) == 2
    assert bool(prompts) == (scenario != "invalid")
    for batch in prompts:
        assert all(item.force_confirmation for item in batch.requests)
    content = path.read_bytes()
    if scenario in {"allow", "create", "create_alias", "add_secret"}:
        assert b"NEW_SYNTHETIC_PRIVATE" in content
    elif scenario == "external":
        assert content == b"TOKEN=EXTERNAL_PRIVATE\n"
    else:
        assert content == b"TOKEN=OLD_SYNTHETIC_PRIVATE\nPORT=3000\n"
    for secret in (
        b"OLD_SYNTHETIC_PRIVATE",
        b"NEW_SYNTHETIC_PRIVATE",
        b"EXTERNAL_PRIVATE",
    ):
        assert secret.decode() not in repr(requests[1])
        assert secret.decode() not in repr(prompts)
        for file in data.rglob("*"):
            if file.is_file():
                assert secret not in file.read_bytes(), str(file.relative_to(data))


@pytest.mark.parametrize("scenario", ["allow", "create", "invalid"])
def test_text_protocol_protects_new_secrets_before_recording(
    tmp_path, monkeypatch, scenario
):
    """旧文本协议使用真实tool字段并在解析前过滤秘密；传参：隔离目录与场景；返回：无。"""
    path, prompts, requests, data = _run(
        tmp_path, monkeypatch, scenario, protocol="text_json"
    )
    assert len(requests) == 2
    assert bool(prompts) == (scenario != "invalid")
    assert (b"NEW_SYNTHETIC_PRIVATE" in path.read_bytes()) == (scenario != "invalid")
    for secret in (b"OLD_SYNTHETIC_PRIVATE", b"NEW_SYNTHETIC_PRIVATE"):
        assert secret.decode() not in repr(requests[1])
        for file in data.rglob("*"):
            if file.is_file():
                assert secret not in file.read_bytes(), str(file.relative_to(data))


@pytest.mark.parametrize("scenario", ["auto", "save_failed"])
def test_secret_changes_require_committed_confirmation_in_every_mode(
    tmp_path, monkeypatch, scenario
):
    """自动模式仍逐项确认，批准保存失败没有写入；传参：目录与场景；返回：无。"""
    path, prompts, _, _ = _run(tmp_path, monkeypatch, scenario)
    assert len(prompts) == 1 and all(
        item.force_confirmation for item in prompts[0].requests
    )
    assert (b"NEW_SYNTHETIC_PRIVATE" in path.read_bytes()) == (scenario == "auto")
