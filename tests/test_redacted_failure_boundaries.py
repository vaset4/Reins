"""敏感候选在发布、非法输入和供应商失败窗口的生产链路验收。

作者：xxx
时间：2026-09-24 23:45:00
"""

import json
from contextlib import contextmanager
from dataclasses import replace

import pytest

from approval import ApprovalDecision
from approval.session import ApprovalMode
from llm.client import RealLLMClient
from scripts.testing.llm import (
    _ScriptedTurn,
    _test_config,
    _test_connection,
    _test_model_registry,
)
from llm.messages import ToolCallPart
from llm.provider_adapter import AdapterRegistry
from llm.provider_result import ProviderError
from runtime.ledger_writer import LedgerWriter
from tests.support.approval import install_approval
from tests.test_file_batch_tracking import _runtime
from tests.test_redacted_model_evidence import RecordingAdapter

SECRET = "BOUNDARY_SYNTHETIC_PRIVATE"


@pytest.mark.parametrize(
    "failure", ["deny", "save_failed", "cancel_locked", "mode_locked"]
)
def test_new_config_failure_never_creates_parent_directory(
    tmp_path, monkeypatch, failure
):
    """批准前失败与锁内撤权/取消均不得创建目录；传参：目录、替换器和窗口；返回：无。"""
    from tools import redacted_files

    args = {"path": "new-folder/.env", "content": f"TOKEN={SECRET}\n"}
    loop, context, _ = _runtime(tmp_path, [ToolCallPart("write", "file_write", args)])
    install_approval(
        monkeypatch,
        lambda _request: (
            ApprovalDecision.DENY if failure == "deny" else ApprovalDecision.ONCE
        ),
    )
    monkeypatch.setattr("approval._config_path", lambda: tmp_path / "approval.yaml")
    original_lock = redacted_files.file_edit_lock
    lock_entries = []

    @contextmanager
    def stop_inside_lock(path):
        """在真实执行线程取得锁后变更边界；传参：目标；返回：文件锁上下文。"""
        with original_lock(path):
            lock_entries.append(path)
            if failure == "cancel_locked":
                loop.cancellation.cancel("cancel at file publication")
            elif failure == "mode_locked":
                loop.approval_session.set_mode(ApprovalMode.READ_ONLY)
            yield

    monkeypatch.setattr(redacted_files, "file_edit_lock", stop_inside_lock)
    record = LedgerWriter.record_once

    def fail_decision(writer, event):
        """只中断批准保存窗口；传参：事实写者和事件；返回：原写入结果或明确错误。"""
        if failure == "save_failed" and event.event == "approval.decided":
            raise OSError("approval storage interrupted")
        return record(writer, event)

    monkeypatch.setattr(LedgerWriter, "record_once", fail_decision)
    list(loop.run_stream(context))
    assert not (tmp_path / "new-folder").exists()
    records = loop.operations.for_session(context.session_id)
    writes = [row for row in records if row["call"]["tool_name"] == "file_write"]
    assert len(writes) == 1 and writes[0].get("result", {}).get("status") != "ok"
    if failure.endswith("_locked"):
        assert len(lock_entries) == 1
        error = writes[0]["result"]["error"]
        assert (
            "cancelled" in error
            if failure == "cancel_locked"
            else "current mode does not allow this write" in error
        )
    _assert_no_secret(tmp_path / "data")


class PartialFailureAdapter(RecordingAdapter):
    """完整工具块之后供应商失败，仍走真实响应装配与尝试保存。"""

    def stream(self, request, **kwargs):
        """将首轮终止改为供应商错误；传参：真实请求与连接；返回：事件流。"""
        events = list(super().stream(request, **kwargs))
        error = ProviderError(
            "invalid_provider_response",
            "stream_decode",
            False,
            f"failed after {SECRET}",
            "test",
            "test",
            self.api_family,
        )
        if len(self.requests) == 1:
            events[-1] = replace(
                events[-1], kind="response_error", error=error, stop_reason=None
            )
        return iter(events)


@pytest.mark.parametrize(
    "failure", ["malformed_text", "invalid_key", "nested_key", "native_partial"]
)
def test_invalid_or_partial_candidate_does_not_escape_into_evidence(
    tmp_path, monkeypatch, failure
):
    """破损协议和非法字典键不能扩大秘密传播，且不得写文件；传参：目录、替换器、场景；返回：无。"""
    args = {"path": ".env", "content": f"TOKEN={SECRET}\n"}
    if failure == "invalid_key":
        args["secret_replacements"] = {SECRET: "other synthetic value"}
    elif failure == "nested_key":
        args["secret_replacements"] = [{SECRET: "other synthetic value"}]
    call = ToolCallPart("write", "file_write", args)
    loop, context, _ = _runtime(tmp_path, [])
    protocol = "text_json" if failure == "malformed_text" else "native_tool_calls"
    first = (
        _ScriptedTurn(
            text=json.dumps(
                {"type": "run_tools", "tool": "file_write", "arguments": args}
            )[:-1]
        )
        if failure == "malformed_text"
        else _ScriptedTurn(calls=(call,))
    )
    final = _ScriptedTurn(
        text='{"type":"final","content":"收到失败"}'
        if failure == "malformed_text"
        else "收到失败"
    )
    adapter = (
        PartialFailureAdapter if failure == "native_partial" else RecordingAdapter
    )((first, final))
    loop.llm_client = RealLLMClient(
        _test_config(),
        adapter_registry=AdapterRegistry([adapter]),
        model_registry=_test_model_registry(),
        connection=_test_connection(),
        protocol_mode=protocol,
    )
    install_approval(monkeypatch, lambda _request: ApprovalDecision.ONCE)
    monkeypatch.setattr("approval._config_path", lambda: tmp_path / "approval.yaml")
    monkeypatch.setenv("REINS_TRACE_LEVEL", "debug")
    list(loop.run_stream(context))
    assert not (tmp_path / ".env").exists()
    for request in adapter.requests:
        assert SECRET not in repr(request)
    _assert_no_secret(tmp_path / "data")


def _assert_no_secret(data):
    """扫描实际数据根全部持久文件；传参：数据目录；返回：无。"""
    for path in data.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), str(path.relative_to(data))
