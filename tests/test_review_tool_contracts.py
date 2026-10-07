"""【审查修复】【工具合同】验证加载状态按需读取和代码执行时间。

作者：xxx
时间：2026-10-03 11:00:00
"""

from pathlib import Path

import pytest

from llm.messages import AssistantMessage, TextPart, ToolCallPart, ToolResultMessage
from runtime.capability_catalog import loaded_tools_from_operations
from runtime.file_content import ContentFiles
from runtime.persistence import RuntimeStore
from tests.test_stage8_file_capture import capture_env as _capture_env, execute
from tools.exec_channel import ExecResult
from tools.types import ToolError

capture_env = _capture_env


def test_loading_state_does_not_expand_unrelated_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """大量其他结果不参与加载/未决状态恢复；参数：隔离根/读取计数；返回：无。"""
    store = RuntimeStore(tmp_path)
    with store.transaction() as batch:
        for number in range(30):
            batch.put(
                "tool_operation",
                f"op-{number}",
                {
                    "operation_id": f"op-{number}",
                    "updated_at": "now",
                    "call": {"call_id": f"call-{number}"},
                    "state": "completed",
                    "result": {"output": f"unrelated-{number}" * 2000},
                },
                session_id="loading-session",
            )
        batch.put(
            "tool_operation",
            "op-load",
            {
                "operation_id": "op-load",
                "updated_at": "now",
                "call": {"call_id": "load"},
                "state": "completed",
                "result": {"meta": {"loaded_tool_names": ["memory_manage"]}},
            },
            session_id="loading-session",
        )
        batch.put(
            "tool_operation",
            "op-pending",
            {
                "operation_id": "op-pending",
                "updated_at": "now",
                "call": {"call_id": "pending"},
                "state": "unknown",
                "result": {"output": "unknown-effects" * 2000},
            },
            session_id="loading-session",
        )
    reads = []
    original = ContentFiles.read

    def read_content(self, reference, **options):
        """记录实际正文读取；参数：内容引用/分页；返回：真实正文。"""
        reads.append(reference.path)
        return original(self, reference, **options)

    monkeypatch.setattr(ContentFiles, "read", read_content)
    messages = [
        AssistantMessage(
            message_id="calls",
            content=(
                ToolCallPart("load", "capabilities", {}),
                ToolCallPart("pending", "terminal_tool", {}),
                ToolCallPart("call-0", "file_read", {}),
            ),
        ),
        ToolResultMessage(
            "loaded", "load", "capabilities", (TextPart("loaded"),), "success"
        ),
    ]
    assert loaded_tools_from_operations(
        tmp_path, "loading-session", messages
    ) == frozenset({"memory_manage", "operation_status", "resume_operation"})
    assert reads == []


@pytest.mark.parametrize("requested,expected", [(0.25, 0.25), (120, 20), (None, 20)])
def test_code_timeout_reaches_backend(capture_env, monkeypatch, requested, expected):
    """模型时间参数进入实际后端且不超过宿主；参数：环境/替身/请求时间/预期；返回：无。"""
    seen = []

    def run(spec):
        """捕获真正提交给执行通道的规格；参数：执行规格；返回：确定的后端回执。"""
        seen.append(spec.timeout_seconds)
        return ExecResult(stdout="done", stderr="", exit_code=0)

    monkeypatch.setattr("tools.code_execution_tool._CHANNEL.run", run)
    args = {"code": "print('done')"}
    if requested is not None:
        args["timeout_seconds"] = requested
    result = execute(capture_env, "code_execution_tool", args)
    assert isinstance(result, dict), result
    assert seen == [expected]


@pytest.mark.parametrize("value", [0, -1, "5", True])
def test_invalid_code_timeout_is_rejected(capture_env, value):
    """时间参数必须是正数；参数：环境/非法输入；返回：无，不执行脚本。"""
    result = capture_env["registry"].prepare_tool_execution(
        "code_execution_tool",
        {"code": "print('unused')", "timeout_seconds": value},
        capture_env["lease"],
    )
    assert isinstance(result, ToolError)
