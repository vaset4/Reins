"""工具状态在实时事件与重连历史之间保持一致。

作者：xxx
时间：2026-09-30 01:00:00
"""

from dataclasses import asdict

import pytest

from app.background.sessions import session_history
from frontends.tui.projection import ConversationProjection
from runtime.session_messages import ToolExchange, append_tool_exchange
from runtime.stream_events import ToolExecutionCompleted
from runtime.tool_results import _render_tool_conversation, execution_display_details
from runtime.types import RunToolsResult


@pytest.mark.parametrize(
    ("execution", "expected"),
    [
        (
            {"execution_state": "not_started", "approval_state": "denied"},
            "未派发 · 用户拒绝",
        ),
        (
            {"execution_state": "not_started", "approval_state": "cancelled"},
            "未派发 · 审批已中断",
        ),
        (
            {"execution_state": "not_started", "approval_state": "unavailable"},
            "未派发 · 审批服务故障",
        ),
        ({"execution_state": "unknown"}, "执行结果未知，需核对副作用"),
        (
            {"execution_state": "completed", "tool_error_category": "cancelled"},
            "已中断",
        ),
        (
            {"execution_state": "completed", "tool_error_category": "invalid_input"},
            "失败",
        ),
    ],
)
def test_live_and_persisted_tool_states_agree(tmp_path, execution, expected):
    """真实持久回执与实时投影保留相同执行/审批语义；传参：状态与预期；返回：无。"""
    result = RunToolsResult.error_result(
        action="read",
        tool_name="read",
        error="操作没有完成",
        meta={**execution, "call_id": "call"},
    )
    append_tool_exchange(
        tmp_path,
        "session",
        ToolExchange(
            "call",
            "read",
            {},
            _render_tool_conversation(result),
            result.status,
            result.error,
        ),
        run_id="run",
    )
    live = ConversationProjection()
    live.apply({"session_id": "session", "history": [], "cursor": 0})
    event = ToolExecutionCompleted(
        "read",
        result.output,
        "call",
        is_error=True,
        execution=execution_display_details(result.meta),
    )
    live.apply(
        {
            "session_id": "session",
            "events": [
                {
                    "sequence": 1,
                    "run_id": "run",
                    "type": "ToolExecutionCompleted",
                    "data": asdict(event),
                }
            ],
        }
    )
    restored = ConversationProjection()
    restored.apply(
        {
            "session_id": "session",
            "history": session_history(tmp_path, "session"),
            "cursor": 1,
        }
    )
    assert live.cards["tool:session:run:call"].state == expected
    assert restored.cards["tool:session:run:call"].state == expected


def test_display_fields_exclude_diagnostics():
    """显示接口只带执行语义，不转发诊断字典；传参：无；返回：无。"""
    assert execution_display_details(
        {"execution_state": "unknown", "diagnostics": {"credential": "private"}}
    ) == {"execution_state": "unknown"}


def test_executor_emits_same_unknown_effects_as_saved_history(tmp_path):
    """真实工具执行器把未知副作用交给实时与历史显示；传参：隔离根；返回：无。"""
    from llm.messages import ToolCallPart
    from scripts.testing.llm import from_test_native_tool_then_final
    from tests.test_background_sessions import session_for
    from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
    from tools.types import ToolError, ToolErrorCategory

    def interrupted(arguments):
        """在工具边界返回执行中断；传参：工具参数；返回：明确未知效果。"""
        return ToolError(
            ToolErrorCategory.CANCELLED,
            "工作被中断",
            partial_state="需要检查实际文件",
            details={"execution_state": "unknown"},
            diagnostics={"internal": "diagnostic-only"},
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "interrupted_action",
            "动作",
            {},
            "agent",
            ToolRisk.SAFE,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            executor=interrupted,
        )
    )
    client = from_test_native_tool_then_final(
        [ToolCallPart("call", "interrupted_action", {})], "已报告执行状态"
    )
    session = session_for(tmp_path, client, registry=registry)
    try:
        session.submit("执行", input_id="state-input", model_config={})
        assert session.runtime.wait_idle(10)
        snapshot = session.snapshot(after=0)
        event = next(
            row for row in snapshot["events"] if row["type"] == "ToolExecutionCompleted"
        )
        assert event["data"]["execution"]["execution_state"] == "unknown"
        assert "diagnostic-only" not in str(event["data"])
        projection = ConversationProjection()
        projection.apply(session.snapshot(history=True))
        card = next(card for card in projection.cards.values() if card.role == "tool")
        assert card.state == "执行结果未知，需核对副作用"
    finally:
        session.close()
