from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from llm.messages import MessageContractError, ToolCallPart
from llm.parser import parse_llm_response, parse_tool_call_parts


def test_parse_final_response() -> None:
    plan = parse_llm_response(
        '{"type":"final","content":"hello"}',
        allowed_tool_names=set(),
    )

    assert plan.final_output == "hello"
    assert plan.run_tools_request is None


def test_native_protocol_plain_text_becomes_final_answer() -> None:
    plan = parse_llm_response(
        "hello in plain text",
        protocol_mode="native_tool_calls",
        allowed_tool_names=set(),
    )

    assert plan.final_output == "hello in plain text"
    assert plan.model_error is None


@pytest.mark.parametrize("protocol_mode", ["native_tool_calls", "text_json"])
def test_business_json_answer_respects_protocol_mode(protocol_mode: str) -> None:
    """业务JSON在原生模式保留正文，文本动作模式仍要求type；传参：协议模式；返回：无。"""
    raw_text = json.dumps(
        {
            "recommendation": "CEDAR",
            "budget_yuan": 4800,
            "may_purchase": False,
            "pending_checks": ["核实发票税率", "完成备份还原演练"],
        },
        ensure_ascii=False,
    )
    plan = parse_llm_response(
        raw_text, protocol_mode=protocol_mode, allowed_tool_names=set()
    )

    assert plan.run_tools_request is None
    if protocol_mode == "native_tool_calls":
        assert plan.final_output == raw_text
        assert plan.model_error is None
    else:
        assert plan.model_error is not None
        assert plan.model_error.category == "invalid_model_protocol"


@pytest.mark.parametrize(
    "raw_text", ['{"type":"unknown"}', '{"type":null}', '{"incomplete":']
)
def test_native_protocol_keeps_invalid_actions_and_json_as_errors(
    raw_text: str,
) -> None:
    """未知动作及破损JSON不因正文支持而吞错；传参：错误响应；返回：无。"""
    plan = parse_llm_response(
        raw_text, protocol_mode="native_tool_calls", allowed_tool_names=set()
    )

    assert plan.run_tools_request is None
    assert plan.model_error is not None
    assert plan.model_error.category == "invalid_model_protocol"


def test_native_protocol_rejects_text_json_run_tools() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","tool":"file_read","arguments":{"path":"app/cli.py"}}',
        protocol_mode="native_tool_calls",
        allowed_tool_names={"file_read"},
    )

    assert plan.run_tools_request is None
    assert plan.model_error is not None
    assert plan.model_error.category == "invalid_model_protocol"
    assert "native text response attempted run_tools" in plan.model_error.raw_summary


def test_parse_run_tools_response() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","action":"inspect","payload":"file app/cli.py"}',
        allowed_tool_names={"inspect"},
    )

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.action == "inspect"
    assert plan.run_tools_request.payload == "file app/cli.py"


def test_parse_structured_tool_response() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","tool":"file_read","arguments":{"path":"app/cli.py"}}',
        allowed_tool_names={"file_read"},
    )

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "file_read"
    assert plan.run_tools_request.arguments == {"path": "app/cli.py"}
    assert plan.run_tools_request.action == "file_read"
    assert plan.run_tools_request.payload == ""


def test_parse_directory_tool_alias_response() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","tool":"list_directory","arguments":{"directory":"tools"}}',
        allowed_tool_names={"list"},
    )

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "list"
    assert plan.run_tools_request.arguments == {"path": "tools"}
    assert plan.run_tools_request.target_scope == "tools"


def test_parse_structured_write_tool_response() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","tool":"file_write","arguments":{"path":"notes.txt","content":"hello"}}',
        allowed_tool_names={"file_write"},
    )

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "file_write"
    assert plan.run_tools_request.arguments == {
        "path": "notes.txt",
        "content": "hello",
    }
    assert plan.run_tools_request.target_scope == "notes.txt"


def test_parse_structured_write_tool_response_with_alias_arguments() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","tool":"file_write","arguments":{"file_path":"notes.txt","text":"hello"}}',
        allowed_tool_names={"file_write"},
    )

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "file_write"
    assert plan.run_tools_request.arguments == {
        "path": "notes.txt",
        "content": "hello",
    }
    assert plan.run_tools_request.target_scope == "notes.txt"


def test_parse_structured_write_tool_response_prefers_canonical_arguments() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","tool":"file_write","arguments":{"path":"notes.txt","target_path":"ignored.txt","content":"hello","body":"ignored body"}}',
        allowed_tool_names={"file_write"},
    )

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "file_write"
    assert plan.run_tools_request.arguments == {
        "path": "notes.txt",
        "content": "hello",
    }
    assert plan.run_tools_request.target_scope == "notes.txt"


def test_parse_structured_patch_tool_response_with_alias_arguments() -> None:
    plan = parse_llm_response(
        json.dumps(
            {
                "type": "run_tools",
                "tool": "file_patch",
                "arguments": {
                    "target_path": "notes.txt",
                    "old_content": "hello",
                    "new_content": "hi",
                    "expected_sha256": "0" * 64,
                },
            }
        ),
        allowed_tool_names={"file_patch"},
    )

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "file_patch"
    assert plan.run_tools_request.arguments == {
        "path": "notes.txt",
        "old_text": "hello",
        "new_text": "hi",
        "expected_sha256": "0" * 64,
    }
    assert plan.run_tools_request.target_scope == "notes.txt"


def test_parse_readonly_tool_does_not_accept_write_alias_arguments() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","tool":"file_read","arguments":{"target_path":"notes.txt"}}',
        allowed_tool_names={"file_read"},
    )

    assert plan.run_tools_request is None
    assert plan.final_output == "MODEL_PROTOCOL_ERROR: invalid run_tools request"


def test_parse_legacy_inspect_response_still_works() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","action":"inspect","payload":"file app/cli.py"}',
        allowed_tool_names={"inspect"},
    )

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "inspect"
    assert plan.run_tools_request.payload == "file app/cli.py"


def test_parse_error_response_preserves_explicit_failure() -> None:
    """错误信封保留失败类别和说明，不产生成功答案；传参：无；返回：无。"""
    plan = parse_llm_response(
        '{"type":"error","message":"cannot comply"}',
        allowed_tool_names=set(),
    )

    assert plan.final_output is None and plan.run_tools_request is None
    assert plan.model_error is not None
    assert plan.model_error.category == "model_reported_error"
    assert plan.model_error.render_output() == "MODEL_RESPONSE_ERROR: cannot comply"


def test_parse_invalid_action_becomes_protocol_error() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","action":"delete","payload":"workspace"}',
        allowed_tool_names=set(),
    )

    assert plan.final_output == "MODEL_PROTOCOL_ERROR: invalid run_tools request"


def test_parse_invalid_inspect_payload_becomes_protocol_error() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","action":"inspect","payload":"{}"}',
        allowed_tool_names={"inspect"},
    )

    assert plan.final_output == "MODEL_PROTOCOL_ERROR: invalid run_tools request"


def test_parse_invalid_json_becomes_protocol_error() -> None:
    plan = parse_llm_response("not-json", allowed_tool_names=set())

    assert plan.final_output is not None
    assert plan.final_output.startswith("MODEL_PROTOCOL_ERROR:")
    assert plan.model_error is not None
    assert plan.model_error.category == "invalid_model_protocol"
    assert plan.model_error.retryable is False
    assert plan.model_error.stage == "parse"


def test_parse_structured_tool_rejects_current_turn_excluded_tool() -> None:
    plan = parse_llm_response(
        '{"type":"run_tools","tool":"file_read","arguments":{"path":"app/cli.py"}}',
        allowed_tool_names={"list"},
    )

    assert plan.run_tools_request is None
    assert plan.model_error is not None
    assert plan.model_error.category == "invalid_tool_arguments"
    assert "not allowed in current request" in plan.model_error.raw_summary


# --- typed 入口：吃 ToolCallPart，不再经 Wire dict ---


def test_parse_tool_call_parts_single_call() -> None:
    """单个工具调用块直接成为 run_tools_request，call_id 原样带出。
    作者：xxx
    时间：2026-08-30 15:20:00
    传参：无
    返回：无"""
    plan = parse_tool_call_parts(
        (
            ToolCallPart(
                call_id="call-typed-1",
                tool_name="list",
                arguments={"path": "tools"},
            ),
        ),
        allowed_tool_names={"list"},
    )

    assert plan.model_error is None
    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "list"
    assert plan.run_tools_request.arguments == {"path": "tools"}
    assert plan.run_tools_request.call_id == "call-typed-1"


def test_parse_tool_call_parts_multiple_calls_keep_pending_and_call_ids() -> None:
    """一次多个调用块：首条进 run_tools_request，全部进 pending_tool_calls 且 call_id 逐条对应。
    作者：xxx
    时间：2026-08-30 15:20:00
    传参：无
    返回：无"""
    plan = parse_tool_call_parts(
        (
            ToolCallPart(
                call_id="call-typed-1",
                tool_name="list",
                arguments={"path": "tools"},
            ),
            ToolCallPart(
                call_id="call-typed-2",
                tool_name="file_read",
                arguments={"path": "x"},
            ),
        ),
        allowed_tool_names={"list", "file_read"},
    )

    assert plan.model_error is None
    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "list"
    assert plan.run_tools_request.call_id == "call-typed-1"

    pending = plan.prompt_context.get("pending_tool_calls", [])
    assert len(pending) == 2
    assert [item.tool_name for item in pending] == ["list", "file_read"]
    assert [item.call_id for item in pending] == ["call-typed-1", "call-typed-2"]
    assert [item.arguments for item in pending] == [{"path": "tools"}, {"path": "x"}]


def test_tool_call_part_rejects_non_object_arguments() -> None:
    """工具调用参数不是 JSON object 时在内容块边界就被拒，非法参数进不了 typed 入口。
    作者：LKX
    时间：2026-08-31 14:09:39
    传参：无
    返回：无
    注：原 Wire dict 入口把 arguments=null 判成 invalid_tool_arguments 回喂模型；typed 栈把这道
        闸前移到 ToolCallPart 构造处，参数非 object 一律抛合同错，parse_tool_call_parts 拿到的
        arguments 已经是校验过的 mapping，故这里盯构造边界"""
    for arguments in (None, "path=tools", [1, 2]):
        with pytest.raises(MessageContractError) as exc_info:
            ToolCallPart(
                call_id="call-typed-1",
                tool_name="file_read",
                arguments=arguments,
            )

        assert exc_info.value.code == "invalid_json_value"
        assert exc_info.value.path == "tool_call_part.arguments"
        assert exc_info.value.detail == "value must be a JSON object"


def test_parse_tool_call_parts_rejects_current_turn_excluded_tool() -> None:
    """工具不在本轮准许清单时，保留调用身份并携带阻止执行的校验错误。
    作者：xxx
    时间：2026-08-30 15:20:00
    传参：无
    返回：无"""
    plan = parse_tool_call_parts(
        (
            ToolCallPart(
                call_id="call-typed-1",
                tool_name="list",
                arguments={"path": "tools"},
            ),
        ),
        allowed_tool_names={"file_read"},
    )

    assert plan.model_error is None
    assert plan.run_tools_request is not None
    assert plan.run_tools_request.call_id == "call-typed-1"
    assert plan.run_tools_request.validation_error is not None
    assert "not allowed in current request" in plan.run_tools_request.validation_error


def test_parse_tool_call_parts_rejects_failed_argument_validation() -> None:
    """缺少file_read必填path时，错误归属原调用并阻止该调用执行。
    作者：xxx
    时间：2026-08-30 15:20:00
    传参：无
    返回：无"""
    plan = parse_tool_call_parts(
        (
            ToolCallPart(
                call_id="call-typed-1",
                tool_name="file_read",
                arguments={},
            ),
        ),
        allowed_tool_names={"file_read"},
    )

    assert plan.model_error is None
    assert plan.run_tools_request is not None
    assert plan.run_tools_request.call_id == "call-typed-1"
    assert plan.run_tools_request.validation_error is not None
    assert "tool=file_read" in plan.run_tools_request.validation_error
    assert "'path' is a required property" in plan.run_tools_request.validation_error


def test_parse_tool_call_parts_preserves_valid_call_after_invalid_call() -> None:
    """首项参数错误只阻止该项，后续独立合法调用保留身份与参数。
    作者：xxx
    时间：2026-08-30 15:20:00
    传参：无
    返回：无"""
    plan = parse_tool_call_parts(
        (
            ToolCallPart(
                call_id="call-typed-1",
                tool_name="file_read",
                arguments={},
            ),
            ToolCallPart(
                call_id="call-typed-2",
                tool_name="list",
                arguments={"path": "tools"},
            ),
        ),
        allowed_tool_names={"list", "file_read"},
    )

    assert plan.model_error is None
    requests = plan.prompt_context["pending_tool_calls"]
    assert len(requests) == 2
    assert [request.call_id for request in requests] == ["call-typed-1", "call-typed-2"]
    assert "'path' is a required property" in requests[0].validation_error
    assert requests[1].validation_error is None
    assert requests[1].tool_name == "list"
    assert requests[1].arguments == {"path": "tools"}


def test_parse_tool_call_parts_thaws_frozen_nested_arguments() -> None:
    """ToolCallPart 深冻结的嵌套参数要解冻成普通 dict/list 才交下游。
    作者：xxx
    时间：2026-08-30 15:20:00
    传参：无
    返回：无
    注：冻结容器传下去不会当场报错，但序列化工具参数时会 TypeError，所以这里直接盯容器类型"""
    from tools.tool_registry import ToolDefinition, ToolRegistry

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="metadata_echo",
            description="Return structured metadata",
            parameters={
                "path": {"type": "string"},
                "content": {"type": "string"},
                "meta": {"type": "object"},
            },
            toolset="agent",
            risk_level="safe",
            readonly=True,
            target_scope_rule="logical_scope",
            source="builtin",
            idempotent="yes",
        )
    )
    part = ToolCallPart(
        call_id="call-typed-1",
        tool_name="metadata_echo",
        arguments={"path": "notes.txt", "content": "hello", "meta": {"a": [1, 2]}},
    )
    assert isinstance(part.arguments["meta"], Mapping)
    assert not isinstance(part.arguments["meta"], dict)

    plan = parse_tool_call_parts(
        (part,), allowed_tool_names={"metadata_echo"}, registry=registry
    )

    assert plan.model_error is None
    assert plan.run_tools_request is not None
    arguments = plan.run_tools_request.arguments
    assert type(arguments) is dict
    assert type(arguments["meta"]) is dict
    assert type(arguments["meta"]["a"]) is list
    assert json.dumps(arguments, sort_keys=True) == json.dumps(
        {"path": "notes.txt", "content": "hello", "meta": {"a": [1, 2]}},
        sort_keys=True,
    )


def test_parse_tool_call_parts_empty_input_is_empty_response() -> None:
    """一个调用块都没有时判空响应，且摘要里不出现 Wire 字段名。
    作者：xxx
    时间：2026-08-30 15:20:00
    传参：无
    返回：无"""
    plan = parse_tool_call_parts((), allowed_tool_names={"list"})

    assert plan.run_tools_request is None
    assert plan.model_error is not None
    assert plan.model_error.category == "empty_response"
    assert "tool_calls" not in plan.model_error.summary
    assert "tool_calls" not in plan.model_error.raw_summary
