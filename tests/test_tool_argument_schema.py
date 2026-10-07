"""通过真实工具注册和执行入口验证参数合同。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from runtime.lease import from_trigger
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


def _definition(parameters: dict[str, object]) -> ToolDefinition:
    """注册原样回传参数的只读工具；传参：声明；返回：工具定义。"""
    return ToolDefinition(
        name="schema_probe",
        description="Validate typed arguments",
        parameters=parameters,
        toolset="agent",
        risk_level=TOOL_RISK_SAFE,
        readonly=True,
        target_scope_rule=TARGET_SCOPE_LOGICAL,
        source=TOOL_SOURCE_BUILTIN,
        idempotent=IDEMPOTENT_YES,
        executor=lambda args: {
            key: value for key, value in args.items() if not key.startswith("__")
        },
    )


@pytest.mark.parametrize(
    "value, schema",
    [
        (0, {"type": "integer", "minimum": 0}),
        (False, {"type": "boolean"}),
        (
            {"count": 2},
            {"type": "object", "properties": {"count": {"type": "integer"}}},
        ),
        (None, {"type": ["string", "null"]}),
        ("", {"type": "string"}),
    ],
)
def test_valid_json_values_reach_executor(
    value: object, schema: dict[str, object]
) -> None:
    """合法类型、空值与空串按声明执行；传参：值与类型约束；返回：无。"""
    registry = ToolRegistry()
    registry.register(_definition({"value": {**schema, "required": True}}))
    arguments = {"value": value}
    result = registry.execute_tool(
        "schema_probe", arguments, from_trigger("user", task_id="typed")
    )
    assert result == arguments


def test_nested_constraints_are_shared_by_model_and_executor() -> None:
    """模型与执行器保留相同嵌套约束，错误定位到字段；传参：无；返回：无。"""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["count"],
                    "additionalProperties": False,
                    "properties": {"count": {"type": "integer", "minimum": 1}},
                },
            }
        },
        "required": ["items"],
    }
    definition = _definition(schema)
    registry = ToolRegistry()
    registry.register(definition)
    assert definition.format_for_openai_tool()["function"]["parameters"] == schema
    lease = from_trigger("user", task_id="nested")
    good = {"items": [{"count": 3}]}
    assert registry.execute_tool("schema_probe", good, lease) == good
    bad = registry.execute_tool("schema_probe", {"items": [{"count": False}]}, lease)
    assert isinstance(bad, ToolError)
    assert bad.category is ToolErrorCategory.INVALID_INPUT
    assert "items[0].count" in bad.message
    assert "integer" in bad.message


def test_invalid_enum_and_extra_fields_are_rejected_without_dispatch() -> None:
    """拒绝枚举外取值与未声明字段；传参：无；返回：无。"""
    registry = ToolRegistry()
    registry.register(
        _definition({"mode": {"type": "string", "enum": ["read"], "required": True}})
    )
    lease = from_trigger("user", task_id="enum")
    for arguments in ({"mode": "write"}, {"mode": "read", "extra": 1}):
        result = registry.execute_tool("schema_probe", arguments, lease)
        assert isinstance(result, ToolError)
        assert result.category is ToolErrorCategory.INVALID_INPUT


def test_schema_copy_cannot_change_registered_argument_contract() -> None:
    """输出的模型Schema不能修改注册参数来源；传参：无；返回：无。"""
    original = {"value": {"type": "integer", "required": True}}
    before = deepcopy(original)
    definition = _definition(original)
    rendered = definition.format_for_openai_tool()
    rendered["function"]["parameters"]["properties"]["value"]["type"] = "string"
    assert original == before
    assert (
        definition.format_for_openai_tool()["function"]["parameters"]["properties"][
            "value"
        ]["type"]
        == "integer"
    )


def test_file_write_and_empty_patch_preserve_exact_text(tmp_path: Path) -> None:
    """写文件保留缩进和换行，空替换可删除片段；传参：隔离目录；返回：无。"""
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path / "data")
    lease = from_trigger(
        "user",
        task_id="editing",
        capabilities={
            "fs": {
                "project_root": str(tmp_path),
                "read": [str(tmp_path)],
                "write": [str(tmp_path)],
            },
        },
    )
    content = "  first\n  remove\n\n"
    write = registry.execute_tool(
        "file_write", {"path": "sample.txt", "content": content}, lease
    )
    assert not isinstance(write, ToolError)
    assert (tmp_path / "sample.txt").read_bytes() == content.encode("utf-8")
    patch = registry.execute_tool(
        "file_patch",
        {
            "path": "sample.txt",
            "old_text": "  remove\n",
            "new_text": "",
            "expected_sha256": write["meta"]["content_sha256"],
        },
        lease,
    )
    assert not isinstance(patch, ToolError)
    assert (tmp_path / "sample.txt").read_bytes() == b"  first\n\n"
