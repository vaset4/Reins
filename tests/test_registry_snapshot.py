"""请求定义与执行版本在更新、撤销和外部修改期间保持一致。

作者：xxx
时间：2026-09-14 21:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

from dataclasses import replace

import pytest

from llm.messages import ToolCallPart
from runtime.agent_loop import State
from runtime.lease import Lease
from tests.test_session_runtime import capture_requests
from tests.test_tool_batch_execution import make_run
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
from tools.types import ToolError


def _definition(executor):
    """声明一个可观察实际执行版本的读取工具；传参：后端；返回：定义。"""
    return ToolDefinition(
        "lookup",
        "查询资料",
        {"key": {"type": "string", "required": True}},
        "agent",
        ToolRisk.SAFE,
        True,
        "logical_scope",
        "builtin",
        idempotent=Idempotent.YES,
        executor=executor,
    )


def test_published_definition_isolated_from_caller_mutations():
    """发布后改动传入或取出的对象不能悄悄更换定义；传参：无；返回：无。"""
    registry = ToolRegistry()
    original = _definition(lambda _args: "original")
    registry.register(original)
    original.description = "mutated input"
    retrieved = registry.get("lookup")
    retrieved.parameters["properties"]["key"]["type"] = "integer"
    assert registry.get("lookup").description == "查询资料"
    assert registry.get("lookup").parameters["properties"]["key"]["type"] == "string"


@pytest.mark.parametrize("revoke", [False, True])
def test_request_uses_prepared_definition_but_current_authorization(
    tmp_path, monkeypatch, revoke
):
    """请求期间更新用旧版执行，撤销则不执行，后续请求使用新定义；传参：目录与撤销分支；返回：无。"""
    effects = []
    registry = ToolRegistry()
    registry.register(_definition(lambda _args: effects.append("v1") or "v1-result"))
    original_version = registry.version
    client = from_test_native_tool_then_final(
        [ToolCallPart("snapshot-call", "lookup", {"key": "资料"})], "已处理"
    )

    def update_definition():
        """在实际请求发出后发布新定义；传参：无；返回：无。"""
        registry.replace(
            replace(
                _definition(lambda _args: effects.append("v2") or "v2-result"),
                description="查询新版资料",
                risk_level=ToolRisk.DENY if revoke else ToolRisk.SAFE,
            )
        )

    requests = capture_requests(client, monkeypatch, before_first=update_definition)
    loop, context, _ = make_run(tmp_path, registry, [])
    loop.llm_client = client
    assert loop.run(context) is State.DONE
    assert effects == ([] if revoke else ["v1"])
    assert requests[0].tools[0].description == "查询资料"
    assert (
        "v1-result" in str(requests[-1])
        if not revoke
        else "not allowed" in str(requests[-1])
    )
    operation = loop.operations.for_session(context.session_id)[0]
    assert operation["call"]["registry_version"] == original_version
    assert registry.version != original_version
    if not revoke:
        assert requests[-1].tools[0].description == "查询新版资料"
        assert (
            operation["call"]["execution_request"]["registry_version"]
            == original_version
        )


def test_revocation_after_preparation_blocks_real_executor():
    """审批后但执行前撤销仍然阻止副作用；传参：无；返回：无。"""
    effects = []
    registry = ToolRegistry()
    registry.register(_definition(lambda _args: effects.append("ran")))
    snapshot = registry.snapshot()
    prepared = snapshot.prepare_tool_execution("lookup", {"key": "x"}, Lease())
    registry.replace(replace(registry.get("lookup"), risk_level=ToolRisk.DENY))
    result = snapshot.execute_prepared_tool(prepared)
    assert isinstance(result, ToolError)
    assert effects == []


def test_failed_catalog_publication_keeps_entire_previous_version():
    """同批目录包含重复名字时不发布半个目录；传参：无；返回：无。"""
    registry = ToolRegistry()
    registry.register(_definition(lambda _args: "v1"))
    version = registry.version
    with pytest.raises(ValueError, match="duplicate"):
        registry.publish(
            [_definition(None), _definition(None)], replace_names=("lookup",)
        )
    assert registry.version == version
    assert registry.get("lookup").executor({}) == "v1"
