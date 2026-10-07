"""从真实请求发现未提前命名的工具，以及分页读取技能正文。

作者：xxx
时间：2026-09-14 21:00:00
"""

from __future__ import annotations

import json
from dataclasses import replace
from uuid import uuid4

import pytest

from context.engine import recall_context_body
from llm.client import RealLLMClient
from llm.config import LLMProviderConfig
from llm.provider_adapter import AdapterRegistry
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.resolved_target import resolve_model_target
from memory.records import MemorySource
from runtime.agent_loop import State
from runtime.capability_catalog import browse_tools, loaded_tools_from_operations
from runtime.lease import Lease
from runtime.session_messages import materialize_messages
from runtime.watchdog import Watchdog
from skills.store import SkillStore, build_skill_markdown
from tests.test_tool_batch_execution import make_run
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import Idempotent, ToolDefinition, ToolRisk
from tools.types import ToolError


class _DiscoveryModel:
    """用实际返回目录决定下一次调用的协议替身，不预先注入目标工具名。"""

    def __init__(self):
        """初始化实际请求记录；传参：无；返回：无。"""
        self.requests = []
        self.discovered_name = ""

    def stream(self, body, _connection):
        """产生合法Chat流，把实际发现的名字接成原生调用；传参：发送正文和连接；返回：事件流。"""
        self.requests.append(body)
        action = self.next_action(body)
        if action is None:
            delta, finish = {"content": "已核对账单"}, "stop"
        else:
            name, args = action
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"catalog-{len(self.requests)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
            finish = "tool_calls"
        yield {
            "id": f"response-{len(self.requests)}",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        yield {
            "id": f"response-{len(self.requests)}",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        }

    def next_action(self, body):
        """依据上一页和加载回执选择下一步；传参：真实模型输入；返回：工具/参数或结束。"""
        results = [item for item in body["messages"] if item["role"] == "tool"]
        if not results:
            return "capabilities", {"action": "list", "query": "账单", "limit": 1}
        envelope = json.loads(results[-1]["content"])
        assert envelope["status"] == "ok", envelope
        output = json.loads(envelope["output"])
        if "loaded_tool_names" in output:
            self.discovered_name = output["loaded_tool_names"][0]
            assert self.discovered_name in {
                item["function"]["name"] for item in body["tools"]
            }
            return self.discovered_name, {"amount": 21}
        if "items" in output:
            match = next(
                (item for item in output["items"] if "差额" in item["description"]),
                None,
            )
            if match is not None:
                return "capabilities", {
                    **match["load_action"],
                    "registry_version": output["registry_version"],
                }
            return "capabilities", {
                "action": "list",
                "query": "账单",
                "limit": 1,
                "cursor": output["next_cursor"],
            }
        assert output == {"difference": 21}
        return None


def test_catalog_pages_load_native_schema_and_execute_discovered_tool(tmp_path):
    """真实装配先看基础工具，翻页加载后执行目标，已加载材料可跨运行恢复；传参：隔离根；返回：无。"""
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    effects = []
    for prefix, description in (("a", "导出账单"), ("z", "计算账单差额")):
        registry.register(
            ToolDefinition(
                f"{prefix}_{uuid4().hex}",
                description,
                {"amount": {"type": "number", "required": True}},
                "agent",
                ToolRisk.SAFE,
                True,
                "logical_scope",
                "builtin",
                idempotent=Idempotent.YES,
                deferred=True,
                executor=lambda args: (
                    effects.append(args["amount"]) or {"difference": args["amount"]}
                ),
            )
        )
    scenario = _DiscoveryModel()
    target = resolve_model_target(
        cli_overrides={
            "base_url": "https://fixture.invalid/v1",
            "model": "fixture",
            "api_key": "synthetic-catalog-token",
        }
    )
    client = RealLLMClient(
        LLMProviderConfig(
            target.base_url, target.model, target.api_key, target.timeout_seconds
        ),
        resolved_target=target,
        adapter_registry=AdapterRegistry(
            [OpenAIChatAdapter(stream_factory=scenario.stream)]
        ),
    )
    loop, context, _ = make_run(tmp_path, registry, [])
    loop.llm_client = client
    assert loop.run(context) is State.DONE
    assert effects == [21]
    assert scenario.discovered_name not in {
        item["function"]["name"] for item in scenario.requests[0]["tools"]
    }
    assert len(scenario.requests) == 5
    messages = materialize_messages(tmp_path, context.session_id)
    assert loaded_tools_from_operations(tmp_path, context.session_id, messages) == {
        scenario.discovered_name
    }
    assert (
        loaded_tools_from_operations(tmp_path, context.session_id, messages[:1])
        == set()
    )


def test_catalog_cursor_and_load_respect_definition_updates_and_permissions(tmp_path):
    """目录变更使旧游标失效，加载不能打开越界能力；传参：隔离根；返回：无。"""
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    first = browse_tools(
        registry, {"action": "list", "limit": 1}, lease=Lease(), policy=None
    )
    definition = registry.get("terminal_tool")
    registry.replace(replace(definition, description="新的命令说明"))
    with pytest.raises(ValueError, match="catalog changed"):
        browse_tools(
            registry,
            {"action": "list", "limit": 1, "cursor": first["next_cursor"]},
            lease=Lease(),
            policy=None,
        )
    with pytest.raises(ValueError, match="current authorization"):
        browse_tools(
            registry,
            {"action": "load", "name": "terminal_tool"},
            lease=Lease(capabilities={"terminal": {"enabled": False}}),
            policy=None,
        )


def test_skill_paging_reaches_body_and_checks_version_at_common_execution_boundary(
    tmp_path,
):
    """第四个方法可分页发现并读正文，旧版本及越界身份不被执行；传参：隔离根；返回：无。"""
    store = SkillStore(tmp_path)
    for index in range(4):
        store.create_skill(
            f"method-{index}",
            build_skill_markdown(
                name=f"方法{index}", body=f"完整方法正文{index}，保留账单原始金额"
            ),
        )
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    watchdog = Watchdog(Lease())
    watchdog.data_root = tmp_path
    first = registry.execute_tool(
        "skill_search", {"limit": 3}, Lease(), watchdog=watchdog
    )
    assert len(first["items"]) == 3 and first["has_more"]
    second = registry.execute_tool(
        "skill_search",
        {"limit": 3, "cursor": first["next_cursor"]},
        Lease(),
        watchdog=watchdog,
    )
    discovered = second["items"][0]
    store.touch_skill(discovered["skill_id"])
    body = registry.execute_tool(
        "skill_read", discovered["read_action"], Lease(), watchdog=watchdog
    )
    assert body["body"] == "完整方法正文3，保留账单原始金额"
    store.revise_skill(
        discovered["skill_id"],
        build_skill_markdown(name="修订方法", body="新正文"),
        expected_version=discovered["version"],
        reason="加入新反馈",
        sources=(MemorySource("user_input", "fixture:feedback"),),
        publish=True,
    )
    pinned = registry.execute_tool(
        "skill_read", discovered["read_action"], Lease(), watchdog=watchdog
    )
    assert pinned["body"] == "完整方法正文3，保留账单原始金额"
    store.withdraw_version(
        discovered["skill_id"], discovered["version"], reason="旧方法已撤回"
    )
    withdrawn = registry.execute_tool(
        "skill_read", discovered["read_action"], Lease(), watchdog=watchdog
    )
    assert isinstance(withdrawn, ToolError) and "withdrawn" in withdrawn.message
    escaped = registry.execute_tool(
        "skill_read", {"skill_id": "../outside"}, Lease(), watchdog=watchdog
    )
    assert isinstance(escaped, ToolError) and "single storage name" in escaped.message


@pytest.mark.parametrize("script", [None, "def main(args):\n    return args\n"])
def test_recalled_skill_exposes_callable_pinned_read_and_script_access(
    tmp_path, script
):
    """召回给出可调用的固定版本读取入口，只为实际脚本提供执行入口；传参：目录和脚本；返回：无。"""
    store = SkillStore(tmp_path)
    original = store.create_skill(
        "meeting-minutes",
        build_skill_markdown(name="会议占用", body="合并会议时间区间"),
        script=script,
        meta={"script_entry": "script.py:main"} if script else {},
    )
    store.revise_skill(
        original.skill_id,
        build_skill_markdown(name="会议占用", body="新的会议计算说明"),
        expected_version=original.version,
        reason="新规则",
        sources=(MemorySource("user_input", "fixture:change"),),
        publish=True,
    )
    text = recall_context_body(
        tmp_path,
        task_summary="整理会议",
        task_tags=[],
        skill_refs=[f"skill:{original.skill_id}@{original.version}"],
    )
    links = [
        line.removeprefix("skill_access=")
        for line in text.splitlines()
        if line.startswith("skill_access=")
    ]
    assert len(links) == 1, (
        "recalled methods need an unambiguous tool entry and version arguments"
    )
    access = json.loads(links[0])
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    watchdog = Watchdog(Lease(), data_root=tmp_path)
    action = access["read_action"]
    result = registry.execute_tool(
        action["tool"], action["arguments"], Lease(), watchdog=watchdog
    )
    assert not isinstance(result, ToolError), result
    assert (
        result["skill_id"] == original.skill_id
        and result["version"] == original.version
    )
    assert result["body"] == "合并会议时间区间"
    if script is None:
        assert "execution_tool" not in access and "load_execution_action" not in access
    else:
        load = access["load_execution_action"]
        assert load["tool"] == "capabilities"
        selected = browse_tools(
            registry,
            load["arguments"],
            lease=Lease(capabilities={"code_execution": {"enabled": True}}),
            policy=None,
        )
        assert (
            selected["tool"]["function"]["name"]
            == access["execution_tool"]
            == "skill_run"
        )
