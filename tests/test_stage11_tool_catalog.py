"""【工具体系】【发现与权限】验证常用核心、中文领域加载和整合后的真实动作。

作者：xxx
时间：2026-10-02 14:30:00
"""

from contextlib import closing

from llm.tool_selection import select_tools
from llm.toolset_policy import ToolsetPolicy
from runtime.capability_catalog import browse_tools
from runtime.lease import Lease
from tools.builtin_tools import build_tool_registry
from tools.types import ToolError

CORE = {
    "capabilities",
    "ask_user",
    "goal",
    "memory_query",
    "read_artifact",
    "list",
    "find_path",
    "grep",
    "file_read",
    "file_write",
    "file_patch",
    "terminal_tool",
}


def test_default_core_retains_all_other_domains_for_chinese_discovery(tmp_path):
    """常驻十二项，其余合法能力可由中文需求成组发现并加载；参数：隔离目录；返回：无。"""
    with closing(
        build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    ) as registry:
        assert select_tools(registry).allowed_tool_names == CORE
        found = browse_tools(
            registry,
            {"action": "search", "query": "帮我联网查网页"},
            lease=None,
            policy=None,
        )
        assert {"web_search", "web_fetch"} <= {row["name"] for row in found["items"]}
        loaded = browse_tools(
            registry, {"action": "load", "domain": "web"}, lease=None, policy=None
        )
        assert set(loaded["loaded_tool_names"]) == {
            "web_search",
            "web_fetch",
            "web_scan",
        }
        selected = select_tools(
            registry, loaded_tools=frozenset(loaded["loaded_tool_names"])
        )
        assert (
            CORE | {"web_search", "web_fetch", "web_scan"}
            == selected.allowed_tool_names
        )
        assert all(
            registry.get(name) is None
            for name in (
                "memory_search",
                "memory_archive",
                "todo_add",
                "todo_list",
                "todo_update",
            )
        )


def test_domain_loading_does_not_grant_disabled_network_or_write_permissions(tmp_path):
    """按需加载仍遵守当前网络和文件写入权限，隐藏工具不能被发现；参数：隔离目录；返回：无。"""
    lease = Lease(
        capabilities={
            "network": {"enabled": False},
            "fs": {"read": [str(tmp_path)], "write": []},
        }
    )
    with closing(
        build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    ) as registry:
        files = browse_tools(
            registry,
            {"action": "load", "domain": "files"},
            lease=lease,
            policy=ToolsetPolicy(read_only=True),
        )
        assert {"file_read", "find_path"} <= set(files["loaded_tool_names"])
        assert not {"file_write", "file_patch"} & set(files["loaded_tool_names"])
        web = browse_tools(
            registry, {"action": "list", "domain": "web"}, lease=lease, policy=None
        )
        assert web["items"] == [] and web["unavailable"]
        assert not {"echo", "inspect", "ocr", "screenshot"} & {
            row["name"] for row in web["items"]
        }


def test_todo_list_is_readonly_but_writes_retain_approval_risk(tmp_path):
    """合并后的查询与修改仍有各自风险和幂等性质；参数：隔离目录；返回：无。"""
    with closing(
        build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    ) as registry:
        definition = registry.get("todo")
        assert definition.action_readonly({"action": "list"})
        assert not definition.action_readonly({"action": "add", "content": "核对金额"})
        assert definition.action_risk({"action": "list"}).value == "safe"
        assert (
            definition.action_risk(
                {"action": "update", "idx": 0, "status": "done"}
            ).value
            == "confirm"
        )
        assert not isinstance(
            registry.validate_model_request(
                tool_name="todo", arguments={"action": "list"}
            ),
            ToolError,
        )
        assert isinstance(
            registry.validate_model_request(
                tool_name="memory_manage",
                arguments={"action": "archive", "memory_id": "known"},
            ),
            ToolError,
        )
        assert select_tools(
            registry, policy=ToolsetPolicy(enabled_toolsets=("safe_read",))
        ).allowed_tool_names


def test_consolidated_actions_keep_preset_scope_at_dispatch(tmp_path):
    """旧预设只允许的动作在合并后仍受限制，禁用不会误删其他合法动作；参数：隔离根；返回：无。"""
    from llm.toolset_policy import allowed_tool_actions
    from runtime.session_state import SessionStateStore
    from runtime.tool_policy import RuntimeToolPolicy
    from runtime.types import RunContext, RunToolsRequest, Trigger

    with closing(
        build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    ) as registry:
        policy = ToolsetPolicy(enabled_toolsets=("safe_read",))
        assert allowed_tool_actions(policy, registry.get("todo")) == {"list"}
        assert allowed_tool_actions(policy, registry.get("memory_manage")) == {
            "archive",
            "restore",
        }
        definitions = {
            row.name: row
            for row in select_tools(registry, policy=policy).selected_definitions
        }
        assert definitions["todo"].parameters["properties"]["action"]["enum"] == [
            "list"
        ]
        run = RunContext(
            session_id="source",
            trigger=Trigger.USER,
            capability_lease=Lease(),
            payload={"toolset_policy": {"enabled_toolsets": ["safe_read"]}},
        )
        boundary = RuntimeToolPolicy(registry, SessionStateStore(tmp_path), {})
        assert (
            boundary.validate(
                run, RunToolsRequest("todo", arguments={"action": "list"})
            )
            is None
        )
        assert (
            boundary.validate(
                run,
                RunToolsRequest("todo", arguments={"action": "add", "content": "新项"}),
            )
            is not None
        )
        disabled = ToolsetPolicy(disabled_toolsets=("research",))
        remaining = select_tools(registry, policy=disabled, include_deferred=True)
        assert "memory_query" in remaining.allowed_tool_names
        assert "search" not in allowed_tool_actions(
            disabled, registry.get("memory_query")
        )


def test_todo_actions_use_existing_task_records_without_completing_goal(tmp_path):
    """真实新增、查询和更新沿用待办原件，全部勾选不改变正式目标状态；参数：隔离根；返回：无。"""
    from tasks.store import TaskStore

    with (
        closing(TaskStore(tmp_path)) as tasks,
        closing(
            build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
        ) as registry,
    ):
        goal = tasks.create_task("核对阶段十一")
        execute = registry.get("todo").executor
        scope = {"task_id": goal.task_id, "__data_root__": str(tmp_path)}
        added = execute({**scope, "action": "add", "content": "检查金额"})
        assert added["status"] == "pending"
        execute({**scope, "action": "update", "idx": added["idx"], "status": "done"})
        assert execute({**scope, "action": "list"})[0]["status"] == "done"
        assert tasks.require_task(goal.task_id).status == goal.status


def test_readonly_mode_uses_actual_host_session_and_retains_read_actions(tmp_path):
    """实际模式变化同时收窄请求和派发，空白名单仍可审批；参数：隔离根；返回：无。"""
    from approval.session import ApprovalMode, ApprovalSession
    from runtime.session_state import SessionStateStore
    from runtime.tool_policy import RuntimeToolPolicy
    from runtime.types import RunContext, RunToolsRequest, Trigger
    from llm.toolset_policy import policy_from_mapping, policy_to_mapping

    with closing(
        build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    ) as registry:
        session = ApprovalSession()
        boundary = RuntimeToolPolicy(
            registry, SessionStateStore(tmp_path), {}, approval_session=session
        )
        run = RunContext(
            session_id="readonly",
            trigger=Trigger.USER,
            capability_lease=Lease(),
            payload={},
        )
        write = RunToolsRequest(
            "file_write", arguments={"path": "result.txt", "content": "value"}
        )
        assert boundary.validate(run, write) is None
        session.set_mode(ApprovalMode.READ_ONLY)
        assert boundary.validate(run, write) is not None
        assert policy_from_mapping(
            policy_to_mapping(boundary.resolve(run)), source="test"
        ).read_only
        assert (
            boundary.validate(
                run, RunToolsRequest("todo", arguments={"action": "list"})
            )
            is None
        )
        assert (
            boundary.validate(
                run,
                RunToolsRequest("todo", arguments={"action": "add", "content": "item"}),
            )
            is not None
        )
        assert (
            boundary.validate(
                run, RunToolsRequest("schedule", arguments={"action": "list"})
            )
            is None
        )
        assert (
            boundary.validate(
                run, RunToolsRequest("terminal_tool", arguments={"command": "pwd"})
            )
            is None
        )
        tools = select_tools(
            registry, policy=boundary.resolve(run), include_deferred=True
        )
        assert "file_write" not in tools.allowed_tool_names
        assert next(
            row for row in tools.selected_definitions if row.name == "todo"
        ).parameters["properties"]["action"]["enum"] == ["list"]
        assert (
            "schedule"
            not in select_tools(
                registry,
                policy=ToolsetPolicy(read_only=True, disabled_toolsets=("agent",)),
                include_deferred=True,
            ).allowed_tool_names
        )
