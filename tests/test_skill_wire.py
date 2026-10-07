from __future__ import annotations

from pathlib import Path

from runtime.lease import from_trigger
from runtime.watchdog import Watchdog
from skills.store import SkillStore, build_skill_markdown
from tools import builtin_tools
from tools.tool_registry import (
    IDEMPOTENT_NO,
    IDEMPOTENT_YES,
    TOOL_RISK_CONFIRM,
    TOOL_RISK_SAFE,
    TOOLSET_AGENT,
    ToolRegistry,
)
from tools.types import ToolError, ToolErrorCategory


class _Watchdog(Watchdog):
    """使用真实预算与执行边界，只绑定本次隔离目录。"""

    def __init__(self, data_root: Path | None = None) -> None:
        """绑定目录和正常运行租约；传参：隔离根；返回：无。"""
        super().__init__(from_trigger("user", task_id="task-1"), data_root=data_root)


def _make_active_skill(
    store: SkillStore, skill_id: str, *, body: str = "run pytest"
) -> None:
    # 造一个 active、带默认脚本 main(args)->{'ok':args} 的可执行 skill
    store.create_skill(
        skill_id,
        build_skill_markdown(
            name=skill_id,
            body=body,
            trigger_keywords=["pytest"],
        ),
        script="def main(args):\n    return {'ok': args}\n",
        meta={"script_entry": "script.py:main"},
    )


def _lease_with_exec(tmp_path: Path):
    # lease 给 code_execution 能力 + write 域覆盖 tmp_path，使 skill.root 写检查放行
    return from_trigger(
        "user",
        task_id="task-1",
        capabilities={
            "fs": {"project_root": str(tmp_path), "read": [], "write": [str(tmp_path)]},
            "code_execution": {"enabled": True},
        },
    )


def test_skill_run_and_search_registered_with_expected_flags() -> None:
    # AC1/AC2：skill_run/skill_search 注册且风险分级符合决策
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    run_def = registry.get("skill_run")
    search_def = registry.get("skill_search")

    assert run_def is not None
    assert run_def.risk_level == TOOL_RISK_CONFIRM
    assert run_def.readonly is False
    assert run_def.idempotent == IDEMPOTENT_NO
    assert run_def.toolset == TOOLSET_AGENT
    assert run_def.model_visible is True
    assert run_def.exec_boundary is False

    assert search_def is not None
    assert search_def.risk_level == TOOL_RISK_SAFE
    assert search_def.readonly is True
    assert search_def.idempotent == IDEMPOTENT_YES
    assert search_def.model_visible is True


def test_skill_run_executes_active_skill(monkeypatch, tmp_path: Path) -> None:
    # AC3：走真实 registry 执行路径，返回 execute_skill_script 的 payload
    from approval import ApprovalDecision

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    _make_active_skill(SkillStore(tmp_path), "pytest-runner")
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    result = registry.execute_tool(
        "skill_run",
        {"skill_id": "pytest-runner", "args": {"x": 1}},
        _lease_with_exec(tmp_path),
        watchdog=_Watchdog(tmp_path),
    )

    assert not isinstance(result, ToolError)
    assert isinstance(result, dict)
    assert result["skill_id"] == "pytest-runner"
    assert result["exit_code"] == 0
    assert '"ok"' in str(result["stdout"])


def test_skill_run_missing_capability_returns_body_gate_error(
    monkeypatch, tmp_path: Path
) -> None:
    # AC4：lease 缺 code_execution 能力 → 本体闸门返回 skill_execution_disabled
    from approval import ApprovalDecision

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    _make_active_skill(SkillStore(tmp_path), "pytest-runner")
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)
    lease = from_trigger(
        "user",
        task_id="task-1",
        capabilities={"fs": {"project_root": str(tmp_path), "read": [], "write": []}},
    )

    result = registry.execute_tool(
        "skill_run",
        {"skill_id": "pytest-runner", "args": {}},
        lease,
        watchdog=_Watchdog(tmp_path),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "skill_execution_disabled"


def test_skill_run_archived_skill_returns_permission_error(
    monkeypatch, tmp_path: Path
) -> None:
    # AC5：archived skill → skill_archived PERMISSION 错误
    from approval import ApprovalDecision

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    store = SkillStore(tmp_path)
    _make_active_skill(store, "pytest-runner")
    store.archive_skill("pytest-runner")
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    result = registry.execute_tool(
        "skill_run",
        {"skill_id": "pytest-runner", "args": {}},
        _lease_with_exec(tmp_path),
        watchdog=_Watchdog(tmp_path),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "skill_archived"


def test_skill_run_nonexistent_skill_surfaces_real_error(
    monkeypatch, tmp_path: Path
) -> None:
    # AC5：skill 不存在 → 传出真实错误，不伪成功（load_skill 找不到文件诚实失败）
    from approval import ApprovalDecision

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    SkillStore(tmp_path)  # 建空 skills 根，不造任何 skill
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    result = registry.execute_tool(
        "skill_run",
        {"skill_id": "does-not-exist", "args": {}},
        _lease_with_exec(tmp_path),
        watchdog=_Watchdog(tmp_path),
    )

    assert isinstance(result, ToolError)


def test_skill_run_no_data_root_fails_closed(monkeypatch, tmp_path: Path) -> None:
    # AC5b：confirm-risk 工具缺 data root 时由统一审批边界先行拒绝
    from approval import ApprovalDecision

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    result = registry.execute_tool(
        "skill_run",
        {"skill_id": "pytest-runner", "args": {}},
        _lease_with_exec(tmp_path),
        watchdog=_Watchdog(None),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "approval_data_root_required"


def test_skill_search_returns_recall_list(monkeypatch, tmp_path: Path) -> None:
    # AC6：skill_search 经 registry → 返回召回列表
    _make_active_skill(SkillStore(tmp_path), "pytest-runner")
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    result = registry.execute_tool(
        "skill_search",
        {"query": "pytest"},
        from_trigger("user", task_id="task-1"),
        watchdog=_Watchdog(tmp_path),
    )

    assert isinstance(result, dict)
    assert result["items"][0]["skill_id"] == "pytest-runner"


def test_skill_search_empty_returns_empty_list(tmp_path: Path) -> None:
    # AC6：无 skill 时返回空列表而非报错
    SkillStore(tmp_path)  # 建空 skills 根
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    result = registry.execute_tool(
        "skill_search",
        {"query": "pytest"},
        from_trigger("user", task_id="task-1"),
        watchdog=_Watchdog(tmp_path),
    )

    assert result["items"] == [] and result["has_more"] is False


def test_skill_search_no_data_root_fails_closed(tmp_path: Path) -> None:
    # AC5b：skill_search 收到 __data_root__ 非 str/Path → 清晰 ToolError
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    result = registry.execute_tool(
        "skill_search",
        {"query": "pytest"},
        from_trigger("user", task_id="task-1"),
        watchdog=_Watchdog(None),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.INVALID_INPUT
    assert result.message == "skill_search_no_data_root"


def test_skill_invoke_stays_retired() -> None:
    # AC7：skill_invoke 仍无注册
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    assert registry.get("skill_invoke") is None
