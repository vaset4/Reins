"""【审查修复】【跨目录执行】实际目录参与协调但不扩大恢复承诺。

作者：xxx
时间：2026-10-03 11:00:00
"""

from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_stage8_file_capture import capture_env as _capture_env, execute, points
from tools.types import ToolError
from tools.workspace_coordination import workspace_write_window

capture_env = _capture_env


@pytest.mark.parametrize("external_argument", ["cwd", "absolute_path"])
def test_external_execution_conflicts_with_existing_writer(
    capture_env, external_argument
):
    """外部目录或显式绝对目标被占用时不能启动执行；参数：环境/目标写法；返回：无。"""
    env = capture_env
    external = env["root"].parent / "external"
    external.mkdir()
    capabilities = env["lease"].capabilities
    env["lease"] = replace(
        env["lease"],
        capabilities={
            **capabilities,
            "fs": {
                **capabilities["fs"],
                "read": [str(env["root"]), str(external)],
                "write": [str(env["root"]), str(external)],
            },
        },
    )
    target = external / "effect.txt"
    args = (
        {"command": "echo changed > effect.txt", "cwd": str(external)}
        if external_argument == "cwd"
        else {"command": f'echo changed > "{target}"', "cwd": str(env["root"])}
    )
    with workspace_write_window(external, subtree=True, owner="existing-writer"):
        result = execute(env, "terminal_tool", args)
        assert isinstance(result, ToolError), result
        assert (
            "busy" in result.message
            and result.details["execution_state"] == "not_started"
        )
        assert not target.exists()
    result = execute(env, "terminal_tool", args)
    assert isinstance(result, dict) and result["exit_code"] == 0, result
    assert target.read_text().strip() == "changed"
    point = points(env)[0]
    assert point["workspace_root"] == str(env["root"])
    assert all(
        not Path(entry["path"]).is_relative_to(external) for entry in point["entries"]
    )


def test_ungranted_external_cwd_is_rejected(capture_env):
    """新增协调不授予目录写权限；参数：环境；返回：无。"""
    external = capture_env["root"].parent / "denied"
    external.mkdir()
    result = capture_env["registry"].prepare_tool_execution(
        "terminal_tool",
        {"command": "echo changed > effect.txt", "cwd": str(external)},
        capture_env["lease"],
    )
    assert isinstance(result, ToolError)
    assert not (external / "effect.txt").exists()
