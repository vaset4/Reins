"""验证敏感材料在同步客户端、后台宿主和实际发布边界的生命周期。

作者：xxx
时间：2026-09-24 23:00:00
"""

from scripts.testing.llm import from_test_native_tool_then_final
from contextlib import closing
from dataclasses import replace
import pytest

from llm.messages import ToolCallPart
from llm.types import LLMPlan
from runtime.agent_loop import AgentLoop
from runtime.workspaces import WorkspaceStore
from runtime.default_capabilities import build_local_agent_capabilities
from runtime.lease import from_trigger
from runtime.types import RunContext, RunToolsRequest, Trigger
from runtime.tool_operations import ToolOperationStore
from tasks.store import TaskStore
from tests.support.approval import install_approval
from approval import ApprovalDecision
from tools.builtin_tools import build_tool_registry


class SynchronousClient:
    """仅实现公开同步入口，复用生产主循环。"""

    def __init__(self, *, private_key=False):
        """选择配置或私钥输入；传参：场景；返回：无。"""
        self.private_key = private_key

    def plan(self, task, *, context):
        """返回包含新值与正文副本的候选；传参：任务和上下文；返回：计划。"""
        args = {"path": ".env", "content": "TOKEN=SYNC_PRIVATE\n"}
        if self.private_key:
            args = {
                "path": "plain.txt",
                "content": "-----BEGIN PRIVATE KEY-----\nSYNC_PRIVATE\n-----END PRIVATE KEY-----\n",
            }
        return LLMPlan(
            run_tools_request=RunToolsRequest(
                action="file_write", tool_name="file_write", arguments=args
            ),
            reasoning_content="设置 SYNC_PRIVATE",
            raw_model_response={"text": "SYNC_PRIVATE"},
        )

    def continue_from_run_tools(self, task, result, *, context):
        """执行后结束当前回复；传参：任务、结果和上下文；返回：计划。"""
        return LLMPlan(final_output="已处理")


@pytest.mark.parametrize("private_key", [False, True])
def test_synchronous_client_is_protected_before_session_and_operation_records(
    tmp_path, monkeypatch, private_key
):
    """同步客户端不会绕过共同证据边界；传参：目录与替换器；返回：无。"""
    data = tmp_path / "data"
    with closing(TaskStore(data)) as tasks:
        task = tasks.create_task("配置令牌")
    install_approval(monkeypatch, lambda _: ApprovalDecision.ONCE)
    lease = from_trigger(
        "user",
        task_id=task.task_id,
        capabilities=build_local_agent_capabilities(tmp_path, data),
    )
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": task.goal},
        capability_lease=lease,
    )
    WorkspaceStore(data).bind_session(context.session_id, tmp_path)
    registry = build_tool_registry(repo_root=tmp_path, data_root=data)
    list(
        AgentLoop(
            data,
            llm_client=SynchronousClient(private_key=private_key),
            tool_registry=registry,
        ).run_stream(context)
    )
    if private_key:
        assert not (tmp_path / "plain.txt").exists()
    else:
        assert (tmp_path / ".env").read_bytes() == b"TOKEN=SYNC_PRIVATE\n"
    for file in data.rglob("*"):
        if file.is_file():
            assert b"SYNC_PRIVATE" not in file.read_bytes(), str(file.relative_to(data))


def test_background_rebuild_retains_view_until_host_shutdown(tmp_path):
    """真实后台每轮重建注册表后仍能编辑原视图；传参：目录；返回：无。"""
    project = tmp_path / "project"
    project.mkdir()
    data = tmp_path / "data"
    path = project / ".env"
    path.write_bytes(b"TOKEN=BACKGROUND_PRIVATE\nPORT=3000\n")
    first = from_test_native_tool_then_final(
        [ToolCallPart("read", "file_read", {"path": ".env"})], "已读取"
    )
    from app.background.sessions import (
        BackgroundSession,
        SessionRecord,
        SessionServices,
    )
    from functools import partial

    services = SessionServices(
        project,
        data,
        lambda _options: first,
        partial(build_tool_registry, repo_root=project, data_root=data),
    )
    session = BackgroundSession(SessionRecord("session-background"), services)
    try:
        session.submit("读取配置", input_id="read", model_config={})
        assert session.runtime.wait_idle(10)
        # 1. 从实际宿主的读取结果取编号，不人工构造另一个视图
        records = ToolOperationStore(data).for_session(session.record.session_id)
        view = next(
            row["result"]["meta"]
            for row in records
            if row["call"]["tool_name"] == "file_read"
        )
        second = from_test_native_tool_then_final(
            [
                ToolCallPart(
                    "patch",
                    "file_patch",
                    {
                        "path": ".env",
                        "view_id": view["view_id"],
                        "expected_sha256": view["content_sha256"],
                        "old_text": "PORT=3000",
                        "new_text": "PORT=4000",
                    },
                )
            ],
            "已修改",
        )
        session.services = replace(session.services, make_llm=lambda _: second)
        session.submit("把端口改为4000", input_id="patch", model_config={})
        assert session.runtime.wait_idle(10)
        assert path.read_bytes() == b"TOKEN=BACKGROUND_PRIVATE\nPORT=4000\n"
    finally:
        session.close()
    assert not session.redacted_files._views
