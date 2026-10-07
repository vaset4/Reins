"""历史模型快照未带强度字段时仍使用历史默认。

作者：xxx
"""

from contextlib import closing
from types import SimpleNamespace

import pytest

from app.background.sessions import BackgroundSession, SessionRecord, SessionServices
from app.scheduled_run import create_scheduler
from llm.profiles import ModelProfile
from llm.providers.openai_chat import OpenAIChatAdapter
from schedules.store import ScheduleStore
from runtime.workspaces import WorkspaceStore
from tests.test_scheduled_execution import TEST_NOW
from tools.tool_registry import ToolRegistry


@pytest.mark.parametrize("boundary", ["background", "scheduled"])
@pytest.mark.parametrize("effort", [None, "high"])
def test_persisted_snapshot_keeps_old_default_at_execution_boundary(
    tmp_path, monkeypatch, boundary, effort
):
    """旧字段缺失与显式强度经真实执行入口后进入请求；参数：目录、替换器、入口、档位；返回：无。"""
    import app.cli as cli

    profile = ModelProfile(
        "selected", "proxy", "http://localhost/v1", "grok-4.7", reasoning_effort="xhigh"
    )
    monkeypatch.setattr(cli, "_load_named_model_profile", lambda name: profile)
    monkeypatch.setattr(
        cli, "SecretsVault", lambda: SimpleNamespace(get=lambda name: None)
    )
    config = {
        "profile_name": "selected",
        "model": "grok-4.7",
        "base_url": "http://localhost/v1",
    }
    if effort is not None:
        config["reasoning_effort"] = effort
    captured = []

    def execute(context, **options):
        """捕获执行入口构建客户端的真实请求，不调用远端；参数：上下文及依赖；返回：完成响应。"""
        request = options["llm_client"].prepare_request("核对资料", {}).request
        captured.append(OpenAIChatAdapter().build_request(request, model_id="grok-4.7"))
        return SimpleNamespace(status="done", output="完成", task_id=context.task_id)

    def factory(options):
        """通过真实模型工厂重建冻结配置；参数：公开快照；返回：客户端。"""
        return cli.build_llm_client(options, project_root=tmp_path)

    if boundary == "background":
        monkeypatch.setattr("app.background.sessions.execute_context", execute)
        session = BackgroundSession(
            SessionRecord("legacy-model"),
            SessionServices(tmp_path, tmp_path, factory, ToolRegistry),
        )
        try:
            session.submit("核对资料", input_id="legacy-input", model_config=config)
            assert session.runtime.wait_idle(5)
        finally:
            session.close()
    else:
        monkeypatch.setattr("app.scheduled_run.execute_context", execute)
        with closing(ScheduleStore(tmp_path)) as store:
            store.create_schedule(
                "legacy-model",
                "at:2026-09-14T19:00:00+08:00",
                kind="work",
                workspace_id=WorkspaceStore(tmp_path).register(tmp_path).workspace_id,
                prompt="核对资料",
                model_config=config,
            )
        with closing(
            create_scheduler(
                project_root=tmp_path,
                data_root=tmp_path,
                llm_factory=factory,
                registry_factory=ToolRegistry,
            )
        ) as scheduler:
            scheduler.run_due_jobs(now=TEST_NOW)
    assert len(captured) == 1
    assert captured[0].get("reasoning_effort") == effort
    assert (
        cli.build_llm_client(
            {"profile_name": "selected"}
        ).resolved_target.reasoning_effort
        == "xhigh"
    )
