"""设置通过真实模型工厂、后台执行与共享命令交接。

作者：xxx
时间：2026-09-30 10:00:00
"""

import asyncio
import json
from threading import Event, RLock
from types import SimpleNamespace

import pytest

from app.background.frontend import BackgroundSessionHost
from app.background.sessions import BackgroundSession, SessionRecord, SessionServices
from app.cli import build_llm_client
from app.repl.session_host import SessionHostConfig
from frontends.tui.bridge import TuiBridge
from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from frontends.tui.settings import SettingsScreen, connection_description
from llm.profiles import load_model_profiles
from llm.public_config import public_model_config
from runtime.default_capabilities import build_local_agent_capabilities
from runtime.lease import from_trigger
from runtime.workspaces import WorkspaceStore
from tests.frontends.tui.test_interactive import DisplayBridge
from tests.test_mcp_standard_integration import _configured_service
from tools.mcp_client.registry import attach_mcp_registry
from tools.tool_registry import ToolRegistry


def configured_bridge(tmp_path, monkeypatch):
    """隔离所有配置读取，只替换凭据提供者；参数：临时目录与补丁；返回：真实桥接、宿主和回执。"""
    path = tmp_path / "models.json"
    provider = {
        "model_provider": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "credential": "test-key",
        "api_mode": "chat_completions",
        "models": {"old": {"model": "model-old"}, "new": {"model": "model-new"}},
    }
    path.write_text(
        json.dumps(
            {
                "active_provider": "proxy",
                "active_model": "old",
                "providers": {"proxy": provider},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("llm.profiles.MODELS_JSON_CONFIG_PATH", path)
    monkeypatch.setattr(
        "app.cli.SecretsVault", lambda: SimpleNamespace(get=lambda _: "fixture-secret")
    )
    events = []
    old = build_llm_client({"profile_name": "proxy:old"}, project_root=tmp_path)
    bridge = TuiBridge(
        project_root=tmp_path,
        data_root=tmp_path,
        llm_client=old,
        event_sink=lambda kind, data: events.append((kind, data)),
    )
    bridge.state.session_id = "session-settings"
    host = object.__new__(BackgroundSessionHost)
    host.config = SessionHostConfig(
        bridge.state, ToolRegistry(), old, tmp_path, tmp_path
    )
    host.session_id = bridge.state.session_id
    host._connection_lock = RLock()
    host._input_error = None
    host.prepare_command = lambda _: None
    host.handle_control = lambda _: False
    bridge.host, bridge.store, bridge.registry = host, object(), host.config.registry
    return bridge, host, events


def test_shared_model_choice_rebuilds_real_client_and_preserves_previous(
    tmp_path, monkeypatch
):
    """真实配置切换经工厂进入后续输入，原客户端不变；参数：隔离根；返回：无。"""
    bridge, host, events = configured_bridge(tmp_path, monkeypatch)
    previous = host.config.llm_client
    bridge.submit("/model profile use proxy:new")
    assert load_model_profiles().active == "proxy:new"
    assert public_model_config(host.config.llm_client)["model"] == "model-new"
    assert public_model_config(previous)["model"] == "model-old"
    assert host.config.llm_client is bridge.llm_client
    sent = []
    host.client = SimpleNamespace(
        call=lambda method, **params: sent.append((method, params)) or {}
    )
    host._completion_menu = SimpleNamespace(choose=lambda _: None)
    host._sync = lambda _: None
    host._present = lambda _: None
    host.submit("使用所选模型")
    assert sent[0][0] == "submit" and sent[0][1]["model_config"]["model"] == "model-new"
    assert "api_key" not in sent[0][1]["model_config"]
    assert any(kind == "input-model" for kind, _ in events)
    assert "fixture-secret" not in str(events)


def test_failed_reload_keeps_old_input_model_and_saved_state(tmp_path, monkeypatch):
    """凭据解析失败时选择不落盘，旧输入模型不变；参数：隔离根；返回：无。"""
    bridge, host, events = configured_bridge(tmp_path, monkeypatch)
    previous = host.config.llm_client
    monkeypatch.setattr(
        "app.cli.SecretsVault",
        lambda: (_ for _ in ()).throw(OSError("fixture vault failure")),
    )
    with pytest.raises(RuntimeError, match="vault failure"):
        bridge.submit("/model profile use proxy:new")
    assert load_model_profiles().active == "proxy:old"
    assert host.config.llm_client is previous and bridge.llm_client is previous
    assert not any(kind == "input-model" for kind, _ in events)


def test_execution_settings_observe_real_mcp_and_do_not_replace_running_model(
    tmp_path, monkeypatch
):
    """后台设置读取真实SDK连接，新接纳配置不冒充当前执行模型；参数：隔离根；返回：无。"""
    entered, release = Event(), Event()
    client = SimpleNamespace(
        resolved_target=SimpleNamespace(model="running-model", provider="fixture")
    )
    options_seen = []

    def make_model(options):
        """记录真实后台传入的公开配置；参数：运行配置；返回：测试模型。"""
        options_seen.append(options)
        return client

    def execute(context, **kwargs):
        """在真实连接生命周期内等待显示查询；参数：执行上下文；返回：明确结束状态。"""
        entered.set()
        assert release.wait(10)
        return SimpleNamespace(status="done")

    with _configured_service(tmp_path, monkeypatch, "stdio"):
        registry = ToolRegistry()
        lease = from_trigger(
            "user", capabilities=build_local_agent_capabilities(tmp_path, tmp_path)
        )
        owner = attach_mcp_registry(lease, registry)
        monkeypatch.setattr("app.background.sessions.execute_context", execute)
        services = SessionServices(
            tmp_path, tmp_path / "data", make_model, lambda: registry
        )
        session = BackgroundSession(SessionRecord("session-settings"), services)
        assert session.settings()["mcp"] is None
        try:
            session.submit(
                "检查连接",
                input_id="input-settings",
                model_config={"model": "running-model"},
            )
            assert entered.wait(5)
            status = session.settings()
            assert status["mcp"] == owner.describe()
            assert status["mcp"]["servers"][0]["status"] == "running"
            assert status["mcp"]["servers"][0]["protocol_version"]
            session.submit(
                "后续要求",
                input_id="input-next-settings",
                model_config={"model": "next-model"},
            )
            assert session.settings()["model"]["model"] == "running-model"
            assert options_seen == [
                {"model": "running-model", "reasoning_effort": "default"}
            ]
            assert session.settings()["approval_mode"] == "workspace"
        finally:
            release.set()
            assert session.runtime.wait_idle(10)
            session.close()
        assert session.settings()["mcp"] is None
        assert "未知" in connection_description(session.settings())


def test_settings_modal_preserves_draft_and_returns_explicit_shared_command():
    """设置选择通过共用提交，浮层不会覆盖草稿；参数：无；返回：无。"""

    async def scenario():
        """运行真实Textual选择交互；参数：无；返回：无。"""
        from textual.widgets import Button, Select, Static

        bridge = DisplayBridge()
        bridge.release.set()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.pause()
            app.query_one(Composer).load_text("保留我的草稿")
            data = {
                "session_id": "",
                "input_model": {},
                "profiles": [],
                "approval_mode": "workspace",
            }
            app.push_screen(SettingsScreen(data), app.settings_selected)
            await pilot.pause()
            guidance = "\n".join(
                str(widget.content) for widget in app.screen.query(Static)
            )
            assert "~/.reins/models.json" in guidance and "~/.reins/.env" in guidance
            assert (
                "models.yaml" not in guidance and "/model profile set" not in guidance
            )
            app.screen.query_one("#mode-choice", Select).value = "read_only"
            app.screen.query_one("#apply-mode", Button).scroll_visible(immediate=True)
            app.screen.query_one("#apply-mode", Button).focus()
            await pilot.pause()
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert bridge.submitted == ["/mode read_only"]
            assert app.query_one(Composer).text == "保留我的草稿"

    asyncio.run(scenario())


def test_settings_rpc_reads_actual_mode_and_shared_command_changes_it(
    tmp_path, monkeypatch
):
    """设置RPC读取实际会话权限，切换仍经过原授权入口；参数：隔离根；返回：无。"""
    from app.background.server import BackgroundServer

    monkeypatch.setattr("approval._config_path", lambda: tmp_path / "approval.yaml")
    services = SessionServices(
        tmp_path, tmp_path / "data", lambda _: None, ToolRegistry
    )
    session = BackgroundSession(SessionRecord("session-mode"), services)
    service = SimpleNamespace(
        attach=lambda _: session,
        workspaces=WorkspaceStore(services.data_root),
        services=services,
    )
    server = BackgroundServer(service, "fixture-token")
    try:
        before = server.dispatch("settings", {"session_id": "session-mode"})
        assert before["approval_mode"] == "workspace" and before["model"] is None
        for mode in ("read_only", "auto", "workspace"):
            server.dispatch(
                "approval_control",
                {
                    "session_id": "session-mode",
                    "command": f"/mode {mode}",
                    "action_id": mode,
                },
            )
            assert (
                server.dispatch("settings", {"session_id": "session-mode"})[
                    "approval_mode"
                ]
                == mode
            )
        assert session.pending_approval is None
    finally:
        server.server_close()
        session.close()
