"""从用户实际前端入口验证整批决定与会话复用。

作者：xxx
时间：2026-09-24 12:00:00
"""

from scripts.testing.llm import _from_scripted
from functools import partial
from threading import Event

from app.repl.session_host import SessionHost, SessionHostConfig
from app.repl.slash_commands import ReplState
from scripts.testing.llm import _ScriptedTurn
from llm.messages import ToolCallPart
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


def _execute(effects, name, _args):
    """记录真实执行顺序；传参：效果列表、动作和参数；返回：结果正文。"""
    effects.append(name)
    return name


def _services():
    """装配前三项操作及下一轮复用操作；传参：无；返回：目录、模型和效果列表。"""
    effects = []
    registry = ToolRegistry()
    for name in ("a", "b", "c"):
        registry.register(
            ToolDefinition(
                name,
                f"执行动作{name}",
                {},
                "agent",
                ToolRisk.CONFIRM,
                False,
                "logical_scope",
                "builtin",
                idempotent=Idempotent.NO,
                executor=partial(_execute, effects, name),
            )
        )
    client = _from_scripted(
        [
            _ScriptedTurn(
                calls=tuple(ToolCallPart(name, name, {}) for name in ("a", "b", "c"))
            ),
            _ScriptedTurn(text="本批结束"),
            _ScriptedTurn(calls=(ToolCallPart("next-a", "a", {}),)),
            _ScriptedTurn(text="第二轮结束"),
        ]
    )
    return registry, client, effects


def test_local_cli_batch_choice_and_session_reuse(tmp_path, monkeypatch):
    """CLI整批展示只执行A/C，下一轮复用当前宿主对A的授权；传参：目录和替换器；返回：无。"""
    registry, client, effects = _services()
    host = SessionHost(
        SessionHostConfig(
            ReplState(session_id="session"), registry, client, tmp_path, tmp_path
        )
    )
    reached, shown = Event(), []
    original = host.approvals._present_batch

    def present(identity, batch):
        """记录真实界面已展示的批次；传参：交互身份与批次；返回：无。"""
        original(identity, batch)
        shown.append(identity)
        reached.set()

    monkeypatch.setattr(host.approvals, "_present_batch", present)
    monkeypatch.setattr("approval.batch._batch_backend", host.approvals.request_batch)
    monkeypatch.setattr("approval._config_path", lambda: tmp_path / "config.yaml")
    try:
        host.submit("执行已选动作")
        assert reached.wait(10) and effects == []
        host.handle_control(f"/approve {shown[0]} 1=session 2=deny 3=once")
        host.wait_idle()
        assert effects == ["a", "c"]
        host.submit("再次执行A")
        host.wait_idle()
        assert effects == ["a", "c", "a"] and len(shown) == 1
    finally:
        host.close()


def test_interactive_tui_batch_choice_has_no_implicit_default(tmp_path, monkeypatch):
    """正式浮层缺项不派发，完整选择只运行授权项；传参：目录和替换器；返回：无。"""
    import asyncio
    import pytest
    from frontends.tui.approval_dialog import ApprovalScreen
    from frontends.tui.interactive import InteractiveTui
    from tests.frontends.tui.test_interactive import DisplayBridge

    async def scenario():
        """把当前浮层接到真实执行宿主审批通道；参数：无；返回：无。"""
        registry, client, effects = _services()
        host = SessionHost(
            SessionHostConfig(
                ReplState(session_id="session"), registry, client, tmp_path, tmp_path
            )
        )
        reached, snapshot, batches = Event(), {}, []
        original = host.approvals._present_batch

        def present(identity, batch):
            """保存真实批次身份并通知界面；传参：身份与批次；返回：无。"""
            original(identity, batch)
            batches.append(batch)
            snapshot.update(
                identity=identity,
                batch_id=batch.batch_id,
                requests=[
                    {
                        "tool": item.tool,
                        "args": dict(item.args),
                        "force_confirmation": True,
                    }
                    for item in batch.requests
                ],
            )
            reached.set()

        monkeypatch.setattr(host.approvals, "_present_batch", present)
        monkeypatch.setattr(
            "approval.batch._batch_backend", host.approvals.request_batch
        )
        monkeypatch.setattr("approval._config_path", lambda: tmp_path / "config.yaml")
        try:
            host.submit("执行已选动作")
            assert await asyncio.to_thread(reached.wait, 10)
            app = InteractiveTui(DisplayBridge())
            async with app.run_test(size=(120, 45)) as pilot:
                app.push_screen(ApprovalScreen(snapshot), host.approvals.answer)
                await pilot.pause()
                screen = app.screen
                await pilot.press("enter")
                assert effects == [] and app.screen is screen
                with pytest.raises(ValueError, match="missing"):
                    host.approvals.answer(f"/approve {snapshot['identity']} 1=once")
                assert effects == [] and host.approvals._batch is batches[0]
                assert host.approvals._pending == snapshot["identity"]
                await pilot.press("down", "enter", "enter", "enter")
                await asyncio.to_thread(host.wait_idle)
                assert effects == ["a", "c"]
        finally:
            host.close()

    asyncio.run(scenario())


def test_background_frontend_reconnection_submits_original_batch(tmp_path, monkeypatch):
    """断开显示后重连原审批，经生产RPC只执行用户选项；传参：目录和替换器；返回：无。"""
    from app.background.frontend import BackgroundSessionHost
    from app.background.server import BackgroundServer
    from app.background.service import BackgroundService
    from app.background.sessions import SessionServices

    registry, client, effects = _services()
    service = BackgroundService(
        SessionServices(tmp_path, tmp_path, lambda _: client, lambda: registry)
    )
    session = service.create_session(tmp_path)
    server = BackgroundServer(service, "synthetic-control-token")
    reached, shown = Event(), []
    original = session.approvals._present_batch

    def present(identity, batch):
        """记录后台实际等待的唯一批次；传参：身份和批次；返回：无。"""
        original(identity, batch)
        shown.append(identity)
        reached.set()

    class Connection:
        """只替换本机传输，保留生产服务分派。"""

        def call(self, method, **params):
            """分派真实RPC方法；传参：方法和参数；返回：服务器结果。"""
            return server.dispatch(method, params)

    monkeypatch.setattr(session.approvals, "_present_batch", present)
    monkeypatch.setattr("approval.batch._batch_backend", service.approve_batch)
    monkeypatch.setattr("approval._config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(BackgroundSessionHost, "_connect", lambda _: Connection())
    config = SessionHostConfig(
        ReplState(session_id=session.record.session_id),
        registry,
        client,
        tmp_path,
        tmp_path,
    )
    first = BackgroundSessionHost(config)
    second = None
    try:
        first.submit("执行已选动作")
        assert reached.wait(10)
        first.close()
        second = BackgroundSessionHost(config)
        assert session.snapshot()["approval"]["identity"] == shown[0] and effects == []
        second.handle_control(f"/approve {shown[0]} 1=session 2=deny 3=once")
        assert session.runtime.wait_idle(10) and effects == ["a", "c"]
        second.submit("再次执行A")
        assert (
            session.runtime.wait_idle(10)
            and effects == ["a", "c", "a"]
            and len(shown) == 1
        )
    finally:
        first.close()
        if second is not None:
            second.close()
        session.close()
        server.server_close()
