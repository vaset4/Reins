"""真实审批浮层的选择与草稿保护回归。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from approval import ApprovalDecision, ApprovalRequest
from approval.batch_types import ApprovalBatch
from approval.channel import ApprovalChannel
from runtime.lease import Lease

from textual.widgets import OptionList

from frontends.tui.approval_dialog import ApprovalScreen
from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from tests.frontends.tui.test_interactive import DisplayBridge
from app.background.events import EventBuffer
from runtime.stream_events import ModelRequestStarted, ToolApprovalRequested


def test_batch_keyboard_selection_submits_identity_without_touching_draft():
    """方向键逐项选择后只提交原批次且保留草稿；参数：无；返回：无。"""

    async def scenario():
        bridge = DisplayBridge()
        bridge.release.set()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("未发送草稿")
            request = {
                "identity": "original",
                "batch_id": "batch",
                "requests": [
                    {
                        "tool": "exec",
                        "args": {"command": "first"},
                        "force_confirmation": True,
                    },
                    {
                        "tool": "exec",
                        "args": {"command": "second"},
                        "force_confirmation": True,
                    },
                ],
            }
            app.receive(
                "snapshot",
                {
                    "session_id": "session",
                    "status": "waiting_approval",
                    "history": [],
                    "approval": request,
                },
            )
            await pilot.pause()
            assert isinstance(app.screen, ApprovalScreen)
            await pilot.press("enter", "down", "enter", "enter")
            await app.workers.wait_for_complete()
            assert bridge.submitted == ["/approve original 1=once 2=deny"]
            assert composer.text == "未发送草稿"

    asyncio.run(scenario())


def test_batch_dialog_reaches_real_channel_once_and_rejects_stale_answer(tmp_path):
    """浮层决定经真实审批通道只消费一次，旧编号不能复用；参数：隔离目录；返回：无。"""

    async def scenario():
        shown, effects = Event(), []
        snapshot = {}

        def present(identity, batch):
            """保存通道实际生成身份；参数：请求编号和批次；返回：无。"""
            snapshot.update(
                identity=identity,
                batch_id=batch.batch_id,
                requests=[
                    {
                        "tool": request.tool,
                        "args": dict(request.args),
                        "force_confirmation": True,
                    }
                    for request in batch.requests
                ],
            )
            shown.set()

        channel = ApprovalChannel(lambda *_: None, present_batch=present)
        request = ApprovalRequest(
            "exec",
            {},
            "confirm",
            Lease(task_id="task"),
            tmp_path,
            "执行",
            operation_id="op",
        )

        def execute():
            """用真实通道阻止授权前副作用；参数：无；返回：无。"""
            answer = channel.request_batch(ApprovalBatch("batch", (request,)))
            if answer.choices[0].decision is ApprovalDecision.ONCE:
                effects.append("executed")

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            assert await asyncio.to_thread(shown.wait, 2)
            app = InteractiveTui(DisplayBridge())
            async with app.run_test(size=(120, 45)) as pilot:
                await pilot.pause()
                app.push_screen(ApprovalScreen(snapshot), channel.answer)
                await pilot.pause()
                assert effects == []
                await pilot.press("enter", "enter")
                await asyncio.to_thread(future.result, 2)
                assert effects == ["executed"]
                with pytest.raises(ValueError, match="已结束"):
                    channel.answer(f"/approve {snapshot['identity']} 1=once")

    asyncio.run(scenario())


def test_single_mouse_and_cancel_are_explicit():
    """鼠标选择允许、Escape稍后不产生授权；参数：无；返回：无。"""

    async def scenario():
        app = InteractiveTui(DisplayBridge())
        answers = []
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            request = {
                "identity": "one",
                "tool": "exec",
                "args": {},
                "force_confirmation": True,
            }
            app.push_screen(ApprovalScreen(request), answers.append)
            await pilot.pause()
            assert app.screen.query_one(OptionList).option_count == 3
            await pilot.click("#approval-options", offset=(3, 1))
            await pilot.pause()
            assert answers == ["/approve one once"]
            app.push_screen(ApprovalScreen(request), answers.append)
            await pilot.pause()
            await pilot.press("escape")
            assert answers == ["/approve one once", None]

    asyncio.run(scenario())


def test_activity_refresh_keeps_approval_focus_and_command_draft():
    """审批中的活动刷新不抢焦点，Esc稍后后命令草稿仍在；参数：无；返回：无。"""

    async def scenario():
        """用真实事件缓存驱动等待到审批的快照；参数：无；返回：无。"""
        bridge = DisplayBridge()
        app = InteractiveTui(bridge)
        events = EventBuffer()
        events.emit(ModelRequestStarted("request", 1, 3), run_id="run")
        async with app.run_test(size=(100, 40)) as pilot:
            app.receive(
                "snapshot",
                {
                    "session_id": "session",
                    "status": "running",
                    "history": [],
                    **events.read(None),
                },
            )
            await pilot.pause()
            composer = app.query_one(Composer)
            await pilot.press("/", "m")
            events.emit(
                ToolApprovalRequested("exec", {}, "confirm", "执行", "call"),
                run_id="run",
            )
            request = {
                "identity": "pending",
                "tool": "exec",
                "args": {},
                "force_confirmation": True,
            }
            snapshot = {
                "session_id": "session",
                "status": "waiting_approval",
                "approval": request,
                **events.read(app.projection.cursor),
            }
            app.receive("snapshot", snapshot)
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ApprovalScreen)
            options = screen.query_one(OptionList)
            await pilot.press("down")
            app.receive("snapshot", {**snapshot, **events.read(app.projection.cursor)})
            app.refresh_activity()
            await pilot.pause()
            assert (
                app.screen is screen and options.has_focus and options.highlighted == 1
            )
            assert "等待审批" in app.projection.activity_status()
            assert composer.text == "/m" and bridge.submitted == []
            await pilot.press("escape")
            assert composer.text == "/m" and bridge.submitted == []
            composer.focus()
            await pilot.pause()
            assert app.query_one("#command-hints").display

    asyncio.run(scenario())
