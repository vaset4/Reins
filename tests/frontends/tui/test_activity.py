"""活动浮层的可读结果、明确控制与草稿隔离。

作者：xxx
时间：2026-09-30 12:00:00
"""

import asyncio

from frontends.tui.activity import ActivityScreen, activity_text
from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from frontends.tui.paged_text import PagedText
from frontends.tui.settings import SettingsScreen
from tests.frontends.tui.test_interactive import DisplayBridge


def activity_data():
    """提供已知失败和未知执行两类后台事实；参数：无；返回：快照。"""
    return {
        "session_id": "session-ui",
        "current": {
            "run_id": "run-ui",
            "status": "paused",
            "active": False,
            "stopped": False,
            "error": None,
            "scheduled": False,
        },
        "task": {
            "goal": "检查原始资料",
            "status": "active",
            "todos": [{"idx": 0, "content": "比较日期", "status": "blocked"}],
        },
        "members": [
            {
                "name": "资料调查",
                "task": "寻找凭据",
                "status": "failed",
                "backend": "internal",
                "session_id": "session-child",
                "run_id": "run-child",
                "cancel_requested": False,
                "output": "原始文件不存在",
            }
        ],
        "work": [],
        "host_status": "running",
        "errors": {},
        "unread_notifications": 2,
    }


class ActivityBridge(DisplayBridge):
    """仅替换远程传输，保留真实Textual活动交互。"""

    def __init__(self):
        """记录明确提交动作；参数：无；返回：无。"""
        super().__init__()
        self.controls = []

    def browse(self, method, **params):
        """返回对应入口的后端事实；参数：查询动作；返回：快照。"""
        return activity_data() if method == "activity" else {"sessions": []}

    def control_activity(self, choice):
        """记录原始身份且只返回真实接纳形状；参数：选择；返回：确认。"""
        self.controls.append(choice)
        return {"session_id": choice["session_id"], "message": "已接纳停止请求"}

    def settings(self):
        """返回执行设置，快捷键测试不触碰真实配置；参数：无；返回：空配置目录。"""
        return {
            "session_id": "session-ui",
            "profiles": [],
            "model": None,
            "mcp": None,
            "approval_mode": "workspace",
            "input_model": {},
        }


def test_f7_shows_plan_and_failure_and_cancels_original_run_without_losing_draft():
    """活动快捷键展示错误及计划，停止携带原运行且保留草稿；参数：无；返回：无。"""

    async def scenario():
        """运行实际界面及按钮事件；参数：无；返回：无。"""
        bridge = ActivityBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(110, 42)) as pilot:
            app.receive(
                "snapshot",
                {"session_id": "session-ui", "history": [], "status": "paused"},
            )
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("尚未提交的中文\n第二行")
            await pilot.press("f7")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ActivityScreen)
            text = app.screen.query_one(PagedText).text
            assert "[blocked] 比较日期" in text and "原始文件不存在" in text
            assert "停止当前运行会同时向所属子执行传递取消" in text
            await pilot.click("#activity-cancel")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert bridge.controls == [
                {"session_id": "session-ui", "run_id": "run-ui", "action": "cancel"}
            ]
            assert composer.text == "尚未提交的中文\n第二行"

    asyncio.run(scenario())


def test_activity_continue_and_close_preserve_draft_on_narrow_screen():
    """窄屏继续会提交明确模型要求，关闭不会停止；参数：无；返回：无。"""

    async def scenario():
        """验证窄屏控件与草稿；参数：无；返回：无。"""
        bridge = ActivityBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(70, 28)) as pilot:
            app.receive(
                "snapshot",
                {"session_id": "session-ui", "history": [], "status": "paused"},
            )
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("保留草稿")
            await pilot.press("f7")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert bridge.controls == []
            await pilot.press("f7")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#activity-continue")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert bridge.controls[0]["action"] == "submit"
            assert "先核对已有结果与停止原因" in bridge.controls[0]["text"]
            assert composer.text == "保留草稿"

    asyncio.run(scenario())


def test_unknown_child_state_is_not_rendered_as_running_or_success():
    """无结束证据不能被补成执行中或成功；参数：无；返回：无。"""
    data = activity_data()
    data["members"][0].update(status="unknown", output="")
    text = activity_text(data)
    assert "状态未知（尚无结束回执，未查询执行存活）" in text
    assert "运行结果" not in text


def test_f6_opens_settings_from_composer_focus_without_changing_selection():
    """F6在输入框焦点下打开设置，不触发TextArea原有选择快捷键；参数：无；返回：无。"""

    async def scenario():
        """使用真实快捷键与浮层生命周期；参数：无；返回：无。"""
        app = InteractiveTui(ActivityBridge())
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.receive(
                "snapshot",
                {"session_id": "session-ui", "history": [], "status": "idle"},
            )
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("保留选择和草稿")
            composer.move_cursor((0, 3))
            selection = composer.selection
            composer.focus()
            await pilot.press("f6")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert composer.text == "保留选择和草稿" and composer.selection == selection

    asyncio.run(scenario())
