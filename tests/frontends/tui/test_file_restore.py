"""文件恢复界面的选择确认、独立取消与过期回复隔离。

作者：xxx
时间：2026-09-30 20:00:00
"""

import asyncio
from threading import Event

import pytest
from textual.widgets import Button, Checkbox, OptionList, Static

from frontends.tui.composer import Composer
from frontends.tui.file_restore import FileRestorePanel
from frontends.tui.interactive import InteractiveTui
from frontends.tui.paged_text import RemotePagedText, TEXT_PAGE_CHARACTERS
from tests.frontends.tui.test_interactive import DisplayBridge
from tests.frontends.tui.test_request_inspection import until

DIFF_TEXT = "文件差异行\n" * TEXT_PAGE_CHARACTERS + "完整差异末尾"


class RestoreBridge(DisplayBridge):
    """只替换RPC传输，正式面板与事件循环保持真实。"""

    def __init__(self):
        """准备可观察的读取与确认；参数：无；返回：无。"""
        super().__init__()
        self.release.set()
        self.calls, self.executions, self.cancellations = [], [], []
        self.job = {}
        self.slow_entered, self.slow_release = Event(), Event()
        self.fail_old = False
        self.slow_old = False
        self.slow_status = False

    def query_file_restore(self, **params):
        """返回已核实的目录、冲突和分段差异；参数：查询；返回：协议数据。"""
        self.calls.append(params)
        action = params["action"]
        if action == "status":
            if self.slow_status and self.job:
                self.slow_entered.set()
                assert self.slow_release.wait(5)
            return dict(self.job)
        if action == "list":
            if self.slow_old and params["session_id"] == "old":
                self.slow_entered.set()
                assert self.slow_release.wait(5)
                if self.fail_old:
                    raise OSError("旧会话读取失败")
            return {
                "workspace_root": "C:/用户项目",
                "turns": [{"input_id": "input", "started_at": "20:00"}],
                "points": [
                    {
                        "point_id": params["session_id"],
                        "entry_count": 4,
                        "status": "complete",
                    }
                ],
                "next_cursor": None,
            }
        if action == "diff":
            offset, limit = params["offset"], params["limit"]
            return {
                "text": DIFF_TEXT[offset : offset + limit],
                "total_chars": len(DIFF_TEXT),
            }
        entries = [
            {
                "entry_id": "safe",
                "path": "原有用户稿.md",
                "state": "ready",
                "default_selected": True,
                "restorable": True,
            },
            {
                "entry_id": "changed",
                "path": "外部修改.txt",
                "state": "conflict",
                "default_selected": False,
                "restorable": True,
            },
            {
                "entry_id": "binary",
                "path": "图片.png",
                "state": "source_unknown",
                "default_selected": False,
                "binary": True,
                "restorable": True,
            },
            {
                "entry_id": "secret",
                "path": ".env",
                "state": "source_unknown",
                "default_selected": False,
                "sensitive": True,
                "restorable": True,
            },
        ]
        result = {"entries": entries, "workspace_root": "C:/用户项目"}
        if action == "preview":
            result.update(
                plan_id="plan",
                can_execute=True,
                confirmation_text="覆盖选中文件；来源待确认的文件可能包含外部编辑",
            )
        return result

    def execute_file_restore(self, **params):
        """仅记录明确提交的计划确认；参数：计划与确认；返回：运行中作业。"""
        self.executions.append(params)
        self.job = {
            "operation_id": "operation",
            "status": "running",
            "entries": [
                {"path": "原有用户稿.md", "status": "restored"},
                {"path": "外部修改.txt", "status": "pending"},
            ],
        }
        return dict(self.job)

    def cancel_file_restore(self, **params):
        """取消仅更新本恢复作业；参数：操作身份；返回：逐项结果。"""
        self.cancellations.append(params)
        self.job = {
            "operation_id": "operation",
            "status": "cancelled",
            "entries": [
                {"path": "原有用户稿.md", "status": "restored"},
                {"path": "外部修改.txt", "status": "cancelled"},
            ],
        }
        return dict(self.job)


def test_restore_selection_requires_fresh_preview_and_keeps_chat_usable():
    """预览和选择不执行，确认只提交一次，取消保留已恢复结果及草稿；参数：无；返回：无。"""

    async def scenario():
        bridge = RestoreBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(140, 50)) as pilot:
            app.receive(
                "snapshot",
                {"session_id": "session", "data_space_id": "space", "history": []},
            )
            await until(pilot, lambda: app.projection.session_id == "session")
            composer = app.query_one(Composer)
            composer.load_text("保留我的草稿")
            await pilot.press("f10")
            panel = app.query_one(FileRestorePanel)
            await until(pilot, lambda: bool(panel.rows))
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            await until(pilot, lambda: "safe" in panel.rows)
            assert panel.choices == {
                "safe": "restore",
                "changed": "keep",
                "binary": "keep",
                "secret": "keep",
            }
            assert bridge.executions == [] and composer.text == "保留我的草稿"
            panel.query_one("#restore-preview", Button).press()
            await until(pilot, lambda: bool(panel.plan))
            assert panel.query_one("#restore-execute", Button).disabled
            panel.query_one(OptionList).highlighted = 1
            panel.query_one("#restore-choice-copy", Button).press()
            await until(pilot, lambda: panel.choices["changed"] == "copy")
            assert (
                not panel.plan and panel.query_one("#restore-execute", Button).disabled
            )
            panel.query_one("#restore-preview", Button).press()
            await until(pilot, lambda: bool(panel.plan))
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            body = panel.query_one(RemotePagedText)
            await until(pilot, lambda: bool(body.text))
            assert body.text == DIFF_TEXT[:TEXT_PAGE_CHARACTERS]
            assert all(
                call.get("limit") == TEXT_PAGE_CHARACTERS
                for call in bridge.calls
                if call["action"] == "diff"
            )
            panel.query_one("#restore-confirm", Checkbox).value = True
            await until(
                pilot, lambda: not panel.query_one("#restore-execute", Button).disabled
            )
            panel.query_one("#restore-execute", Button).press()
            panel.query_one("#restore-execute", Button).press()
            await until(pilot, lambda: panel.job.get("status") == "running")
            assert (
                len(bridge.executions) == 1
                and bridge.executions[0]["confirmation"]["accepted"]
            )
            panel.action_close()
            assert not panel.display and not bridge.cancellations
            composer.focus()
            composer.load_text("恢复期间的新草稿")
            panel.display = True
            panel.query_one("#restore-cancel", Button).press()
            await until(pilot, lambda: panel.job.get("status") == "cancelled")
            assert bridge.cancellations[0]["operation_id"] == "operation"
            assert panel.job["entries"][0]["status"] == "restored"
            assert composer.text == "恢复期间的新草稿" and bridge.submitted == []

    asyncio.run(scenario())


def test_sensitive_restore_needs_specific_confirmation_for_current_plan():
    """敏感配置要本计划的额外确认，改变文件选择即清除确认；参数：无；返回：无。"""

    async def scenario():
        bridge = RestoreBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 44)) as pilot:
            panel = app.query_one(FileRestorePanel)
            panel.open({"session_id": "session", "data_space_id": "space"})
            await until(pilot, lambda: bool(panel.rows))
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            await until(pilot, lambda: "secret" in panel.rows)
            panel.query_one(OptionList).highlighted = 3
            panel.query_one("#restore-choice-restore", Button).press()
            await until(pilot, lambda: panel.choices["secret"] == "restore")
            panel.query_one("#restore-preview", Button).press()
            await until(pilot, lambda: bool(panel.plan))
            panel.query_one("#restore-confirm", Checkbox).value = True
            await pilot.pause()
            assert panel.query_one("#restore-sensitive", Checkbox).display
            assert panel.query_one("#restore-execute", Button).disabled
            panel.query_one("#restore-sensitive", Checkbox).value = True
            await until(
                pilot, lambda: not panel.query_one("#restore-execute", Button).disabled
            )
            panel.query_one("#restore-choice-keep", Button).press()
            await until(pilot, lambda: not panel.plan)
            assert not panel.query_one("#restore-sensitive", Checkbox).value
            assert not panel.query_one("#restore-confirm", Checkbox).value
            assert not bridge.executions

    asyncio.run(scenario())


def test_slow_restore_status_does_not_block_explicit_cancel():
    """进度读取挂起时取消仍走独立请求，晚到状态不能撤销取消结果；参数：无；返回：无。"""

    async def scenario():
        bridge = RestoreBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 44)) as pilot:
            panel = app.query_one(FileRestorePanel)
            panel.owner = {"session_id": "session", "data_space_id": "space"}
            panel.display = True
            panel.request_operation(
                "execute",
                {**panel.owner, "plan_id": "plan", "confirmation": {"accepted": True}},
            )
            await until(pilot, lambda: panel.job.get("status") == "running")
            bridge.slow_status = True
            panel.poll_operation()
            assert await asyncio.to_thread(bridge.slow_entered.wait, 3)
            assert not panel.query_one("#restore-cancel", Button).disabled
            panel.query_one("#restore-cancel", Button).press()
            await until(pilot, lambda: panel.job.get("status") == "cancelled")
            bridge.slow_release.set()
            await app.workers.wait_for_complete()
            assert panel.job["status"] == "cancelled" and len(bridge.cancellations) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("fail_old", [False, True])
def test_restore_late_directory_reply_cannot_replace_new_session(fail_old):
    """旧会话成功或失败均不能覆盖新的选择和草稿；参数：是否旧请求失败；返回：无。"""

    async def scenario():
        bridge = RestoreBridge()
        bridge.slow_old, bridge.fail_old = True, fail_old
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one(FileRestorePanel)
            panel.open({"session_id": "old", "data_space_id": "space"})
            assert await asyncio.to_thread(bridge.slow_entered.wait, 3)
            panel.open({"session_id": "new", "data_space_id": "space"})
            await until(pilot, lambda: "point:new" in panel.rows)
            app.query_one(Composer).load_text("新会话草稿")
            bridge.slow_release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "point:new" in panel.rows and "point:old" not in panel.rows
            assert "旧会话读取失败" not in str(
                panel.query_one("#restore-notice", Static).render()
            )
            assert app.query_one(Composer).text == "新会话草稿"

    asyncio.run(scenario())
