"""正式全屏输入的粘贴与本终端停止边界。

作者：xxx
时间：2026-10-06 13:10:00
"""

import asyncio

from textual import events

from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from tests.frontends.tui.test_interactive import DisplayBridge


class CancellationBridge(DisplayBridge):
    """记录正式界面发到当前连接的停止请求。"""

    def __init__(self):
        """初始化停止记录；参数：无；返回：无。"""
        super().__init__()
        self.cancellations = []

    def cancel(self):
        """记录宿主收到的停止；参数：无；返回：无。"""
        self.cancellations.append("current-session")


def test_paste_escape_is_text_and_local_escape_preserves_draft():
    """粘贴中的 Esc 不停止，独立 Esc 只停止当前连接且保留草稿；参数：无；返回：无。"""

    async def scenario():
        """使用正式控件和消息循环验证停止边界；参数：无；返回：无。"""
        bridge = CancellationBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            app.receive(
                "snapshot",
                {"session_id": "current-session", "status": "running", "history": []},
            )
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.focus()
            # 1. 输入内容保持单份草稿，正文中的控制字符不是运行控制指令
            draft = "第一行\n\x1b第二行"
            composer.post_message(events.Paste(draft))
            await pilot.pause()
            assert composer.text == draft
            assert bridge.cancellations == [] and bridge.submitted == []
            # 2. 本终端独立停止按键到达当前宿主，未发送内容不被消费
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            assert bridge.cancellations == ["current-session"]
            assert composer.text == draft and bridge.submitted == []

    asyncio.run(scenario())
