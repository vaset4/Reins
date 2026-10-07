"""在独立进程中以真实 Textual 按键驱动正式入口。

作者：xxx
时间：2026-09-30 10:00:00
"""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

from textual.app import App
from textual.pilot import Pilot

from frontends.tui.composer import Composer
from frontends.tui.main import main
from frontends.tui.widgets import MessageCard

WAIT_SECONDS = 15
INPUT_TEXT = "写出本地结果"
FINAL_TEXT = "后台文件已完成"


async def drive(pilot: Pilot) -> None:
    """等待真实连接后发送或读取历史并断开；参数：按键驱动；返回：无。"""
    app = pilot.app
    mode = sys.argv[2]
    # 1. 连接和提交均等待后台回执，不注入快照或伪造已接纳事件
    async with asyncio.timeout(WAIT_SECONDS):
        while not app.connected or not app.projection.session_id:
            await pilot.pause(0.05)
        if mode == "send":
            app.query_one(Composer).focus()
            await pilot.press(*INPUT_TEXT, "enter")
            while INPUT_TEXT not in app.input_history.get(
                app.projection.session_id, []
            ):
                await pilot.pause(0.05)
        elif mode == "reconnect":
            while not any(
                card.card.text == FINAL_TEXT for card in app.query(MessageCard)
            ):
                await pilot.pause(0.05)
    # 2. 从已挂载控件读取实际内容，供父进程核对重复消息及工具
    if mode == "inspect":
        inspect = runpy.run_path(
            str(Path(__file__).with_name("tui_inspection_driver.py"))
        )["inspect"]
        print(
            json.dumps(
                await inspect(pilot, json.loads(sys.argv[3])), ensure_ascii=False
            ),
            flush=True,
        )
        await pilot.press("ctrl+q")
        return
    cards = [card.card for card in app.query(MessageCard)]
    print(
        json.dumps(
            [{"key": card.key, "role": card.role, "text": card.text} for card in cards],
            ensure_ascii=False,
        ),
        flush=True,
    )
    await pilot.press("ctrl+q")


def run_headless(self: App, **kwargs):
    """仅替换终端设备为官方无头驱动；参数：真实应用与运行选项；返回：正式退出码。"""
    return ORIGINAL_RUN(self, **kwargs, headless=True, size=(120, 40), auto_pilot=drive)


ORIGINAL_RUN = App.run

if __name__ == "__main__":
    with patch.object(App, "run", run_headless):
        raise SystemExit(main([]))
