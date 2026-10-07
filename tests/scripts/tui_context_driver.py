"""【TUI】【上下文联验】正式入口连接真实后台，读取和暂停均保留聊天草稿。

作者：xxx
时间：2026-10-01 15:00:00
"""

from __future__ import annotations

import json
from unittest.mock import patch

from textual.app import App
from textual.widgets import Button, OptionList

from frontends.tui.composer import Composer
from frontends.tui.context_management import ContextPanel
from frontends.tui.main import main
from frontends.tui.paged_text import RemotePagedText
from tests.scripts.tui_restore_driver import wait_for

ORIGINAL_RUN = App.run


async def drive(pilot):
    """读取真实工作并分别暂停自动任务；参数：界面驾驶器；返回：无。"""
    app = pilot.app
    await wait_for(pilot, lambda: app.connected and app.projection.session_id)
    composer = app.query_one(Composer)
    composer.load_text("上下文查看期间保留的草稿")
    app.action_context_management()
    panel = app.query_one(ContextPanel)
    await wait_for(pilot, lambda: bool(panel.rows) and bool(panel.enabled))
    panel.query_one(OptionList).focus()
    await pilot.press("enter")
    body = panel.query_one(RemotePagedText)
    await wait_for(pilot, lambda: bool(body.text))
    detail = body.text
    assert "已取消" in detail
    for domain in ("history", "knowledge"):
        panel.query_one(f"#context-{domain}", Button).press()
        await wait_for(pilot, lambda: panel.enabled.get(domain) is False)
    panel.action_close()
    assert composer.text == "上下文查看期间保留的草稿"
    print(
        json.dumps(
            {
                "detail": detail,
                "draft": composer.text,
                "session_id": app.projection.session_id,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    await pilot.press("ctrl+q")


def run_headless(self: App, **kwargs):
    """只替换终端设备，使用正式启动和连接；参数：应用与选项；返回：实际退出码。"""
    return ORIGINAL_RUN(self, **kwargs, headless=True, size=(130, 48), auto_pilot=drive)


if __name__ == "__main__":
    with patch.object(App, "run", run_headless):
        raise SystemExit(main([]))
