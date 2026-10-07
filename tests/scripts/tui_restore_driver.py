"""正式 TUI 与真实后台的文件恢复交互驾驶器。

作者：xxx
时间：2026-09-30 20:40:00
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Button, Checkbox, OptionList

from frontends.tui.composer import Composer
from frontends.tui.file_restore import FileRestorePanel
from frontends.tui.main import main
from frontends.tui.paged_text import RemotePagedText

WAIT_SECONDS = 15


async def wait_for(pilot: Pilot, predicate) -> None:
    """等待真实后台引起的控件状态；参数：驾驶器、结果条件；返回：无，超时直接失败。"""
    async with asyncio.timeout(WAIT_SECONDS):
        while not predicate():
            await pilot.pause(0.03)


async def drive(pilot: Pilot) -> None:
    """选择已保存文件并明确预览确认，保留原草稿；参数：正式界面；返回：无。"""
    app = pilot.app
    await wait_for(pilot, lambda: app.connected and app.projection.session_id)
    composer = app.query_one(Composer)
    composer.load_text("恢复时保留的草稿")
    await pilot.press("f10")
    panel = app.query_one(FileRestorePanel)
    await wait_for(pilot, lambda: bool(panel.rows))
    panel.query_one(OptionList).focus()
    await pilot.press("enter")
    await wait_for(pilot, lambda: bool(panel.source) and not panel._pending)
    assert panel.rows and any(
        row["path"].endswith("background-result.txt") for row in panel.rows.values()
    )
    assert not panel.job
    panel.query_one("#restore-preview", Button).press()
    await wait_for(pilot, lambda: bool(panel.plan))
    assert panel.query_one("#restore-execute", Button).disabled
    panel.query_one(OptionList).focus()
    await pilot.press("enter")
    body = panel.query_one(RemotePagedText)
    await wait_for(pilot, lambda: bool(body.text))
    difference = body.text
    panel.query_one("#restore-confirm", Checkbox).value = True
    await wait_for(
        pilot, lambda: not panel.query_one("#restore-execute", Button).disabled
    )
    panel.query_one("#restore-execute", Button).press()
    await wait_for(
        pilot,
        lambda: (
            panel.job.get("status")
            in {"completed", "partial", "failed", "needs_reconciliation"}
        ),
    )
    assert panel.job["status"] == "completed", panel.job
    assert composer.text == "恢复时保留的草稿"
    print(
        json.dumps(
            {"operation": panel.job, "difference": difference, "draft": composer.text},
            ensure_ascii=False,
        ),
        flush=True,
    )
    await pilot.press("ctrl+q")


def run_headless(self: App, **kwargs):
    """只替换终端设备，保留正式启动和后台连接；参数：应用、选项；返回：实际退出码。"""
    return ORIGINAL_RUN(self, **kwargs, headless=True, size=(130, 48), auto_pilot=drive)


ORIGINAL_RUN = App.run

if __name__ == "__main__":
    with patch.object(App, "run", run_headless):
        raise SystemExit(main([]))
