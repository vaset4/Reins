"""用正式TUI控件验证真实后台请求和跨工作区切换。

作者：xxx
时间：2026-09-30 21:00:00
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.widgets import Button, Input, OptionList, Static

from frontends.tui.composer import Composer
from frontends.tui.evidence import EvidencePanel
from frontends.tui.paged_text import RemotePagedText
from frontends.tui.settings import SettingsScreen
from frontends.tui.widgets import MessageCard

WAIT_SECONDS = 10


async def wait_for(pilot, predicate):
    """等待真实控件回执，超时不冒充完成；参数：驾驶器与条件；返回：无。"""
    async with asyncio.timeout(WAIT_SECONDS):
        while not predicate():
            await pilot.pause(0.02)


async def inspect(pilot, config):
    """从正式目录下钻、复制及导出，再切换原工作区；参数：驾驶器和隔离配置；返回：核验结果。"""
    app = pilot.app
    composer = app.query_one(Composer)
    session_a = app.projection.session_id
    assert session_a == config["session_a"]
    leaf_a = app.projection.snapshot["leaf_id"]
    composer.load_text("A工作区未发送草稿")
    model_a = await inspect_settings(pilot)
    await pilot.press("f9")
    panel = app.query_one(EvidencePanel)
    assert panel.display, str(app.query_one("#notice", Static).render())
    await wait_for(
        pilot,
        lambda: (
            bool(panel.rows)
            or "失败" in str(panel.query_one("#evidence-notice", Static).render())
        ),
    )
    assert panel.rows, str(panel.query_one("#evidence-notice", Static).render())
    for level in ("requests", "attempts"):
        previous = dict(panel.rows)
        panel.query_one(OptionList).focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: panel.level == level and panel.rows != previous)
    panel.query_one(OptionList).focus()
    await pilot.press("enter")
    body = panel.query_one(RemotePagedText)
    await wait_for(pilot, lambda: bool(body.text))
    next(button for button in body.query(Button) if button.name == "copy").press()
    await wait_for(pilot, lambda: bool(app.clipboard))
    request = json.loads(app.clipboard)
    # 【TUI】【真实导出】1. 路径由用户输入，完成由后台任务回执确认
    panel.open_export()
    panel.query_one("#export-path", Input).value = config["export_path"]
    panel.query_one("#export-start", Button).press()
    composer.load_text("导出期间仍可编辑")
    await wait_for(pilot, lambda: panel.job.get("status") in {"completed", "failed"})
    assert panel.job["status"] == "completed", panel.job
    assert composer.text == "导出期间仍可编辑"
    source_page = body.page
    panel.query_one("#evidence-close", Button).press()
    await pilot.pause()
    await pilot.press("f9")
    assert body.page == source_page
    relations = await inspect_cards(pilot)
    assert app.projection.snapshot["leaf_id"] == leaf_a
    # 【TUI】【工作区切换】2. 真实握手决定命令根，新建仍属于当前界面工作区
    app.switch_session(config["session_b"])
    await wait_for(pilot, lambda: app.projection.session_id == config["session_b"])
    await wait_for(
        pilot, lambda: app.bridge.project_root == Path(config["workspace_b"])
    )
    assert app.bridge.require_host().config.project_root == Path(config["workspace_b"])
    model_b = await inspect_settings(pilot)
    composer.load_text("B工作区独立草稿")
    app.action_new_session()
    await wait_for(pilot, lambda: app.projection.session_id != config["session_b"])
    assert app.projection.snapshot["project_root"] == config["workspace_b"]
    app.switch_session(session_a)
    await wait_for(pilot, lambda: app.projection.session_id == session_a)
    await wait_for(
        pilot,
        lambda: (
            app.bridge.project_root == Path(app.projection.snapshot["project_root"])
        ),
    )
    assert composer.text == "导出期间仍可编辑"
    assert app.drafts[config["session_b"]] == "B工作区独立草稿"
    assert await inspect_settings(pilot) == model_a
    return {
        "request": request,
        "relations": relations,
        "export": panel.job,
        "draft": composer.text,
        "model_a": model_a,
        "model_b": model_b,
        "workspace_a": str(app.bridge.project_root),
        "workspace_b": config["workspace_b"],
    }


async def inspect_settings(pilot):
    """读取所属工作区模型并关闭浮层，不应用配置；参数：驾驶器；返回：公开输入模型。"""
    app = pilot.app
    composer = app.query_one(Composer)
    draft = composer.text
    composer.focus()
    await pilot.press("f6")
    await wait_for(pilot, lambda: isinstance(app.screen, SettingsScreen))
    assert not app.screen.data["execution_error"], app.screen.data["execution_error"]
    model = app.screen.data["input_model"]
    assert model["model"] and model["provider"]
    await pilot.press("escape")
    assert composer.text == draft
    return model


async def inspect_cards(pilot):
    """从真实持久卡片核对请求关联、原结果和实际回喂；参数：驾驶器；返回：有序关联身份。"""
    app = pilot.app
    panel = app.query_one(EvidencePanel)
    body = panel.query_one(RemotePagedText)
    relations = {}
    for role in ("assistant", "tool"):
        card = next(item for item in app.query(MessageCard) if item.card.role == role)
        generation = panel.generation
        card.query_one(".card-requests", Button).press()
        await wait_for(
            pilot,
            lambda: (
                panel.generation > generation
                and panel.level == "locate"
                and bool(panel.rows)
                and not panel.query_one(OptionList).disabled
                and all("run_id" in row for row in panel.rows.values())
            ),
        )
        relations[role] = [dict(row) for row in panel.rows.values()]
        assert all(row["run_id"] == card.card.run_id for row in panel.rows.values())
        assert len(relations[role]) == (1 if role == "assistant" else 2), (
            role,
            relations[role],
        )
    panel.query_one("#evidence-tools", Button).press()
    await wait_for(
        pilot,
        lambda: (
            panel.level == "tools"
            and bool(panel.rows)
            and not panel.query_one(OptionList).disabled
            and all("operation_id" in row for row in panel.rows.values())
        ),
    )
    panel.query_one(OptionList).focus()
    await pilot.press("enter")
    await wait_for(pilot, lambda: bool(body.text))
    original = json.loads(body.text)
    assert (
        original["status"] == "ok"
        and original["meta"]["execution_state"] == "completed"
    )
    panel.query_one("#evidence-response", Button).press()
    await wait_for(pilot, lambda: panel.section == "tool_feedback" and bool(body.text))
    feedback = json.loads(body.text)
    assert len(feedback) == 1
    assert feedback[0]["content"]["kind"] == "tool_result"
    assert feedback[0]["content"]["status"] == "success"
    assert (
        json.loads(feedback[0]["content"]["content"][0]["text"])["output"]
        == original["output"]
    )
    assert feedback[0]["attempt_id"] in {row["attempt_id"] for row in relations["tool"]}
    return relations
