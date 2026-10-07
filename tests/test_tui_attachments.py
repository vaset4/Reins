"""附件真实入站、文件权限及草稿语法验证。

作者：xxx
时间：2026-09-29 22:00:00
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import Input

from app.background.attachments import compose_attachment_input, validate_paths
from app.session_assembly import user_lease
from frontends.tui.attachments import parse_attachment_draft
from frontends.tui.composer import Composer
from frontends.tui.bridge import TuiBridge
from frontends.tui.interactive import InteractiveTui
from llm.messages import model_visible_text
from scripts.testing.llm import from_test_sequence
from tests.test_background_sessions import session_for
from tests.test_session_runtime import capture_requests
from tests.frontends.tui.test_interactive import DisplayBridge
from tools.redacted_files import RedactedFiles
from tools.tool_registry import ToolRegistry


def test_windows_paths_and_plain_at_mentions_are_preserved():
    """独立指令解析不能损坏反斜线和正文；参数：无；返回：无。"""
    draft = parse_attachment_draft(
        '请查看 @someone\n@file "C:\\工作区\\a b.txt"\n@ref code.py'
    )
    assert draft.text == "请查看 @someone"
    assert draft.attachment_paths == ("C:\\工作区\\a b.txt",)
    assert draft.reference_paths == ("code.py",)
    with pytest.raises(ValueError, match="引号"):
        parse_attachment_draft('@file "abc')


def test_bridge_passes_attachment_paths_without_reading_them(tmp_path):
    """界面桥接只转交用户选择，文件读取归后台；参数：无；返回：无。"""
    accepted = []

    def submit(text, **options):
        """记录RPC前的真实调用参数；参数：正文与路径；返回：接纳身份。"""
        accepted.append((text, options))
        return "input-test"

    bridge = TuiBridge(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        llm_client=object(),
        event_sink=lambda *_: None,
    )
    bridge.host = SimpleNamespace(
        handle_control=lambda _: False,
        submit=submit,
        session_id="session",
        config=SimpleNamespace(project_root=tmp_path),
    )
    assert bridge.submit('@file "does not exist.txt"\n@ref code.py') is False
    assert accepted == [
        (
            "",
            {
                "attachment_paths": ("does not exist.txt",),
                "reference_paths": ("code.py",),
            },
        )
    ]


def test_attachment_content_reaches_actual_model_request(tmp_path, monkeypatch):
    """后台实际模型请求包含读取快照且引用不冒充正文；参数：临时根及捕获器；返回：无。"""
    client = from_test_sequence(["完成"])
    requests = capture_requests(client, monkeypatch)
    session = session_for(tmp_path, client, registry=ToolRegistry())
    (tmp_path / "材料.txt").write_text("实际附件内容：订单数量为37", encoding="utf-8")
    (tmp_path / "ref.txt").write_text("未读取的引用秘密标记", encoding="utf-8")
    try:
        session.submit(
            "",
            input_id="input-attachment",
            model_config={},
            attachment_paths=("材料.txt",),
            reference_paths=("ref.txt",),
        )
        assert session.runtime.wait_idle(10)
        assert len(requests) == 1
        body = str(requests[0])
        assert "实际附件内容：订单数量为37" in body
        assert "尚未读取文件内容" in body
        assert "未读取的引用秘密标记" not in body
        inbound = session.messages.read_entries(session.record.session_id)[0]
        assert "实际附件内容：订单数量为37" in model_visible_text(inbound.message)
    finally:
        session.runtime.close()


def test_attachment_permissions_redaction_and_failure(tmp_path):
    """脱敏复用宿主视图，越界及二进制失败不伪成功；参数：临时根；返回：无。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text(
        "API_KEY=never-publish-this-key\n", encoding="utf-8"
    )
    (workspace / "binary.bin").write_bytes(b"abc\x00def")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    options = dict(
        project_root=workspace,
        lease=user_lease(
            task_id="test", project_root=workspace, data_root=tmp_path / "data"
        ),
        redacted_files=RedactedFiles(),
        session_id="session-attachment",
        reference_paths=(),
    )
    body = compose_attachment_input("阅读", attachment_paths=(".env",), **options)
    assert "never-publish-this-key" not in body.text
    assert "<redacted:" in body.text
    with pytest.raises(ValueError, match="REJECTED_PATH"):
        compose_attachment_input("", attachment_paths=("../outside.txt",), **options)
    with pytest.raises(ValueError, match="二进制"):
        compose_attachment_input("", attachment_paths=("binary.bin",), **options)
    with pytest.raises(ValueError):
        validate_paths("not-an-array")


def test_failed_attachment_is_not_accepted(tmp_path):
    """读取失败不创建入站消息或启动模型；参数：临时根；返回：无。"""
    session = session_for(
        tmp_path, from_test_sequence(["不应执行"]), registry=ToolRegistry()
    )
    try:
        with pytest.raises(ValueError, match="附件读取失败"):
            session.submit(
                "分析",
                input_id="input-missing",
                model_config={},
                attachment_paths=("missing.txt",),
            )
        assert not session.runtime.active
        assert not session.messages.read_entries(session.record.session_id)
    finally:
        session.runtime.close()


def test_file_picker_keeps_draft_on_cancel_failure_and_session_switch():
    """真实全屏路径框、切会话与失败回执不丢附件草稿；参数：无；返回：无。"""

    async def scenario():
        """驱动Textual事件循环；参数：无；返回：无。"""
        app = InteractiveTui(DisplayBridge())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.receive(
                "snapshot", {"session_id": "one", "status": "idle", "history": []}
            )
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("阅读这个附件")
            await pilot.press("ctrl+o")
            await pilot.pause()
            app.screen.query_one(Input).value = '"C:\\工作区\\a b.txt"'
            await pilot.press("enter")
            await pilot.pause()
            expected = '阅读这个附件\n@file "C:\\工作区\\a b.txt"'
            assert composer.text == expected
            await pilot.press("ctrl+o")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert composer.text == expected
            app.receive(
                "submitted", {"text": expected, "session_id": "one", "ok": False}
            )
            app.receive(
                "snapshot", {"session_id": "two", "status": "idle", "history": []}
            )
            await pilot.pause()
            app.receive(
                "snapshot", {"session_id": "one", "status": "idle", "history": []}
            )
            await pilot.pause()
            assert composer.text == expected

    asyncio.run(scenario())
