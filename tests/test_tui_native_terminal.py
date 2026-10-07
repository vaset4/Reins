"""原生 Windows 终端中的输入、缩放、取消和后台独立运行验证。

作者：xxx
时间：2026-09-29 19:00:00
"""

from __future__ import annotations

from pathlib import Path

from app.background.client import connect, is_running
from llm.messages import TextPart, UserMessage
from runtime.session_message_store import SessionMessageStore
from tests.scripts.native_tui_terminal import NativeTuiTerminal
from tests.test_background_process import REPOSITORY, process_setup, wait_until

__all__ = ["process_setup"]

PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"
ENTER = "\r"
CTRL_J = "\n"
ESCAPE = "\x1b"
F1 = "\x1bOP"


def test_native_paste_resize_send_disconnect(process_setup, tmp_path):
    """中文多行保持单草稿，缩放后发送且断开继续执行；参数：隔离进程、证据目录；返回：无。"""
    project, model = process_setup
    root = Path.home() / ".reins/data"
    expected = "中文第一行\n第二行\n缩放后仍可输入"
    with NativeTuiTerminal(REPOSITORY, tmp_path / "native-input.vt") as terminal:
        # 1. 正式应用通过真实控制台进入全屏与括号粘贴模式
        terminal.wait(lambda: "普通输入" in terminal.output and is_running(root))
        terminal.wait(lambda: "background-test-model" in terminal.output)
        assert "\x1b[?1049h" in terminal.output
        assert "\x1b[?2004h" in terminal.output
        client = connect(root)
        session = client.call("attach")["session_id"]
        terminal.send(PASTE_START + "中文第一行\n第二行" + PASTE_END)
        terminal.wait(lambda: "第二行" in terminal.output)
        assert not model.packets
        # 2. 真正调整终端设备尺寸后继续编辑，Ctrl+J 仍是草稿内换行
        terminal.resize(70, 22)
        terminal.send(CTRL_J + "缩放后仍可输入")
        terminal.wait(lambda: "缩放后仍可输入" in terminal.output)
        assert not model.packets
        terminal.send(ENTER)
        terminal.wait(lambda: bool(model.packets))
        entries = SessionMessageStore(root).read_entries(session)
        inbound = [row for row in entries if row.type == "inbound"]
        assert len(inbound) == 1
        assert isinstance(inbound[0].message, UserMessage)
        assert inbound[0].message.content == (TextPart(expected),)
        terminal.disconnect()
    # 3. 前台已退出时才释放模型，文件工具仍在真实后台执行
    assert is_running(root) and not (project / "background-result.txt").exists()
    model.first_release.set()
    model.final_release.set()
    wait_until(lambda: client.call("poll", session_id=session)["status"] == "done")
    assert (project / "background-result.txt").read_text(encoding="utf-8") == "只写一次"
    assert len(model.packets) == 2


def test_native_escape_cancels_run_without_closing_terminal(process_setup, tmp_path):
    """原生 Esc 停止当前运行，随后仍可编辑并用 Ctrl+Q 断开；参数：隔离进程、证据目录；返回：无。"""
    project, model = process_setup
    root = Path.home() / ".reins/data"
    with NativeTuiTerminal(REPOSITORY, tmp_path / "native-escape.vt") as terminal:
        terminal.wait(
            lambda: "background-test-model" in terminal.output and is_running(root)
        )
        client = connect(root)
        session = client.call("attach")["session_id"]
        terminal.send("等待取消" + ENTER)
        terminal.wait(lambda: bool(model.packets))
        # 浮层内首次 Esc 只关闭帮助，不能停止后台运行
        terminal.send(F1)
        terminal.wait(lambda: "浮层内 Esc" in terminal.output)
        position = len(terminal.output)
        terminal.send(ESCAPE)
        terminal.wait(lambda: "等待取消" in terminal.output[position:])
        assert not client.call("poll", session_id=session)["stopped"]
        terminal.send(ESCAPE)
        terminal.wait(lambda: client.call("poll", session_id=session)["stopped"])
        terminal.send(PASTE_START + "取消后保留草稿" + PASTE_END)
        terminal.wait(lambda: "取消后保留草稿" in terminal.output)
        terminal.disconnect()
    model.first_release.set()
    model.final_release.set()
    state = wait_until(lambda: not client.call("poll", session_id=session)["active"])
    assert state and not (project / "background-result.txt").exists()
