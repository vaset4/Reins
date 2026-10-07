"""验证百万字符已保存工具结果下的原生终端输入与缩放。

作者：xxx
时间：2026-09-29 20:15:00
"""

from __future__ import annotations

import time
from pathlib import Path

from app.background.client import ensure_running
from llm.messages import (
    AssistantMessage,
    StopReason,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
)
from runtime.session_message_store import SessionMessageStore
from tests.scripts.native_tui_terminal import NativeTuiTerminal
from tests.test_background_process import REPOSITORY, process_setup
from tests.test_tui_native_terminal import PASTE_END, PASTE_START

__all__ = ["process_setup"]

LARGE_RESULT_CHARACTERS = 1000000
INPUT_LATENCY_SECONDS = 2.0


def test_native_large_saved_result_keeps_input_responsive(process_setup, tmp_path):
    """真实终端加载长工具记录后仍能及时输入和缩放；参数：隔离进程与证据根；返回：无。"""
    project, model = process_setup
    root = Path.home() / ".reins/data"
    client = ensure_running(project_root=project, data_root=root)
    identity = client.call("attach")["session_id"]
    messages = SessionMessageStore(root)
    # 【TUI】【原生大结果】1. 合成已保存工具记录，实际查询、排版和终端均走正式实现
    messages.append_message(
        identity,
        AssistantMessage(
            "native-large-call",
            (ToolCallPart("native-large", "huge_saved_output", {}),),
            stop_reason=StopReason.TOOL_CALL,
        ),
        run_id="native-large-run",
    )
    output = "原" * LARGE_RESULT_CHARACTERS + "完整末尾"
    messages.append_message(
        identity,
        ToolResultMessage(
            "native-large-result",
            "native-large",
            "huge_saved_output",
            (TextPart(output),),
            "success",
        ),
        run_id="native-large-run",
    )
    with NativeTuiTerminal(REPOSITORY, tmp_path / "native-large.vt") as terminal:
        terminal.wait(
            lambda: (
                "huge_saved_output" in terminal.output
                and "background-test-model" in terminal.output
            )
        )
        position = len(terminal.output)
        started = time.monotonic()
        terminal.send(PASTE_START + "大结果下继续输入" + PASTE_END)
        terminal.wait(lambda: "大结果下继续输入" in terminal.output[position:])
        elapsed = time.monotonic() - started
        assert elapsed < INPUT_LATENCY_SECONDS
        # 【TUI】【原生大结果】2. 缩放后换行仍留在同一草稿，不调用模型
        terminal.resize(70, 22)
        terminal.send("\n" + "缩放后继续编辑")
        terminal.wait(lambda: "缩放后继续编辑" in terminal.output)
        assert not model.packets
        terminal.record(
            f"large_result_chars={len(output)} input_latency_seconds={elapsed:.3f}"
        )
        terminal.disconnect()
    print(
        f"native_large_result: chars={len(output)}, input_latency_seconds={elapsed:.3f}"
    )
