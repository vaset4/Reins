"""共享命令的前置选项与全屏回答接线。

作者：xxx
时间：2026-09-30 00:30:00
"""

from types import SimpleNamespace

from app.repl.console import get_console
from app.repl.slash_commands import SlashCommand, SlashCommandResult
from frontends.tui.bridge import TuiBridge


def test_command_prompt_includes_preceding_options_and_preserves_final_output(
    tmp_path, capsys
):
    """追问先展示命令已打印选项，结果不泄漏到全屏终端外；传参：隔离根；返回：无。"""
    received = []

    def sink(kind, payload):
        """记录界面事件并模拟明确回答；传参：事件和内容；返回：无。"""
        received.append((kind, payload))
        if kind == "prompt":
            assert "1. 第一项" in payload["label"] and "2. 第二项" in payload["label"]
            payload["answer"].set_result("2")

    def choose(args, context):
        """复用共用控制台和追问函数；传参：命令参数和上下文；返回：用户选择。"""
        get_console().print("1. 第一项\n2. 第二项")
        answer = context.prompt_fn("请选择")
        return SlashCommandResult(message=f"已选择：{answer}")

    bridge = TuiBridge(
        project_root=tmp_path, data_root=tmp_path, llm_client=object(), event_sink=sink
    )
    bridge.state.session_id = "session"
    bridge.host = SimpleNamespace(
        config=SimpleNamespace(project_root=tmp_path),
        session_id="session",
        handle_control=lambda text: False,
        prepare_command=lambda text: None,
    )
    bridge.store = object()
    bridge.registry = object()
    bridge.commands.register(SlashCommand("choose", "选择", choose))
    assert not bridge.submit("/choose")
    assert any(
        kind == "output" and payload == "已选择：2" for kind, payload in received
    )
    assert capsys.readouterr().out == ""
