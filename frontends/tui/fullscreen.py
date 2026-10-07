from __future__ import annotations

from pathlib import Path

from app.repl.console import enable_utf8_console
from llm.base import LLMClient
from prompt_toolkit.application import Application
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from tools.tool_registry import ToolRegistry

from frontends.tui.session import FullscreenTui


def run_fullscreen_tui(
    *,
    project_root: Path,
    data_root: Path,
    llm_client: LLMClient,
    tool_registry: ToolRegistry | None = None,
) -> int:
    enable_utf8_console()
    session = FullscreenTui(
        project_root=project_root,
        data_root=data_root,
        llm_client=llm_client,
        tool_registry=tool_registry,
    )
    application = _build_application(session)
    session.attach_app(application)
    session.start_approval_backend()
    try:
        result = application.run()
        return _result_code(result)
    finally:
        session.stop_approval_backend()
        session.detach_app()
        session.close()


def _build_application(
    session: FullscreenTui,
    *,
    input_obj: Input | None = None,
    output_obj: Output | None = None,
) -> Application[object]:
    body = Window(
        content=FormattedTextControl(session.transcript_fragments),
        wrap_lines=True,
    )
    input_area = TextArea(
        height=1,
        prompt=[("class:prompt", "❯ ")],
        multiline=False,
        accept_handler=lambda buff: _submit_buffer(session, buff),
    )
    root = HSplit(
        [
            body,
            input_area,
            Window(FormattedTextControl(session.footer_fragments), height=2),
        ]
    )
    return Application(
        layout=Layout(root, focused_element=input_area),
        key_bindings=_key_bindings(session),
        style=_style(),
        full_screen=True,
        input=input_obj,
        mouse_support=True,
        output=output_obj,
        refresh_interval=0.25,
    )


def _submit_buffer(session: FullscreenTui, buff: object) -> bool:
    text = getattr(buff, "text", "")
    reset = getattr(buff, "reset", None)
    if callable(reset):
        reset()
    session.submit(str(text))
    return True


def _result_code(result: object) -> int:
    if result is None:
        return 0
    if isinstance(result, int):
        return result
    raise TypeError(f"unexpected TUI result: {result!r}")


def _key_bindings(session: FullscreenTui) -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("c-c")
    def _exit_binding(_event: object) -> None:
        session.exit(0)

    @bindings.add(Keys.PageUp)
    @bindings.add(Keys.ControlPageUp)
    @bindings.add(Keys.ScrollUp)
    def _page_up(_event: object) -> None:
        session.scroll_up()

    @bindings.add(Keys.PageDown)
    @bindings.add(Keys.ControlPageDown)
    @bindings.add(Keys.ScrollDown)
    def _page_down(_event: object) -> None:
        session.scroll_down()

    @bindings.add(Keys.ControlUp)
    def _line_up(_event: object) -> None:
        session.scroll_up(4)

    @bindings.add(Keys.ControlDown)
    def _line_down(_event: object) -> None:
        session.scroll_down(4)

    return bindings


def _style() -> Style:
    return Style.from_dict(
        {
            "footer": "#7f8790",
            "footer.active": "#c9d1d9",
            "footer.pending": "bold #f2cc60",
            "prompt": "bold #f97316",
            "empty": "#7f8790",
            "user.block": "bg:#273241 #f8fafc",
            "assistant.body": "#d6dee7",
            "status.line": "#7f8790",
            "working": "#5eead4",
            "tool.pending.border": "#60a5fa",
            "tool.pending.title": "bold #93c5fd",
            "tool.pending.body": "#cbd5e1",
            "tool.success.border": "#4ade80",
            "tool.success.title": "bold #86efac",
            "tool.success.body": "#d1fae5",
            "tool.error.border": "#f87171",
            "tool.error.title": "bold #fca5a5",
            "tool.error.body": "#fecaca",
            "approval.pending.border": "#f2cc60",
            "approval.pending.title": "bold #fde68a",
            "approval.pending.body": "#fef3c7",
            "error.error.border": "#f87171",
            "error.error.title": "bold #fca5a5",
            "error.error.body": "#fecaca",
        }
    )


__all__ = ["run_fullscreen_tui"]
