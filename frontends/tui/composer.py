"""聊天输入的草稿编辑、历史、补全和 Vim 按键。

作者：xxx
时间：2026-09-29 21:30:00
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from textual import events
from textual.message import Message
from textual.widgets import TextArea


class Composer(TextArea):
    """普通输入默认；模式切换只改变编辑行为，不更换草稿。"""

    class Submitted(Message):
        """携带用户明确发送的整份草稿。"""

        def __init__(self, text: str) -> None:
            """保存草稿；传参：原文；返回：无。"""
            super().__init__()
            self.text = text

    class HintsChanged(Message):
        """通知界面更新当前模式及补全候选。"""

    def __init__(self, *, id: str | None = None) -> None:
        """初始化普通编辑与本会话输入历史；传参：控件编号；返回：无。"""
        super().__init__(id=id)
        self.vim = False
        self.vim_state: Literal["insert", "normal", "visual"] = "insert"
        self._pending = ""
        self._history: tuple[str, ...] = ()
        self._history_index = 0
        self._history_draft = ""
        self.commands: tuple[str, ...] = ()
        self._completion_value = ""
        self._completion_index = -1
        self.candidates: tuple[str, ...] = ()
        self._linewise_clipboard: str | None = None
        self.hint_matches: tuple[str, ...] = ()
        self.hint_index = 0
        self._hint_prefix = ""
        self._hint_navigated = False
        self._hint_dismissed: str | None = None

    def command_hints_visible(self) -> bool:
        """同步草稿对应候选并判断菜单可见性；参数：无；返回：是否展示。"""
        if self.text != self._hint_prefix:
            self._hint_prefix = self.text
            self.hint_matches = tuple(
                name for name in self.commands if name.startswith(self.text)
            )
            self.hint_index = 0
            self._hint_navigated = False
            self._hint_dismissed = None
        return (
            self.has_focus
            and self.text.startswith("/")
            and not any(char.isspace() for char in self.text)
            and (not self.vim or self.vim_state == "insert")
            and self._hint_dismissed != self.text
        )

    def accept_command_hint(self, index: int) -> None:
        """将所选命令填入草稿但不执行；参数：候选索引；返回：无。"""
        value = self.hint_matches[index]
        self.replace(value, (0, 0), self.document.end)
        self.move_cursor(self.document.end)
        self.command_hints_visible()
        self._hint_dismissed = value
        self._hint_navigated = False
        self.post_message(self.HintsChanged())

    def _command_hint_key(self, key: str) -> bool:
        """菜单打开时接管选择和收起，完整命令仍可直接提交；参数：按键；返回：是否消费。"""
        if not self.command_hints_visible():
            return False
        if key == "escape":
            if self.vim:
                self.vim_state = "normal"
            else:
                self._hint_dismissed = self.text
        elif key in {"up", "down"} and self.hint_matches:
            self.hint_index = (self.hint_index + (1 if key == "down" else -1)) % len(
                self.hint_matches
            )
            self._hint_navigated = True
        elif (
            key == "enter"
            and self.hint_matches
            and (self._hint_navigated or self.text not in self.commands)
        ):
            self.accept_command_hint(self.hint_index)
        else:
            return False
        self.post_message(self.HintsChanged())
        return True

    @property
    def hints(self) -> str:
        """提供与当前模式一致的提示；传参：无；返回：提示文本。"""
        mode = f"Vim {self.vim_state.upper()}" if self.vim else "普通输入"
        escape = (
            "Esc 导航" if self.vim and self.vim_state == "insert" else "Esc 停止运行"
        )
        if self.vim_state == "visual" and self.vim:
            escape = "Esc 取消选择"
        candidates = " · " + "  ".join(self.candidates) if self.candidates else ""
        return f"{mode} · F2 切换 · Enter 发送 / Ctrl+J 换行 · Ctrl+↑↓ 历史 · Tab 补全 · {escape}{candidates}"

    def toggle_vim(self) -> None:
        """保留文本和光标切换输入模式；传参：无；返回：无。"""
        self.vim = not self.vim
        self.vim_state = "normal" if self.vim else "insert"
        self._pending = ""
        self.post_message(self.HintsChanged())

    def on_focus(self) -> None:
        """回到草稿时恢复输入提示；参数：无；返回：无。"""
        self.post_message(self.HintsChanged())

    def on_blur(self) -> None:
        """离开草稿时通知界面隐藏命令提示；参数：无；返回：无。"""
        self.post_message(self.HintsChanged())

    def set_history(self, values: Sequence[str]) -> None:
        """绑定当前会话已接纳输入，未发送草稿不进历史；传参：有序原文；返回：无。"""
        self._history = tuple(values)
        self._history_index = len(self._history)
        self._history_draft = self.text

    def browse_history(self, direction: int) -> None:
        """浏览历史并在返回末尾时恢复原草稿；传参：前后方向；返回：无。"""
        if self._history_index == len(self._history):
            self._history_draft = self.text
        target = max(0, min(len(self._history), self._history_index + direction))
        if target == self._history_index:
            return
        self._history_index = target
        text = (
            self._history[target]
            if target < len(self._history)
            else self._history_draft
        )
        self.load_text(text)
        self.move_cursor(self.document.end)

    def complete_command(self, direction: int) -> bool:
        """只补全首个命令词，不把参数或普通文本改写；传参：循环方向；返回：是否消费 Tab。"""
        if not self.text.startswith("/") or any(char.isspace() for char in self.text):
            return False
        if self.text != self._completion_value:
            self.candidates = tuple(
                name for name in self.commands if name.startswith(self.text)
            )
            self._completion_index = -1 if direction > 0 else 0
        if self.candidates:
            self._completion_index = (self._completion_index + direction) % len(
                self.candidates
            )
            value = self.candidates[self._completion_index]
            self.replace(value, (0, 0), self.document.end)
            self.move_cursor(self.document.end)
            self._completion_value = value
        self.post_message(self.HintsChanged())
        return True

    async def _on_key(self, event: events.Key) -> None:
        """区分发送、编辑和终端控制，粘贴不经过发送路径；传参：按键；返回：无。"""
        key = event.key
        handled = True
        if self._command_hint_key(key):
            pass
        elif key == "enter":
            if self.text.strip():
                self.post_message(self.Submitted(self.text))
        elif key == "ctrl+j":
            self.insert("\n")
        elif key in {"ctrl+up", "ctrl+down"}:
            self.browse_history(-1 if key == "ctrl+up" else 1)
        elif key in {"tab", "shift+tab"}:
            handled = self.complete_command(-1 if key == "shift+tab" else 1)
        elif self.vim:
            handled = self._vim_key(event)
        else:
            handled = False
        if handled:
            event.stop()
            event.prevent_default()
            return
        if key not in {"tab", "shift+tab"} and self.candidates:
            self.candidates = ()
            self._completion_value = ""
            self.post_message(self.HintsChanged())
        await super()._on_key(event)

    async def _on_paste(self, event: events.Paste) -> None:
        """中文及多行粘贴留在同一草稿，Vim切入编辑；传参：粘贴正文；返回：无。"""
        self.vim_state = "insert"
        self._pending = ""
        normalized = events.Paste(event.text.replace("\r\n", "\n").replace("\r", "\n"))
        await super()._on_paste(normalized)
        event.stop()
        event.prevent_default()
        self.post_message(self.HintsChanged())

    def _vim_key(self, event: events.Key) -> bool:
        """处理Vim模式和导航，未处理的控制键交给终端；传参：按键；返回：是否消费。"""
        key = (event.character or event.key) if event.is_printable else event.key
        if event.key == "escape":
            if self.vim_state == "normal" and not self._pending:
                return False
            self.move_cursor(self.cursor_location)
            self.vim_state, self._pending = "normal", ""
            self.post_message(self.HintsChanged())
            return True
        if self.vim_state == "insert":
            return False
        motions = {
            "h": self.action_cursor_left,
            "j": self.action_cursor_down,
            "k": self.action_cursor_up,
            "l": self.action_cursor_right,
            "w": self.action_cursor_word_right,
            "b": self.action_cursor_word_left,
            "0": self.action_cursor_line_start,
            "$": self.action_cursor_line_end,
        }
        if not self._pending and key in motions:
            motions[key](self.vim_state == "visual")
            return True
        if key == "G":
            self.move_cursor(self.document.end, select=self.vim_state == "visual")
        elif self._pending or key in {"g", "d", "y"}:
            self._vim_sequence(key)
        elif not self._vim_edit(key):
            return event.is_printable
        self.post_message(self.HintsChanged())
        return True

    def _vim_sequence(self, key: str) -> None:
        """处理整行复制、删除和跳到开头；传参：当前键；返回：无。"""
        sequence = self._pending + key
        self._pending = ""
        if self.vim_state == "visual" and key in {"d", "y"}:
            self.action_cut() if key == "d" else self.action_copy()
            self.move_cursor(self.cursor_location)
            self.vim_state = "normal"
        elif sequence == "gg":
            self.move_cursor((0, 0), select=self.vim_state == "visual")
        elif sequence in {"dd", "yy"}:
            row, _column = self.cursor_location
            self._linewise_clipboard = self.document.get_line(row) + "\n"
            self.app.copy_to_clipboard(self._linewise_clipboard)
            if sequence == "dd":
                if row > 0 and row == self.document.line_count - 1:
                    start = (row - 1, len(self.document.get_line(row - 1)))
                    self.delete(start, self.document.end)
                    self.move_cursor(start)
                else:
                    self.action_delete_line()
        elif len(sequence) == 1:
            self._pending = key

    def _vim_edit(self, key: str) -> bool:
        """执行常用插入、选择和修改动作；传参：普通模式键；返回：是否识别。"""
        edits = {
            "x": self.action_delete_right,
            "D": self.action_delete_to_end_of_line,
            "u": self.action_undo,
            "ctrl+r": self.action_redo,
        }
        if key in edits:
            edits[key]()
        elif key == "p":
            if (
                self._linewise_clipboard is not None
                and self.app.clipboard == self._linewise_clipboard
            ):
                self.action_cursor_line_end()
                self.insert("\n" + self._linewise_clipboard.removesuffix("\n"))
            else:
                self.action_paste()
        elif key == "v":
            self.vim_state = "normal" if self.vim_state == "visual" else "visual"
            self.move_cursor(self.cursor_location)
        elif key in {"i", "a", "I", "A", "o", "O"}:
            self._vim_insert(key)
        else:
            return False
        return True

    def _vim_insert(self, key: str) -> None:
        """选择插入位置，换行只修改草稿；传参：插入动作；返回：无。"""
        row, column = self.cursor_location
        if key == "a":
            self.move_cursor((row, min(column + 1, len(self.document.get_line(row)))))
        elif key in {"A", "o"}:
            self.action_cursor_line_end()
        elif key in {"I", "O"}:
            self.move_cursor((row, 0))
        if key in {"o", "O"}:
            self.insert("\n")
            if key == "O":
                self.move_cursor((row, 0))
        self.vim_state = "insert"
