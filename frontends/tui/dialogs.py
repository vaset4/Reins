"""不覆盖聊天草稿的详情、审批及确认浮层。

作者：xxx
时间：2026-09-29 18:00:00
"""

from __future__ import annotations


from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static
from frontends.tui.approval_dialog import ApprovalScreen as ApprovalScreen
from frontends.tui.paged_text import PagedText


class PromptScreen(ModalScreen[str | None]):
    """命令追问使用独立输入，不占用聊天草稿。"""

    BINDINGS = [("escape", "dismiss(None)", "取消命令")]

    def __init__(self, label: str) -> None:
        """保存问题；传参：共用命令的提示；返回：无。"""
        super().__init__()
        self.label = label

    def compose(self) -> ComposeResult:
        """生成追问输入；传参：无；返回：问题与输入控件。"""
        with Vertical(classes="dialog"):
            with VerticalScroll(classes="dialog-body"):
                yield Static(self.label, markup=False)
            yield Input(id="prompt-answer")
            yield Button("确认", id="prompt-confirm", variant="primary")

    def on_mount(self) -> None:
        """聚焦追问框；传参：无；返回：无。"""
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """返回完整回答；传参：输入事件；返回：无。"""
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """确认当前回答；传参：按钮事件；返回：无。"""
        self.dismiss(self.query_one(Input).value)


class DetailScreen(ModalScreen[str | None]):
    """显示完整文本，可选择从指定树节点继续。"""

    BINDINGS = [("escape", "dismiss(None)", "关闭")]

    def __init__(self, title: str, body: str, *, branch_id: str | None = None) -> None:
        """保存详情与可选分支身份；传参：标题、正文、节点；返回：无。"""
        super().__init__()
        self.heading, self.body, self.branch_id = title, body, branch_id

    def compose(self) -> ComposeResult:
        """生成可滚动完整内容；传参：无；返回：控件。"""
        with Vertical(classes="dialog"):
            yield Label(self.heading, classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                yield PagedText(self.body, title=self.heading)
            with Horizontal(classes="dialog-actions"):
                if self.branch_id:
                    yield Button("从这里继续", id="branch-confirm", variant="primary")
                yield Button("关闭", id="close-dialog")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """返回明确选择；传参：点击；返回：无。"""
        self.dismiss(self.branch_id if event.button.id == "branch-confirm" else None)
