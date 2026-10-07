"""绑定真实请求的键盘和鼠标审批浮层。

作者：xxx
时间：2026-09-30 12:00:00
"""

from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

CHOICE_LABELS = {
    "once": "允许本次",
    "session": "本会话相同范围",
    "task": "本任务相同范围",
    "permanent": "永久允许相同范围",
    "deny": "拒绝",
}


def describe_operation(item: dict[str, Any]) -> str:
    """展示批准的工具、范围及完整参数；参数：单个动作；返回：阅读文本。"""
    resource = item.get("resource")
    target = (
        f"{resource['action']} · {resource['target']}"
        if resource
        else "所展示的完整参数"
    )
    return f"{item['tool']}\n范围：{target}\n参数：\n{json.dumps(item['args'], ensure_ascii=False, indent=2)}"


class ApprovalScreen(ModalScreen[str | None]):
    """逐项选择后一次提交整批，未选项不产生授权。"""

    BINDINGS = [("escape", "dismiss(None)", "稍后处理")]
    DEFAULT_CSS = """
    ApprovalScreen .dialog { height: 85%; }
    ApprovalScreen .dialog-body { height: 1fr; }
    ApprovalScreen #approval-options { height: 10; }
    ApprovalScreen #approval-summary { height: auto; }
    """

    def __init__(self, request: dict[str, Any]) -> None:
        """保存展示身份；参数：真实后台快照；返回：无。"""
        super().__init__()
        self.request = request
        self.items = request["requests"] if "batch_id" in request else [request]
        self.index = 0
        self.choices: list[str] = []

    def compose(self) -> ComposeResult:
        """展示动作原文与可选择范围；参数：无；返回：控件。"""
        with Vertical(classes="dialog"):
            yield Label(
                f"需要授权 · {self.request['identity']}", classes="dialog-title"
            )
            yield Static(id="approval-summary", markup=False)
            with VerticalScroll(classes="dialog-body"):
                yield Static(id="approval-detail", markup=False)
            yield OptionList(id="approval-options")
            yield Input(
                placeholder="补充要求（取消原待审批动作）", id="approval-correction"
            )
            with Horizontal(classes="dialog-actions"):
                yield Button("提交补充要求", id="correction-submit")
                yield Button("稍后处理", id="approval-later")

    def on_mount(self) -> None:
        """首次聚焦选择列表；参数：无；返回：无。"""
        self.show_item()

    def show_item(self) -> None:
        """更新当前项目与可用授权范围；参数：无；返回：无。"""
        options = self.query_one(OptionList)
        options.clear_options()
        if self.index == len(self.items):
            self.query_one("#approval-summary", Static).update(
                "已选完全部操作，请确认提交"
            )
            self.query_one("#approval-detail", Static).update(
                "\n".join(
                    f"{index + 1}. {CHOICE_LABELS[choice]}\n{describe_operation(item)}\n"
                    for index, (item, choice) in enumerate(
                        zip(self.items, self.choices)
                    )
                )
            )
            options.add_options(
                [
                    Option("提交以上选择", id="submit"),
                    Option("重新选择", id="restart"),
                    Option("取消本批操作", id="cancel"),
                ]
            )
        else:
            item = self.items[self.index]
            self.query_one("#approval-detail", Static).update(describe_operation(item))
            self.query_one("#approval-summary", Static).update(
                f"操作 {self.index + 1}/{len(self.items)} · ↑↓ 选择，Enter 确认，也可鼠标点击\n"
                "复用授权仅限此工具与所展示范围，不代表整个工作区或所有工具"
            )
            rows = [Option("允许本次", id="once")]
            if not item.get("force_confirmation"):
                if item.get("session_id"):
                    rows.append(Option("本会话内允许相同工具与范围", id="session"))
                if item.get("task_id"):
                    rows.append(Option("本任务内允许相同工具与范围", id="task"))
                rows.append(
                    Option("永久允许相同工具与范围（保存后持续生效）", id="permanent")
                )
            rows.append(Option("拒绝此操作", id="deny"))
            if len(self.items) > 1 and self.index == 0:
                rows.extend(
                    [
                        Option("本批全部仅允许这一次", id="all-once"),
                        Option("本批全部拒绝", id="all-deny"),
                    ]
                )
            rows.append(
                Option(
                    "取消本批操作" if "batch_id" in self.request else "取消此操作",
                    id="cancel",
                )
            )
            options.add_options(rows)
        options.highlighted = 0
        options.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """收集明确选择并绑定原请求提交；参数：所选项；返回：无。"""
        event.stop()
        choice = event.option.id
        if choice == "cancel":
            self.finish("cancel" if "batch_id" in self.request else "cancelled")
        elif choice == "submit":
            self.finish(
                " ".join(
                    f"{index}={value}" for index, value in enumerate(self.choices, 1)
                )
            )
        elif choice == "restart":
            self.index, self.choices = 0, []
            self.show_item()
        elif choice in {"all-once", "all-deny"}:
            self.choices = [choice.removeprefix("all-")] * len(self.items)
            self.index = len(self.items)
            self.show_item()
        elif choice:
            if "batch_id" not in self.request:
                self.finish(choice)
                return
            self.choices.append(choice)
            self.index += 1
            self.show_item()

    def finish(self, choice: str) -> None:
        """返回原审批编号命令；参数：明确范围或取消；返回：无。"""
        self.dismiss(f"/approve {self.request['identity']} {choice}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """保留稍后处理与纠正入口；参数：点击事件；返回：无。"""
        event.stop()
        if event.button.id == "approval-later":
            self.dismiss(None)
        elif event.button.id == "correction-submit":
            text = self.query_one(Input).value.strip()
            if text:
                self.dismiss(text)
