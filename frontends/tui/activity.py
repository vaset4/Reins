"""活动浮层展示真实工作，模型继续自主组织计划与协作。

作者：xxx
时间：2026-09-30 12:00:00
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from frontends.tui.paged_text import PagedText


def activity_text(data: dict[str, Any]) -> str:
    """将状态事实翻译为可读内容；参数：后台活动快照；返回：完整展示正文。"""
    current, task = data["current"], data["task"]
    lines = [
        f"当前运行：{current['run_id'] or '尚未开始'} · {current['status']}",
        f"执行中：{'是' if current['active'] else '否'} · 已请求停止/暂停：{'是' if current['stopped'] else '否'}",
    ]
    if current["error"]:
        lines.append(f"执行错误：{current['error']}")
    if task is not None:
        lines.extend(
            [f"\n当前工作：{task['goal']} · {task['status']}", "模型维护的计划："]
        )
        lines.extend(
            f"{item['idx'] + 1}. [{item['status']}] {item['content']}"
            for item in task["todos"]
        )
        if not task["todos"]:
            lines.append("尚无已保存计划")
    else:
        lines.append("\n当前没有已保存目标，可直接在聊天里提出要求")
    lines.append("\n本次工作中的子代理：")
    for member in data["members"]:
        status = (
            "状态未知（尚无结束回执，未查询执行存活）"
            if member["status"] == "unknown"
            else member["status"]
        )
        lines.extend(
            [
                f"{member['name']} · {status} · {member['backend']}",
                f"任务：{member['task']}",
                f"会话：{member['session_id']} · 运行：{member['run_id'] or '尚未开始'}",
            ]
        )
        if member["cancel_requested"]:
            lines.append("已请求取消，实际停止以执行回执为准")
        if member["output"]:
            lines.append(f"运行结果：\n{member['output']}")
    if not data["members"]:
        lines.append("暂无已登记子代理")
    lines.extend(
        [
            "\n停止当前运行会同时向所属子执行传递取消；已产生的效果保留",
            "子代理分工、单独接续或纠正，请在补充要求中告诉模型",
            f"\n后台：{data['host_status']} · 未读通知：{data['unread_notifications']}",
        ]
    )
    lines.extend(
        f"后台异常：{name} · {error}" for name, error in data["errors"].items()
    )
    lines.append(
        "通知详情 /notifications；所有目标 /tasks；后台状态 /background；观察入口 /dashboard"
    )
    return "\n".join(lines)


class ActivityScreen(ModalScreen[dict[str, str] | None]):
    """只返回用户明确动作，不在浮层内复制运行状态机。"""

    BINDINGS = [("escape", "dismiss(None)", "关闭")]

    def __init__(self, data: dict[str, Any]) -> None:
        """保留查询身份以防旧选择控制新运行；参数：活动快照；返回：无。"""
        super().__init__()
        self.data = data

    def compose(self) -> ComposeResult:
        """展示计划、协作和实际工作入口；参数：无；返回：活动控件。"""
        current = self.data["current"]
        options = [
            (
                f"{row['title']} · {row['status']} · {'执行中' if row['active'] else '未执行'}",
                row["session_id"],
            )
            for row in self.data["work"]
        ]
        with Vertical(classes="dialog activity-dialog"):
            yield Label("当前工作与后台活动", classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                yield PagedText(activity_text(self.data), title="活动详情")
                yield Select(
                    options,
                    prompt="查看另一项后台工作",
                    id="activity-work",
                    disabled=not options,
                )
                yield Button("打开所选工作", id="activity-open", disabled=not options)
                yield Button("上下文与记忆维护", id="activity-context")
                yield Input(
                    placeholder="补充要求或纠正子代理分工，交给模型处理",
                    id="activity-input",
                )
            with Horizontal(classes="dialog-actions"):
                yield Button("发送补充要求", id="activity-submit")
                yield Button(
                    "继续当前工作", id="activity-continue", disabled=current["active"]
                )
            with Horizontal(classes="dialog-actions"):
                label = (
                    "停止本次及后续唤醒"
                    if current["scheduled"]
                    else "停止当前运行及子执行"
                )
                yield Button(label, id="activity-cancel", disabled=current["stopped"])
            with Horizontal(classes="dialog-actions"):
                yield Button("刷新", id="activity-refresh")
                yield Button("关闭", id="activity-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """提交明确选择及原身份，取消关闭不提交输入；参数：按钮事件；返回：无。"""
        identity = event.button.id
        if not identity or not identity.startswith("activity-"):
            return
        event.stop()
        choice = {
            "session_id": self.data["session_id"],
            "run_id": self.data["current"]["run_id"],
        }
        if identity == "activity-close":
            self.dismiss(None)
        elif identity == "activity-context":
            self.dismiss({**choice, "action": "context_management"})
        elif identity == "activity-open":
            value = self.query_one("#activity-work", Select).value
            if value is not Select.BLANK:
                self.dismiss({**choice, "action": "open", "target": str(value)})
        elif identity in {"activity-continue", "activity-submit"}:
            text = (
                "请继续当前工作，先核对已有结果与停止原因，再决定下一步。"
                if identity == "activity-continue"
                else self.query_one("#activity-input", Input).value
            )
            if text.strip():
                self.dismiss({**choice, "action": "submit", "text": text})
        elif identity in {"activity-refresh", "activity-cancel"}:
            self.dismiss({**choice, "action": identity.removeprefix("activity-")})
