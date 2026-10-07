"""以执行端事实展示设置，所有变更交还共享命令。

作者：xxx
时间：2026-09-30 10:00:00
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select, Static

MODE_OPTIONS = [
    ("只读", "read_only"),
    ("工作区内写自动放行", "workspace"),
    ("自动放行", "auto"),
]
REASONING_LABELS = {
    "default": "模型默认",
    "none": "关闭",
    "minimal": "最低",
    "low": "低",
    "medium": "中",
    "high": "高",
    "xhigh": "极高",
    "max": "最高",
}


def model_description(model: dict[str, Any] | None) -> str:
    """呈现公开模型身份，缺失明确未知；参数：后台或输入模型；返回：说明。"""
    if not model:
        return "未知"
    effort = str(model.get("reasoning_effort") or "default")
    return (
        f"{model.get('provider', '未知')} / {model.get('model', '未知')} · 配置 {model.get('profile_name', '未命名')}"
        f" · 思考 {REASONING_LABELS.get(effort, effort)}"
    )


def connection_description(data: dict[str, Any]) -> str:
    """仅翻译执行端连接事实，不把配置当在线；参数：设置快照；返回：连接状态。"""
    mcp = data.get("mcp")
    if mcp is None:
        return "MCP：未知（当前没有可查询的执行端连接）"
    rows = [
        f"{row['server']}：{row.get('status', '未知')} · {row.get('transport', '未知')}"
        + (f" · {row['error']}" if row.get("error") else "")
        for row in mcp.get("servers", [])
    ]
    rows.extend(
        f"{name}：失败 · {reason}" for name, reason in mcp.get("errors", {}).items()
    )
    return "MCP 执行端连接\n" + ("\n".join(rows) or "尚未建立服务器连接")


class SettingsScreen(ModalScreen[str | None]):
    """设置浮层只返回用户明确点击的命令，不占用草稿。"""

    BINDINGS = [("escape", "dismiss(None)", "关闭")]

    def __init__(self, data: dict[str, Any]) -> None:
        """保存本次查询快照；参数：真实设置；返回：无。"""
        super().__init__()
        self.data = data
        self.profiles = {row["name"]: row for row in data["profiles"]}
        current = (data.get("input_model") or {}).get("profile_name") or data.get(
            "saved_profile"
        )
        self.selected_profile = (
            current if current in self.profiles else next(iter(self.profiles), "")
        )

    def model_controls(self) -> ComposeResult:
        """呈现已配置供应商、模型和实际支持强度；参数：无；返回：选择控件。"""
        selected = self.profiles.get(self.selected_profile, {})
        providers = sorted(
            {
                row.get("provider_group", row["provider"])
                for row in self.profiles.values()
            }
        )
        provider = selected.get("provider_group", selected.get("provider"))
        yield Label("供应商")
        yield Select(
            [(name, name) for name in providers],
            id="provider-choice",
            allow_blank=not providers,
            value=provider if provider else Select.NULL,
            disabled=not providers,
        )
        yield Label("模型")
        options = self.model_options(provider)
        yield Select(
            options,
            prompt="尚未配置模型",
            id="model-choice",
            allow_blank=not options,
            value=self.selected_profile or Select.NULL,
            disabled=not options,
        )
        yield Label("思考强度")
        yield Select(
            self.reasoning_options(selected),
            id="reasoning-choice",
            allow_blank=False,
            value=selected.get("reasoning_effort") or "default",
            disabled=not selected.get("reasoning_options"),
        )
        yield Static(
            self.reasoning_description(selected),
            id="reasoning-description",
            markup=False,
        )
        yield Button("应用模型与思考强度", id="apply-model", disabled=not options)
        if not self.profiles:
            yield Static(
                "尚无模型配置。请在 ~/.reins/models.json 配置供应商、模型和凭据引用，"
                "在 ~/.reins/.env 配置对应密钥。",
                markup=False,
            )

    def model_options(self, provider: str | None) -> list[tuple[str, str]]:
        """按供应商筛选真实配置；参数：供应商分组；返回：模型标签与配置身份。"""
        return [
            (f"{row['model']} · {name}", name)
            for name, row in self.profiles.items()
            if row.get("provider_group", row["provider"]) == provider
        ]

    def reasoning_options(self, profile: dict[str, Any]) -> list[tuple[str, str]]:
        """使用后端声明的能力生成选项；参数：模型配置；返回：强度标签和值。"""
        return [
            (REASONING_LABELS.get(value, value), value)
            for value in ["default", *profile.get("reasoning_options", [])]
        ]

    def reasoning_description(self, profile: dict[str, Any]) -> str:
        """说明能力未知时不能承诺强度生效；参数：模型配置；返回：用户提示。"""
        return (
            "上下键选择，Enter 确认；Tab 切换控件"
            if profile.get("reasoning_options")
            else "此模型未声明可调思考强度，使用模型默认设置"
        )

    def compose(self) -> ComposeResult:
        """展示实际模型、可选配置及权限；参数：无；返回：设置控件。"""
        data = self.data
        with Vertical(classes="dialog"):
            yield Label("模型、连接与权限", classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                yield Static(
                    "后续新运行：" + model_description(data.get("input_model")),
                    markup=False,
                )
                yield Static(
                    "当前执行模型：" + model_description(data.get("model")),
                    markup=False,
                )
                yield Static(
                    "模型切换不改变已接纳运行；定时任务沿用自己的配置", markup=False
                )
                yield from self.model_controls()
                yield Static(
                    "当前权限："
                    + dict((value, label) for label, value in MODE_OPTIONS).get(
                        str(data.get("approval_mode")), "未知（定时任务沿用原授权）"
                    ),
                    markup=False,
                )
                yield Select(
                    MODE_OPTIONS,
                    prompt="选择权限模式",
                    id="mode-choice",
                    disabled=data.get("scheduled", False),
                )
                yield Static(
                    "切换权限会中断当前待审批动作；不会扩大租约、跳过密钥替换或定时控制确认",
                    markup=False,
                )
                yield Button(
                    "应用权限", id="apply-mode", disabled=data.get("scheduled", False)
                )
                yield Static(connection_description(data), markup=False)
                yield Button(
                    "上下文与记忆：查看、暂停或取消", id="open-context-management"
                )
                yield Static(
                    "模型目录：~/.reins/models.json；密钥：~/.reins/.env；授权查询 /mode，撤回 /revoke；F1 帮助",
                    markup=False,
                )
            with Horizontal(classes="dialog-actions"):
                yield Button("刷新状态", id="refresh-settings")
                yield Button("关闭", id="close-settings")

    def on_select_changed(self, event: Select.Changed) -> None:
        """供应商变化重建模型及强度选项；参数：真实选择事件；返回：无。"""
        if event.value is Select.NULL:
            return
        if event.select.id == "provider-choice":
            model = self.query_one("#model-choice", Select)
            options = self.model_options(str(event.value))
            if model.value in {value for _, value in options}:
                return
            model.set_options(options)
            model.value = options[0][1] if options else Select.NULL
        elif event.select.id == "model-choice":
            profile = self.profiles[str(event.value)]
            choice = self.query_one("#reasoning-choice", Select)
            choice.set_options(self.reasoning_options(profile))
            choice.value = profile.get("reasoning_effort") or "default"
            choice.disabled = not profile.get("reasoning_options")
            self.query_one("#reasoning-description", Static).update(
                self.reasoning_description(profile)
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """只提交明确选择，空选项不生效；参数：按钮事件；返回：无。"""
        event.stop()
        identity = event.button.id
        if identity in {"close-settings", "refresh-settings"}:
            self.dismiss("refresh" if identity == "refresh-settings" else None)
        elif identity == "open-context-management":
            self.dismiss("context_management")
        elif identity in {"apply-model", "apply-mode"}:
            value = self.query_one(
                "#model-choice" if identity == "apply-model" else "#mode-choice", Select
            ).value
            if value is not Select.NULL:
                if identity == "apply-model":
                    effort = self.query_one("#reasoning-choice", Select).value
                    self.dismiss(
                        f"/model profile use {value} reasoning_effort={effort}"
                    )
                else:
                    self.dismiss(f"/mode {value}")
