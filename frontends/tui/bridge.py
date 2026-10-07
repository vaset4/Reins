"""全屏界面与共用后台、命令接口之间的窄连接。

作者：xxx
时间：2026-09-29 18:00:00
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from io import StringIO
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from app.background.frontend import BackgroundSessionHost
from app.repl import _describe_model, _render_dashboard_view
from app.repl.console import capture_console
from app.repl.session_host import local_session_resources
from app.repl.slash_commands import (
    ReplState,
    SlashCommandContext,
    create_default_registry,
)
from llm.base import LLMClient
from llm.profiles import load_model_profiles
from llm.public_config import public_model_config
from llm.reasoning import reasoning_options
from llm.resolved_target import requires_api_key
from tools.tool_registry import ToolRegistry
from frontends.tui.attachments import parse_attachment_draft


class TuiBridge:
    """复用真实后台宿主；只转发输入、选择及已提交事实。"""

    def __init__(
        self,
        *,
        project_root: Path,
        data_root: Path,
        llm_client: LLMClient | None,
        event_sink: Callable[[str, Any], None],
        registry: ToolRegistry | None = None,
        startup_error: str = "",
    ) -> None:
        """保存连接依赖；传参：目录、模型和界面消息入口；返回：无。"""
        self.project_root, self.data_root = project_root, data_root
        self.llm_client, self.event_sink = llm_client, event_sink
        self.state = ReplState(defer_session=True)
        self.commands = create_default_registry()
        self._resources = ExitStack()
        self._registry = registry
        self.host: BackgroundSessionHost | None = None
        self._prompt_lock = Lock()
        self._pending_prompt: Future[str] | None = None
        self._closed = False
        self._command_output: StringIO | None = None
        self._context_lock = RLock()
        self.execution_error = startup_error

    def start(self) -> None:
        """在线程内连接后台并取得命令依赖；传参：无；返回：无。"""
        try:
            self.store, self.registry, host = self._resources.enter_context(
                local_session_resources(
                    self.state,
                    project_root=self.project_root,
                    data_root=self.data_root,
                    llm_client=self.llm_client,
                    tool_registry=self._registry,
                    host_factory=partial(
                        BackgroundSessionHost, event_sink=self.event_sink
                    ),
                )
            )
            assert isinstance(host, BackgroundSessionHost)
            self.host = host
            self.sync_workspace()
        except BaseException:
            self._resources.close()
            raise

    def close(self) -> None:
        """断开显示资源，后台已接纳工作继续；传参：无；返回：无。"""
        with self._prompt_lock:
            self._closed = True
            if self._pending_prompt is not None and not self._pending_prompt.done():
                self._pending_prompt.set_exception(
                    EOFError("界面已关闭，命令输入已取消")
                )
        self._resources.close()

    def submit(self, text: str) -> bool:
        """冻结当前桥接上下文后提交，避免切换中借用前一客户端；参数：完整草稿；返回：退出标志。"""
        with self._context_lock:
            self.sync_workspace()
            return self._submit(text)

    def _submit(self, text: str) -> bool:
        """路由普通输入与现有命令，返回是否退出界面；传参：完整草稿；返回：退出标志。"""
        host = self.require_host()
        if host.handle_control(text):
            return False
        if self.execution_error and text.split(maxsplit=1)[0] != "/model":
            raise RuntimeError(f"当前会话仅可阅读：{self.execution_error}")
        if not text.startswith("/"):
            draft = parse_attachment_draft(text)
            if draft.attachment_paths or draft.reference_paths:
                identity = host.submit(
                    draft.text,
                    attachment_paths=draft.attachment_paths,
                    reference_paths=draft.reference_paths,
                )
            else:
                identity = host.submit(text)
            self.event_sink(
                "accepted",
                {"identity": identity, "text": text, "session_id": host.session_id},
            )
            return False
        host.prepare_command(text)
        candidate = self.prepare_profile_client(text)
        context = SlashCommandContext(
            self.state,
            self.store,
            self.registry,
            self.llm_client,
            self.project_root,
            self.data_root,
            self._prompt,
        )
        # 【Reins】【共用命令】捕获现有 Rich 回执，全屏期间不直接写终端
        captured = StringIO()
        self._command_output = captured
        try:
            with capture_console(file=captured):
                result = self.commands.dispatch(text, context)
                if result is not None and result.enter_dashboard:
                    _render_dashboard_view(
                        self.state,
                        self.store,
                        self.registry,
                        self.project_root,
                        self.data_root,
                    )
        finally:
            self._command_output = None
        output = captured.getvalue().strip()
        if output:
            self.event_sink("output", output)
        if result is None:
            return False
        if result.model_config_pending_restart:
            if candidate is not None:
                host.set_input_model(candidate)
                self.llm_client = candidate
                if self.project_root.is_dir():
                    self.execution_error = ""
                self.event_sink(
                    "output", "模型与思考强度已应用到后续新运行；已接纳运行保持原配置"
                )
                self.event_sink("input-model", self.model_label())
            else:
                self.reload_input_model()
        elif result.message:
            self.event_sink("output", result.message)
        if result.session_input is not None:
            identity = host.submit(result.session_input)
            self.event_sink(
                "accepted",
                {
                    "identity": identity,
                    "text": result.session_input,
                    "session_id": host.session_id,
                },
            )
        if self.state.session_id != host.session_id:
            host.attach_session(self.state.session_id)
            self.sync_workspace()
        if result.clear_screen:
            self.event_sink("clear", None)
        return result.should_exit

    def prepare_profile_client(self, text: str) -> LLMClient | None:
        """持久化选择前校验模型与凭据；参数：共用命令；返回：待应用客户端，错误时不改配置。"""
        from app.cli import build_llm_client

        fields = text.split()
        if fields[:3] != ["/model", "profile", "use"] or len(fields) < 4:
            return None
        options: dict[str, object] = {"profile_name": fields[3]}
        for field in fields[4:]:
            key, separator, value = field.partition("=")
            if key != "reasoning_effort" or not separator:
                raise ValueError("选择模型只接受 reasoning_effort 参数")
            options[key] = value
        client = build_llm_client(options, project_root=self.project_root)
        target = getattr(client, "resolved_target", None)
        if target is None:
            raise ValueError("模型配置不完整，尚不能用于新输入")
        if requires_api_key(target.base_url) and not target.api_key_present:
            raise ValueError("该供应商尚未配置可用凭据，模型选择未保存")
        return client

    def require_host(self) -> BackgroundSessionHost:
        """在真实连接建立后提供宿主；传参：无；返回：宿主，未连接时明确失败。"""
        if self.host is None:
            raise RuntimeError("后台尚未连接")
        return self.host

    def attach_session(self, identity: str | None) -> None:
        """确认会话归属后重建命令与输入依赖；参数：已有会话或新建；返回：无。"""
        with self._context_lock:
            self.require_host().attach_session(identity)
            self.sync_workspace()

    def sync_workspace(self) -> None:
        """以后台确认的原目录刷新依赖，失败仍允许历史阅读；参数：无；返回：无。"""
        from app.cli import build_llm_client
        from tools.builtin_tools import build_tool_registry

        host = self.require_host()
        root = host.config.project_root
        if (
            root == self.project_root
            and not self.execution_error
            and root.is_dir()
            and self.llm_client is not None
        ):
            return
        # 【TUI】【工作区切换】1. 原目录不可用时不借用旧客户端或创建替代目录
        self.project_root = root
        self.execution_error = "正在加载所属工作区配置"
        try:
            if not root.is_dir():
                raise ValueError(f"原工作区不可用：{root}")
            client = build_llm_client({}, project_root=root)
            target = getattr(client, "resolved_target", None)
            if target is None or (
                requires_api_key(target.base_url) and not target.api_key_present
            ):
                raise ValueError("原工作区模型配置或凭据不可用")
            registry = build_tool_registry(repo_root=root, data_root=self.data_root)
        except Exception as exc:
            self.execution_error = str(exc)
            self.event_sink("output", f"会话历史可读，继续执行不可用：{exc}")
            self.event_sink("input-model", "不可用，请检查所属工作区配置")
            return
        previous = self.registry
        self.registry, self.llm_client = registry, client
        host.config = replace(host.config, registry=registry)
        host.set_input_model(client)
        self.execution_error = ""
        previous.close()
        self.event_sink("input-model", self.model_label())

    def query_evidence(self, **params: Any) -> dict[str, Any]:
        """转发按页查询，独立连接不占聊天控制锁；参数：只读查询；返回：真实记录。"""
        return self.require_host().browse("query_evidence", payload=params)

    def query_context_management(self, **params: Any) -> dict[str, Any]:
        """通过独立连接查看或控制自动整理；参数：固定空间/会话及动作；返回：后台真实状态。"""
        return self.require_host().browse("context_management", payload=params)

    def export_evidence(self, **params: Any) -> dict[str, Any]:
        """启动、查询或取消独立导出；参数：任务动作；返回：实际发布状态。"""
        return self.require_host().browse("export_evidence", payload=params)

    def query_file_restore(self, **params: Any) -> dict[str, Any]:
        """读取恢复点、差异或进度，不依赖模型配置；参数：持久来源；返回：后台查询结果。"""
        return self.require_host().browse("file_restore_query", payload=params)

    def execute_file_restore(self, **params: Any) -> dict[str, Any]:
        """提交本次预览确认，文件写入由后台持有；参数：计划与确认；返回：恢复操作。"""
        return self.require_host().browse("file_restore_execute", payload=params)

    def cancel_file_restore(self, **params: Any) -> dict[str, Any]:
        """停止恢复作业尚未开始的文件；参数：操作身份；返回：逐项状态。"""
        return self.require_host().browse("file_restore_cancel", payload=params)

    def settings(self) -> dict[str, Any]:
        """固定当前工作区查询设置，切换不能混入前一配置；参数：无；返回：真实设置。"""
        with self._context_lock:
            return self._settings()

    def _settings(self) -> dict[str, Any]:
        """合并后台事实与可选模型目录；参数：无；返回：不含凭据的设置。"""
        host = self.require_host()
        self.sync_workspace()
        result = host.browse("settings", session_id=host.session_id)
        profiles = load_model_profiles()
        return {
            **result,
            "input_model": {}
            if self.execution_error or self.llm_client is None
            else public_model_config(self.llm_client),
            "execution_error": self.execution_error,
            "saved_profile": profiles.active,
            "profiles": [
                {
                    "name": name,
                    "model": profile.model,
                    "provider": profile.provider,
                    "provider_group": name.partition(":")[0]
                    if profiles.source == "models_json"
                    else profile.provider,
                    "reasoning_effort": profile.reasoning_effort,
                    "reasoning_options": reasoning_options(
                        profile.model, profile.api_mode
                    ),
                }
                for name, profile in sorted(profiles.profiles.items())
            ],
        }

    def reload_input_model(self, profile_name: str | None = None) -> None:
        """保存后重新校验配置，再发布后续输入模型；参数：明确选择的配置名；返回：无，失败保留旧客户端。"""
        from app.cli import build_llm_client

        options: dict[str, object] = (
            {"profile_name": profile_name} if profile_name else {}
        )
        try:
            client = build_llm_client(options, project_root=self.project_root)
            if getattr(client, "resolved_target", None) is None:
                raise ValueError("模型配置不完整，尚不能用于新输入")
            self.require_host().set_input_model(client)
            self.llm_client = client
            if self.project_root.is_dir():
                self.execution_error = ""
        except Exception as exc:
            self.event_sink(
                "output", f"配置已保存，但应用失败：{exc}；后续输入仍使用原模型"
            )
            return
        self.event_sink(
            "output", "配置已保存并应用到后续新运行；已接纳的运行和定时任务保持原模型"
        )
        self.event_sink("input-model", self.model_label())

    def model_label(self) -> str:
        """显示实际客户端模型与容量；传参：无；返回：非用量的配置说明。"""
        if self.execution_error or self.llm_client is None:
            return f"不可执行：{self.execution_error or '模型尚未配置'}"
        target = getattr(self.llm_client, "resolved_target", None)
        capacity = getattr(target, "context_window", None)
        label = (
            f"{_describe_model(self.llm_client)} · 容量 {capacity:,}"
            if capacity
            else _describe_model(self.llm_client)
        )
        return (
            f"{label} · 思考 {getattr(target, 'reasoning_effort', None) or '模型默认'}"
        )

    def _prompt(self, label: str) -> str:
        """等待全屏浮层回答，不读取终端；传参：问题；返回：明确输入，取消时抛出 EOFError。"""
        answer: Future[str] = Future()
        with self._prompt_lock:
            if self._closed:
                raise EOFError("界面已关闭")
            self._pending_prompt = answer
        try:
            preceding = (
                self._command_output.getvalue().strip()
                if self._command_output is not None
                else ""
            )
            question = f"{preceding}\n\n{label}" if preceding else label
            self.event_sink("prompt", {"label": question, "answer": answer})
            return answer.result()
        finally:
            with self._prompt_lock:
                self._pending_prompt = None
