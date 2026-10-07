"""聊天界面作为后台客户端，退出只停止展示和输入连接。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from io import StringIO
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any
from uuid import uuid4

from rich.text import Text
from rich.console import Console

from app.background.client import (
    BackgroundClient,
    BackgroundUnavailable,
    ensure_running,
)
from app.background.events import decode_event
from app.repl.console import get_console
from app.repl.render import EventRenderer
from app.repl.session_host import SessionHost, SessionHostConfig
from approval import ApprovalDecision, ApprovalRequest, ApprovalUnavailable
from approval.channel import ApprovalChannel
from approval.batch_types import ApprovalBatch, BatchDecision
from llm.public_config import ephemeral_api_key, public_model_config
from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurnComplete,
)

FRONTEND_POLL_SECONDS = 0.15


class RemoteApprovals(ApprovalChannel):
    """前台只传递编号决定，审批等待仍由后台持有。"""

    def __init__(self, host: BackgroundSessionHost) -> None:
        """绑定前台连接；传参：宿主客户端；返回：无。"""
        self.host = host

    def request(self, request: ApprovalRequest) -> ApprovalDecision:
        """拒绝在展示进程启动工具审批；传参：请求；返回：无，明确指出入口错误。"""
        raise ApprovalUnavailable(
            "interactive execution belongs to the background host"
        )

    def answer(self, command: str) -> None:
        """把明确决定交给原审批；传参：完整命令；返回：无。"""
        self.host.client.call(
            "approve", session_id=self.host.session_id, command=command
        )

    def request_batch(self, request: ApprovalBatch) -> BatchDecision:
        """展示进程不能成为授权执行者；传参：批次；返回：无，明确拒绝错误路由。"""
        raise ApprovalUnavailable(
            "interactive execution belongs to the background host"
        )

    def interrupt(self) -> None:
        """显式取消待审批动作；传参：无；返回：无。"""
        self.host.client.call("interrupt_approval", session_id=self.host.session_id)

    def close(self) -> None:
        """关闭显示不会撤销仍在后台等待的审批；传参：无；返回：无。"""
        return


class BackgroundSessionHost(SessionHost):
    """复用聊天渲染和命令界面，所有模型与工具均在独立进程执行。"""

    def __init__(
        self,
        config: SessionHostConfig,
        *,
        event_sink: Callable[[str, Any], None] | None = None,
    ) -> None:
        """启动或连接后台并恢复最近会话；传参：现有界面配置；返回：无。"""
        if config.extensions is not None:
            raise ValueError(
                "in-process extension callbacks require an explicitly injected local SessionHost"
            )
        self.config = config
        self._event_sink = event_sink
        self._connection_lock = RLock()
        self.client = self._connect()
        self.approvals = RemoteApprovals(self)
        self._closed = Event()
        self._cursor: int | None = None
        self._snapshot: dict[str, Any] = {}
        self._approval_id: str | None = None
        self._last_error: str | None = None
        self._input_error: str | None = None
        self._renderer = EventRenderer(trace_on=config.state.trace_on)
        self._need_complete_turn = False
        from app.completion_ui import CompletionMenu

        self._completion_menu = CompletionMenu()
        initial = config.state.session_id if config.state.current_task_id else None
        self._attach(initial)
        self._monitor = Thread(
            target=self._poll, name="reins-frontend-events", daemon=True
        )
        status = self.client.call("status")
        self._present(
            Text(
                f"本机后台已连接。关闭窗口后继续运行；/background 查看状态，/stop 停止当前工作。\n"
                f"未读通知：{status['unread_notifications']}，输入 /notifications 查看。",
                style="dim",
            )
        )
        if status["errors"]:
            self._present(Text(str(status["errors"]), style="red"))
        self._monitor.start()

    def _present(self, value: Any) -> None:
        """将命令回执送到所连界面；传参：Rich 内容；返回：无。"""
        if self._event_sink is None:
            get_console().print(value)
            return
        output = StringIO()
        Console(file=output, width=120, color_system=None).print(value)
        self._event_sink("output", output.getvalue().rstrip())

    def attach_session(self, identity: str | None) -> None:
        """只切换显示会话，不停止其他运行；传参：会话编号；返回：无。"""
        with self._connection_lock:
            self._attach(identity, create=identity is None)

    def refresh_history(self) -> None:
        """从后台重读当前分支并重设事件游标；传参：无；返回：无。"""
        self.attach_session(self.session_id)

    def browse(self, method: str, **params: Any) -> dict[str, Any]:
        """调用后台的会话列表与分支接口；传参：动作和参数；返回：真实投影。"""
        # 【后台连接】【只读查询】1. 客户端每次请求自建认证连接，长读取不占状态轮询和控制锁
        if method != "branch":
            return self.client.call(method, **params)
        with self._connection_lock:
            result = self.client.call(method, **params)
            if params.get("session_id") == self.session_id:
                self._attach(self.session_id)
            return result

    def control_activity(self, choice: dict[str, str]) -> dict[str, str]:
        """将活动浮层的选择交给原会话，切换后拒绝旧选择；参数：明确动作与身份；返回：接纳信息。"""
        with self._connection_lock:
            if choice["session_id"] != self.session_id:
                raise ValueError("所选会话已变化，请重新打开活动")
            if choice["action"] == "cancel":
                self._sync(
                    self.client.call(
                        "cancel",
                        session_id=self.session_id,
                        expected_run_id=choice["run_id"],
                    )
                )
                return {
                    "message": "已请求停止；实际停止状态以执行回执为准",
                    "session_id": self.session_id,
                }
            if choice["action"] != "submit":
                raise ValueError("unknown activity action")
            text = choice["text"]
            if not text.strip():
                raise ValueError("请输入补充要求")
            identity = self.submit(text)
            return {
                "input_id": identity,
                "text": text,
                "session_id": self.session_id,
                "message": "已接纳要求，将由当前模型结合已有结果继续处理",
            }

    @property
    def active(self) -> bool:
        """读取最近的后台运行状态；传参：无；返回：是否仍在执行。"""
        return bool(self._snapshot.get("active", False))

    def set_input_model(self, client: Any) -> None:
        """替换后续输入的模型配置，已接纳执行不变；参数：已校验客户端；返回：无。"""
        self.config = replace(self.config, llm_client=client)
        self._input_error = None

    def submit(
        self,
        text: str,
        *,
        attachment_paths: tuple[str, ...] = (),
        reference_paths: tuple[str, ...] = (),
    ) -> str:
        """将输入连同稳定编号持久接纳到后台；传参：正文；返回：可核对的输入编号。"""
        with self._connection_lock:
            return self._submit(
                text, attachment_paths=attachment_paths, reference_paths=reference_paths
            )

    def _submit(
        self,
        text: str,
        *,
        attachment_paths: tuple[str, ...],
        reference_paths: tuple[str, ...],
    ) -> str:
        """在已冻结选择下接纳输入；参数：正文及附件引用；返回：同一会话的输入编号。"""
        if self.config.state.session_id != self.session_id:
            self._attach(self.config.state.session_id)
        if self._input_error is not None:
            raise BackgroundUnavailable(self._input_error)
        if self.config.llm_client is None:
            raise BackgroundUnavailable(
                "当前工作区没有可执行模型配置；已保存历史仍可查看"
            )
        choice = (
            self._completion_menu.choose(text)
            if not attachment_paths and not reference_paths
            else None
        )
        if choice is not None:
            snapshot = self.client.call(
                "confirm_completion",
                session_id=self.session_id,
                model_config=public_model_config(self.config.llm_client),
                api_key=ephemeral_api_key(self.config.llm_client),
                **choice,
            )
            self._sync(snapshot)
            return str(snapshot["input_id"])
        identity = f"input-{uuid4().hex}"
        snapshot = self.client.call(
            "submit",
            session_id=self.session_id,
            input_id=identity,
            text=text,
            model_config=public_model_config(self.config.llm_client),
            api_key=ephemeral_api_key(self.config.llm_client),
            task_id=self.config.state.current_task_id,
            attachment_paths=list(attachment_paths),
            reference_paths=list(reference_paths),
        )
        self._sync(snapshot)
        self._present("[dim]后台已接纳，将在下一次决策中使用。[/dim]")
        return identity

    def handle_control(self, text: str) -> bool:
        """处理后台、停止、通知和审批命令；传参：用户输入；返回：是否已消费。"""
        try:
            if text.split(maxsplit=1)[0] == "/background":
                self._background_command(text)
            elif text.split(maxsplit=1)[0] == "/notifications":
                self._notifications_command(text)
            elif text in {"/stop", "/pause"}:
                self.cancel()
                self._present("[dim]已请求停止，实际执行状态以运行记录为准。[/dim]")
            elif text.startswith("/approve"):
                self.approvals.answer(text)
            elif text.split(maxsplit=1)[0] in {"/mode", "/revoke"}:
                result = self.client.call(
                    "approval_control",
                    session_id=self.session_id,
                    command=text,
                    action_id=uuid4().hex,
                )
                self._sync(result)
                self._present(Text(result["message"]))
            else:
                return False
        except (ValueError, BackgroundUnavailable) as exc:
            self._present(Text(str(exc), style="red"))
        return True

    def prepare_command(self, text: str) -> None:
        """仅会修改当前持久焦点的高级命令等待运行交接；传参：命令；返回：无。"""
        if (
            text.split(maxsplit=1)[0] in {"/task", "/toolsets"}
            and len(text.split()) > 1
        ):
            self.wait_idle()

    def cancel(self) -> None:
        """将明确停止传到后台及子执行者；传参：无；返回：无。"""
        self._sync(self.client.call("cancel", session_id=self.session_id))

    def wait_idle(self) -> None:
        """显式切换持久状态前等待本会话交接；传参：无；返回：无。"""
        while self.client.call("poll", session_id=self.session_id)["active"]:
            if self._closed.wait(FRONTEND_POLL_SECONDS):
                return

    def close(self) -> None:
        """断开前台展示，不等待或取消后台工作；传参：无；返回：无。"""
        self._closed.set()
        self._monitor.join(timeout=FRONTEND_POLL_SECONDS * 2)
        self.config.registry.close()

    def _connect(self) -> BackgroundClient:
        """使用当前启动目录连接单实例后台；传参：无；返回：认证客户端。"""
        return ensure_running(
            project_root=self.config.project_root, data_root=self.config.data_root
        )

    def _attach(self, identity: str | None, *, create: bool = False) -> None:
        """选择显示会话并从唯一正文恢复历史；传参：可选会话编号；返回：无。"""
        snapshot = self.client.call(
            "create_session" if create else "attach",
            session_id=identity,
            project_root=str(self.config.project_root),
        )
        root = snapshot.get("project_root")
        if not isinstance(root, str) or not root:
            raise BackgroundUnavailable("会话未返回持久工作区，不能继续使用之前的目录")
        self._adopt_workspace(Path(root), snapshot)
        self.session_id = snapshot["session_id"]
        self.config.state.session_id = self.session_id
        self._cursor = snapshot["cursor"]
        self._renderer = EventRenderer(trace_on=self.config.state.trace_on)
        self._need_complete_turn = bool(snapshot["active"])
        self._sync(snapshot)
        self._show_history(snapshot)

    def _adopt_workspace(self, root: Path, snapshot: dict[str, Any]) -> None:
        """接收后台确认的归属并更新REPL本地配置；参数：原工作区与快照；返回：无，失效配置阻止输入。"""
        previous = self.config
        error = snapshot.get("workspace_error")
        selected = replace(previous, project_root=root)
        if self._event_sink is not None or previous.project_root == root:
            self.config, self._input_error = selected, error
            return
        from app.cli import build_llm_client
        from tools.builtin_tools import build_tool_registry
        from tools.tool_registry import ToolRegistry

        try:
            if error is not None:
                raise BackgroundUnavailable(error)
            client = build_llm_client({}, project_root=root)
            registry = build_tool_registry(repo_root=root, data_root=selected.data_root)
        except (OSError, RuntimeError, ValueError) as exc:
            error = f"原工作区执行配置不可用：{exc}"
            client = None
            registry = ToolRegistry()
            self._present(Text(error, style="red"))
        try:
            previous.registry.close()
        except BaseException:
            registry.close()
            raise
        self.config = replace(selected, llm_client=client, registry=registry)
        self._input_error = error

    def _sync(self, snapshot: dict[str, Any]) -> None:
        """更新同一会话的界面投影；传参：后台快照；返回：无。"""
        if snapshot["session_id"] != self.config.state.session_id:
            return
        self._snapshot = snapshot
        for field in ("current_task_id", "compatibility_task_id", "current_run_id"):
            setattr(self.config.state, field, snapshot[field])
        error = snapshot.get("error")
        if error and error != self._last_error:
            self._present(Text(error, style="red"))
        self._last_error = error
        confirmation_text = self._completion_menu.update(
            snapshot.get("completion_requests", [])
        )
        if self._event_sink is not None:
            self._event_sink("snapshot", snapshot)
            return
        if confirmation_text:
            self._present(Text(confirmation_text, style="yellow"))

    def _poll(self) -> None:
        """持续接收同一条事件流，断档时重读正文，连接失败明确显示；传参：无；返回：无。"""
        try:
            while not self._closed.wait(FRONTEND_POLL_SECONDS):
                with self._connection_lock:
                    if self.config.state.session_id != self.session_id:
                        self._attach(self.config.state.session_id)
                    snapshot = self.client.call(
                        "poll", session_id=self.session_id, after=self._cursor
                    )
                    if self._closed.is_set():
                        return
                    if (
                        snapshot.get("event_epoch") != self._snapshot.get("event_epoch")
                        and "history" not in snapshot
                    ):
                        self._attach(self.session_id)
                        continue
                    self._sync(snapshot)
                    self._cursor = snapshot["cursor"]
                    if snapshot["gap"]:
                        self._show_history(snapshot)
                        self._renderer = EventRenderer(
                            trace_on=self.config.state.trace_on
                        )
                        self._need_complete_turn = True
                    self._renderer.trace_on = self.config.state.trace_on
                    self._render_events(snapshot["events"])
                    self._show_approval(snapshot)
        except Exception as exc:
            if not self._closed.is_set():
                if self._event_sink is not None:
                    self._event_sink("disconnected", str(exc))
                self._present(
                    Text(
                        f"后台连接中断：{exc}\n输入 /background 查看状态或 /background start 重新连接。",
                        style="red",
                    )
                )

    def _render_events(self, rows: list[dict[str, Any]]) -> None:
        """中途接回时等待本轮完整正文，避免只展示断档后的尾巴；传参：缓存事件；返回：无。"""
        if self._event_sink is not None:
            return
        for row in rows:
            event = decode_event(row)
            if self._need_complete_turn and isinstance(
                event, (AssistantTextDelta, AssistantReasoningDelta)
            ):
                continue
            self._renderer.render(event)
            if isinstance(event, AssistantTurnComplete):
                self._need_complete_turn = False

    def _show_history(self, snapshot: dict[str, Any]) -> None:
        """显示主存已保存对话，不执行历史工具；传参：接续快照；返回：无。"""
        if self._event_sink is not None:
            return
        history = snapshot.get("history", [])
        if history:
            self._present("[dim]已接回保存的会话：[/dim]")
        for row in history:
            label = "你" if row["role"] == "user" else "Reins"
            self._present(Text(f"{label}：{row['text']}"))
        for question in snapshot.get("questions", []):
            self._present(Text(f"等待你的答复：{question}", style="yellow"))

    def _show_approval(self, snapshot: dict[str, Any]) -> None:
        """展示当前编号和范围，重连仍指向同一审批；传参：快照；返回：无。"""
        if self._event_sink is not None:
            return
        request = snapshot.get("approval")
        if request is None or request["identity"] == self._approval_id:
            return
        self._approval_id = request["identity"]
        if "batch_id" in request:
            self._present(
                Text(
                    f"{request['body']}\n输入 /approve {request['identity']} 后跟全部选择。",
                    style="yellow",
                )
            )
            return
        resource = request.get("resource")
        scope = (
            f"{resource['action']}: {resource['target']}"
            if resource
            else str(request["args"])
        )
        self._present(
            Text(
                f"需要授权：{request['tool']}\n范围：{scope}\n"
                f"输入 /approve {request['identity']} once|task|permanent|deny；直接输入新要求会取消原待审批动作。",
                style="yellow",
            )
        )

    def _background_command(self, text: str) -> None:
        """展示、重连或显式停止整个后台；传参：用户命令；返回：无。"""
        fields = text.split()
        action = fields[1] if len(fields) > 1 else "status"
        if action == "stop":
            stopped = self.client.stop()
            self._present(
                "[dim]后台已退出，运行记录已保留。[/dim]"
                if stopped
                else "[dim]后台仍在交接，尚未确认退出。[/dim]"
            )
            return
        if action == "start":
            previous_space = self.client.data_space_id
            self.client = self._connect()
            self._attach(
                self.session_id if self.client.data_space_id == previous_space else None
            )
            if not self._monitor.is_alive():
                self._monitor = Thread(
                    target=self._poll, name="reins-frontend-events", daemon=True
                )
                self._monitor.start()
        elif action != "status":
            raise ValueError("用法：/background [status|start|stop]")
        status = self.client.call("status")
        active = sum(row["active"] for row in status["sessions"])
        self._present(
            Text(
                f"后台：{status['status']}；执行中会话：{active}；定时工作：{len(status['running_occurrences'])}；"
                f"未读通知：{status['unread_notifications']}"
            )
        )
        if status["errors"]:
            self._present(Text(str(status["errors"]), style="red"))

    def _notifications_command(self, text: str) -> None:
        """查看完整通知后记可见，只有明确 read 才记已读；传参：用户命令；返回：无。"""
        fields = text.split()
        if len(fields) == 1:
            rows = self.client.call("notifications", action="list")["notifications"]
            for row in rows:
                self._present(
                    Text(
                        f"{row['notification_id']} · {row['title']} · {row['delivery_status']}"
                    )
                )
            self._present(
                "[dim]/notifications show|read|retry 通知编号[/dim]"
                if rows
                else "暂无未读通知。"
            )
            return
        if len(fields) != 3 or fields[1] not in {"show", "read", "retry"}:
            raise ValueError("用法：/notifications [show|read|retry 通知编号]")
        action, identity = fields[1:]
        if action == "show":
            row = self.client.call("notifications", action="get", identity=identity)
            self._present(Text(f"{row['title']}\n{row['message']}"))
            self.client.call("notifications", action="visible", identity=identity)
        else:
            self.client.call("notifications", action=action, identity=identity)
            self._present(
                "已确认已读。"
                if action == "read"
                else "已接纳重发；原送达情况未知时可能再次出现。"
            )
