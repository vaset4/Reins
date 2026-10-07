"""后台进程的本机控制端口；只接受私有令牌认证的结构化请求。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import secrets
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from uuid import uuid4

import approval
from approval.batch import register_batch_backend
from app.background.service import BackgroundService
from app.background.sessions import BackgroundSession, SessionServices
from app.background.scheduled_session import ScheduledSession
from app.background.inspection import InspectionService
from reins_secrets.store import SecretsVault
from runtime.schema_meta import ensure_current_schema
from schedules.persistence import claim_file, write_record

MAX_RPC_BYTES = 8 * 1024 * 1024
_LOG = logging.getLogger(__name__)


class BackgroundServer(ThreadingHTTPServer):
    """只监听 IPv4 loopback 的后台控制服务。"""

    daemon_threads = True

    def __init__(self, service: BackgroundService, token: str) -> None:
        """绑定实例和认证状态；传参：后台组合根和私有令牌；返回：无。"""
        self.service, self.token = service, token
        self.instance = uuid4().hex
        self.data_space_id = service.workspaces.database.data_space_id
        self.inspection = InspectionService(service.services.data_root)
        super().__init__(("127.0.0.1", 0), BackgroundRequest)

    def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """分派有限控制动作，业务推进仍归模型；传参：方法和结构化参数；返回：回执。"""
        if method == "ping":
            return {
                "instance": self.instance,
                "data_space_id": self.data_space_id,
                "data_root": str(self.service.services.data_root),
                "pid": os.getpid(),
                "status": "stopping" if self.service.stopping.is_set() else "running",
            }
        if method == "status":
            return {
                "instance": self.instance,
                "data_space_id": self.data_space_id,
                "data_root": str(self.service.services.data_root),
                **self.service.status(),
            }
        if method == "stop":
            self.service.stop()
            Thread(
                target=self.shutdown, name="reins-host-shutdown", daemon=True
            ).start()
            return {"status": "stopping"}
        if method == "notifications":
            return self.service.notifications(
                _text(params, "action"), params.get("identity")
            )
        if method == "context_management":
            from app.background.context_management import ContextManagement

            payload = params.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("context management payload must be an object")
            return {
                "instance": self.instance,
                **ContextManagement(self.service.services.data_root).query(payload),
            }
        if method in {
            "file_restore_query",
            "file_restore_execute",
            "file_restore_cancel",
        }:
            payload = params.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("file restore payload must be an object")
            action = {
                "file_restore_query": self.service.file_restore.query,
                "file_restore_execute": self.service.file_restore.execute,
                "file_restore_cancel": self.service.file_restore.cancel,
            }[method]
            result = action(payload)
            if (
                method == "file_restore_query"
                and payload.get("action") == "status"
                and not result
            ):
                return {}
            return {
                "instance": self.instance,
                "data_space_id": self.data_space_id,
                **result,
            }
        if method in {"query_evidence", "export_evidence"}:
            payload = params.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("evidence payload must be an object")
            result = (
                self.inspection.query(payload)
                if method == "query_evidence"
                else self.inspection.export(payload)
            )
            return {
                "instance": self.instance,
                "data_space_id": self.data_space_id,
                **result,
            }
        if method in {"attach", "create_session"}:
            project_root = (
                Path(_text(params, "project_root"))
                if "project_root" in params
                else None
            )
            session: BackgroundSession | ScheduledSession
            if method == "create_session":
                if project_root is None:
                    raise ValueError("new session requires project_root")
                session = self.service.create_session(project_root)
            else:
                session = self.service.attach(
                    params.get("session_id"), project_root=project_root
                )
            return {"instance": self.instance, **session.snapshot(history=True)}
        if method == "tool_result_detail":
            from app.background.tool_results import tool_result_detail

            source = {
                key: _text(params, key)
                for key in ("session_id", "run_id", "entry_id", "call_id")
            }
            return tool_result_detail(self.service.services.data_root, **source)
        if method == "session_search":
            from app.background.navigation import session_search

            return session_search(self.service.services.data_root, params)
        if method in {"sessions", "session_tree", "history_page"}:
            from app.background.navigation import (
                history_page,
                list_sessions,
                session_tree,
            )

            if method == "sessions":
                query = params.get("query", "")
                if not isinstance(query, str):
                    raise ValueError("session query must be text")
                return list_sessions(
                    self.service.services.data_root,
                    query,
                    before=params.get("before"),
                    workspace_id=params.get("workspace_id"),
                    purpose=params.get("purpose", "chat"),
                    owner_session_id=params.get("owner_session_id"),
                )
            if method == "history_page":
                options: dict[str, Any] = {
                    key: _text(params, key)
                    for key in ("leaf_id", "before")
                    if params.get(key) is not None
                }
                if "limit" in params:
                    options["limit"] = params["limit"]
                return history_page(
                    self.service.services.data_root,
                    _text(params, "session_id"),
                    **options,
                )
            return session_tree(
                self.service.services.data_root,
                _text(params, "session_id"),
                after=params.get("after", 0),
                view=params.get("view", "turns"),
                query=params.get("query", ""),
                selected_entry=params.get("selected_entry"),
            )
        return {
            "instance": self.instance,
            "data_space_id": self.data_space_id,
            **self._session_action(method, params),
        }

    def server_close(self) -> None:
        """关闭查询导出资源与HTTP端口；参数：无；返回：无，不重放执行。"""
        try:
            self.inspection.close()
        finally:
            super().server_close()

    def _session_action(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """把会话输入、停止和审批交给所属执行者；传参：动作及参数；返回：当前状态。"""
        session = self.service.attach(_text(params, "session_id"))
        if method in {"activity", "settings", "poll"}:
            return self._session_view(session, method, params)
        if method in {"submit", "confirm_completion"}:
            return self._session_input(session, method, params)
        if method == "branch":
            if not isinstance(session, BackgroundSession):
                raise ValueError("定时会话不能切换交互分支")
            session.branch(_text(params, "entry_id"))
            return session.snapshot(history=True)
        if method == "approval_control":
            message = session.approval_control(
                _text(params, "command"), _text(params, "action_id")
            )
            return {"message": message, **session.snapshot()}
        if method == "cancel":
            expected = params.get("expected_run_id")
            if expected is not None and not isinstance(expected, str):
                raise ValueError("expected_run_id must be text")
            session.cancel(expected_run_id=expected)
        elif method == "approve":
            session.approvals.answer(_text(params, "command"))
        elif method == "interrupt_approval":
            session.approvals.interrupt()
        else:
            raise ValueError(f"unknown background method: {method}")
        return session.snapshot()

    def _session_view(
        self,
        session: BackgroundSession | ScheduledSession,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """查询执行端事实，不触发模型或更改权限；参数：所属会话、查询及参数；返回：只读视图。"""
        if method == "activity":
            from app.background.activity import activity_snapshot

            return activity_snapshot(self.service, _text(params, "session_id"))
        if method == "settings":
            if isinstance(session, BackgroundSession):
                return session.settings()
            return {
                "session_id": _text(params, "session_id"),
                "model": None,
                "approval_mode": None,
                "mcp": None,
                "execution_attached": False,
                "scheduled": True,
            }
        after = params.get("after")
        if after is not None and (
            not isinstance(after, int) or isinstance(after, bool) or after < 0
        ):
            raise ValueError("event cursor must be a non-negative integer")
        return session.snapshot(after=after)

    def _session_input(
        self,
        session: BackgroundSession | ScheduledSession,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """校验并接纳用户输入或确认，所有执行仍归会话；参数：所属会话、输入动作及参数；返回：接纳回执。"""
        from app.background.attachments import validate_paths

        if method == "confirm_completion":
            accepted = params.get("accepted")
            if type(accepted) is not bool:
                raise ValueError("completion choice must be a boolean")
            config = params.get("model_config")
            if isinstance(session, BackgroundSession) and not isinstance(config, dict):
                raise ValueError("model_config must be an object")
            options: dict[str, Any] = (
                {"model_config": config}
                if isinstance(session, BackgroundSession)
                else {}
            )
            identity = session.confirm_completion(
                question_id=_text(params, "question_id"),
                action_id=_text(params, "action_id"),
                accepted=accepted,
                api_key=params.get("api_key"),
                **options,
            )
            return {"input_id": identity, **session.snapshot()}
        config = params.get("model_config")
        if not isinstance(config, dict):
            raise ValueError("model_config must be an object")
        attachments = validate_paths(params.get("attachment_paths", []))
        references = validate_paths(params.get("reference_paths", []))
        text = params.get("text")
        if not isinstance(text, str) or (
            not text.strip() and not attachments and not references
        ):
            raise ValueError("input must contain text or attachments")
        options = {"attachment_paths": attachments, "reference_paths": references}
        if not isinstance(session, BackgroundSession) and (attachments or references):
            raise ValueError("定时会话不接纳文件附件")
        identity = session.submit(
            text,
            input_id=_text(params, "input_id"),
            model_config=config,
            api_key=params.get("api_key"),
            task_id=params.get("task_id"),
            **(options if isinstance(session, BackgroundSession) else {}),
        )
        return {"input_id": identity, **session.snapshot()}


class BackgroundRequest(BaseHTTPRequestHandler):
    """处理一次小型本机 RPC，认证失败不解析业务正文。"""

    server: BackgroundServer

    def do_POST(self) -> None:
        """认证、校验再分派控制请求；传参：HTTP 请求；返回：JSON 回执。"""
        expected = f"Bearer {self.server.token}"
        if self.path != "/rpc" or not hmac.compare_digest(
            self.headers.get("Authorization", ""), expected
        ):
            self._respond(403, {"error": "background authentication failed"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if (
                self.headers.get("Content-Type") != "application/json"
                or not 0 < length <= MAX_RPC_BYTES
            ):
                raise ValueError("invalid RPC content type or length")
            value = json.loads(self.rfile.read(length))
            if (
                not isinstance(value, dict)
                or value.get("instance") != self.server.instance
            ):
                raise ValueError(
                    "background instance changed; reconnect before issuing a command"
                )
            if value.get("data_space_id") != self.server.data_space_id:
                raise ValueError(
                    "data space changed; reconnect before issuing a command"
                )
            params = value.get("params")
            if not isinstance(params, dict):
                raise ValueError("RPC params must be an object")
            result = self.server.dispatch(_text(value, "method"), params)
        except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
            self._respond(400, {"error": str(exc)})
        except Exception as exc:
            _LOG.exception("【本机后台】【控制失败】请求未能完成")
            self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
        else:
            self._respond(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        """常规访问不记录正文或令牌；传参：HTTP 日志格式；返回：无。"""
        return

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        """返回明确 JSON 状态，前台断开不取消后台工作；传参：状态及内容；返回：无。"""
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            _LOG.info("【本机后台】【界面断开】接纳状态保留，界面可用原输入编号核对")


def serve(*, project_root: Path, data_root: Path) -> int:
    """持有唯一进程锁运行服务；传参：已解析目录；返回：进程退出码。"""
    ensure_current_schema(data_root)
    with claim_file(data_root / "runtime" / "host.lock") as acquired:
        if not acquired:
            return 0
        dependencies = workspace_services(project_root, data_root=data_root)
        service = BackgroundService(dependencies)
        token = secrets.token_urlsafe(32)
        SecretsVault(data_root / "runtime" / "control.key").set("control_token", token)
        server = BackgroundServer(service, token)
        approval.register_approval_backend(service.approve)
        register_batch_backend(service.approve_batch)
        try:
            service.start()
            write_record(
                data_root / "runtime" / "endpoint.json",
                {
                    "port": server.server_port,
                    "instance": server.instance,
                    "data_space_id": server.data_space_id,
                    "pid": os.getpid(),
                },
            )
            server.serve_forever(poll_interval=0.2)
        finally:
            service.close()
            server.server_close()
            approval.register_approval_backend(None)
            register_batch_backend(None)
    return 0


def workspace_services(project_root: Path, *, data_root: Path) -> SessionServices:
    """冻结每个工作区的配置及工具工厂；参数：原目录与共享数据空间；返回：惰性执行依赖。"""
    from app.cli import build_llm_client
    from tools.builtin_tools import build_tool_registry

    return SessionServices(
        project_root,
        data_root,
        partial(build_llm_client, project_root=project_root),
        partial(build_tool_registry, repo_root=project_root, data_root=data_root),
        partial(workspace_services, data_root=data_root),
    )


def main() -> int:
    """解析独立进程所需目录；传参：命令行；返回：服务退出码。"""
    parser = argparse.ArgumentParser(description="Reins local background host")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    return serve(
        project_root=args.project_root.resolve(), data_root=args.data_root.resolve()
    )


def _text(params: dict[str, Any], name: str) -> str:
    """校验本机控制边界的必要文本；传参：参数和字段；返回：非空原文。"""
    value = params.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
