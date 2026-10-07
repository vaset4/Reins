"""受认证的本机连接与独立后台进程启动。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import json
import importlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

from reins_secrets.store import SecretsVault
from schedules.persistence import claim_file, read_record

RPC_TIMEOUT_SECONDS = 10
STARTUP_TIMEOUT_SECONDS = 20
STARTUP_POLL_SECONDS = 0.1
STOP_TIMEOUT_SECONDS = 10
MILLISECONDS_PER_SECOND = 1000


class BackgroundUnavailable(RuntimeError):
    """无法确认后台的当前状态，调用方不得改在前台重复执行。"""


@dataclass(frozen=True, slots=True)
class BackgroundClient:
    """连接只包含受认证的本机端点，不持有执行循环。"""

    port: int
    token: str = field(repr=False)
    instance: str
    pid: int = 0
    data_space_id: str = ""

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        """调用后台控制入口，写入失败不自动重投；传参：方法及参数；返回：明确回执。"""
        connection = HTTPConnection("127.0.0.1", self.port, timeout=RPC_TIMEOUT_SECONDS)
        try:
            body = json.dumps(
                {
                    "method": method,
                    "params": params,
                    "instance": self.instance,
                    "data_space_id": self.data_space_id,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            connection.request(
                "POST",
                "/rpc",
                body=body,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            if response.status != 200:
                raise BackgroundUnavailable(
                    str(payload.get("error", f"background HTTP {response.status}"))
                )
            if not isinstance(payload, dict):
                raise BackgroundUnavailable("background returned an invalid response")
            return payload
        except (OSError, ValueError) as exc:
            raise BackgroundUnavailable(
                f"无法连接本机后台，提交结果需要重新核对：{exc}"
            ) from exc
        finally:
            connection.close()

    def stop(self, *, timeout: float = STOP_TIMEOUT_SECONDS) -> bool:
        """在发出停止前持有真实进程句柄，等待进程退出而非只等锁释放；传参：等待秒数；返回：是否已退出。"""
        importlib.import_module("pywintypes")
        api = importlib.import_module("win32api")
        con = importlib.import_module("win32con")
        event = importlib.import_module("win32event")
        handle = api.OpenProcess(con.SYNCHRONIZE, False, self.pid)
        try:
            self.call("stop")
            result = event.WaitForSingleObject(
                handle, int(timeout * MILLISECONDS_PER_SECOND)
            )
            return bool(result == event.WAIT_OBJECT_0)
        finally:
            handle.Close()


def connect(data_root: Path) -> BackgroundClient:
    """连接已经发布的后台，不启动进程；传参：数据根；返回：通过实例核对的客户端。"""
    root = data_root / "runtime"
    endpoint_path, key_path = root / "endpoint.json", root / "control.key"
    if not endpoint_path.is_file() or not key_path.is_file():
        raise BackgroundUnavailable("本机后台尚未启动")
    endpoint = read_record(endpoint_path)
    port = endpoint.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
        raise BackgroundUnavailable("后台端口记录无效")
    token = SecretsVault(key_path).get("control_token")
    if not token:
        raise BackgroundUnavailable("后台认证记录缺失")
    space = endpoint.get("data_space_id")
    if not isinstance(space, str) or not space:
        raise BackgroundUnavailable("后台使用旧数据空间协议，请先完成数据切换")
    client = BackgroundClient(
        port, token, str(endpoint["instance"]), data_space_id=space
    )
    response = client.call("ping")
    from runtime.persistence import RuntimeStore

    if (
        response.get("data_space_id") != space
        or RuntimeStore(data_root).data_space_id != space
    ):
        raise BackgroundUnavailable("后台数据空间身份已变化，请重新连接")
    return BackgroundClient(
        port, token, str(endpoint["instance"]), int(response["pid"]), space
    )


def is_running(data_root: Path) -> bool:
    """以进程实际持锁状态判断宿主是否仍存在；传参：数据根；返回：是否有执行者。"""
    path = data_root / "runtime" / "host.lock"
    if not path.is_file():
        return False
    with claim_file(path) as acquired:
        return not acquired


def ensure_running(*, project_root: Path, data_root: Path) -> BackgroundClient:
    """按单实例锁启动独立后台，启动观察超时不重启；传参：两个根目录；返回：已连接客户端。"""
    from runtime.schema_meta import ensure_current_schema

    ensure_current_schema(data_root)

    try:
        return connect(data_root)
    except BackgroundUnavailable:
        pass
    process = None
    with claim_file(data_root / "runtime" / "launch.lock") as launch_owner:
        if launch_owner and not is_running(data_root):
            process = _launch(project_root=project_root, data_root=data_root)
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                return connect(data_root)
            except BackgroundUnavailable:
                if process is not None and process.poll() is not None:
                    raise BackgroundUnavailable(
                        f"后台启动失败（退出码 {process.returncode}），见 {data_root / 'runtime' / 'host.log'}"
                    )
                time.sleep(STARTUP_POLL_SECONDS)
    raise BackgroundUnavailable(
        "未在观察时间内接通后台；进程状态仍需核对，没有重复启动或转到前台执行"
    )


def _launch(*, project_root: Path, data_root: Path) -> subprocess.Popen[bytes]:
    """以无窗口进程运行已授权宿主；传参：目录；返回：用于启动观察的进程句柄。"""
    root = data_root / "runtime"
    root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "app.background.server",
        "--project-root",
        str(project_root),
        "--data-root",
        str(data_root),
    ]
    with (root / "host.log").open("ab", buffering=0) as output:
        return subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
