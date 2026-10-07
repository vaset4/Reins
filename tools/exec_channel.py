from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from runtime.cancellation import CancellationToken
from tools.exec_process_tree import CREATE_SUSPENDED, ProcessTree

# ExecChannel：唯一受控代码执行核（GA 式落盘临时脚本 + 受控 header +
# sys.executable -X utf8 -u + type 分流 py/shell + 流式读 + 超时杀进程组）。
#
# 安全缺口（显式，非静默降级）：LocalSubprocessBackend **不是 OS 沙箱**。它只做
# 子进程 + env 机密剥离 + 输出脱敏 + 进程组 kill——一旦代码放行，删文件/联网/改系统
# 在本机仍不可控。Windows Job 只核验和停止进程树，不提供文件或网络隔离。
# OS 级硬隔离由 ExecBackend 抽象预留接口。fs 边界由调用方在 tool_registry 闸门经
# path_security 强制（exec_boundary），ExecChannel 本身不替代该闸门。

ExecKind = Literal["py", "shell"]
_VALID_KINDS: frozenset[str] = frozenset({"py", "shell"})

# 子进程 env 机密黑名单（借 hermes local.py:19-138）：名字含这些片段即剥离，
# 防机密下发到放行的子进程。安全前缀白名单放行系统必需变量。
_SECRET_ENV_MARKERS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
)
_SECRET_REDACTION = "[REDACTED]"
# 自动 env 输出脱敏的最小长度护栏：短的 secret 命名配置值（如 AUTH_FLAG=ok）不自动
# 打码，避免污染正常输出；显式 ExecSpec.redact_values 不受此限。
_MIN_AUTO_REDACT_LEN = 8
_PROCESS_POLL_SECONDS = 0.05
_PROCESS_STOP_SECONDS = 2.0

# 受控 header：编码兼容（utf-8 失败回退 gbk）+ 报错引导，纯文本无安全逻辑。
_PY_HEADER = (
    "# -*- coding: utf-8 -*-\n"
    "import sys as _sys\n"
    "try:\n"
    "    _sys.stdout.reconfigure(encoding='utf-8')\n"
    "    _sys.stderr.reconfigure(encoding='utf-8')\n"
    "except Exception:\n"
    "    pass\n"
)


@dataclass(frozen=True, slots=True)
class ExecSpec:
    code: str
    kind: ExecKind
    cwd: Path
    timeout_seconds: float = 30.0
    env_passthrough: Mapping[str, str] | None = None
    redact_values: Sequence[str] = field(default_factory=tuple)
    cancellation: CancellationToken | None = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unsupported exec kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    cancelled: bool = False
    execution_state: str = "completed"
    pid: int | None = None
    stop_error: str = ""


class ExecBackend(ABC):
    # 预留接口：OS 隔离后端可实现此基类而不改调用方。
    # 本卡只提供 LocalSubprocessBackend（非 OS 沙箱）。
    @abstractmethod
    def run(self, spec: ExecSpec) -> ExecResult: ...


def strip_secret_env(source: Mapping[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for name, value in source.items():
        if _is_secret_name(name):
            continue
        cleaned[name] = value
    return cleaned


def redact_secrets(text: str, values: Sequence[str]) -> str:
    redacted = text
    for value in values:
        if value:
            redacted = redacted.replace(value, _SECRET_REDACTION)
    return redacted


def _is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_ENV_MARKERS)


def _secret_env_values(source: Mapping[str, str]) -> tuple[str, ...]:
    # 自动脱敏被剥离的 secret 命名 env 值（AC4/R5：输出脱敏生效，不依赖调用方喂值）。
    # 仅用 os.environ，不碰 SecretsVault（C16 边界）。长度护栏见 _MIN_AUTO_REDACT_LEN。
    return tuple(
        value
        for name, value in source.items()
        if _is_secret_name(name) and len(value) >= _MIN_AUTO_REDACT_LEN
    )


class LocalSubprocessBackend(ExecBackend):
    # 非 OS 沙箱：子进程 + env 剥机密 + 输出脱敏 + 进程组 kill（hermes local.py 式
    # 最小安全层，不依赖容器）。代码副作用一旦放行不受 OS 隔离约束——见模块头说明。
    def run(self, spec: ExecSpec) -> ExecResult:
        if spec.cancellation is not None and spec.cancellation.cancelled:
            return ExecResult("", "", -1, cancelled=True, execution_state="not_started")
        if spec.kind == "py":
            return self._run_python(spec)
        return self._run_shell(spec)

    def _run_python(self, spec: ExecSpec) -> ExecResult:
        script_path = _write_temp_script(spec.code, spec.cwd)
        cmd = [sys.executable, "-X", "utf8", "-u", str(script_path)]
        try:
            return self._spawn(cmd, spec)
        finally:
            script_path.unlink(missing_ok=True)

    def _run_shell(self, spec: ExecSpec) -> ExecResult:
        """将命令正文原样交给cmd，保留路径和脚本引号；参数：执行规格；返回：真实进程结果。"""
        shell = subprocess.list2cmdline([os.environ.get("COMSPEC", "cmd.exe")])
        # 1. 【代码执行】【命令引号】cmd正文不是C运行库argv，不能让list2cmdline把内部引号改成反斜线转义
        cmd = f'{shell} /s /c "{spec.code}"'
        return self._spawn(cmd, spec)

    def _spawn(self, cmd: str | list[str], spec: ExecSpec) -> ExecResult:
        """【代码执行】【启动子进程】按运行配置启动并收集真实结果；传参：命令和执行规格；返回：进程证据。"""
        env = _build_env(spec.env_passthrough)
        redact = (*spec.redact_values, *_secret_env_values(os.environ))
        launch_cwd = spec.cwd
        if os.name == "nt" and spec.kind == "py":
            # 1. 【代码执行】【长路径启动】CreateProcess限制初始目录长度，Python头部再进入完整工作目录
            launch_cwd = Path(spec.cwd.resolve().anchor)
        tree = ProcessTree() if os.name == "nt" else None
        try:
            if spec.cancellation is not None and spec.cancellation.cancelled:
                return ExecResult(
                    "", "", -1, cancelled=True, execution_state="not_started"
                )
            process = subprocess.Popen(
                cmd,
                cwd=str(launch_cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_creation_flags()
                | (CREATE_SUSPENDED if tree is not None else 0),
                start_new_session=_new_session(),
            )
            if tree is not None:
                try:
                    # 2. 【代码执行】【进程树归属】用户代码恢复前纳入Job，不能等根进程退出后才猜后代
                    tree.attach(int(getattr(process, "_handle")))
                    if spec.cancellation is not None and spec.cancellation.cancelled:
                        # 3. 【代码执行】【取消启动】挂起进程尚未运行用户代码，确认终止后保持未开始语义
                        process.kill()
                        process.communicate(timeout=_PROCESS_STOP_SECONDS)
                        spec.cancellation.report_backend(
                            backend="local_subprocess",
                            pid=process.pid,
                            supports_stop=True,
                            stop_confirmed=True,
                            process_tree_stopped=True,
                        )
                        return ExecResult(
                            "",
                            "",
                            -1,
                            cancelled=True,
                            execution_state="not_started",
                            pid=process.pid,
                        )
                    tree.resume(process.pid)
                except BaseException:
                    process.kill()
                    process.wait(timeout=_PROCESS_STOP_SECONDS)
                    raise
            return _collect(process, spec, redact, tree=tree)
        finally:
            if tree is not None:
                tree.close()


def _collect(
    process: subprocess.Popen[str],
    spec: ExecSpec,
    redact_values: Sequence[str],
    *,
    tree: ProcessTree | None = None,
) -> ExecResult:
    """等待进程真实结果并响应停止，停止失败仍等待迟到结果；传参：进程、执行配置和脱敏值；返回：执行证据。"""
    token = spec.cancellation or CancellationToken()
    token.report_backend(
        backend="local_subprocess", pid=process.pid, supports_stop=True
    )
    deadline = time.monotonic() + spec.timeout_seconds
    timed_out, cancelled, stopping, confirmed = False, False, False, False
    stop_error = ""
    while True:
        timed_out = (
            timed_out or time.monotonic() >= deadline or token.reason == "timeout"
        )
        cancelled = cancelled or (token.cancelled and not timed_out)
        if (timed_out or cancelled) and not stopping:
            stopping = True
            try:
                _kill_process_tree(process, tree=tree)
                confirmed = True
                token.report_backend(stop_confirmed=True)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                stop_error = str(exc)
                token.report_backend(stop_confirmed=False, stop_error=stop_error)
        try:
            stdout, stderr = process.communicate(timeout=_PROCESS_POLL_SECONDS)
            if tree is not None:
                try:
                    active = tree.active_count()
                except OSError as exc:
                    token.report_backend(
                        process_tree_stopped=False, process_query_error=str(exc)
                    )
                    time.sleep(_PROCESS_POLL_SECONDS)
                    continue
                if active:
                    time.sleep(_PROCESS_POLL_SECONDS)
                    continue
            break
        except subprocess.TimeoutExpired:
            continue
    token.report_backend(process_tree_stopped=True)
    return ExecResult(
        stdout=redact_secrets(stdout or "", redact_values),
        stderr=redact_secrets(stderr or "", redact_values),
        exit_code=process.returncode if process.returncode is not None else -1,
        timed_out=timed_out,
        cancelled=cancelled,
        pid=process.pid,
        execution_state="stopped" if confirmed else "completed",
        stop_error=stop_error,
    )


def _build_env(passthrough: Mapping[str, str] | None) -> dict[str, str]:
    base = strip_secret_env(os.environ)
    if passthrough:
        base.update(dict(passthrough))
    return base


def _write_temp_script(code: str, cwd: Path) -> Path:
    """【代码执行】【脚本落盘】在实际目录保存临时脚本并固定执行位置；传参：代码和目录；返回：绝对脚本路径。"""
    execution_dir = cwd.resolve()
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".py",
        dir=str(execution_dir),
        delete=False,
    )
    with handle:
        handle.write(_PY_HEADER)
        # 1. 【代码执行】【工作目录】只切换子进程目录，保持相对文件及同目录模块按原规格解析
        handle.write(f"import os as _os\n_os.chdir({str(execution_dir)!r})\n")
        handle.write(code)
    return Path(handle.name)


def _creation_flags() -> int:
    # Windows 创建独立进程组；实际进程树由启动前绑定的 Job 查询和终止
    # POSIX 不用 creationflags（恒 0）
    if os.name == "nt":
        return subprocess.CREATE_NEW_PROCESS_GROUP
    return 0


def _new_session() -> bool:
    # POSIX 用 start_new_session（setsid）建立独立进程组以便 killpg；Windows 不支持。
    return os.name != "nt"


def _kill_process_tree(
    process: subprocess.Popen[str], *, tree: ProcessTree | None = None
) -> None:
    """终止实际进程树并核对系统回执；传参：子进程；返回：无，未确认停止时抛错。"""
    if tree is not None:
        tree.terminate()
        deadline = time.monotonic() + _PROCESS_STOP_SECONDS
        while tree.active_count():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Windows Job stop requested but process tree still active"
                )
            time.sleep(_PROCESS_POLL_SECONDS)
    elif os.name == "nt":
        _taskkill(process.pid)
    else:
        _killpg(process)
    process.wait(timeout=_PROCESS_STOP_SECONDS)


def _taskkill(pid: int) -> None:
    """Windows按PID终止整棵进程树；传参：根进程PID；返回：无，失败不伪造已停止。"""
    result = subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True,
        check=False,
        timeout=_PROCESS_STOP_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"taskkill failed for pid {pid}: exit {result.returncode}")


def _killpg(process: subprocess.Popen[str]) -> None:
    # POSIX-only 分支（_kill_process_tree 已按 os.name 分流，Windows 永不到此）。
    import signal

    try:
        pgid = os.getpgid(process.pid)  # type: ignore[attr-defined]
        os.killpg(pgid, signal.SIGTERM)  # type: ignore[attr-defined]
        time.sleep(1)
        os.killpg(pgid, signal.SIGKILL)  # type: ignore[attr-defined]
    except (ProcessLookupError, PermissionError):
        process.kill()


class ExecChannel:
    # 唯一受控执行通道。三入口（code_execution/terminal/skill）全委托此处。
    # 不替代 tool_registry 闸门——lease/approval/path_security/exec_boundary 在闸门
    # 强制；ExecChannel 是闸门之后的受控执行核（见 design §1）。
    def __init__(self, backend: ExecBackend) -> None:
        self._backend = backend

    def run(self, spec: ExecSpec) -> ExecResult:
        """在真实本地后端边界接入捕获；传参：执行规格；返回：真实进程结果。"""
        from runtime.file_capture import observe_local_execution

        return observe_local_execution(lambda: self._backend.run(spec))
