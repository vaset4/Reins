from __future__ import annotations

import argparse
import getpass
import sys
from dataclasses import replace
from pathlib import Path

from app.gateway import run_gateway_stdio
from app.repl import run_repl
from app.run_task import (
    RunTaskResponse,
    execute_resume,
    inspect_resume,
    run_task,
    resolve_resume_identity,
)
from app.startup import StartupIdentity, resolve_startup_identity
from llm.base import LLMClient
from llm.client import MissingConfigurationLLMClient, RealLLMClient
from llm.config import (
    LLMProviderConfig,
    load_project_llm_defaults,
    load_saved_config,
)
from llm.profiles import ModelProfile, load_model_profiles
from llm.resolved_target import resolve_model_target
from reins_secrets.store import SecretsVault
from runtime.schema_meta import UnsupportedSchemaError, ensure_current_schema


EXIT_OK = 0
EXIT_SERVICE_ERROR = 1
EXIT_PAUSED = 3


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local agent core.")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--task", help="Task input for the runtime loop.")
    _add_llm_options(parser)

    gateway_parser = subparsers.add_parser("gateway", help="Run stdin/stdout gateway.")
    _add_llm_options(gateway_parser)

    chat_parser = subparsers.add_parser("chat", help="Run one chat-style input.")
    chat_parser.add_argument("--session", dest="session_id")
    chat_parser.add_argument(
        "--task", dest="initial_task", help="One-shot prompt; omit to enter REPL."
    )
    _add_llm_options(chat_parser)

    repl_parser = subparsers.add_parser(
        "repl", help="Enter the interactive REPL (default when no subcommand is given)."
    )
    repl_parser.add_argument(
        "--resume",
        dest="resume_task_id",
        help="Resume a paused or active task by task_id.",
    )
    _add_llm_options(repl_parser)

    resume_parser = subparsers.add_parser(
        "resume", help="Inspect or execute a checkpoint resume."
    )
    resume_parser.add_argument("--checkpoint", required=True, dest="checkpoint_id")
    resume_parser.add_argument(
        "--execute", action="store_true", help="Create and run a new resume segment."
    )
    resume_parser.add_argument(
        "--decision",
        choices=("skip", "replay"),
        help="Resolve a pending tool while executing the resume.",
    )
    _add_llm_options(resume_parser)

    secret_parser = subparsers.add_parser("secret", help="Manage local secrets.")
    secret_subparsers = secret_parser.add_subparsers(dest="secret_command")
    secret_set = secret_subparsers.add_parser("set", help="Set one secret value.")
    secret_set.add_argument("name")
    secret_subparsers.add_parser("list", help="List secret names.")
    secret_delete = secret_subparsers.add_parser("delete", help="Delete one secret.")
    secret_delete.add_argument("name")

    background = subparsers.add_parser(
        "background",
        help="Inspect, start or explicitly stop the local background host.",
    )
    background.add_argument(
        "background_action",
        choices=("status", "start", "stop"),
        nargs="?",
        default="status",
    )
    background.add_argument("--data-root", type=Path, dest="background_data_root")

    return parser


def build_llm_client(
    cli_overrides: dict[str, object],
    project_root: Path | None = None,
) -> LLMClient:
    """按共同配置优先级构造模型，定时任务可固定命名配置；传参：公开覆盖与项目根；返回：模型客户端。"""
    resolved_root = project_root or Path(__file__).resolve().parent.parent
    profile_name = cli_overrides.get("profile_name")
    active_profile = (
        _load_named_model_profile(str(profile_name))
        if profile_name is not None
        else _load_active_model_profile()
    )
    file_defaults = _merge_model_defaults(
        load_project_llm_defaults(resolved_root),
        active_profile,
    )
    model_overrides = (
        {**active_profile.as_config(), **cli_overrides}
        if profile_name is not None and active_profile is not None
        else cli_overrides
    )
    saved = load_saved_config()

    try:
        vault: SecretsVault | None = SecretsVault()
    except Exception as exc:
        raise RuntimeError(f"secrets vault unavailable: {exc}") from exc

    target = resolve_model_target(
        cli_overrides=model_overrides,
        saved_config=saved,
        file_defaults=file_defaults,
        secrets_vault=vault,
    )
    if active_profile is not None:
        target = replace(
            target,
            profile_name=active_profile.name,
            credential_name=active_profile.credential,
        )

    config = LLMProviderConfig(
        base_url=target.base_url,
        model=target.model,
        api_key=target.api_key,
        timeout_seconds=target.timeout_seconds,
    )

    if config.is_configured:
        return RealLLMClient(config=config, resolved_target=target)
    return MissingConfigurationLLMClient(config.missing_required_fields)


def main() -> int:
    """解析启动身份并分派 CLI 子命令

    参数：无，命令行参数从 sys.argv 读取
    返回：子命令退出码
    """
    identity = resolve_startup_identity()
    parser = build_run_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.command == "resume" and args.decision is not None and not args.execute:
        parser.error("resume --decision requires --execute")

    if args.command == "secret":
        return _run_secret_command(args)
    if args.command == "background":
        if args.background_data_root is not None:
            identity = replace(identity, data_root=args.background_data_root.resolve())
        return _run_background_command(args.background_action, identity)
    # 【Schema Startup】【CLI 入口】模型和业务运行前只通过唯一 gate 验证 data root
    try:
        ensure_current_schema(identity.data_root)

    except UnsupportedSchemaError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_SERVICE_ERROR
    return _run_runtime_command(args, identity)


def _run_runtime_command(args: argparse.Namespace, identity: StartupIdentity) -> int:
    """在数据格式检查后启动会话或恢复运行。

    传参：args 为已解析命令；identity 为启动目录；返回：实际入口退出码
    """
    if args.command == "resume":
        return _run_resume_command(args, identity)
    llm_client = build_llm_client(
        _build_cli_overrides(args), project_root=identity.project_root
    )
    if args.command == "gateway":
        return run_gateway_stdio(identity=identity, llm_client=llm_client)
    task = args.initial_task if args.command == "chat" else args.task
    if task and args.command != "repl":
        response = run_task(
            task=task,
            project_root=identity.project_root,
            data_root=identity.data_root,
            llm_client=llm_client,
            session_id=args.session_id if args.command == "chat" else None,
        )
        _print_response(response)
        return EXIT_OK
    # 【会话】【CLI 入口】未指定一次性目标时进入连续聊天；repl 可显式选择已有任务
    from app.background.frontend import BackgroundSessionHost

    return run_repl(
        project_root=identity.project_root,
        data_root=identity.data_root,
        llm_client=llm_client,
        initial_task_id=getattr(args, "resume_task_id", None),
        host_factory=BackgroundSessionHost,
    )


def _run_background_command(action: str, identity: StartupIdentity) -> int:
    """查询、启动或明确停止后台，查询不暗中启动；传参：动作和目录；返回：真实状态退出码。"""
    from app.background.client import (
        BackgroundUnavailable,
        connect,
        ensure_running,
        is_running,
    )

    try:
        if action != "start" and not is_running(identity.data_root):
            print("本机后台已停止。")
            return EXIT_OK
        client = (
            ensure_running(
                project_root=identity.project_root, data_root=identity.data_root
            )
            if action == "start"
            else connect(identity.data_root)
        )
        if action == "stop":
            if not client.stop():
                print("已请求停止，后台仍在交接；尚未确认退出。")
                return EXIT_PAUSED
            print("本机后台已退出，运行记录已保留。")
            return EXIT_OK
        status = client.call("status")
        print(
            f"本机后台：{status['status']}；会话：{len(status['sessions'])}；未读通知：{status['unread_notifications']}"
        )
        if status["errors"]:
            print(status["errors"], file=sys.stderr)
            return EXIT_SERVICE_ERROR
        return EXIT_OK
    except (BackgroundUnavailable, UnsupportedSchemaError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_SERVICE_ERROR


def _add_llm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-base-url", dest="llm_base_url")
    parser.add_argument("--llm-model", dest="llm_model")
    parser.add_argument("--llm-api-key", dest="llm_api_key")
    parser.add_argument("--llm-timeout-seconds", dest="llm_timeout_seconds", type=float)


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "base_url": getattr(args, "llm_base_url", None),
        "model": getattr(args, "llm_model", None),
        "api_key": getattr(args, "llm_api_key", None),
        "timeout_seconds": getattr(args, "llm_timeout_seconds", None),
    }


def _load_active_model_profile() -> ModelProfile | None:
    profiles = load_model_profiles()
    return profiles.active_profile


def _load_named_model_profile(name: str) -> ModelProfile:
    """按保存的名称读取配置，不随界面当前选择漂移；传参：名称；返回：指定模型配置。"""
    profiles = load_model_profiles()
    if name not in profiles.profiles:
        raise ValueError(f"model profile not found: {name}")
    return profiles.profiles[name]


def _merge_model_defaults(
    legacy_defaults: dict[str, object],
    active_profile: ModelProfile | None,
) -> dict[str, object]:
    if active_profile is None:
        return legacy_defaults
    merged = dict(legacy_defaults)
    merged.update(active_profile.as_config())
    return merged


def _run_secret_command(args: argparse.Namespace) -> int:
    vault = SecretsVault()
    if args.secret_command == "set":
        value = getpass.getpass("Secret value: ")
        vault.set(str(args.name), value)
        print(f"secret set: {args.name}")
        return 0
    if args.secret_command == "list":
        for name in vault.list_names():
            print(name)
        return 0
    if args.secret_command == "delete":
        deleted = vault.delete(str(args.name))
        status = "deleted" if deleted else "not found"
        print(f"secret {status}: {args.name}")
        return 0
    return 2


def _run_resume_command(args: argparse.Namespace, identity: StartupIdentity) -> int:
    """分派 resume inspect/execute 并映射公开退出码

    参数：args 为已完成 argparse 校验的参数；identity 为共享启动身份
    返回：inspect/done=0、service/failed=1、paused=3
    """
    try:
        if not args.execute:
            response = inspect_resume(
                args.checkpoint_id,
                identity.project_root,
                data_root=identity.data_root,
            )
        else:
            checkpoint_id, project_root = resolve_resume_identity(
                args.checkpoint_id, identity.data_root
            )
            llm_client = build_llm_client(
                _build_cli_overrides(args), project_root=project_root
            )
            response = execute_resume(
                checkpoint_id,
                project_root,
                data_root=identity.data_root,
                decision=args.decision,
                llm_client=llm_client,
            )
            exit_code = _resume_exit_code(response.status)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"RESUME_ERROR: {exc}", file=sys.stderr)
        return EXIT_SERVICE_ERROR
    if not args.execute:
        print(response.output)
        return EXIT_OK
    # 【CLI Resume】【失败输出】运行失败保留真实身份，但不输出成功形态的 TASK 头
    if response.status == "failed":
        print(
            f"RESUME_ERROR: task_id={response.task_id} "
            f"run_id={response.run_id} segment_id={response.segment_id} "
            f"status={response.status}\n{response.output}",
            file=sys.stderr,
        )
        return exit_code
    _print_response(response)
    return exit_code


def _resume_exit_code(status: str) -> int:
    """把真实 resume 终态转换为 CLI 退出码

    参数：status 为 execute_resume 返回的 done/paused/failed
    返回：done=0、failed=1、paused=3；未知状态显式失败
    """
    if status == "done":
        return EXIT_OK
    if status == "failed":
        return EXIT_SERVICE_ERROR
    if status == "paused":
        return EXIT_PAUSED
    raise ValueError(f"unsupported resume status: {status}")


def _print_response(response: RunTaskResponse) -> None:
    print(
        f"TASK task_id={response.task_id} "
        f"run_id={response.run_id} "
        f"segment_id={response.segment_id} "
        f"status={response.status}"
    )
    print(response.output)


if __name__ == "__main__":
    raise SystemExit(main())
