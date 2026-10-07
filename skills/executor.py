from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import path_security
from runtime.lease import Lease
from runtime.cancellation import CancellationToken
from skills.store import SKILL_STATE_ACTIVE, Skill, SkillStore
from tools.exec_channel import ExecChannel, ExecSpec, LocalSubprocessBackend
from tools.types import ToolError, ToolErrorCategory

_CHANNEL = ExecChannel(LocalSubprocessBackend())
_WRAPPER_MODULE = "_reins_skill_script"
# skill 脚本子进程默认超时秒数，与 execute_skill_script 签名默认值同源
_DEFAULT_TIMEOUT_SECONDS = 30.0


def executor(args: dict[str, object]) -> dict[str, object] | ToolError:
    """
    skill_run 工具的薄 executor：把 registry 注入的环境值取出后透传给
    execute_skill_script 本体，不预判 skill 业务闸门（state/capabilities/path）。

    2026-07-13 22:00:00 作者 xxx

    :param args: registry 组装的参数字典，含模型侧 skill_id/args/script_name 与
        registry 注入的 __lease__/__data_root__/__timeout_seconds__
    :return: execute_skill_script 的原始 dict 结果或真实 ToolError
    """
    # 1. 环境前置 fail-closed：lease 缺失属环境注入问题，本体闸门够不着，先拦
    lease = args.get("__lease__")
    if not isinstance(lease, Lease):
        return ToolError(
            ToolErrorCategory.PERMISSION, "skill_run_no_lease", retryable=False
        )
    # 2. 环境前置 fail-closed：data_root 非 str/Path 会让 SkillStore 抛晦涩 TypeError
    data_root = args.get("__data_root__")
    if not isinstance(data_root, str | Path):
        return ToolError(
            ToolErrorCategory.INVALID_INPUT, "skill_run_no_data_root", retryable=False
        )
    # 3. args 缺失才补 {}（本体要 dict）；存在但非 dict 时原样透传，
    #    让脚本 json 序列化后诚实失败，不静默改写模型入参
    raw_args = args.get("args", {})
    return execute_skill_script(
        data_root,
        skill_id=str(args.get("skill_id", "")).strip(),
        version=cast(str | None, args.get("version")),
        args=cast("dict[str, object]", raw_args),
        lease=lease,
        script_name=str(args.get("script_name", "")).strip(),
        timeout_seconds=_timeout_arg(args.get("__timeout_seconds__")),
        cancellation=cast(CancellationToken | None, args.get("__cancellation__")),
    )


def _timeout_arg(value: object) -> float:
    """
    把 registry 注入的 __timeout_seconds__ 归一成 float 超时秒数。

    2026-07-13 22:00:00 作者 xxx

    :param value: registry 注入值，可能是 int/float、数字字符串或缺失
    :return: 归一后的超时秒数；无法解析时回退默认值
    """
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return _DEFAULT_TIMEOUT_SECONDS


def execute_skill_script(
    data_root: Path | str,
    *,
    skill_id: str,
    version: str | None = None,
    args: dict[str, object],
    lease: Lease,
    script_name: str = "",
    timeout_seconds: float = 30.0,
    channel: ExecChannel | None = None,
    cancellation: CancellationToken | None = None,
) -> dict[str, object] | ToolError:
    store = SkillStore(data_root)
    skill = store.load_skill(skill_id, version=version)
    permission = _execution_permission(skill, lease)
    if permission is not None:
        return permission
    script_path = _resolve_script(skill, script_name)
    if script_path is None:
        return ToolError(ToolErrorCategory.INVALID_INPUT, "skill_script_not_found")
    result = (channel or _CHANNEL).run(
        ExecSpec(
            code=_wrapper_code(script_path, args),
            kind="py",
            cwd=skill.root,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )
    )
    if result.timed_out or result.cancelled:
        store.update_skill_stats(skill_id, success=False, version=skill.version)
        return ToolError(
            ToolErrorCategory.TIMEOUT
            if result.timed_out
            else ToolErrorCategory.CANCELLED,
            "skill_script_interrupted",
            retryable=False,
            partial_state="previous side effects are not rolled back",
            details={
                "execution_state": result.execution_state,
                "pid": result.pid,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "stop_error": result.stop_error,
            },
        )
    success = result.exit_code == 0
    store.update_skill_stats(skill_id, success=success, version=skill.version)
    if not success:
        message = (
            result.stderr.strip() or result.stdout.strip() or "skill_script_failed"
        )
        return ToolError(ToolErrorCategory.UNKNOWN, message, retryable=False)
    return {
        "skill_id": skill_id,
        "version": skill.version,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "script_status": "exited_zero",
        "task_effect": "unknown",
    }


def _execution_permission(skill: Skill, lease: Lease) -> ToolError | None:
    if skill.frontmatter.state != SKILL_STATE_ACTIVE:
        return ToolError(
            ToolErrorCategory.PERMISSION, "skill_archived", retryable=False
        )
    missing = _missing_capabilities(skill.frontmatter.required_capabilities, lease)
    if missing:
        return ToolError(
            ToolErrorCategory.PERMISSION,
            "skill_missing_capabilities: " + ", ".join(missing),
            retryable=False,
        )
    if not _capability_enabled("code_execution", lease):
        return ToolError(
            ToolErrorCategory.PERMISSION, "skill_execution_disabled", retryable=False
        )
    if (
        path_security.check_write(skill.root, lease)
        is not path_security.Decision.ALLOWED
    ):
        return ToolError(
            ToolErrorCategory.PERMISSION, "skill_path_denied", retryable=False
        )
    return None


def _missing_capabilities(required: list[str], lease: Lease) -> list[str]:
    return [name for name in required if not _capability_enabled(name, lease)]


def _capability_enabled(name: str, lease: Lease) -> bool:
    capability = lease.capabilities.get(name)
    if capability is None:
        return False
    if isinstance(capability, Mapping):
        return capability.get("enabled") is not False
    if isinstance(capability, bool):
        return capability
    return True


def _resolve_script(skill: Skill, script_name: str) -> Path | None:
    if not script_name.strip():
        return skill.script_path
    relative = Path(script_name.strip())
    if relative.parts[:1] != ("scripts",):
        relative = Path("scripts") / relative
    path = (skill.root / relative).resolve()
    scripts_root = (skill.root / "scripts").resolve()
    try:
        path.relative_to(scripts_root)
    except ValueError:
        return None
    if path.relative_to(skill.root).as_posix() not in skill.resources:
        return None
    return path if path.is_file() else None


def _wrapper_code(script_path: Path, args: dict[str, object]) -> str:
    payload = json.dumps(args, ensure_ascii=False)
    return "\n".join(
        [
            "import importlib.util, json",
            f"script_path = {str(script_path)!r}",
            f"args = json.loads({payload!r})",
            f"spec = importlib.util.spec_from_file_location({_WRAPPER_MODULE!r}, script_path)",
            "if spec is None or spec.loader is None:",
            "    raise RuntimeError('skill_import_failed')",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "main = getattr(module, 'main', None)",
            "if not callable(main):",
            "    raise RuntimeError('skill_main_missing')",
            "print(json.dumps({'result': main(args)}, ensure_ascii=False, default=str))",
        ]
    )


__all__ = ["execute_skill_script"]
