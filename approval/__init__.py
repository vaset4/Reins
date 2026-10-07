from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import cast
import hashlib
import json

import yaml

from runtime.lease import Lease
from runtime.cancellation import CancellationToken
from tasks.ids import utc_now
from tasks.store import TaskStore


class ApprovalDecision(str, Enum):
    ONCE = "once"
    SESSION = "session"
    TASK = "task"
    PERMANENT = "permanent"
    DENY = "deny"
    CANCELLED = "cancelled"


class ApprovalUnavailable(RuntimeError):
    """审批设施未能给出决定，不能被解释成用户拒绝或同意。"""


@dataclass(frozen=True, slots=True)
class ApprovalResource:
    """界面明确展示的动作与精确资源范围，不表示整个父目录的授权。"""

    action: str
    kind: str
    target: str


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    tool: str
    args: Mapping[str, object]
    risk: str
    lease: Lease
    data_root: Path
    message: str
    ts: str = field(default_factory=utc_now)
    resource: ApprovalResource | None = None
    cancellation: CancellationToken | None = field(
        default=None, compare=False, repr=False
    )
    superseded: Callable[[], bool] | None = field(
        default=None, compare=False, repr=False
    )
    session_id: str = ""
    run_id: str = ""
    owner_session_id: str = ""
    operation_id: str = ""
    batch_id: str = ""
    definition_version: str = ""
    intent_digest: str = ""
    readonly: bool = False
    force_confirmation: bool = False
    resource_identity: str = ""


ApprovalBackend = Callable[[ApprovalRequest], ApprovalDecision]
_backend: ApprovalBackend | None = None


def register_approval_backend(fn: ApprovalBackend | None) -> None:
    global _backend
    _backend = fn


def request_approval(req: ApprovalRequest) -> ApprovalDecision:
    """复用明确范围或请求新决定；传参：审批请求；返回：用户决定，设施失败抛独立错误。"""
    if not _is_usable_data_root(req.data_root):
        raise ApprovalUnavailable("approval data root is not a directory")
    if _approval_interrupted(req):
        return ApprovalDecision.CANCELLED
    if not req.force_confirmation and (
        _has_task_grant(req) or _has_permanent_grant(req)
    ):
        return ApprovalDecision.ONCE
    try:
        backend: ApprovalBackend
        if _backend is None:
            from approval.tui import tui_request_approval

            backend = tui_request_approval
        else:
            backend = _backend
        decision = backend(req)
    except (KeyboardInterrupt, EOFError):
        return ApprovalDecision.CANCELLED
    except Exception as exc:
        raise ApprovalUnavailable(f"approval backend failed: {exc}") from exc
    if not isinstance(decision, ApprovalDecision):
        raise ApprovalUnavailable("approval backend returned an invalid decision")
    if _approval_interrupted(req):
        return ApprovalDecision.CANCELLED
    if req.force_confirmation:
        if decision not in {
            ApprovalDecision.ONCE,
            ApprovalDecision.DENY,
            ApprovalDecision.CANCELLED,
        }:
            raise ApprovalUnavailable(
                "protected replacement only accepts a decision for this operation"
            )
        return decision
    _persist_decision(req, decision)
    return decision


def _approval_interrupted(req: ApprovalRequest) -> bool:
    """提交授权前核对停止与运行中新要求；传参：原审批；返回：是否已失效。"""
    return (req.cancellation is not None and req.cancellation.cancelled) or (
        req.superseded is not None and req.superseded()
    )


def _persist_decision(req: ApprovalRequest, decision: ApprovalDecision) -> None:
    if decision is ApprovalDecision.SESSION:
        raise ApprovalUnavailable("session approval requires an active session host")
    if decision in (ApprovalDecision.ONCE, ApprovalDecision.TASK):
        _append_goal_grant(req, decision.value)
    elif decision is ApprovalDecision.PERMANENT:
        _append_permanent_grant(req)


def _has_task_grant(req: ApprovalRequest) -> bool:
    return any(_grant_matches(req, grant, "task") for grant in _task_grants(req))


def _has_permanent_grant(req: ApprovalRequest) -> bool:
    config = _read_yaml(_config_path())
    return any(
        _grant_matches(req, grant, "permanent")
        for grant in _grant_list(config.get("permanent_grants"))
    )


def _task_grants(req: ApprovalRequest) -> list[Mapping[str, object]]:
    grants: list[Mapping[str, object]] = []
    task_caps = req.lease.capabilities.get("task")
    if isinstance(task_caps, Mapping):
        grants.extend(_grant_list(task_caps.get("grants")))
    task = TaskStore(req.data_root).load_task(req.lease.task_id)
    if task is not None:
        grants.extend(_grant_list(task.grants))
    return grants


def _append_goal_grant(req: ApprovalRequest, scope: str) -> None:
    """通过任务写者保存授权；传参：请求和授权范围；返回：无，保存失败直接暴露。"""
    store = TaskStore(req.data_root)
    try:
        store.append_grant(req.lease.task_id, _grant_entry(req, scope))
    finally:
        store.close()


def _append_permanent_grant(req: ApprovalRequest) -> None:
    path = _config_path()
    data = _read_yaml(path)
    grants = [dict(item) for item in _grant_list(data.get("permanent_grants"))]
    grants.append(_grant_entry(req, "permanent"))
    data["permanent_grants"] = grants
    _write_yaml(path, data)


def _grant_entry(req: ApprovalRequest, scope: str) -> dict[str, object]:
    """保存用户看到的授权范围；传参：请求与复用范围；返回：可持久授权记录。"""
    entry: dict[str, object] = {
        "tool": req.tool,
        "args": dict(req.args),
        "scope": scope,
        "ts": utc_now(),
    }
    if req.resource is not None and scope != "once":
        entry["resource"] = asdict(req.resource)
    return entry


def _grant_matches(
    req: ApprovalRequest, grant: Mapping[str, object], scope: str
) -> bool:
    """新授权按动作/资源复用，旧授权保留全参数语义；传参：请求、授权及范围；返回：是否匹配。"""
    if grant.get("tool") != req.tool or grant.get("scope") != scope:
        return False
    if (
        grant.get("definition_version") is not None
        and grant["definition_version"] != req.definition_version
    ):
        return False
    resource = grant.get("resource")
    if resource is not None:
        if isinstance(resource, Mapping) and resource.get("kind") == "directory":
            if (
                req.resource is None
                or req.resource.kind != "file"
                or resource.get("action") != req.resource.action
            ):
                return False
            return Path(req.resource.target).is_relative_to(
                Path(str(resource["target"]))
            )
        return req.resource is not None and resource == asdict(req.resource)
    if grant.get("args_digest") is not None:
        return grant["args_digest"] == req.intent_digest
    return grant.get("args") == dict(req.args)


def persist_grant(req: ApprovalRequest, grant: dict[str, object]) -> None:
    """将已落事件的授权写入原有权限载体；传参：申请及授权；返回：无，失败明确上抛。"""
    from contextlib import closing
    from tools.file_persistence import file_edit_lock

    if grant["scope"] == "task":
        with closing(TaskStore(req.data_root)) as store:
            store.append_grant(req.lease.task_id, grant)
    elif grant["scope"] == "permanent":
        path = _config_path()
        with file_edit_lock(path, wait=True):
            data = _read_yaml(path)
            grants = [dict(item) for item in _grant_list(data.get("permanent_grants"))]
            existing = next(
                (item for item in grants if item.get("grant_id") == grant["grant_id"]),
                None,
            )
            if existing is not None:
                if existing != grant:
                    raise ApprovalUnavailable(
                        "persistent grant identity has different content"
                    )
                return
            _write_yaml(path, {**data, "permanent_grants": [*grants, grant]})


def _grant_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def grant_identity(grant: Mapping[str, object]) -> str:
    """给旧权限确定可撤销身份，仍保持原精确匹配语义；传参：权限载体；返回：记录ID或旧记录摘要。"""
    return str(
        grant.get("grant_id")
        or "legacy-"
        + hashlib.sha256(
            json.dumps(dict(grant), sort_keys=True).encode("utf-8")
        ).hexdigest()
    )


def _read_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cast(dict[str, object], data) if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: Mapping[str, object]) -> None:
    """原子发布配置授权，不留半份权限文件；传参：路径与完整配置；返回：无。"""
    from schedules.persistence import atomic_text

    atomic_text(path, yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True))


def _is_usable_data_root(data_root: Path) -> bool:
    """判断授权请求携带的数据根是否具备目录语义

    参数：data_root 为授权 task/once grant 的数据命名空间
    返回：路径可作为目录使用时返回 True
    """
    if not isinstance(data_root, Path):
        return False
    try:
        resolved = data_root.expanduser().resolve()
        return not resolved.exists() or resolved.is_dir()
    except OSError:
        return False


def _config_path() -> Path:
    return Path.home() / ".reins" / "config.yaml"
