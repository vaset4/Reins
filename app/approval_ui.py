"""用户界面的会话模式与明确授权撤销。

作者：xxx
时间：2026-09-24 12:00:00
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from uuid import uuid4

import approval
from approval.session import ApprovalMode, ApprovalSession
from runtime.ledger import LedgerStore, new_ledger_event
from runtime.ledger_writer import LedgerWriter
from runtime.session_message_store import SessionMessageStore
from tasks.store import TaskStore
from tools.file_persistence import file_edit_lock

MODE_LABELS = {
    ApprovalMode.READ_ONLY: "只读",
    ApprovalMode.WORKSPACE: "工作区内写自动放行",
    ApprovalMode.AUTO: "自动放行",
}


def approval_control(
    command: str,
    session: ApprovalSession,
    *,
    data_root: Path,
    session_id: str,
    task_id: str | None,
    action_id: str | None = None,
) -> str:
    """执行用户显式模式或撤销命令；传参：命令、宿主权限与会话身份；返回：实际结果说明。"""
    fields = command.split()
    if fields[0] == "/mode":
        if len(fields) > 2:
            raise ValueError("用法：/mode [read_only|workspace|auto]")
        if len(fields) == 2:
            session.set_mode(ApprovalMode(fields[1]))
        grants = _grants(session, data_root, task_id)
        lines = [
            f"当前模式：{MODE_LABELS[session.mode]}；密钥替换和定时控制仍需明确授权。"
        ]
        lines.extend(
            f"{_identity(grant)} · {grant.get('scope', 'permanent')} · {grant.get('tool', '旧命令授权')} · {grant.get('resource') or '精确参数'}"
            for grant in grants
        )
        lines.append(
            "/mode read_only|workspace|auto 切换模式；/revoke 授权编号 撤回某项授权"
        )
        return "\n".join(lines)
    if fields[0] != "/revoke" or len(fields) != 2 or not session_id:
        raise ValueError("用法：/revoke 授权编号（先用 /mode 查看）")
    identity = fields[1]
    candidates = _grants(session, data_root, task_id)
    grant = next((item for item in candidates if _identity(item) == identity), None)
    ledger = LedgerStore(data_root)
    original = next(
        (
            event
            for event in ledger.read_events()
            if event.event_id == identity and event.event == "approval.decided"
        ),
        None,
    )
    if original is not None:
        scope = original.payload.get("scope")
        owned = (
            scope == "permanent"
            or (scope == "task" and task_id is not None and original.task_id == task_id)
            or (
                scope in {"session", "once"}
                and original.session_id == session_id
                and original.payload.get("session_instance") == session.instance_id
            )
        )
        if original.source != "user_action" or not owned:
            original = None
    if grant is None and original is None:
        raise ValueError("未找到当前会话或目标可管理的授权")
    # 【审批】【撤销优先】先停止实际权限复用，再记录来源和审计；审计失败不能恢复允许
    session.revoke(identity)
    if grant is not None:
        _remove_grant(grant, data_root=data_root, task_id=task_id)
    chosen_action = action_id or uuid4().hex
    source = SessionMessageStore(data_root).accept_input(
        session_id,
        f"撤销授权：{identity}",
        input_id=f"approval-input-{chosen_action}",
        task_id=task_id,
        input_kind="approval",
    )
    LedgerWriter(ledger).record_once(
        new_ledger_event(
            "approval.revoked",
            f"approval-revoked-{chosen_action}",
            "user_action",
            {
                "action_id": chosen_action,
                "grant_id": identity,
                "source_input_id": source.entry_id,
            },
            session_id=session_id,
            task_id=task_id,
        )
    )
    return "授权已撤回，后续操作将重新核对；已经发生的操作不会被撤销。"


def _grants(
    session: ApprovalSession, root: Path, task_id: str | None
) -> tuple[dict[str, object], ...]:
    """列出当前可管理的真实权限载体；传参：宿主、数据根和目标；返回：不包含单次权限的快照。"""
    grants = [
        dict(item)
        for item in approval._grant_list(
            approval._read_yaml(approval._config_path()).get("permanent_grants")
        )
    ]
    if task_id:
        with closing(TaskStore(root)) as store:
            record = store.load_task(task_id)
            if record is not None:
                grants.extend(
                    dict(item) for item in record.grants if item.get("scope") == "task"
                )
    grants.extend(session.grants())
    return tuple(grant for grant in grants if not session.is_revoked(_identity(grant)))


def _identity(grant: dict[str, object]) -> str:
    """给旧授权提供可明确选择的身份，不扩大其范围；传参：原授权；返回：原ID或稳定旧记录摘要。"""
    return approval.grant_identity(grant)


def _remove_grant(
    grant: dict[str, object], *, data_root: Path, task_id: str | None
) -> None:
    """通过原写者撤回准确匹配的持久权限；传参：原权限、根和目标；返回：无。"""
    if grant.get("scope", "permanent") == "task":
        if task_id is None:
            raise ValueError("task authorization requires its original goal")
        with closing(TaskStore(data_root)) as store:
            store.revoke_grant(task_id, grant)
    elif grant.get("scope", "permanent") == "permanent":
        path = approval._config_path()
        with file_edit_lock(path, wait=True):
            data = approval._read_yaml(path)
            grants = [
                dict(item)
                for item in approval._grant_list(data.get("permanent_grants"))
                if item != grant
            ]
            approval._write_yaml(path, {**data, "permanent_grants": grants})
