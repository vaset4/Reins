"""先完成整批授权，再把固定意图交给执行器。

作者：xxx
时间：2026-09-24 12:00:00
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import approval
from approval import ApprovalDecision, ApprovalRequest, ApprovalUnavailable
from approval.batch_types import (
    ApprovalBatch,
    ApprovalChoice,
    BatchDecision,
    validate_batch_decision,
)
from approval.session import ApprovalMode, ApprovalSession
from llm.messages import JsonValue, thaw_json_value
from runtime.ledger import LedgerStore, new_ledger_event
from runtime.ledger_writer import LedgerWriter
from runtime.session_message_store import SessionMessageStore

BatchBackend = Callable[[ApprovalBatch], BatchDecision]
_batch_backend: BatchBackend | None = None


def register_batch_backend(backend: BatchBackend | None) -> None:
    """绑定宿主已有输入通道；传参：批次交互入口或释放；返回：无。"""
    global _batch_backend
    _batch_backend = backend


def request_batch_decision(batch: ApprovalBatch) -> BatchDecision:
    """通过宿主一次收取整批选择；传参：固定展示集合；返回：完整决定或取消。"""
    try:
        if _batch_backend is not None:
            result = _batch_backend(batch)
        elif len(batch.requests) == 1 and approval._backend is not None:
            # 【审批】【单项入口】既有单项界面只处理唯一项目，不把一个决定扩散到整批
            request = batch.requests[0]
            choice = approval._backend(request)
            result = (
                BatchDecision(cancelled=True)
                if choice is ApprovalDecision.CANCELLED
                else BatchDecision((ApprovalChoice(request.operation_id, choice),))
            )
        else:
            from approval.cli import cli_request_batch

            result = cli_request_batch(batch)
        validate_batch_decision(batch, result)
        return result
    except (KeyboardInterrupt, EOFError):
        return BatchDecision(cancelled=True)
    except ApprovalUnavailable:
        raise
    except Exception as exc:
        raise ApprovalUnavailable(f"approval batch interaction failed: {exc}") from exc


def intent_digest(request: ApprovalRequest) -> str:
    """固定实际参数、资源与工具合同；传参：规范化申请；返回：不含正文的意图哈希。"""
    value = {
        "tool": request.tool,
        "args": thaw_json_value(cast(JsonValue, request.args)),
        "contract": request.definition_version,
        "resource": asdict(request.resource) if request.resource else None,
        "force_confirmation": request.force_confirmation,
        "resource_identity": request.resource_identity,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Authorization:
    """一次操作的已提交授权凭据；拒绝与取消也保留真实决定。"""

    request: ApprovalRequest
    decision: ApprovalDecision
    event_id: str
    grant: Mapping[str, object]
    source: str

    def evidence(self) -> dict[str, object]:
        """提供可写入操作记录的授权证据；传参：无；返回：不含候选正文的事实。"""
        return {
            "event_id": self.event_id,
            "batch_id": self.request.batch_id,
            "operation_id": self.request.operation_id,
            "intent_digest": self.request.intent_digest,
            "definition_version": self.request.definition_version,
            "decision": self.decision.value,
            "grant_id": self.grant.get("grant_id"),
            "scope": self.grant.get("scope"),
            "source": self.source,
        }


class BatchAuthorizer:
    """整批决定及派发前复核；持久权限继续由任务和配置写者保存。"""

    def __init__(
        self,
        session: ApprovalSession,
        messages: SessionMessageStore,
        ledger: LedgerStore,
        *,
        prompt: BatchBackend = request_batch_decision,
    ) -> None:
        """注入实际宿主权限与事实写者；传参：会话、消息、事件及交互；返回：无。"""
        self.session, self.messages, self.ledger, self.prompt = (
            session,
            messages,
            ledger,
            prompt,
        )
        self._decisions: dict[str, BatchDecision] = {}

    def authorize(
        self, requests: tuple[ApprovalRequest, ...]
    ) -> dict[str, Authorization]:
        """整批展示结束后提交逐项证据，任何交互失败都不派发；传参：全部意图；返回：逐项凭据。"""
        if not requests:
            return {}
        identities = {
            (request.batch_id, request.session_id, request.run_id)
            for request in requests
        }
        if len(identities) != 1 or len(
            {request.operation_id for request in requests}
        ) != len(requests):
            raise ApprovalUnavailable(
                "approval batch must contain unique operations from the same run"
            )
        covered: dict[str, tuple[str, dict[str, object]]] = {}
        pending = []
        for request in requests:
            self._record(
                request,
                "approval.requested",
                f"approval-request-{request.operation_id}",
                "runtime",
                {},
            )
            coverage = self._coverage(request)
            if coverage is None:
                pending.append(request)
            else:
                covered[request.operation_id] = coverage
        batch = ApprovalBatch(requests[0].batch_id, tuple(pending))
        decision = self._decisions.get(batch.batch_id)
        if pending and decision is None:
            decision = self.prompt(batch)
            validate_batch_decision(batch, decision)
            self._decisions[batch.batch_id] = decision
        if any(approval._approval_interrupted(request) for request in requests):
            decision = BatchDecision(cancelled=True)
        if decision is not None and decision.cancelled:
            return self._cancelled(requests)
        source_input = (
            self._accept_choice(batch, decision)
            if pending and decision is not None
            else None
        )
        if pending and source_input is None:
            return self._cancelled(requests)
        choices = (
            {choice.operation_id: choice for choice in decision.choices}
            if decision is not None
            else {}
        )
        # 【审批】【整批提交】所有回执先完整验证，再写事件和权限；本方法返回前执行器不能开始
        result = {}
        for request in requests:
            if request.operation_id in covered:
                source, grant = covered[request.operation_id]
                choice = ApprovalChoice(
                    request.operation_id,
                    ApprovalDecision.DENY
                    if source == "deny"
                    else ApprovalDecision.ONCE,
                )
                action_id = f"policy-{request.operation_id}"
            else:
                source, grant = "user_action", {}
                choice = choices[request.operation_id]
                assert decision is not None
                action_id = decision.action_id
            result[request.operation_id] = self._commit(
                request,
                choice,
                action_id=action_id,
                source=source,
                inherited=grant,
                source_input=source_input,
            )
        return result

    def _coverage(
        self, request: ApprovalRequest
    ) -> tuple[str, dict[str, object]] | None:
        """核对当前策略及真正载体，先拒绝再按长期到临时范围查询；传参：申请；返回：来源或需询问。"""
        if request.risk == "deny":
            return "deny", {}
        if (
            request.lease.trigger != "cron"
            and self.session.mode is ApprovalMode.READ_ONLY
            and not request.readonly
        ):
            return "deny", {}
        if request.force_confirmation:
            return ("deny", {}) if request.lease.trigger == "cron" else None
        if request.risk == "safe":
            return "policy", {}
        grant = self._find_grant(request)
        if grant is not None:
            return "grant" if grant.get("decision_event_id") else "legacy", grant
        if request.lease.trigger == "cron":
            return "deny", {}
        if self.session.mode is ApprovalMode.AUTO and request.tool != "schedule":
            return "policy", {}
        return None

    def _find_grant(self, request: ApprovalRequest) -> dict[str, object] | None:
        """读取实际权限载体，缺失审计引用视为错误；传参：申请；返回：匹配授权或无。"""
        permanent = approval._grant_list(
            approval._read_yaml(approval._config_path()).get("permanent_grants")
        )
        if request.lease.trigger == "cron":
            schedule = request.lease.capabilities.get("schedule")
            if isinstance(schedule, Mapping):
                permanent.extend(
                    approval._grant_list(schedule.get("required_permanent_grants"))
                )
        groups = [("permanent", permanent)]
        if request.lease.trigger != "cron":
            groups.extend(
                [
                    ("task", approval._task_grants(request)),
                    ("session", list(self.session.grants())),
                ]
            )
        candidates = [
            (scope, grant)
            for scope, grants in groups
            for grant in grants
            if not self.session.is_revoked(approval.grant_identity(grant))
            and _grant_matches(request, grant, scope)
        ]
        if not candidates:
            return None
        # 1. 【审批】【权限原件】只核对匹配候选的撤销与批准来源，二者固定在同一提交前缀
        with self.ledger.database.snapshot():
            revoked = {
                row.payload.get("grant_id")
                for row in self.ledger.read_events_of_type("approval.revoked")
            }
            for scope, grant in candidates:
                identity = approval.grant_identity(grant)
                if identity in revoked or self.session.is_revoked(identity):
                    continue
                reference = grant.get("decision_event_id")
                if reference:
                    event = (
                        self.ledger.read_event(reference)
                        if isinstance(reference, str)
                        else None
                    )
                    if (
                        event is None
                        or event.event != "approval.decided"
                        or event.source != "user_action"
                    ):
                        raise ApprovalUnavailable(
                            "approval grant references a missing or invalid decision"
                        )
                    expected = {
                        "grant_id": grant.get("grant_id"),
                        "scope": grant.get("scope"),
                        "tool": grant.get("tool"),
                        "granted_resource": grant.get("resource"),
                        "definition_version": grant.get("definition_version"),
                        "intent_digest": grant.get("args_digest"),
                    }
                    if any(
                        event.payload.get(key) != value
                        for key, value in expected.items()
                    ):
                        raise ApprovalUnavailable(
                            "approval grant no longer matches its recorded decision"
                        )
                return {"scope": scope, **grant, "grant_id": identity}
        return None

    def _accept_choice(
        self, batch: ApprovalBatch, decision: BatchDecision
    ) -> str | None:
        """保存用户实际选择的唯一正文，不把它误当本批新要求；传参：展示批次和回执；返回：来源身份。"""
        first = batch.requests[0]
        by_operation = {choice.operation_id: choice for choice in decision.choices}
        text = f"工具授权（{batch.batch_id}）：" + "；".join(
            f"{index}={by_operation[request.operation_id].decision.value}:{by_operation[request.operation_id].resource_scope}"
            for index, request in enumerate(batch.requests, 1)
        )
        identity = f"approval-input-{decision.action_id}"
        sessions = {
            session
            for request in batch.requests
            for session in (request.session_id, request.owner_session_id)
            if session
        }
        inputs = self._input_identities(sessions)
        # 1. 【审批】【协作核对】回调可能传播协作消息并取宿主锁，不能持有事实写锁等待它
        if any(approval._approval_interrupted(request) for request in batch.requests):
            return None
        with self.messages.write_lock(first.session_id):
            # 2. 【审批】【原子接纳】锁内直接重核本会话和父会话的新要求，取消或输入变化都使旧决定失效
            cancelled = any(
                request.cancellation is not None and request.cancellation.cancelled
                for request in batch.requests
            )
            if cancelled or self._input_identities(sessions) != inputs:
                return None
            self.messages.accept_input(
                first.session_id,
                text,
                input_id=identity,
                run_id=first.run_id,
                task_id=first.lease.task_id,
                input_source="user",
                input_kind="approval",
            )
        return identity

    def _input_identities(self, sessions: set[str]) -> dict[str, tuple[str, ...]]:
        """在同一原件快照读取语义输入，审批正文不算新要求；参数：实际会话集合；返回：按接纳顺序的输入身份。"""
        with self.messages.database.snapshot():
            return {
                session: tuple(
                    entry.entry_id
                    for entry in self.messages.pending_inputs(
                        session, include_delivered=True
                    )
                )
                for session in sessions
            }

    def _commit(
        self,
        request: ApprovalRequest,
        choice: ApprovalChoice,
        *,
        action_id: str,
        source: str,
        inherited: dict[str, object],
        source_input: str | None,
    ) -> Authorization:
        """事件先于实际授权载体，失败不能派发；传参：意图、选择及来源；返回：完整凭据。"""
        identity = f"approval-decision-{action_id}-{request.operation_id}"
        grant = inherited or self._grant(request, choice, identity)
        self._record(
            request,
            "approval.decided",
            identity,
            source,
            {
                "action_id": action_id,
                "decision": choice.decision.value,
                "scope": grant.get("scope"),
                "resource_scope": choice.resource_scope,
                "grant_id": grant.get("grant_id"),
                "granted_resource": grant.get("resource"),
                "session_instance": self.session.instance_id,
                "source_input_id": source_input if source == "user_action" else None,
                "inherited_event_id": inherited.get("decision_event_id"),
            },
        )
        if source == "user_action" and choice.decision not in {
            ApprovalDecision.DENY,
            ApprovalDecision.ONCE,
        }:
            approval.persist_grant(request, grant)
            if choice.decision is ApprovalDecision.SESSION:
                self.session.remember(grant)
        return Authorization(request, choice.decision, identity, grant, source)

    def _grant(
        self, request: ApprovalRequest, choice: ApprovalChoice, event_id: str
    ) -> dict[str, object]:
        """固定用户看到的资源范围，不能隐式扩大父目录；传参：意图、选择与事件；返回：权限载体。"""
        resource = asdict(request.resource) if request.resource is not None else None
        if request.force_confirmation and choice.decision not in {
            ApprovalDecision.ONCE,
            ApprovalDecision.DENY,
        }:
            raise ApprovalUnavailable(
                "this protected action requires confirmation for this operation only"
            )
        if choice.resource_scope == "directory":
            if (
                resource is None
                or resource["kind"] != "file"
                or choice.decision is ApprovalDecision.ONCE
            ):
                raise ApprovalUnavailable(
                    "directory scope requires a displayed file and a reusable grant"
                )
            resource = {
                **resource,
                "kind": "directory",
                "target": str(Path(resource["target"]).parent),
            }
        return {
            "grant_id": event_id,
            "decision_event_id": event_id,
            "tool": request.tool,
            "scope": choice.decision.value,
            "resource": resource,
            "args_digest": request.intent_digest,
            "definition_version": request.definition_version,
            "operation_id": request.operation_id,
            "session_instance": self.session.instance_id,
        }

    def _record(
        self,
        request: ApprovalRequest,
        event: str,
        identity: str,
        source: str,
        payload: Mapping[str, object],
    ) -> None:
        """追加无候选正文的类型化审批证据；传参：意图和事件字段；返回：无。"""
        fields = {
            "batch_id": request.batch_id,
            "operation_id": request.operation_id,
            "tool": request.tool,
            "intent_digest": request.intent_digest,
            "definition_version": request.definition_version,
            "resource": asdict(request.resource) if request.resource else None,
            **payload,
        }
        LedgerWriter(self.ledger).record_once(
            new_ledger_event(
                event,
                identity,
                source,
                fields,
                task_id=request.lease.task_id,
                session_id=request.session_id,
                run_id=request.run_id,
            )
        )

    def _cancelled(
        self, requests: tuple[ApprovalRequest, ...]
    ) -> dict[str, Authorization]:
        """整批取消保留未取得决定的事实；传参：全部意图；返回：全部取消凭据。"""
        return {
            request.operation_id: Authorization(
                request, ApprovalDecision.CANCELLED, "", {}, "cancelled"
            )
            for request in requests
        }

    def validate(self, request: ApprovalRequest, authorization: Authorization) -> None:
        """实际派发前核对意图、模式、撤销和权限载体；传参：当前申请与原批准；返回：无，失效抛错。"""
        actual = (
            request.batch_id,
            request.operation_id,
            request.session_id,
            request.run_id,
            request.lease.task_id,
        )
        original = authorization.request
        expected = (
            original.batch_id,
            original.operation_id,
            original.session_id,
            original.run_id,
            original.lease.task_id,
        )
        if actual != expected or intent_digest(request) != original.intent_digest:
            raise ValueError("approved operation intent changed")
        if request.risk == "deny":
            raise ValueError("current boundary denies this operation")
        if request.lease.trigger == "cron" and (
            request.force_confirmation
            or (request.risk == "confirm" and self._find_grant(request) is None)
        ):
            raise ValueError(
                "cron operation requires its original permanent authorization"
            )
        if approval._approval_interrupted(request):
            raise ValueError("approved operation was cancelled or superseded")
        if self.session.is_revoked(str(authorization.grant.get("grant_id", ""))):
            raise ValueError("approval was revoked before dispatch")
        if (
            request.lease.trigger != "cron"
            and self.session.mode is ApprovalMode.READ_ONLY
            and not request.readonly
        ):
            raise ValueError("current mode does not allow this write")
        if authorization.source in {"grant", "legacy"} or authorization.decision in {
            ApprovalDecision.SESSION,
            ApprovalDecision.TASK,
            ApprovalDecision.PERMANENT,
        }:
            if self._find_grant(request) != dict(authorization.grant):
                raise ValueError("approval carrier no longer authorizes this operation")
        elif authorization.source == "policy" and self._coverage(request) != (
            "policy",
            {},
        ):
            raise ValueError("automatic approval policy changed before dispatch")
        event = self.ledger.read_event(authorization.event_id)
        if event is None:
            raise ApprovalUnavailable("operation authorization event is missing")
        # 1. 【审批】【授权原件】只展开当前凭据指向的事件，并核对当时实际发布的决定与资源合同
        fields = {
            "batch_id": original.batch_id,
            "operation_id": original.operation_id,
            "tool": original.tool,
            "intent_digest": original.intent_digest,
            "definition_version": original.definition_version,
            "decision": authorization.decision.value,
            "scope": authorization.grant.get("scope"),
            "grant_id": authorization.grant.get("grant_id"),
            "granted_resource": authorization.grant.get("resource"),
        }
        if (
            event.event,
            event.source,
            event.session_id,
            event.run_id,
            event.task_id,
        ) != (
            "approval.decided",
            authorization.source,
            original.session_id,
            original.run_id,
            original.lease.task_id,
        ) or any(event.payload.get(key) != value for key, value in fields.items()):
            raise ApprovalUnavailable(
                "operation authorization event does not match approved decision"
            )


def _grant_matches(
    request: ApprovalRequest, grant: Mapping[str, object], scope: str
) -> bool:
    """仅旧cron授权沿用原通配合同，新权限使用事件绑定的资源合同；传参：申请、载体、范围；返回：是否匹配。"""
    if request.lease.trigger == "cron" and not grant.get("decision_event_id"):
        from tools.tool_registry import _grant_matches as legacy_matches

        return legacy_matches(request.tool, dict(request.args), dict(grant))
    return approval._grant_matches(request, grant, scope)
