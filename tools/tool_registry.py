from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import time
import warnings
from builtins import ExceptionGroup
from collections.abc import Callable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from functools import partial
from threading import RLock
from typing import Any, Protocol, cast
from urllib.parse import urlparse
from uuid import uuid4

import yaml

import approval
import path_security
from approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResource,
    ApprovalUnavailable,
)
from approval.batch import Authorization, BatchAuthorizer, intent_digest
from llm.retry_utils import retry_delays
from llm.messages import freeze_json_object
from runtime.lease import Lease
from runtime.cancellation import CancellationToken
from runtime.types import RunToolsRequest
from runtime.watchdog import Watchdog
from tools.inspection_parser import parse_inspection_payload
from tools.argument_schema import normalize_tool_schema, tool_argument_error
from tools.types import (
    ToolError,
    ToolErrorCategory,
    ToolVisibilityReport,
    ToolVisibilityStatus,
    VisibilityReportSummary,
)
from tools.redacted_files import ProtectedEdit, RedactedFiles

TOOL_SOURCE_BUILTIN = "builtin"
TOOL_SOURCE_MCP_RESERVED = "mcp_reserved"
TOOL_SOURCE_PLUGIN_RESERVED = "plugin_reserved"

_LOG = logging.getLogger(__name__)


class ToolRisk(str, Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    DENY = "deny"


class Idempotent(str, Enum):
    YES = "yes"
    CONDITIONAL = "conditional"
    NO = "no"


class MCPToolRiskError(ValueError):
    pass


class FileDispatchError(RuntimeError):
    """文件锁内发现的派发失效，保留原错误类别和执行边界。"""

    def __init__(self, error: ToolError) -> None:
        """保存既有工具错误；传参：未开始错误；返回：异常对象。"""
        super().__init__(error.message)
        self.error = error


TOOL_RISK_SAFE = TOOL_RISK_READONLY = ToolRisk.SAFE
TOOL_RISK_CONFIRM = TOOL_RISK_WRITE = TOOL_RISK_EXTERNAL_OR_DELEGATE = ToolRisk.CONFIRM
IDEMPOTENT_YES, IDEMPOTENT_CONDITIONAL, IDEMPOTENT_NO = (
    Idempotent.YES,
    Idempotent.CONDITIONAL,
    Idempotent.NO,
)

TOOLSET_FILE = "file"
TOOLSET_WEB = "web"
TOOLSET_MEMORY = "memory"
TOOLSET_AGENT = "agent"

TARGET_SCOPE_PATH = "path"
TARGET_SCOPE_DOMAIN = "domain"
TARGET_SCOPE_LOGICAL = "logical_scope"

AvailabilityCheck = Callable[[], tuple[bool, str | None]]
ToolExecutor = Callable[[dict[str, object]], object]


class CapabilitySource(Protocol):
    """由工具目录持有的外部能力连接，刷新不扩大当前租约。"""

    def refresh(self, lease: Lease, *, force: bool = False) -> None: ...
    def describe(self) -> dict[str, object]: ...
    def close(self) -> None: ...


_MODEL_TOOL_NAME_ALIASES: dict[str, str] = {
    "dir": "list",
    "directory_list": "list",
    "file_list": "list",
    "list_directory": "list",
    "list_files": "list",
    "ls": "list",
    "read_dir": "list",
    "read_directory": "list",
}

_READONLY_TOOL_ARGUMENT_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "list": {
        "path": (
            "dir",
            "directory",
            "directory_path",
            "directoryPath",
            "folder",
            "folder_path",
            "folderPath",
        )
    }
}

_WRITE_TOOL_ARGUMENT_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "file_write": {
        "path": (
            "file",
            "file_path",
            "filepath",
            "filePath",
            "target_path",
            "targetPath",
            "target_file",
            "targetFile",
        ),
        "content": (
            "text",
            "body",
            "contents",
            "new_content",
            "newContent",
            "new_text",
            "newText",
        ),
    },
    "file_patch": {
        "path": (
            "file",
            "file_path",
            "filepath",
            "filePath",
            "target_path",
            "targetPath",
            "target_file",
            "targetFile",
        ),
        "old_text": (
            "old",
            "oldText",
            "old_content",
            "oldContent",
            "old_body",
            "oldBody",
        ),
        "new_text": (
            "new",
            "newText",
            "new_content",
            "newContent",
            "new_body",
            "newBody",
            "replacement",
            "replace_with",
            "replaceWith",
            "content",
            "text",
            "body",
        ),
    },
}


def _always_available() -> tuple[bool, str | None]:
    return True, None


@dataclass(slots=True)
class ToolDefinition:
    # 工具定义是运行时边界，不只是一个 callable。模型只看到安全元数据；
    # 真正执行时还会检查风险、作用域和幂等性。
    name: str
    description: str
    parameters: dict[str, object]
    toolset: str
    risk_level: ToolRisk | str
    readonly: bool
    target_scope_rule: str
    source: str
    model_visible: bool = True
    target_scope_parameters: tuple[str, ...] = ()
    # exec_boundary: 显式声明该工具会跑任意代码/命令（terminal/code_execution），
    # 其 fs 目标（cwd / 落盘脚本）必须经 path_security 强制 workspace 边界 + deny 列表，
    # 不能落入 TARGET_SCOPE_PATH 的单 path 解析（它们无单一 path 参数）。
    # 去掉此标志即可禁用边界（显式、可禁用、有文档）。
    exec_boundary: bool = False
    idempotent: Idempotent | str | None = None
    executor: ToolExecutor | None = field(default=None, repr=False, compare=False)
    parallel_safe: bool = False
    runtime_action: bool = False
    semantics: tuple[str, ...] = ()
    deferred: bool = False
    domain: str = ""
    search_terms: tuple[str, ...] = ()
    readonly_actions: tuple[str, ...] = ()
    mcp_server: str = ""
    availability_check: AvailabilityCheck = field(
        default=_always_available,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """创建时收敛为唯一的标准参数声明；传参：构造字段；返回：无。"""
        self.parameters = normalize_tool_schema(self.parameters)
        if any(
            item not in {"polling", "retryable", "verification"}
            for item in self.semantics
        ):
            raise ValueError("unknown tool semantic")
        self.semantics = tuple(sorted(set(self.semantics)))
        if not isinstance(self.domain, str) or any(
            not isinstance(term, str) or not term for term in self.search_terms
        ):
            raise ValueError("tool domain and search terms must be text")
        self.search_terms = tuple(self.search_terms)
        self.readonly_actions = tuple(self.readonly_actions)

    @property
    def effective_semantics(self) -> tuple[str, ...]:
        """只有内建声明可赋予运行语义，外部标签不授予豁免；传参：无；返回：生效标签。"""
        return self.semantics if self.source == TOOL_SOURCE_BUILTIN else ()

    @property
    def risk(self) -> ToolRisk:
        return _normalize_risk(self.risk_level)

    def check_available(self) -> tuple[bool, str | None]:
        return self.availability_check()

    def required_parameter_names(self) -> list[str]:
        """读取对象级必填字段；传参：无；返回：字段名列表。"""
        return list(cast(list[str], self.parameters.get("required", [])))

    def format_for_model(self) -> str:
        parameter_parts: list[str] = []
        properties = cast(dict[str, object], self.parameters.get("properties", {}))
        for key, spec in properties.items():
            if isinstance(spec, dict):
                type_name = str(spec.get("type", "object"))
                description = str(spec.get("description", "")).strip()
                required = " required" if key in self.required_parameter_names() else ""
                if description:
                    parameter_parts.append(
                        f"{key}<{type_name}{required}>: {description}"
                    )
                else:
                    parameter_parts.append(f"{key}<{type_name}{required}>")
            else:
                parameter_parts.append(f"{key}<object>")

        parameter_text = ", ".join(parameter_parts) or "(none)"
        return (
            f"{self.name}: {self.description}; "
            f"toolset={self.toolset}; "
            f"risk={self.risk.value}; "
            f"idempotent={self.idempotent}; "
            f"readonly={str(self.readonly).lower()}; "
            f"semantics={','.join(self.effective_semantics)}; "
            f"target_scope_rule={self.target_scope_rule}; "
            f"source={self.source}; "
            f"parameters={parameter_text}"
        )

    def format_for_openai_tool(self) -> dict[str, object]:
        """生成与执行校验同源的完整工具定义；传参：无；返回：独立模型视图。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters),
            },
        }

    def action_readonly(self, arguments: Mapping[str, object]) -> bool:
        """按明确动作确定只读性质，复合工具不能整体冒充只读；参数：已校验参数；返回：动作是否只读。"""
        return self.readonly or arguments.get("action") in self.readonly_actions

    def action_risk(self, arguments: Mapping[str, object]) -> ToolRisk:
        """查询动作沿用安全风险，修改保留工具审批要求；参数：已校验参数；返回：实际动作风险。"""
        if (
            self.risk is not ToolRisk.DENY
            and arguments.get("action") in self.readonly_actions
        ):
            return ToolRisk.SAFE
        return self.risk


@dataclass(slots=True)
class PreparedToolExecution:
    """已完成安全与审批检查、等待一次 executor 调用的工具执行对象"""

    definition: ToolDefinition
    arguments: dict[str, object]
    lease: Lease
    watchdog: Any
    max_retries: int
    cancellation: CancellationToken | None = None
    on_late: Callable[[object], None] | None = None
    operation_id: str = ""
    registry_version: str = ""
    definition_version: str = ""
    approval_request: ApprovalRequest | None = None
    authorization: Authorization | None = None
    authorizer: BatchAuthorizer | None = field(default=None, repr=False)
    approval_target: PreparedToolExecution | None = field(default=None, repr=False)
    protected_edit: ProtectedEdit | None = field(default=None, repr=False)
    file_session_id: str = ""
    file_resource: dict[str, object] | None = None
    capture_identity: dict[str, str] = field(default_factory=dict)
    _consumed: bool = field(default=False, init=False, repr=False)

    def authorization_target(self) -> PreparedToolExecution:
        """取得恢复包装内真正产生效果的准备请求；传参：无；返回：最终受授权的操作。"""
        target = self
        while target.approval_target is not None:
            target = target.approval_target
        return target

    def with_authorization(
        self, authorization: Authorization, authorizer: BatchAuthorizer
    ) -> PreparedToolExecution:
        """返回绑定真正目标授权的独立准备树；传参：凭据与核验服务；返回：新准备结果。"""
        if self.approval_target is not None:
            return replace(
                self,
                approval_target=self.approval_target.with_authorization(
                    authorization, authorizer
                ),
            )
        return replace(self, authorization=authorization, authorizer=authorizer)


class ToolRegistry:
    def __init__(
        self,
        *,
        redacted_files: RedactedFiles | None = None,
        data_root: Path | str | None = None,
    ) -> None:
        """创建目录唯一写者；传参：无；返回：无，快照共享执行依赖但不共享可改定义。"""
        self._definitions: dict[str, ToolDefinition] = {}
        self._availability_cache: dict[str, tuple[bool, str | None]] = {}
        self._cache_session_id: str | None = None
        self._data_root = Path(data_root).resolve() if data_root is not None else None
        self.redacted_files = (
            redacted_files if redacted_files is not None else RedactedFiles()
        )
        self._owns_redacted_files = redacted_files is None
        self._lock = RLock()
        self._registry_id = f"registry-{uuid4().hex}"
        self._revision = 0
        self._definition_versions: dict[str, str] = {}
        self._snapshot_owner: ToolRegistry | None = None
        self._sources: dict[str, CapabilitySource] = {}

    def attach_source(self, name: str, source: CapabilitySource) -> None:
        """为连接设置唯一生命周期owner；传参：来源名和连接集合；返回：无，重复owner明确失败。"""
        with self._lock:
            if self._snapshot_owner is not None or name in self._sources:
                raise ValueError(
                    f"capability source already owned or snapshot is read-only: {name}"
                )
            self._sources[name] = source

    def source(self, name: str) -> CapabilitySource | None:
        """取得当前目录所拥有的连接集合；传参：来源名；返回：已存在的owner。"""
        return self._sources.get(name)

    def refresh_sources(self, lease: Lease, *, force: bool = False) -> None:
        """在新请求前发布已知变更，显式force才重连失败来源；传参：当前租约及刷新选项；返回：无。"""
        for source in tuple(self._sources.values()):
            source.refresh(lease, force=force)

    def source_status(self) -> dict[str, object]:
        """返回可发现的连接错误及实际协议版本；传参：无；返回：来源状态，不含凭据。"""
        owner = self._snapshot_owner or self
        return {name: source.describe() for name, source in owner._sources.items()}

    def bind_redacted_files(self, files: RedactedFiles) -> None:
        """在宿主装配时注入跨轮共享的秘密内存；传参：宿主服务；返回：无，注册表关闭不销毁宿主状态。"""
        if self._snapshot_owner is not None:
            raise ValueError("cannot rebind a request registry snapshot")
        if self._owns_redacted_files:
            self.redacted_files.close()
        self.redacted_files, self._owns_redacted_files = files, False

    def close(self) -> None:
        """宿主退出时关闭全部已拥有连接；传参：无；返回：无，清理其余连接后仍暴露任何失败。"""
        errors: list[Exception] = []
        if self._owns_redacted_files:
            self.redacted_files.close()
        for source in tuple(self._sources.values()):
            try:
                source.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("capability sources failed to close", errors)

    @property
    def version(self) -> str:
        """取得完整目录发布版本；传参：无；返回：本进程唯一目录身份及修订。"""
        return f"{self._registry_id}:{self._revision}"

    def definition_version(self, name: str) -> str:
        """取得单个执行定义的发布版本；传参：工具名；返回：版本，未知工具为空。"""
        return self._definition_versions.get(_canonical_tool_name(name), "")

    def register(self, definition: ToolDefinition) -> None:
        """发布一个新工具；传参：定义；返回：无，重复名字明确失败。"""
        self.publish((definition,))

    def replace(self, definition: ToolDefinition) -> None:
        """显式发布既有工具的新版本；传参：新定义；返回：无，在途快照继续引用原执行版本。"""
        if self.get(definition.name) is None:
            raise ValueError(f"unknown tool: {definition.name}")
        self.publish((definition,), replace_names=(definition.name,))

    def publish(
        self,
        definitions: Sequence[ToolDefinition],
        *,
        replace_names: Sequence[str] = (),
    ) -> None:
        """整批校验后原子发布目录；传参：新定义、被替换或移除的名字；返回：无，不提交部分更新。"""
        if self._snapshot_owner is not None:
            raise ValueError("request registry snapshot is read-only")
        prepared = [_publication_definition(item) for item in definitions]
        names = [item.name for item in prepared]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tool names in registry publication")
        replaced = set(replace_names)
        with self._lock:
            conflicts = set(names) & (self._definitions.keys() - replaced)
            if conflicts:
                raise ValueError(f"tool already registered: {sorted(conflicts)}")
            updated = {
                key: value
                for key, value in self._definitions.items()
                if key not in replaced
            }
            versions = {
                key: value
                for key, value in self._definition_versions.items()
                if key not in replaced
            }
            self._revision += 1
            for item in prepared:
                updated[item.name] = item
                versions[item.name] = f"{self.version}:{item.name}"
            self._definitions, self._definition_versions = updated, versions
            self._availability_cache = {
                key: value
                for key, value in self._availability_cache.items()
                if key not in replaced | set(names)
            }

    def snapshot(self) -> ToolRegistry:
        """固定一次请求的目录、Schema和执行器；传参：无；返回：只读目录，授权仍引用当前写者。"""
        if self._snapshot_owner is not None:
            return self
        with self._lock:
            snapshot = ToolRegistry(
                redacted_files=self.redacted_files, data_root=self._data_root
            )
            snapshot._registry_id, snapshot._revision = (
                self._registry_id,
                self._revision,
            )
            snapshot._definitions = dict(self._definitions)
            snapshot._definition_versions = dict(self._definition_versions)
            snapshot._availability_cache = dict(self._availability_cache)
            snapshot._snapshot_owner = self
            return snapshot

    def get(self, name: str) -> ToolDefinition | None:
        """返回独立定义视图；传参：工具名；返回：副本，修改不会绕过目录发布。"""
        definition = self._definitions.get(_canonical_tool_name(name))
        return _copy_definition(definition) if definition is not None else None

    def list_definitions(
        self, *, model_visible_only: bool = False, available_only: bool = False
    ) -> list[ToolDefinition]:
        definitions = list(self._definitions.values())
        if model_visible_only:
            definitions = [item for item in definitions if item.model_visible]
        if available_only:
            definitions = [
                item for item in definitions if self._is_tool_available(item.name)
            ]
        return [_copy_definition(item) for item in definitions]

    def list_tool_names(self, *, model_visible_only: bool = False) -> list[str]:
        return [
            item.name
            for item in self.list_definitions(model_visible_only=model_visible_only)
        ]

    def refresh_availability_cache(self, session_id: str) -> None:
        """Session 开始时刷新 availability 缓存"""
        if self._cache_session_id != session_id:
            self._availability_cache.clear()
            self._cache_session_id = session_id
            # 执行所有工具的 availability_check 并缓存结果
            for tool_name, definition in self._definitions.items():
                try:
                    available, reason = definition.check_available()
                    self._availability_cache[tool_name] = (available, reason)
                except Exception as exc:
                    self._availability_cache[tool_name] = (False, str(exc))

    def invalidate_tool_availability(self, tool_name: str) -> None:
        """工具首次调用失败时使缓存失效"""
        self._availability_cache.pop(tool_name, None)

    def _is_tool_available(self, tool_name: str) -> bool:
        """检查工具是否可用(使用缓存)"""
        if tool_name in self._availability_cache:
            available, _reason = self._availability_cache[tool_name]
            return available
        # 缓存未命中,执行 availability_check
        definition = self._definitions.get(tool_name)
        if definition is None:
            return False
        try:
            available, reason = definition.check_available()
            self._availability_cache[tool_name] = (available, reason)
            return available
        except Exception as exc:
            self._availability_cache[tool_name] = (False, str(exc))
            return False

    def get_visibility_report(
        self,
    ) -> tuple[list[ToolVisibilityReport], VisibilityReportSummary]:
        """生成所有工具的可见性报告"""
        reports: list[ToolVisibilityReport] = []
        visible_count = 0
        hidden_count = 0
        unavailable_count = 0

        for tool_name, definition in self._definitions.items():
            # 获取 availability 状态
            if tool_name in self._availability_cache:
                available, reason = self._availability_cache[tool_name]
            else:
                try:
                    available, reason = definition.check_available()
                    self._availability_cache[tool_name] = (available, reason)
                except Exception as exc:
                    available, reason = False, str(exc)
                    self._availability_cache[tool_name] = (available, reason)

            # 确定 visibility_status
            if not definition.model_visible:
                visibility_status = ToolVisibilityStatus.HIDDEN
                status_reason = "model_visible=False"
                hidden_count += 1
            elif not available:
                # 根据工具类型和原因细分状态
                if tool_name.startswith("mcp_"):
                    if reason and "not configured" in reason.lower():
                        visibility_status = ToolVisibilityStatus.NOT_CONFIGURED
                    elif reason and "not discovered" in reason.lower():
                        visibility_status = ToolVisibilityStatus.NOT_DISCOVERED
                    else:
                        visibility_status = ToolVisibilityStatus.UNAVAILABLE
                else:
                    visibility_status = ToolVisibilityStatus.UNAVAILABLE
                status_reason = reason or "availability_check failed"
                unavailable_count += 1
            else:
                visibility_status = ToolVisibilityStatus.VISIBLE
                status_reason = None
                visible_count += 1

            reports.append(
                ToolVisibilityReport(
                    name=tool_name,
                    source=definition.source,
                    toolset=definition.toolset,
                    risk_level=definition.risk,
                    readonly=definition.readonly,
                    idempotent=definition.idempotent,  # type: ignore
                    visibility_status=visibility_status,
                    reason=status_reason,
                    model_visible=definition.model_visible,
                    available=available,
                )
            )

        # 【工具体系】【数量口径】候选数量不代表发送数量，实际表面由请求选择记录给出
        warnings: list[str] = []

        summary = VisibilityReportSummary(
            total_tools=len(self._definitions),
            visible_tools=visible_count,
            hidden_tools=hidden_count,
            unavailable_tools=unavailable_count,
            warnings=warnings,
        )

        return reports, summary

    def normalize_request(self, request: RunToolsRequest) -> RunToolsRequest:
        tool_name = _canonical_tool_name(str(request.tool_name or request.action))
        arguments = _normalize_tool_arguments(
            tool_name=tool_name,
            arguments=dict(request.arguments),
        )
        payload = str(request.payload).strip()

        if not arguments and payload:
            arguments["payload"] = payload
        if not payload and isinstance(arguments.get("payload"), str):
            payload = str(arguments["payload"]).strip()

        target_scope = request.target_scope
        tool_name = _canonical_tool_name(tool_name)
        definition = self.get(tool_name)
        if definition is not None and target_scope is None:
            target_scope = extract_target_scope(
                rule=definition.target_scope_rule,
                arguments=arguments,
                payload=payload,
            )

        return RunToolsRequest(
            action=request.action or tool_name,
            payload=payload,
            tool_name=tool_name,
            arguments=arguments,
            target_scope=target_scope,
            call_id=request.call_id,
            validation_error=request.validation_error,
        )

    def validate_model_request(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
    ) -> RunToolsRequest | ToolError:
        """解析模型请求并返回可执行参数或具体错误。

        传参：tool_name/arguments 为工具及输入；返回：请求，或字段、可用性错误
        """
        tool_name = _canonical_tool_name(tool_name)
        definition = self.get(tool_name)
        if definition is None:
            return ToolError(
                ToolErrorCategory.INVALID_INPUT, f"unknown tool: {tool_name}"
            )

        available, reason = definition.check_available()
        if not available:
            return ToolError(
                ToolErrorCategory.PERMISSION, f"tool unavailable: {reason or tool_name}"
            )

        normalized_arguments = _normalize_tool_arguments(
            tool_name=tool_name,
            arguments=arguments,
        )
        error = tool_argument_error(definition.parameters, normalized_arguments)
        if error is not None:
            return ToolError(ToolErrorCategory.INVALID_INPUT, f"{tool_name} {error}")

        payload = _build_payload_for_tool(
            tool_name=tool_name,
            arguments=normalized_arguments,
        )
        if tool_name == "inspect" and parse_inspection_payload(payload) is None:
            return ToolError(
                ToolErrorCategory.INVALID_INPUT, "inspect: unsupported payload"
            )

        target_scope = extract_target_scope(
            rule=definition.target_scope_rule,
            arguments=normalized_arguments,
            payload=payload,
        )
        return RunToolsRequest(
            action=tool_name,
            payload=payload,
            tool_name=tool_name,
            arguments=normalized_arguments,
            target_scope=target_scope,
        )

    def prepare_tool_execution(
        self,
        tool_name: str,
        args: dict[str, object],
        lease: Lease,
        *,
        watchdog: Any | None = None,
        max_retries: int = 3,
        cancellation: CancellationToken | None = None,
        on_late: Callable[[object], None] | None = None,
        operation_id: str = "",
        superseded: Callable[[], bool] | None = None,
        approval_batch: str | None = None,
    ) -> PreparedToolExecution | ToolError:
        """完成工具副作用前的解析、安全校验和审批
        作者/时间：LKX，2026-08-16 00:00:00
        传参：tool_name/args 为请求；lease 为租约；watchdog/max_retries 为执行配置；返回 prepared 或 ToolError
        """
        active_watchdog = watchdog or Watchdog(lease, data_root=self._data_root)
        definition = self._resolve_execution_definition(tool_name, lease)
        if isinstance(definition, ToolError):
            return definition
        authorization_error = self._authorization_error(definition)
        if authorization_error is not None:
            return authorization_error
        arguments = deepcopy(
            _normalize_tool_arguments(tool_name=tool_name, arguments=args)
        )
        error = tool_argument_error(definition.parameters, arguments)
        if error is not None:
            return ToolError(ToolErrorCategory.INVALID_INPUT, f"{tool_name} {error}")
        effective_risk = _prepared_execution_risk(definition, arguments, lease)
        if isinstance(effective_risk, ToolError):
            return effective_risk
        prepared = PreparedToolExecution(
            definition=definition,
            arguments=arguments,
            lease=lease,
            watchdog=active_watchdog,
            max_retries=max_retries,
            cancellation=cancellation,
            on_late=on_late,
            operation_id=operation_id,
            registry_version=self.version,
            definition_version=self.definition_version(definition.name),
            file_session_id=getattr(
                getattr(active_watchdog, "budget_owner", None), "session_id", ""
            )
            or self._registry_id,
        )
        file_prepared = self._prepare_file_edit(prepared)
        if isinstance(file_prepared, ToolError):
            return file_prepared
        prepared = file_prepared
        arguments = prepared.arguments
        force_confirmation = (
            prepared.protected_edit is not None
            and prepared.protected_edit.replacement_count > 0
        )
        if force_confirmation:
            effective_risk = ToolRisk.CONFIRM
        if approval_batch is not None:
            request = _batch_approval_request(
                prepared, effective_risk, approval_batch, superseded
            )
            if isinstance(request, ToolError):
                return request
            return replace(prepared, approval_request=request)
        approval_error = _execution_approval_error(
            effective_risk,
            tool_name=tool_name,
            arguments=_approval_arguments(prepared),
            lease=lease,
            watchdog=active_watchdog,
            resource=_approval_resource(definition, arguments, lease),
            cancellation=cancellation,
            superseded=superseded,
            force_confirmation=force_confirmation,
        )
        if approval_error is not None:
            return approval_error
        return prepared

    def _prepare_file_edit(
        self, prepared: PreparedToolExecution
    ) -> PreparedToolExecution | ToolError:
        """在审批前固定敏感配置候选，准备不发布或创建目录；传参：准备操作；返回：固定候选或明确错误。"""
        if (
            prepared.definition.name not in {"file_write", "file_patch"}
            or prepared.definition.source != TOOL_SOURCE_BUILTIN
        ):
            return prepared
        if self.redacted_files.has_private_key_input(
            prepared.arguments, session_id=prepared.file_session_id
        ):
            return ToolError(
                ToolErrorCategory.PERMISSION,
                "private key content cannot be written by file tools",
                retryable=False,
                details={"execution_state": "not_started"},
            )
        requested = Path(str(prepared.arguments.get("path", "")))
        path = path_security.resolve_target(requested, prepared.lease)
        from tools.file_resources import describe_resource

        prepared = replace(
            prepared,
            file_resource=describe_resource(
                prepared.definition, prepared.arguments, prepared.lease
            ),
        )
        if not path_security.uses_redacted_files(requested, prepared.lease):
            return prepared
        try:
            arguments = self.redacted_files.protect_arguments(
                prepared.arguments,
                session_id=prepared.file_session_id,
                request_id=prepared.operation_id or uuid4().hex,
                call_id=prepared.definition.name,
            )
            arguments = self.redacted_files.protect_file_text(
                arguments,
                session_id=prepared.file_session_id,
                request_id=prepared.operation_id or uuid4().hex,
                call_id=prepared.definition.name,
            )
            edit = self.redacted_files.prepare(
                prepared.definition.name,
                arguments,
                session_id=prepared.file_session_id,
                path=path,
            )
            return replace(prepared, arguments=arguments, protected_edit=edit)
        except (OSError, ValueError) as exc:
            return ToolError(
                ToolErrorCategory.INVALID_INPUT,
                str(exc),
                retryable=False,
                details={"execution_state": "not_started"},
            )

    def validate_prepared_tool(
        self, prepared: PreparedToolExecution
    ) -> ToolError | None:
        """派发前重新核对真实参数、路径和授权，延迟准备没有执行权；传参：准备结果；返回：未开始错误。"""
        error = self._authorization_error(prepared.definition)
        if error is None and prepared.approval_target is not None:
            return self.validate_prepared_tool(prepared.approval_target)
        if error is not None or prepared.approval_request is None:
            return error
        if prepared.authorization is None or prepared.authorizer is None:
            return ToolError(
                ToolErrorCategory.PERMISSION,
                "operation has no committed authorization",
                retryable=False,
                details={"execution_state": "not_started"},
            )
        decision = prepared.authorization.decision
        if decision is ApprovalDecision.DENY:
            return ToolError(
                ToolErrorCategory.PERMISSION,
                "approval_denied",
                retryable=False,
                details={"approval_state": "denied", "execution_state": "not_started"},
            )
        if decision is ApprovalDecision.CANCELLED:
            return ToolError(
                ToolErrorCategory.CANCELLED,
                "approval interaction cancelled",
                retryable=False,
                details={
                    "approval_state": "cancelled",
                    "execution_state": "not_started",
                },
            )
        risk = _prepared_execution_risk(
            prepared.definition, prepared.arguments, prepared.lease
        )
        if isinstance(risk, ToolError):
            return risk
        if (
            prepared.protected_edit is not None
            and prepared.protected_edit.replacement_count
        ):
            risk = ToolRisk.CONFIRM
        current = replace(
            prepared.approval_request,
            args=_approval_arguments(prepared),
            lease=prepared.lease,
            risk=risk.value,
            operation_id=prepared.operation_id,
            resource=_approval_resource(
                prepared.definition, prepared.arguments, prepared.lease
            ),
        )
        try:
            prepared.authorizer.validate(current, prepared.authorization)
        except ApprovalUnavailable as exc:
            return ToolError(
                ToolErrorCategory.TRANSPORT,
                "approval facility unavailable",
                retryable=False,
                details={
                    "approval_state": "unavailable",
                    "execution_state": "not_started",
                },
                diagnostics={"approval_cause": str(exc)},
            )
        except ValueError as exc:
            return ToolError(
                ToolErrorCategory.PERMISSION,
                str(exc),
                retryable=False,
                details={"execution_state": "not_started"},
            )
        return None

    def execute_prepared_tool(
        self,
        prepared: PreparedToolExecution,
        *,
        runtime_executor: Callable[[PreparedToolExecution], object] | None = None,
    ) -> object:
        """执行一次已通过安全与审批边界的 prepared 工具请求

        作者：LKX
        时间：2026-08-16 00:00:00
        传参：prepared 为 prepare_tool_execution 的成功返回值
        返回：executor/retry 产生的真实结果；重复消费抛 RuntimeError
        """
        if prepared._consumed:
            raise RuntimeError("prepared tool execution already consumed")
        prepared._consumed = True
        error = self.validate_prepared_tool(prepared)
        if error is not None:
            return error
        if prepared.definition.runtime_action:
            if runtime_executor is None:
                return ToolError(
                    ToolErrorCategory.INVALID_INPUT,
                    "tool requires an active session runtime",
                    retryable=False,
                )
            return runtime_executor(prepared)
        return _execute_with_retry(
            prepared.definition,
            prepared.arguments,
            prepared.watchdog,
            prepared.lease,
            max_retries=prepared.max_retries,
            cancellation=prepared.cancellation,
            on_late=prepared.on_late,
            operation_id=prepared.operation_id,
            authorize=partial(self.validate_prepared_tool, prepared),
            internal_arguments={
                "__redacted_files__": self.redacted_files,
                "__session_id__": prepared.file_session_id,
                "__data_root__": self._data_root,
                "__capture_identity__": prepared.capture_identity,
                "__protected_edit__": prepared.protected_edit,
                "__validate_file__": partial(self._validate_file_dispatch, prepared),
            },
        )

    def _validate_file_dispatch(self, prepared: PreparedToolExecution) -> None:
        """文件锁内再次核对取消、路径和授权；传参：已展示操作；返回：无，失效时阻止发布。"""
        token = prepared.cancellation or getattr(
            prepared.watchdog, "cancellation", None
        )
        if token is not None and token.cancelled:
            raise FileDispatchError(
                ToolError(
                    ToolErrorCategory.CANCELLED,
                    "cancelled before file publication",
                    retryable=False,
                    details={"execution_state": "not_started"},
                )
            )
        error = self.validate_prepared_tool(prepared)
        if error is None:
            risk = _prepared_execution_risk(
                prepared.definition, prepared.arguments, prepared.lease
            )
            error = risk if isinstance(risk, ToolError) else None
        if error is not None:
            raise FileDispatchError(error)
        if prepared.file_resource is not None:
            from tools.file_resources import describe_resource

            current = describe_resource(
                prepared.definition, prepared.arguments, prepared.lease
            )
            if not current.get("known") or current != prepared.file_resource:
                raise FileDispatchError(
                    ToolError(
                        ToolErrorCategory.INVALID_INPUT,
                        "file resource changed after approval; read it again",
                        retryable=False,
                        details={"execution_state": "not_started"},
                    )
                )
        if prepared.protected_edit is not None:
            path = path_security.resolve_target(
                Path(str(prepared.arguments["path"])), prepared.lease
            )
            if path != prepared.protected_edit.path:
                raise FileDispatchError(
                    ToolError(
                        ToolErrorCategory.PERMISSION,
                        "file target changed after approval",
                        retryable=False,
                    )
                )

    def execute_tool(
        self,
        tool_name: str,
        args: dict[str, object],
        lease: Lease,
        *,
        watchdog: Any | None = None,
        max_retries: int = 3,
    ) -> object:
        """兼容一步式工具执行，内部只转发两阶段 owner

        作者：LKX
        时间：2026-08-16 00:00:00
        传参：tool_name/args/lease 为请求；watchdog/max_retries 为执行配置
        返回：prepare 的 ToolError，或 execute_prepared_tool 的真实结果
        """
        prepared = self.prepare_tool_execution(
            tool_name,
            args,
            lease,
            watchdog=watchdog,
            max_retries=max_retries,
        )
        if isinstance(prepared, ToolError):
            return prepared
        prepared.watchdog.reserve_tool_step()
        return self.execute_prepared_tool(prepared)

    def _resolve_execution_definition(
        self, tool_name: str, lease: Lease
    ) -> ToolDefinition | ToolError:
        """只执行已在本目录发布的定义；传参：工具名与租约；返回：定义或明确的未加载错误。"""
        del lease
        definition = self.get(tool_name)
        if definition is not None:
            return definition
        return ToolError(
            ToolErrorCategory.INVALID_INPUT,
            f"unknown tool: {tool_name}; discover and load its definition first",
            retryable=False,
        )

    def _authorization_error(self, definition: ToolDefinition) -> ToolError | None:
        """每次真实派发重查撤销和授权合同；传参：请求所见定义；返回：拒绝原因或None。"""
        owner = self._snapshot_owner or self
        current = owner.get(definition.name)
        if current is None or _authorization_contract(
            current
        ) != _authorization_contract(definition):
            return ToolError(
                ToolErrorCategory.PERMISSION,
                "tool authorization changed; reload its current definition",
                retryable=False,
                details={
                    "execution_state": "not_started",
                    "registry_version": owner.version,
                },
            )
        available, reason = current.check_available()
        if not available:
            owner.invalidate_tool_availability(definition.name)
            return ToolError(
                ToolErrorCategory.PERMISSION,
                f"tool unavailable: {reason or definition.name}",
                retryable=False,
                details={"execution_state": "not_started"},
            )
        return None


def _copy_definition(definition: ToolDefinition) -> ToolDefinition:
    """复制可变Schema并保留后端句柄；传参：已发布定义；返回：独立视图，不重复编译Schema。"""
    result = copy(definition)
    result.parameters = deepcopy(definition.parameters)
    return result


def _publication_definition(definition: ToolDefinition) -> ToolDefinition:
    """在目录发布前统一校验边界元数据；传参：候选定义；返回：独立且规范化的定义。"""
    result = _copy_definition(definition)
    result.parameters = normalize_tool_schema(result.parameters)
    result.risk_level = _normalize_risk(result.risk_level)
    if result.name.startswith("mcp_") and result.risk is ToolRisk.SAFE:
        raise MCPToolRiskError("mcp tools cannot be registered as safe")
    if result.idempotent is None:
        raise ValueError(f"tool idempotent is required: {result.name}")
    result.idempotent = _normalize_idempotent(result.idempotent)
    return result


def _authorization_contract(definition: ToolDefinition) -> tuple[object, ...]:
    """收敛影响已有授权的定义字段；传参：定义；返回：权限比较值，不包含可继续使用的旧实现。"""
    return (
        definition.risk,
        definition.readonly,
        definition.target_scope_rule,
        definition.target_scope_parameters,
        definition.exec_boundary,
        definition.runtime_action,
        definition.source,
        definition.idempotent,
        definition.mcp_server,
        definition.parameters,
        definition.effective_semantics,
        definition.readonly_actions,
    )


def _approval_arguments(prepared: PreparedToolExecution) -> dict[str, object]:
    """把真实候选摘要纳入授权绑定，秘密仍只在内存；传参：准备请求；返回：审批可见的不可变意图材料。"""
    result = dict(prepared.arguments)
    if prepared.protected_edit is not None:
        edit = prepared.protected_edit
        result["protected_file"] = {
            "path": str(edit.path),
            "previous_sha256": edit.previous_sha256,
            "candidate_sha256": edit.candidate_sha256,
            "replacement_count": edit.replacement_count,
        }
    return result


def _batch_approval_request(
    prepared: PreparedToolExecution,
    risk: ToolRisk,
    batch_id: str,
    superseded: Callable[[], bool] | None,
) -> ApprovalRequest | ToolError:
    """冻结实际授权意图，读取阶段不弹窗或产生业务副作用；传参：准备结果、风险、批次及新输入检查；返回：申请。"""
    root = _approval_data_root(prepared.watchdog)
    if root is None:
        return ToolError(
            ToolErrorCategory.PERMISSION, "approval_data_root_required", retryable=False
        )
    owner = getattr(prepared.watchdog, "budget_owner", None)
    definition = prepared.definition
    contract = hashlib.sha256(
        json.dumps(_authorization_contract(definition), sort_keys=True).encode("utf-8")
    ).hexdigest()
    resource_identity = (
        hashlib.sha256(
            json.dumps(dict(prepared.file_resource), sort_keys=True).encode("utf-8")
        ).hexdigest()
        if prepared.file_resource is not None
        else ""
    )
    message = f"授权操作：{definition.name}"
    if prepared.protected_edit is not None:
        message += f"；受保护值变更 {prepared.protected_edit.replacement_count} 项，绑定当前文件版本和候选内容"
    request = ApprovalRequest(
        definition.name,
        freeze_json_object(_approval_arguments(prepared), path="approval.arguments"),
        risk.value,
        prepared.lease,
        root,
        message,
        resource=_approval_resource(definition, prepared.arguments, prepared.lease),
        cancellation=prepared.cancellation,
        superseded=superseded,
        session_id=getattr(owner, "session_id", ""),
        run_id=getattr(owner, "run_id", ""),
        owner_session_id=getattr(
            getattr(getattr(prepared.watchdog, "shared_budget", None), "owner", None),
            "session_id",
            "",
        ),
        operation_id=prepared.operation_id,
        batch_id=batch_id,
        definition_version=contract,
        resource_identity=resource_identity,
        readonly=_operation_is_readonly(definition, prepared.arguments),
        force_confirmation=prepared.protected_edit is not None
        and prepared.protected_edit.replacement_count > 0,
    )
    return replace(request, intent_digest=intent_digest(request))


def _operation_is_readonly(
    definition: ToolDefinition, arguments: Mapping[str, object]
) -> bool:
    """按具体操作识别只读能力，混合工具不能整类降风险；传参：定义与参数；返回：是否只读。"""
    if definition.name == "schedule":
        return arguments.get("action") == "list"
    if definition.name == "terminal_tool":
        from tools.terminal_tool import is_readonly_command

        return is_readonly_command(str(arguments.get("command", "")))
    return definition.action_readonly(arguments)


def _prepared_execution_risk(
    definition: ToolDefinition,
    arguments: dict[str, object],
    lease: Lease,
) -> ToolRisk | ToolError:
    """计算已通过 domain/path/exec 安全检查后的实际风险

    作者：LKX
    时间：2026-08-16 00:00:00
    传参：definition/arguments 为工具请求；lease 为路径与域名能力边界
    返回：有效 ToolRisk，或 fail-closed ToolError
    """
    domain_error = _check_domain_security(definition, arguments, lease)
    if domain_error is not None:
        return domain_error
    if (
        definition.source == TOOL_SOURCE_BUILTIN
        and definition.name == "code_execution_tool"
        and path_security.task_workspace(lease) is None
    ):
        return ToolError(
            ToolErrorCategory.PERMISSION, "code_execution_no_workspace", retryable=False
        )
    if definition.name == "terminal_tool":
        from tools.terminal_tool import _check_terminal_permission

        command_error = _check_terminal_permission(
            str(arguments.get("command", "")), lease
        )
        if command_error is not None:
            return command_error
    path_decision = _check_path_security(definition, arguments, lease)
    if path_decision is path_security.Decision.DENY:
        return ToolError(
            ToolErrorCategory.PERMISSION, "path_security_deny", retryable=False
        )
    exec_decision: path_security.Decision | None = None
    if definition.exec_boundary:
        exec_decision = _check_exec_boundary(definition, arguments, lease)
        if exec_decision is path_security.Decision.DENY:
            return ToolError(
                ToolErrorCategory.PERMISSION, "path_security_deny", retryable=False
            )
    return _effective_risk(definition, path_decision, exec_decision, lease, arguments)


def _execution_approval_error(
    effective_risk: ToolRisk,
    *,
    tool_name: str,
    arguments: dict[str, object],
    lease: Lease,
    watchdog: Any,
    resource: ApprovalResource | None = None,
    cancellation: CancellationToken | None = None,
    superseded: Callable[[], bool] | None = None,
    force_confirmation: bool = False,
) -> ToolError | None:
    """在 executor 前应用 cron grant、风险拒绝和交互审批

    作者/时间：LKX，2026-08-16 00:00:00
    传参：effective_risk、tool_name、arguments、lease、watchdog；返回 ToolError 或 None
    """
    has_cron_grant = False
    if lease.trigger == "cron" and effective_risk is ToolRisk.CONFIRM:
        if force_confirmation:
            return ToolError(
                ToolErrorCategory.PERMISSION,
                "secret replacement requires explicit user confirmation",
                retryable=False,
            )
        has_cron_grant = _has_permanent_grant(lease, tool_name, arguments)
        if not has_cron_grant:
            return ToolError(
                ToolErrorCategory.PERMISSION,
                "requires_permanent_grant_in_cron_mode",
                retryable=False,
            )
    if effective_risk is ToolRisk.DENY:
        return ToolError(
            ToolErrorCategory.PERMISSION, "tool_risk_denied", retryable=False
        )
    if effective_risk is not ToolRisk.CONFIRM or has_cron_grant:
        return None
    data_root = _approval_data_root(watchdog)
    if data_root is None:
        return ToolError(
            ToolErrorCategory.PERMISSION,
            "approval_data_root_required",
            retryable=False,
        )
    try:
        decision = approval.request_approval(
            ApprovalRequest(
                tool=tool_name,
                args=arguments,
                risk=effective_risk.value,
                lease=lease,
                data_root=data_root,
                message=f"Approve tool: {tool_name}",
                resource=resource,
                cancellation=cancellation,
                superseded=superseded,
                session_id=getattr(
                    getattr(watchdog, "budget_owner", None), "session_id", ""
                ),
                run_id=getattr(getattr(watchdog, "budget_owner", None), "run_id", ""),
                owner_session_id=getattr(
                    getattr(getattr(watchdog, "shared_budget", None), "owner", None),
                    "session_id",
                    "",
                ),
                force_confirmation=force_confirmation,
            )
        )
    except ApprovalUnavailable as exc:
        # 【审批】【设施故障】审批通道没给出决定，归为通道故障；正文用稳定描述，原始原因另存供排查
        return ToolError(
            ToolErrorCategory.TRANSPORT,
            "approval facility unavailable",
            retryable=False,
            details={"approval_state": "unavailable"},
            diagnostics={"approval_cause": str(exc)},
        )
    if decision is ApprovalDecision.DENY:
        # 【审批】【用户拒绝】与中断、设施故障区分开，模型据此换方法而不是重试同一动作
        return ToolError(
            ToolErrorCategory.PERMISSION,
            "approval_denied",
            retryable=False,
            details={"approval_state": "denied"},
        )
    if decision is ApprovalDecision.CANCELLED:
        return ToolError(
            ToolErrorCategory.CANCELLED,
            "approval interaction cancelled",
            retryable=False,
            details={"approval_state": "cancelled"},
        )
    return None


def _approval_resource(
    definition: ToolDefinition, arguments: dict[str, object], lease: Lease
) -> ApprovalResource | None:
    """仅将单文件操作规范化为精确资源授权；传参：可信工具声明/参数；返回：明确范围或全参数审批。"""
    if definition.name not in {"file_write", "file_patch"}:
        return None
    target = _definition_path_argument(definition, arguments)
    if target is None:
        return None
    return ApprovalResource(
        definition.name,
        "file",
        str(path_security.resolve_target(Path(target.strip()), lease)).casefold(),
    )


def _approval_data_root(watchdog: Any) -> Path | None:
    """读取当前 watchdog 携带的授权数据根并拒绝无效路径

    参数：watchdog 为当前工具执行的运行看门狗
    返回：可用于 ApprovalRequest 的规范化目录路径，缺失或无效时返回 None
    """
    value = getattr(watchdog, "data_root", None)
    if isinstance(value, str):
        if not value.strip():
            return None
        value = Path(value)
    if not isinstance(value, Path):
        return None
    try:
        resolved = value.expanduser().resolve()
    except OSError:
        return None
    return resolved if not resolved.exists() or resolved.is_dir() else None


def extract_target_scope(
    *,
    rule: str,
    arguments: dict[str, object],
    payload: str,
) -> str | None:
    if rule == TARGET_SCOPE_PATH:
        path_value = _first_path_argument(arguments)
        if path_value is not None:
            return path_value
        if payload.startswith("dir "):
            return payload.removeprefix("dir ").strip() or None
        if payload.startswith("file "):
            return payload.removeprefix("file ").strip() or None
        if payload.startswith("search "):
            search_target = payload.removeprefix("search ").split(" for ", maxsplit=1)[
                0
            ]
            return search_target.strip() or None
        if payload == "workspace":
            return "."

    if rule == TARGET_SCOPE_DOMAIN:
        domain_value = arguments.get("domain")
        if isinstance(domain_value, str):
            normalized = domain_value.strip()
            return normalized or None
        url_value = arguments.get("url")
        if isinstance(url_value, str) and url_value.strip():
            hostname = urlparse(url_value.strip()).hostname
            if hostname:
                return hostname

    if rule == TARGET_SCOPE_LOGICAL:
        scope_value = arguments.get("scope")
        if isinstance(scope_value, str):
            normalized = scope_value.strip()
            return normalized or None
        query_value = arguments.get("query")
        if isinstance(query_value, str):
            normalized = query_value.strip()
            return normalized or None

    return None


def _first_path_argument(arguments: dict[str, object]) -> str | None:
    preferred_names = (
        "path",
        "file_path",
        "filepath",
        "filePath",
        "target_path",
        "targetPath",
    )
    for name in preferred_names:
        value = arguments.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for name, value in arguments.items():
        if "path" in name.casefold() and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """复制参数并保留空串、缩进与换行；传参：模型参数；返回：独立参数副本。"""
    return deepcopy(arguments)


def _normalize_tool_arguments(
    *,
    tool_name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    normalized = _normalize_arguments(arguments)
    alias_map = _READONLY_TOOL_ARGUMENT_ALIASES.get(
        tool_name
    ) or _WRITE_TOOL_ARGUMENT_ALIASES.get(tool_name)
    if alias_map is None:
        return normalized

    # 写文件类工具最容易出现“接近正确”的参数名；
    # 这里仅做窄范围的同义字段提升，避免把安全的别名直接判成协议错误。
    canonicalized = dict(normalized)
    for canonical_name, aliases in alias_map.items():
        if canonical_name in canonicalized:
            for alias_name in aliases:
                canonicalized.pop(alias_name, None)
            continue
        promoted_value = _promote_alias_value(
            arguments=canonicalized,
            aliases=aliases,
        )
        if promoted_value is None:
            continue
        canonicalized[canonical_name] = promoted_value
        for alias_name in aliases:
            canonicalized.pop(alias_name, None)
    return canonicalized


def _canonical_tool_name(tool_name: str) -> str:
    normalized = tool_name.strip()
    return _MODEL_TOOL_NAME_ALIASES.get(normalized, normalized)


def _promote_alias_value(
    *,
    arguments: dict[str, object],
    aliases: tuple[str, ...],
) -> str | None:
    for alias_name in aliases:
        value = arguments.get(alias_name)
        if isinstance(value, str):
            return value
    return None


def _build_payload_for_tool(
    *,
    tool_name: str,
    arguments: dict[str, object],
) -> str:
    payload_value = arguments.get("payload")
    if isinstance(payload_value, str) and payload_value.strip():
        return payload_value.strip()

    if tool_name == "file_read":
        return ""
    if tool_name == "list":
        return ""
    if tool_name == "grep":
        return ""
    if tool_name == "web_search":
        return ""
    if tool_name == "web_fetch":
        return ""
    if tool_name == "web_scan":
        return ""

    return ""


def execute_tool(
    tool_name: str,
    args: dict[str, object],
    lease: Lease,
    *,
    watchdog: Any | None = None,
    max_retries: int = 3,
) -> object:
    return get_default_tool_registry().execute_tool(
        tool_name, args, lease, watchdog=watchdog, max_retries=max_retries
    )


def _backoff_sleep(seconds: float) -> None:
    """工具重试之间的退避等待。

    作者：LKX
    时间：2026-08-28 23:10:00
    传参：seconds 为本次要等待的秒数
    返回：无

    可重试的工具错误都是"对方现在不行、过一会儿可能就行"这一类（网络抖动、对方重启、
    被限流），不等就重发必然撞上同一个错误，还会加重对方负载。单独抽成模块级函数是为了
    留一个替换点：测试要验的是重试流程本身，不需要真等，替掉这里即可，与 retry_utils
    和 watchdog 把 sleep 做成可替换参数是同一个约定。
    """
    time.sleep(seconds)


def _execute_with_retry(
    definition: ToolDefinition,
    arguments: dict[str, object],
    watchdog: Any,
    lease: Lease,
    *,
    max_retries: int,
    cancellation: CancellationToken | None = None,
    on_late: Callable[[object], None] | None = None,
    operation_id: str = "",
    authorize: Callable[[], ToolError | None] | None = None,
    internal_arguments: Mapping[str, object] | None = None,
) -> object:
    delays = retry_delays(max_retries=max_retries)
    attempts = 0
    restore_evidence: dict[str, list[str]] = {
        "restore_point_ids": [],
        "restore_record_errors": [],
    }
    while True:
        error = authorize() if authorize is not None else None
        if error is not None:
            return error
        result = _run_once(
            definition,
            arguments,
            watchdog,
            lease,
            cancellation=cancellation,
            on_late=on_late,
            operation_id=operation_id,
            internal_arguments=internal_arguments,
        )
        result, restore_evidence = _merge_restore_evidence(result, restore_evidence)
        if not _should_retry(result, definition, attempts, max_retries):
            if isinstance(result, ToolError):
                watchdog.record_tool_failure(definition.name, arguments)
            return result
        if cancellation is None:
            _backoff_sleep(delays[attempts])
        elif cancellation.wait(delays[attempts]):
            return ToolError(
                ToolErrorCategory.CANCELLED,
                "cancelled before retry",
                retryable=False,
                details={"execution_state": "completed"},
            )
        watchdog.reserve_tool_step()
        attempts += 1


def _merge_restore_evidence(
    result: object, evidence: dict[str, list[str]]
) -> tuple[object, dict[str, list[str]]]:
    """合并同一操作各次真实重试的恢复证据；参数：当前结果、累计引用；返回：保留全部捕获身份的结果。"""
    details = (
        result.details
        if isinstance(result, ToolError)
        else result.get("meta", {})
        if isinstance(result, dict)
        else {}
    )
    merged = {
        name: list(dict.fromkeys([*saved, *cast(list[str], details.get(name, []))]))
        for name, saved in evidence.items()
    }
    if not any(merged.values()):
        return result, merged
    if isinstance(result, ToolError):
        return replace(result, details={**result.details, **merged}), merged
    if isinstance(result, dict):
        return {**result, "meta": {**details, **merged}}, merged
    return {"content": str(result), "meta": dict(merged)}, merged


def _run_once(
    definition: ToolDefinition,
    arguments: dict[str, object],
    watchdog: Any,
    lease: Lease,
    *,
    cancellation: CancellationToken | None = None,
    on_late: Callable[[object], None] | None = None,
    operation_id: str = "",
    internal_arguments: Mapping[str, object] | None = None,
) -> object:
    if definition.executor is None:
        return ToolError(
            ToolErrorCategory.INVALID_INPUT, f"tool executor missing: {definition.name}"
        )
    executor_args = dict(arguments)
    executor_args.update(internal_arguments or {})
    executor_args["__lease__"] = lease
    executor_args["__task_id__"] = lease.task_id
    selected_data_root = getattr(watchdog, "data_root", None)
    executor_args["__data_root__"] = (
        selected_data_root
        if selected_data_root is not None
        else executor_args.get("__data_root__")
    )
    token = cancellation or CancellationToken(getattr(watchdog, "cancellation", None))
    executor_args["__cancellation__"] = token
    executor_args["__operation_id__"] = operation_id
    executor_args["__timeout_seconds__"] = getattr(
        watchdog, "tool_timeout_seconds", 30.0
    )
    # 1. 【代码执行】【时间合同】请求可缩短子进程时限，不能延长宿主整次工具的时间窗口
    if definition.name == "code_execution_tool" and "timeout_seconds" in arguments:
        executor_args["__timeout_seconds__"] = min(
            float(cast(float, arguments["timeout_seconds"])),
            float(cast(float, executor_args["__timeout_seconds__"])),
        )
    try:
        from runtime.file_capture import run_captured_tool

        result = watchdog.run_tool_with_timeout(
            lambda: run_captured_tool(definition, executor_args),
            cancellation=token,
            on_late=on_late,
        )
    except FileDispatchError as exc:
        return exc.error
    except TimeoutError:
        return ToolError(ToolErrorCategory.TIMEOUT, "tool timed out")
    except ConnectionError as exc:
        return ToolError(ToolErrorCategory.TRANSPORT, str(exc))
    except Exception as exc:
        return ToolError(ToolErrorCategory.UNKNOWN, str(exc), retryable=False)
    return _normalize_tool_error(result)


def _should_retry(
    result: object, definition: ToolDefinition, attempts: int, max_retries: int
) -> bool:
    return (
        isinstance(result, ToolError)
        and result.retryable
        and (
            definition.idempotent == Idempotent.YES
            or result.details.get("execution_state") == "not_started"
        )
        and attempts < max_retries
    )


def _check_path_security(
    definition: ToolDefinition, arguments: dict[str, object], lease: Lease
) -> path_security.Decision | None:
    if definition.target_scope_rule != TARGET_SCOPE_PATH:
        return None
    target = _definition_path_argument(definition, arguments)
    if target is None:
        return None
    path = Path(target)
    filtered = definition.source == TOOL_SOURCE_BUILTIN and definition.name in {
        "file_read",
        "file_write",
        "file_patch",
        "grep",
        "find_path",
        "list",
    }
    return (
        path_security.check_read(path, lease, filtered=filtered)
        if definition.readonly
        else path_security.check_write(path, lease, filtered=filtered)
    )


def _definition_path_argument(
    definition: ToolDefinition, arguments: dict[str, object]
) -> str | None:
    for name in definition.target_scope_parameters:
        value = arguments.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return extract_target_scope(
        rule=definition.target_scope_rule, arguments=arguments, payload=""
    )


_EXEC_TEXT_KEYS = ("command", "code")


def _effective_risk(
    definition: ToolDefinition,
    path_decision: "path_security.Decision | None",
    exec_decision: "path_security.Decision | None",
    lease: Lease,
    arguments: dict[str, object],
) -> ToolRisk:
    """按具体操作与已核对资源计算风险；传参：定义、路径/执行判定、租约和参数；返回：审批风险。"""
    if definition.risk is ToolRisk.DENY:
        return ToolRisk.DENY
    if definition.name == "schedule":
        return ToolRisk.SAFE if arguments.get("action") == "list" else ToolRisk.CONFIRM
    decision = exec_decision if definition.exec_boundary else path_decision
    if decision is path_security.Decision.ALLOWED:
        if definition.exec_boundary:
            from tools.terminal_tool import is_readonly_command

            # 【审批】【具体命令】工作区路径可写不代表任意命令免审批，cron沿用永久授权边界
            if (
                lease.trigger != "cron"
                and definition.name == "terminal_tool"
                and is_readonly_command(str(arguments.get("command", "")))
            ):
                return ToolRisk.SAFE
            return definition.action_risk(arguments)
        return ToolRisk.SAFE
    if decision is path_security.Decision.CONFIRM:
        return ToolRisk.CONFIRM
    return definition.action_risk(arguments)


def _check_exec_boundary(
    definition: ToolDefinition, arguments: dict[str, object], lease: Lease
) -> path_security.Decision | None:
    # exec 工具（terminal/code_execution）的 fs 目标是 cwd 与命令/代码里点名的路径。复用
    # path_security 同一规则源（deny 列表 + workspace 限定 + roots），不另起一套。这是尽力而为
    # 的字符串扫描、非 OS 沙箱：挡手滑与朴素注入，挡不住刻意混淆（base64 路径、env 间接）。
    cwd_decision = _check_exec_cwd(arguments, lease)
    if cwd_decision is path_security.Decision.DENY:
        return path_security.Decision.DENY
    if not _exec_guard_enabled(lease):
        return (
            None  # 关闭扫描：回退旧行为（仅 cwd 边界，exec 工具仍按声明风险弹确认）。
        )
    text = _exec_command_text(arguments)
    if not text:
        return None
    scan = path_security.classify_exec_targets(text, lease)
    decision = _combine_exec_decision(cwd_decision, scan)
    if decision is not path_security.Decision.ALLOWED:
        _LOG.info(
            "【执行边界】【路径检查】exec_guard %s on %s: %s",
            decision.value,
            definition.name,
            text[:200],
        )
    return decision


def _check_exec_cwd(
    arguments: dict[str, object], lease: Lease
) -> path_security.Decision | None:
    cwd_value = arguments.get("cwd")
    if isinstance(cwd_value, str) and cwd_value.strip():
        target: Path | None = Path(cwd_value.strip())
    else:
        target = path_security.task_workspace(lease)
    if target is None:
        return None
    return path_security.check_write(target, lease)


def _combine_exec_decision(
    cwd_decision: "path_security.Decision | None",
    scan: path_security.Decision,
) -> path_security.Decision:
    if scan is path_security.Decision.DENY:
        return path_security.Decision.DENY
    if path_security.Decision.CONFIRM in (cwd_decision, scan):
        return path_security.Decision.CONFIRM
    return path_security.Decision.ALLOWED


def _exec_guard_enabled(lease: Lease) -> bool:
    # master 开关：缺省视为 True。设 exec_guard.enabled=False 即一键回退旧行为（可禁用、有文档）。
    guard = lease.capabilities.get("exec_guard")
    if isinstance(guard, Mapping):
        return guard.get("enabled") is not False
    return True


def _exec_command_text(arguments: dict[str, object]) -> str:
    parts = [
        str(arguments[key])
        for key in _EXEC_TEXT_KEYS
        if isinstance(arguments.get(key), str)
    ]
    return " ".join(parts)


def _check_domain_security(
    definition: ToolDefinition, arguments: dict[str, object], lease: Lease
) -> ToolError | None:
    if definition.target_scope_rule != TARGET_SCOPE_DOMAIN:
        return None
    network = lease.capabilities.get("network")
    if not isinstance(network, Mapping):
        return None
    if network.get("enabled") is False:
        return ToolError(
            ToolErrorCategory.PERMISSION, "network_disabled", retryable=False
        )
    domain = extract_target_scope(
        rule=definition.target_scope_rule, arguments=arguments, payload=""
    )
    if domain is None:
        return None
    denied = [str(item).casefold() for item in network.get("deny_domains", []) if item]
    if any(fnmatch.fnmatch(domain.casefold(), pattern) for pattern in denied):
        return ToolError(
            ToolErrorCategory.PERMISSION,
            "network_domain_denied",
            retryable=False,
        )
    return None


def _has_permanent_grant(lease: Lease, tool_name: str, args: dict[str, object]) -> bool:
    grant_entries = _permanent_grants_from_config()
    schedule = lease.capabilities.get("schedule")
    grants = (
        schedule.get("required_permanent_grants")
        if isinstance(schedule, dict)
        else None
    )
    if isinstance(grants, list):
        grant_entries.extend(dict(grant) for grant in grants if isinstance(grant, dict))
    return any(_grant_matches(tool_name, args, grant) for grant in grant_entries)


def _permanent_grants_from_config() -> list[dict[str, object]]:
    path = Path.home() / ".reins" / "config.yaml"
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    grants = data.get("permanent_grants") if isinstance(data, dict) else None
    if not isinstance(grants, list):
        return []
    return [dict(item) for item in grants if isinstance(item, dict)]


def _grant_matches(
    tool_name: str, args: dict[str, object], grant: dict[str, object]
) -> bool:
    if grant.get("tool") == tool_name:
        grant_args = grant.get("args")
        return isinstance(grant_args, dict) and _args_match(args, grant_args)
    values = grant.get(tool_name)
    return isinstance(values, list) and any(
        _value_matches(args, item) for item in values
    )


def _args_match(args: dict[str, object], grant_args: dict[object, object]) -> bool:
    normalized_grant_args = _normalize_arguments(
        {str(key): value for key, value in grant_args.items()}
    )
    if dict(args) == normalized_grant_args:
        return True
    for key, expected in normalized_grant_args.items():
        actual = args.get(str(key))
        if isinstance(expected, str) and (
            not isinstance(actual, str) or not fnmatch.fnmatch(actual, expected)
        ):
            return False
        if not isinstance(expected, str) and actual != expected:
            return False
    return True


def _value_matches(args: dict[str, object], expected: object) -> bool:
    if not isinstance(expected, str):
        return False
    payload = " ".join(str(value) for value in args.values())
    return fnmatch.fnmatch(payload, expected)


def _normalize_tool_error(result: object) -> object:
    if isinstance(result, ToolError):
        return result
    category = getattr(result, "category", None)
    if category is None:
        return result
    value = category.value if hasattr(category, "value") else str(category)
    normalized_category = (
        ToolErrorCategory(value)
        if value in ToolErrorCategory._value2member_map_
        else ToolErrorCategory.UNKNOWN
    )
    return ToolError(
        normalized_category,
        str(getattr(result, "message", "")),
        retryable=bool(getattr(result, "retryable", False)),
        partial_state=str(getattr(result, "partial_state", "")),
    )


def _normalize_risk(value: ToolRisk | str) -> ToolRisk:
    if isinstance(value, ToolRisk):
        return value
    mapping = {
        "readonly": ToolRisk.SAFE,
        "write": ToolRisk.CONFIRM,
        "external_or_delegate": ToolRisk.CONFIRM,
    }
    if value in mapping:
        warnings.warn(
            f"legacy tool risk value is deprecated: {value}",
            DeprecationWarning,
            stacklevel=3,
        )
        return mapping[value]
    return ToolRisk(value)


def _normalize_idempotent(value: Idempotent | str) -> Idempotent:
    if isinstance(value, Idempotent):
        return value
    return Idempotent(value)


_DEFAULT_TOOL_REGISTRY = ToolRegistry()
_DEFAULT_BUILTINS_REGISTERED = False


def get_default_tool_registry() -> ToolRegistry:
    global _DEFAULT_BUILTINS_REGISTERED
    if not _DEFAULT_BUILTINS_REGISTERED:
        from tools import builtin_tools

        builtin_tools.register_tools(_DEFAULT_TOOL_REGISTRY)
        _DEFAULT_BUILTINS_REGISTERED = True
    return _DEFAULT_TOOL_REGISTRY
