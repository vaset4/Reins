from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.tool_registry import Idempotent, ToolRisk


class ToolErrorCategory(str, Enum):
    BUSINESS = "business"
    CANCELLED = "cancelled"
    TIMEOUT, PERMISSION, INVALID_INPUT, TRANSPORT, UNKNOWN = (
        "timeout",
        "permission",
        "invalid_input",
        "transport",
        "unknown",
    )

    @property
    def retryable(self) -> bool:
        return self in {ToolErrorCategory.TIMEOUT, ToolErrorCategory.TRANSPORT}


@dataclass(slots=True, init=False)
class ToolError:
    category: ToolErrorCategory
    message: str
    retryable: bool
    partial_state: str = ""
    details: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __init__(
        self,
        category: ToolErrorCategory,
        message: str,
        retryable: bool | None = None,
        partial_state: str = "",
        *,
        details: dict[str, object] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        """区分模型可见失败与内部诊断；传参：类别、描述、重试/状态及两类详情；返回：错误对象。"""
        self.category = category
        self.message = message
        self.retryable = category.retryable if retryable is None else retryable
        self.partial_state = partial_state
        self.details = dict(details or {})
        self.diagnostics = dict(diagnostics or {})


class ToolVisibilityStatus(str, Enum):
    """工具可见性状态"""

    VISIBLE = "visible"  # 对模型可见且可用
    HIDDEN = "hidden"  # model_visible=False
    UNAVAILABLE = "unavailable"  # availability_check 失败
    APPROVAL_REQUIRED = "approval_required"  # 需要审批(但对模型可见)
    NOT_CONFIGURED = "not_configured"  # MCP 未配置/未启用
    NOT_DISCOVERED = "not_discovered"  # MCP 懒注册失败
    DENIED = "denied"  # 被权限/安全规则拒绝


@dataclass
class ToolVisibilityReport:
    """单个工具的可见性报告"""

    name: str
    source: str  # builtin | mcp_reserved | plugin_reserved
    toolset: str | None  # file | web | memory | agent
    risk_level: ToolRisk
    readonly: bool
    idempotent: Idempotent
    visibility_status: ToolVisibilityStatus
    reason: str | None  # 不可见/不可用的原因说明
    model_visible: bool  # 原始 model_visible 字段
    available: bool  # availability_check 结果


@dataclass
class VisibilityReportSummary:
    """可见性报告汇总"""

    total_tools: int
    visible_tools: int
    hidden_tools: int
    unavailable_tools: int
    warnings: list[str]
