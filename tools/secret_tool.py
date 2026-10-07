from __future__ import annotations

from dataclasses import dataclass

from reins_secrets.store import SecretsVault
from tools.types import ToolError, ToolErrorCategory
from tools.tool_registry import (
    TARGET_SCOPE_LOGICAL,
    TOOL_SOURCE_BUILTIN,
    ToolDefinition,
)
from tools.tool_registry import ToolRegistry

IDEMPOTENT_CONDITIONAL = "conditional"


@dataclass(slots=True)
class SecretToolDefinition(ToolDefinition):
    idempotent: str = IDEMPOTENT_CONDITIONAL


def secret_use(
    name: str,
    action: str = "sign_request",
    **kwargs: object,
) -> dict[str, str]:
    SecretsVault().use(name, action, **kwargs)
    return {"status": "ok", "action_result_summary": "secret action completed"}


def secret_list_names() -> list[str]:
    return SecretsVault().list_names()


def register_tools(registry: ToolRegistry) -> None:
    # 参数校验按声明严格执行，漏声明的字段会被整条拒绝，因此动作真正需要的
    # url（签名请求的目标地址）与 headers（附加请求头）必须在此声明，
    # 否则 _secret_use_executor 的额外参数转交永远收不到值
    params: dict[str, object] = {
        "name": {"type": "string", "required": True},
        "action": {"type": "string", "required": False},
        "scope": {"type": "string", "required": False},
        "url": {"type": "string", "required": False},
        "headers": {"type": "object", "required": False},
    }
    empty: dict[str, object] = {}
    for name, parameters in (("secret_use", params), ("secret_list_names", empty)):
        if registry.get(name) is None:
            registry.register(_definition(name, parameters))


def _definition(name: str, parameters: dict[str, object]) -> SecretToolDefinition:
    executor = _secret_use_executor if name == "secret_use" else _secret_list_executor
    return SecretToolDefinition(
        name=name,
        description="Use or list stored secrets without returning secret values.",
        parameters=parameters,
        toolset="secret",
        risk_level="safe",
        readonly=True,
        target_scope_rule=TARGET_SCOPE_LOGICAL,
        source=TOOL_SOURCE_BUILTIN,
        model_visible=True,
        idempotent=IDEMPOTENT_CONDITIONAL,
        executor=executor,
    )


def _secret_use_executor(args: dict[str, object]) -> object:
    name = str(args.get("name", "")).strip()
    action = str(args.get("action", "sign_request")).strip() or "sign_request"
    kwargs = {
        key: value
        for key, value in args.items()
        if key not in {"name", "action", "scope"} and not key.startswith("__")
    }
    try:
        return secret_use(name, action, **kwargs)
    except (TypeError, ValueError, RuntimeError) as exc:
        return ToolError(ToolErrorCategory.INVALID_INPUT, str(exc), retryable=False)


def _secret_list_executor(_args: dict[str, object]) -> list[str]:
    return secret_list_names()
