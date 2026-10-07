from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from llm.config import LLMProviderConfig
from llm.messages import TextPart, UserMessage
from llm.model_registry import ModelRegistry, ModelSelectionError, ModelSelector
from llm.model_request import ModelRequest, RequestMetadata, build_request
from llm.production_target import (
    production_allowed_model_keys,
    production_connection,
    production_model_registry,
)
from llm.provider_adapter import (
    AdapterRegistry,
    ProviderAdapterError,
    validate_adapter_target,
)
from llm.provider_connection import ProviderConnectionError, ResolvedConnection
from llm.provider_result import ProviderCallResult
from llm.provider_stream import StreamAssembler, StreamProtocolError
from llm.providers.builtin import build_builtin_adapter_registry
from llm.resolved_target import (
    ResolvedModelTarget,
    requires_api_key,
    sanitize_base_url,
)


# 探针真实调用供应商并可能计费，限制输出额度以验证鉴权和端点可达
_PROBE_PROMPT: Final[str] = "Reply with OK."
_PROBE_MESSAGE_ID: Final[str] = "doctor-probe"
_PROBE_OUTPUT_TOKENS: Final[int] = 64

# StreamAssembler 判空消息用的错误码；这一类是"连上了但没内容"，与协议违规分开报
_EMPTY_MESSAGE_CODE: Final[str] = "empty_assistant_message"

# Adapter 契约错误按码前缀归类；只列探针真能撞到的码，其余归 provider_error
_ADAPTER_CATEGORY_BY_CODE: Final[Mapping[str, str]] = {
    "unknown_api_family": "missing_config",
    "adapter_family_mismatch": "missing_config",
    "unsupported_capability": "missing_config",
    "invalid_provider_response": "invalid_provider_response",
    "invalid_model_protocol": "invalid_model_protocol",
}
_UNCLASSIFIED_ADAPTER_CATEGORY: Final[str] = "provider_error"


@dataclass(frozen=True, slots=True)
class ModelDoctorReport:
    ok: bool
    profile_name: str
    provider: str
    model: str
    base_url_host: str
    credential_source: str
    credential_name: str
    credential_present: bool
    error_category: str
    message: str
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class _DoctorStack:
    """保存本次体检可注入的 typed Provider 栈三件套。

    作者：LKX
    时间：2026-08-31 15:20:00
    传参：adapter_registry 为已定好的 Adapter 注册表；model_registry/connection 为
          未注入时留 None，探针发请求时再从 target 推
    返回：不可变的依赖容器
    """

    adapter_registry: AdapterRegistry
    model_registry: ModelRegistry | None
    connection: ResolvedConnection | None


@dataclass(frozen=True, slots=True)
class _ProbeFailure:
    """保存一次探针失败要写进报告的三个事实。

    作者：LKX
    时间：2026-08-31 15:20:00
    传参：category 取 ProviderErrorCategory 值域；message 为脱敏摘要；
          status_code 仅 HTTP 边界失败时有值
    返回：不可变的失败事实
    """

    category: str
    message: str
    status_code: int | None = None


def run_model_doctor(
    target: ResolvedModelTarget | None,
    *,
    adapter_registry: AdapterRegistry | None = None,
    model_registry: ModelRegistry | None = None,
    connection: ResolvedConnection | None = None,
) -> ModelDoctorReport:
    """按 typed Provider 栈体检当前模型配置并交出可读诊断。

    作者：LKX
    时间：2026-08-31 15:20:00
    传参：target 为已解析模型配置，缺失即报缺配置；adapter_registry/model_registry/
          connection 为可注入的 typed Provider 栈三件套，未注入时按生产口径推
    返回：ModelDoctorReport；体检失败以报告字段呈现，不向调用方抛异常
    """
    if target is None:
        return _report(None, False, "missing_config", "model target not available")
    # 1. 本地能判死的配置错先判掉，缺 base_url、缺模型名、缺远端密钥都不该发网络请求
    local_error = _local_error(target)
    if local_error:
        category, message = local_error
        return _report(target, False, category, message)
    stack = _DoctorStack(
        adapter_registry=adapter_registry or build_builtin_adapter_registry(),
        model_registry=model_registry,
        connection=connection,
    )
    failure = _probe(target, stack)
    if failure is None:
        return _report(target, True, "", "provider profile check passed")
    return _report(
        target,
        False,
        failure.category,
        failure.message,
        status_code=failure.status_code,
    )


def _probe(target: ResolvedModelTarget, stack: _DoctorStack) -> _ProbeFailure | None:
    """发一次真实探针请求，把四类边界异常翻成失败事实。

    作者：LKX
    时间：2026-08-31 15:20:00
    传参：target 为已过本地校验的模型配置；stack 为本次可用的 Provider 栈
    返回：失败事实；调用成功返回 None

    体检命令的契约是"把故障讲清楚"而不是把异常抛给终端（app/repl/model_commands.py:169
    直接读 report.ok，没有 try），所以这里照 RealLLMClient._call_provider 的做法把选型、
    连接、Adapter 契约、流协议四类异常转成报告分类，原文一律保留在 message 里。
    """
    try:
        return _provider_failure(_stream_probe(target, stack))
    except ModelSelectionError as exc:
        return _ProbeFailure("missing_config", str(exc))
    except ProviderConnectionError as exc:
        return _ProbeFailure("missing_config", str(exc))
    except ProviderAdapterError as exc:
        return _ProbeFailure(_adapter_category(exc), str(exc))
    except StreamProtocolError as exc:
        return _ProbeFailure(_stream_category(exc), str(exc))


def _stream_probe(
    target: ResolvedModelTarget, stack: _DoctorStack
) -> ProviderCallResult:
    """按 Selector -> Adapter -> StreamAssembler 走一遍与生产同构的调用链。

    作者：LKX
    时间：2026-08-31 15:20:00
    传参：target 为模型配置来源；stack 为 Provider 栈三件套
    返回：ProviderCallResult

    模型注册表与连接都在这里才从 target 推：体检可能在本地校验阶段就返回，
    提前构造连接会白解析一次凭据。
    """
    request = _probe_request(min(_PROBE_OUTPUT_TOKENS, target.output_token_limit))
    registry = stack.model_registry or production_model_registry(
        target, _config(target)
    )
    decision = ModelSelector(registry).select(
        production_allowed_model_keys(),
        request.required_capabilities,
        request.optional_preferences,
    )
    adapter = stack.adapter_registry.require(decision.selected.api_family)
    validate_adapter_target(adapter, decision.selected)
    connection = stack.connection or production_connection(target)
    return StreamAssembler().assemble(
        list(adapter.stream(request, model=decision.selected, connection=connection))
    )


def _provider_failure(result: ProviderCallResult) -> _ProbeFailure | None:
    """把 Provider 边界错误按分类、摘要、HTTP 状态摊进失败事实。

    作者：LKX
    时间：2026-08-31 15:20:00
    传参：result 为流组装结果
    返回：失败事实；result 带回助手消息即视为体检通过，返回 None

    http_status 直接读 ProviderError 字段，不再从响应体里翻 status_code：
    HTTP 状态是 Adapter 在传输边界就拿到的事实，落到响应体里反而会被 Provider 的
    自定义错误格式盖掉。
    """
    error = result.error
    if error is None:
        return None
    return _ProbeFailure(error.category, error.summary, status_code=error.http_status)


def _probe_request(max_output_tokens: int) -> ModelRequest:
    """构造体检用的最小 canonical 请求：一条用户消息、不带工具。

    作者：LKX
    时间：2026-08-31 15:20:00
    传参：max_output_tokens 为探针的实际输出上限
    返回：ModelRequest

    不带工具定义，所以 build_request 推出的必需能力里没有原生工具，只剩流式；
    体检要验的是"这套配置能不能发出去、回得来"，不是模型的工具能力。
    """
    return build_request(
        instructions=(),
        messages=(UserMessage(_PROBE_MESSAGE_ID, (TextPart(_PROBE_PROMPT),)),),
        tools=(),
        metadata=RequestMetadata(),
        max_output_tokens=max_output_tokens,
    )


def _adapter_category(exc: ProviderAdapterError) -> str:
    """按 Adapter 错误码前缀给出报告分类。

    作者：LKX
    时间：2026-08-31 15:20:00
    传参：exc 为 Adapter 或注册表抛出的契约错误
    返回：ProviderErrorCategory 值域内的分类；未登记的新码归 provider_error
    """
    code = str(exc).split(":", 1)[0]
    return _ADAPTER_CATEGORY_BY_CODE.get(code, _UNCLASSIFIED_ADAPTER_CATEGORY)


def _stream_category(exc: StreamProtocolError) -> str:
    """区分"连上了但没内容"与流协议违规。

    作者：LKX
    时间：2026-08-31 15:20:00
    传参：exc 为 StreamAssembler 抛出的协议错误
    返回：empty_response 或 invalid_model_protocol
    """
    code = str(exc).split(":", 1)[0]
    if code == _EMPTY_MESSAGE_CODE:
        return "empty_response"
    return "invalid_model_protocol"


def _local_error(target: ResolvedModelTarget) -> tuple[str, str] | None:
    if not target.base_url.strip():
        return "missing_config", "base_url is missing"
    if not target.model.strip():
        return "missing_config", "model is missing"
    if target.unsupported_reason:
        return "missing_config", target.unsupported_reason
    if requires_api_key(target.base_url) and not target.api_key_present:
        return "missing_config", "api_key for remote HTTPS endpoint is missing"
    return None


def _config(target: ResolvedModelTarget) -> LLMProviderConfig:
    return LLMProviderConfig(
        base_url=target.base_url,
        model=target.model,
        api_key=target.api_key,
        timeout_seconds=target.timeout_seconds,
    )


def _report(
    target: ResolvedModelTarget | None,
    ok: bool,
    error_category: str,
    message: str,
    *,
    status_code: int | None = None,
) -> ModelDoctorReport:
    return ModelDoctorReport(
        ok=ok,
        profile_name=target.profile_name if target else "",
        provider=target.provider if target else "",
        model=target.model if target else "",
        base_url_host=sanitize_base_url(target.base_url) if target else "",
        credential_source=target.credential_source if target else "",
        credential_name=target.credential_name if target else "",
        credential_present=target.api_key_present if target else False,
        error_category=error_category,
        message=message,
        status_code=status_code,
    )


__all__ = ["ModelDoctorReport", "run_model_doctor"]
