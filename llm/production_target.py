from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping
from urllib.parse import urlsplit

from llm.api_modes import API_MODE_FAMILIES
from llm.config import LLMProviderConfig
from llm.model_registry import (
    CapabilityValue,
    ModelDescriptor,
    ModelRegistry,
    ModelSelectionError,
)
from llm.model_request import Capability, PreferenceKind
from llm.reasoning import reasoning_options
from llm.provider_connection import (
    ConnectionProfile,
    HeaderPolicy,
    ProviderConnectionError,
    ResolvedConnection,
    resolve_connection,
)
from llm.resolved_target import ResolvedModelTarget


# 单模型配置只产出一个候选，这个 key 同时是注册表索引和 ModelSelector 的允许候选
PRODUCTION_MODEL_KEY: Final[str] = "production"

# 无 profile 时 target.profile_name 是空串，而 ModelDescriptor 要求 connection_profile_key
# 非空白，所以给它一个固定名字；单模型配置下连接只有一份，这个名字不需要区分来源
_DEFAULT_CONNECTION_PROFILE_KEY: Final[str] = "production_default"

# credential_ref 是引用名不是密钥值（ConnectionProfile 可被持久化），无命名凭据时用这个固定引用
_DEFAULT_CREDENTIAL_REF: Final[str] = "production_default_credential"


@dataclass(frozen=True, slots=True, repr=False)
class _InlineCredential:
    """把已解析到内存的密钥值按 CredentialProvider 端口交出。

    作者：LKX
    时间：2026-08-30 16:40:00
    传参：credential_ref 为本次允许解析的引用名；secret 为已解析密钥，缺失时为 None
    返回：满足 CredentialProvider 协议的解析器
    """

    credential_ref: str
    secret: str | None

    def resolve(self, credential_ref: str) -> str:
        """按引用名交出密钥值，引用名不符即失败。

        作者：LKX
        时间：2026-08-30 16:40:00
        传参：credential_ref 为 ConnectionProfile 声明的引用名
        返回：密钥值；密钥缺失时返回空串，由 resolve_connection 判成 missing_config
        """
        # 一个解析器只持有一份密钥，引用名不符说明连接和凭据被配错，不能交出无关密钥
        if credential_ref != self.credential_ref:
            raise ProviderConnectionError(f"unknown_credential_ref:{credential_ref}")
        return self.secret or ""

    def __repr__(self) -> str:
        """返回不含密钥值的调试摘要。

        作者：LKX
        时间：2026-08-30 16:40:00
        传参：无
        返回：脱敏字符串
        """
        return f"_InlineCredential(credential_ref={self.credential_ref!r}, secret=<redacted>)"


def production_model_registry(
    target: ResolvedModelTarget,
    config: LLMProviderConfig,
) -> ModelRegistry:
    """把用户配置翻成生产唯一候选模型的能力事实注册表。

    作者：LKX
    时间：2026-08-30 16:40:00
    传参：target 为已解析配置与凭据来源；config 为发送用的 Wire 模型名来源
    返回：只含一个 PRODUCTION_MODEL_KEY 候选的 ModelRegistry；api_mode 非法时抛 ModelSelectionError

    config_source / credential_source / context_window_source / context_window_defaulted /
    api_key_present / unsupported_reason 六个字段不进 descriptor：ModelObservation 直接从
    resolved_target 读它们，复制进来会造出第二份事实。
    """
    # 1. api_mode 决定用哪一家协议的 Adapter，取值超出映射表即失败，不落默认 family
    api_family = _api_family_for(target.api_mode)
    descriptor = ModelDescriptor(
        model_key=PRODUCTION_MODEL_KEY,
        provider=target.provider,
        model_id=config.model,
        api_family=api_family,
        capabilities=_production_capabilities(target),
        preference_values={
            PreferenceKind.REASONING_LEVEL: frozenset(
                reasoning_options(target.model, target.api_mode)
            )
        },
        connection_profile_key=_connection_profile_key(target),
    )
    return ModelRegistry([descriptor])


def production_connection(target: ResolvedModelTarget) -> ResolvedConnection:
    """把用户配置翻成发送边界的连接与鉴权 Header。

    作者：LKX
    时间：2026-08-30 16:40:00
    传参：target 为连接地址、超时与密钥来源
    返回：只驻留内存的 ResolvedConnection；密钥缺失时抛 ProviderConnectionError

    连接事实一律取自 target，它是 LLMProviderConfig 的上游（app/cli.py:117-122 用 target 造
    config），所以这里不需要第二个入参。密钥缺失走 resolve_connection 自身的
    missing_config:credential 失败路径，不传哨兵占位串、不给无鉴权端点留第二条构造路径。
    """
    profile = ConnectionProfile(
        key=_connection_profile_key(target),
        base_url=target.base_url,
        timeout_seconds=target.timeout_seconds,
        credential_ref=_credential_ref(target),
    )
    # 1. 密钥值只在 CredentialProvider.resolve 的返回值里出现，不进可持久化的 ConnectionProfile
    credentials = _InlineCredential(profile.credential_ref, target.api_key)
    return resolve_connection(
        profile, credentials, HeaderPolicy(_api_family_for(target.api_mode))
    )


def production_allowed_model_keys() -> tuple[str, ...]:
    """列出生产允许 ModelSelector 选择的候选 key。

    作者：LKX
    时间：2026-08-30 16:40:00
    传参：无
    返回：单模型配置下只有 PRODUCTION_MODEL_KEY 一个候选
    """
    return (PRODUCTION_MODEL_KEY,)


def _api_family_for(api_mode: str) -> str:
    """把用户配置的 api_mode 映射成 Adapter 的 api_family。"""
    try:
        return API_MODE_FAMILIES[api_mode]
    except KeyError as exc:
        raise ModelSelectionError(f"unsupported_api_mode:{api_mode}") from exc


def _production_capabilities(
    target: ResolvedModelTarget,
) -> Mapping[Capability, CapabilityValue]:
    """声明Adapter承接能力与配置允许的窗口额度；传参：目标；返回：选择器约束，不自动推断厂商上限。"""
    endpoint = urlsplit(target.base_url)
    return {
        # 1. 【模型配置】【缓存支持】官方文档声明Claude缓存；兼容网关不继承这一支持事实
        Capability.PROMPT_CACHE: (
            target.api_mode == "anthropic_messages"
            and endpoint.scheme == "https"
            and endpoint.hostname == "api.anthropic.com"
            and target.model.startswith("claude-")
        ),
        # 生产默认走原生工具调用协议且请求带工具定义，今天是隐含假设，此处显式化
        Capability.NATIVE_TOOLS: True,
        # 【模型输入】【图片】声明API适配器已支持图像；实际端点拒绝仍保留其错误，不切模型或丢图片
        Capability.IMAGE_INPUT: True,
        # OpenAIChatAdapter.translate_stream 只吃 chunk 流，非流式响应会被判成
        # invalid_provider_response，所以生产请求必须流式
        Capability.STREAMING: True,
        # 真实端点会产出推理正文（reasoning_content），Adapter 把它译成 thinking 内容块
        Capability.REASONING: True,
        # token 类能力，CapabilityRequirement 对它按 minimum 比较，值取用户配置的窗口
        Capability.CONTEXT_WINDOW_TOKENS: target.context_window,
        Capability.OUTPUT_TOKENS: target.output_token_limit,
    }


def _connection_profile_key(target: ResolvedModelTarget) -> str:
    """取 descriptor 与 ConnectionProfile 共用的连接引用名。"""
    return target.profile_name.strip() or _DEFAULT_CONNECTION_PROFILE_KEY


def _credential_ref(target: ResolvedModelTarget) -> str:
    """取凭据引用名；这是引用不是密钥值。"""
    return target.credential_name.strip() or _DEFAULT_CREDENTIAL_REF


__all__ = [
    "PRODUCTION_MODEL_KEY",
    "production_allowed_model_keys",
    "production_connection",
    "production_model_registry",
]
