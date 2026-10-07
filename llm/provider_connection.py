from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Mapping, Protocol


_CLIENT_VERSION: Final[str] = "0.1.0"
# 客户端身份只在这里定义一次：Header 构造与探针脚本的默认 User-Agent 都取这个值
DEFAULT_USER_AGENT: Final[str] = f"Reins/{_CLIENT_VERSION}"
_RESERVED_HEADERS: Final[Mapping[str, frozenset[str]]] = {
    "openai_chat": frozenset(
        {
            "authorization",
            "content-type",
            "accept",
            "user-agent",
            "openai-organization",
            "openai-project",
            "x-reins-tenant",
        }
    ),
    "openai_responses": frozenset(
        {
            "authorization",
            "content-type",
            "accept",
            "user-agent",
            "openai-organization",
            "openai-project",
            "x-reins-tenant",
        }
    ),
    "anthropic_messages": frozenset(
        {
            "x-api-key",
            "anthropic-version",
            "anthropic-beta",
            "content-type",
            "accept",
            "user-agent",
            "x-reins-tenant",
        }
    ),
}


class ProviderConnectionError(ValueError):
    """表示连接配置、credential 或 Header 策略失败。"""


class CredentialProvider(Protocol):
    """定义发送前解析 credential reference 的窄端口。"""

    def resolve(self, credential_ref: str) -> str:
        """解析 credential reference 并返回仅驻留内存的 secret。"""
        ...


@dataclass(frozen=True, slots=True)
class ConnectionProfile:
    """保存可持久化且不含 secret value 的连接配置。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：URL/timeout/credential_ref 为连接事实；租户与 extra header 为结构化选项
    返回：深拷贝 Header 的不可变 profile
    """

    key: str
    base_url: str
    timeout_seconds: float
    credential_ref: str
    organization: str = ""
    project: str = ""
    tenant: str = ""
    feature_headers: tuple[str, ...] = ()
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验连接事实并冻结受控 header。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法配置抛 ProviderConnectionError
        """
        for name in ("key", "base_url", "credential_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProviderConnectionError(f"connection_{name}_required")
        if self.timeout_seconds <= 0:
            raise ProviderConnectionError("connection_timeout_invalid")
        headers = dict(self.extra_headers)
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in headers.items()
        ):
            raise ProviderConnectionError("connection_extra_header_invalid")
        object.__setattr__(self, "extra_headers", MappingProxyType(headers))
        object.__setattr__(self, "feature_headers", tuple(self.feature_headers))


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedConnection:
    """保存发送边界已解析 secret 与最终 Header，不允许持久化。"""

    base_url: str
    timeout_seconds: float
    credential: str
    headers: Mapping[str, str]

    def __repr__(self) -> str:
        """返回不含 credential 和鉴权 Header 值的调试摘要。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：脱敏字符串
        """
        return (
            "ResolvedConnection(base_url="
            f"{self.base_url!r}, timeout_seconds={self.timeout_seconds!r}, headers=<redacted>)"
        )


class HeaderPolicy:
    """按 API family 统一拥有 auth、版本、租户和 client identity Header。"""

    def __init__(self, api_family: str) -> None:
        """选择一个闭合 API-family Header 策略。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：api_family 为注册 family
        返回：无
        """
        if api_family not in _RESERVED_HEADERS:
            raise ProviderConnectionError(f"unknown_header_policy:{api_family}")
        self.api_family = api_family

    def build(self, profile: ConnectionProfile, credential: str) -> Mapping[str, str]:
        """构建最终 Header 并拒绝大小写不敏感的 reserved override。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：profile 为无 secret 配置；credential 为发送前解析值
        返回：不可变最终 Header
        """
        reserved = _RESERVED_HEADERS[self.api_family]
        for name in profile.extra_headers:
            if name.lower() in reserved:
                raise ProviderConnectionError(f"reserved_header_override:{name}")
        headers = _family_headers(self.api_family, profile, credential)
        headers.update(profile.extra_headers)
        return MappingProxyType(headers)


def resolve_connection(
    profile: ConnectionProfile,
    credentials: CredentialProvider,
    policy: HeaderPolicy,
) -> ResolvedConnection:
    """在发送前解析单一 credential 并应用 HeaderPolicy。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：profile 为连接引用；credentials 为注入解析器；policy 为 family 策略
    返回：只驻留内存的 ResolvedConnection
    """
    credential = credentials.resolve(profile.credential_ref)
    if not isinstance(credential, str) or not credential.strip():
        raise ProviderConnectionError("missing_config:credential")
    headers = policy.build(profile, credential)
    return ResolvedConnection(
        profile.base_url, profile.timeout_seconds, credential, headers
    )


def _family_headers(
    api_family: str,
    profile: ConnectionProfile,
    credential: str,
) -> dict[str, str]:
    """按 API family 拼装内容协商、客户端身份与鉴权 Header。

    作者：LKX
    时间：2026-08-31 17:20:00
    传参：api_family 为注册 family；profile 为无 secret 连接配置；credential 为已解析 secret
    返回：该 family 的最终 Header 字典
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if profile.tenant:
        headers["X-Reins-Tenant"] = profile.tenant
    if api_family == "anthropic_messages":
        headers.update({"x-api-key": credential, "anthropic-version": "2023-06-01"})
        if profile.feature_headers:
            headers["anthropic-beta"] = ",".join(profile.feature_headers)
        return headers
    headers["Authorization"] = f"Bearer {credential}"
    if profile.organization:
        headers["OpenAI-Organization"] = profile.organization
    if profile.project:
        headers["OpenAI-Project"] = profile.project
    return headers


__all__ = [
    "DEFAULT_USER_AGENT",
    "ConnectionProfile",
    "CredentialProvider",
    "HeaderPolicy",
    "ProviderConnectionError",
    "ResolvedConnection",
    "resolve_connection",
]
