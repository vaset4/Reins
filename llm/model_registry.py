from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TypeAlias

from llm.model_request import (
    Capability,
    CapabilityRequirement,
    ModelPreference,
    PreferenceKind,
)


CapabilityValue: TypeAlias = bool | int


class ModelSelectionError(ValueError):
    """表示模型注册事实或显式候选选择失败。"""


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """保存一个模型的稳定身份、能力事实和连接引用。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：model/provider/family 为身份；capabilities/preferences 为审计事实
    返回：不可变模型描述
    """

    model_key: str
    provider: str
    model_id: str
    api_family: str
    capabilities: Mapping[Capability, CapabilityValue]
    preference_values: Mapping[PreferenceKind, frozenset[str]]
    connection_profile_key: str
    limits: Mapping[str, int] = field(default_factory=dict)
    continuity_constraints: Mapping[str, object] = field(default_factory=dict)
    priority: int = 0

    def __post_init__(self) -> None:
        """校验并冻结模型事实。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法事实抛 ModelSelectionError
        """
        for name in (
            "model_key",
            "provider",
            "model_id",
            "api_family",
            "connection_profile_key",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ModelSelectionError(f"invalid_model_descriptor:{name}")
        capabilities = dict(self.capabilities)
        if any(
            not isinstance(key, Capability)
            or not isinstance(value, (bool, int))
            or isinstance(value, int)
            and not isinstance(value, bool)
            and value < 0
            for key, value in capabilities.items()
        ):
            raise ModelSelectionError("invalid_model_descriptor:capability")
        preferences = {}
        for key, values in self.preference_values.items():
            if not isinstance(key, PreferenceKind):
                raise ModelSelectionError("invalid_model_descriptor:preference")
            if isinstance(values, (str, bytes, bytearray)):
                raise ModelSelectionError("invalid_model_descriptor:preference_values")
            copied = frozenset(values)
            if any(not isinstance(value, str) or not value.strip() for value in copied):
                raise ModelSelectionError("invalid_model_descriptor:preference_values")
            preferences[key] = copied
        limits = dict(self.limits)
        if any(
            not isinstance(key, str) or not isinstance(value, int) or value <= 0
            for key, value in limits.items()
        ):
            raise ModelSelectionError("invalid_model_descriptor:limits")
        object.__setattr__(self, "capabilities", MappingProxyType(capabilities))
        object.__setattr__(self, "preference_values", MappingProxyType(preferences))
        object.__setattr__(self, "limits", MappingProxyType(limits))
        object.__setattr__(
            self,
            "continuity_constraints",
            MappingProxyType(dict(self.continuity_constraints)),
        )


class ModelRegistry:
    """保存按 stable model key 唯一索引的模型事实快照。"""

    def __init__(
        self, descriptors: list[ModelDescriptor] | tuple[ModelDescriptor, ...]
    ) -> None:
        """构造不可变模型索引并拒绝重复 key。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：descriptors 为本次选择可见的模型事实
        返回：无
        """
        values: dict[str, ModelDescriptor] = {}
        for descriptor in descriptors:
            if descriptor.model_key in values:
                raise ModelSelectionError(f"duplicate_model_key:{descriptor.model_key}")
            values[descriptor.model_key] = descriptor
        self._values = MappingProxyType(values)

    def require(self, model_key: str) -> ModelDescriptor:
        """按 stable key 取得模型事实。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：model_key 为注册键
        返回：唯一 ModelDescriptor；未知时抛 ModelSelectionError
        """
        try:
            return self._values[model_key]
        except KeyError as exc:
            raise ModelSelectionError(f"unknown_model_key:{model_key}") from exc


@dataclass(frozen=True, slots=True)
class ModelRejection:
    """记录一个候选未满足 required capability 的原因。"""

    model_key: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """保存确定性选择、拒绝原因、偏好 notice 和排序依据。"""

    selected: ModelDescriptor
    rejections: tuple[ModelRejection, ...] = ()
    notices: tuple[str, ...] = ()
    reason: str = "highest_optional_score_then_priority_then_allowed_order"


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    descriptor: ModelDescriptor
    optional_score: int
    allowed_index: int
    notices: tuple[str, ...] = field(default_factory=tuple)


class ModelSelector:
    """只在调用方显式允许的候选中执行可审计选择。"""

    def __init__(self, registry: ModelRegistry) -> None:
        """注入模型事实注册表。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：registry 为唯一模型事实 owner
        返回：无
        """
        self._registry = registry

    def select(
        self,
        allowed_keys: tuple[str, ...],
        required: frozenset[CapabilityRequirement],
        preferences: tuple[ModelPreference, ...],
    ) -> SelectionDecision:
        """过滤 required 并按偏好、优先级、允许顺序确定模型。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：allowed_keys 为显式候选；required/preferences 为 canonical 请求意图
        返回：含拒绝和 notice 的 SelectionDecision
        """
        if not allowed_keys:
            raise ModelSelectionError("empty_allowed_candidates")
        ranked: list[_RankedCandidate] = []
        rejections: list[ModelRejection] = []
        for index, key in enumerate(allowed_keys):
            descriptor = self._registry.require(key)
            reasons = _missing_requirements(descriptor, required)
            if reasons:
                rejections.append(ModelRejection(key, reasons))
                continue
            ranked.append(_rank_candidate(descriptor, preferences, index))
        if not ranked:
            raise ModelSelectionError("required_capability_unsatisfied")
        selected = max(
            ranked,
            key=lambda item: (
                item.optional_score,
                item.descriptor.priority,
                -item.allowed_index,
            ),
        )
        return SelectionDecision(
            selected.descriptor, tuple(rejections), selected.notices
        )


def _missing_requirements(
    descriptor: ModelDescriptor,
    required: frozenset[CapabilityRequirement],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for requirement in sorted(required, key=lambda item: item.capability.value):
        value = descriptor.capabilities.get(requirement.capability, False)
        satisfied = (
            value >= requirement.minimum
            if requirement.minimum is not None and isinstance(value, int)
            else value is True
        )
        if not satisfied:
            reasons.append(
                f"missing_required_capability:{requirement.capability.value}"
            )
    return tuple(reasons)


def _rank_candidate(
    descriptor: ModelDescriptor,
    preferences: tuple[ModelPreference, ...],
    allowed_index: int,
) -> _RankedCandidate:
    score = 0
    notices: list[str] = []
    for preference in preferences:
        supported = descriptor.preference_values.get(preference.kind, frozenset())
        if preference.value in supported:
            score += 1
        else:
            notices.append(
                f"optional_preference_unsupported:{preference.kind.value}:{preference.value}"
            )
    return _RankedCandidate(descriptor, score, allowed_index, tuple(notices))


__all__ = [
    "CapabilityValue",
    "ModelDescriptor",
    "ModelRegistry",
    "ModelRejection",
    "ModelSelectionError",
    "ModelSelector",
    "SelectionDecision",
]
