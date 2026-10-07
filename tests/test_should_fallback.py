"""Tests for should_fallback logic in RealLLMClient."""

from __future__ import annotations

from llm.client import RealLLMClient, _ObservationFacts, _StageTiming
from llm.config import LLMProviderConfig
from llm.resolved_target import ResolvedModelTarget


def _make_config() -> LLMProviderConfig:
    return LLMProviderConfig(
        base_url="http://stub.local/v1",
        model="stub-model",
        api_key="test-key",
        timeout_seconds=10,
    )


def _make_target() -> ResolvedModelTarget:
    return ResolvedModelTarget(
        provider="openai_compatible",
        model="stub-model",
        base_url="http://stub.local/v1",
        api_mode="chat_completions",
        timeout_seconds=10,
        config_source="cli",
        credential_source="env",
        api_key_present=True,
    )


def _timing() -> _StageTiming:
    """构造一个固定起始时刻的阶段计时对象。

    作者：LKX
    时间：2026-08-31 14:09:39
    传参：无
    返回：_StageTiming；本文件只断言 fallback 与选型字段，不断言耗时
    """
    return _StageTiming(started_at="2026-05-19T00:00:00+00:00")


def _facts(*, success: bool, error_category: str | None) -> _ObservationFacts:
    """构造一次调用的观测事实，只给出成败与错误分类。

    作者：LKX
    时间：2026-08-31 14:09:39
    传参：success 为本轮是否成功；error_category 为失败分类，成功时传 None
    返回：_ObservationFacts；attempt_count 与用量走默认值
    """
    return _ObservationFacts(success=success, error_category=error_category)


class TestUpdateFallbackState:
    def test_success_clears_counters_no_fallback(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        client._consecutive_error_counts["rate_limited"] = 5
        should_fallback, reason = client._update_fallback_state(True, None)
        assert should_fallback is False
        assert reason == ""
        assert client._consecutive_error_counts == {}

    def test_auth_triggers_immediate_fallback(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        should_fallback, reason = client._update_fallback_state(False, "auth")
        assert should_fallback is True
        assert reason == "auth"

    def test_billing_triggers_immediate_fallback(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        should_fallback, reason = client._update_fallback_state(False, "billing")
        assert should_fallback is True
        assert reason == "billing"

    def test_model_not_found_triggers_immediate_fallback(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        should_fallback, reason = client._update_fallback_state(
            False, "model_not_found"
        )
        assert should_fallback is True
        assert reason == "model_not_found"

    def test_missing_config_triggers_immediate_fallback(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        should_fallback, reason = client._update_fallback_state(False, "missing_config")
        assert should_fallback is True
        assert reason == "missing_config"

    def test_rate_limited_x2_does_not_trigger(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        client._update_fallback_state(False, "rate_limited")
        should_fallback, reason = client._update_fallback_state(False, "rate_limited")
        assert should_fallback is False
        assert reason == ""

    def test_rate_limited_x3_triggers_fallback(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        client._update_fallback_state(False, "rate_limited")
        client._update_fallback_state(False, "rate_limited")
        should_fallback, reason = client._update_fallback_state(False, "rate_limited")
        assert should_fallback is True
        assert "rate_limited" in reason
        assert "3" in reason

    def test_overloaded_x3_triggers_fallback(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        for _ in range(2):
            client._update_fallback_state(False, "overloaded")
        should_fallback, reason = client._update_fallback_state(False, "overloaded")
        assert should_fallback is True
        assert "overloaded" in reason

    def test_server_error_x3_triggers_fallback(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        for _ in range(2):
            client._update_fallback_state(False, "server_error")
        should_fallback, reason = client._update_fallback_state(False, "server_error")
        assert should_fallback is True

    def test_timeout_does_not_trigger(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        for _ in range(5):
            should_fallback, _ = client._update_fallback_state(False, "timeout")
            assert should_fallback is False

    def test_transport_error_does_not_trigger(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        for _ in range(5):
            should_fallback, _ = client._update_fallback_state(False, "transport_error")
            assert should_fallback is False

    def test_success_after_failures_resets_counter(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        client._update_fallback_state(False, "rate_limited")
        client._update_fallback_state(False, "rate_limited")
        client._update_fallback_state(True, None)
        should_fallback, _ = client._update_fallback_state(False, "rate_limited")
        assert should_fallback is False

    def test_empty_category_no_fallback(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        should_fallback, reason = client._update_fallback_state(False, None)
        assert should_fallback is False
        assert reason == ""


class TestObservationFallbackFields:
    def test_observation_includes_fallback_fields_on_success(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        obs = client._observation(
            "plan",
            _timing(),
            _facts(success=True, error_category=None),
        )
        assert obs.should_fallback is False
        assert obs.fallback_reason == ""
        assert obs.provider == "openai_compatible"
        assert obs.config_source == "cli"
        assert obs.credential_source == "env"
        assert obs.base_url_host == "http://stub.local"

    def test_observation_records_fallback_reason_on_auth_error(self):
        client = RealLLMClient(config=_make_config(), resolved_target=_make_target())
        obs = client._observation(
            "plan",
            _timing(),
            _facts(success=False, error_category="auth"),
        )
        assert obs.should_fallback is True
        assert obs.fallback_reason == "auth"

    def test_observation_without_resolved_target_uses_defaults(self):
        client = RealLLMClient(config=_make_config(), resolved_target=None)
        obs = client._observation(
            "plan",
            _timing(),
            _facts(success=True, error_category=None),
        )
        assert obs.provider == "openai_compatible"
        assert obs.config_source == ""
        assert obs.credential_source == ""
        assert obs.base_url_host == ""
