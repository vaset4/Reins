from __future__ import annotations

from types import SimpleNamespace

from app.repl.model_commands import handle_model_command
from llm.resolved_target import ResolvedModelTarget


def test_model_set_exposes_pending_restart_contract(monkeypatch) -> None:
    saved: dict[str, object] = {}

    def fake_save(updates: dict[str, object]) -> None:
        saved.update(updates)

    monkeypatch.setattr("app.repl.model_commands.save_user_config", fake_save)

    result = handle_model_command("set model=next-model", object())

    assert saved == {"model": "next-model"}
    assert result.model_config_pending_restart is True
    assert result.model_config_applied_to_active_client is False
    assert result.message is not None
    assert "current REPL client is unchanged" in result.message


def test_model_profile_use_exposes_pending_restart_contract(monkeypatch) -> None:
    switched: list[str] = []
    monkeypatch.setattr(
        "app.repl.model_commands.switch_active_model_profile",
        switched.append,
    )

    result = handle_model_command("profile use fast", object())

    assert switched == ["fast"]
    assert result.model_config_pending_restart is True
    assert result.model_config_applied_to_active_client is False
    assert result.message is not None
    assert "Pending restart" in result.message


def _model_show_message(target: ResolvedModelTarget) -> str:
    ctx = SimpleNamespace(llm_client=SimpleNamespace(resolved_target=target))
    result = handle_model_command("show", ctx)
    assert result.message is not None
    return result.message


def _target(*, context_window: int, defaulted: bool) -> ResolvedModelTarget:
    return ResolvedModelTarget(
        provider="openai",
        model="gpt-x",
        base_url="https://api.openai.com/v1",
        api_mode="chat_completions",
        timeout_seconds=30.0,
        config_source="builtin_default" if defaulted else "saved_config",
        credential_source="none",
        api_key_present=False,
        context_window=context_window,
        context_window_defaulted=defaulted,
    )


def test_model_show_marks_defaulted_context_window() -> None:
    message = _model_show_message(_target(context_window=30000, defaulted=True))
    assert "30000 (defaulted)" in message


def test_model_show_omits_marker_when_context_window_configured() -> None:
    message = _model_show_message(_target(context_window=200000, defaulted=False))
    assert "200000" in message
    assert "(defaulted)" not in message
