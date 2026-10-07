from __future__ import annotations

from typing import TYPE_CHECKING

from llm.client import RealLLMClient
from llm.doctor import ModelDoctorReport, run_model_doctor
from llm.profiles import (
    load_model_profiles,
    switch_active_model_profile,
)
from llm.resolved_target import ResolvedModelTarget, sanitize_base_url

if TYPE_CHECKING:
    from app.repl.slash_commands import SlashCommandResult


def handle_model_command(args: str, ctx: object) -> "SlashCommandResult":
    sub, _, rest = args.partition(" ")
    sub = sub.strip().lower()
    if not sub or sub == "show":
        return _result(_render_model_show(ctx))
    if sub == "profile":
        return _model_profile(rest.strip())
    if sub == "doctor":
        return _result(_render_doctor(ctx))
    return _result(_usage())


def _render_model_show(ctx: object) -> str:
    target = _resolved_target(ctx)
    if target is None:
        return "Model configuration not available."
    cred_label = "present" if target.api_key_present else "absent"
    credential = _credential_label(target, cred_label)
    profile = target.profile_name or "(none)"
    lines = [
        f"Profile    : {profile}",
        f"Provider   : {target.provider}",
        f"Model      : {target.model}",
        f"Reasoning  : {target.reasoning_effort or 'default'}",
        f"Endpoint   : {sanitize_base_url(target.base_url)}",
        f"Base URL   : {target.base_url}",
        f"API mode   : {target.api_mode}",
        f"Context    : {target.context_window}{' (defaulted)' if target.context_window_defaulted else ''}",
        f"Output cap : {target.output_token_limit}",
        f"Config src : {target.config_source}",
        f"Credential : {credential}",
    ]
    if not target.profile_name:
        lines.append(
            "Note       : no active profile; current config uses legacy sources."
        )
    return "\n".join(lines)


def _model_profile(args: str) -> "SlashCommandResult":
    sub, _, rest = args.partition(" ")
    sub = sub.strip().lower()
    if sub == "list":
        return _result(_render_profile_list())
    if sub == "show":
        return _result(_render_profile_show(rest.strip()))
    if sub == "use":
        return _profile_use(rest.strip())
    return _result(_profile_usage())


def _render_profile_list() -> str:
    try:
        config = load_model_profiles()
    except Exception as exc:
        return f"Failed to load profiles: {exc}"
    if not config.profiles:
        return "No model profiles configured."
    lines = []
    for name in sorted(config.profiles):
        profile = config.profiles[name]
        marker = "*" if name == config.active else " "
        endpoint = sanitize_base_url(profile.base_url)
        lines.append(f"{marker} {name}: {profile.provider} {profile.model} {endpoint}")
    return "\n".join(lines)


def _render_profile_show(name: str) -> str:
    try:
        config = load_model_profiles()
    except Exception as exc:
        return f"Failed to load profiles: {exc}"
    profile_name = name or config.active
    profile = config.profiles.get(profile_name)
    if profile is None:
        return f"Profile not found: {profile_name or '(none)'}"
    credential = profile.credential or "(none)"
    lines = [
        f"Profile    : {profile.name}",
        f"Active     : {profile.name == config.active}",
        f"Provider   : {profile.provider}",
        f"Model      : {profile.model}",
        f"Reasoning  : {profile.reasoning_effort or 'default'}",
        f"Endpoint   : {sanitize_base_url(profile.base_url)}",
        f"Base URL   : {profile.base_url}",
        f"API mode   : {profile.api_mode}",
        f"Context    : {profile.context_window}",
        f"Credential : {credential}",
    ]
    return "\n".join(lines)


def _profile_use(args: str) -> "SlashCommandResult":
    """同时保存配置与可选强度；传参：命令参数；返回：生效待加载或明确失败。"""
    name, _, rest = args.partition(" ")
    if not name:
        return _result(
            "Usage: /model profile use <provider:model> [reasoning_effort=value]"
        )
    effort = None
    if rest.strip():
        updates = _parse_updates(rest.strip())
        if isinstance(updates, str):
            return _result(updates)
        if set(updates) != {"reasoning_effort"}:
            return _result("Only reasoning_effort is allowed when selecting a profile.")
        effort = str(updates["reasoning_effort"])
    try:
        if effort is None:
            switch_active_model_profile(name)
        else:
            switch_active_model_profile(name, reasoning_effort=effort)
    except Exception as exc:
        return _result(f"Failed to switch profile: {exc}")
    return _pending_restart_result(
        f"Active profile: {name}\nPending restart: current REPL client is unchanged."
    )


def _render_doctor(ctx: object) -> str:
    target = _resolved_target(ctx)
    report = run_model_doctor(target)
    base_url = target.base_url if target is not None else "(unknown)"
    lines = [
        f"Status     : {'ok' if report.ok else 'failed'}",
        f"Profile    : {report.profile_name or '(none)'}",
        f"Provider   : {report.provider or '(unknown)'}",
        f"Model      : {report.model or '(unknown)'}",
        f"Endpoint   : {report.base_url_host or '(unknown)'}",
        f"Base URL   : {base_url}",
        f"Credential : {_report_credential(report)}",
    ]
    if report.error_category:
        lines.append(f"Category   : {report.error_category}")
    if report.status_code is not None:
        lines.append(f"HTTP       : {report.status_code}")
    lines.append(f"Message    : {report.message}")
    return "\n".join(lines)


def _resolved_target(ctx: object) -> ResolvedModelTarget | None:
    client = getattr(ctx, "llm_client", None)
    target = getattr(client, "resolved_target", None)
    if isinstance(client, RealLLMClient) and isinstance(target, ResolvedModelTarget):
        return target
    return target if isinstance(target, ResolvedModelTarget) else None


def _credential_label(target: ResolvedModelTarget, presence: str) -> str:
    source = f"source: {target.credential_source}"
    if target.credential_name:
        return f"{presence} ({source}, name: {target.credential_name})"
    return f"{presence} ({source})"


def _report_credential(report: ModelDoctorReport) -> str:
    presence = "present" if report.credential_present else "absent"
    if report.credential_name:
        return f"{presence} (source: {report.credential_source}, name: {report.credential_name})"
    return f"{presence} (source: {report.credential_source or 'none'})"


def _parse_updates(args: str) -> dict[str, object] | str:
    updates: dict[str, object] = {}
    for token in args.split():
        if "=" not in token:
            return f"Invalid format: {token} (expected key=value)"
        key, _, value = token.partition("=")
        updates[key.strip()] = value.strip()
    return updates


def _result(message: str) -> "SlashCommandResult":
    from app.repl.slash_commands import SlashCommandResult

    return SlashCommandResult(message=message)


def _pending_restart_result(message: str) -> "SlashCommandResult":
    from app.repl.slash_commands import SlashCommandResult

    return SlashCommandResult(
        message=message,
        model_config_pending_restart=True,
        model_config_applied_to_active_client=False,
    )


def _usage() -> str:
    return "Usage: /model show | /model profile ... | /model doctor"


def _profile_usage() -> str:
    return (
        "Usage: /model profile list | show [provider:model] | "
        "use <provider:model> [reasoning_effort=value]"
    )


__all__ = ["handle_model_command"]
