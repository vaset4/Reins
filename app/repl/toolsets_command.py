from __future__ import annotations

from llm.toolset_policy import known_toolset_names
from runtime.session_state import SessionState, SessionStateStore
from tasks.ids import utc_now

# 默认态的真实行为：不做任何收窄，只减掉 opt-in 那组
_DEFAULT_DESCRIPTION = "all tools except the opt-in group"

# enable 的语义是"收窄到恰好这些"而不是"在默认基础上追加"，所以想在保留全部工具的
# 同时拿到密钥/剪贴板，唯一走得通的姿势是 enable full。不写明的话用户会敲
# enable secret，然后发现文件读不了、命令跑不了
_OPT_IN_HINT = (
    "opt-in group (secret / clipboard / redact / vision) is off by default; "
    "use /toolsets enable full to include it, or /toolsets enable opt_in_only for those alone"
)


def handle_toolsets(args: str, ctx: object) -> object:
    from app.repl.slash_commands import SlashCommandResult

    sub, values = _split_args(args)
    if sub in {"", "show"}:
        return SlashCommandResult(message=_render_show(ctx))
    if sub == "reset":
        _save(ctx, enabled=None, disabled=None)
        return SlashCommandResult(message=f"Toolsets reset: {_DEFAULT_DESCRIPTION}.")
    if sub in {"enable", "disable"}:
        names = _parse_names(values)
        if not names:
            return SlashCommandResult(message=f"Usage: /toolsets {sub} <name[,name]>")
        unknown = _unknown_names(ctx, names)
        if unknown:
            return SlashCommandResult(
                message=f"Unknown toolset(s): {', '.join(unknown)}"
            )
        _apply_update(ctx, sub, names)
        return SlashCommandResult(message=_render_show(ctx))
    return SlashCommandResult(
        message="Usage: /toolsets show | /toolsets enable <names> | "
        "/toolsets disable <names> | /toolsets reset"
    )


def _split_args(args: str) -> tuple[str, str]:
    sub, _, rest = args.strip().partition(" ")
    return sub.lower(), rest.strip()


def _parse_names(value: str) -> list[str]:
    return [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]


def _unknown_names(ctx: object, names: list[str]) -> list[str]:
    registry = getattr(ctx, "registry")
    known = known_toolset_names(registry)
    return sorted(name for name in names if name not in known)


def _apply_update(ctx: object, sub: str, names: list[str]) -> None:
    state = _state(ctx)
    enabled = names if sub == "enable" else state.toolsets_enabled
    disabled = names if sub == "disable" else state.toolsets_disabled
    _save(ctx, enabled=enabled, disabled=disabled)


def _save(
    ctx: object,
    *,
    enabled: list[str] | None,
    disabled: list[str] | None,
) -> SessionState:
    state = _state(ctx)
    state.toolsets_enabled = list(enabled) if enabled is not None else None
    state.toolsets_disabled = list(disabled) if disabled is not None else None
    state.toolsets_updated_at = utc_now()
    state.updated_at = state.toolsets_updated_at
    _sync_repl_state(ctx, state)
    SessionStateStore(getattr(ctx, "data_root")).save(state)
    return state


def _state(ctx: object) -> SessionState:
    repl_state = getattr(ctx, "repl_state")
    store = SessionStateStore(getattr(ctx, "data_root"))
    session_id = str(getattr(repl_state, "session_id"))
    state = store.load(session_id) or SessionState(session_id=session_id)
    if state.toolsets_enabled is None:
        state.toolsets_enabled = getattr(repl_state, "toolsets_enabled", None)
    if state.toolsets_disabled is None:
        state.toolsets_disabled = getattr(repl_state, "toolsets_disabled", None)
    return state


def _sync_repl_state(ctx: object, state: SessionState) -> None:
    repl_state = getattr(ctx, "repl_state")
    repl_state.toolsets_enabled = state.toolsets_enabled
    repl_state.toolsets_disabled = state.toolsets_disabled


def _render_show(ctx: object) -> str:
    state = _state(ctx)
    return (
        "Toolset policy\n"
        f"enabled : {_format_names(state.toolsets_enabled, f'({_DEFAULT_DESCRIPTION})')}\n"
        f"disabled: {_format_names(state.toolsets_disabled, '(none)')}\n"
        f"{_OPT_IN_HINT}"
    )


def _format_names(value: list[str] | None, fallback: str) -> str:
    if value is None:
        return fallback
    return ", ".join(value) if value else "(empty)"
