from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 直连探针与生产发送路径共用同一个客户端身份，避免两处 User-Agent 字面量各自漂移
from llm.provider_connection import DEFAULT_USER_AGENT  # noqa: E402

DEFAULT_MESSAGE = "Reply with OK."
DEFAULT_TIMEOUT_SECONDS = 30.0

if TYPE_CHECKING:
    from llm.resolved_target import ResolvedModelTarget


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ok: bool
    status_code: int | None
    body: str
    error: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        target = _resolve_target(args)
    except Exception as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    _print_target(target)
    missing = _missing_target_reason(target)
    if missing:
        print(f"local_check: failed ({missing})")
        return 2
    result = _post_chat_completion(
        target,
        message=args.message,
        user_agent=args.user_agent,
    )
    _print_result(result, show_body=args.show_body)
    return 0 if result.ok else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the active Reins OpenAI-compatible API profile."
    )
    parser.add_argument("--base-url", dest="base_url", help="Override base URL.")
    parser.add_argument("--model", help="Override model.")
    parser.add_argument("--api-key-env", help="Read API key from this env var.")
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header. Pass an empty string to omit it.",
    )
    parser.add_argument("--show-body", action="store_true", help="Print full body.")
    return parser.parse_args(argv)


def _resolve_target(
    args: argparse.Namespace,
) -> ResolvedModelTarget:
    profiles_module = importlib.import_module("llm.profiles")
    resolved = importlib.import_module("llm.resolved_target")
    secrets_store = importlib.import_module("reins_secrets.store")
    cli_overrides = _cli_overrides(args)
    if args.api_key_env:
        cli_overrides["api_key"] = _env_value(args.api_key_env)
    profiles = profiles_module.load_model_profiles()
    active_profile = profiles.active_profile
    target = resolved.resolve_model_target(
        cli_overrides=cli_overrides,
        file_defaults=active_profile.as_config() if active_profile else None,
        secrets_vault=secrets_store.SecretsVault(),
    )
    return cast(
        "ResolvedModelTarget",
        replace(
            target,
            profile_name=profiles.active,
            credential_name=active_profile.credential if active_profile else "",
        ),
    )


def _cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "base_url": args.base_url,
        "model": args.model,
        "timeout_seconds": args.timeout_seconds,
    }


def _env_value(name: str) -> str:
    import os

    value = os.getenv(name)
    if not value:
        raise ValueError(f"environment variable is empty: {name}")
    return value


def _missing_target_reason(target: ResolvedModelTarget) -> str:
    resolved = importlib.import_module("llm.resolved_target")
    if target.unsupported_reason:
        return target.unsupported_reason
    if not target.base_url.strip():
        return "base_url is missing"
    if not target.model.strip():
        return "model is missing"
    if resolved.requires_api_key(target.base_url) and not target.api_key_present:
        return "api_key is missing for remote HTTPS endpoint"
    return ""


def _post_chat_completion(
    target: ResolvedModelTarget,
    *,
    message: str,
    user_agent: str,
) -> ProbeResult:
    endpoint = f"{target.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": target.model,
        "messages": [{"role": "user", "content": message}],
    }
    headers = _headers(target, user_agent)
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=target.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            return ProbeResult(True, response.status, body, "")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        return ProbeResult(False, exc.code, body, f"HTTP {exc.code}")
    except (TimeoutError, URLError, OSError) as exc:
        return ProbeResult(False, None, "", str(exc))


def _headers(target: ResolvedModelTarget, user_agent: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    if user_agent.strip():
        headers["User-Agent"] = user_agent.strip()
    return headers


def _print_target(target: ResolvedModelTarget) -> None:
    resolved = importlib.import_module("llm.resolved_target")
    print(f"profile       : {target.profile_name or '(none)'}")
    print(f"provider      : {target.provider}")
    print(f"model         : {target.model or '(missing)'}")
    print(f"base url      : {target.base_url or '(missing)'}")
    print(f"endpoint host : {resolved.sanitize_base_url(target.base_url)}")
    print(f"config source : {target.config_source}")
    print(f"credential    : {_credential_summary(target)}")


def _credential_summary(target: ResolvedModelTarget) -> str:
    presence = "present" if target.api_key_present else "absent"
    if target.credential_name:
        return f"{presence} ({target.credential_source}:{target.credential_name})"
    return f"{presence} ({target.credential_source or 'none'})"


def _print_result(result: ProbeResult, *, show_body: bool) -> None:
    print(f"ok            : {result.ok}")
    if result.status_code is not None:
        print(f"http status   : {result.status_code}")
    if result.error:
        print(f"error         : {result.error}")
    if result.body:
        label = "body" if show_body else "body preview"
        print(f"{label:<14}: {_body(result.body, show_body=show_body)}")


def _body(value: str, *, show_body: bool) -> str:
    if show_body or len(value) <= 500:
        return value
    return value[:500] + "\n...[truncated]..."


if __name__ == "__main__":
    raise SystemExit(main())
