"""Single owner for run-observability secret redaction (PRD R1/R3).

``run_evidence`` (raw evidence) and ``run_facts`` (queryable summaries) both
write data that humans and replay tooling read back, so secrets must be masked
once, here, instead of relying on every reader to re-mask. Two stores used to
keep two divergent strategies — evidence did value-pattern matching, facts did
key-based masking — which meant a secret one store caught the other could leak.

This module merges both layers behind one pattern source:

- key-based: ``looks_secret_key(key)`` -> the whole value becomes ``<redacted>``.
- value-pattern: ``redact_secret_text(text)`` masks secret-shaped substrings
  (``sk-...`` API keys, ``Bearer ...`` tokens) embedded in otherwise innocuous
  strings.

``redact_value`` applies both layers recursively and is the entry point both
stores reuse. R3 (only stricter, never looser): callers may add their own length
truncation on top, but secret protection is never relaxed below this owner.

``tools.exec_channel.redact_secrets`` is a deliberately separate owner (C16, exec
channel). It keeps its OWN marker set (``_SECRET_ENV_MARKERS``, matched against
upper-cased env-var NAMES) and does not import this module's constants — the two
are independent owners by design, not a shared source. They are kept aligned by
cross-reference and manual review, not by literal constant sharing: changing
either side requires checking the other so neither relaxes below the other (R3).
The two cover different surfaces (exec masks env-var names; this owner masks
mapping keys + value patterns), so neither is a strict superset of the other.

``memory.safety_scan`` is the THIRD owner, on the memory/skill write path. Same
arrangement: independent constants, aligned by review, neither below the other.
It is a *blocking* gate (unsafe content is refused outright and, at recall time,
excluded from the prompt) whereas this module *masks and continues*, so their
rules are intentionally not merged — a shape worth masking here is not
automatically a shape worth refusing a whole memory over. The direction that
must hold is that safety_scan never falls below this owner on real credential
shapes; ``tests/test_secret_redaction.py`` asserts it instead of trusting review.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

REDACTED = "<redacted>"

# Secret-name markers. Single source shared by evidence and facts (R1: the two
# stores must not diverge). Exec-channel keeps its own markers (see module
# docstring) and is aligned by review, not by importing these.
SECRET_KEY_PARTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
)
SECRET_TOKEN_PATTERNS: tuple[str, ...] = (
    "access_token",
    "refresh_token",
    "bearer_token",
    "auth_token",
    "api_token",
)
# Exact key names that are always secret regardless of substring rules.
_SECRET_KEY_EXACT: frozenset[str] = frozenset({"authorization", "cookie", "set-cookie"})
# Secret-shaped values that may appear inside otherwise innocuous strings.
#
# NOTE on the boundary asymmetry below: the two original rules have no
# left-boundary assertion, so `risk-assessment-review` masks to `ri<redacted>`.
# That over-masks (fails safe) but corrupts the readable records this module
# exists to serve. It is left as-is deliberately: adding a boundary RELAXES a
# live redaction rule, and this module's own contract is "only stricter, never
# looser" (see R3 in the docstring), so it needs explicit sign-off rather than a
# drive-by fix. The vendor rules added below all carry the boundary, so the
# defect is not spreading. `memory/safety_scan.py` — a blocking gate, where a
# false positive hides a memory permanently — uses bounded copies throughout.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?<![A-Za-z0-9])gh[porsu]_[A-Za-z0-9]{36,}"),
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"),
    re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{35}"),
)


def looks_secret_key(key: str) -> bool:
    """True when a mapping key names a secret whose value must be fully masked."""
    lowered = key.lower()
    if lowered in _SECRET_KEY_EXACT:
        return True
    if any(pat in lowered for pat in SECRET_TOKEN_PATTERNS):
        return True
    return any(part in lowered for part in SECRET_KEY_PARTS)


def redact_secret_text(value: str) -> str:
    """Mask secret-shaped substrings (value-pattern layer)."""
    redacted = value
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(_replace_secret_match, redacted)
    return redacted


def redact_value(value: object, *, key_hint: str = "") -> Any:
    """Apply both redaction layers recursively.

    ``key_hint`` carries the enclosing mapping key so a secret-named key masks
    its whole value (key-based layer); strings are additionally scrubbed for
    secret-shaped substrings (value-pattern layer). Callers keep their own
    truncation policy; this owner only guarantees secret protection.
    """
    if key_hint and looks_secret_key(key_hint):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(key): redact_value(item, key_hint=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item, key_hint=key_hint) for item in value]
    if isinstance(value, str):
        return redact_secret_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_secret_text(str(value))


def _replace_secret_match(match: re.Match[str]) -> str:
    if match.lastindex:
        return f"{match.group(1)}{REDACTED}"
    return REDACTED


__all__ = [
    "REDACTED",
    "SECRET_KEY_PARTS",
    "SECRET_TOKEN_PATTERNS",
    "looks_secret_key",
    "redact_secret_text",
    "redact_value",
]
