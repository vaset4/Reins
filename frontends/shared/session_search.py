"""会话搜索通过实体 Store 读取统一数据库，不依赖物理会话目录。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from frontends.shared.session_search_fields import fact_search_fields
from runtime.run_facts import FactReadWarning, RunFactStore
from runtime.session_state import SessionState, SessionStateStore

# --- public dataclasses --------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryWarning:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    session_id: str
    last_run_id: str | None
    status: str | None
    focus_task_id: str | None
    summary_snippet: str
    hit_source: str
    updated_at: str
    trigger_badge: str | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: list[SearchHit]
    warnings: list[QueryWarning]
    scanned_count: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class PreciseResult:
    """ID-based lookup outcome.

    `kind` is "session" or "run" so callers can render the right banner.
    `hit` carries the rendered SearchHit; `payload` is the raw object
    (SessionState or list[fact dict]) for callers that want details. `warnings`
    carries tolerant badge diagnostics without weakening precise run reads.
    """

    kind: str
    hit: SearchHit
    payload: Any
    warnings: tuple[QueryWarning, ...] = ()


# --- constants ------------------------------------------------------------

MIN_TOKEN_LENGTH = 2
SCAN_LIMIT = 50
SCAN_TIMEOUT_SECONDS = 2.0
DEEP_SCAN_RUNS_PER_SESSION = 3
DEEP_SCAN_LINES_PER_RUN = 50
CORRUPT_RUN_FACTS_CODE = "corrupt_run_facts"
SNIPPET_BEFORE = 20
SNIPPET_AFTER = 60
SNIPPET_TOTAL_LIMIT = 120

_SESSION_ID_PATTERN = re.compile(r"^session-[0-9a-f]{32}$")
_RUN_ID_PATTERN = re.compile(r"^run-[0-9a-f]{32}$")

_SESSION_FIELDS_FOR_SEARCH = (
    "session_id",
    "summary",
    "last_run_status",
    "last_run_event",
    "focus_task_id",
    "compatibility_task_id",
)
# --- backend protocol -----------------------------------------------------


class SearchBackend(Protocol):
    def search_sessions(self, query: str, limit: int) -> list[SearchHit]: ...


# --- file-scan backend ----------------------------------------------------


class FileScanBackend:
    """Store-backed session search.

    Reads canonical session and run entities through their stores.
    """

    def __init__(
        self,
        data_root: Path,
        *,
        scan_limit: int = SCAN_LIMIT,
        timeout_seconds: float = SCAN_TIMEOUT_SECONDS,
    ) -> None:
        self._data_root = data_root
        self._session_states = SessionStateStore(data_root)
        self._run_facts = RunFactStore(data_root)
        self._scan_limit = scan_limit
        self._timeout_seconds = timeout_seconds

    def search_sessions(self, query: str, limit: int) -> list[SearchHit]:
        # Kept for protocol compatibility; the richer entry point is `scan`.
        result = self.scan(query, limit=limit, deep_scan=False)
        return result.hits

    def scan(
        self,
        query: str,
        *,
        limit: int,
        deep_scan: bool,
    ) -> SearchResult:
        warnings: list[QueryWarning] = []
        tokens, token_warnings = _tokenize(query)
        warnings.extend(token_warnings)
        if not tokens:
            return SearchResult(
                hits=[], warnings=warnings, scanned_count=0, truncated=False
            )

        sessions = self._session_states.list_recent(limit=self._scan_limit)
        scored, scanned, truncated, scan_warnings = self._collect_scored_hits(
            sessions, tokens, deep_scan=deep_scan
        )
        warnings.extend(scan_warnings)

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        hits = [hit for _, hit in scored[:limit]]
        return SearchResult(
            hits=hits, warnings=warnings, scanned_count=scanned, truncated=truncated
        )

    def _collect_scored_hits(
        self,
        sessions: list[SessionState],
        tokens: list[str],
        *,
        deep_scan: bool,
    ) -> tuple[list[tuple[int, SearchHit]], int, bool, list[QueryWarning]]:
        scored: list[tuple[int, SearchHit]] = []
        scanned = 0
        truncated = False
        warnings: list[QueryWarning] = []
        deadline = time.monotonic() + self._timeout_seconds
        for session in sessions:
            if time.monotonic() > deadline:
                truncated = True
                warnings.append(
                    QueryWarning(
                        code="scan_timeout",
                        message="搜索超过 2 秒，结果可能不完整。",
                    )
                )
                break
            scanned += 1
            scored_hit, session_warnings = self._score_session(
                session, tokens, deep_scan=deep_scan
            )
            warnings.extend(session_warnings)
            if scored_hit is not None:
                scored.append(scored_hit)
        if scanned >= self._scan_limit and len(sessions) >= self._scan_limit:
            truncated = True
            warnings.append(
                QueryWarning(
                    code="scan_limit_reached",
                    message=(
                        f"已达扫描上限 {self._scan_limit} 个会话，结果可能不完整。"
                    ),
                )
            )
        return scored, scanned, truncated, warnings

    # ---- internals ------------------------------------------------------

    def _score_session(
        self,
        session: SessionState,
        tokens: list[str],
        *,
        deep_scan: bool,
    ) -> tuple[tuple[int, SearchHit] | None, list[QueryWarning]]:
        field_values = [
            (name, str(getattr(session, name) or ""))
            for name in _SESSION_FIELDS_FOR_SEARCH
        ]
        hit_field_count, matched_token = _count_matching_fields(field_values, tokens)
        if hit_field_count > 0:
            snippet_source, snippet_token = _pick_snippet_source(
                field_values, tokens, matched_token
            )
            snippet = _make_snippet(snippet_source, snippet_token)
            badge, warnings = self._read_trigger_badge(session.session_id)
            hit = _build_hit(
                session,
                snippet=snippet,
                hit_source="session_summary",
                badge=badge,
            )
            return (hit_field_count, hit), warnings

        if not deep_scan:
            return None, []

        deep_match, warnings = self._deep_scan_session(session.session_id, tokens)
        if deep_match is None:
            return None, warnings

        snippet, _, fact_score = deep_match
        badge, badge_warnings = self._read_trigger_badge(session.session_id)
        hit = _build_hit(
            session,
            snippet=snippet,
            hit_source="run_fact",
            badge=badge,
        )
        return (fact_score, hit), _merge_query_warnings(warnings, badge_warnings)

    def _deep_scan_session(
        self, session_id: str, tokens: list[str]
    ) -> tuple[tuple[str, str, int] | None, list[QueryWarning]]:
        """用 tolerant facts 查找 session 命中并保留损坏 warning

        参数：session_id 为会话身份；tokens 为已校验搜索词
        返回：首个 deep match 与去重后的结构化 warning
        """
        run_list = self._run_facts.list_runs_for_session_tolerant(
            session_id, limit=DEEP_SCAN_RUNS_PER_SESSION
        )
        warnings = _map_fact_warnings(run_list.warnings)
        for run in run_list.runs:
            read = self._run_facts.read_run_tolerant(run.run_id)
            warnings = _merge_query_warnings(
                warnings, _map_fact_warnings(read.warnings)
            )
            for fact in read.facts[-DEEP_SCAN_LINES_PER_RUN:]:
                field_values = fact_search_fields(fact)
                hit_field_count, matched_token = _count_matching_fields(
                    field_values, tokens
                )
                if hit_field_count == 0:
                    continue
                snippet_source, snippet_token = _pick_snippet_source(
                    field_values, tokens, matched_token
                )
                snippet = _make_snippet(snippet_source, snippet_token)
                return (snippet, snippet_source, hit_field_count), warnings
        return None, warnings

    def _read_trigger_badge(
        self, session_id: str
    ) -> tuple[str | None, list[QueryWarning]]:
        """从 tolerant facts 读取 trigger badge 并显式返回损坏 warning

        参数：session_id 为会话身份
        返回：可信 trigger badge 与去重后的 QueryWarning
        """
        run_list = self._run_facts.list_runs_for_session_tolerant(session_id, limit=1)
        warnings = _map_fact_warnings(run_list.warnings)
        if not run_list.runs:
            return None, warnings
        read = self._run_facts.read_run_tolerant(run_list.runs[0].run_id)
        warnings = _merge_query_warnings(warnings, _map_fact_warnings(read.warnings))
        for fact in read.facts:
            if fact.get("event") == "run:start":
                trigger = fact.get("trigger")
                return (str(trigger) if trigger is not None else None), warnings
        return None, warnings


# --- service --------------------------------------------------------------


class SessionSearchService:
    """User-facing search service.

    Exposes ID resolution (`resolve_id`) and free-text search (`search`).
    Reads retained session and run facts through their source-owning Stores.
    The TUI directory's full-text index is owned by SessionMessageStore.
    """

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._session_states = SessionStateStore(data_root)
        self._run_facts = RunFactStore(data_root)
        self._backend = FileScanBackend(data_root)

    def resolve_id(self, raw_input: str) -> PreciseResult | None:
        text = raw_input.strip()
        if not text:
            return None
        if _SESSION_ID_PATTERN.match(text):
            session = self._session_states.load(text)
            if session is None:
                return None
            badge, warnings = self._backend._read_trigger_badge(session.session_id)
            hit = _build_hit(
                session,
                snippet=_truncate_snippet(session.summary or ""),
                hit_source="session_summary",
                badge=badge,
            )
            return PreciseResult(
                kind="session", hit=hit, payload=session, warnings=tuple(warnings)
            )
        if _RUN_ID_PATTERN.match(text):
            run = self._run_facts.find_run(text)
            if run is None:
                return None
            facts = self._run_facts.read_run(text)
            if not facts:
                return None
            session_id = str(facts[0].get("session_id", "")) or ""
            session = self._session_states.load(session_id) if session_id else None
            if session is None and session_id:
                session = SessionState(session_id=session_id, last_run_id=text)
            if session is None:
                return None
            badge, warnings = self._backend._read_trigger_badge(session.session_id)
            snippet = _facts_snippet(facts)
            hit = _build_hit(
                session,
                snippet=snippet,
                hit_source="run_fact",
                badge=badge,
                override_run_id=text,
            )
            return PreciseResult(
                kind="run", hit=hit, payload=facts, warnings=tuple(warnings)
            )
        return None

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        deep_scan: bool = False,
    ) -> SearchResult:
        text = query.strip()
        if not text:
            return SearchResult(hits=[], warnings=[], scanned_count=0, truncated=False)

        if _SESSION_ID_PATTERN.match(text) or _RUN_ID_PATTERN.match(text):
            precise = self.resolve_id(text)
            if precise is not None:
                return SearchResult(
                    hits=[precise.hit],
                    warnings=list(precise.warnings),
                    scanned_count=1,
                    truncated=False,
                )
            warnings = [
                QueryWarning(
                    code="precise_lookup_miss",
                    message="未找到精确 ID，已按关键词搜索。",
                )
            ]
            base = self._backend.scan(text, limit=limit, deep_scan=deep_scan)
            return SearchResult(
                hits=base.hits,
                warnings=warnings + list(base.warnings),
                scanned_count=base.scanned_count,
                truncated=base.truncated,
            )

        return self._backend.scan(text, limit=limit, deep_scan=deep_scan)


# --- helpers --------------------------------------------------------------


def _tokenize(query: str) -> tuple[list[str], list[QueryWarning]]:
    raw_tokens = [token for token in query.lower().split() if token]
    warnings: list[QueryWarning] = []
    valid: list[str] = []
    has_short = False
    for token in raw_tokens:
        if len(token) < MIN_TOKEN_LENGTH:
            has_short = True
            continue
        valid.append(token)
    if has_short:
        warnings.append(
            QueryWarning(
                code="token_too_short",
                message="搜索词需至少 2 字符，已忽略过短的输入。",
            )
        )
    return valid, warnings


def _count_matching_fields(
    field_values: list[tuple[str, str]],
    tokens: list[str],
) -> tuple[int, str | None]:
    """Return (hit_field_count, first_matched_token).

    AND across tokens: every token must hit at least one field. The
    return value counts how many fields had any token match (used for
    sort ranking).
    """
    lowered = [(name, value.lower()) for name, value in field_values]
    for token in tokens:
        if not any(token in value for _, value in lowered):
            return 0, None
    matched_fields: set[str] = set()
    first_token: str | None = None
    for token in tokens:
        for name, value in lowered:
            if token in value:
                matched_fields.add(name)
                if first_token is None:
                    first_token = token
    return len(matched_fields), first_token


def _pick_snippet_source(
    field_values: list[tuple[str, str]],
    tokens: list[str],
    preferred_token: str | None,
) -> tuple[str, str]:
    """Return (source_text, token_used_for_snippet)."""
    token = preferred_token or (tokens[0] if tokens else "")
    for _, value in field_values:
        if token and token in value.lower() and value:
            return value, token
    # No usable source — fall back to first non-empty field.
    for _, value in field_values:
        if value:
            return value, token
    return "", token


def _make_snippet(source: str, token: str) -> str:
    if not source:
        return ""
    if not token:
        return _truncate_snippet(source)
    lower = source.lower()
    index = lower.find(token.lower())
    if index < 0:
        return _truncate_snippet(source)
    start = max(0, index - SNIPPET_BEFORE)
    end = min(len(source), index + len(token) + SNIPPET_AFTER)
    snippet = source[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(source):
        snippet = snippet + "…"
    return _truncate_snippet(snippet)


def _truncate_snippet(text: str) -> str:
    if len(text) <= SNIPPET_TOTAL_LIMIT:
        return text
    return text[: SNIPPET_TOTAL_LIMIT - 1] + "…"


def _facts_snippet(facts: list[dict[str, Any]]) -> str:
    """Render a short snippet describing the latest fact for a run."""
    if not facts:
        return ""
    last = facts[-1]
    event = str(last.get("event", "")) or "fact"
    status = str(last.get("status", "")) or ""
    parts = [event]
    if status:
        parts.append(f"status={status}")
    tool = last.get("tool")
    if isinstance(tool, Mapping):
        name = str(tool.get("name", "")) or ""
        if name:
            parts.append(f"tool={name}")
    return _truncate_snippet(" | ".join(parts))


def _map_fact_warnings(
    warnings: tuple[FactReadWarning, ...],
) -> list[QueryWarning]:
    """把 runtime warning 映射成不含绝对路径的查询边界消息

    参数：warnings 为 RunFactStore 返回的损坏行
    返回：code 固定且含 session/run/line/reason 的 QueryWarning
    """
    return [
        QueryWarning(
            code=CORRUPT_RUN_FACTS_CODE,
            message=(
                f"session={warning.session_id}; run={warning.run_id}; "
                f"line={warning.line_number}; reason={warning.message}"
            ),
        )
        for warning in warnings
    ]


def _merge_query_warnings(
    current: list[QueryWarning], extra: list[QueryWarning]
) -> list[QueryWarning]:
    """合并重复读取同一坏行产生的 warning，并保持首次发现顺序

    参数：current 为已有 warning；extra 为新读取 warning
    返回：按 QueryWarning 完整值去重的新列表
    """
    merged = list(current)
    seen = set(current)
    for warning in extra:
        if warning in seen:
            continue
        merged.append(warning)
        seen.add(warning)
    return merged


def _build_hit(
    session: SessionState,
    *,
    snippet: str,
    hit_source: str,
    badge: str | None,
    override_run_id: str | None = None,
) -> SearchHit:
    return SearchHit(
        session_id=session.session_id,
        last_run_id=override_run_id or session.last_run_id or None,
        status=session.last_run_status or None,
        focus_task_id=session.focus_task_id,
        summary_snippet=snippet,
        hit_source=hit_source,
        updated_at=session.updated_at,
        trigger_badge=badge,
    )


__all__ = [
    "FileScanBackend",
    "PreciseResult",
    "QueryWarning",
    "SearchBackend",
    "SearchHit",
    "SearchResult",
    "SessionSearchService",
]
