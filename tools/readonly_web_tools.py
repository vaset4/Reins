from __future__ import annotations

from html.parser import HTMLParser
from typing import Callable, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from runtime.types import RunToolsRequest, RunToolsResult

WebFetcher = Callable[[str], str]


class ReadOnlyWebToolExecutor:
    def __init__(
        self,
        *,
        fetcher: WebFetcher | None = None,
        max_chars: int = 4000,
        max_results: int = 5,
    ) -> None:
        self._fetcher = fetcher or _fetch_url
        self._max_chars = max_chars
        self._max_results = max_results

    def execute(self, request: RunToolsRequest) -> RunToolsResult:
        tool_name = request.tool_name or request.action

        try:
            if tool_name == "web_search":
                content = self._web_search(request)
            elif tool_name == "web_fetch":
                content = self._web_fetch(request)
            elif tool_name == "web_scan":
                content = self._web_scan(request)
            else:
                return RunToolsResult.error_result(
                    action=request.action,
                    tool_name=tool_name,
                    error=f"INVALID_REQUEST: unsupported readonly web tool: {tool_name}",
                    summary="invalid request",
                    target_scope=request.target_scope,
                )
        except Exception as exc:
            # 网页只读工具允许失败，但失败要回到统一结果骨架，
            # 不能把网络层异常直接漏回 runtime。
            return RunToolsResult.error_result(
                action=request.action,
                tool_name=tool_name,
                error=f"RUN_TOOLS_ERROR: {exc}",
                summary="tool execution failed",
                target_scope=request.target_scope,
            )

        return RunToolsResult.ok(
            action=request.action,
            tool_name=tool_name,
            content=content,
            summary="tool executed",
            target_scope=request.target_scope,
        )

    def _web_search(self, request: RunToolsRequest) -> str:
        query = str(request.arguments.get("query", "")).strip()
        if not query:
            raise ValueError("missing query")

        domain = str(request.arguments.get("domain", "")).strip()
        search_query = query
        if domain:
            search_query = f"site:{domain} {query}"

        url = f"https://duckduckgo.com/html/?q={quote_plus(search_query)}"
        html = self._fetcher(url)
        parsed = _HTMLSummaryParser()
        parsed.feed(html)

        lines: list[str] = []
        if parsed.title:
            lines.append(parsed.title)
        for href, text in parsed.links[: self._max_results]:
            lines.append(f"{text} -> {href}")
        if not lines:
            lines.append("No visible search results")
        return _truncate("\n".join(lines), self._max_chars)

    def _web_fetch(self, request: RunToolsRequest) -> str:
        url = str(request.arguments.get("url", "")).strip()
        if not url:
            raise ValueError("missing url")

        html = self._fetcher(url)
        parsed = _HTMLSummaryParser()
        parsed.feed(html)

        text_parts: list[str] = []
        if parsed.title:
            text_parts.append(parsed.title)
        text_parts.extend(parsed.texts)
        if not text_parts:
            text_parts.append(html)
        return _truncate("\n".join(text_parts), self._max_chars)

    def _web_scan(self, request: RunToolsRequest) -> str:
        url = str(request.arguments.get("url", "")).strip()
        if not url:
            raise ValueError("missing url")

        html = self._fetcher(url)
        parsed = _HTMLSummaryParser()
        parsed.feed(html)

        lines: list[str] = []
        if parsed.title:
            lines.append(f"title: {parsed.title}")
        if parsed.headings:
            lines.append("headings: " + " | ".join(parsed.headings[:3]))
        if parsed.texts:
            lines.append("text: " + " ".join(parsed.texts[:4]))
        if parsed.links:
            lines.append("links: " + " | ".join(href for href, _ in parsed.links[:3]))
        if not lines:
            lines.append("scan: no visible content")
        return _truncate("\n".join(lines), self._max_chars)


class _HTMLSummaryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.headings: list[str] = []
        self.texts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._current_link: str | None = None
        self._current_heading = False
        self._in_title = False
        self._suppress_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._suppress_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in {"h1", "h2", "h3"}:
            self._current_heading = True
        if tag == "a":
            for key, value in attrs:
                if key == "href" and isinstance(value, str):
                    self._current_link = value.strip() or None
                    break

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._suppress_depth > 0:
            self._suppress_depth -= 1
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in {"h1", "h2", "h3"}:
            self._current_heading = False
        if tag == "a":
            self._current_link = None

    def handle_data(self, data: str) -> None:
        if self._suppress_depth > 0:
            return

        text = data.strip()
        if not text:
            return

        if self._in_title and not self.title:
            self.title = text
            return

        if self._current_heading:
            self.headings.append(text)

        if self._current_link:
            self.links.append((self._current_link, text))
            return

        self.texts.append(text)


def _fetch_url(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": "xiangmu-readonly-web/1.0"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=10) as response:
            return cast(bytes, response.read()).decode("utf-8", errors="ignore")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        raise RuntimeError(detail or f"HTTP {exc.code}") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        raise RuntimeError(str(reason or exc)) from exc
    except TimeoutError as exc:
        raise RuntimeError("request timed out") from exc


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 14] + "\n...[truncated]"
