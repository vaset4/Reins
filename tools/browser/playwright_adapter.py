from __future__ import annotations

import atexit
import fnmatch
import threading
from collections.abc import Callable
from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from queue import SimpleQueue
from typing import Any, TypeVar, cast
from urllib.parse import urlparse
from urllib.request import url2pathname

from artifacts.store import ArtifactStore
from runtime.workspaces import WorkspaceStore
from context.artifact_ref import store_large_output
import path_security
from runtime.lease import Lease
from tasks.ids import new_ulid
from tools.types import ToolError, ToolErrorCategory

VIEWPORT = {"width": 1280, "height": 800}
EXTRACT_THRESHOLD = 4096

_session: BrowserSession | None = None
_sync_playwright_factory: Any | None = None
_browser_owner: _BrowserThreadOwner | None = None
_browser_owner_lock = threading.Lock()
_BrowserResult = TypeVar("_BrowserResult")


class _BrowserThreadOwner:
    """为 Playwright sync API 持有唯一、可复用的线程。"""

    def __init__(self) -> None:
        self._queue: SimpleQueue[tuple[Callable[[], object], Future[object]] | None] = (
            SimpleQueue()
        )
        self._state_lock = threading.Lock()
        self._closed = False
        self._thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name="reins-browser",
            daemon=True,
        )
        self._thread.start()

    def run(self, operation: Callable[[], _BrowserResult]) -> _BrowserResult:
        """在浏览器专属线程执行操作并同步返回结果。

        传参：operation 为不接收参数的浏览器操作。
        返回：operation 的执行结果。
        """
        if self._thread_id == threading.get_ident():
            return operation()
        future: Future[object] = Future()
        with self._state_lock:
            if self._closed:
                raise RuntimeError("browser thread owner is closed")
            self._queue.put((cast(Callable[[], object], operation), future))
        return cast(_BrowserResult, future.result())

    def close(self, operation: Callable[[], None]) -> None:
        """在 owner 线程执行关闭操作并停止该线程。

        传参：operation 为 session 和 Playwright 关闭操作。
        返回：无。
        """
        if self._thread_id == threading.get_ident():
            operation()
            with self._state_lock:
                self._closed = True
                self._queue.put(None)
            return
        future: Future[object] = Future()
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put((operation, future))
            self._queue.put(None)
        try:
            future.result()
        finally:
            self._thread.join()

    def _serve(self) -> None:
        """串行消费浏览器操作，保持 Playwright 线程亲和。

        传参：无。
        返回：无。
        """
        self._thread_id = threading.get_ident()
        while True:
            item = self._queue.get()
            if item is None:
                return
            operation, future = item
            try:
                future.set_result(operation())
            except BaseException as exc:
                future.set_exception(exc)


class BrowserSession:
    def __init__(
        self,
        *,
        profile: str,
        headless: bool,
        profile_root: Path | None = None,
        playwright_factory: Any | None = None,
    ) -> None:
        self.profile = profile
        self.headless = headless
        self.profile_dir = (profile_root or _default_profile_root()) / profile
        self.storage_state_path = self.profile_dir / "storage_state.json"
        self._playwright_factory = playwright_factory
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None

    def navigate(self, url: str) -> dict[str, object]:
        page = self._page_or_start()
        response = page.goto(url)
        return {
            "url": getattr(page, "url", url),
            "status": getattr(response, "status", None),
        }

    def click(
        self,
        *,
        selector: str = "",
        text: str = "",
        expect_download: bool = False,
        data_root: Path | None = None,
        task_id: str = "",
        project_root: Path | None = None,
    ) -> dict[str, object]:
        page = self._page_or_start()
        if expect_download:
            return self._click_with_download(
                page,
                selector=selector,
                text=text,
                data_root=data_root,
                task_id=task_id,
                project_root=project_root,
            )
        if selector:
            page.click(selector)
            target = selector
        elif text:
            page.get_by_text(text).click()
            target = text
        else:
            return {"clicked": False, "reason": "missing selector or text"}
        return {"clicked": True, "target": target}

    def type_text(self, selector: str, text: str) -> dict[str, object]:
        page = self._page_or_start()
        page.fill(selector, text)
        return {"typed": True, "selector": selector, "chars": len(text)}

    def extract_text(self, selector: str = "") -> str:
        page = self._page_or_start()
        if selector:
            return str(page.locator(selector).inner_text())
        return str(page.locator("body").inner_text())

    @property
    def current_url(self) -> str:
        page = self._page
        if page is None:
            return ""
        return str(getattr(page, "url", ""))

    def screenshot(self, path: Path, *, full_page: bool = False) -> dict[str, int]:
        page = self._page_or_start()
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path), full_page=full_page)
        viewport = getattr(page, "viewport_size", None)
        if not isinstance(viewport, Mapping):
            viewport = VIEWPORT
        return {
            "width": int(viewport.get("width", VIEWPORT["width"])),
            "height": int(viewport.get("height", VIEWPORT["height"])),
        }

    def close(self) -> None:
        if self._context is not None:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            self._context.storage_state(path=str(self.storage_state_path))
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def _click_with_download(
        self,
        page: Any,
        *,
        selector: str,
        text: str,
        data_root: Path | None,
        task_id: str,
        project_root: Path | None,
    ) -> dict[str, object]:
        if data_root is None or not task_id or project_root is None:
            return {"clicked": False, "reason": "missing download workspace"}
        if not selector and not text:
            return {"clicked": False, "reason": "missing selector or text"}
        downloads_dir = project_root / ".reins" / "workspace" / task_id / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        with page.expect_download() as download_info:
            if selector:
                page.click(selector)
                target = selector
            else:
                page.get_by_text(text).click()
                target = text
        download = download_info.value
        filename = str(getattr(download, "suggested_filename", "download.bin"))
        path = downloads_dir / filename
        download.save_as(str(path))
        artifact_id = f"art-{new_ulid()}"
        ArtifactStore(data_root).create_artifact(
            task_id,
            "download",
            str(path),
            filename,
            path.stat().st_size,
            artifact_id=artifact_id,
            workspace_id=WorkspaceStore(data_root).register(project_root).workspace_id,
        )
        return {"clicked": True, "target": target, "download_artifact_id": artifact_id}

    def _page_or_start(self) -> Any:
        if self._page is not None:
            return self._page
        factory = self._playwright_factory or _sync_playwright()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = factory().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        kwargs: dict[str, object] = {
            "viewport": VIEWPORT,
            "accept_downloads": True,
        }
        if self.storage_state_path.is_file():
            kwargs["storage_state"] = str(self.storage_state_path)
        self._context = self._browser.new_context(**kwargs)
        self._page = self._context.new_page()
        return self._page


def get_session(lease: Lease) -> BrowserSession:
    """在浏览器 owner 线程中取得匹配租约配置的 session。

    传参：lease 为当前工具调用的能力租约。
    返回：可复用的 BrowserSession。
    """
    global _session
    profile = _profile_from_lease(lease)
    headless = _headless_from_lease(lease)
    if _session is None or _session.profile != profile or _session.headless != headless:
        _close_active_session()
        _session = BrowserSession(
            profile=profile,
            headless=headless,
            playwright_factory=_sync_playwright_factory,
        )
    return _session


def close_session() -> None:
    """在创建 Playwright 的同一线程关闭 session 和浏览器 owner。

    传参：无。
    返回：无；关闭异常保持可见。
    """
    global _browser_owner
    with _browser_owner_lock:
        owner = _browser_owner
    if owner is None:
        _close_active_session()
        return
    try:
        owner.close(_close_active_session)
    finally:
        with _browser_owner_lock:
            if _browser_owner is owner:
                _browser_owner = None


def _browser_call(operation: Callable[[], _BrowserResult]) -> _BrowserResult:
    """把一次浏览器调用交给持久单线程 owner。

    传参：operation 为需要保持 Playwright 线程亲和的操作。
    返回：operation 的执行结果。
    """
    global _browser_owner
    with _browser_owner_lock:
        if _browser_owner is None:
            _browser_owner = _BrowserThreadOwner()
        owner = _browser_owner
    return owner.run(operation)


def _close_active_session() -> None:
    """关闭当前全局 session，并清除可复用引用。

    传参：无。
    返回：无；底层关闭错误保持可见。
    """
    global _session
    if _session is None:
        return
    try:
        _session.close()
    finally:
        _session = None


def _click_on_browser_thread(
    lease: Lease,
    *,
    selector: str,
    text: str,
    expect_download: bool,
    data_root: Path | None,
    task_id: str,
    project_root: Path | None,
) -> object:
    """在线程亲和边界内复查域名并执行点击。

    传参：lease 为能力租约，其余参数为点击目标和下载工作区。
    返回：点击结果或权限错误。
    """
    session = get_session(lease)
    denied = _current_domain_error(session, lease)
    if denied is not None:
        return denied
    return session.click(
        selector=selector,
        text=text,
        expect_download=expect_download,
        data_root=data_root,
        task_id=task_id,
        project_root=project_root,
    )


def _type_on_browser_thread(lease: Lease, *, selector: str, text: str) -> object:
    """在线程亲和边界内复查域名并输入文本。

    传参：lease 为能力租约，selector 为目标元素，text 为输入内容。
    返回：输入结果或权限错误。
    """
    session = get_session(lease)
    denied = _current_domain_error(session, lease)
    if denied is not None:
        return denied
    return session.type_text(selector, text)


def _extract_on_browser_thread(lease: Lease, selector: str) -> str | ToolError:
    """在线程亲和边界内复查域名并提取页面文本。

    传参：lease 为能力租约，selector 为可选 CSS 选择器。
    返回：页面文本或权限错误。
    """
    session = get_session(lease)
    denied = _current_domain_error(session, lease)
    if denied is not None:
        return denied
    return session.extract_text(selector)


def navigate_executor(args: dict[str, object]) -> object:
    lease = _lease_arg(args)
    permission = _browser_enabled(lease)
    if permission is not None:
        return permission
    url = str(args.get("url", "")).strip()
    denied = _deny_domain_error(url, lease)
    if denied is not None:
        return denied
    try:
        result = _browser_call(lambda: get_session(lease).navigate(url))
        result.setdefault("summary", "browser navigated")
        return result
    except ValueError as exc:
        return ToolError(ToolErrorCategory.INVALID_INPUT, str(exc), retryable=False)
    except Exception as exc:
        return ToolError(
            ToolErrorCategory.UNKNOWN,
            f"browser_unavailable: {exc}",
            retryable=False,
        )


def click_executor(args: dict[str, object]) -> object:
    lease = _lease_arg(args)
    permission = _browser_enabled(lease)
    if permission is not None:
        return permission
    return _browser_call(
        lambda: _click_on_browser_thread(
            lease,
            selector=str(args.get("selector", "")).strip(),
            text=str(args.get("text", "")).strip(),
            expect_download=_bool_arg(args.get("expect_download")),
            data_root=_path_arg(args.get("__data_root__")),
            task_id=str(args.get("__task_id__", "")).strip(),
            project_root=_project_root_from_lease(lease),
        )
    )


def type_executor(args: dict[str, object]) -> object:
    lease = _lease_arg(args)
    permission = _browser_enabled(lease)
    if permission is not None:
        return permission
    return _browser_call(
        lambda: _type_on_browser_thread(
            lease,
            selector=str(args.get("selector", "")).strip(),
            text=str(args.get("text", "")),
        )
    )


def extract_executor(args: dict[str, object]) -> object:
    lease = _lease_arg(args)
    permission = _browser_enabled(lease)
    if permission is not None:
        return permission
    try:
        text = _browser_call(
            lambda: _extract_on_browser_thread(
                lease, str(args.get("selector", "")).strip()
            )
        )
    except Exception as exc:
        return ToolError(
            ToolErrorCategory.UNKNOWN,
            f"browser_unavailable: {exc}",
            retryable=False,
        )
    if isinstance(text, ToolError):
        return text
    data_root = _path_arg(args.get("__data_root__")) or Path.home() / ".reins" / "data"
    task_id = str(args.get("__task_id__", "")).strip() or lease.task_id
    ref = store_large_output(
        data_root,
        task_id,
        text,
        type="html_dump",
        ext="txt",
        summary=text[:500].strip() or "browser extract",
    )
    if ref is None:
        return {"text": text, "summary": _extract_summary(text)}
    return {
        "artifact_id": ref.artifact_id,
        "summary": ref.summary[:500],
        "key_excerpt": ref.key_excerpt,
        "meta": {"artifact_id": ref.artifact_id, "artifact_type": "html_dump"},
    }


def screenshot_executor(args: dict[str, object]) -> object:
    lease = _lease_arg(args)
    permission = _browser_enabled(lease)
    if permission is not None:
        return permission
    data_root = _path_arg(args.get("__data_root__")) or Path.home() / ".reins" / "data"
    task_id = str(args.get("__task_id__", "")).strip() or lease.task_id
    artifact_id = f"art-{new_ulid()}"
    ArtifactStore(data_root)
    relative_path = f"assets/{artifact_id}.png"
    path = data_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dimensions = _browser_call(
            lambda: get_session(lease).screenshot(
                path,
                full_page=_bool_arg(args.get("full_page")),
            )
        )
    except Exception as exc:
        return ToolError(
            ToolErrorCategory.UNKNOWN,
            f"browser_unavailable: {exc}",
            retryable=False,
        )
    ArtifactStore(data_root).create_artifact(
        task_id,
        "screenshot",
        relative_path,
        "browser screenshot",
        path.stat().st_size,
        artifact_id=artifact_id,
        session_id=str(args.get("__session_id__") or "") or None,
    )
    return {
        "artifact_id": artifact_id,
        **dimensions,
        "summary": "browser screenshot saved",
        "meta": {"artifact_id": artifact_id, "artifact_type": "screenshot"},
    }


def _extract_summary(text: str) -> str:
    summary = " ".join(text.strip().split())
    return summary[:500] or "browser extract"


def _sync_playwright() -> Any:
    from playwright.sync_api import sync_playwright

    return sync_playwright


def _default_profile_root() -> Path:
    return Path.home() / ".reins" / "profile" / "browser"


def _lease_arg(args: Mapping[str, object]) -> Lease:
    lease = args.get("__lease__")
    if isinstance(lease, Lease):
        return lease
    raise ValueError("missing lease")


def _browser_enabled(lease: Lease) -> ToolError | None:
    browser = _browser_capabilities(lease)
    if browser.get("enabled") is False:
        return ToolError(
            ToolErrorCategory.PERMISSION, "browser_disabled", retryable=False
        )
    return None


def _browser_capabilities(lease: Lease) -> Mapping[str, object]:
    browser = lease.capabilities.get("browser")
    return browser if isinstance(browser, Mapping) else {}


def _profile_from_lease(lease: Lease) -> str:
    profile = str(_browser_capabilities(lease).get("profile", "default")).strip()
    if not profile:
        return "default"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in profile)
    return safe.strip("._") or "default"


def _headless_from_lease(lease: Lease) -> bool:
    value = _browser_capabilities(lease).get("headless", True)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no"}
    return True


def _deny_domain_error(url: str, lease: Lease) -> ToolError | None:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return _file_url_error(parsed, lease)
    hostname = parsed.hostname
    if not hostname:
        return ToolError(
            ToolErrorCategory.INVALID_INPUT, "invalid_url", retryable=False
        )
    denied = _str_list(_browser_capabilities(lease).get("deny_domains", []))
    if any(fnmatch.fnmatch(hostname.lower(), pattern) for pattern in denied):
        return ToolError(
            ToolErrorCategory.PERMISSION,
            "browser_domain_denied",
            retryable=False,
        )
    return None


def _current_domain_error(session: object, lease: Lease) -> ToolError | None:
    url = _current_url(session)
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme in {"", "about"}:
        return None
    if parsed.scheme not in {"file", "http", "https"}:
        return None
    return _deny_domain_error(url, lease)


def _current_url(session: object) -> str:
    value = getattr(session, "current_url", "")
    if callable(value):
        value = value()
    return str(value).strip() if value else ""


def _file_url_error(parsed: Any, lease: Lease) -> ToolError | None:
    if parsed.netloc not in {"", "localhost"}:
        return ToolError(
            ToolErrorCategory.INVALID_INPUT, "invalid_url", retryable=False
        )
    path_text = url2pathname(parsed.path)
    if not path_text:
        return ToolError(
            ToolErrorCategory.INVALID_INPUT, "invalid_url", retryable=False
        )
    decision = path_security.check_read(Path(path_text), lease)
    if decision is not path_security.Decision.ALLOWED:
        return ToolError(
            ToolErrorCategory.PERMISSION,
            "browser_file_path_denied",
            retryable=False,
        )
    return None


def _path_arg(value: object) -> Path | None:
    if isinstance(value, str | Path):
        return Path(value)
    return None


def _project_root_from_lease(lease: Lease) -> Path | None:
    fs = lease.capabilities.get("fs")
    if not isinstance(fs, Mapping):
        return None
    value = fs.get("project_root")
    return Path(value) if isinstance(value, str | Path) else None


def _bool_arg(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).lower() for item in value if item]


atexit.register(close_session)


__all__ = [
    "BrowserSession",
    "click_executor",
    "close_session",
    "extract_executor",
    "get_session",
    "navigate_executor",
    "screenshot_executor",
    "type_executor",
]
