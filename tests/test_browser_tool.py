from __future__ import annotations

import json
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import cast

from artifacts.store import ArtifactStore
from runtime.lease import Lease, from_trigger
from tasks.index_sync import connect_index
from tools import browser as browser_tools
from tools import builtin_tools
from tools.browser import playwright_adapter
from tools.browser.playwright_adapter import BrowserSession
from tools.tool_registry import Idempotent, ToolRegistry, ToolRisk
from tools.types import ToolError, ToolErrorCategory
from pytest import MonkeyPatch


class _FakeLocator:
    def __init__(self, text: str) -> None:
        self._text = text

    def inner_text(self) -> str:
        return self._text


class _FakePage:
    def __init__(self, text: str = "hello") -> None:
        self.url = ""
        self.viewport_size = {"width": 1280, "height": 800}
        self.text = text
        self.clicked: list[str] = []
        self.typed: list[tuple[str, str]] = []

    def goto(self, url: str) -> object:
        self.url = url
        return type("Response", (), {"status": 200})()

    def click(self, selector: str) -> None:
        self.clicked.append(selector)

    def get_by_text(self, text: str) -> "_FakePage":
        self.clicked.append(text)
        return self

    def fill(self, selector: str, text: str) -> None:
        self.typed.append((selector, text))

    def locator(self, _selector: str) -> _FakeLocator:
        return _FakeLocator(self.text)

    def screenshot(self, path: str, full_page: bool = False) -> None:
        Path(path).write_bytes(b"fake-png-full" if full_page else b"fake-png")


class _FakeContext:
    def __init__(self, page: _FakePage, label: str) -> None:
        self.page = page
        self.label = label
        self.closed = False

    def new_page(self) -> _FakePage:
        return self.page

    def storage_state(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"cookies": [{"name": self.label}]}),
            encoding="utf-8",
        )

    def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self, page: _FakePage, label: str) -> None:
        self.page = page
        self.label = label
        self.context_kwargs: list[dict[str, object]] = []
        self.closed = False

    def new_context(self, **kwargs: object) -> _FakeContext:
        self.context_kwargs.append(kwargs)
        return _FakeContext(self.page, self.label)

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser) -> None:
        self.browser = browser
        self.launch_kwargs: list[dict[str, object]] = []

    def launch(self, **kwargs: object) -> _FakeBrowser:
        self.launch_kwargs.append(kwargs)
        return self.browser


class _FakePlaywright:
    def __init__(self, browser: _FakeBrowser) -> None:
        self.chromium = _FakeChromium(browser)
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeStarter:
    def __init__(self, playwright: _FakePlaywright) -> None:
        self.playwright = playwright

    def start(self) -> _FakePlaywright:
        return self.playwright


class _ThreadAffineSession:
    """模拟只允许创建线程访问和关闭的 Playwright session。"""

    def __init__(self) -> None:
        self._thread_id: int | None = None
        self.calls: list[str] = []
        self._current_url = "https://example.com"

    @property
    def current_url(self) -> str:
        """返回当前页面地址，并校验读取线程。

        传参：无。
        返回：当前页面地址。
        """
        self._record("current_url")
        return self._current_url

    def navigate(self, url: str) -> dict[str, object]:
        """记录导航调用并返回成功响应。

        传参：url 为目标地址。
        返回：导航结果。
        """
        self._record("navigate")
        self._current_url = url
        return {"url": url, "status": 200}

    def extract_text(self, _selector: str = "") -> str:
        """记录文本提取调用。

        传参：_selector 为可选选择器。
        返回：固定页面文本。
        """
        self._record("extract")
        return "Reins browser fixture"

    def screenshot(self, path: Path, *, full_page: bool = False) -> dict[str, int]:
        """记录截图调用并写入非空图片证据。

        传参：path 为截图路径，full_page 表示是否截取完整页面。
        返回：固定视口尺寸。
        """
        self._record("screenshot")
        path.write_bytes(b"thread-affine-png" if full_page else b"png")
        return {"width": 1280, "height": 800}

    def close(self) -> None:
        """记录 session 关闭调用。

        传参：无。
        返回：无。
        """
        self._record("close")

    def _record(self, name: str) -> None:
        thread_id = threading.get_ident()
        if self._thread_id is None:
            self._thread_id = thread_id
        assert thread_id == self._thread_id
        self.calls.append(name)


def _factory(label: str, text: str = "hello") -> tuple[object, _FakeBrowser]:
    browser = _FakeBrowser(_FakePage(text), label)
    playwright = _FakePlaywright(browser)
    return lambda: _FakeStarter(playwright), browser


def _lease_denying_bank_domain() -> Lease:
    return from_trigger(
        "user",
        task_id="task-1",
        capabilities={
            "fs": {"read": [], "write": []},
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {
                "enabled": True,
                "profile": "default",
                "deny_domains": ["bank.example.com"],
                "headless": True,
            },
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
    )


def test_browser_tools_register_five_definitions() -> None:
    registry = ToolRegistry()

    builtin_tools.register_tools(registry)

    expected = {
        "browser_navigate": (ToolRisk.CONFIRM, Idempotent.CONDITIONAL),
        "browser_click": (ToolRisk.CONFIRM, Idempotent.NO),
        "browser_type": (ToolRisk.CONFIRM, Idempotent.NO),
        "browser_screenshot": (ToolRisk.SAFE, Idempotent.YES),
        "browser_extract": (ToolRisk.SAFE, Idempotent.YES),
    }
    for name, (risk, idempotent) in expected.items():
        definition = registry.get(name)
        assert definition is not None
        assert definition.risk is risk
        assert definition.idempotent is idempotent
        assert definition.executor is not None


def test_browser_navigate_happy_with_mock_playwright(tmp_path: Path) -> None:
    fake_factory, fake_browser = _factory("default")
    session = BrowserSession(
        profile="default",
        headless=True,
        profile_root=tmp_path,
        playwright_factory=fake_factory,
    )

    result = session.navigate("https://example.com")
    session.close()

    assert result == {"url": "https://example.com", "status": 200}
    assert fake_browser.context_kwargs[0]["viewport"] == {"width": 1280, "height": 800}
    assert (tmp_path / "default" / "storage_state.json").is_file()


def test_browser_executors_keep_thread_affine_session(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """不同 Watchdog worker 调用浏览器时仍由同一线程操作和关闭 session。"""
    data_root = tmp_path / "data"
    connect_index(data_root).close()
    lease = from_trigger("user", task_id="task-thread-affinity")
    session = _ThreadAffineSession()
    monkeypatch.setattr(playwright_adapter, "_session", session)
    monkeypatch.setattr(playwright_adapter, "get_session", lambda _lease: session)
    common = {
        "__lease__": lease,
        "__data_root__": data_root,
        "__task_id__": lease.task_id,
    }
    executors = [ThreadPoolExecutor(max_workers=1) for _ in range(3)]
    try:
        results = [
            executors[0]
            .submit(
                playwright_adapter.navigate_executor,
                common | {"url": "https://example.com"},
            )
            .result(),
            executors[1].submit(playwright_adapter.extract_executor, common).result(),
            executors[2]
            .submit(
                playwright_adapter.screenshot_executor,
                common | {"full_page": "true"},
            )
            .result(),
        ]
    finally:
        try:
            playwright_adapter.close_session()
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)

    assert all(not isinstance(result, ToolError) for result in results)
    assert {"navigate", "extract", "screenshot", "close"}.issubset(session.calls)


def test_browser_deny_domain_does_not_start_playwright(
    monkeypatch: MonkeyPatch,
) -> None:
    lease = from_trigger(
        "user",
        task_id="task-1",
        capabilities={
            "fs": {"read": [], "write": []},
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {
                "enabled": True,
                "profile": "default",
                "deny_domains": ["bank.example.com"],
            },
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
    )
    monkeypatch.setattr(
        playwright_adapter,
        "get_session",
        lambda _lease: (_ for _ in ()).throw(AssertionError("started playwright")),
    )

    result = playwright_adapter.navigate_executor(
        {"url": "https://bank.example.com/login", "__lease__": lease}
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "browser_domain_denied"


def test_browser_click_rechecks_current_domain(monkeypatch: MonkeyPatch) -> None:
    class _Session:
        current_url = "https://bank.example.com/login"
        clicked = False

        def click(self, **_kwargs: object) -> dict[str, object]:
            self.clicked = True
            return {"clicked": True}

    session = _Session()
    monkeypatch.setattr(playwright_adapter, "get_session", lambda _lease: session)

    result = playwright_adapter.click_executor(
        {"selector": "#submit", "__lease__": _lease_denying_bank_domain()}
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "browser_domain_denied"
    assert session.clicked is False


def test_browser_type_rechecks_current_domain(monkeypatch: MonkeyPatch) -> None:
    class _Session:
        current_url = "https://bank.example.com/login"
        typed = False

        def type_text(self, _selector: str, _text: str) -> dict[str, object]:
            self.typed = True
            return {"typed": True}

    session = _Session()
    monkeypatch.setattr(playwright_adapter, "get_session", lambda _lease: session)

    result = playwright_adapter.type_executor(
        {
            "selector": "#email",
            "text": "user@example.com",
            "__lease__": _lease_denying_bank_domain(),
        }
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "browser_domain_denied"
    assert session.typed is False


def test_browser_extract_rechecks_current_domain(monkeypatch: MonkeyPatch) -> None:
    class _Session:
        current_url = "https://bank.example.com/login"
        extracted = False

        def extract_text(self, _selector: str = "") -> str:
            self.extracted = True
            return "secret"

    session = _Session()
    monkeypatch.setattr(playwright_adapter, "get_session", lambda _lease: session)

    result = playwright_adapter.extract_executor(
        {"selector": "body", "__lease__": _lease_denying_bank_domain()}
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "browser_domain_denied"
    assert session.extracted is False


def test_browser_file_url_without_read_lease_does_not_start_playwright(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    fixture = tmp_path / "sample.html"
    fixture.write_text("<h1>Hello</h1>", encoding="utf-8")
    lease = from_trigger(
        "user",
        task_id="task-1",
        capabilities={
            "fs": {"read": [], "write": []},
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {
                "enabled": True,
                "profile": "default",
                "deny_domains": [],
                "headless": True,
            },
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
    )
    monkeypatch.setattr(
        playwright_adapter,
        "get_session",
        lambda _lease: (_ for _ in ()).throw(AssertionError("started playwright")),
    )

    result = playwright_adapter.navigate_executor(
        {"url": fixture.resolve().as_uri(), "__lease__": lease}
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "browser_file_path_denied"


def test_browser_extract_large_text_uses_artifact(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    connect_index(tmp_path).close()
    lease = from_trigger("user", task_id="task-1")

    class _Session:
        def extract_text(self, _selector: str = "") -> str:
            return "x" * 10_000

    monkeypatch.setattr(playwright_adapter, "get_session", lambda _lease: _Session())

    result = playwright_adapter.extract_executor(
        {"__lease__": lease, "__data_root__": tmp_path, "__task_id__": "task-1"}
    )

    payload = cast(dict[str, object], result)
    artifact_id = str(payload["artifact_id"])
    assert artifact_id.startswith("art-")
    assert len(str(payload["summary"])) == 500
    artifact_path = ArtifactStore(tmp_path).read_path(artifact_id)
    assert artifact_path.read_text(encoding="utf-8") == "x" * 10_000
    record = ArtifactStore(tmp_path).load_artifact(artifact_id)
    assert record is not None
    assert record.type == "html_dump"
    assert record.retention_until is not None
    assert datetime.fromisoformat(record.retention_until) == (
        datetime.fromisoformat(record.created_at) + timedelta(days=7)
    )


def test_browser_screenshot_uses_artifact(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    connect_index(tmp_path).close()
    lease = from_trigger("user", task_id="task-1")

    class _Session:
        def screenshot(self, path: Path, *, full_page: bool = False) -> dict[str, int]:
            path.write_bytes(b"png")
            return {"width": 1280, "height": 800}

    monkeypatch.setattr(playwright_adapter, "get_session", lambda _lease: _Session())

    result = playwright_adapter.screenshot_executor(
        {"__lease__": lease, "__data_root__": tmp_path, "__task_id__": "task-1"}
    )

    payload = cast(dict[str, object], result)
    artifact_id = str(payload["artifact_id"])
    record = ArtifactStore(tmp_path).load_artifact(artifact_id)
    assert record is not None
    assert record.type == "screenshot"
    assert record.retention_until is not None
    assert datetime.fromisoformat(record.retention_until) == (
        datetime.fromisoformat(record.created_at) + timedelta(days=7)
    )
    assert ArtifactStore(tmp_path).read_path(artifact_id).read_bytes() == b"png"
    assert payload["width"] == 1280
    assert payload["height"] == 800
    assert payload["summary"] == "browser screenshot saved"
    assert payload["meta"] == {
        "artifact_id": artifact_id,
        "artifact_type": "screenshot",
    }


def test_browser_tools_unavailable_without_playwright(
    monkeypatch: MonkeyPatch,
) -> None:
    def _missing_playwright(name: str, package: str | None = None) -> ModuleSpec | None:
        assert name == "playwright"
        assert package is None
        return None

    registry = ToolRegistry()
    browser_tools.register_tools(registry)
    monkeypatch.setattr(importlib.util, "find_spec", _missing_playwright)

    result = registry.execute_tool("browser_extract", {}, from_trigger("user"))

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert "Playwright optional dependency is not installed" in result.message


def test_browser_profiles_do_not_share_storage_state(tmp_path: Path) -> None:
    factory_a, browser_a = _factory("profile-a")
    session_a = BrowserSession(
        profile="profile-a",
        headless=True,
        profile_root=tmp_path,
        playwright_factory=factory_a,
    )
    session_a.navigate("https://example.com/a")
    session_a.close()

    factory_b, browser_b = _factory("profile-b")
    session_b = BrowserSession(
        profile="profile-b",
        headless=True,
        profile_root=tmp_path,
        playwright_factory=factory_b,
    )
    session_b.navigate("https://example.com/b")
    session_b.close()

    state_a = json.loads(
        (tmp_path / "profile-a" / "storage_state.json").read_text(encoding="utf-8")
    )
    state_b = json.loads(
        (tmp_path / "profile-b" / "storage_state.json").read_text(encoding="utf-8")
    )
    assert state_a != state_b
    assert "storage_state" not in browser_b.context_kwargs[0]
    assert browser_a.closed
    assert browser_b.closed
