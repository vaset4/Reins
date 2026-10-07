from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import closing
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast

import pytest

from approval import ApprovalDecision
from artifacts.store import ArtifactStore
from runtime.lease import Lease, from_trigger
from runtime.watchdog import WatchdogDecision
from tests.acceptance.helpers.sandbox_fixtures import (
    SandboxProject,
    create_sandbox_project,
    lease_for,
)
from tools import builtin_tools
from tools.browser.playwright_adapter import close_session
from tools.readonly_file_tools import ReadOnlyFileToolExecutor
from tools.readonly_inspection import ReadOnlyInspectionExecutor
from tools.readonly_web_tools import ReadOnlyWebToolExecutor
from tools.tool_registry import MCPToolRiskError, ToolRegistry
from tools.types import ToolError, ToolErrorCategory
from tools.write_file_tools import WriteFileToolExecutor


FIXTURES = Path(__file__).parent / "fixtures"


class _Watchdog:
    def __init__(self, data_root: Path | None = None, timeout: float = 5.0) -> None:
        self.data_root = data_root
        self.tool_timeout_seconds = timeout
        self.failures: list[tuple[str, dict[str, object]]] = []
        self.steps_reserved = 0

    def run_tool_with_timeout(
        self, operation: Callable[[], object], *, cancellation=None, on_late=None
    ) -> object:
        return operation()

    def record_tool_failure(self, tool: str, args: dict[str, object]) -> None:
        self.failures.append((tool, dict(args)))

    def reserve_tool_step(self, *, operation_id: str = "") -> WatchdogDecision:
        """记录一次工具派发；传参：操作身份；返回：放行决定，步数不再拦派发。"""
        self.steps_reserved += 1
        return WatchdogDecision(False)


class TestFileTools:
    def test_file_happy_workspace_write_then_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        registry = _registry(monkeypatch, sandbox)
        target = sandbox.workspace_scratch / "dod2.txt"

        write_result = registry.execute_tool(
            "file_write",
            {"path": str(target), "content": "workspace ok"},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )
        read_result = registry.execute_tool(
            "file_read",
            {"path": str(target)},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(
            sandbox, "file_happy", {"write": write_result, "read": read_result}
        )
        assert not isinstance(write_result, ToolError)
        assert cast(dict[str, object], read_result)["content"] == "workspace ok"

    def test_file_sandbox_block_out_of_bounds_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        result = _registry(monkeypatch, sandbox).execute_tool(
            "file_write",
            {"path": str(sandbox.home / "outside.txt"), "content": "no"},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(sandbox, "file_block", result)
        _assert_tool_error(result, ToolErrorCategory.PERMISSION)


class TestWebTools:
    def test_web_happy_fetches_readme_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        html = "<html><head><title>README</title></head><body><h1>Hello</h1><p>Docs</p></body></html>"
        registry = _registry(
            monkeypatch,
            sandbox,
            web_fetcher=lambda _url: html,
        )

        result = registry.execute_tool(
            "web_fetch",
            {"url": "https://docs.example.com/readme"},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(sandbox, "web_happy", result)
        payload = cast(dict[str, object], result)
        assert "README" in str(payload["content"])
        assert payload["summary"] == "tool executed"

    def test_web_sandbox_block_deny_domain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        result = _registry(monkeypatch, sandbox).execute_tool(
            "web_fetch",
            {"url": "https://deny.example.com/readme"},
            _lease(sandbox, network_deny=["deny.example.com"]),
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(sandbox, "web_block", result)
        _assert_tool_error(result, ToolErrorCategory.PERMISSION)
        assert cast(ToolError, result).message == "network_domain_denied"


class TestTerminalTools:
    def test_terminal_happy_git_status_whitelist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        subprocess.run(
            ["git", "init"],
            cwd=sandbox.project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        result = _registry(monkeypatch, sandbox).execute_tool(
            "terminal_tool",
            {"command": "git status", "cwd": str(sandbox.project_root)},
            _lease(sandbox, terminal_allow=["git status"]),
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(sandbox, "terminal_happy", result)
        payload = cast(dict[str, object], result)
        assert payload["exit_code"] == 0
        assert "on branch" in str(payload["stdout"]).lower()

    def test_terminal_sandbox_block_default_deny_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        result = _registry(monkeypatch, sandbox).execute_tool(
            "terminal_tool",
            {"command": r"del /f /s /q C:\\"},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(sandbox, "terminal_block", result)
        _assert_tool_error(result, ToolErrorCategory.PERMISSION)
        assert cast(ToolError, result).message == "terminal_command_denied"


class TestCodeExecutionTools:
    def test_code_execution_happy_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        result = _registry(monkeypatch, sandbox).execute_tool(
            "code_execution_tool",
            {"code": 'print("hi")'},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(sandbox, "code_happy", result)
        assert cast(dict[str, object], result)["stdout"] == "hi\n"

    def test_code_execution_sandbox_block_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        result = _registry(monkeypatch, sandbox).execute_tool(
            "code_execution_tool",
            {"code": "while True:\n    pass", "timeout_seconds": 1},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root, timeout=3.0),
            max_retries=0,
        )

        _record_trace(sandbox, "code_block", result)
        _assert_tool_error(result, ToolErrorCategory.TIMEOUT)


class TestBrowserTools:
    def test_browser_happy_headless_file_fixture_extract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_playwright_chromium()
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        registry = _registry(monkeypatch, sandbox)
        url = (FIXTURES / "sample.html").resolve().as_uri()

        try:
            navigate = registry.execute_tool(
                "browser_navigate",
                {"url": url},
                _lease(sandbox, fs_read=[str(FIXTURES)]),
                watchdog=_Watchdog(sandbox.data_root),
            )
            extract = registry.execute_tool(
                "browser_extract",
                {},
                _lease(sandbox, fs_read=[str(FIXTURES)]),
                watchdog=_Watchdog(sandbox.data_root),
            )
        finally:
            close_session()

        _record_trace(
            sandbox, "browser_happy", {"navigate": navigate, "extract": extract}
        )
        assert cast(dict[str, object], navigate)["status"] is not None
        assert "Hello" in str(cast(dict[str, object], extract)["text"])

    def test_browser_sandbox_block_deny_domain_before_playwright(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        result = _registry(monkeypatch, sandbox).execute_tool(
            "browser_navigate",
            {"url": "https://bank.example.com/login"},
            _lease(sandbox, browser_deny=["bank.example.com"]),
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(sandbox, "browser_block", result)
        _assert_tool_error(result, ToolErrorCategory.PERMISSION)
        assert cast(ToolError, result).message == "browser_domain_denied"


class TestVisionTools:
    def test_vision_happy_screenshot_ocr_redact_pixels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        source_png = _email_png(tmp_path)
        _mock_vision(monkeypatch, source_png)
        registry = _registry(monkeypatch, sandbox)

        screenshot = registry.execute_tool(
            "screenshot",
            {"monitor": "1"},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )
        artifact_id = str(cast(dict[str, object], screenshot)["artifact_id"])
        ocr = registry.execute_tool(
            "ocr",
            {"artifact_id": artifact_id},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )
        redacted = registry.execute_tool(
            "redact",
            {"artifact_id": artifact_id},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )

        redacted_id = str(cast(dict[str, object], redacted)["redacted_artifact_id"])
        record = ArtifactStore(sandbox.data_root).load_artifact(redacted_id)
        assert record is not None
        image = Image.open(sandbox.data_root / record.path)
        _record_trace(
            sandbox,
            "vision_happy",
            {"screenshot": screenshot, "ocr": ocr, "redact": redacted},
        )
        assert "test@example.com" in str(cast(dict[str, object], ocr)["text"])
        assert image.getpixel((13, 13)) == (0, 0, 0)

    def test_vision_sandbox_block_redact_approval_denied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        artifact_id = _artifact_png(sandbox.data_root)
        registry = _registry(monkeypatch, sandbox)
        monkeypatch.setattr(
            "tools.tool_registry.approval.request_approval",
            lambda _req: ApprovalDecision.DENY,
        )

        result = registry.execute_tool(
            "redact",
            {"artifact_id": artifact_id, "regions": [{"x": 1, "y": 1, "w": 5, "h": 5}]},
            _lease(sandbox),
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(sandbox, "vision_block", result)
        _assert_tool_error(result, ToolErrorCategory.PERMISSION)
        assert cast(ToolError, result).message == "approval_denied"


class TestMCPTools:
    def test_mcp_happy_lazy_load_echo_and_shutdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools.mcp_client.registry import attach_mcp_registry

        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        registry = _registry(monkeypatch, sandbox)
        lease = _lease(
            sandbox, mcp_allow=["echo"], mcp_config=FIXTURES / "mcp_test.yaml"
        )
        attach_mcp_registry(lease, registry)

        result = registry.execute_tool(
            "mcp_echo_echo",
            {"text": "hello"},
            lease,
            watchdog=_Watchdog(sandbox.data_root),
        )

        _record_trace(sandbox, "mcp_happy", result)
        assert result["content"]["structuredContent"]["text"] == "hello"
        # 真实进程归目录所有：关掉目录就得收干净，不再依赖进程级全局服务器表
        registry.close()
        assert [
            server["status"] for server in registry.source_status()["mcp"]["servers"]
        ] == ["closed"]

    def test_mcp_sandbox_block_disallowed_server_not_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools.mcp_client.registry import attach_mcp_registry

        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        registry = _registry(monkeypatch, sandbox)
        lease = _lease(
            sandbox, mcp_allow=["echo"], mcp_config=FIXTURES / "mcp_test.yaml"
        )
        attach_mcp_registry(lease, registry)

        # 同一份配置里有两个服务器，授权只放行 echo：先坐实配置真的接通了，
        # 否则「没注册」可能只是因为整份配置都没生效
        assert registry.get("mcp_echo_echo") is not None
        assert registry.get("mcp_filesystem_read_file") is None
        with closing(registry):
            result = registry.execute_tool(
                "mcp_filesystem_read_file",
                {"filePath": str(sandbox.home / ".env")},
                lease,
                watchdog=_Watchdog(sandbox.data_root),
            )

        _record_trace(sandbox, "mcp_block", result)
        _assert_tool_error(result, ToolErrorCategory.INVALID_INPUT)

    def test_mcp_sandbox_block_safe_risk_override_raises_visible_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools.mcp_client.registry import MCPRegistry

        sandbox = create_sandbox_project(tmp_path, monkeypatch)
        registry = _registry(monkeypatch, sandbox)
        config = tmp_path / "mcp_safe_lock.yaml"
        config.write_text(
            """
servers:
  echo:
    command: ["python", "tests/fixtures/echo_mcp_server.py"]
    transport: stdio
    tool_risk_overrides:
      echo: safe
    tools:
      echo:
        description: Echo one text value.
        inputSchema:
          type: object
          properties:
            text:
              type: string
          required: [text]
""",
            encoding="utf-8",
        )

        with pytest.raises(MCPToolRiskError) as exc_info:
            MCPRegistry(
                _lease(sandbox, mcp_allow=["echo"], mcp_config=config), registry
            ).register_allowed_tools()

        _record_trace(
            sandbox,
            "mcp_risk_lock",
            {
                "error_type": type(exc_info.value).__name__,
                "message": str(exc_info.value),
            },
        )
        assert "safe" in str(exc_info.value)


def _registry(
    monkeypatch: pytest.MonkeyPatch,
    sandbox: SandboxProject,
    *,
    web_fetcher: Callable[[str], str] | None = None,
) -> ToolRegistry:
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    inspection = ReadOnlyInspectionExecutor(sandbox.project_root, 50, 4000, 50)
    monkeypatch.setattr(builtin_tools, "_INSPECTION", inspection)
    monkeypatch.setattr(
        builtin_tools, "_FILE_TOOLS", ReadOnlyFileToolExecutor(inspection)
    )
    monkeypatch.setattr(
        builtin_tools, "_WRITE_FILE_TOOLS", WriteFileToolExecutor(sandbox.project_root)
    )
    if web_fetcher is not None:
        monkeypatch.setattr(
            builtin_tools, "_WEB_TOOLS", ReadOnlyWebToolExecutor(fetcher=web_fetcher)
        )
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)
    return registry


def _lease(
    sandbox: SandboxProject,
    *,
    fs_read: list[str] | None = None,
    network_deny: list[str] | None = None,
    terminal_allow: list[str] | None = None,
    browser_deny: list[str] | None = None,
    mcp_allow: list[str] | None = None,
    mcp_config: Path | None = None,
) -> Lease:
    base = lease_for(sandbox)
    capabilities = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in base.capabilities.items()
    }
    fs = cast(dict[str, object], capabilities["fs"])
    fs["read"] = [
        *cast(list[str], fs["read"]),
        *(fs_read or []),
    ]
    capabilities["network"] = {
        "enabled": True,
        "deny_domains": network_deny or [],
    }
    capabilities["terminal"] = {
        "enabled": True,
        "allow_commands": terminal_allow or [],
    }
    capabilities["browser"] = {
        "enabled": True,
        "profile": "default",
        "deny_domains": browser_deny or [],
        "headless": True,
    }
    capabilities["mcp"] = {
        "enabled": True,
        "allow_servers": mcp_allow or [],
    }
    if mcp_config is not None:
        cast(dict[str, object], capabilities["mcp"])["config_path"] = str(mcp_config)
    return from_trigger("user", task_id=sandbox.task_id, capabilities=capabilities)


def _mock_vision(monkeypatch: pytest.MonkeyPatch, source_png: Path) -> None:
    def find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name in {"mss", "mss.tools", "pytesseract", "PIL"}:
            return ModuleSpec(name, loader=None)
        return None

    class ImageGrab:
        width = 80
        height = 40
        size = (80, 40)
        rgb = b""

    class Capture:
        monitors = [{"all": True}, {"left": 0, "top": 0, "width": 80, "height": 40}]

        def __enter__(self) -> "Capture":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def grab(self, _monitor: object) -> ImageGrab:
            return ImageGrab()

    def to_png(_rgb: bytes, _size: tuple[int, int], *, output: str) -> None:
        Path(output).write_bytes(source_png.read_bytes())

    monkeypatch.setattr("tools.vision.importlib.util.find_spec", find_spec)
    monkeypatch.setattr("tools.vision.shutil.which", lambda _name: "tesseract.exe")
    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=lambda: Capture()))
    monkeypatch.setitem(sys.modules, "mss.tools", SimpleNamespace(to_png=to_png))
    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(
            Output=SimpleNamespace(DICT="dict"),
            image_to_string=lambda *_args, **_kwargs: "test@example.com",
            image_to_data=lambda *_args, **_kwargs: {
                "text": ["test@example.com"],
                "left": [10],
                "top": [10],
                "width": [60],
                "height": [12],
            },
        ),
    )


def _require_playwright_chromium() -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:
        if os.environ.get("GITHUB_ACTIONS") != "true" and _is_winerror_5(exc):
            pytest.skip(f"playwright chromium unavailable locally: {exc}")
        raise


def _is_winerror_5(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if getattr(current, "winerror", None) == 5 or "WinError 5" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _email_png(tmp_path: Path) -> Path:
    from PIL import Image, ImageDraw

    path = tmp_path / "screenshot_with_email.png"
    image = Image.new("RGB", (80, 40), "white")
    ImageDraw.Draw(image).text((10, 10), "test@example.com", fill="black")
    image.save(path)
    return path


def _artifact_png(data_root: Path) -> str:
    from PIL import Image

    artifact_id = "art-dod2-redact-block"
    relative_path = f"tasks/task-1/artifacts/{artifact_id}.png"
    path = data_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), "white").save(path)
    ArtifactStore(data_root).create_artifact(
        "task-1",
        "screenshot",
        relative_path,
        "redact block fixture",
        path.stat().st_size,
        artifact_id=artifact_id,
    )
    return artifact_id


def _record_trace(sandbox: SandboxProject, name: str, result: object) -> None:
    trace_dir = sandbox.data_root / "tasks" / sandbox.task_id / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": name,
        "result": _jsonable(result),
    }
    (trace_dir / f"dod2-{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _jsonable(value: object) -> object:
    if isinstance(value, ToolError):
        return {
            "category": value.category.value,
            "message": value.message,
            "retryable": value.retryable,
            "partial_state": value.partial_state,
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, subprocess.CompletedProcess):
        return {"returncode": value.returncode}
    return value


def _assert_tool_error(result: object, category: ToolErrorCategory) -> None:
    assert isinstance(result, ToolError)
    assert result.category is category
