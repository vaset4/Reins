from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from artifacts.store import ArtifactStore
from tasks.index_sync import connect_index
from tools.tool_registry import Idempotent, ToolRegistry, ToolRisk
from tools.vision import register_tools
from tools.vision.ocr import detect_sensitive_regions, ocr
from tools.vision.redact import redact
from tools.vision.screenshot import screenshot


def test_vision_tools_register_expected_risk_and_idempotency() -> None:
    registry = ToolRegistry()

    register_tools(registry)

    expected = {
        "screenshot": (ToolRisk.SAFE, Idempotent.YES),
        "ocr": (ToolRisk.SAFE, Idempotent.YES),
        "redact": (ToolRisk.CONFIRM, Idempotent.CONDITIONAL),
    }
    for name, (risk, idempotent) in expected.items():
        definition = registry.get(name)
        assert definition is not None
        assert definition.risk is risk
        assert definition.idempotent is idempotent
        assert definition.executor is not None


def test_ocr_is_hidden_when_tesseract_binary_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.machinery import ModuleSpec
    from tools import vision

    monkeypatch.setattr(
        vision.importlib.util,
        "find_spec",
        lambda name, package=None: (
            ModuleSpec(name, loader=None) if name == "pytesseract" else None
        ),
    )
    monkeypatch.setattr(vision.shutil, "which", lambda _name: None)
    registry = ToolRegistry()

    vision.register_tools(registry)

    definition = registry.get("ocr")
    assert definition is not None
    assert not definition.model_visible
    assert definition.check_available() == (False, "tesseract binary is not installed")


def test_screenshot_writes_screenshot_artifact_with_retention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    connect_index(tmp_path).close()

    class _Image:
        width = 20
        height = 10
        size = (20, 10)
        rgb = b"\x00\x00\x00" * 200

    class _Capture:
        monitors = [{"all": True}, {"left": 0, "top": 0, "width": 20, "height": 10}]

        def __enter__(self) -> "_Capture":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def grab(self, monitor: object) -> _Image:
            assert monitor == self.monitors[1]
            return _Image()

    def _to_png(_rgb: bytes, _size: tuple[int, int], *, output: str) -> None:
        Path(output).write_bytes(b"png")

    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=lambda: _Capture()))
    monkeypatch.setitem(sys.modules, "mss.tools", SimpleNamespace(to_png=_to_png))

    result = screenshot(1, data_root=tmp_path, task_id="task-1")

    artifact_id = str(result["artifact_id"])
    record = ArtifactStore(tmp_path).load_artifact(artifact_id)
    assert record is not None
    assert record.type == "screenshot"
    assert record.retention_until is not None
    assert datetime.fromisoformat(record.retention_until) == (
        datetime.fromisoformat(record.created_at) + timedelta(days=7)
    )
    assert result["width"] == 20
    assert result["height"] == 10
    assert result["monitor"] == 1


def test_ocr_reads_artifact_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_id = _png_artifact(tmp_path)
    calls: list[dict[str, object]] = []

    def _image_to_string(path: str, *, lang: str) -> str:
        calls.append({"path": path, "lang": lang})
        return "fixed text"

    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(image_to_string=_image_to_string),
    )

    result = ocr(artifact_id, "eng", data_root=tmp_path)

    assert result == {"text": "fixed text", "lang": "eng"}
    assert Path(calls[0]["path"]) == ArtifactStore(tmp_path).read_path(artifact_id)


def test_detect_sensitive_regions_from_mock_ocr_boxes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_id = _png_artifact(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(
            Output=SimpleNamespace(DICT="dict"),
            image_to_data=lambda *_args, **_kwargs: {
                "text": [
                    "Email",
                    "a@example.com",
                    "Card",
                    "4111",
                    "1111",
                    "1111",
                    "1111",
                ],
                "left": [0, 10, 0, 10, 30, 50, 70],
                "top": [0, 0, 20, 20, 20, 20, 20],
                "width": [8, 50, 8, 15, 15, 15, 15],
                "height": [10, 10, 10, 10, 10, 10, 10],
            },
        ),
    )

    regions = detect_sensitive_regions(artifact_id, data_root=tmp_path)

    assert len(regions) == 5
    assert any(region["x"] <= 10 and region["w"] >= 50 for region in regions)


def test_redact_explicit_regions_writes_black_pixels(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    artifact_id = _png_artifact(tmp_path, size=(80, 80), color="red")
    regions = [
        {"x": 5, "y": 5, "w": 10, "h": 10},
        {"x": 25, "y": 5, "w": 10, "h": 10},
        {"x": 5, "y": 25, "w": 10, "h": 10},
        {"x": 25, "y": 25, "w": 10, "h": 10},
    ]

    result = redact(artifact_id, regions, data_root=tmp_path)

    redacted_id = str(result["redacted_artifact_id"])
    record = ArtifactStore(tmp_path).load_artifact(redacted_id)
    assert record is not None
    assert record.type == "screenshot"
    assert record.retention_until is not None
    image = Image.open(tmp_path / record.path)
    assert result["regions_redacted"] == 4
    for region in regions:
        typed = cast(dict[str, int], region)
        assert image.getpixel((typed["x"] + 1, typed["y"] + 1)) == (0, 0, 0)


def test_redact_auto_mode_uses_ocr_regions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image
    from tools.vision import redact as redact_module

    artifact_id = _png_artifact(tmp_path, size=(40, 40), color="red")
    monkeypatch.setattr(
        redact_module,
        "detect_sensitive_regions",
        lambda *_args, **_kwargs: [{"x": 1, "y": 1, "w": 8, "h": 8}],
    )

    result = redact_module.redact(artifact_id, None, data_root=tmp_path)

    record = ArtifactStore(tmp_path).load_artifact(str(result["redacted_artifact_id"]))
    assert record is not None
    image = Image.open(tmp_path / record.path)
    assert image.getpixel((2, 2)) == (0, 0, 0)


def _png_artifact(
    tmp_path: Path,
    *,
    size: tuple[int, int] = (20, 20),
    color: str = "white",
) -> str:
    pytest.importorskip("PIL")
    from PIL import Image

    connect_index(tmp_path).close()
    artifact_id = "art-test"
    relative_path = f"tasks/task-1/artifacts/{artifact_id}.png"
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    ArtifactStore(tmp_path).create_artifact(
        "task-1",
        "screenshot",
        relative_path,
        "test screenshot",
        path.stat().st_size,
        artifact_id=artifact_id,
    )
    return artifact_id
