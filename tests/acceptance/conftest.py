from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.acceptance.helpers.mock_tools import CallRecorder


@pytest.fixture
def reins_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / ".reins" / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("REINS_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture
def cassette_dir(tmp_path: Path) -> Path:
    path = tmp_path / "cassettes"
    path.mkdir()
    return path


@pytest.fixture
def mock_screenshot(monkeypatch: pytest.MonkeyPatch) -> CallRecorder:
    recorder = CallRecorder(result=b"\x89PNG\r\n\x1a\n")
    monkeypatch.setitem(sys.modules, "reins_screenshot", recorder)
    return recorder


@pytest.fixture
def mock_notification(monkeypatch: pytest.MonkeyPatch) -> CallRecorder:
    recorder = CallRecorder(result=True)
    monkeypatch.setitem(sys.modules, "win10toast", recorder)
    return recorder


@pytest.fixture
def mock_clipboard(monkeypatch: pytest.MonkeyPatch) -> CallRecorder:
    recorder = CallRecorder(result="")
    monkeypatch.setitem(sys.modules, "pyperclip", recorder)
    return recorder


@pytest.fixture(autouse=True)
def isolated_reins_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    monkeypatch.setenv("REINS_HOME", str(tmp_path / ".reins"))
    yield
