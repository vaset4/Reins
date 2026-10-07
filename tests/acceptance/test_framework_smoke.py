from __future__ import annotations

import sys

from tests.acceptance.helpers.cassette import Cassette
from tests.acceptance.helpers.subprocess_helper import ReinsSubprocess
from tests.acceptance.helpers.synth_data import make_memory, make_task


def test_mock_fixtures_intercept_calls(
    mock_screenshot,
    mock_notification,
    mock_clipboard,
) -> None:
    import pyperclip
    import reins_screenshot
    import win10toast

    assert reins_screenshot("screen") == b"\x89PNG\r\n\x1a\n"
    assert win10toast("title", msg="body") is True
    assert pyperclip("copy") == ""
    assert mock_screenshot.calls[0]["args"] == ("screen",)
    assert mock_notification.calls[0]["kwargs"] == {"msg": "body"}
    assert mock_clipboard.calls[0]["args"] == ("copy",)


def test_reins_subprocess_noop_start_stop() -> None:
    args = [sys.executable, "-c", "import time; time.sleep(30)"]
    with ReinsSubprocess(args=args) as process:
        assert process.pid is not None


def test_cassette_records_and_replays(cassette_dir) -> None:
    path = cassette_dir / "mock_llm.json"
    first = Cassette(path).replay_or_record(
        method="POST",
        url="mock://llm",
        body={"messages": ["hello"]},
        response_factory=lambda: {"text": "world"},
    )
    second = Cassette(path).replay_or_record(
        method="POST",
        url="mock://llm",
        body={"messages": ["hello"]},
        response_factory=lambda: {"text": "changed"},
    )
    assert first == {"text": "world"}
    assert second == {"text": "world"}


def test_synth_data_validates() -> None:
    make_task().validate()
    make_memory().validate()
