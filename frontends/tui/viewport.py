from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TranscriptViewport:
    offset: int = 0
    follow_latest: bool = True

    def jump_to_latest(self, max_scroll: int) -> None:
        if self.follow_latest:
            self.offset = max(0, max_scroll)

    def scroll_up(self, lines: int) -> None:
        self.follow_latest = False
        self.offset = max(0, self.offset - lines)

    def scroll_down(self, lines: int, max_scroll: int) -> None:
        self.offset = min(self.offset + lines, max(0, max_scroll))
        self.follow_latest = self.offset >= max(0, max_scroll)

    def follow(self) -> None:
        self.follow_latest = True


__all__ = ["TranscriptViewport"]
