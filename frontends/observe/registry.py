from __future__ import annotations

from typing import Iterable

from frontends.observe.panels.base import Panel, PanelDescriptor


class PanelRegistry:
    """Registry of inspection panels, ordered by registration order.

    Adding a new panel = one new backend file + one frontend file +
    one line in ``panels/__init__.py``. Server, router, and shell stay
    unchanged.
    """

    def __init__(self) -> None:
        self._panels: list[Panel] = []
        self._by_id: dict[str, Panel] = {}

    def register(self, panel: Panel) -> None:
        if not panel.id:
            raise ValueError("panel.id must be non-empty")
        if panel.id in self._by_id:
            raise ValueError(f"panel id {panel.id!r} already registered")
        self._panels.append(panel)
        self._by_id[panel.id] = panel

    def register_many(self, panels: Iterable[Panel]) -> None:
        for panel in panels:
            self.register(panel)

    def all(self) -> list[Panel]:
        return list(self._panels)

    def get(self, panel_id: str) -> Panel | None:
        return self._by_id.get(panel_id)

    def descriptors(self) -> list[PanelDescriptor]:
        return [panel.descriptor() for panel in self._panels]


def build_default_registry() -> PanelRegistry:
    """Construct registry from the panels exported by ``panels/__init__``."""
    from frontends.observe import panels as panel_pkg

    registry = PanelRegistry()
    registry.register_many(panel_pkg.ALL_PANELS)
    return registry


__all__ = ["PanelRegistry", "build_default_registry"]
