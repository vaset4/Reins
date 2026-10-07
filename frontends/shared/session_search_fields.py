from __future__ import annotations

import json
from typing import Any, Mapping


FACT_FIELDS_FOR_DEEP_SCAN = (
    "event",
    "trigger",
    "from_state",
    "to_state",
    "close_reason",
)


def fact_search_fields(fact: Mapping[str, Any]) -> list[tuple[str, str]]:
    fields = [(name, str(fact.get(name) or "")) for name in FACT_FIELDS_FOR_DEEP_SCAN]
    tool = fact.get("tool")
    if isinstance(tool, Mapping):
        fields.extend(
            [
                ("tool_summary", str(tool.get("summary") or "")),
                ("tool_output", str(tool.get("output_summary") or "")),
                ("tool_name", str(tool.get("name") or "")),
                ("tool_status", str(tool.get("status") or "")),
                ("tool_error_category", str(tool.get("error_category") or "")),
                ("tool_error", str(tool.get("error") or "")),
            ]
        )
    summary = fact.get("summary")
    if isinstance(summary, Mapping):
        fields.append(("summary", json.dumps(summary, ensure_ascii=False)))
    checkpoint = fact.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        fields.extend(
            [
                ("checkpoint_state", str(checkpoint.get("state") or "")),
                ("checkpoint_reason", str(checkpoint.get("reason") or "")),
            ]
        )
    return fields


__all__ = ["fact_search_fields"]
