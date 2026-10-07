from __future__ import annotations

import os
from enum import Enum


class TraceLevel(str, Enum):
    OFF = "off"
    BASIC = "basic"
    DEBUG = "debug"


_DEFAULT_TRACE_LEVEL = TraceLevel.BASIC


def get_trace_level() -> TraceLevel:
    env_value = os.environ.get("REINS_TRACE_LEVEL", "").strip().lower()
    if env_value in ("off", "basic", "debug"):
        return TraceLevel(env_value)
    return _DEFAULT_TRACE_LEVEL


def should_write_raw(level: TraceLevel | None = None) -> bool:
    effective = level if level is not None else get_trace_level()
    return effective == TraceLevel.DEBUG


def should_write_parsed_plan(level: TraceLevel | None = None) -> bool:
    effective = level if level is not None else get_trace_level()
    return effective in (TraceLevel.BASIC, TraceLevel.DEBUG)


__all__ = [
    "TraceLevel",
    "get_trace_level",
    "should_write_parsed_plan",
    "should_write_raw",
]
