"""只缓存界面事件；断档后的正文从 Session 主存重新读取。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from threading import RLock
from typing import Any, cast, get_args
from uuid import uuid4
from time import time

from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurnComplete,
    AssistantStreamClosed,
    StreamEvent,
)

EVENT_CACHE_SIZE = 4096
_EVENT_TYPES = {kind.__name__: kind for kind in get_args(StreamEvent)}


class EventBuffer:
    """缓存有限的展示事件，游标失效只触发重新读正文，不触发工作重放。"""

    def __init__(self) -> None:
        """建立当前进程的事件序号；传参：无；返回：无。"""
        self._lock = RLock()
        self._sequence = 0
        self._epoch = uuid4().hex
        self._events: deque[dict[str, Any]] = deque(maxlen=EVENT_CACHE_SIZE)
        self._streams: dict[str, dict[str, Any]] = {}
        self._closed_streams: set[str] = set()
        self._activity: dict[str, Any] = {}

    def emit(self, event: StreamEvent, *, run_id: str = "") -> None:
        """保存同一 AgentLoop 的展示事件；传参：真实事件；返回：无。"""
        with self._lock:
            if isinstance(
                event,
                (AssistantTextDelta, AssistantReasoningDelta, AssistantTurnComplete),
            ):
                if event.message_id in self._closed_streams:
                    return
            self._update_activity(event, run_id)
            self._sequence += 1
            self._events.append(
                {
                    "sequence": self._sequence,
                    "run_id": run_id,
                    "type": type(event).__name__,
                    "data": asdict(event),
                }
            )
            if (
                isinstance(event, (AssistantTextDelta, AssistantReasoningDelta))
                and event.message_id
            ):
                current = self._streams.setdefault(
                    event.message_id,
                    {
                        "message_id": event.message_id,
                        "run_id": run_id,
                        "text": "",
                        "reasoning": "",
                    },
                )
                channel = (
                    "text" if isinstance(event, AssistantTextDelta) else "reasoning"
                )
                current[channel] += event.text
            elif (
                isinstance(event, (AssistantTurnComplete, AssistantStreamClosed))
                and event.message_id
            ):
                self._streams.pop(event.message_id, None)
                if not isinstance(event, AssistantStreamClosed) or not event.retrying:
                    self._closed_streams.add(event.message_id)

    def _update_activity(self, event: StreamEvent, run_id: str) -> None:
        """缓存最新可观察活动供重连读取；参数：真实事件、运行编号；返回：无。"""
        kind = type(event).__name__
        labels = {
            "ModelRequestStarted": "等待模型响应",
            "ModelRetryScheduled": "等待重试",
            "AssistantTextDelta": "正在接收回答",
            "AssistantReasoningDelta": "正在接收思考",
            "ToolExecutionStarted": "工具执行中",
            "ToolExecutionCompleted": "工具已返回",
            "ToolApprovalRequested": "等待审批",
            "LifecycleChanged": "运行状态",
            "SegmentPaused": "已暂停",
            "AssistantStreamClosed": "输出已停止",
            "AssistantTurnComplete": "模型响应已完成",
        }
        if kind not in labels:
            return
        # 【TUI】【运行状态】1. 同类增量保留起始时间，避免每个 token 都重置计时
        if (
            kind in {"AssistantTextDelta", "AssistantReasoningDelta"}
            and self._activity.get("kind") == kind
        ):
            return
        self._activity = {
            "kind": kind,
            "label": labels[kind],
            "started_at": time(),
            "run_id": run_id,
            "data": asdict(event),
        }
        # 【TUI】【运行状态】2. 状态快照不复制模型正文，只保留当前尝试与真实原因
        self._activity["data"].pop("text", None)
        self._activity["data"].pop("content", None)
        self._activity["data"].pop("output", None)
        self._activity["data"].pop("args", None)

    def read(
        self, after: int | None, *, include_streams: bool = False
    ) -> dict[str, Any]:
        """读取游标之后的缓存；传参：已显示序号；返回：事件、当前序号和断档标记。"""
        with self._lock:
            gap = after is not None and (
                after > self._sequence
                or bool(self._events and after < self._events[0]["sequence"] - 1)
            )
            events = (
                [
                    row
                    for row in self._events
                    if after is not None and row["sequence"] > after
                ]
                if not gap
                else []
            )
            return {
                "events": events,
                "cursor": self._sequence,
                "gap": gap,
                "event_epoch": self._epoch,
                "activity": dict(self._activity),
                "streams": [dict(value) for value in self._streams.values()]
                if include_streams or gap
                else [],
            }


def decode_event(value: dict[str, Any]) -> StreamEvent:
    """从受认证的宿主还原渲染事件；传参：事件对象；返回：原生事件，未知类型明确失败。"""
    kind = _EVENT_TYPES.get(value["type"])
    if kind is None:
        raise ValueError(f"unknown background event: {value['type']}")
    return cast(StreamEvent, kind(**value["data"]))
