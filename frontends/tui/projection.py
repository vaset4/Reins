"""后台事实和有序流事件的显示投影，不承担执行与持久化。

作者：xxx
时间：2026-09-29 18:00:00
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any
from frontends.tui.tool_status import tool_display_state


@dataclass(frozen=True, slots=True)
class ToolResultSource:
    """原件读取仅携带会话事实身份，不接受产物路径。"""

    session_id: str
    run_id: str
    entry_id: str
    call_id: str


@dataclass(frozen=True, slots=True)
class Card:
    """一张具有稳定身份的消息卡片；工具结果始终使用 call_id 配对。"""

    key: str
    role: str
    text: str = ""
    reasoning: str = ""
    title: str = ""
    args: str = ""
    state: str = ""
    elapsed: float | None = None
    result_source: ToolResultSource | None = None
    run_id: str = ""
    message_id: str = ""
    entry_id: str = ""
    call_id: str = ""


class ConversationProjection:
    """只保存渲染所需投影，完整历史以后台快照为准。"""

    def __init__(self, *, retained_cards: int | None = None) -> None:
        """初始化显示状态；传参：实时正文缓存窗大小，历史页不指定；返回：无。"""
        self.session_id = ""
        self.cards: dict[str, Card] = {}
        self.cursor = 0
        self.status = "正在连接后台"
        self.usage: dict[str, int] = {}
        self.snapshot: dict[str, Any] = {}
        self._stream_key: str | None = None
        self._started: dict[str, float] = {}
        self.event_epoch = ""
        self.data_space_id = ""
        self._retired_spaces: set[str] = set()
        self._retired_epochs: set[str] = set()
        self._stream_run_id = ""
        self._message_keys: dict[str, str] = {}
        self._closed_streams: set[str] = set()
        self._committed_keys: set[str] = set()
        self._retained_cards = retained_cards
        self._reading_keys: set[str] = set()
        self.has_older_cards = False

    def apply(self, snapshot: dict[str, Any]) -> set[str]:
        """按事件游标合并快照并去重；传参：真实后台快照；返回：变化卡片编号。"""
        before = dict(self.cards)
        identity = snapshot["session_id"]
        if not self._accept_space(snapshot):
            return set()
        epoch = str(snapshot.get("event_epoch", ""))
        if epoch and epoch in self._retired_epochs:
            return set()
        if identity != self.session_id and "history" not in snapshot:
            return set()
        if (
            identity == self.session_id
            and epoch != self.event_epoch
            and self.event_epoch
        ):
            if "history" not in snapshot:
                raise ValueError("后台事件实例已变化，需要重新取得会话快照")
            self._retired_epochs.add(self.event_epoch)
            self.cursor = 0
            self._stream_key = None
            self._started.clear()
            self._closed_streams.clear()
        if identity != self.session_id:
            self.cards.clear()
            self.cursor = 0
            self.usage = {}
            self._stream_key = None
            self._started.clear()
            self._closed_streams.clear()
            self._message_keys.clear()
            self._committed_keys.clear()
            self._reading_keys.clear()
            self.has_older_cards = False
        self.event_epoch = epoch
        self.session_id = identity
        self.snapshot = snapshot
        self.status = str(snapshot.get("status", "idle"))
        if "history" in snapshot:
            self._history(snapshot["history"])
            self.cursor = int(snapshot.get("cursor", 0))
            for stream in snapshot.get("streams", []):
                message_id = stream["message_id"]
                if (
                    message_id in self._message_keys
                    or message_id in self._closed_streams
                ):
                    continue
                key = f"stream:{self.session_id}:{message_id}"
                self.cards[key] = Card(
                    key,
                    "assistant",
                    stream["text"],
                    stream["reasoning"],
                    run_id=str(stream.get("run_id", "")),
                    message_id=message_id,
                )
        for event in snapshot.get("events", []):
            sequence = int(event["sequence"])
            if sequence <= self.cursor:
                continue
            self._event(
                event, str(event.get("run_id") or snapshot.get("current_run_id", ""))
            )
            self.cursor = sequence
        self._release_old_cards()
        return {
            key
            for key in before.keys() | self.cards.keys()
            if before.get(key) != self.cards.get(key)
        }

    def _accept_space(self, snapshot: dict[str, Any]) -> bool:
        """空间变化只接受完整握手，旧空间消息永久作废；参数：后台快照；返回：是否可应用。"""
        space = str(snapshot.get("data_space_id", ""))
        if space and space in self._retired_spaces:
            return False
        if self.data_space_id and space != self.data_space_id:
            if "history" not in snapshot:
                raise ValueError("数据空间已变化，需要重新取得完整会话快照")
            self._retired_spaces.add(self.data_space_id)
            self.session_id, self.event_epoch, self._stream_run_id = "", "", ""
            self._retired_epochs.clear()
        self.data_space_id = space
        return True

    def activity_status(self) -> str:
        """根据后台时间显示真实等待与退避；参数：无；返回：当前状态文字。"""
        if self.snapshot.get("error") and not self.snapshot.get("active"):
            return f"失败 · {self.snapshot['error']}"
        activity = self.snapshot.get("activity")
        if not activity:
            return self.status
        kind, data = activity["kind"], activity["data"]
        elapsed = max(0, time.time() - activity["started_at"])
        label = str(activity["label"])
        if kind == "ModelRetryScheduled":
            remaining = max(0, data["wait_seconds"] - elapsed)
            return (
                f"重试 {data['attempt_index']}/{data['max_attempts']} · "
                f"{remaining:.1f}s 后重试 · {data['error_category']}"
            )
        if kind == "ModelRequestStarted":
            label += f" · 尝试 {data['attempt_index']}/{data['max_attempts']}"
        elif kind == "LifecycleChanged":
            labels = {
                "running": "运行中",
                "done": "已完成",
                "failed": "失败",
                "paused": "已暂停",
            }
            return (
                f"{labels.get(data['lifecycle'], data['lifecycle'])} · {data['reason']}"
            )
        elif kind in {"SegmentPaused", "AssistantStreamClosed"}:
            return f"{label} · {data['reason']}"
        elif kind in {"ToolExecutionStarted", "ToolExecutionCompleted"}:
            label += f" · {data['tool_name']}"
        if kind in {"ToolExecutionCompleted", "AssistantTurnComplete"}:
            return label
        timing = "已等待" if kind == "ModelRequestStarted" else "已用时"
        return f"{label} · {timing} {elapsed:.0f}s · Esc 取消"

    def accepted(self, identity: str, text: str) -> None:
        """仅在后台已接纳后显示用户正文；传参：输入编号和原文；返回：无。"""
        if identity in self._committed_keys:
            return
        self.cards[identity] = Card(identity, "user", text)
        self._committed_keys.add(identity)
        self._release_old_cards()

    def current_key(self, key: str) -> str:
        """把阅读中的流身份解析到提交后的卡片；参数：原卡片键；返回：当前显示键。"""
        prefix = f"stream:{self.session_id}:"
        return (
            self._message_keys.get(key[len(prefix) :], key)
            if key.startswith(prefix)
            else key
        )

    def retain_window(self, keys: Iterable[str]) -> None:
        """保护正在阅读的一窗，释放其余已保存旧正文；参数：显示键；返回：无。"""
        self._reading_keys = {self.current_key(key) for key in keys}
        self._release_old_cards()

    def _release_old_cards(self) -> None:
        """仅回收有持久事实的旧正文，保留活跃输出和轻量去重身份；参数：无；返回：无。"""
        if self._retained_cards is None:
            return
        recent = set(list(self.cards)[-self._retained_cards :])
        # 【TUI】【长连接缓存】1. 最近一窗、当前阅读和未提交输出都继续持有完整正文
        removable = (
            self._committed_keys.intersection(self.cards) - recent - self._reading_keys
        )
        for key in removable:
            del self.cards[key]
        self.has_older_cards = self.has_older_cards or bool(removable)

    def _history(self, rows: list[dict[str, Any]]) -> None:
        """从当前分支正文重建卡片，调用与结果按身份合并；传参：持久消息；返回：无。"""
        cards: dict[str, Card] = {}
        for row in rows:
            role = row["role"]
            key = str(row["entry_id"])
            if row.get("message_id"):
                self._message_keys[row["message_id"]] = key
            text = str(row.get("text", ""))
            if role == "tool":
                call_id = str(row.get("tool_call_id", key))
                tool_key = self._tool_key(str(row.get("run_id") or ""), call_id)
                old = cards.get(
                    tool_key, Card(tool_key, "tool", title=row.get("tool_name", "工具"))
                )
                state = tool_display_state(row.get("status"), row.get("execution", {}))
                details = [text]
                if row.get("error"):
                    details.append(f"错误：{row['error']}")
                if row.get("artifact_refs"):
                    details.append("产物：\n" + "\n".join(row["artifact_refs"]))
                source = (
                    ToolResultSource(**row["result_source"])
                    if row.get("result_source")
                    else None
                )
                cards[tool_key] = replace(
                    old,
                    text="\n\n".join(details),
                    state=state,
                    result_source=source,
                    run_id=str(row.get("run_id") or ""),
                    entry_id=key,
                    call_id=call_id,
                )
                self._committed_keys.add(tool_key)
                continue
            reasoning = str(row.get("reasoning") or "")
            if text or reasoning:
                cards[key] = Card(
                    key,
                    role,
                    text,
                    reasoning,
                    run_id=str(row.get("run_id") or ""),
                    message_id=str(row.get("message_id") or ""),
                    entry_id=key,
                )
                self._committed_keys.add(key)
            for call in row.get("tool_calls", []):
                call_id = str(call.get("call_id", call.get("id", "")))
                tool_key = self._tool_key(str(row.get("run_id") or ""), call_id)
                cards[tool_key] = Card(
                    tool_key,
                    "tool",
                    title=str(call.get("name", call.get("tool_name", "工具"))),
                    args=json.dumps(
                        call.get("arguments", call.get("args", {})),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    state="等待结果",
                    run_id=str(row.get("run_id") or ""),
                    message_id=str(row.get("message_id") or ""),
                    call_id=call_id,
                )
        self.cards = cards
        self._stream_key = None

    def _event(self, row: dict[str, Any], run_id: str) -> None:
        """消费一个真实流事件，不将重试或工具参数当正文；传参：事件和运行身份；返回：无。"""
        kind, data = row["type"], row["data"]
        if kind in {
            "AssistantTextDelta",
            "AssistantReasoningDelta",
            "AssistantTurnComplete",
            "AssistantStreamClosed",
        }:
            if run_id != self._stream_run_id:
                self._stream_key = None
                self._stream_run_id = run_id
            self._assistant(
                kind, data, f"stream:{self.session_id}:{run_id}:{row['sequence']}"
            )
        elif kind in {"ToolExecutionStarted", "ToolExecutionCompleted"}:
            self._tool(kind, data, run_id)
        elif kind == "ModelRetryScheduled":
            self.status = (
                f"网络重试 {data['attempt_index']}/{data['max_attempts']} · "
                f"等待 {data['wait_seconds']:g}s · {data['error_category']}"
            )
        elif kind == "LifecycleChanged":
            self.status = f"{data['lifecycle']} · {data['reason']}"
        elif kind == "SegmentPaused":
            self.status = f"已暂停 · {data['reason']} · /resume 继续"

    def _assistant(self, kind: str, data: dict[str, Any], identity: str) -> None:
        """合并单轮正文与思考，用完整收尾替换重连后的局部片段；传参：事件及身份；返回：无。"""
        message_id = data.get("message_id", "")
        if kind == "AssistantStreamClosed":
            self._close_stream(data)
            return
        if message_id in self._closed_streams:
            return
        if (
            message_id in self._message_keys
            or data.get("entry_id") in self._committed_keys
        ):
            if kind == "AssistantTurnComplete":
                self.usage = dict(data.get("usage", {}))
            return
        key = (
            f"stream:{self.session_id}:{message_id}"
            if message_id
            else self._stream_key or identity
        )
        card = self.cards.get(
            key,
            Card(key, "assistant", run_id=self._stream_run_id, message_id=message_id),
        )
        self._stream_key = key
        if kind == "AssistantTextDelta":
            card = replace(card, text=card.text + data["text"])
        elif kind == "AssistantReasoningDelta":
            card = replace(card, reasoning=card.reasoning + data["text"])
        else:
            if data.get("content") is not None:
                card = replace(card, text=data["content"])
            self.usage = dict(data.get("usage", {}))
            self._stream_key = None
            if data.get("entry_id"):
                entry_id = data["entry_id"]
                if key in self._reading_keys:
                    self._reading_keys.remove(key)
                    self._reading_keys.add(entry_id)
                self.cards = {
                    entry_id if old_key == key else old_key: old_card
                    for old_key, old_card in self.cards.items()
                }
                key = entry_id
                card = replace(card, key=key, entry_id=entry_id)
                self._committed_keys.add(key)
                if message_id:
                    self._message_keys[message_id] = key
        if card.text or card.reasoning:
            self.cards[key] = card

    def _close_stream(self, data: dict[str, Any]) -> None:
        """关闭指定请求的局部显示，取消片段只保留在当前连接；参数：真实关闭事件；返回：无。"""
        message_id, reason = data["message_id"], data["reason"]
        key = f"stream:{self.session_id}:{message_id}"
        labels = {
            "cancelled": "输出已中断，局部内容未保存为回答",
            "superseded": "旧输出已被新输入替代",
            "retry": "失败尝试的输出已清理，等待重试",
            "model_request_failed": "请求失败，未完成输出已清理",
        }
        self.status = labels.get(reason, f"输出已停止：{reason}，未保存为回答")
        if reason == "cancelled" and key in self.cards:
            card = self.cards[key]
            self.cards[key] = replace(
                card, text=f"【输出已中断，未保存】\n\n{card.text}", state="中断"
            )
        else:
            self.cards.pop(key, None)
            notice_key = f"closure:{self.session_id}:{message_id}:{reason}"
            self.cards[notice_key] = Card(notice_key, "status", self.status)
        self._stream_key = None
        if not data.get("retrying"):
            self._closed_streams.add(message_id)

    def _tool(self, kind: str, data: dict[str, Any], run_id: str) -> None:
        """按调用编号更新工具完整结果及本次可观察耗时；传参：工具事件；返回：无。"""
        key = self._tool_key(run_id, data["call_id"])
        if key in self._committed_keys:
            self._started.pop(key, None)
            return
        card = self.cards.get(
            key,
            Card(
                key,
                "tool",
                title=data["tool_name"],
                run_id=run_id,
                call_id=data["call_id"],
            ),
        )
        if kind == "ToolExecutionStarted":
            self._started[key] = time.monotonic()
            card = replace(
                card,
                args=json.dumps(data["args"], ensure_ascii=False, indent=2),
                state="执行中",
            )
        else:
            started = self._started.pop(key, None)
            execution = data.get("execution", {})
            output = data["output"]
            if execution.get("partial_state"):
                output += f"\n\n执行影响：{execution['partial_state']}"
            state = tool_display_state(
                "error" if data.get("is_error") else "success", execution
            )
            card = replace(
                card,
                text=output,
                state=state,
                elapsed=time.monotonic() - started if started is not None else None,
            )
            # 【TUI】【长连接缓存】2. 完成事件由后台在会话工具交换提交成功之后发出
            self._committed_keys.add(key)
        self.cards[key] = card

    def _tool_key(self, run_id: str, call_id: str) -> str:
        """工具编号只在所属会话和运行内配对；传参：运行及调用编号；返回：显示身份。"""
        return f"tool:{self.session_id}:{run_id}:{call_id}"
