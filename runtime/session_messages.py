from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from llm.messages import (
    AgentMessage,
    AssistantMessage,
    StopReason,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    freeze_json_object,
    thaw_json_value,
)
from runtime.session_message_store import SessionMessageStore
from tasks.ids import new_ulid

_ASSISTANT_ROLE: Final = "assistant"
_TOOL_ROLE: Final = "tool"

HistoryRow = dict[str, object]
HistoryRowGroup = tuple[HistoryRow, ...]


@dataclass(frozen=True, slots=True)
class ToolExchange:
    """描述一次工具调用及其结果，用于一次性提交成两条 canonical 消息。

    作者：xxx
    时间：2026-08-27 20:00:00
    传参：call_id/tool_name/args 描述调用；rendered 为模型可见结果文本；
    status 为运行时状态串；error 为可选失败原因
    返回：不可变工具交换值对象
    """

    call_id: str
    tool_name: str
    args: Mapping[str, object] = field(default_factory=dict)
    rendered: str = ""
    status: str = "ok"
    error: str | None = None
    artifact_refs: tuple[str, ...] = ()


def append_user_message(
    data_root: Path | str,
    session_id: str,
    text: str,
    *,
    run_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """把一条用户输入提交为 canonical 用户消息。

    作者：xxx
    时间：2026-08-27 20:00:00
    传参：data_root 为 data 根目录；session_id 为会话标识；text 为用户原文；
    run_id/task_id 为可选关联证据
    返回：已保存的 message_id，供本次运行引用；同一输入只提交一次
    """
    message = UserMessage(_new_message_id(), (TextPart(text),))
    _append(
        data_root,
        session_id,
        message,
        run_id=run_id,
        task_id=task_id,
    )
    return message.message_id


def append_assistant_message(
    data_root: Path | str,
    session_id: str,
    text: str,
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    message_id: str | None = None,
    reasoning: str = "",
) -> str:
    """把模型正文或可见思考提交为 canonical assistant 消息。

    作者：xxx
    时间：2026-08-27 20:00:00
    传参：data_root 为 data 根目录；session_id 为会话标识；text 为模型可见回答；
    run_id/task_id 为可选关联证据
    message_id沿用流式消息身份，reasoning为可见思考；返回：已提交entry_id
    """
    # 1. 工具调用前可能只有思考，未生成正文时不创建空 TextPart
    parts: tuple[TextPart | ThinkingPart, ...] = (
        (TextPart(text),) if text.strip() else ()
    )
    if reasoning.strip():
        parts = (ThinkingPart(reasoning, "visible"), *parts)
    # 2. 正文和思考都为空时由消息合同拒绝，不保存虚构回答
    return _append(
        data_root,
        session_id,
        AssistantMessage(
            message_id or _new_message_id(), parts, stop_reason=StopReason.END_TURN
        ),
        run_id=run_id,
        task_id=task_id,
    )


def append_tool_exchange(
    data_root: Path | str,
    session_id: str,
    exchange: ToolExchange,
    *,
    run_id: str | None = None,
    task_id: str | None = None,
) -> None:
    """把一次工具调用与其结果提交为关联的两条 canonical 消息。

    作者：xxx
    时间：2026-08-27 20:00:00
    传参：data_root/session_id 指向会话；exchange 为本次工具交换；
    run_id/task_id 为可选关联证据
    返回：无；assistant tool_call 与 tool_result 通过 call_id 关联
    """
    # 1. 生产调用在派发前已公告；旧恢复入口可补公告，但不能重复提交既有结果
    messages = materialize_messages(data_root, session_id)
    for message in messages:
        if (
            isinstance(message, ToolResultMessage)
            and message.call_id == exchange.call_id
        ):
            if _joined_text(message.content) != exchange.rendered:
                raise ValueError("conflicting tool result for existing call_id")
            return
    announced = {
        part.call_id
        for message in messages
        if isinstance(message, AssistantMessage)
        for part in message.content
        if isinstance(part, ToolCallPart)
    }
    if exchange.call_id not in announced:
        append_tool_calls(
            data_root, session_id, (exchange,), run_id=run_id, task_id=task_id
        )
    # 2. 运行时状态串收敛到冻结值域，非 ok 一律记为 error 并保留原始原因
    if exchange.status == "ok":
        status: Literal["success", "error"] = "success"
    else:
        status = "error"
    _append(
        data_root,
        session_id,
        ToolResultMessage(
            _new_message_id(),
            exchange.call_id,
            exchange.tool_name,
            (TextPart(exchange.rendered),),
            status,
            error=None if status == "success" else (exchange.error or exchange.status),
            artifact_refs=exchange.artifact_refs,
        ),
        run_id=run_id,
        task_id=task_id,
    )


def append_tool_calls(
    data_root: Path | str,
    session_id: str,
    calls: Sequence[ToolExchange],
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    source_message_id: str | None = None,
) -> None:
    """派发前保存整组调用；传参：会话、调用、归属及模型请求消息ID；返回：无，失败阻止执行。"""
    parts = tuple(
        ToolCallPart(
            call.call_id,
            call.tool_name,
            freeze_json_object(call.args, path="tool_calls.args"),
        )
        for call in calls
    )
    # 【模型调用】【思考续接】来源ID关联同一响应的前置思考，旧恢复记录不猜测来源
    message_id = (
        f"{source_message_id}:tool-calls" if source_message_id else _new_message_id()
    )
    _append(
        data_root,
        session_id,
        AssistantMessage(message_id, parts, stop_reason=StopReason.TOOL_CALL),
        run_id=run_id,
        task_id=task_id,
    )


def materialize_messages(
    data_root: Path | str, session_id: str
) -> tuple[AgentMessage, ...]:
    """恢复当前分支的线性模型消息序列。

    作者：xxx
    时间：2026-08-27 20:00:00
    传参：data_root 为 data 根目录；session_id 为会话标识
    返回：root-to-leaf 顺序的 AgentMessage；会话尚未产生任何消息时返回空 tuple
    """
    store = SessionMessageStore(data_root)
    if not store.exists(session_id):
        return ()
    return store.materialize(session_id).messages


def rows_from_messages(
    messages: Sequence[AgentMessage],
) -> list[dict[str, object]]:
    """把 canonical AgentMessage 序列转成既有的 role/content 历史行形状。

    作者：xxx
    时间：2026-08-27 22:40:00
    传参：messages 为当前分支的 root-to-leaf 消息序列
    返回：与 prompt 裁剪和 REPL 显示兼容的历史行；未建模的消息种类被跳过

    这是展示与 prompt 边界上的形状转换，不是第二个消息 owner：数据只来自
    Session Store。typed prompt 组装要等 Provider Adapter 接线那一卡。
    """
    rows: list[dict[str, object]] = []
    for message in messages:
        row = _row_from_message(message)
        if row is not None:
            rows.append(row)
    return rows


def read_history_rows(
    data_root: Path | str, session_id: str, *, limit: int = 0
) -> list[dict[str, object]]:
    """按会话读出历史行，供 REPL 显示与手工压缩使用。

    作者：xxx
    时间：2026-08-27 22:40:00
    传参：data_root 为 data 根目录；session_id 为会话标识；limit 为保留的末尾行数，
    非正数表示全量
    返回：时间顺序的历史行；会话不存在时返回空 list
    """
    rows = rows_from_messages(materialize_messages(data_root, session_id))
    if limit > 0 and len(rows) > limit:
        return tail_history_rows(rows, max_rows=limit)
    return rows


def group_history_rows(rows: Sequence[HistoryRow]) -> tuple[HistoryRowGroup, ...]:
    """把历史行按"不可分割的工具调用组"切块，供所有尾部裁剪共用。

    作者：LKX
    时间：2026-08-28 21:30:00
    传参：rows 为时间顺序的历史行
    返回：分组后的行块 tuple，拼回去与入参逐行等同；不修改也不复制入参行对象

    一次工具调用在存储行层面是多行：assistant 行带 tool_calls 公告调用 id，紧随其后的
    role=tool 行用 tool_call_id 回指同一个 id。工具结果脱离发起它的公告就没有意义，
    所以这两类行必须当成一个整体被保留或丢弃，裁剪时不允许从中间切开。

    公告缺失的孤立 role=tool 行自成一组（组首是 tool 行），交由下游 prompt 组装显式报错，
    这里不掩盖，也不替它补造公告。
    """
    groups: list[HistoryRowGroup] = []
    current: list[HistoryRow] = []
    open_call_ids: set[str] = set()
    for row in rows:
        # 1. 工具结果回指当前组已公告的调用 id 时并入当前组，其余行一律另起一组
        if current and _extends_group(row, open_call_ids):
            current.append(row)
            continue
        if current:
            groups.append(tuple(current))
        current = [row]
        open_call_ids = _announced_call_ids(row)
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def tail_history_rows(rows: Sequence[HistoryRow], *, max_rows: int) -> list[HistoryRow]:
    """按行数上限取历史尾部，且不切开任何工具调用组。

    作者：LKX
    时间：2026-08-28 21:30:00
    传参：rows 为时间顺序的历史行；max_rows 为保留行数上限，非正数表示全量
    返回：新的行 list；实际行数可少于 max_rows，绝不多于

    上限落在某个工具调用组中间时整组丢弃，而不是留下半组：留下的半组要么是没有公告的
    孤立工具结果，要么是没有结果的悬空调用，两种都会让模型读到不成立的调用图。
    """
    if max_rows <= 0:
        return list(rows)
    selected: list[HistoryRowGroup] = []
    kept = 0
    # 1. 从尾往前整组累加，遇到装不下的组就停，保持历史是连续的一段尾部
    for group in reversed(group_history_rows(rows)):
        if kept + len(group) > max_rows:
            break
        selected.append(group)
        kept += len(group)
    selected.reverse()
    return [row for group in selected for row in group]


def _extends_group(row: HistoryRow, open_call_ids: set[str]) -> bool:
    """判断这条行是否是当前组已公告调用的工具结果。"""
    if str(row.get("role", "")).strip() != _TOOL_ROLE:
        return False
    return str(row.get("tool_call_id") or "") in open_call_ids


def _announced_call_ids(row: HistoryRow) -> set[str]:
    """取出 assistant 行公告的全部工具调用 id；非公告行返回空集合。"""
    if str(row.get("role", "")).strip() != _ASSISTANT_ROLE:
        return set()
    calls = row.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        return set()
    return {
        str(call["id"])
        for call in calls
        if isinstance(call, Mapping) and call.get("id")
    }


def _row_from_message(message: AgentMessage) -> dict[str, object] | None:
    """把单条 AgentMessage 转成历史行；工具调用保留 OpenAI 兼容的展示字段。"""
    if isinstance(message, UserMessage):
        return {"role": "user", "content": _joined_text(message.content)}
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "content": _joined_text(message.content),
            "tool_call_id": message.call_id,
        }
    if isinstance(message, AssistantMessage):
        return _assistant_row(message)
    return None


def _assistant_row(message: AssistantMessage) -> dict[str, object]:
    """组装 assistant 历史行，带上本条消息发起的工具调用。"""
    row: dict[str, object] = {
        "role": "assistant",
        "content": _joined_text(message.content),
    }
    calls = [
        {
            "id": part.call_id,
            "type": "function",
            "function": {
                "name": part.tool_name,
                "arguments": json.dumps(
                    thaw_json_value(part.arguments), ensure_ascii=False
                ),
            },
        }
        for part in message.content
        if isinstance(part, ToolCallPart)
    ]
    if calls:
        row["tool_calls"] = calls
    return row


def _joined_text(parts: Sequence[object]) -> str:
    """拼接消息里的模型可见文本块。"""
    return "".join(part.text for part in parts if isinstance(part, TextPart))


def _append(
    data_root: Path | str,
    session_id: str,
    message: AgentMessage,
    *,
    run_id: str | None,
    task_id: str | None,
) -> str:
    """把一条消息追加到会话的 canonical entries 文件。

    作者：xxx
    时间：2026-08-28 18:40:00
    传参：data_root 为 data 根目录；session_id 为会话标识；message 为待落盘消息；
    run_id/task_id 为可选关联证据
    返回：已提交条目编号，供流式收尾关联同一事实
    """
    return (
        SessionMessageStore(data_root)
        .append_message(
            session_id, message, run_id=run_id or None, task_id=task_id or None
        )
        .entry_id
    )


def _new_message_id() -> str:
    """生成本次运行内稳定唯一的消息标识。"""
    return f"msg-{new_ulid()}"


__all__ = [
    "HistoryRow",
    "HistoryRowGroup",
    "ToolExchange",
    "append_assistant_message",
    "append_tool_exchange",
    "append_user_message",
    "group_history_rows",
    "materialize_messages",
    "read_history_rows",
    "rows_from_messages",
    "tail_history_rows",
]
