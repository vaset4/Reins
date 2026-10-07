from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4
from contextlib import AbstractContextManager

from runtime.persistence import (
    RuntimeStore,
    FileTransaction,
    PreparedPayload,
    record_key,
)

from llm.messages import (
    AgentMessage,
    AssistantMessage,
    MessageContractError,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    UserContentPart,
    TextPart,
    agent_message_from_mapping,
    agent_message_to_mapping,
    validate_message_sequence,
)

SessionEntryType = Literal["message", "branch", "inbound", "delivery"]
SESSION_ENTRY_TYPES = frozenset({"message", "branch", "inbound", "delivery"})
_BEIJING = timezone(timedelta(hours=8))


class SessionMessageStoreError(ValueError):
    """表示 Session Message Store 文件或 Session Tree 合同失败。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：code 为稳定错误码；session_id 为会话标识；detail 为失败原因；
    line_number/entry_id 为可选定位信息
    返回：携带稳定定位字段的 ValueError
    """

    def __init__(
        self,
        code: str,
        session_id: str,
        detail: str,
        *,
        line_number: int | None = None,
        entry_id: str | None = None,
    ) -> None:
        """初始化可定位的 Session Message Store 错误。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：code/session_id/detail 描述失败；line_number/entry_id 指向文件位置
        返回：无；构造异常对象
        """
        self.code = code
        self.session_id = session_id
        self.detail = detail
        self.line_number = line_number
        self.entry_id = entry_id
        parts = [code, f"session={session_id}"]
        if line_number is not None:
            parts.append(f"line={line_number}")
        if entry_id is not None:
            parts.append(f"entry={entry_id}")
        super().__init__(f"{' '.join(parts)}: {detail}")


@dataclass(frozen=True, slots=True)
class SessionHeader:
    """保存一个 Session Message Store 文件的唯一会话头。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：session_id 为会话标识；created_at 为北京时间创建时间
    返回：不可变 Session header
    """

    session_id: str
    created_at: str

    def to_mapping(self) -> dict[str, object]:
        """序列化严格 header，供 JSONL 文件写入。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：新的普通 JSON mapping
        """
        return {
            "type": "session",
            "session_id": self.session_id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class _EntryParseContext:
    """保存 Entry 解析期间不泄露文件路径的错误定位。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：session_id 为会话标识；line_number 为 JSONL 行号
    返回：不可变解析定位
    """

    session_id: str
    line_number: int


@dataclass(frozen=True, slots=True)
class _EntryIdentity:
    """保存已解析的 Entry 身份、父关系和运行关联字段。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：entry/parent/timestamp 为树身份；run/task 为可选关联
    返回：供 SessionEntry 构造使用的不可变字段集合
    """

    entry_id: str
    parent_id: str | None
    timestamp: str
    run_id: str | None
    task_id: str | None


@dataclass(frozen=True, slots=True)
class SessionEntry:
    """保存 Session Tree 中的一条不可变消息或分支事实。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：type/entry_id/parent_id/session_id/timestamp 为树事实；message 为消息事实；
    run_id/task_id 为可选运行关联
    返回：不可变 Session Entry
    """

    type: SessionEntryType
    entry_id: str
    parent_id: str | None
    session_id: str
    timestamp: str
    run_id: str | None = None
    task_id: str | None = None
    message: AgentMessage | None = None
    input_id: str | None = None
    input_source: Literal["user", "agent"] | None = None
    input_kind: Literal["approval"] | None = None

    def __post_init__(self) -> None:
        """校验 Entry 身份、类型和消息边界。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：字段来自严格 JSONL 解析或 Store append
        返回：无；非法 Entry 字段抛 ValueError，Store 边界会转换为稳定错误
        """
        _validate_entry_identity(self)
        _validate_entry_payload(self)

    def to_mapping(self) -> dict[str, object]:
        """序列化 Entry，保持消息与树控制字段分离。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：新的普通 JSON mapping
        """
        payload: dict[str, object] = {
            "type": self.type,
            "entry_id": self.entry_id,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }
        if self.run_id is not None:
            payload["run_id"] = self.run_id
        if self.task_id is not None:
            payload["task_id"] = self.task_id
        if self.message is not None:
            payload["message"] = agent_message_to_mapping(self.message)
        if self.input_id is not None:
            payload["input_id"] = self.input_id
        if self.input_source is not None:
            payload["input_source"] = self.input_source
        if self.input_kind is not None:
            payload["input_kind"] = self.input_kind
        return payload


def _validate_entry_identity(entry: SessionEntry) -> None:
    """校验 Session Entry 的树身份与可选关联字段。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：entry 为刚完成 dataclass 字段赋值的 SessionEntry
    返回：无；非法类型、身份或关联字段抛 ValueError
    """
    if not isinstance(entry.type, str) or entry.type not in SESSION_ENTRY_TYPES:
        raise ValueError(f"unknown session entry type: {entry.type!r}")
    for name in ("entry_id", "session_id", "timestamp"):
        if not _non_empty_text(getattr(entry, name)):
            raise ValueError(f"{name} must be non-empty")
    if entry.parent_id is not None and not _non_empty_text(entry.parent_id):
        raise ValueError("parent_id must be null or non-empty")
    for name in ("run_id", "task_id"):
        value = getattr(entry, name)
        if value is not None and not _non_empty_text(value):
            raise ValueError(f"{name} must be null or non-empty")


def _validate_entry_payload(entry: SessionEntry) -> None:
    """校验 message 与 branch Entry 的载荷互斥规则。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：entry 为已通过身份字段校验的 SessionEntry
    返回：无；message 缺失或 branch 携带消息时抛 ValueError
    """
    if entry.type in {"message", "inbound"} and not _is_agent_message(entry.message):
        raise ValueError("message entry requires AgentMessage")
    if entry.type in {"branch", "delivery"} and entry.message is not None:
        raise ValueError("branch entry cannot carry message")
    if entry.type == "inbound" and (
        not isinstance(entry.message, UserMessage)
        or entry.message.message_id != entry.entry_id
    ):
        raise ValueError("inbound entry must own its user input identity")
    if entry.type == "delivery" and not _non_empty_text(entry.input_id):
        raise ValueError("delivery requires input_id")
    if entry.type != "delivery" and entry.input_id is not None:
        raise ValueError("only delivery can reference input_id")
    if entry.input_source not in (None, "user", "agent") or (
        entry.input_source is not None and entry.type != "inbound"
    ):
        raise ValueError("input_source must classify an inbound user or agent message")
    if entry.input_kind is not None and (
        entry.input_kind != "approval"
        or entry.type != "inbound"
        or entry.input_source != "user"
    ):
        raise ValueError("approval input must be an explicit user control action")


@dataclass(frozen=True, slots=True)
class MaterializedSession:
    """表示从当前 leaf 沿父链重建出的线性模型上下文。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：session_id/leaf_id 为恢复身份；entries 为当前路径；messages 为模型消息；
    pending_tool_calls 为路径末尾尚未返回结果的调用标识
    返回：不可变 materialize 结果
    """

    session_id: str
    leaf_id: str | None
    entries: tuple[SessionEntry, ...]
    messages: tuple[AgentMessage, ...]
    pending_tool_calls: tuple[str, ...] = ()

    @property
    def pending_tool_call_ids(self) -> tuple[str, ...]:
        """返回尚未收到工具结果的调用标识。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：与 pending_tool_calls 相同顺序的不可变调用标识 tuple
        """
        return self.pending_tool_calls


@dataclass(frozen=True, slots=True)
class SessionHistoryPage:
    """固定分支叶上的一页完整交互，游标引用已有条目。"""

    session_id: str
    leaf_id: str | None
    entries: tuple[SessionEntry, ...]
    next_before: str | None


DEFAULT_HISTORY_PAGE_SIZE = 60
SESSION_TITLE_LENGTH = 64


class SessionMessageStore:
    """管理工作区文件中不可变会话条目的严格 Session Message Store。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：data_root 为 V2.1 durable data 根目录
    返回：提供 Session header、Entry 追加、分支和当前路径恢复
    """

    def __init__(self, data_root: Path | str) -> None:
        """初始化 Session Message Store 的 durable data 根目录。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：data_root 为包含 sessions 目录的 V2.1 data 根目录
        返回：无；构造不持有长生命周期文件句柄的 Store
        """
        self._data_root = Path(data_root)
        self.database = RuntimeStore(data_root)

    def create_session(self, session_id: str, *, created_at: str | None = None) -> Path:
        """批次创建会话身份；参数：会话编号与时间；返回：原件日志路径，已有身份不覆盖。"""
        self._validate_session_id(session_id)
        chosen = _now() if created_at is None else created_at
        self._validate_timestamp(chosen, session_id, "created_at")
        with self.database.transaction() as batch:
            if self.exists(session_id):
                raise self._error(
                    "session_exists", session_id, "session already exists"
                )
            batch.put(
                "session",
                session_id,
                {
                    "session_id": session_id,
                    "created_at": chosen,
                    "leaf_id": None,
                    "title": "",
                    "title_search": "",
                },
                session_id=session_id,
            )
        return self.database.source_path("session", session_id)

    def exists(self, session_id: str) -> bool:
        """按身份查询会话存在性；参数：会话编号；返回：是否已持久化。"""
        self._validate_session_id(session_id)
        with self.database.snapshot() as source:
            return source.get("session", session_id) is not None

    def list_session_ids(self) -> tuple[str, ...]:
        """列出持久会话供搜索和会话树使用；参数：无；返回：按创建时间倒序的编号。"""
        with self.database.snapshot() as source:
            rows = source.list("session")
            return tuple(
                row["session_id"]
                for row in sorted(
                    rows,
                    key=lambda row: (row["created_at"], row["session_id"]),
                    reverse=True,
                )
            )

    def current_leaf(self, session_id: str) -> str | None:
        """只读取当前已提交叶身份；传参：会话；返回：叶编号，新空会话返回None。"""
        self._validate_session_id(session_id)
        with self.database.snapshot() as source:
            row = source.get("session", session_id)
            return row["leaf_id"] if row is not None else None

    def read_entries(self, session_id: str) -> tuple[SessionEntry, ...]:
        """严格读取 header 与全部 Entry，不跳过任何损坏行。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：session_id 为目标会话
        返回：按文件顺序排列的不可变 Entry tuple；文件缺失抛稳定错误
        """
        with self.database.snapshot():
            rows = self._read_rows(session_id)
        return tuple(row for row in rows[1:] if isinstance(row, SessionEntry))

    def tree_page(
        self,
        session_id: str,
        *,
        after: int = 0,
        limit: int = DEFAULT_HISTORY_PAGE_SIZE,
        view: str = "turns",
        query: str = "",
        selected_entry: str | None = None,
    ) -> dict[str, Any]:
        """读取轮次/完整历史的轻量目录；参数：会话、分页、筛选及原选择；返回：摘要与实际分页位置。"""
        from runtime.session_tree import original_tree_label, project_tree_page

        self._validate_session_id(session_id)
        if type(after) is not int or after < 0 or type(limit) is not int or limit < 1:
            raise ValueError(
                "tree page cursor and limit must be nonnegative and positive integers"
            )
        with self.database.snapshot() as source:
            rows = source.list_raw("session_entry", session_id=session_id)
            originals = {row.payload["entry_id"]: row for row in rows}
            labels = None
            if query:
                labels = {
                    row.payload["entry_id"]: original_tree_label(
                        self.database,
                        row.payload,
                        originals[row.payload["input_id"]]
                        if row.payload["type"] == "delivery"
                        else row,
                    )
                    for row in rows
                }
            page = project_tree_page(
                [row.payload for row in rows],
                after=after,
                limit=limit,
                view=view,
                query=query,
                selected_entry=selected_entry,
                labels=labels,
            )
            selected = []
            for item in page["entries"]:
                row = originals[item["entry_id"]]
                original = (
                    originals[row.payload["input_id"]]
                    if row.payload["type"] == "delivery"
                    else row
                )
                label = (
                    labels[item["entry_id"]]
                    if labels is not None
                    else original_tree_label(self.database, row.payload, original)
                )
                selected.append({**item, "label": label})
            return {**page, "entries": selected}

    def read(self, session_id: str) -> tuple[SessionEntry, ...]:
        """读取 Session Entry，作为 read_entries 的简洁别名。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：session_id 为目标会话
        返回：严格校验后的 Entry tuple
        """
        return self.read_entries(session_id)

    def history_page(
        self,
        session_id: str,
        *,
        leaf_id: str | None = None,
        before: str | None = None,
        limit: int = DEFAULT_HISTORY_PAGE_SIZE,
    ) -> SessionHistoryPage:
        """按稳定分支锚点读取正文页；传参：会话、固定叶、排他游标、目标条数；返回：完整配对页。"""
        self._validate_session_id(session_id)
        if type(limit) is not int or limit < 1:
            raise ValueError("history page limit must be a positive integer")
        with self.database.snapshot() as source:
            header = source.get("session", session_id)
            if header is None:
                raise self._error(
                    "session_not_found", session_id, "session does not exist"
                )
            anchor = leaf_id if leaf_id is not None else header["leaf_id"]
            cursor = anchor
            if before is not None:
                ancestors: dict[str, Any] = {}
                while cursor is not None and cursor not in ancestors:
                    raw = source.raw("session_entry", record_key(session_id, cursor))
                    if raw is None:
                        raise self._error(
                            "entry_not_found",
                            session_id,
                            "history anchor is missing",
                            entry_id=cursor,
                        )
                    ancestors[cursor] = raw.payload
                    cursor = raw.payload["parent_id"]
                if before not in ancestors:
                    raise self._error(
                        "invalid_page_cursor",
                        session_id,
                        "cursor is outside the selected branch",
                        entry_id=before,
                    )
                cursor = ancestors[before]["parent_id"]
            entries: list[SessionEntry] = []
            seen: set[str] = set()
            while cursor is not None:
                if cursor in seen:
                    raise self._error(
                        "cycle",
                        session_id,
                        "history parent chain contains a cycle",
                        entry_id=cursor,
                    )
                seen.add(cursor)
                row = source.get("session_entry", record_key(session_id, cursor))
                if row is None:
                    raise self._error(
                        "entry_not_found",
                        session_id,
                        "history anchor is missing",
                        entry_id=cursor,
                    )
                entry = self._parse_entry(row, session_id, len(seen) + 1)
                if entry.entry_id != cursor or entry.parent_id != row["parent_id"]:
                    raise self._error(
                        "invalid_parent",
                        session_id,
                        "history record disagrees with indexed identity",
                        entry_id=cursor,
                    )
                entries.append(entry)
                cursor = entry.parent_id
                # 【会话】【历史分页】1. 工具结果不能成为页首，向前补到对应调用或用户输入
                if (
                    len(entries) >= limit
                    and entry.message is not None
                    and not isinstance(entry.message, ToolResultMessage)
                ):
                    break
            next_before = (
                entries[-1].entry_id if entries and cursor is not None else None
            )
            return SessionHistoryPage(
                session_id, anchor, tuple(reversed(entries)), next_before
            )

    def append_message(
        self,
        session_id: str,
        message: AgentMessage,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> SessionEntry:
        """追加一条 message Entry，并把当前 leaf 移到新 Entry。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：session_id/message 为消息归属和内容；run_id/task_id 为可选证据关联；
        返回：已持久化的新 Message Entry
        """
        if not _is_agent_message(message):
            raise self._error(
                "invalid_message", session_id, "value must be AgentMessage"
            )
        prepared = self.database.prepare_payload(
            {"message": agent_message_to_mapping(message)}, session_id=session_id
        )
        with self.write_lock(session_id):
            return self._append_message(
                session_id, message, run_id=run_id, task_id=task_id, prepared=prepared
            )

    def _append_message(
        self,
        session_id: str,
        message: AgentMessage,
        *,
        run_id: str | None,
        task_id: str | None,
        prepared: PreparedPayload,
    ) -> SessionEntry:
        """在会话锁内提交消息及父关系；传参：消息与归属；返回：持久Entry。"""
        if not _is_agent_message(message):
            raise self._error(
                "invalid_message", session_id, "value must be AgentMessage"
            )
        entries = self.read_entries_or_create(session_id)
        parent_id = entries[-1].entry_id if entries else None
        # 1. 新消息始终认事务当前 leaf 为父节点
        entry = self._new_entry(
            session_id,
            "message",
            parent_id=parent_id,
            message=message,
            run_id=run_id,
            task_id=task_id,
        )
        # 2. 写入前重新验证 leaf，追加后不改历史字节
        self._append_entry(
            session_id, entry, expected_parent=parent_id, prepared=prepared
        )
        return entry

    def branch(
        self,
        session_id: str,
        target_entry_id: str,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> SessionEntry:
        """追加 branch Entry，将当前 leaf 持久化回退到目标 Entry。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：session_id 为目标会话；target_entry_id 为已存在回退目标；其余为关联信息
        返回：已持久化的新 Branch Entry；历史 Entry 不被修改或删除
        """
        with self.write_lock(session_id):
            return self._branch(
                session_id, target_entry_id, run_id=run_id, task_id=task_id
            )

    def _branch(
        self,
        session_id: str,
        target_entry_id: str,
        *,
        run_id: str | None,
        task_id: str | None,
    ) -> SessionEntry:
        """在会话锁内提交分支，避免入站输入改变父关系；传参：目标与归属；返回：分支记录。"""
        entries = self.read_entries(session_id)
        if not any(item.entry_id == target_entry_id for item in entries):
            raise self._error(
                "missing_target",
                session_id,
                "branch target entry does not exist",
                entry_id=target_entry_id,
            )
        current_leaf = entries[-1].entry_id if entries else None
        # 1. branch 的父节点就是回退目标，旧分支继续保留
        entry = self._new_entry(
            session_id,
            "branch",
            parent_id=target_entry_id,
            message=None,
            run_id=run_id,
            task_id=task_id,
        )
        # 2. 当前 leaf 仅用于并发变化检测，不写入第二个 cursor
        self._append_entry(session_id, entry, expected_parent=current_leaf)
        return entry

    def materialize(
        self, session_id: str, *, at_entry_id: str | None = None
    ) -> MaterializedSession:
        """从当前持久 leaf 沿父链恢复当前分支的线性消息。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：session_id 为目标会话；at_entry_id 可固定已获授权的历史分支范围
        返回：当前路径 Entry、AgentMessage 和末尾 pending tool call；损坏或非法图抛错
        """
        entries = self.read_entries(session_id)
        if not entries:
            return MaterializedSession(session_id, None, (), (), ())
        # 1. 从最后提交的 Entry 沿 parent_id 回溯当前分支
        by_id = {entry.entry_id: entry for entry in entries}
        path: list[SessionEntry] = []
        if at_entry_id is not None and at_entry_id not in by_id:
            raise self._error(
                "entry_not_found",
                session_id,
                "source branch entry is missing",
                entry_id=at_entry_id,
            )
        cursor: SessionEntry | None = (
            by_id[at_entry_id] if at_entry_id is not None else entries[-1]
        )
        seen: set[str] = set()
        while cursor is not None:
            if cursor.entry_id in seen:
                raise self._error(
                    "cycle",
                    session_id,
                    "parent chain contains a cycle",
                    entry_id=cursor.entry_id,
                )
            seen.add(cursor.entry_id)
            path.append(cursor)
            cursor = (
                by_id.get(cursor.parent_id) if cursor.parent_id is not None else None
            )
        # 2. 反转为模型消费顺序，并跳过不携带消息的 branch Entry
        path.reverse()
        messages = _project_entry_messages(path, session_id)
        _validate_materialized_messages(
            session_id,
            messages,
            entry_id=path[-1].entry_id,
        )
        # 3. pending tool call 只作为中断恢复状态返回
        pending = _pending_tool_call_ids(messages)
        return MaterializedSession(
            session_id=session_id,
            leaf_id=path[-1].entry_id,
            entries=tuple(path),
            messages=messages,
            pending_tool_calls=pending,
        )

    def read_entries_or_create(self, session_id: str) -> tuple[SessionEntry, ...]:
        """读取现有会话，或在首次追加时原子创建 header。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：session_id 为目标会话
        返回：严格校验后的 Entry tuple
        """
        with self.write_lock(session_id):
            if not self.exists(session_id):
                self.create_session(session_id)
            return self.read_entries(session_id)

    def write_lock(self, session_id: str) -> AbstractContextManager[FileTransaction]:
        """复合会话写入共用文件批次；参数：会话身份；返回：可重入的跨进程提交窗口。"""
        self._validate_session_id(session_id)
        return self.database.transaction()

    def accept_input(
        self,
        session_id: str,
        text: str,
        *,
        input_id: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        input_source: Literal["user", "agent"] = "user",
        input_kind: Literal["approval"] | None = None,
        content: tuple[UserContentPart, ...] | None = None,
    ) -> SessionEntry:
        """保存可恢复输入正文，工具组未闭合时也不插入user消息；传参：输入和归属；返回：唯一入站记录。"""
        identity = input_id or f"input-{uuid4().hex}"
        message = UserMessage(
            identity, content if content is not None else (TextPart(text),)
        )
        prepared = self.database.prepare_payload(
            {"message": agent_message_to_mapping(message)}, session_id=session_id
        )
        with self.write_lock(session_id):
            entries = self.read_entries_or_create(session_id)
            existing = next(
                (item for item in entries if item.entry_id == identity), None
            )
            if existing is not None:
                if (
                    existing.type != "inbound"
                    or existing.message != message
                    or (existing.input_source or "user") != input_source
                    or existing.input_kind != input_kind
                ):
                    raise self._error(
                        "input_identity_conflict",
                        session_id,
                        "input_id has different content",
                    )
                return existing
            parent = entries[-1].entry_id if entries else None
            entry = SessionEntry(
                "inbound",
                identity,
                parent,
                session_id,
                _now(),
                run_id=run_id,
                task_id=task_id,
                message=message,
                input_source=input_source,
                input_kind=input_kind,
            )
            self._append_entry(
                session_id, entry, expected_parent=parent, prepared=prepared
            )
            return entry

    def pending_inputs(
        self, session_id: str, *, include_delivered: bool = False
    ) -> tuple[SessionEntry, ...]:
        """读取当前分支待交付输入；传参：会话及是否包含已投影记录；返回：按接纳顺序的正文引用。"""
        if not self.exists(session_id):
            return ()
        entries = self.materialize(session_id).entries
        delivered = {entry.input_id for entry in entries if entry.type == "delivery"}
        return tuple(
            entry
            for entry in entries
            if entry.type == "inbound"
            and entry.input_kind is None
            and (include_delivered or entry.entry_id not in delivered)
        )

    def deliver_inputs(
        self, session_id: str, *, run_id: str, task_id: str | None
    ) -> tuple[str, ...]:
        """调用组闭合后只投影一次入站正文；传参：会话和接收运行；返回：本次交付的输入编号。"""
        with self.write_lock(session_id):
            if not self.exists(session_id):
                return ()
            current = self.materialize(session_id)
            if current.pending_tool_calls:
                return ()
            pending = self.pending_inputs(session_id)
            parent = current.leaf_id
            for inbound in pending:
                entry = SessionEntry(
                    "delivery",
                    f"entry-{uuid4().hex}",
                    parent,
                    session_id,
                    _now(),
                    run_id=run_id,
                    task_id=task_id,
                    input_id=inbound.entry_id,
                )
                self._append_entry(session_id, entry, expected_parent=parent)
                parent = entry.entry_id
            return tuple(entry.entry_id for entry in pending)

    def _append_entry(
        self,
        session_id: str,
        entry: SessionEntry,
        *,
        expected_parent: str | None,
        prepared: PreparedPayload | None = None,
    ) -> None:
        """在严格读取通过后追加单行并 fsync，确保旧字节不被重写。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：session_id/entry 为目标文件及新事实；expected_parent 为读取时 leaf
        返回：无；追加失败抛稳定错误
        """
        current = self.read_entries(session_id)
        actual_parent = current[-1].entry_id if current else None
        if actual_parent != expected_parent:
            raise self._error(
                "concurrent_write", session_id, "session leaf changed during append"
            )
        if any(item.entry_id == entry.entry_id for item in current):
            raise self._error(
                "duplicate_entry_id",
                session_id,
                "entry_id already exists",
                entry_id=entry.entry_id,
            )
        # 1. 候选 Entry 在落盘前通过完整树和消息引用校验
        candidate = (*current, entry)
        self._validate_parent_graph(candidate, session_id)
        self._validate_entry_paths(candidate, session_id)
        # 【会话树】【追加事实】校验与提交共用事务，失败不会发布半条消息或错误的当前叶
        with self.database.transaction() as batch:
            payload = entry.to_mapping()
            candidate_payload: Mapping[str, Any] | PreparedPayload = (
                prepared.with_fields(
                    {key: value for key, value in payload.items() if key != "message"}
                )
                if prepared
                else payload
            )
            batch.put(
                "session_entry",
                record_key(session_id, entry.entry_id),
                candidate_payload,
                session_id=session_id,
            )
            header = batch.get("session", session_id)
            if header is None:
                raise self._error(
                    "session_not_found", session_id, "session does not exist"
                )
            header = {**header, "leaf_id": entry.entry_id}
            if (
                entry.type == "inbound"
                and isinstance(entry.message, UserMessage)
                and entry.input_kind is None
            ):
                from llm.messages import model_visible_text

                # 【会话】【目录标题】1. 首条输入提交时维护有界标题，列表不解码全部输入正文
                title = model_visible_text(entry.message)[:SESSION_TITLE_LENGTH]
                if not header["title"]:
                    header = {
                        **header,
                        "title": title,
                        "title_search": title.casefold(),
                    }
            batch.put("session", session_id, header, session_id=session_id)

    def _read_rows(self, session_id: str) -> tuple[SessionHeader | SessionEntry, ...]:
        """读取会话和不可变条目，损坏不返回旧前缀；参数：会话；返回：头信息和完整事实。"""
        self._validate_session_id(session_id)
        with self.database.snapshot() as source:
            header = source.get("session", session_id)
            if header is None:
                raise self._error(
                    "session_not_found", session_id, "session does not exist"
                )
            rows = source.list("session_entry", session_id=session_id)
            entries = tuple(
                self._parse_entry(row, session_id, index + 2)
                for index, row in enumerate(rows)
            )
            actual_leaf = entries[-1].entry_id if entries else None
            if header["leaf_id"] != actual_leaf:
                raise self._error(
                    "invalid_leaf",
                    session_id,
                    "session leaf does not match committed entries",
                )
        self._validate_parent_graph(entries, session_id)
        self._validate_entry_paths(entries, session_id)
        return (SessionHeader(session_id, header["created_at"]), *entries)

    def _parse_entry(
        self, row: Mapping[str, object], session_id: str, line_number: int
    ) -> SessionEntry:
        """解析一行 message/branch Entry，并调用 llm.messages 严格 serde。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：row 为 JSON object；session_id 为期望会话；line_number 为定位行
        返回：严格校验后的不可变 SessionEntry
        """
        context = _EntryParseContext(session_id, line_number)
        entry_type = self._validate_entry_schema(row, context)
        identity = self._parse_entry_identity(row, context)
        message = self._parse_entry_message(
            row,
            entry_type,
            context=context,
            entry_id=identity.entry_id,
        )
        try:
            return SessionEntry(
                type=entry_type,
                entry_id=identity.entry_id,
                parent_id=identity.parent_id,
                session_id=session_id,
                timestamp=identity.timestamp,
                run_id=identity.run_id,
                task_id=identity.task_id,
                message=message,
                input_id=_optional_text(row.get("input_id"), "input_id", context),
                input_source=cast(
                    Literal["user", "agent"] | None, row.get("input_source")
                ),
                input_kind=cast(Literal["approval"] | None, row.get("input_kind")),
            )
        except ValueError as exc:
            raise self._error(
                "invalid_entry",
                session_id,
                str(exc),
                line_number=line_number,
                entry_id=identity.entry_id,
            ) from exc

    def _validate_entry_schema(
        self,
        row: Mapping[str, object],
        context: _EntryParseContext,
    ) -> SessionEntryType:
        """校验 Entry discriminator 与 message/branch 字段集合。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：row 为 JSON object；context 为会话和行号定位
        返回：已收窄为 message 或 branch 的 Entry 类型
        """
        entry_type = row.get("type")
        if not isinstance(entry_type, str) or entry_type not in SESSION_ENTRY_TYPES:
            raise self._error(
                "invalid_entry",
                context.session_id,
                "entry type must be message or branch",
                line_number=context.line_number,
            )
        common = {
            "type",
            "entry_id",
            "parent_id",
            "session_id",
            "timestamp",
            "run_id",
            "task_id",
        }
        allowed = common | (
            {"message"} if entry_type in {"message", "inbound"} else set()
        )
        if entry_type == "delivery":
            allowed.add("input_id")
        if entry_type == "inbound":
            allowed.add("input_source")
            allowed.add("input_kind")
        required = {"type", "entry_id", "parent_id", "session_id", "timestamp"}
        if entry_type in {"message", "inbound"}:
            required.add("message")
        if entry_type == "delivery":
            required.add("input_id")
        self._validate_keys(
            row,
            allowed,
            required,
            session_id=context.session_id,
            line_number=context.line_number,
        )
        return cast(SessionEntryType, entry_type)

    def _parse_entry_identity(
        self,
        row: Mapping[str, object],
        context: _EntryParseContext,
    ) -> _EntryIdentity:
        """解析并校验 Entry 的身份、父关系与可选运行关联。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：row 为字段已冻结的 Entry object；context 为会话和行号定位
        返回：不可变 Entry 身份字段集合
        """
        entry_id = _required_text(row.get("entry_id"), "entry_id", context)
        entry_session = _required_text(row.get("session_id"), "session_id", context)
        timestamp = _required_text(row.get("timestamp"), "timestamp", context)
        if entry_session != context.session_id:
            raise self._error(
                "session_mismatch",
                context.session_id,
                "entry session_id does not match requested session",
                line_number=context.line_number,
                entry_id=entry_id,
            )
        parent_id = _optional_text(
            row.get("parent_id"), "parent_id", context, code="invalid_parent"
        )
        return _EntryIdentity(
            entry_id=entry_id,
            parent_id=parent_id,
            timestamp=timestamp,
            run_id=_optional_text(row.get("run_id"), "run_id", context),
            task_id=_optional_text(row.get("task_id"), "task_id", context),
        )

    def _parse_entry_message(
        self,
        row: Mapping[str, object],
        entry_type: SessionEntryType,
        *,
        context: _EntryParseContext,
        entry_id: str,
    ) -> AgentMessage | None:
        """通过 llm.messages 严格 serde 解析 message Entry 载荷。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：row/type 为 Entry 数据；context/entry_id 为错误定位
        返回：message Entry 的 AgentMessage；branch Entry 返回 None
        """
        if entry_type in {"branch", "delivery"}:
            return None
        try:
            return agent_message_from_mapping(
                row.get("message"),
                path=f"entry[{context.line_number}].message",
            )
        except MessageContractError as exc:
            raise self._error(
                "invalid_message",
                context.session_id,
                exc.detail,
                line_number=context.line_number,
                entry_id=entry_id,
            ) from exc

    def _validate_parent_graph(
        self, entries: Sequence[SessionEntry], session_id: str
    ) -> None:
        """验证根节点、父链存在性、顺序和循环。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：entries 为文件顺序 Entry；session_id 为错误定位会话
        返回：无；非法 parent、断链或循环抛稳定错误
        """
        by_id = {entry.entry_id: entry for entry in entries}
        positions = {entry.entry_id: index for index, entry in enumerate(entries)}
        for index, entry in enumerate(entries):
            line_number = index + 2
            if index == 0 and entry.parent_id is None:
                if entry.type == "branch":
                    raise self._error(
                        "invalid_parent",
                        session_id,
                        "first entry must be a root message, not a branch",
                        line_number=line_number,
                        entry_id=entry.entry_id,
                    )
                continue
            if entry.parent_id is None:
                raise self._error(
                    "invalid_parent",
                    session_id,
                    "only the first entry may have null parent_id",
                    line_number=line_number,
                    entry_id=entry.entry_id,
                )
            if entry.parent_id not in by_id:
                raise self._error(
                    "missing_parent",
                    session_id,
                    "parent entry does not exist",
                    line_number=line_number,
                    entry_id=entry.entry_id,
                )
            if positions[entry.parent_id] < index:
                continue
            code = "cycle" if _has_cycle(entry, by_id) else "invalid_parent"
            detail = (
                "parent chain contains a cycle"
                if code == "cycle"
                else "parent entry must appear earlier in file"
            )
            raise self._error(
                code,
                session_id,
                detail,
                line_number=line_number,
                entry_id=entry.entry_id,
            )

    def _validate_entry_paths(
        self,
        entries: Sequence[SessionEntry],
        session_id: str,
    ) -> None:
        """验证每个保留分支的消息引用图。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：entries 为已通过父图校验的 Entry；session_id 为错误定位会话
        返回：无；任一 leaf 路径存在非法消息引用时抛稳定错误
        """
        by_id = {entry.entry_id: entry for entry in entries}
        parent_ids = {
            entry.parent_id for entry in entries if entry.parent_id is not None
        }
        leaves = [entry for entry in entries if entry.entry_id not in parent_ids]
        for leaf in leaves:
            # 1. 单独恢复每个保留 leaf 的根到叶路径
            path: list[SessionEntry] = []
            cursor: SessionEntry | None = leaf
            while cursor is not None:
                path.append(cursor)
                cursor = by_id.get(cursor.parent_id) if cursor.parent_id else None
            # 2. 每条 materialized 路径独立校验 message_id 与工具调用图
            messages = _project_entry_messages(tuple(reversed(path)), session_id)
            _validate_materialized_messages(
                session_id,
                messages,
                entry_id=leaf.entry_id,
            )

    def _validate_keys(
        self,
        row: Mapping[str, object],
        allowed: set[str],
        required: set[str],
        *,
        session_id: str,
        line_number: int,
    ) -> None:
        """校验 JSON object 的冻结字段集合。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：row 为目标对象；allowed/required 为字段合同；session_id/line_number 为定位
        返回：无；未知或缺失字段抛稳定错误
        """
        unknown = set(row) - allowed
        if unknown:
            raise self._error(
                "unknown_field",
                session_id,
                f"unknown fields: {sorted(unknown)}",
                line_number=line_number,
            )
        missing = required - set(row)
        if missing:
            raise self._error(
                "missing_field",
                session_id,
                f"missing fields: {sorted(missing)}",
                line_number=line_number,
            )

    def _new_entry(
        self,
        session_id: str,
        entry_type: SessionEntryType,
        *,
        parent_id: str | None,
        message: AgentMessage | None,
        run_id: str | None,
        task_id: str | None,
    ) -> SessionEntry:
        """构造待追加 Entry 并统一生成身份和北京时间时间戳。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：session_id/type/parent/message 为树事实；运行关联和稳定身份为可选值
        返回：通过字段校验但尚未持久化的 SessionEntry
        """
        chosen_id = f"entry-{uuid4().hex}"
        chosen_timestamp = _now()
        try:
            return SessionEntry(
                type=entry_type,
                entry_id=chosen_id,
                parent_id=parent_id,
                session_id=session_id,
                timestamp=chosen_timestamp,
                run_id=run_id,
                task_id=task_id,
                message=message,
            )
        except ValueError as exc:
            raise self._error(
                "invalid_entry", session_id, str(exc), entry_id=chosen_id
            ) from exc

    def _validate_timestamp(
        self, value: object, session_id: str, field_name: str
    ) -> None:
        """确认 Store 生成或接收的时间字段为非空文本。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：value 为待检时间；session_id/field_name 为错误定位
        返回：无；空时间值抛 invalid_timestamp
        """
        if not _non_empty_text(value):
            raise self._error(
                "invalid_timestamp", session_id, f"{field_name} must be non-empty text"
            )

    def _validate_session_id(self, session_id: str) -> None:
        """拒绝空会话标识和路径穿越。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：session_id 为路径中的候选会话标识
        返回：无；非法路径组件抛 invalid_session_id
        """
        if (
            not _non_empty_text(session_id)
            or Path(session_id).name != session_id
            or session_id in {".", ".."}
        ):
            raise self._error(
                "invalid_session_id",
                str(session_id),
                "session_id must be a single non-empty path component",
            )

    @staticmethod
    def _error(
        code: str, session_id: str, detail: str, **kwargs: object
    ) -> SessionMessageStoreError:
        """创建不泄露 data root 的稳定 Store 错误。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：code/session_id/detail 为错误主体；kwargs 为行号或 Entry 定位
        返回：尚未抛出的 SessionMessageStoreError
        """
        return SessionMessageStoreError(
            code, session_id, detail, **cast(dict[str, Any], kwargs)
        )


def _validate_materialized_messages(
    session_id: str,
    messages: Sequence[AgentMessage],
    *,
    entry_id: str | None = None,
) -> None:
    """校验一条 materialized 消息路径及末尾 pending call。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：session_id 为错误定位；messages 为根到叶消息；entry_id 为 leaf 标识
    返回：无；消息图非法或 pending call 不在末尾时抛稳定错误
    """
    try:
        validate_message_sequence(messages, allow_pending=True)
    except MessageContractError as exc:
        raise SessionMessageStoreError(
            "invalid_message_sequence",
            session_id,
            exc.detail,
            entry_id=entry_id,
        ) from exc


def _pending_tool_call_ids(messages: Sequence[AgentMessage]) -> tuple[str, ...]:
    """计算已发起但尚未收到结果的工具调用。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：messages 为已通过基本消息 serde 的根到叶序列
    返回：按首次出现顺序排列的 pending call_id tuple
    """
    pending: dict[str, None] = {}
    for message in messages:
        if isinstance(message, ToolResultMessage):
            pending.pop(message.call_id, None)
            continue
        if not isinstance(message, AssistantMessage):
            continue
        for part in message.content:
            if isinstance(part, ToolCallPart):
                pending[part.call_id] = None
    return tuple(pending)


def _has_cycle(entry: SessionEntry, by_id: Mapping[str, SessionEntry]) -> bool:
    """沿指定 Entry 的父链检测循环。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：entry 为起点；by_id 为同文件 Entry 索引
    返回：父链形成循环时为 True，否则为 False
    """
    seen: set[str] = set()
    cursor: SessionEntry | None = entry
    while cursor is not None and cursor.parent_id is not None:
        if cursor.entry_id in seen:
            return True
        seen.add(cursor.entry_id)
        cursor = by_id.get(cursor.parent_id)
    return cursor is not None and cursor.entry_id in seen


def _reject_json_constant(value: str) -> object:
    """拒绝 NaN/Infinity 等非标准 JSON 常量。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为 json parser 识别的非标准常量名
    返回：不返回；始终抛 ValueError
    """
    raise ValueError(f"non-standard JSON constant: {value}")


def _non_empty_text(value: object) -> bool:
    """判断任意值是否为非空白文本。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为候选字段值
    返回：字符串包含非空白字符时为 True
    """
    return isinstance(value, str) and bool(value.strip())


def _required_text(
    value: object,
    field_name: str,
    context: _EntryParseContext,
) -> str:
    """解析 Entry 中必填且不可为空白的文本字段。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value/field_name 为字段值与名称；context 为会话和行号定位
    返回：已验证的非空文本
    """
    if not _non_empty_text(value):
        raise SessionMessageStoreError(
            "invalid_entry",
            context.session_id,
            f"{field_name} must be non-empty text",
            line_number=context.line_number,
        )
    return cast(str, value)


def _optional_text(
    value: object,
    field_name: str,
    context: _EntryParseContext,
    *,
    code: str = "invalid_entry",
) -> str | None:
    """解析可空但不可为空白的 Entry 文本字段。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value/field_name 为字段值与名称；context 为定位；code 为错误码
    返回：None 或已验证的非空文本
    """
    if value is None:
        return None
    if not _non_empty_text(value):
        raise SessionMessageStoreError(
            code,
            context.session_id,
            f"{field_name} must be null or non-empty text",
            line_number=context.line_number,
        )
    return cast(str, value)


def _is_agent_message(value: object) -> bool:
    """判断值是否为 Child 1 冻结的 AgentMessage 联合类型。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为 append_message 收到的候选消息
    返回：User/Assistant/ToolResultMessage 时为 True
    """
    return isinstance(value, (UserMessage, AssistantMessage, ToolResultMessage))


def _now() -> str:
    """生成 Store 默认使用的北京时间 ISO 时间戳。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：无
    返回：精确到秒且携带 UTC+8 偏移的时间文本
    """
    return datetime.now(_BEIJING).isoformat(timespec="seconds")


__all__ = [
    "MaterializedSession",
    "SessionEntry",
    "SessionHeader",
    "SessionMessageStore",
    "SessionMessageStoreError",
]


def _project_entry_messages(
    entries: Sequence[SessionEntry], session_id: str
) -> tuple[AgentMessage, ...]:
    """按交付位置投影唯一正文，入站和分支控制记录不重复进模型；传参：分支路径与会话；返回：消息序列。"""
    inputs: dict[str, AgentMessage] = {}
    delivered: set[str] = set()
    messages: list[AgentMessage] = []
    for entry in entries:
        if entry.type == "inbound" and entry.input_kind is None:
            inputs[entry.entry_id] = cast(AgentMessage, entry.message)
        elif entry.type == "message":
            messages.append(cast(AgentMessage, entry.message))
        elif entry.type == "delivery":
            if entry.input_id not in inputs or entry.input_id in delivered:
                raise SessionMessageStoreError(
                    "invalid_input_delivery",
                    session_id,
                    "input must belong to this branch and be delivered once",
                )
            messages.append(inputs[entry.input_id])
            delivered.add(entry.input_id)
    return tuple(messages)
