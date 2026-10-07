"""保存协作关系、定向信件与共同决定，接收者会话仍独占模型消息。

作者：xxx
时间：2026-09-15 01:00:00
"""

from __future__ import annotations

import json
from typing import Any

from runtime.session_message_store import SessionMessageStore
from runtime.types import RunContext
from tasks.ids import utc_now


class CollaborationStore:
    """一个主会话拥有一份协作记录；邮箱的接纳与模型的实际观察分别留证。"""

    def __init__(self, messages: SessionMessageStore, root_session_id: str) -> None:
        """复用现有会话写锁与原子文件发布；传参：会话存储和整合者身份；返回：无。"""
        self.messages, self.root_session_id = messages, root_session_id
        self._db = messages.database

    def read(self) -> dict[str, Any]:
        """读取完整事实，已有损坏文件不能被空记录替代；传参：无；返回：独立记录。"""
        with self._db.snapshot() as source:
            value = source.get("collaboration", self.root_session_id)
            if value is None:
                return {
                    "schema_version": 1,
                    "root_session_id": self.root_session_id,
                    "revision": 0,
                    "members": {},
                    "messages": {},
                    "decisions": {},
                    "user_inputs": [],
                }
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != 1
                or value.get("root_session_id") != self.root_session_id
            ):
                raise ValueError("invalid collaboration record")
            for key in ("members", "messages", "decisions"):
                if not isinstance(value.get(key), dict):
                    raise ValueError(f"invalid collaboration {key}")
            if (
                not isinstance(value.get("user_inputs"), list)
                or type(value.get("revision")) is not int
            ):
                raise ValueError("invalid collaboration input history or revision")
            return value

    def add_member(self, member: dict[str, Any]) -> dict[str, Any]:
        """按发起操作去重登记执行者；传参：已验证的成员身份；返回：原记录或新成员。"""
        with self._db.transaction():
            current = self.read()
            existing = current["members"].get(member["agent_id"])
            if existing is not None:
                return dict(existing)
            if any(
                item["name"].casefold() == member["name"].casefold()
                for item in current["members"].values()
            ):
                raise ValueError("agent name is already in use in this collaboration")
            self._save(
                {
                    **current,
                    "members": {**current["members"], member["agent_id"]: member},
                }
            )
            return dict(member)

    def update_member(self, agent_id: str, **detail: object) -> dict[str, Any]:
        """更新运行关联和产物引用，生命周期另由运行事实负责；传参：成员与关联字段；返回：新记录。"""
        with self._db.transaction():
            current = self.read()
            member = {**current["members"][agent_id], **detail}
            self._save({**current, "members": {**current["members"], agent_id: member}})
            return member

    def send(
        self,
        source: RunContext,
        recipient: str,
        *,
        message_id: str,
        text: str,
        kind: str = "peer",
        reference: str = "",
    ) -> dict[str, Any]:
        """先持久化定向信件再交付，发送者由运行身份确定；传参：来源、接收者与内容；返回：信件凭据。"""
        envelope = {
            "message_id": message_id,
            "sender_session_id": source.session_id,
            "sender_run_id": source.run_id,
            "recipient_session_id": recipient,
            "kind": kind,
            "text": text,
            "reference": reference,
        }
        existing = self._existing_message(self.read(), envelope)
        # 1. 【协作】【重复投递】已接纳原件就是持久凭据，重投只核对身份内容，不再进入空写事务
        if existing is not None and existing["accepted"]:
            return dict(existing)
        with self._db.transaction():
            current = self.read()
            if self._existing_message(current, envelope) is None:
                self._save(
                    {
                        **current,
                        "messages": {
                            **current["messages"],
                            message_id: {
                                **envelope,
                                "accepted": False,
                                "created_at": utc_now(),
                            },
                        },
                    }
                )
            self.deliver_pending()
            return dict(self.read()["messages"][message_id])

    def _existing_message(
        self, current: dict[str, Any], envelope: dict[str, str]
    ) -> dict[str, Any] | None:
        """核对重投身份对应的原始内容和接收者；参数：已读邮箱与待投信件；返回：原凭据或不存在。"""
        existing: dict[str, Any] | None = current["messages"].get(
            envelope["message_id"]
        )
        if existing is not None and any(
            existing.get(key) != value
            for key, value in envelope.items()
            if key != "sender_run_id"
        ):
            raise ValueError(
                "collaboration message identity has different content or recipient"
            )
        return existing

    def deliver_pending(self) -> tuple[str, ...]:
        """接收者Session提交后才确认邮箱接纳；传参：无；返回：本次接纳的接收者，崩溃可按原ID重投。"""
        if all(envelope["accepted"] for envelope in self.read()["messages"].values()):
            return ()
        with self._db.transaction():
            current = self.read()
            accepted: list[str] = []
            envelopes = dict(current["messages"])
            for identity, envelope in current["messages"].items():
                if envelope["accepted"]:
                    continue
                recipient = envelope["recipient_session_id"]
                text = json.dumps(
                    {
                        key: envelope[key]
                        for key in (
                            "kind",
                            "sender_session_id",
                            "sender_run_id",
                            "reference",
                            "text",
                        )
                    },
                    ensure_ascii=False,
                )
                self.messages.accept_input(
                    recipient,
                    text,
                    input_id=identity,
                    input_source="user"
                    if envelope["kind"] == "user_update"
                    else "agent",
                )
                envelopes[identity] = {**envelope, "accepted": True}
                accepted.append(recipient)
            if accepted:
                self._save({**current, "messages": envelopes})
            return tuple(accepted)

    def remember_user_input(self, input_id: str) -> int:
        """给共同用户输入分配稳定版本，正文引用原Session；传参：原输入ID；返回：版本号。"""
        with self._db.transaction():
            current = self.read()
            previous = current["user_inputs"]
            if input_id in previous:
                return int(previous.index(input_id)) + 1
            self._save({**current, "user_inputs": [*previous, input_id]})
            return len(previous) + 1

    def decide(
        self,
        source: RunContext,
        *,
        key: str,
        text: str,
        expected_version: int,
        operation_id: str,
    ) -> dict[str, Any]:
        """由整合者按版本发布共同决定；传参：运行、主题、正文与预期版本；返回：有来源的新版本。"""
        if source.session_id != self.root_session_id:
            raise ValueError(
                "only the integrating parent can publish a shared decision; send a proposal instead"
            )
        with self._db.transaction():
            current = self.read()
            versions = current["decisions"].get(key, [])
            existing = next(
                (item for item in versions if item.get("operation_id") == operation_id),
                None,
            )
            if existing is not None:
                if (
                    existing["text"] != text
                    or existing["source_session_id"] != source.session_id
                ):
                    raise ValueError("shared decision operation identity changed")
                return dict(existing)
            if expected_version != len(versions):
                raise ValueError(
                    f"shared decision version conflict: expected {expected_version}, current {len(versions)}"
                )
            decision = {
                "key": key,
                "version": len(versions) + 1,
                "text": text,
                "source_session_id": source.session_id,
                "source_run_id": source.run_id,
                "source_input_id": source.payload.get("input_message_id"),
                "created_at": utc_now(),
                "operation_id": operation_id,
                "source_task_id": source.focus_task_id,
            }
            self._save(
                {
                    **current,
                    "decisions": {**current["decisions"], key: [*versions, decision]},
                }
            )
            return decision

    def _save(self, value: dict[str, Any]) -> None:
        """以同一事实源原子发布修订；传参：新记录；返回：无，必要写入失败直接暴露。"""
        with self._db.transaction() as batch:
            batch.put(
                "collaboration",
                self.root_session_id,
                {**value, "revision": value["revision"] + 1},
                session_id=self.root_session_id,
                expected_revision=value["revision"],
            )
