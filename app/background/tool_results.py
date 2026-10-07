"""按已保存工具交换核验原件归属，仅为明确的详情阅读读取产物。

作者：xxx
时间：2026-09-29 20:00:00
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any

from artifacts.store import ArtifactStore
from llm.messages import ToolResultMessage, model_visible_text
from runtime.session_message_store import SessionEntry, SessionMessageStore
from runtime.tool_operations import ToolOperationStore
from tools.read_artifact import read_artifact


def retained_result_source(entry: SessionEntry) -> dict[str, str] | None:
    """仅发布工具原件的会话来源，不预读产物；参数：持久条目；返回：来源或无原件提示。"""
    message = entry.message
    if not isinstance(message, ToolResultMessage) or not message.artifact_refs:
        return None
    try:
        payload = json.loads(model_visible_text(message))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
        return None
    if not payload["meta"].get("result_artifact_id"):
        return None
    return {
        "session_id": entry.session_id,
        "run_id": entry.run_id or "",
        "entry_id": entry.entry_id,
        "call_id": message.call_id,
    }


def tool_result_detail(
    data_root: Path, *, session_id: str, run_id: str, entry_id: str, call_id: str
) -> dict[str, Any]:
    """核对会话、运行、调用和操作记录后读取完整原件；参数：数据根及来源身份；返回：完整正文。"""
    page = SessionMessageStore(data_root).history_page(
        session_id, leaf_id=entry_id, limit=1
    )
    entry = page.entries[-1]
    message = entry.message
    if (
        entry.entry_id != entry_id
        or entry.run_id != run_id
        or not isinstance(message, ToolResultMessage)
        or message.call_id != call_id
    ):
        raise ValueError("工具结果不属于指定会话、运行或调用")
    payload = json.loads(model_visible_text(message))
    if not isinstance(payload, dict) or payload.get("tool_name") != message.tool_name:
        raise ValueError("已保存回执没有可核验的工具原件来源")
    meta = payload.get("meta", {})
    if (
        not isinstance(meta, dict)
        or meta.get("call_id") != call_id
        or not isinstance(meta.get("operation_id"), str)
    ):
        raise ValueError("已保存回执缺少对应操作身份")
    operation = ToolOperationStore(data_root).load(
        {
            "session_id": session_id,
            "run_id": run_id,
            "operation_id": meta["operation_id"],
        }
    )
    if operation is None:
        raise ValueError("工具操作记录不存在，无法核验原件归属")
    call = operation["call"]
    artifact_id = meta.get("result_artifact_id")
    saved_meta = (operation.get("result") or {}).get("meta", {})
    # 【TUI】【工具原件】1. UI 不能指定产物或路径；正文提示必须与同一操作的持久引用一致
    if (
        call["call_id"] != call_id
        or call["tool_name"] != message.tool_name
        or not isinstance(artifact_id, str)
        or artifact_id not in message.artifact_refs
        or saved_meta.get("result_artifact_id") != artifact_id
    ):
        raise ValueError("工具原件引用与已提交操作不一致")
    with closing(ArtifactStore(data_root)) as artifacts:
        artifact = artifacts.load_artifact(artifact_id)
    if artifact is None:
        raise FileNotFoundError(f"工具原件不存在：{artifact_id}")
    if call.get("task_id") is not None and artifact.task_id != call["task_id"]:
        raise ValueError("工具原件不属于该操作的目标")
    return {
        "session_id": session_id,
        "run_id": run_id,
        "entry_id": entry_id,
        "call_id": call_id,
        "artifact_id": artifact_id,
        "text": read_artifact(data_root, artifact_id, mode="full"),
    }
