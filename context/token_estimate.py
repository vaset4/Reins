from __future__ import annotations

import json
import re
from math import ceil
from typing import Sequence

from llm.messages import (
    AgentMessage,
    ImagePart,
    agent_message_to_mapping,
    content_part_to_mapping,
)

# 单条消息除正文外的固定包装开销（角色标识与消息分隔），历史预算与裁剪共用
MESSAGE_OVERHEAD_TOKENS = 4

_CJK_RANGES = re.compile(
    r"[一-鿿㐀-䶿豈-﫿"
    r"\U00020000-\U0002a6df\U0002a700-\U0002b73f"
    r"\U0002b740-\U0002b81f\U0002b820-\U0002ceaf"
    r"\U0002ceb0-\U0002ebef\U00030000-\U0003134f"
    r"　-〿＀-￯]+"
)

CJK_CHARS_PER_TOKEN = 1.5
WORDS_TO_TOKENS_RATIO = 1.3
LATIN_CHARS_PER_TOKEN = 4
IMAGE_PATCH_PIXELS = 32


def estimate_tokens(text: str) -> int:
    """估算中英文、代码和无空格长串；传参：正文；返回：估算token数，不替代供应商用量。"""
    if not text:
        return 0
    cjk_chars = 0
    non_cjk_parts: list[str] = []
    last_end = 0
    for match in _CJK_RANGES.finditer(text):
        cjk_chars += len(match.group())
        non_cjk_parts.append(text[last_end : match.start()])
        last_end = match.end()
    non_cjk_parts.append(text[last_end:])

    non_cjk_text = " ".join(non_cjk_parts)
    word_count = len(non_cjk_text.split())

    cjk_tokens = cjk_chars / CJK_CHARS_PER_TOKEN
    word_tokens = max(
        word_count * WORDS_TO_TOKENS_RATIO, len(non_cjk_text) / LATIN_CHARS_PER_TOKEN
    )

    return max(1, round(cjk_tokens + word_tokens))


def estimate_messages_tokens(messages: list[dict[str, object]]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(str(msg.get("content", "")))
        total += MESSAGE_OVERHEAD_TOKENS
    return total


def estimate_agent_messages_tokens(messages: Sequence[AgentMessage]) -> int:
    """按完整内容与包装估算 canonical 消息序列的 token 占用。

    作者：LKX
    时间：2026-08-30 14:20:00
    传参：messages 为按发送顺序排列的 canonical 消息
    返回：估算 token 数

    工具参数、产物引用和Provider状态也占用请求窗口，不能只统计模型可见正文。
    历史选择、请求预算和摘要准备共用此估算；实际消费仍以供应商报告为准。
    """
    total = 0
    for message in messages:
        payload = agent_message_to_mapping(message)
        content: list[dict[str, object]] = []
        for part in message.content:
            if not isinstance(part, ImagePart):
                content.append(content_part_to_mapping(part))
                continue
            # 【上下文】【图片估算】1. 传输base64不是文本token；采用官方32像素patch基础计数，非模型账单
            width, height = part.width, part.height
            if width is None or height is None:
                from llm.image_input import image_base64, image_from_bytes
                import base64

                measured = image_from_bytes(base64.b64decode(image_base64(part)))
                width, height = measured.width, measured.height
            assert width is not None and height is not None
            total += ceil(width / IMAGE_PATCH_PIXELS) * ceil(
                height / IMAGE_PATCH_PIXELS
            )
            content.append(
                {
                    "kind": "image",
                    "mime_type": part.mime_type,
                    "width": width,
                    "height": height,
                }
            )
        payload["content"] = content
        total += (
            estimate_tokens(json.dumps(payload, ensure_ascii=False))
            + MESSAGE_OVERHEAD_TOKENS
        )
    return total


__all__ = [
    "MESSAGE_OVERHEAD_TOKENS",
    "estimate_agent_messages_tokens",
    "estimate_messages_tokens",
    "estimate_tokens",
]
