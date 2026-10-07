"""全屏草稿中的附件与文件引用语法。

作者：xxx
时间：2026-09-29 22:00:00
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttachmentDraft:
    """保存原正文及用户明确选择的路径，不在界面读取文件。"""

    text: str
    attachment_paths: tuple[str, ...]
    reference_paths: tuple[str, ...]


def parse_attachment_draft(text: str) -> AttachmentDraft:
    """解析独立的@file/@ref行；参数：完整草稿；返回：正文和有序路径，保留Windows反斜线。"""
    body: list[str] = []
    attachments: list[str] = []
    references: list[str] = []
    for line in text.splitlines():
        command, _, value = line.strip().partition(" ")
        if command not in {"@file", "@ref"}:
            body.append(line)
            continue
        path = value.strip()
        if path.startswith(('"', "'")):
            if len(path) < 2 or path[-1] != path[0]:
                raise ValueError("文件路径的引号未闭合")
            path = path[1:-1]
        if not path or "\x00" in path:
            raise ValueError("请在 @file 或 @ref 后填写文件路径")
        (attachments if command == "@file" else references).append(path)
    return AttachmentDraft("\n".join(body), tuple(attachments), tuple(references))
