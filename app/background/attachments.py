"""后台附件输入：沿用文件读取权限及会话脱敏，保存实际读取的正文。

作者：xxx
时间：2026-09-29 22:00:00
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.background.image_attachments import read_image_attachment
from llm.messages import TextPart, UserContentPart
from runtime.lease import Lease
from runtime.types import ReadOnlyInspectionRequest
from tools.readonly_inspection import DEFAULT_READ_MAX_CHARS, ReadOnlyInspectionExecutor
from tools.redacted_files import RedactedFiles


def validate_paths(value: object) -> tuple[str, ...]:
    """校验控制接口的路径数组；参数：外部JSON值；返回：有序路径，非法输入明确失败。"""
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(path, str) or not path.strip() or "\x00" in path
        for path in value
    ):
        raise ValueError("attachment paths must be an array of non-empty strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class AttachmentInput:
    """正文摘要用于运行定位，完整内容块用于原始消息与模型输入。"""

    text: str
    content: tuple[UserContentPart, ...]


def compose_attachment_input(
    text: str,
    *,
    attachment_paths: tuple[str, ...],
    reference_paths: tuple[str, ...],
    project_root: Path,
    lease: Lease,
    redacted_files: RedactedFiles,
    session_id: str,
) -> AttachmentInput:
    """组装真实附件材料；参数：正文、路径和宿主读取依赖；返回：可持久化文本，失败不接纳输入。"""
    attachments, references = (
        validate_paths(attachment_paths),
        validate_paths(reference_paths),
    )
    if not attachments and not references:
        return AttachmentInput(text, (TextPart(text),))
    parts: list[UserContentPart] = [TextPart(text)] if text.strip() else []
    # 【附件】【读取】1. 复用原文件读取的范围、脱敏、PDF抽取及分页合同
    reader = ReadOnlyInspectionExecutor(project_root, 1, DEFAULT_READ_MAX_CHARS, 1)
    for target in attachments:
        image = read_image_attachment(target, project_root=project_root, lease=lease)
        if image is not None:
            parts.extend(
                (TextPart(f"[图片附件：{target}，{image.width}×{image.height}]"), image)
            )
            continue
        result = reader.execute(
            ReadOnlyInspectionRequest(action="read_file", target_path=target),
            lease=lease,
            redacted_files=redacted_files,
            session_id=session_id,
        )
        if result.status != "ok":
            if result.meta.get("exception_type") == "UnicodeDecodeError":
                raise ValueError(
                    f"附件不是UTF-8文本：{target}；当前不支持直接发送二进制媒体"
                )
            reason = result.output or result.error or "读取失败"
            raise ValueError(f"附件读取失败：{target}：{reason}")
        if "\x00" in result.output:
            raise ValueError(
                f"附件不是可读取的文本：{target}；当前不支持直接发送二进制媒体"
            )
        metadata = json.dumps(result.meta, ensure_ascii=False, sort_keys=True)
        parts.append(
            TextPart(
                f"[文件附件：{target}]\n读取信息：{metadata}\n{result.output}\n[附件结束]"
            )
        )
    # 【附件】【引用】2. 引用只提供定位信息，不宣称读取成功或越过模型工具权限
    for target in references:
        path = (project_root / target).resolve()
        if not path.is_file():
            raise ValueError(f"文件引用不存在或不是文件：{target}")
        parts.append(
            TextPart(
                f"[文件引用：{path}]\n尚未读取文件内容；需要内容时请通过现有文件工具读取，遵守当前权限"
            )
        )
    return AttachmentInput(
        "\n\n".join(part.text for part in parts if isinstance(part, TextPart)),
        tuple(parts),
    )
