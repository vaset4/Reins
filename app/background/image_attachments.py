"""图片附件读取沿用本地文件权限，原文件变化不影响已接纳图片。

作者：xxx
时间：2026-09-30 11:00:00
"""

from pathlib import Path

import path_security
from llm.image_input import image_from_bytes
from llm.messages import ImagePart
from runtime.lease import Lease
from tools.file_persistence import file_edit_lock

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
IMAGE_HEADER_BYTES = 16


def read_image_attachment(
    target: str, *, project_root: Path, lease: Lease
) -> ImagePart | None:
    """在真实路径权限内识别图片；参数：用户路径、工作区和租约；返回：图片，非图片交原文本读取器。"""
    path = project_root / target
    with file_edit_lock(path):
        resolved = path.resolve()
        if not resolved.is_relative_to(project_root.resolve()):
            raise ValueError(f"REJECTED_PATH: {target}")
        decision = path_security.check_read(path, lease, filtered=True)
        if decision is not path_security.Decision.ALLOWED:
            raise ValueError(f"PERMISSION_DENIED: {decision.value}")
        # 【图片附件】【秘密保护】1. 敏感路径必须留在原脱敏通道，不能把其字节转成图片绕过过滤
        if path_security.uses_redacted_files(path, lease):
            return None
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            header = stream.read(IMAGE_HEADER_BYTES)
            is_image = header.startswith(
                (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")
            ) or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
            if not is_image and path.suffix.lower() not in IMAGE_SUFFIXES:
                return None
            raw = header + stream.read()
        return image_from_bytes(raw)
