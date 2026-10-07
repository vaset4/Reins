"""图片输入的冻结字节与格式校验；不读取路径或远程地址。

作者：xxx
时间：2026-09-30 11:00:00
"""

from __future__ import annotations

import base64
import binascii
from io import BytesIO

from llm.messages import ImagePart

IMAGE_MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


def image_from_bytes(raw: bytes) -> ImagePart:
    """验证真实图片格式并冻结原始字节；参数：已授权读取的字节；返回：可持久化图片块。"""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(raw)) as picture:
            mime = IMAGE_MIME_TYPES.get(picture.format or "")
            if mime is None:
                raise ValueError("图片格式不支持，请使用 PNG、JPEG、GIF 或 WebP")
            if getattr(picture, "n_frames", 1) != 1:
                raise ValueError("当前图片输入不支持动画，请选择静态图片")
            width, height = picture.size
            picture.verify()
        # 【图片输入】【格式校验】1. 解码像素以发现截断数据，不把仅有合法文件头当成图片
        with Image.open(BytesIO(raw)) as picture:
            picture.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("图片损坏或无法解码") from exc
    encoded = base64.b64encode(raw).decode("ascii")
    return ImagePart(f"data:{mime};base64,{encoded}", mime, width=width, height=height)


def image_base64(part: ImagePart, *, max_dimension: int | None = None) -> str:
    """验证冻结图片再交给协议适配器；参数：图片块；返回：base64正文，拒绝可变路径或URL。"""
    prefix = f"data:{part.mime_type};base64,"
    if (
        part.mime_type not in IMAGE_MIME_TYPES.values()
        or not part.source_ref.startswith(prefix)
    ):
        raise ValueError(
            "image input requires a frozen data URI with a supported MIME type"
        )
    encoded = part.source_ref[len(prefix) :]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image input contains invalid base64") from exc
    verified = image_from_bytes(raw)
    if verified.mime_type != part.mime_type:
        raise ValueError("image MIME type does not match its content")
    if (part.width is not None and verified.width != part.width) or (
        part.height is not None and verified.height != part.height
    ):
        raise ValueError("image dimensions do not match its content")
    if max_dimension is not None:
        assert verified.width is not None and verified.height is not None
        if max(verified.width, verified.height) > max_dimension:
            raise ValueError(f"image_dimensions_exceeded:{max_dimension}")
    return encoded
