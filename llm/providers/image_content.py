"""供应商图片传输共用校验，不允许把未读取引用冒充图片。

作者：xxx
时间：2026-09-30 11:00:00
"""

import json
from collections.abc import Mapping

from llm.image_input import image_base64
from llm.messages import ImagePart
from llm.provider_adapter import ProviderAdapterError

# OpenAI官方vision指南的请求边界；伙伴端点的更小限制由其实际错误报告
OPENAI_IMAGE_REQUEST_BYTES = 512 * 1024 * 1024
OPENAI_IMAGE_REQUEST_COUNT = 1500


def validated_image_data(part: ImagePart, *, max_dimension: int | None = None) -> str:
    """把图片合同失败转成供应商边界错误；参数：冻结图片；返回：验证过的base64正文。"""
    try:
        return image_base64(part, max_dimension=max_dimension)
    except ValueError as exc:
        raise ProviderAdapterError(f"invalid_image_input:{exc}") from exc


def validate_image_request_size(body: Mapping[str, object], *, max_bytes: int) -> None:
    """校验含图片的实际JSON传输体积；参数：最终请求体和协议上限；返回：无，超限明确失败。"""
    byte_count = len(
        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if byte_count > max_bytes:
        raise ProviderAdapterError(f"image_request_too_large:{byte_count}>{max_bytes}")
