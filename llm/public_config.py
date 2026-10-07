"""向后台和定时意图传递公开模型选择，凭据始终单独处理。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

from collections.abc import Mapping

from llm.base import LLMClient

PUBLIC_MODEL_FIELDS = frozenset(
    {
        "provider",
        "base_url",
        "model",
        "api_mode",
        "timeout_seconds",
        "context_window",
        "max_output_tokens",
        "profile_name",
        "reasoning_effort",
    }
)


def restore_model_config(config: Mapping[str, object]) -> dict[str, object]:
    """恢复已接纳的公开快照；传参：持久模型配置；返回：独立配置，旧记录未设强度保持历史默认。"""
    # 【模型配置】【运行恢复】旧快照无强度字段表示当时未发送参数，不能继承后来修改的profile
    return {"reasoning_effort": "default", **config}


def public_model_config(client: LLMClient) -> dict[str, object]:
    """取得当前实际模型的公开快照；传参：模型客户端；返回：不含凭据的配置。"""
    target = getattr(client, "resolved_target", None)
    if target is None:
        return {}
    result = {
        name: value
        for name in PUBLIC_MODEL_FIELDS
        if (value := getattr(target, name, None)) is not None and value != ""
    }
    # 默认也必须冻结，后台不能从后来修改的profile重新继承强度
    result["reasoning_effort"] = getattr(target, "reasoning_effort", None) or "default"
    return result


def ephemeral_api_key(client: LLMClient) -> str | None:
    """只提取命令行临时凭据供进程内交接；传参：客户端；返回：临时密钥或无。"""
    target = getattr(client, "resolved_target", None)
    if target is not None and target.credential_source == "cli":
        return str(target.api_key) if target.api_key else None
    return None
