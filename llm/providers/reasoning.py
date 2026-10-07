"""将显式思考选择转换为供应商协议参数。

作者：xxx
时间：2026-09-30 00:00:00
"""

from llm.model_request import ModelRequest, PreferenceKind
from llm.reasoning import validate_reasoning_effort


def reasoning_parameters(
    request: ModelRequest, model: str, api_mode: str
) -> dict[str, object]:
    """翻译思考档位；传参：规范请求、模型及协议；返回：发送字段，不支持时拒绝。"""
    value = next(
        (
            item.value
            for item in request.optional_preferences
            if item.kind == PreferenceKind.REASONING_LEVEL
        ),
        None,
    )
    effort = validate_reasoning_effort(value, model, api_mode)
    if effort is None:
        return {}
    if api_mode == "responses":
        return {"reasoning": {"effort": effort}}
    if api_mode == "anthropic_messages":
        if effort == "none":
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "enabled"}, "output_config": {"effort": effort}}
    if model.lower().rsplit("/", 1)[-1].startswith("deepseek-"):
        # 【模型调用】【思考强度】扩展字段通过SDK extra_body进入HTTP正文，不能作为未知关键字传入
        if effort == "none":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {
            "reasoning_effort": effort,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    return {"reasoning_effort": effort}
