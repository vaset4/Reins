from __future__ import annotations

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def reasoning_options(model: str, api_mode: str) -> tuple[str, ...]:
    """列出已核验的模型强度；传参：真实模型名和协议；返回：档位，未知模型为空。"""
    name = model.lower().rsplit("/", 1)[-1]
    # 【模型配置】【思考强度】供应商可以是代理别名，能力按真实模型及协议确定
    deepseek = {
        "deepseek-flash",
        "deepseek-v4-flash",
        "deepseek-v4.1-flash",
        "deepseek-v4-pro",
        "deepseek-v4-pro-0813",
        "deepseek-v4-flash-vision-exp",
    }
    if name in deepseek and api_mode in {
        "chat_completions",
        "responses",
        "anthropic_messages",
    }:
        return ("none", "low", "high", "max")
    if api_mode not in {"chat_completions", "responses"}:
        return ()
    if api_mode == "chat_completions":
        if name == "glm-5.2":
            return ("high", "max")
        if name in {"glm-5.3", "glm-5.3-flash"}:
            return ("low", "high", "max")
    if name in {"grok-4.6", "grok-4.7"}:
        return ("low", "medium", "high", "xhigh")
    if name in {"grok-4.5", "o1", "o3", "o3-mini", "o4-mini"}:
        return ("low", "medium", "high")
    if name in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        return ("minimal", "low", "medium", "high")
    return ()


def validate_reasoning_effort(value: object, model: str, api_mode: str) -> str | None:
    """校验用户选择；传参：档位、模型及协议；返回：真实档位或默认，不支持则明确报错。"""
    if value is None or value == "default":
        return None
    if not isinstance(value, str) or value not in reasoning_options(model, api_mode):
        raise ValueError(
            f"unsupported reasoning_effort {value!r} for {model} ({api_mode})"
        )
    return value


def combine_reasoning_text(*parts: str) -> str:
    rows = [part.strip() for part in parts if part.strip()]
    return "\n\n".join(rows)


def split_think_blocks(text: str) -> tuple[str, str]:
    """Return visible content plus reasoning extracted from <think> blocks."""
    if THINK_OPEN not in text.casefold():
        return text, ""

    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    cursor = 0
    lowered = text.casefold()
    open_len = len(THINK_OPEN)
    close_len = len(THINK_CLOSE)

    while cursor < len(text):
        start = lowered.find(THINK_OPEN, cursor)
        if start < 0:
            visible_parts.append(text[cursor:])
            break
        visible_parts.append(text[cursor:start])
        body_start = start + open_len
        end = lowered.find(THINK_CLOSE, body_start)
        if end < 0:
            reasoning_parts.append(text[body_start:])
            cursor = len(text)
            break
        reasoning_parts.append(text[body_start:end])
        cursor = end + close_len

    return "".join(visible_parts), combine_reasoning_text(*reasoning_parts)


__all__ = ["combine_reasoning_text", "split_think_blocks"]
