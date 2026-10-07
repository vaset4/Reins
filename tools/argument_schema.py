"""工具声明与实际参数校验共用的 JSON Schema 合同。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.validators import validator_for  # type: ignore[import-untyped]
from referencing import Registry


def normalize_tool_schema(parameters: dict[str, object]) -> dict[str, object]:
    """将内置字段声明或完整对象Schema统一为独立的标准Schema。

    传参：parameters 为工具声明；返回：保留嵌套约束的对象Schema，非法声明直接报错
    """
    if parameters.get("type") == "object":
        schema = deepcopy(parameters)
    else:
        properties: dict[str, object] = {}
        required: list[str] = []
        for name, value in parameters.items():
            if not isinstance(value, dict):
                raise ValueError(
                    f"tool parameter declaration must be an object: {name}"
                )
            spec = deepcopy(value)
            # 【工具】【参数声明】字段级布尔required用于内置声明，嵌套对象的required数组原样保留
            if isinstance(spec.get("required"), bool) and spec.pop("required"):
                required.append(name)
            properties[name] = spec
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    validator = _validator_class(schema)
    validator.check_schema(schema)
    return schema


def tool_argument_error(
    schema: dict[str, object], arguments: dict[str, object]
) -> str | None:
    """按实际发给模型的Schema校验参数，不强制转换类型或补默认值。

    传参：schema 为标准声明；arguments 为模型参数；返回：带字段路径的错误，合法时为None
    """
    validator = _validator_class(schema)(schema, registry=Registry())
    error = next(validator.iter_errors(arguments), None)
    if error is None:
        return None
    path = "$"
    for part in error.absolute_path:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return f"{path}: {error.message}"


def _validator_class(schema: dict[str, object]) -> Any:
    """选择Schema声明的标准草案；传参：Schema；返回：已支持的校验器类型。"""
    validator = (
        validator_for(schema, default=None)
        if "$schema" in schema
        else Draft202012Validator
    )
    if validator is None:
        raise ValueError(f"unsupported tool schema dialect: {schema['$schema']}")
    return validator
