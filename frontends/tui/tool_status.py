"""保持工具回执的执行与审批含义。

作者：xxx
时间：2026-09-30 00:30:00
"""

from collections.abc import Mapping


def tool_display_state(status: str | None, execution: Mapping[str, str]) -> str:
    """按真实结果与执行阶段生成标题；传参：规范结果状态及执行字段；返回：用户可读状态。"""
    state = execution.get("execution_state")
    labels = {
        "not_started": "未派发",
        "unknown": "执行结果未知，需核对副作用",
        "started": "执行中",
        "interrupted": "已中断",
        "cancelled": "已中断",
    }
    if state in labels:
        result = labels[state]
    elif execution.get("tool_error_category") == "cancelled":
        result = "已中断"
    else:
        result = {"success": "完成", "error": "失败", "partial": "部分完成"}.get(
            str(status), "状态未知"
        )
    approval = {
        "denied": "用户拒绝",
        "cancelled": "审批已中断",
        "unavailable": "审批服务故障",
    }.get(execution.get("approval_state", ""))
    return f"{result} · {approval}" if approval else result
