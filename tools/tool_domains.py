"""【工具体系】【领域声明】常用表面与中文领域材料共用注册表元数据。

作者：xxx
时间：2026-10-02 14:35:00
"""

from dataclasses import replace

from tools.tool_registry import TOOL_SOURCE_BUILTIN, ToolRegistry

CORE_TOOL_NAMES = frozenset(
    {
        "capabilities",
        "ask_user",
        "goal",
        "memory_query",
        "read_artifact",
        "list",
        "find_path",
        "grep",
        "file_read",
        "file_write",
        "file_patch",
        "terminal_tool",
    }
)

DOMAIN_TOOLS = {
    "interaction": ("capabilities", "ask_user", "goal"),
    "files": ("list", "find_path", "grep", "file_read", "file_write", "file_patch"),
    "execution": ("terminal_tool", "code_execution_tool"),
    "memory": ("memory_query", "memory_manage", "memory_note"),
    "knowledge": ("knowledge_reflect", "knowledge_read"),
    "skills": ("skill_search", "skill_read", "skill_run", "skill_manage"),
    "history": ("read_history", "read_artifact", "context_inspect", "context_release"),
    "operations": ("operation_status", "resume_operation"),
    "planning": ("todo",),
    "web": ("web_search", "web_fetch", "web_scan"),
    "browser": (
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_extract",
        "browser_screenshot",
    ),
    "collaboration": (
        "delegate",
        "agent_send",
        "agent_status",
        "agent_wait",
        "agent_cancel",
        "agent_decision",
    ),
    "schedules": ("schedule",),
    "notifications": ("notification_send", "notification_status"),
    "sensitive": (
        "clipboard_read",
        "clipboard_write",
        "secret_list_names",
        "secret_use",
        "redact",
    ),
    "internal": ("echo", "inspect", "ocr", "screenshot"),
}

DOMAIN_TERMS = {
    "interaction": ("能力", "工具", "提问", "目标"),
    "files": ("文件", "目录", "路径", "代码搜索", "资料", "读文件", "修改文件"),
    "execution": ("终端", "命令", "执行代码", "运行测试", "脚本"),
    "memory": ("记忆", "偏好", "事实", "记住", "便签", "更正", "归档", "恢复知识"),
    "knowledge": ("知识维护", "提炼", "反思", "来源核验"),
    "skills": ("技能", "方法", "经验", "可复用"),
    "history": ("历史", "上下文", "产物", "原文", "回查"),
    "operations": ("操作状态", "恢复操作", "执行进度", "运行状态"),
    "planning": ("待办", "计划", "清单", "进度"),
    "web": ("网络", "网页", "联网", "搜索资料", "上网", "网站"),
    "browser": ("浏览器", "页面", "点击", "填写", "表单"),
    "collaboration": ("协作", "助手", "分工", "代理", "子任务", "队友"),
    "schedules": ("定时", "提醒", "计划任务", "闹钟"),
    "notifications": ("通知", "发送消息", "通知状态"),
    "sensitive": ("剪贴板", "密钥", "脱敏"),
    "internal": ("内部",),
}


def configure_builtin_catalog(registry: ToolRegistry) -> None:
    """发布常用/领域元数据，不改变风险或授权；参数：当前注册表；返回：无。"""
    domains = {name: domain for domain, names in DOMAIN_TOOLS.items() for name in names}
    definitions = []
    for item in registry.list_definitions():
        if item.source != TOOL_SOURCE_BUILTIN:
            continue
        domain = domains.get(item.name, item.domain or item.toolset)
        definitions.append(
            replace(
                item,
                domain=domain,
                search_terms=DOMAIN_TERMS.get(domain, ()),
                deferred=item.name not in CORE_TOOL_NAMES,
            )
        )
    if definitions:
        registry.publish(
            definitions, replace_names=tuple(item.name for item in definitions)
        )
