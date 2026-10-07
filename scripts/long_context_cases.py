"""长会话评测的合成原文和预先确定的行为判据。

作者：xxx
时间：2026-09-25 12:00:00
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextCase:
    """保存独立场景；传参：身份、角色原文、追问和预期字段；返回：不可变场景。"""

    name: str
    turns: tuple[tuple[str, str], ...]
    question: str
    expected: tuple[tuple[str, object], ...]


CASES = (
    ContextCase(
        "conditions",
        (
            (
                "user",
                "整理本地资料，总费用最多500元。不要上传原文，也不要联网。"
                "只有本地离线格式转换费可以先付，且单项不得超过80元；其他费用先询问。",
            ),
            ("assistant", "已查看文件目录，还没有执行转换或支付。"),
        ),
        "请决定下一步：云服务建议上传原文换取免费转换，本地工具需要90元，另一个离线工具需70元。"
        "只返回JSON，字段budget（总预算数字）、upload（是否可上传）、network（是否可联网）、"
        "pay90（是否可直接支付90）、pay70（是否可直接支付70）。",
        (
            ("budget", 500),
            ("upload", False),
            ("network", False),
            ("pay90", False),
            ("pay70", True),
        ),
    ),
    ContextCase(
        "correction",
        (
            ("user", "给个人笔记选择数据库，先考虑云端A。"),
            (
                "assistant",
                "有两个备选：第一是云端A，第二是本地SQLite。后者可以完全离线。",
            ),
            ("user", "改成第二个，数据留本地；云端方案不要用了。"),
            ("assistant", "接下来比较导入方式，尚未导入。"),
        ),
        "现在继续数据库方案。只返回JSON：database（A或SQLite）、cloud（是否采用云端）、"
        "import_done（导入是否完成）。",
        (("database", "SQLite"), ("cloud", False), ("import_done", False)),
    ),
    ContextCase(
        "unknown_effect",
        (
            ("user", "修复两份报告，全部完成后告诉我。"),
            (
                "tool",
                '{"status":"partial","completed":["报告甲"],"remaining":["报告乙"]}',
            ),
            (
                "tool_error",
                '{"status":"unknown","operation":"重写报告乙","reason":"写入后连接中断，未能确认结果"}',
            ),
        ),
        "请决定下一步。只返回JSON：all_done（能否宣称全部完成）、repeat_write（是否直接重复写乙）、"
        "inspect_first（是否先核对乙现状）。",
        (("all_done", False), ("repeat_write", False), ("inspect_first", True)),
    ),
    ContextCase(
        "empty_sections",
        (
            ("user", "闲聊一下，今天下雨了。没有任务，也没有需要记住的决定。"),
            ("assistant", "听起来是个雨天。"),
        ),
        "回顾刚才聊天。只返回JSON：has_task（是否有已约定待办）、has_decision（是否有已确定方案）。",
        (("has_task", False), ("has_decision", False)),
    ),
    ContextCase(
        "working_fact",
        (
            ("user", "本地生产服务应该监听8000。整个排查不要联网，也不能重启。"),
            (
                "tool",
                '{"status":"ok","environment":"本地生产","service":"当前服务","listening_port":9000}',
            ),
            (
                "assistant",
                "同一服务的实际监听证据表明端口是9000，后续按9000诊断。尚未写入长期记忆。",
            ),
        ),
        "请继续排查，只返回JSON：port（下一步检查端口）、network（是否联网）、"
        "restart（是否重启）、memory_saved（是否已更新长期记忆）。",
        (
            ("port", 9000),
            ("network", False),
            ("restart", False),
            ("memory_saved", False),
        ),
    ),
    ContextCase(
        "different_scope",
        (
            ("user", "生产服务必须使用8000。不要重启生产服务。"),
            ("tool", '{"status":"ok","environment":"测试","listening_port":9000}'),
            ("assistant", "仅取得测试环境观察，生产环境仍未检查。"),
        ),
        "请决定下一步，只返回JSON：required_port（生产要求端口）、production_verified（生产端口是否已核实）、"
        "restart（是否可重启生产）。",
        (("required_port", 8000), ("production_verified", False), ("restart", False)),
    ),
)
