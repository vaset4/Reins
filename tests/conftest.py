"""测试全局夹具。

作者：LKX
时间：2026-08-28 23:45:00
"""

import os

import pytest

os.environ.setdefault("REINS_TRACE_LEVEL", "debug")


@pytest.fixture(autouse=True)
def no_retry_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """全测试禁掉重试退避的真实等待。

    作者：LKX
    时间：2026-08-28 23:45:00
    传参：monkeypatch 为 pytest 提供的替换器，随用例自动还原
    返回：无

    退避在生产里必须真等：可重试的错误都是"对方现在不行、过一会儿可能就行"，不等就重发
    必然撞上同一个错误。但测试要验的是重试流程与产出，不是等待本身——一条故意触发工具
    超时的用例会让进程按 2/4/8 秒白等十几秒。

    替换的是"等待这个动作"而不是"该等多久"：退避秒数表保持真实，
    所以断言退避时长的用例（test_llm_retry）看到的仍是 1/2/4 真值。
    也不动全局 time.sleep，轮询与子进程超时那类用例需要真实时间。
    """
    # 1. 工具重试退避
    monkeypatch.setattr("tools.tool_registry._backoff_sleep", lambda _seconds: None)
    # 2. provider 重试退避：LLM 调用报可重试错误之间的等待
    monkeypatch.setattr("llm.client.sleep", lambda _seconds: None)
