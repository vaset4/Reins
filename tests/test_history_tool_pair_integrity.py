"""历史尾部裁剪不得切开工具调用配对，切开也不得伪造缺失的调用公告。

作者：LKX
时间：2026-08-30 16:20:00
背景：一次工具调用横跨 assistant(tool_calls) 与 ToolResultMessage 两条消息。按条切尾会
留下没有公告的孤立工具结果，而旧的补救函数会补造 name="tool"、arguments="{}" 的假公告，
让模型读到一次从未发生过的调用。这里的用例全部走真实链路造数据：
append_* -> materialize_messages -> ProductionContextBuilder -> compose_model_request。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from context.production_builder import ProductionContextBuilder
from llm.messages import (
    AgentMessage,
    AssistantMessage,
    MessageContractError,
    StopReason,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    model_visible_text,
    validate_message_sequence,
)
from llm.model_request import ModelRequestContractError, compose_model_request
from runtime.session_messages import (
    ToolExchange,
    append_assistant_message,
    append_tool_exchange,
    append_user_message,
    materialize_messages,
)
from runtime.session_message_store import SessionMessageStore
from tools.tool_registry import (
    IDEMPOTENT_YES,
    TARGET_SCOPE_PATH,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_FILE,
    ToolDefinition,
    ToolRegistry,
)

SESSION_ID = "session-pair-integrity"
SYSTEM_PROMPT = "You are a fixed test system prompt."
TASK_TEXT = "read the config file and report what it says"
CONTEXT_WINDOW = 30_000
# 真实调用的工具名与参数：断言产出里出现的是它们，而不是补造出来的 name="tool"
REAL_TOOL_NAME = "file_read"
REAL_CALL_ID = "call-secrets-1"
REAL_TOOL_TEXT = "permission denied on secrets.yaml"


def _builder(data_root: Path) -> ProductionContextBuilder:
    """创建从真实会话读取历史的生产Builder。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：data_root 为本用例独占数据根
    返回：可直接调用 read_conversation_history 的 ProductionContextBuilder
    """
    return ProductionContextBuilder(
        data_root,
        system_prompt_provider=lambda: SYSTEM_PROMPT,
    )


def _seed_one_tool_pair(data_root: Path) -> None:
    """造"用户提问 + 一次工具调用配对"的真实历史，尾部正好是工具结果。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：data_root 为本用例独占数据根
    返回：无；落盘 3 条消息（user / assistant(tool_calls) / tool_result）
    """
    append_user_message(data_root, SESSION_ID, "now read secrets.yaml")
    append_tool_exchange(
        data_root,
        SESSION_ID,
        ToolExchange(
            call_id=REAL_CALL_ID,
            tool_name=REAL_TOOL_NAME,
            args={"path": "secrets.yaml"},
            rendered=REAL_TOOL_TEXT,
            status="error",
            error="permission denied",
        ),
    )


def _multi_call_messages() -> tuple[AgentMessage, ...]:
    """造"一条 assistant 带两个 tool_calls + 两条结果"的多结果组消息序列。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：无
    返回：通过 validate_message_sequence 的 canonical 消息序列（4 条）

    这个形状是合法契约（见用例内的 validate_message_sequence 断言），但现役 store 只能
    逐条 append，落第一条结果时第二个调用还 pending 且末条不是 assistant，会被
    _validate_materialized_messages 拒绝，所以这里直接给消息序列。
    """
    return (
        UserMessage("msg-multi-user", (TextPart("read both config files"),)),
        AssistantMessage(
            "msg-multi-call",
            (
                ToolCallPart("call-multi-a", REAL_TOOL_NAME, {"path": "a.yaml"}),
                ToolCallPart("call-multi-b", REAL_TOOL_NAME, {"path": "b.yaml"}),
            ),
            stop_reason=StopReason.TOOL_CALL,
        ),
        ToolResultMessage(
            "msg-multi-result-a",
            "call-multi-a",
            REAL_TOOL_NAME,
            (TextPart("contents of a.yaml"),),
            "success",
        ),
        ToolResultMessage(
            "msg-multi-result-b",
            "call-multi-b",
            REAL_TOOL_NAME,
            (TextPart("contents of b.yaml"),),
            "success",
        ),
    )


def _tool_results(
    messages: tuple[AgentMessage, ...],
) -> list[ToolResultMessage]:
    """筛出工具结果消息。"""
    return [item for item in messages if isinstance(item, ToolResultMessage)]


def _announced_ids(messages: tuple[AgentMessage, ...]) -> set[str]:
    """收集序列里 assistant 公告过的全部 tool call id。"""
    announced: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantMessage):
            announced.update(
                part.call_id
                for part in message.content
                if isinstance(part, ToolCallPart)
            )
    return announced


def _assert_no_orphan_tool_results(messages: tuple[AgentMessage, ...]) -> None:
    """断言选中的历史里每条工具结果都有公告它的 assistant 消息。"""
    announced = _announced_ids(messages)
    for result in _tool_results(messages):
        assert result.call_id in announced, (
            f"工具结果 {result.call_id} 的 assistant 公告被切掉了：{messages}"
        )


def _compose(history: tuple[AgentMessage, ...]) -> tuple[AgentMessage, ...]:
    """把选中的历史走真实 compose_model_request 组装成模型可见消息。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：history 为 read_conversation_history 产出的消息序列
    返回：本次请求发给模型的消息序列
    """
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name=REAL_TOOL_NAME,
            description="Read a file.",
            parameters={
                "path": {"type": "string", "description": "Path", "required": True}
            },
            toolset=TOOLSET_FILE,
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_PATH,
            source=TOOL_SOURCE_BUILTIN,
            model_visible=True,
            idempotent=IDEMPOTENT_YES,
            executor=lambda _args: {"content": "ok"},
        )
    )
    bundle = compose_model_request(
        task=TASK_TEXT,
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={
            "conversation_history": history,
            "system_prompt": SYSTEM_PROMPT,
        },
        registry=registry,
        context_window=CONTEXT_WINDOW,
    )
    return bundle.messages


def _fabricated_calls(
    messages: tuple[AgentMessage, ...],
) -> list[ToolCallPart]:
    """挑出被补造出来的假公告：工具名是字面量 tool、或参数是空对象。"""
    fabricated: list[ToolCallPart] = []
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for part in message.content:
            if not isinstance(part, ToolCallPart):
                continue
            if part.tool_name == "tool" or not part.arguments:
                fabricated.append(part)
    return fabricated


def test_limit_cutting_into_pair_drops_whole_group(tmp_path: Path) -> None:
    """条数上限只够装最后那条工具结果时，整组丢弃且不补造假公告。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：tmp_path 为 pytest 提供的独占数据根
    返回：无
    """
    _seed_one_tool_pair(tmp_path)
    stored = materialize_messages(tmp_path, SESSION_ID)
    assert len(_tool_results(stored)) == 1, "前置：存储里应有一条完整工具配对"

    # 1. limit=1 只够装尾部那条工具结果——旧实现正是在这里切出孤立工具结果
    selection = _builder(tmp_path).read_conversation_history(SESSION_ID, limit=1)
    history = selection.messages

    # 2. 孤立工具结果连同它的 assistant 公告一起被丢掉，模型看不到这次调用的任何痕迹
    assert _tool_results(history) == [], f"孤立工具结果被保留了：{history}"
    assert REAL_TOOL_TEXT not in str(history), f"工具结果内容泄漏进历史：{history}"
    _assert_no_orphan_tool_results(history)

    # 3. 全链组装后不得出现任何补造的假调用公告
    messages = _compose(history)
    assert _fabricated_calls(messages) == [], f"仍在补造假公告：{messages}"
    assert REAL_CALL_ID not in str(messages), (
        f"调用 id 以假公告形式回到模型：{messages}"
    )


def test_limit_fitting_whole_group_keeps_pair_linked(tmp_path: Path) -> None:
    """条数上限装得下整组时整组保留，call_id 与 assistant 公告对得上。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：tmp_path 为 pytest 提供的独占数据根
    返回：无
    """
    _seed_one_tool_pair(tmp_path)

    # 1. limit=2 恰好装下 assistant(tool_calls) + tool_result 这一整组
    history = _builder(tmp_path).read_conversation_history(SESSION_ID, limit=2).messages

    # 2. 整组都在，且工具结果回指的是真实调用 id
    results = _tool_results(history)
    assert len(results) == 1, f"整组未被完整保留：{history}"
    assert results[0].call_id == REAL_CALL_ID
    assert REAL_CALL_ID in _announced_ids(history), f"assistant 公告缺失：{history}"
    _assert_no_orphan_tool_results(history)

    # 3. 组装后模型看到的是真实工具名，而不是补造的字面量 tool
    messages = _compose(history)
    assert _fabricated_calls(messages) == [], f"出现补造公告：{messages}"
    assert REAL_TOOL_NAME in str(messages), f"真实工具名丢失：{messages}"


def test_token_budget_path_never_splits_group(tmp_path: Path) -> None:
    """走 token 预算路径（不传 limit）压低预算，同样不切开工具调用组。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：tmp_path 为 pytest 提供的独占数据根
    返回：无
    """
    # 1. 先垫一条超长 assistant 文本，再放工具配对，让紧预算只能保住尾部
    append_assistant_message(tmp_path, SESSION_ID, "word " * 2_000)
    _seed_one_tool_pair(tmp_path)

    # 2. 预算极小：旧实现会在 assistant 公告与工具结果之间切断
    history = _builder(tmp_path).read_conversation_history(SESSION_ID).messages

    _assert_no_orphan_tool_results(history)
    # 3. 预算再紧也至少留一个完整组，历史不会塌成空
    assert history, "历史被裁成空"
    messages = _compose(history)
    assert _fabricated_calls(messages) == [], f"预算路径仍在补造公告：{messages}"


def test_multi_result_group_is_never_partially_kept(tmp_path: Path) -> None:
    """一条 assistant 带两个 tool_calls 的多结果组，不允许只保留一部分。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：tmp_path 为 pytest 提供的独占数据根
    返回：无
    """
    stored = _multi_call_messages()
    # 1. 先证明这个多结果形状本身是合法契约，再让它走真实的裁剪
    validate_message_sequence(stored)
    owner = SessionMessageStore(tmp_path)
    for message in stored:
        owner.append_message(SESSION_ID, message)
    assert len(_tool_results(stored)) == 2, "前置：应有两条工具结果"

    # 2. 逐个上限扫过整组：任何一刀都不能切出"公告在、结果只剩一条"或孤立结果
    for limit in range(1, len(stored) + 1):
        history = (
            _builder(tmp_path)
            .read_conversation_history(SESSION_ID, limit=limit)
            .messages
        )
        _assert_no_orphan_tool_results(history)
        kept = len(_tool_results(history))
        assert kept in (0, 2), f"limit={limit} 把多结果组切成 {kept} 条：{history}"
        assert _fabricated_calls(_compose(history)) == [], f"limit={limit} 补造了公告"

    # 3. 上限装得下整组时两条结果都在
    full = (
        _builder(tmp_path)
        .read_conversation_history(SESSION_ID, limit=len(stored))
        .messages
    )
    assert len(_tool_results(full)) == 2, f"整组未被完整保留：{full}"


def test_orphan_tool_message_raises_contract_error() -> None:
    """直接喂一条孤立工具结果：必须显式报错，不得静默补造。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：无
    返回：无
    """
    messages: tuple[AgentMessage, ...] = (
        ToolResultMessage(
            "msg-orphan",
            "orphan-1",
            REAL_TOOL_NAME,
            (TextPart("r"),),
            "success",
        ),
    )

    with pytest.raises(MessageContractError) as caught:
        validate_message_sequence(messages)

    assert caught.value.code == "orphan_tool_result"
    assert "orphan-1" in str(caught.value)
    # 1. 校验是只读的：入参消息序列不被就地改写，更不会被塞进假公告
    assert [item.kind for item in messages] == ["tool_result"]


def test_composing_orphan_history_rejects_instead_of_fabricating() -> None:
    """组装拿到孤立工具结果时抛错，而不是补一条假公告继续发请求。

    作者：LKX
    时间：2026-08-30 16:20:00
    传参：无
    返回：无
    """
    orphan: tuple[AgentMessage, ...] = (
        ToolResultMessage(
            "msg-orphan",
            REAL_CALL_ID,
            REAL_TOOL_NAME,
            (TextPart(REAL_TOOL_TEXT),),
            "error",
            error="permission denied",
        ),
    )

    # 构造 ModelRequest 时把消息契约错误重包成请求契约错误，错误码原样保留
    with pytest.raises(ModelRequestContractError) as caught:
        _compose(orphan)

    assert caught.value.code == "orphan_tool_result"
    assert REAL_CALL_ID in str(caught.value)
    # 报错前后都没有产生过假公告——错误指向坏数据本身
    assert model_visible_text(orphan[0]) == REAL_TOOL_TEXT
