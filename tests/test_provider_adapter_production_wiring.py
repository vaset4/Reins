"""Provider Adapter 生产接线的契约测试。

作者：xxx
时间：2026-08-28 00:00:00

本文件是 08-28-provider-adapter-production-wiring 的 Phase A 失败契约测试。
每条断言都指向"typed 路径未接线"或"Wire 路径未退休"这一个事实，
不允许通过改断言或加兜底让它变绿——只能通过真正接线和退休来变绿。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from llm.config import LLMProviderConfig
from llm.messages import AgentMessage, TextPart
from llm.model_request import ModelRequest, ModelToolDefinition
from tools.tool_registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]

# 组装一次 compose 所需的最小上下文，避免各条测试重复造相同输入
COMPOSE_TASK = "read the config file and report what it says"
COMPOSE_STAGE = "plan"
COMPOSE_CONTEXT_WINDOW = 30000


def _compose(protocol_mode: str = "native_tool_calls") -> object:
    """按生产签名跑一次 prompt 组装。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：protocol_mode 为协议模式，默认走生产默认的原生工具调用
    返回：compose_model_request 的返回值，类型由实现阶段决定
    """
    from llm.model_request import compose_model_request

    return compose_model_request(
        task=COMPOSE_TASK,
        stage=COMPOSE_STAGE,
        protocol_mode=protocol_mode,
        model_context={"system_prompt": "You are a fixed baseline system prompt."},
        registry=ToolRegistry(),
        context_window=COMPOSE_CONTEXT_WINDOW,
    )


def _rg(pattern: str, *paths: str, extra: tuple[str, ...] = ()) -> str:
    """在仓库内跑一次 ripgrep 并返回命中文本。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：pattern 为正则；paths 为搜索根；extra 为完整的附加参数（含 glob）
    返回：命中行文本；无命中返回空串
    """
    command = ["rg", "-n", pattern, *paths, *extra]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode not in (0, 1):
        pytest.fail(f"ripgrep failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


# --- 1. prompt 组装必须交出 typed ModelRequest ---


def test_compose_yields_model_request_with_agent_messages() -> None:
    """组装结果必须携带 ModelRequest，且历史全为 AgentMessage。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    composed = _compose()
    request = getattr(composed, "request", None)
    assert isinstance(request, ModelRequest), (
        "prompt 组装仍未交出 ModelRequest；今天返回的是携带 dict 元组的 ModelRequestBundle"
    )
    assert request.messages, "ModelRequest.messages 不应为空"
    for message in request.messages:
        assert isinstance(message, AgentMessage), f"消息仍是 Wire 形状：{message!r}"


def test_compose_carries_system_prompt_as_instructions() -> None:
    """system prompt 必须走 ModelRequest.instructions，不再伪造 system 消息。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    composed = _compose()
    request = getattr(composed, "request", None)
    assert isinstance(request, ModelRequest), "组装未交出 ModelRequest"
    assert request.instructions, "instructions 为空说明 system prompt 仍挂在消息序列里"
    assert all(isinstance(part, TextPart) for part in request.instructions)
    roles = [message.kind for message in request.messages]
    assert "system" not in roles, "消息序列里仍有 system 条目，说明 Wire 形状未退休"


def test_compose_tools_are_typed_definitions() -> None:
    """工具必须以 ModelToolDefinition 交付，而不是 openai 函数 dict。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    composed = _compose()
    tools = getattr(getattr(composed, "request", None), "tools", None)
    assert tools is not None, "组装未交出 ModelRequest.tools"
    for tool in tools:
        assert isinstance(tool, ModelToolDefinition), f"工具仍是 Wire dict：{tool!r}"


# --- 2. 生产调用必须经过 ModelSelector 与 Adapter ---


def test_client_exposes_adapter_seam_not_transport_seam() -> None:
    """RealLLMClient 必须以 Adapter/模型注册表为注入面，transport 注入面退休。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    from inspect import signature

    from llm.client import RealLLMClient

    parameters = signature(RealLLMClient.__init__).parameters
    assert "adapter_registry" in parameters, (
        "RealLLMClient 尚无 adapter_registry 注入面，生产仍走 openai_compatible_transport"
    )
    assert "model_registry" in parameters, "RealLLMClient 尚无 model_registry 注入面"
    assert "transport" not in parameters, "transport 注入面仍在，旧 Wire 路径未退休"


def test_production_call_path_has_no_transport_import() -> None:
    """llm/client.py 不得再引用旧 transport 与 dict 中转函数。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    source = (REPO_ROOT / "llm" / "client.py").read_text(encoding="utf-8")
    assert "openai_compatible_transport" not in source, "client 仍引用旧 transport"
    assert "request_llm_completion" not in source, "client 仍走 dict 请求函数"


# --- 3. 能力不满足必须抛 ProviderAdapterError 且不被吞成 unknown ---


def test_unsatisfied_capability_error_reaches_caller() -> None:
    """能力不满足要抛 ProviderAdapterError，不能被重试循环归类成 unknown。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    from llm.client import RealLLMClient
    from llm.model_registry import ModelRegistry
    from llm.provider_adapter import AdapterRegistry, ProviderAdapterError
    from llm.providers.openai_chat import OpenAIChatAdapter

    from tests.provider_test_support import tool_roundtrip_request

    # 1. Adapter 的能力校验在生成器体内，裸调用不执行；必须迭代才会触发
    adapter = OpenAIChatAdapter()
    with pytest.raises(ProviderAdapterError, match="unsupported_capability"):
        list(
            adapter.stream(
                tool_roundtrip_request(),
                model=_descriptor_without_native_tools(),
                connection=_local_connection(),
            )
        )

    # 2. 真正要守的是 client 不吞：注入一个不声明原生工具能力的注册表，
    #    本轮请求带工具定义，选型必然不满足，错误必须以硬阻断分类回到调用方
    client = RealLLMClient(
        _wiring_test_config(),
        adapter_registry=AdapterRegistry([OpenAIChatAdapter()]),
        model_registry=ModelRegistry([_descriptor_without_native_tools()]),
        connection=_local_connection(),
    )
    plan = client.plan(COMPOSE_TASK)

    assert plan.model_error is not None, "能力不满足却没有回喂 model_error"
    assert plan.model_error.category != "unknown", (
        f"能力失败被吞成 unknown：{plan.model_error.summary}"
    )
    assert plan.model_error.category == "missing_config", (
        f"能力失败应按硬阻断分类回喂，实际 {plan.model_error.category}"
    )
    assert not plan.model_error.retryable, "能力不满足重试无用，不得标成可重试"
    assert plan.observation is not None and plan.observation.attempt_count == 0, (
        "派发前发现能力不满足，没有真实网络尝试"
    )


def _wiring_test_config() -> LLMProviderConfig:
    """构造一条不会真正发出请求的 provider 配置。

    作者：LKX
    时间：2026-08-30 23:20:00
    传参：无
    返回：LLMProviderConfig
    """
    return LLMProviderConfig(
        base_url="http://127.0.0.1:1/v1",
        model="text-only-1",
        api_key="unused",
        timeout_seconds=0.01,
    )


def _descriptor_without_native_tools() -> object:
    """构造一个不声明原生工具能力的模型事实。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：ModelDescriptor
    """
    from llm.model_registry import Capability, ModelDescriptor

    return ModelDescriptor(
        model_key="production",
        provider="local",
        model_id="text-only-1",
        api_family="openai_chat",
        capabilities={Capability.STREAMING: True},
        preference_values={},
        connection_profile_key="local",
    )


def _local_connection() -> object:
    """构造一个不会真正发出请求的连接对象。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：ResolvedConnection
    """
    from llm.provider_connection import ResolvedConnection

    return ResolvedConnection(
        base_url="http://127.0.0.1:1",
        timeout_seconds=0.01,
        credential="unused",
        headers={},
    )


# --- 4. 响应必须经 StreamAssembler，usage 缺失为 None ---


def test_missing_usage_stays_none_through_production_path() -> None:
    """usage 缺失必须保持 None，不得在生产路径上被填成 0。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    source = (REPO_ROOT / "llm" / "client.py").read_text(encoding="utf-8")
    assert "StreamAssembler" in source, (
        "client 未使用 StreamAssembler，响应仍由 ProviderResponse dict 承载"
    )


def test_runtime_never_imports_provider_wire_types() -> None:
    """runtime/ 不得 import llm.provider_stream，Wire 类型必须止步于 llm/ 边界。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：无
    返回：无

    流式增量打通后 client 只向上交 provider 中立的 ModelOutputDelta。一旦 runtime
    直接读 ModelStreamEvent 的 api_family / block_id / sequence，换一家 provider
    就得改 runtime，分层就白划了。
    """
    forms = ("from llm.provider_stream", "import llm.provider_stream")
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "runtime").rglob("*.py"))
        if any(form in path.read_text(encoding="utf-8") for form in forms)
    ]
    assert not offenders, f"runtime 侧出现 Provider Wire 类型 import：{offenders}"


# --- 5. 孤立 tool 消息必须在源头被拒绝 ---


def test_truncated_tool_pair_is_rejected_not_fabricated() -> None:
    """历史截断切开 tool 对时必须报错，不能补造假的 tool_calls 条目。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    from llm.prompt_composer import build_request_messages

    assert not hasattr(build_request_messages, "__wrapped__")
    composer = (REPO_ROOT / "llm" / "prompt_composer.py").read_text(encoding="utf-8")
    assert "_repair_orphan_tool_messages" not in composer, (
        "补造函数仍在；S0 已坐实它会把真实调用 file_read({'path':'secrets.yaml'}) "
        "伪造成 name='tool'、arguments='{}'"
    )
    assert "_history_message" not in composer, "dict 历史组装函数仍在"


def test_orphan_tool_result_raises_from_validate_sequence() -> None:
    """typed 校验必须对孤立 tool 结果给出 orphan_tool_result。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    from llm.messages import (
        MessageContractError,
        ToolResultMessage,
        validate_message_sequence,
    )

    orphan = ToolResultMessage(
        "m1",
        "call-baseline-2",
        "file_read",
        (TextPart("permission denied"),),
        "error",
        error="permission denied",
    )
    with pytest.raises(MessageContractError) as caught:
        validate_message_sequence((orphan,))
    assert caught.value.code == "orphan_tool_result"


# --- 7. 开关全仓退休 ---


def test_native_tool_history_flag_fully_retired() -> None:
    """REINS_NATIVE_TOOL_HISTORY 必须全仓零命中，含文档。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    # 排除本测试文件自身：它的 docstring 和 pattern 字面量声明了要扫的字符串，
    # 会被自己扫到。同目录其他测试文件仍在扫描范围内，真实引用照旧会被打中
    hits = _rg(
        "REINS_NATIVE_TOOL_HISTORY",
        ".",
        extra=(
            "-g",
            "!build/**",
            "-g",
            "!.git/**",
            "-g",
            "!.trellis/**",
            "-g",
            "!**/test_provider_adapter_production_wiring.py",
        ),
    )
    assert hits == "", f"开关仍有命中：\n{hits}"


def test_legacy_symbols_fully_retired() -> None:
    """§7 第一条扫描的旧符号必须零命中。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    pattern = (
        "openai_compatible_transport|_native_tool_history_enabled|"
        "_NATIVE_TOOL_HISTORY_ENV|_native_history_row|"
        "_repair_orphan_tool_messages|_history_message\\(|"
        "ModelRequestBundle|bundle_with_messages"
    )
    hits = _rg(
        pattern,
        "app",
        "context",
        "frontends",
        "runtime",
        "tasks",
        "triggers",
        "llm",
        "tools",
        extra=("-g", "*.py"),
    )
    assert hits == "", f"旧符号仍有命中：\n{hits}"


def test_renamed_types_absent_from_docs() -> None:
    """被改名的类型名不得残留在文档里。

    作者：LKX
    时间：2026-09-01 16:20:00
    传参：无
    返回：无

    上一条扫描带 `-g "*.py"` 且只覆盖八个代码目录，文档按构造必然通过，`ModelRequestBundle`
    因此在 .py 全部退休后仍活在 Architecture.md 与 docs/learn/ 里。改名类型的扫描必须覆盖
    文档，否则"退休扫描全绿"这条证据对文档漂移永远失效。

    只扫改名（旧名已不存在于代码，出现即是漂移），不扫仍在用的符号——那样会打中正常引用。

    `.trellis/spec/` 必须一起扫：它是会被注入进后续会话的活指南，一段把已退休类型当现役
    API 写的 spec 会主动误导执行者。但要用 `--hidden`——rg 默认跳过点开头的目录，`.trellis`
    因此从来没被扫到，只加 glob 排除是空操作。

    放过 `.trellis/tasks/` 与 `.trellis/workspace/`：任务卡与开发日志是历史记录，本就该保留
    当时写下的旧名字，改它等于篡改台账。

    只认声明与使用形态（`class X` / `-> X` / `X.` / `f(`），不认裸提及——spec 记录"某类型已
    改名"时必须能写出旧名字，那是正当文档而非漂移。这与 §8 的 `tool_calls` 扫描同一思路：
    靠形态区分真契约引用与正常叙述。
    """
    hits = _rg(
        r"class ModelRequestBundle\b|-> ModelRequestBundle\b|\bModelRequestBundle\."
        r"|\bbundle_with_messages\(|\bbundle_to_evidence\(",
        ".",
        extra=(
            "--hidden",
            "-g",
            "*.md",
            "-g",
            "!build/**",
            "-g",
            "!.trellis/tasks/**",
            "-g",
            "!.trellis/workspace/**",
        ),
    )
    assert hits == "", f"文档把已改名的类型当现役 API 写：\n{hits}"


# --- 8. Wire 形状在 allowlist 外零命中 ---


def test_wire_shapes_absent_outside_allowlist() -> None:
    """Wire 字段名在 context/ 与 llm/（除 Adapter 与 Run Evidence）零命中。

    作者：xxx
    时间：2026-08-28 00:00:00
    传参：无
    返回：无
    """
    # 只把三种形态当作 OpenAI 协议字段名：字典键（前面紧贴引号）、typed 对象取属性
    # （前面紧贴点号）、按关键字传参（后面紧跟等号）。下划线本身是单词字符、不构成 \b
    # 边界，所以 native_tool_calls（protocol_mode 的取值，parser 靠它分派协议）和
    # pending_tool_calls（prompt_context 的内部键名）这类自有标识符不会被当成协议泄漏。
    # 作者：LKX
    # 时间：2026-08-31 10:20:00
    pattern = r"""["'.]tool_calls\b|\btool_calls\s*=|\btool_call_id\b"""
    # llm/response_evidence.py 例外：Run Evidence 的响应字段名按 design §8.1 保留 Wire
    # 词汇，frontends/tui/data/run_evidence_detail.py:131 等消费方直接对
    # response["tool_calls"] 取长度，改名会断掉这份已对外承诺的证据契约。它是有意保留，
    # 不是这道闸的漏网之鱼。
    hits = _rg(
        pattern,
        "context",
        "llm",
        extra=(
            "-g",
            "*.py",
            "-g",
            "!llm/providers/**",
            "-g",
            "!llm/response_evidence.py",
        ),
    )
    assert hits == "", f"Wire 形状泄漏到 Adapter 边界之外：\n{hits}"
