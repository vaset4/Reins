"""使用官方FastMCP运行可重复的标准协议服务，客户端实现不参与服务端协议处理。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anyio
from mcp import types
from mcp.server.fastmcp import Context, FastMCP

_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jD1sAAAAASUVORK5CYII="
_READONLY = types.ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_CRASH_EXIT_CODE = 19


def _mark(event: str) -> None:
    """记录测试进程实际事件；传参：事件名；返回：无。"""
    root = os.environ.get("MCP_TEST_TRACE_DIR")
    if root:
        Path(root, event).write_text(str(os.getpid()), encoding="utf-8")


def build_server(port: int) -> FastMCP:
    """定义真实标准工具及错误、通知、取消路径；传参：端口；返回：官方服务。"""
    server = FastMCP(
        "reins-standard-fixture", host="127.0.0.1", port=port, log_level="ERROR"
    )

    @server.tool(annotations=_READONLY, structured_output=False)
    async def echo(text: str, ctx: Context, delay: float = 0) -> types.CallToolResult:
        """回传正文并发送无请求ID的诊断通知；传参：正文、上下文及延迟；返回：三类内容块与结构化结果。"""
        _mark("echo_called")
        await ctx.info("fixture diagnostic")
        await anyio.sleep(delay)
        value = {"text": text, "pid": os.getpid()}
        return types.CallToolResult(
            content=[
                types.TextContent(type="text", text=json.dumps(value)),
                types.ImageContent(type="image", data=_PIXEL_PNG, mimeType="image/png"),
                types.EmbeddedResource(
                    type="resource",
                    resource=types.TextResourceContents(
                        uri="fixture://material", text="原始资料", mimeType="text/plain"
                    ),
                ),
            ],
            structuredContent=value,
        )

    @server.tool(structured_output=False)
    def business_error() -> types.CallToolResult:
        """以通信成功交付真实业务失败；传参：无；返回：isError与失败证据。"""
        return types.CallToolResult(
            isError=True,
            content=[types.TextContent(type="text", text="账单已撤回")],
            structuredContent={"reason": "withdrawn"},
        )

    @server.tool()
    async def wait_for_cancel() -> str:
        """等待标准取消通知并留下实际处理证据；传参：无；返回：仅取消时退出。"""
        _mark("call_started")
        try:
            await anyio.Event().wait()
        except anyio.get_cancelled_exc_class():
            _mark("call_cancelled")
            raise
        return "unexpected completion"

    @server.tool()
    async def publish_capability(ctx: Context) -> str:
        """发布新工具并发送标准目录通知；传参：服务上下文；返回：发布回执。"""
        server.add_tool(_late_capability, name="late_capability", annotations=_READONLY)
        await ctx.session.send_tool_list_changed()
        return "new capability published"

    @server.tool()
    def crash_connection() -> str:
        """模拟服务已接收动作后的突然退出；传参：无；返回：不会返回，效果必须记为未知。"""
        _mark("call_before_crash")
        os._exit(_CRASH_EXIT_CODE)

    return server


def _late_capability() -> str:
    """返回动态发布后的真实结果；传参：无；返回：正文。"""
    return "late capability result"


def main() -> None:
    """以命令行选择标准传输；传参：CLI参数；返回：无，退出留下进程证据。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport", choices=("stdio", "streamable-http", "sse"), default="stdio"
    )
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    _mark("server_started")
    try:
        build_server(args.port).run(transport=args.transport)
    finally:
        _mark("server_exited")


if __name__ == "__main__":
    main()
