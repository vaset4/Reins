"""【后台服务】【请求查看】向RPC委托共享查询和独立导出任务。

作者：xxx
时间：2026-09-30 12:00:00
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from runtime.request_export import RequestExporter
from runtime.request_inspection import RequestInspection


class InspectionService:
    """持有导出任务生命周期，不占用会话执行或事件轮询锁。"""

    def __init__(self, data_root: Path | str) -> None:
        """创建共享领域读者；参数：数据根；返回：无。"""
        self.inspection = RequestInspection(data_root)
        self.exporter = RequestExporter(self.inspection)

    def query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """委托只读查询；参数：查询载荷；返回：稳定页面。"""
        return self.inspection.query(payload)

    def export(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """委托独立导出；参数：开始、状态或取消载荷；返回：任务状态。"""
        return self.exporter.command(payload)

    def close(self) -> None:
        """关闭本服务拥有的导出任务；参数：无；返回：无，不取消运行。"""
        self.exporter.close()
