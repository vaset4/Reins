"""在评测进程中载入已留存的旧压缩器，不覆盖生产模块或修改冻结源码。

作者：xxx
时间：2026-09-25 20:00:00
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, cast


def load_baseline(source: Path) -> type[Any]:
    """按采样时的指纹加载旧摘要算法；传参：基线source目录；返回：隔离的压缩器类型。"""
    manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    paths = ("context/compaction.py", "runtime/context_preparation.py")
    for name in paths:
        digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if digest != manifest["source_sha256"][name]:
            raise ValueError(f"frozen baseline source changed: {name}")
    compaction = _load(source / paths[0], "_reins_eval_baseline_compaction")
    preparation = _load(
        source / paths[1], "_reins_eval_baseline_preparation", compaction=compaction
    )
    return cast(type[Any], preparation.ContextCompactor)


def _load(path: Path, name: str, *, compaction: ModuleType | None = None) -> ModuleType:
    """给冻结模块独立名字，旧导入只连接旧压缩算法；传参：文件、模块名和依赖；返回：已载入模块。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("baseline source is not a Python module")
    module = importlib.util.module_from_spec(spec)
    if compaction is not None:
        module.__dict__["__builtins__"] = {
            **vars(builtins),
            "__import__": partial(_baseline_import, compaction=compaction),
        }
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _baseline_import(
    name: str, *args: Any, compaction: ModuleType, **kwargs: Any
) -> Any:
    """仅替换冻结压缩器的算法依赖，其他仍用共享生产边界；传参：Python导入参数；返回：模块。"""
    return (
        compaction
        if name == "context.compaction"
        else builtins.__import__(name, *args, **kwargs)
    )
