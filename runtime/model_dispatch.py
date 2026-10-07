"""【模型资源】【交互优先】用进程持有的现有文件锁协调前台与自动维护。

作者：xxx
时间：2026-10-02 11:12:00
"""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from pathlib import Path

from runtime.cancellation import CancellationToken, ExecutionCancelled
from schedules.persistence import claim_file

DISPATCH_POLL_SECONDS = 0.05


@contextmanager
def foreground_turn(data_root: Path, run_id: str) -> Iterator[None]:
    """登记真实交互的存活范围，退出进程自动释放；参数：数据空间/运行身份；返回：交互执行窗口。"""
    identity = sha256(run_id.encode()).hexdigest()
    path = data_root / "runtime/locks/foreground" / f"{identity}.lock"
    claim = ExitStack()
    with dispatch_registry(data_root):
        if not claim.enter_context(claim_file(path)):
            claim.close()
            raise RuntimeError("foreground run already owns its dispatch activity")
    try:
        yield
    finally:
        with dispatch_registry(data_root):
            claim.close()
            path.unlink()


def foreground_active(data_root: Path) -> bool:
    """检查正在执行的交互而非残留锁文件；参数：数据根；返回：是否有存活前台运行。"""
    with dispatch_registry(data_root):
        for path in (data_root / "runtime/locks/foreground").glob("*.lock"):
            with claim_file(path) as acquired:
                if not acquired:
                    return True
            # 1. 【模型资源】【崩溃恢复】可重新认领的文件只代表已退出进程，不阻塞维护
            path.unlink()
    return False


@contextmanager
def dispatch_registry(data_root: Path) -> Iterator[None]:
    """串行化存活登记和清理，防止Windows探测与关闭竞争；参数：数据根；返回：短元数据窗口。"""
    with claim_file(
        data_root / "runtime/locks/dispatch-registry.lock", blocking=True
    ) as acquired:
        if not acquired:
            raise BlockingIOError("model dispatch registry is busy")
        yield


@contextmanager
def maintenance_call(
    data_root: Path, cancellation: CancellationToken
) -> Iterator[None]:
    """下一次维护调用等待交互空闲，已派发请求正常收尾；参数：空间/取消信号；返回：真实模型派发窗口。"""
    path = data_root / "runtime/locks/knowledge-model.lock"
    while not cancellation.cancelled:
        with claim_file(path) as acquired:
            if acquired and not foreground_active(data_root):
                yield
                return
        cancellation.wait(DISPATCH_POLL_SECONDS)
    raise ExecutionCancelled("knowledge model dispatch cancelled before sending")
