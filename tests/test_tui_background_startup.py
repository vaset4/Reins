"""终端所需的独立后台启动合同。

作者：xxx
时间：2026-09-29 21:00:00
"""

from app.background import client


def test_real_background_publishes_endpoint_where_frontend_connects(
    tmp_path, monkeypatch
):
    """在隔离根启动真实宿主并连回同一实例，最终释放进程；传参：临时根；返回：无。"""
    processes = []
    launch = client._launch

    def record_launch(**options):
        """记录真实进程以便失败也能释放；传参：启动参数；返回：进程句柄。"""
        process = launch(**options)
        processes.append(process)
        return process

    monkeypatch.setattr(client, "_launch", record_launch)
    root = tmp_path / "data"
    try:
        connected = client.ensure_running(project_root=tmp_path, data_root=root)
        assert client.is_running(root)
        assert client.connect(root).instance == connected.instance
        assert connected.call("attach")["status"] == "idle"
        assert connected.stop()
        assert not client.is_running(root)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
