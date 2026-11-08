"""维护接口：只允许本机或已登录的同源页面触发，且不依赖运行时核心。"""
from fastapi.testclient import TestClient

from api.app import create_app


class RunnerStub:
    def __init__(self):
        self.calls = 0
        self.payload = {
            "config": {
                "report_cleanup_enabled": True,
                "sweep_interval_minutes": 360,
            },
            "running": True,
            "last_run": {"state": "ok", "deleted": 2},
        }

    def status(self):
        return self.payload

    def run_once(self):
        self.calls += 1
        return {"state": "ok", "deleted": 2, "scanned": 3}


def test_status_returns_defaults_without_runner():
    client = TestClient(create_app(push=False), client=("127.0.0.1", 43210))

    response = client.get("/api/v1/maintenance/status")

    assert response.status_code == 200
    body = response.json()
    assert body["running"] is False
    assert body["last_run"] is None
    assert body["config"] == {
        "report_cleanup_enabled": True,
        "sweep_interval_minutes": 360,
    }


def test_status_uses_runner_snapshot():
    runner = RunnerStub()
    client = TestClient(
        create_app(push=False, maintenance=runner),
        client=("127.0.0.1", 43210),
    )

    assert client.get("/api/v1/maintenance/status").json() == runner.payload


def test_sweep_only_accepts_trusted_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = RunnerStub()
    app = create_app(push=False, maintenance=runner)
    local = TestClient(app, client=("127.0.0.1", 43210))

    response = local.post("/api/v1/maintenance/sweep")

    assert response.status_code == 200
    assert response.json()["deleted"] == 2
    assert runner.calls == 1

    # 远端未配置密码时由访问认证层直接拒绝，请求不会进入业务路由。
    remote = TestClient(app, client=("192.0.2.25", 43210))
    assert remote.post("/api/v1/maintenance/sweep").status_code == 503
    assert runner.calls == 1


def test_sweep_reports_unavailable_without_runner():
    client = TestClient(create_app(push=False), client=("127.0.0.1", 43210))

    assert client.post("/api/v1/maintenance/sweep").status_code == 503
