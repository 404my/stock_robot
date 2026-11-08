"""自动采集控制接口的持久、离线和严格校验测试。"""
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from radar.calendar import CalendarData, TradingCalendar
from radar.collector_store import CollectorStore
from radar.universe import UniverseConfigError, UniverseRepository
from radar.update_service import UpdateStore
from utils.config import Config


@pytest.fixture
def collector_api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # 访问认证层读取当前目录的默认配置（未设密码则外部失败关闭）
    config = Config(config_dir=tmp_path)
    monkeypatch.setattr("api.radar.Config", lambda: config)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    first = date(now.year-2, 1, 1)
    last = date(now.year+1, 12, 31)
    calendar = TradingCalendar(tmp_path / "radar_collector.db", fetcher=lambda: CalendarData(tuple(stamp.date() for stamp in pd.bdate_range(first, last)), first, last, now, "test-fixture"))
    calendar.refresh()
    store = CollectorStore(tmp_path / "radar_collector.db")
    client = TestClient(create_app(core=None, push=False), client=("127.0.0.1", 50123))
    return client, store, config


def test_plan_settings_are_durable_and_revision_conflict_is_409(collector_api):
    client, _, _ = collector_api
    before = client.get("/api/v1/radar/collector/config")
    assert before.status_code == 200
    body = {"enabled": True, "hour": 19, "minute": 15, "revision": before.json()["revision"]}
    saved = client.put("/api/v1/radar/collector/config", json=body)
    assert saved.status_code == 200
    assert saved.json()["hour"] == 19
    assert saved.json()["revision"] != body["revision"]
    assert client.put("/api/v1/radar/collector/config", json=body).status_code == 409
    other = TestClient(create_app(core=None, push=False))
    assert other.get("/api/v1/radar/collector/config").json()["minute"] == 15


@pytest.mark.parametrize("field,value", [("hour", True), ("hour", "18"), ("hour", 24), ("minute", 60), ("minute", 1.5), ("enabled", 1), ("revision", -1), ("revision", True)])
def test_plan_rejects_coerced_or_out_of_range_fields(collector_api, field, value):
    client, _, _ = collector_api
    body = {"enabled": False, "hour": 18, "minute": 30, "revision": 0}
    body[field] = value
    assert client.put("/api/v1/radar/collector/config", json=body).status_code == 422


def test_offline_manual_and_retry_are_durable_and_do_not_require_daemon(collector_api):
    client, store, _ = collector_api
    response = client.post("/api/v1/radar/collector/runs", json={"universe_id": "cn_hk_etf"})
    assert response.status_code == 202
    assert store.list_runs() == []
    legacy = client.post("/api/v1/radar/refresh", json={"universe_id": "cn_hk_etf"})
    assert legacy.status_code == 202
    assert legacy.json()["id"] == response.json()["id"]
    assert len(UpdateStore(store.db_path).list_runs()) == 1
    run = store.enqueue("cn_hk_etf", date(2026, 9, 29), "test", "manual", ["510300"])
    store.finish(run["id"], "failed", "网络失败")
    assert client.post(f'/api/v1/radar/collector/runs/{run["id"]}/retry').status_code == 202
    assert store.get_run(run["id"])["status"] == "queued"


def test_online_manual_deduplicates_and_survives_web_router_recreation(collector_api):
    client, store, _ = collector_api
    store.record_started(worker_id="live-worker")
    first = client.post("/api/v1/radar/refresh", json={"universe_id": "cn_hk_etf"})
    assert first.status_code == 202
    second = client.post("/api/v1/radar/collector/runs", json={"universe_id": "cn_hk_etf"})
    assert second.status_code == 202
    assert first.json()["task_id"] == second.json()["task_id"]
    assert len(UpdateStore(store.db_path).list_runs()) == 1
    other = TestClient(create_app(core=None, push=False))
    restored = other.get(f'/api/v1/radar/refresh/{first.json()["task_id"]}')
    assert restored.status_code == 200
    assert restored.json()["status"] == "queued"
    assert restored.json()["universe_id"] == "cn_hk_etf"
    assert other.get("/api/v1/radar/collector/runs").json()[0]["id"] == first.json()["task_id"]


def test_status_uses_freshness_and_calendar_cache_only(collector_api, monkeypatch):
    client, store, _ = collector_api
    store.record_started(worker_id="live-worker", phase="syncing")
    store.record_heartbeat(worker_id="live-worker", phase="blocked", error_summary="日历不可用")
    monkeypatch.setattr("radar.calendar.fetch_akshare_calendar", lambda **_: pytest.fail("配置查询不得联网"))
    saved = store.update_settings(enabled=True, hour=19, minute=15, revision=store.settings()["revision"])
    response = client.get("/api/v1/radar/collector/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service_online"] is True
    assert payload["runtime"]["phase"] == "blocked"
    assert payload["runtime"]["last_error"] == "日历不可用"
    assert payload["schedule"]["hour"] == saved["hour"]
    assert "19:15:00" in payload["next_scheduled_at"]
    assert "19:15:00" in payload["next_overseas_check_at"]
    with closing(sqlite3.connect(store.db_path)) as conn, conn:
        conn.execute("UPDATE collector_runtime SET heartbeat_at=?", ((datetime.now().astimezone()-timedelta(seconds=61)).isoformat(),))
    payload = client.get("/api/v1/radar/collector/status").json()
    assert payload["service_online"] is False
    assert payload["runtime"]["status"] == "offline"


def test_missing_calendar_reports_null_next_plan_without_network(collector_api, monkeypatch):
    client, store, _ = collector_api
    store.update_settings(enabled=True, hour=18, minute=30, revision=store.settings()["revision"])
    with closing(sqlite3.connect(store.db_path)) as conn, conn:
        conn.execute("DELETE FROM radar_calendar_cache")
    monkeypatch.setattr("radar.calendar.fetch_akshare_calendar", lambda **_: pytest.fail("查询不得联网"))
    payload = client.get("/api/v1/radar/collector/status").json()
    assert payload["next_scheduled_at"] is None
    assert "18:30:00" in payload["next_overseas_check_at"]
    assert "日历" in payload["schedule_error"]


def test_partial_retry_keeps_success_items(collector_api):
    client, store, _ = collector_api
    store.record_started(worker_id="live-worker")
    run = store.enqueue("cn_hk_etf", date(2026, 9, 29), "test", "manual", ["510300", "510500"])
    store.record_item(run["id"], "510300", "completed")
    store.record_item(run["id"], "510500", "failed", "网络失败")
    store.finish(run["id"], "partial", "部分失败")
    response = client.post(f'/api/v1/radar/collector/runs/{run["id"]}/retry')
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert [item["status"] for item in response.json()["items"]] == ["completed", "pending"]


class FakeStartup:
    platform = "win32"
    def __init__(self, *args, **kwargs):
        self.enabled = False
    def status(self):
        return {"editable": True, "supported": True, "provider": "task_scheduler", "registered": self.enabled, "enabled": self.enabled, "error": None}
    def migrate_legacy(self, *, state_dir):
        return {"unsafe_overlap": False, "errors": []}
    def set_enabled(self, enabled):
        self.enabled = enabled
        return self.status()


def test_startup_registration_is_not_claimed_as_service_online(collector_api, monkeypatch):
    client, store, _ = collector_api
    fake = FakeStartup()
    monkeypatch.setattr("api.radar.CollectorStartup", lambda *_, **__: fake)
    monkeypatch.setattr("api.radar._STARTUP_HEARTBEAT_WAIT_SECONDS", 0)
    response = client.put("/api/v1/radar/collector/startup", json={"enabled": True})
    assert response.status_code == 503
    assert fake.enabled is True
    assert store.settings()["enabled"] is False
    assert client.get("/api/v1/radar/collector/startup").json()["service_online"] is False


def test_startup_is_local_only_and_linux_read_only(collector_api, monkeypatch):
    client, _, _ = collector_api
    fake = FakeStartup()
    fake.platform = "linux"
    fake.status = lambda: {"editable": False, "supported": True, "provider": "systemd", "enabled": False}
    monkeypatch.setattr("api.radar.CollectorStartup", lambda *_, **__: fake)
    assert client.put("/api/v1/radar/collector/startup", json={"enabled": True}).status_code == 403
    remote = TestClient(create_app(core=None, push=False), client=("192.0.2.25", 50124))
    assert remote.put("/api/v1/radar/collector/startup", json={"enabled": True}).status_code == 503
    assert client.put("/api/v1/radar/collector/config", json={"enabled": False, "hour": 18, "minute": 30, "revision": client.get("/api/v1/radar/collector/config").json()["revision"]}).status_code == 200


def test_performance_reuses_covered_history_and_expands_only_earlier_range(collector_api, monkeypatch):
    client, _, _ = collector_api
    class Provider:
        adjustment = "unadjusted"
        def __init__(self):
            self.calls = []
        def supports(self, asset_type):
            return True
        def fetch_spot(self):
            raise NotImplementedError
        def fetch_daily(self, symbol, start, end):
            self.calls.append((start, end))
            days = pd.bdate_range(start, end)
            return pd.DataFrame({"date": [day.date() for day in days], "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "amount": 100.0})
    provider = Provider()
    monkeypatch.setattr("api.radar.FallbackETFDataProvider", lambda *_: provider)
    monkeypatch.setattr("api.radar.fetch_benchmark_closes", lambda item, start, end, **kwargs: pd.Series(10.0, index=pd.bdate_range(start, end)))
    params = {"universe_id": "cn_hk_etf", "symbol": "510300", "start_date": "2026-09-01", "end_date": "2026-09-30"}
    first = client.get("/api/v1/radar/performance", params=params)
    assert first.status_code == 200, first.json()
    params["start_date"] = "2026-09-10"
    second = client.get("/api/v1/radar/performance", params=params)
    assert second.status_code == 200, second.json()
    assert len(provider.calls) == 1
    assert second.json()["start_date"] == "2026-09-10"
    params["start_date"] = "2026-08-01"
    assert client.get("/api/v1/radar/performance", params=params).status_code == 200
    assert len(provider.calls) == 2
    assert provider.calls[-1][0] == date(2026, 8, 1)


def test_status_keeps_runtime_and_audit_when_pool_configuration_is_broken(collector_api, monkeypatch):
    client, store, _ = collector_api
    store.record_started(worker_id="live-worker")
    run = store.enqueue("cn_hk_etf", date(2026, 9, 29), "broken-config", "manual", ["510300"])
    store.record_event("error", "标的池配置无效")

    def broken_load(_):
        raise UniverseConfigError("解析标的池失败: broken.yaml")

    monkeypatch.setattr(UniverseRepository, "load_all", broken_load)
    response = client.get("/api/v1/radar/collector/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service_online"] is True
    assert payload["runtime"]["worker_id"] == "live-worker"
    assert payload["runs"][0]["id"] == run["id"]
    assert payload["recent_events"][0]["message"] == "标的池配置无效"
    assert "broken.yaml" in payload["config_error"]
    assert payload["items"] == []


def test_status_scope_excludes_disabled_pools(collector_api, monkeypatch):
    client, _, _ = collector_api
    repository = UniverseRepository(Path(__file__).parents[2] / "config/radar_universes")
    pools = repository.load_all()
    disabled_id = pools[0].id
    enabled_id = pools[1].id
    pools = [item.model_copy(update={"enabled": item.id != disabled_id}) for item in pools]
    monkeypatch.setattr(UniverseRepository, "load_all", lambda _: pools)
    payload = client.get("/api/v1/radar/collector/status").json()
    assert {item["universe_id"] for item in payload["items"]} == {enabled_id}
    assert payload["config_error"] is None


def test_startup_refuses_registration_when_legacy_overlap_is_unsafe(collector_api, monkeypatch):
    client, store, _ = collector_api
    fake = FakeStartup()
    fake.migrate_legacy = lambda **_: {"unsafe_overlap": True, "errors": ["旧进程无法停止"], "active_legacy_process_ids": [123]}
    monkeypatch.setattr("api.radar.CollectorStartup", lambda *_, **__: fake)
    response = client.put("/api/v1/radar/collector/startup", json={"enabled": True})
    assert response.status_code == 409
    assert fake.enabled is False
    assert "旧采集" in response.json()["detail"]["message"]
    assert response.json()["detail"]["migration"]["active_legacy_process_ids"] == [123]
    assert "旧采集" in store.recent_events()[0]["message"]


@pytest.mark.parametrize("status", ["queued", "running", "retry_wait"])
def test_retry_pending_task_is_idempotent_without_resetting_attempts(collector_api, status):
    client, store, _ = collector_api
    store.record_started(worker_id="live-worker")
    run = store.enqueue("cn_hk_etf", date(2026, 9, 29), "idempotent", "manual", ["510300"])
    if status != "queued":
        store.claim("live-worker")
    if status == "retry_wait":
        store.finish(run["id"], "failed", "网络错误", worker_id="live-worker")
        store.retry(run["id"], manual=False)
    before = store.get_run(run["id"])
    for _ in range(2):
        response = client.post(f'/api/v1/radar/collector/runs/{run["id"]}/retry')
        assert response.status_code == 202
        assert response.json() == before
    assert store.get_run(run["id"]) == before
