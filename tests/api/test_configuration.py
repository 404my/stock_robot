"""受控配置 API 的边界契约测试。"""
import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.configuration import mask_secret
from api.runtime import ReloadResult, RuntimeManager
from utils.config import Config


class RuntimeStub(RuntimeManager):
    """记录配置路由交给运行时的重载请求。"""

    def __init__(self, result: ReloadResult | None = None) -> None:
        self.reload_configs: list[Config] = []
        self.result = result or ReloadResult(applied=True)

    def reload(self, config: Config) -> ReloadResult:
        self.reload_configs.append(config)
        return self.result


def test_extra_analysis_defaults_off_and_saves_boolean(runtime_config_client):
    config, runtime, client = runtime_config_client
    current = client.get("/api/v1/config").json()
    assert current["config"]["analysis"]["extra_enabled"] is False
    response = client.put("/api/v1/config", json={"config": {"analysis": {"extra_enabled": True}},
                                                  "revision": current["revision"]})
    assert response.status_code == 200
    assert runtime.reload_configs[-1].get("analysis.extra_enabled") is True
    assert Config(config_dir=config.config_dir).get("analysis.extra_enabled") is True


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_extra_analysis_rejects_non_boolean(runtime_config_client, value):
    _, runtime, client = runtime_config_client
    current = client.get("/api/v1/config").json()
    response = client.put("/api/v1/config", json={"config": {"analysis": {"extra_enabled": value}},
                                                  "revision": current["revision"]})
    assert response.status_code == 422 and not runtime.reload_configs


@pytest.fixture
def config_client(tmp_path: Path, monkeypatch):
    """将配置固定到临时项目目录，避免测试读取真实用户配置。"""
    monkeypatch.chdir(tmp_path)
    config = Config()
    config.set("llm.api_key", "abcde12345vwxyz")
    config.set("push.email.smtp_password", "smtp-secret-xyz")
    return config, TestClient(
        create_app(push=False),
        client=("127.0.0.1", 43210),
        raise_server_exceptions=False,
    )


@pytest.fixture
def runtime_config_client(tmp_path: Path, monkeypatch):
    """创建带运行时桩的配置客户端，隔离真实 AgentCore。"""
    monkeypatch.chdir(tmp_path)
    config = Config()
    runtime = RuntimeStub()
    return config, runtime, TestClient(
        create_app(push=False, runtime=runtime),
        client=("127.0.0.1", 43210),
        raise_server_exceptions=False,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("short", "*****"),
        ("12345678", "********"),
        ("123456789", "1234*6789"),
        ("12345678901", "1234***8901"),
        ("123456789012", "12345**89012"),
    ],
)
def test_mask_secret_preserves_only_the_contractually_allowed_characters(value, expected):
    """密钥掩码严格遵循短密钥全遮蔽、长密钥首尾保留规则。"""
    assert mask_secret(value) == expected


def test_get_config_only_returns_safe_whitelist_and_read_only_paths(config_client):
    """公开配置不泄露密钥，且只给出两个只读路径。"""
    config, client = config_client

    response = client.get("/api/v1/config")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"config", "paths", "revision"}
    assert payload["paths"] == {
        "state_dir": str(config.config_dir),
        "config_file": str(config.config_dir / "config.yaml"),
    }
    assert payload["config"]["llm"]["api_key"] == {
        "configured": True,
        "masked": "abcde*****vwxyz",
    }
    assert payload["config"]["push"]["email"]["smtp_password"] == {
        "configured": True,
        "masked": "smtp-*****t-xyz",
    }
    assert "wecom" not in payload["config"]["push"]
    assert "abcde12345vwxyz" not in str(payload)
    assert "smtp-secret-xyz" not in str(payload)
    assert "wecom-secret-xyz" not in str(payload)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("llm.api_key", "abcde12345vwxyz"),
        ("push.email.smtp_password", "smtp-secret-xyz"),
    ],
)
def test_get_credential_only_allows_explicit_secret_keys(config_client, key, expected):
    """完整密钥读取只允许当前版本的明确白名单键。"""
    _, client = config_client

    response = client.get(f"/api/v1/config/credentials/{key}")

    assert response.status_code == 200
    assert response.json() == {"value": expected}


def test_get_credential_rejects_unknown_key(config_client):
    """非白名单密钥不能通过凭据接口读取。"""
    _, client = config_client

    response = client.get("/api/v1/config/credentials/llm.model")

    assert response.status_code == 404
    assert client.get("/api/v1/config/credentials/push.wecom.secret").status_code == 404


def test_get_credential_allows_loopback_client(config_client):
    """本机浏览器仍可按需读取完整凭据。"""
    _, client = config_client

    response = client.get("/api/v1/config/credentials/llm.api_key")

    assert response.status_code == 200
    assert response.json() == {"value": "abcde12345vwxyz"}


def test_get_credential_rejects_cross_site_origin_and_host(config_client):
    """其他网站不能借用户浏览器读取本机完整凭据。"""
    _, client = config_client
    path = "/api/v1/config/credentials/push.email.smtp_password"
    assert client.get(path, headers={"origin": "https://other.example"}).status_code == 403
    # 非回环 Host 不再被视为本机请求，未配置密码时由认证层直接拒绝（失败关闭）
    assert client.get(path, headers={"host": "other.example"}).status_code == 503
    assert client.get(path, headers={"origin": "http://testserver"}).status_code == 200


def test_remote_client_without_password_is_rejected(tmp_path, monkeypatch):
    """对外监听且未配置密码时，远端连已脱敏配置也不放行。"""
    monkeypatch.chdir(tmp_path)
    config = Config()
    config.set("llm.api_key", "remote-secret")
    client = TestClient(create_app(push=False), client=("192.0.2.25", 43210))

    credential = client.get("/api/v1/config/credentials/llm.api_key")
    public_config = client.get("/api/v1/config")

    assert credential.status_code == 503
    assert public_config.status_code == 503


def test_update_deep_merges_only_submitted_values_and_persists_once(config_client, monkeypatch):
    """局部更新不重置同层字段，并在一次写盘后可被新实例读取。"""
    config, client = config_client
    persist_calls = 0
    original_persist = config._persist

    def count_persist():
        nonlocal persist_calls
        persist_calls += 1
        original_persist()

    monkeypatch.setattr(config, "_persist", count_persist)
    monkeypatch.setattr("api.configuration.Config", lambda: config)

    response = client.put("/api/v1/config", json={
        "config": {"llm": {"temperature": 0.8}},
    })

    assert response.status_code == 200
    assert response.json()["restart_required"] is False
    assert response.json()["config"]["llm"]["temperature"] == 0.8
    assert response.json()["config"]["llm"]["model"] == "gpt-4o"
    assert persist_calls == 1
    assert Config(config_dir=config.config_dir).get("llm.temperature") == 0.8


def test_hot_update_persists_then_applies_runtime(runtime_config_client):
    """热更新写盘成功后应立刻交给运行时生效。"""
    config, runtime, client = runtime_config_client

    response = client.put("/api/v1/config", json={
        "config": {"llm": {"temperature": 0.8}},
    })

    assert response.status_code == 200
    assert response.json()["persisted"] is True
    assert response.json()["applied"] is True
    assert response.json()["restart_required"] is False
    assert len(runtime.reload_configs) == 1
    assert Config(config_dir=config.config_dir).get("llm.temperature") == 0.8


def test_listener_only_update_persists_without_runtime_reload(runtime_config_client):
    """仅监听配置变更只写盘并要求重启，不能重建运行时。"""
    config, runtime, client = runtime_config_client

    response = client.put("/api/v1/config", json={
        "config": {"api": {"port": 3000}},
    })

    assert response.status_code == 200
    assert response.json()["persisted"] is True
    assert response.json()["applied"] is False
    assert response.json()["restart_required"] is True
    assert runtime.reload_configs == []
    assert Config(config_dir=config.config_dir).get("api.port") == 3000


def test_mixed_listener_and_hot_update_applies_hot_part(runtime_config_client):
    """混合更新需保留监听端口给重启，并仅以热字段重载运行时。"""
    config, runtime, client = runtime_config_client
    original_port = config.get("api.port")

    response = client.put("/api/v1/config", json={
        "config": {"api": {"port": 3000}, "llm": {"temperature": 0.8}},
    })

    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert response.json()["restart_required"] is True
    assert len(runtime.reload_configs) == 1
    assert runtime.reload_configs[0].get("api.port") == original_port
    assert runtime.reload_configs[0].get("llm.temperature") == 0.8


def test_reload_failure_is_reported_without_secret(tmp_path: Path, monkeypatch):
    """重载失败仍保留写盘结果，且错误响应不泄露新密钥。"""
    monkeypatch.chdir(tmp_path)
    runtime = RuntimeStub(ReloadResult(applied=False, error="模拟运行时重建失败"))
    client = TestClient(
        create_app(push=False, runtime=runtime),
        client=("127.0.0.1", 43210),
        raise_server_exceptions=False,
    )

    response = client.put("/api/v1/config", json={
        "config": {"llm": {"api_key": "new-secret"}},
    })

    assert response.status_code == 200
    assert response.json()["persisted"] is True
    assert response.json()["applied"] is False
    assert response.json()["reload_error"] == "模拟运行时重建失败"
    assert "new-secret" not in response.text


def test_update_without_runtime_reports_unavailable_reload(config_client, monkeypatch):
    """无运行时模式热字段仍可写盘，并返回可恢复的未连接状态。"""
    config, client = config_client
    monkeypatch.setattr("api.configuration.Config", lambda: config)

    response = client.put("/api/v1/config", json={
        "config": {"llm": {"temperature": 0.8}},
    })

    assert response.status_code == 200
    assert response.json()["persisted"] is True
    assert response.json()["applied"] is False
    assert response.json()["restart_required"] is False
    assert "reload_error" in response.json()


def test_config_update_deep_merges_and_persists_once(tmp_path: Path, monkeypatch):
    """Config.update 递归合并后只触发一次持久化。"""
    config = Config(config_dir=tmp_path)
    persist_calls = 0
    original_persist = config._persist

    def count_persist():
        nonlocal persist_calls
        persist_calls += 1
        original_persist()

    monkeypatch.setattr(config, "_persist", count_persist)

    config.update({"data": {"cache_ttl": {"daily": 12}}})

    assert config.get("data.cache_ttl.daily") == 12
    assert config.get("data.cache_ttl.news") == 21600
    assert persist_calls == 1


def test_empty_secret_does_not_overwrite_existing_value(config_client, monkeypatch):
    """编辑页提交空密钥表示保留旧值。"""
    config, client = config_client
    monkeypatch.setattr("api.configuration.Config", lambda: config)

    response = client.put("/api/v1/config", json={
        "config": {"llm": {"api_key": ""}},
    })

    assert response.status_code == 200
    assert config.get("llm.api_key") == "abcde12345vwxyz"
    assert response.json()["config"]["llm"]["api_key"]["masked"] == "abcde*****vwxyz"


def test_update_rejects_malformed_json_without_persisting(config_client):
    """畸形 JSON 应是可恢复的客户端错误，且不能写入配置。"""
    config, client = config_client
    before = copy.deepcopy(config.data)

    response = client.put(
        "/api/v1/config",
        content=b'{"config":',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert Config(config_dir=config.config_dir).data == before


@pytest.mark.parametrize(
    ("body", "path"),
    [
        ({"config": {"llm": {"temperature": float("nan")}}}, "llm.temperature"),
        ({"config": {"llm": {"temperature": float("inf")}}}, "llm.temperature"),
        ({"config": {"llm": {"temperature": float("-inf")}}}, "llm.temperature"),
        ({"config": {"signal": {"thresholds": {"watch": float("nan")}}}},
         "signal.thresholds.watch"),
        ({"config": {"signal": {"thresholds": {"attack": float("nan")}}}},
         "signal.thresholds.attack"),
        ({"config": {"signal": {"thresholds": {"watch": float("inf")}}}},
         "signal.thresholds.watch"),
        ({"config": {"signal": {"thresholds": {"attack": float("-inf")}}}},
         "signal.thresholds.attack"),
    ],
)
def test_update_rejects_non_finite_numbers_without_persisting(config_client, body, path):
    """温度与信号阈值不接受 NaN/正负无穷，也绝不写入文件。"""
    config, client = config_client
    before = config.get(path)

    response = client.put(
        "/api/v1/config",
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert Config(config_dir=config.config_dir).get(path) == before


@pytest.mark.parametrize(
    "body",
    [
        {"config": {"llm": {"temperature": "0"}}},
        {"config": {"llm": {"max_tokens": True}}},
        {"config": {"llm": {"retry_times": "0"}}},
        {"config": {"llm": {"timeout_seconds": False}}},
        {"config": {"data": {"cache_ttl": {"daily": "1"}}}},
        {"config": {"api": {"port": True}}},
        {"config": {"push": {"max_symbols_per_subscription": "1"}}},
        {"config": {"push": {"email": {"smtp_port": False}}}},
        {"config": {"signal": {"thresholds": {"watch": "1"}}}},
    ],
)
def test_update_rejects_numeric_type_boundaries_without_persisting(config_client, body):
    """数值字段拒绝字符串和布尔值，不持久化任何无效更新。"""
    config, client = config_client
    before = copy.deepcopy(config.data)

    response = client.put("/api/v1/config", json=body)

    assert response.status_code == 422
    assert Config(config_dir=config.config_dir).data == before


@pytest.mark.parametrize(
    "body",
    [
        {"config": {"llm": {"temperature": 0}}},
        {"config": {"llm": {"temperature": 2}}},
        {"config": {"llm": {"max_tokens": 1}}},
        {"config": {"llm": {"max_tokens": 128000}}},
        {"config": {"llm": {"retry_times": 0}}},
        {"config": {"llm": {"retry_times": 10}}},
        {"config": {"llm": {"timeout_seconds": 1}}},
        {"config": {"llm": {"timeout_seconds": 600}}},
        {"config": {"data": {"cache_ttl": {"daily": 1}}}},
        {"config": {"data": {"cache_ttl": {"quarterly": 1}}}},
        {"config": {"data": {"cache_ttl": {"news": 1}}}},
        {"config": {"api": {"port": 1}}},
        {"config": {"api": {"port": 65535}}},
        {"config": {"push": {"max_symbols_per_subscription": 1}}},
        {"config": {"push": {"email": {"smtp_port": 1}}}},
        {"config": {"signal": {"thresholds": {"watch": 1, "attack": 10}}}},
    ],
)
def test_update_accepts_closed_interval_endpoints(config_client, body):
    """允许的闭区间端点能够保存为有效配置。"""
    _, client = config_client

    response = client.put("/api/v1/config", json=body)

    assert response.status_code == 200


@pytest.mark.parametrize("field", ["host", "port"])
def test_api_bind_change_requires_restart(config_client, monkeypatch, field):
    """监听地址或端口变更明确提示用户重启服务。"""
    config, client = config_client
    monkeypatch.setattr("api.configuration.Config", lambda: config)
    value = "0.0.0.0" if field == "host" else 3000

    response = client.put("/api/v1/config", json={"config": {"api": {field: value}}})

    assert response.status_code == 200
    assert response.json()["restart_required"] is True
    assert config.get(f"api.{field}") == value


@pytest.mark.parametrize(
    "body",
    [
        {"config": {"unknown": True}},
        {"config": {"llm": {"unknown": True}}},
        {"unexpected": True},
        {"config": {"push": {"email": {"unknown": True}}}},
    ],
)
def test_update_rejects_unknown_fields(config_client, body):
    """白名单以外字段不能被写入配置文件。"""
    _, client = config_client

    response = client.put("/api/v1/config", json=body)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"config": {"llm": {"provider": "gemini"}}},
        {"config": {"llm": {"model": "   "}}},
        {"config": {"llm": {"temperature": -0.01}}},
        {"config": {"llm": {"temperature": 2.01}}},
        {"config": {"llm": {"max_tokens": 0}}},
        {"config": {"llm": {"max_tokens": 128001}}},
        {"config": {"llm": {"retry_times": -1}}},
        {"config": {"llm": {"retry_times": 11}}},
        {"config": {"llm": {"timeout_seconds": 0}}},
        {"config": {"llm": {"timeout_seconds": 601}}},
        {"config": {"data": {"cache_ttl": {"daily": 0}}}},
        {"config": {"api": {"host": " "}}},
        {"config": {"api": {"port": 0}}},
        {"config": {"api": {"port": 65536}}},
        {"config": {"push": {"max_symbols_per_subscription": 0}}},
        {"config": {"push": {"email": {"smtp_host": " "}}}},
        {"config": {"push": {"email": {"smtp_port": 0}}}},
        {"config": {"push": {"email": {"smtp_user": " "}}}},
        {"config": {"push": {"email": {"to_addr": " "}}}},
        {"config": {"push": {"wecom": {"corp_id": " "}}}},
        {"config": {"push": {"wecom": {"agent_id": " "}}}},
        {"config": {"push": {"wecom": {"to_user": " "}}}},
        {"config": {"signal": {"thresholds": {"watch": 0}}}},
        {"config": {"signal": {"thresholds": {"watch": 7}}}},
        {"config": {"signal": {"thresholds": {"attack": 11}}}},
        {"config": {"signal": {"actions": {"attack": {"action": " "}}}}},
        {"config": {"signal": {"actions": {"watch": {"position": " "}}}}},
    ],
)
def test_update_rejects_invalid_config_boundaries(config_client, body):
    """所有受控字段在越界、空白或违反阈值关系时均被拒绝。"""
    _, client = config_client

    response = client.put("/api/v1/config", json=body)

    assert response.status_code == 422


@pytest.mark.parametrize("budget", [None, 2000])
def test_output_budget_round_trips_through_config_api_and_hot_reload(runtime_config_client, budget):
    config, runtime, client = runtime_config_client
    response = client.put("/api/v1/config", json={"config": {"llm": {"max_tokens": budget}}})
    assert response.status_code == 200
    assert response.json()["config"]["llm"]["max_tokens"] == budget
    assert response.json()["applied"] is True
    assert runtime.reload_configs[-1].get("llm.max_tokens") == budget
    assert Config(config_dir=config.config_dir).get("llm.max_tokens") == budget
    assert client.get("/api/v1/config").json()["config"]["llm"]["max_tokens"] == budget


def test_maintenance_update_persists_without_runtime_reload(runtime_config_client):
    """清理开关由维护循环每轮读取，保存后不需要重建 Agent 核心。"""
    config, runtime, client = runtime_config_client

    response = client.put("/api/v1/config", json={
        "config": {"maintenance": {"report_cleanup_enabled": False}},
    })

    assert response.status_code == 200
    body = response.json()
    assert body["persisted"] is True
    assert body["applied"] is False
    assert body["restart_required"] is False
    assert runtime.reload_configs == []
    reloaded = Config(config_dir=config.config_dir)
    assert reloaded.get("maintenance.report_cleanup_enabled") is False
    assert "report_retention_days" not in reloaded.data["maintenance"]


def test_mixed_maintenance_and_hot_update_still_reloads_runtime(runtime_config_client):
    """与热字段一起保存时仍按热字段重建运行时，维护设置一并落盘。"""
    _config, runtime, client = runtime_config_client

    response = client.put("/api/v1/config", json={
        "config": {
            "maintenance": {"report_cleanup_enabled": False},
            "llm": {"temperature": 0.8},
        },
    })

    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert len(runtime.reload_configs) == 1
    assert runtime.reload_configs[0].get("maintenance.report_cleanup_enabled") is False


@pytest.mark.parametrize("value", [7, 14, "7", True])
def test_retention_days_is_not_editable(config_client, value):
    """保留期固定 7 天，不提供按天数修改的配置项。"""
    _, client = config_client

    response = client.put("/api/v1/config", json={
        "config": {"maintenance": {"report_retention_days": value}},
    })

    assert response.status_code == 422
