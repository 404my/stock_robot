"""对外访问认证：内部回环直通、远端 Basic Auth 与同源校验。"""
import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app import create_app
from utils.config import Config

PASSWORD = "s3cret-pass"


def _basic(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"authorization": f"Basic {token}"}


@pytest.fixture
def config(tmp_path, monkeypatch):
    """固定到临时项目目录，避免读写真实用户配置。"""
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.set("api.auth.password", PASSWORD)
    return cfg


def _app() -> FastAPI:
    return create_app(push=False)


def test_internal_loopback_bypasses_auth_without_credentials(config):
    """AstrBot 等本机进程通过 127.0.0.1 调用时不弹密码。"""
    client = TestClient(_app(), client=("127.0.0.1", 43210))

    assert client.get("/api/v1/config").status_code == 200
    assert client.get("/api/v1/config/credentials/llm.api_key").status_code == 200
    assert client.get("/api/v1/config/file").status_code == 200


def test_remote_request_requires_basic_auth(config):
    client = TestClient(_app(), client=("198.51.100.7", 43211))

    response = client.get("/api/v1/config")

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Basic ")


@pytest.mark.parametrize(
    "headers",
    [
        {},
        _basic("admin", "wrong-password"),
        _basic("guest", PASSWORD),
        {"authorization": "Bearer not-basic"},
        {"authorization": "Basic !!!"},
    ],
)
def test_remote_credentials_are_rejected(config, headers):
    client = TestClient(_app(), client=("198.51.100.7", 43211))

    assert client.get("/api/v1/config", headers=headers).status_code == 401


def test_remote_valid_credentials_reach_protected_pages(config):
    """认证通过后可访问 Web UI 与原先仅限本机的配置编辑接口。"""
    client = TestClient(_app(), client=("198.51.100.7", 43211))
    headers = _basic("admin", PASSWORD)

    assert client.get("/api/v1/config", headers=headers).status_code == 200
    assert client.get("/api/v1/config/file", headers=headers).status_code == 200
    assert client.get("/api/v1/config/credentials/llm.api_key", headers=headers).status_code == 200


def test_remote_without_configured_password_fails_closed(tmp_path, monkeypatch):
    """保持默认空密码时，外部请求一律被拒，不会裸奔。"""
    monkeypatch.chdir(tmp_path)
    Config()
    client = TestClient(_app(), client=("198.51.100.7", 43211))

    response = client.get("/api/v1/config")

    assert response.status_code == 503
    assert "STOCK_ROBOT_AUTH_PASSWORD" in response.json()["detail"]


def test_auth_disabled_allows_remote(config):
    config.set("api.auth.enabled", False)
    client = TestClient(_app(), client=("198.51.100.7", 43211))

    assert client.get("/api/v1/config").status_code == 200


def test_environment_overrides_config_password(config, monkeypatch):
    monkeypatch.setenv("STOCK_ROBOT_AUTH_USERNAME", "operator")
    monkeypatch.setenv("STOCK_ROBOT_AUTH_PASSWORD", "env-pass")
    client = TestClient(_app(), client=("198.51.100.7", 43211))

    assert client.get("/api/v1/config", headers=_basic("operator", "env-pass")).status_code == 200
    assert client.get("/api/v1/config", headers=_basic("admin", PASSWORD)).status_code == 401


def test_cross_site_origin_is_rejected(config):
    """同源校验拦住借浏览器缓存的凭据发起的跨站请求。"""
    client = TestClient(_app(), client=("198.51.100.7", 43211))

    response = client.get(
        "/api/v1/config",
        headers={**_basic("admin", PASSWORD), "origin": "https://evil.example"},
    )

    assert response.status_code == 403


def test_remote_host_spoof_does_not_bypass_auth(config):
    """远端伪造 Host 为回环地址仍必须提供密码。"""
    client = TestClient(_app(), client=("198.51.100.7", 43211))

    assert client.get("/api/v1/config", headers={"host": "127.0.0.1:25618"}).status_code == 401


def test_forwarded_origin_matches_reverse_proxy(config):
    """反向代理转发头用于同源比对，认证仍按客户端真实来源执行。"""
    client = TestClient(_app(), client=("198.51.100.7", 43211))

    response = client.get(
        "/api/v1/config",
        headers={
            **_basic("admin", PASSWORD),
            "origin": "https://stock.example",
            "x-forwarded-proto": "https",
            "x-forwarded-host": "stock.example",
        },
    )

    assert response.status_code == 200
