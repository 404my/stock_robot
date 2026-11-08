"""对外访问安全：本机内部请求直通，其余请求强制 HTTP Basic Auth。

监听地址改为 ``0.0.0.0`` 后，Web UI 与 API 可能暴露到局域网或公网。本模块在
ASGI 入口统一拦截：AstrBot 插件等本机进程通过 ``http://127.0.0.1:25618`` 调用时
免验证；外部浏览器必须输入账号密码。反向代理场景下，只有回环来源且 Host 仍为
回环地址才免验证，避免代理把整站流量伪装成本机请求而绕过认证。
"""

from __future__ import annotations

import base64
import binascii
import hmac
import os
from collections.abc import Callable
from typing import Any

from fastapi import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

AUTH_USERNAME_ENV = "STOCK_ROBOT_AUTH_USERNAME"
AUTH_PASSWORD_ENV = "STOCK_ROBOT_AUTH_PASSWORD"

# TestClient 的默认来源为 testclient + testserver，httpx ASGITransport 为 127.0.0.1 + test，
# 网络请求无法伪造这些本地客户端标识。
_INTERNAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "testclient"}
_INTERNAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1", "testserver", "test"}


def _expected_origin(request: Request) -> str:
    """按反向代理转发头还原浏览器所见的源站，用于同源校验。"""
    scheme = (request.headers.get("x-forwarded-proto") or request.url.scheme).split(",")[0].strip()
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host")
            or request.url.netloc).split(",")[0].strip()
    return f"{scheme}://{host}"


def is_same_origin(request: Request) -> bool:
    """无 Origin 头的命令行/插件调用放行；浏览器跨站请求一律不视为同源。"""
    origin = request.headers.get("origin")
    return origin is None or origin == _expected_origin(request)


def is_internal_request(request: Request) -> bool:
    """仅本机回环来源且 Host 为回环地址的请求免验证。"""
    client = request.client
    if client is None or client.host not in _INTERNAL_CLIENT_HOSTS:
        return False
    if request.url.hostname not in _INTERNAL_HOSTNAMES:
        return False
    return is_same_origin(request)


def is_trusted_client(request: Request) -> bool:
    """受保护接口的调用方：本机请求，或已通过访问验证且同源的远端。"""
    if is_internal_request(request):
        return True
    return bool(getattr(request.state, "access_granted", False)) and is_same_origin(request)


def auth_credentials(config: Any) -> tuple[str, str]:
    """账号密码优先读环境变量，未设置时回退 config.yaml 的 api.auth。"""
    username = (os.environ.get(AUTH_USERNAME_ENV, "").strip()
                or str(config.get("api.auth.username", "") or "").strip())
    password = (os.environ.get(AUTH_PASSWORD_ENV, "")
                or str(config.get("api.auth.password", "") or ""))
    return username, password


def _parse_basic(header: str | None) -> tuple[str, str] | None:
    """解析 Authorization: Basic，格式或编码非法时返回 None。"""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "basic" or not token.strip():
        return None
    try:
        decoded = base64.b64decode(token.strip(), validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    username, separator, password = decoded.partition(":")
    return (username, password) if separator else None


class BasicAuthMiddleware:
    """在 ASGI 层覆盖 API、静态页面与 SSE 流，未认证请求不进入路由。"""

    def __init__(self, app: ASGIApp, config_factory: Callable[[], Any]) -> None:
        self.app = app
        self.config_factory = config_factory

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        if request.method == "OPTIONS":
            await self.app(scope, receive, send)
            return
        if not is_same_origin(request):
            await JSONResponse({"detail": "跨站请求被拒绝"}, status_code=403)(scope, receive, send)
            return
        if is_internal_request(request):
            request.state.access_granted = True
            await self.app(scope, receive, send)
            return
        config = self.config_factory()
        if not bool(config.get("api.auth.enabled", True)):
            request.state.access_granted = True
            await self.app(scope, receive, send)
            return
        username, password = auth_credentials(config)
        if not password:
            await JSONResponse(
                {
                    "detail": (
                        "服务已对外监听但未配置访问密码：请在 "
                        f"{config.config_dir / 'config.yaml'} 的 api.auth.password 设置，"
                        f"或设置环境变量 {AUTH_PASSWORD_ENV}"
                    )
                },
                status_code=503,
            )(scope, receive, send)
            return
        supplied = _parse_basic(request.headers.get("authorization"))
        if (supplied is not None
                and hmac.compare_digest(supplied[0].encode("utf-8"), username.encode("utf-8"))
                and hmac.compare_digest(supplied[1].encode("utf-8"), password.encode("utf-8"))):
            request.state.access_granted = True
            await self.app(scope, receive, send)
            return
        await JSONResponse(
            {"detail": "需要登录"},
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="stock-robot", charset="UTF-8"'},
        )(scope, receive, send)


def install_basic_auth(app: Any, config_factory: Callable[[], Any]) -> None:
    """注册访问认证中间件；config_factory 每次请求重新读取，改密码无需重启。"""
    app.add_middleware(BasicAuthMiddleware, config_factory=config_factory)
