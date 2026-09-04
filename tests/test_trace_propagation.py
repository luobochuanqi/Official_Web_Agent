"""OBS-02 trace 透传单测:只测出站 header,不碰内部 contextvar 状态。

CI 无 Langfuse 配置:span 侧行为用 monkeypatch 注入,零外部依赖。
"""

import httpx
import respx

from official_agent.config import Settings
from official_agent.observability import (
    reset_turn_trace_id,
    set_turn_trace_id,
)
from official_agent.tools.client import BackendClient

BASE = "http://backend.test"
LOGIN = f"{BASE}/api/auth/login"
CYCLES = f"{BASE}/api/cycles/open"
SPAN_ID_ZERO = "0" * 16


def make_client() -> BackendClient:
    settings = Settings(
        _env_file=None,
        backend_base_url=BASE,
        backend_service_username="svc-agent",
        backend_service_password="secret",
    )
    return BackendClient(http=httpx.AsyncClient(base_url=BASE), settings=settings)


def login_ok() -> httpx.Response:
    return httpx.Response(
        201, json={"code": 200, "message": "ok", "data": {"token": "tok-1", "user_id": 1}}
    )


def ok() -> httpx.Response:
    return httpx.Response(200, json={"code": 200, "message": "ok", "data": {}})


def trace_ids_of(routes: list[respx.Route]) -> list[str]:
    """各路由最后一个出站请求的 traceparent 的 trace-id 段(第 2 段)。"""
    out = []
    for r in routes:
        tp = r.calls.last.request.headers["traceparent"]
        assert tp.startswith("00-") and tp.endswith(f"-{SPAN_ID_ZERO}-01"), tp
        out.append(tp.split("-")[1])
    return out


@respx.mock
async def test_span_trace_id_injected_on_both_exits(monkeypatch) -> None:
    """主通道(_send)与用户令牌通道(get_as_user)都带 span 的 32-hex,登录不带。"""
    import official_agent.observability as obs

    monkeypatch.setattr(obs, "_active_span_trace_id", lambda: "a" * 32)
    login_route = respx.post(LOGIN)
    login_route.side_effect = login_ok()
    api_route = respx.get(CYCLES)
    api_route.side_effect = ok()
    user_route = respx.get(f"{BASE}/api/me/summary")
    user_route.side_effect = ok()
    client = make_client()

    await client.get("/api/cycles/open")
    await client.get_as_user("/api/me/summary", user_token="user-tok")

    assert trace_ids_of([api_route, user_route]) == ["a" * 32, "a" * 32]
    assert "traceparent" not in login_route.calls.last.request.headers
    await client.aclose()


@respx.mock
async def test_turn_id_fallback_without_span(monkeypatch) -> None:
    """无 span 时 trace-id 段 = 轮级 id 的 W3C 归一值(非 32-hex → sha256,确定性)。"""
    import hashlib

    import official_agent.observability as obs

    monkeypatch.setattr(obs, "_active_span_trace_id", lambda: None)
    respx.post(LOGIN).side_effect = login_ok()
    api_route = respx.get(CYCLES)
    api_route.side_effect = ok()
    user_route = respx.get(f"{BASE}/api/me/summary")
    user_route.side_effect = ok()
    client = make_client()

    token = set_turn_trace_id("cli:u123:a1b2c3d4")
    try:
        await client.get("/api/cycles/open")
        await client.get_as_user("/api/me/summary", user_token="user-tok")
    finally:
        reset_turn_trace_id(token)

    want = hashlib.sha256(b"cli:u123:a1b2c3d4").hexdigest()
    assert trace_ids_of([api_route, user_route]) == [want] * 2
    await client.aclose()
