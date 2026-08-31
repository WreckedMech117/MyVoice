"""Auth + Host-guard unit tests (tech-spec Task 12 / F9)."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from myvoice.services.api_server.security import (
    generate_api_key,
    verify_auth,
    verify_host,
)


def _fake_request(headers: dict, api_key: str = ""):
    settings = SimpleNamespace(http_api_key=api_key)
    state = SimpleNamespace(settings_provider=lambda: settings)
    app = SimpleNamespace(state=state)
    return SimpleNamespace(headers={k.lower(): v for k, v in headers.items()}, app=app)


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.1:7778", "localhost:7778"])
async def test_host_guard_allows_loopback(host):
    await verify_host(_fake_request({"host": host}))  # no raise


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["[::1]", "[::1]:7778"])
async def test_host_guard_allows_ipv6_loopback(host):
    await verify_host(_fake_request({"host": host}))  # no raise


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["evil.com", "192.168.1.5", "example.org:7778", ""])
async def test_host_guard_rejects_non_loopback(host):
    with pytest.raises(HTTPException) as exc:
        await verify_host(_fake_request({"host": host}))
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_auth_keyless_passes_without_header():
    await verify_auth(_fake_request({}, api_key=""))  # no raise


@pytest.mark.asyncio
async def test_auth_missing_header_rejected_when_key_set():
    with pytest.raises(HTTPException) as exc:
        await verify_auth(_fake_request({}, api_key="secret"))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_wrong_key_rejected():
    req = _fake_request({"authorization": "Bearer wrong"}, api_key="secret")
    with pytest.raises(HTTPException) as exc:
        await verify_auth(req)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_correct_key_passes():
    req = _fake_request({"authorization": "Bearer secret"}, api_key="secret")
    await verify_auth(req)  # no raise


@pytest.mark.asyncio
async def test_auth_malformed_scheme_rejected():
    req = _fake_request({"authorization": "Basic secret"}, api_key="secret")
    with pytest.raises(HTTPException) as exc:
        await verify_auth(req)
    assert exc.value.status_code == 401


def test_generate_api_key_is_high_entropy():
    k1, k2 = generate_api_key(), generate_api_key()
    assert k1 != k2
    assert len(k1) >= 32
