"""ApiServerController lifecycle tests (tech-spec G7/F9/F2).

Covers stop()-cancels-in-flight-streams and the keyless start warning without
binding a real socket where possible.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from myvoice.services.api_server.server import ApiServerController


def _controller(api_key=""):
    return ApiServerController(
        tts_service=SimpleNamespace(),
        voice_manager=SimpleNamespace(),
        app_ref=SimpleNamespace(),
        settings_provider=lambda: SimpleNamespace(http_api_key=api_key),
    )


@pytest.mark.asyncio
async def test_stop_cancels_registered_stream_tasks():
    ctrl = _controller()

    async def _never():
        await asyncio.sleep(60)

    task = asyncio.ensure_future(_never())
    ctrl._active_stream_tasks.add(task)

    # No server was started; stop() must still cancel registered gen_tasks (G7)
    # and clear the registry, completing well within the bounded window.
    await asyncio.wait_for(ctrl.stop(), timeout=3.0)

    assert task.cancelled() or task.done()
    assert ctrl._active_stream_tasks == set()
    assert ctrl.is_running is False


@pytest.mark.asyncio
async def test_start_warns_when_keyless(caplog):
    """A keyless enable (e.g. hand-edited config) logs a security warning (F2)."""
    ctrl = _controller(api_key="")

    # Force the readiness wait to fail fast so we don't bind a real port: stub
    # build_app to raise, then assert the warning was emitted before the raise.
    import myvoice.services.api_server.server as server_mod

    def _boom(*a, **k):
        raise RuntimeError("stop here")

    original = server_mod.build_app
    server_mod.build_app = _boom
    try:
        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError):
                await ctrl.start(host="127.0.0.1", port=7799)
    finally:
        server_mod.build_app = original

    assert any("WITHOUT an API key" in r.message for r in caplog.records)
