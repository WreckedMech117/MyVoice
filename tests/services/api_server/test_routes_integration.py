"""Integration tests for the routes against a stubbed TTS service (no GPU).

Drives ``build_app(...)`` with an in-memory ASGI transport (httpx). The stub
``generate_custom_voice`` returns a known response for the buffered path and
publishes to the app's StreamHub for the streaming path, so the full route +
encoder + StreamHub join is exercised without a model.
"""

import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from myvoice.services.api_server.app_factory import build_app
from myvoice.services.api_server.audio_encode import float32_to_int16_bytes
from myvoice.services.api_server.stream_hub import StreamHub


# --- fakes ----------------------------------------------------------------- #


class _FakeProfile:
    voice_type = "bundled"
    description = None


class _FakeVoiceManager:
    def __init__(self, names):
        self._profiles = {n: _FakeProfile() for n in names}

    def get_valid_profiles(self):
        return self._profiles


class _FakeApp:
    def __init__(self):
        self._stream_hub = StreamHub()
        self._api_origin_sessions = set()


class _FakeController:
    def __init__(self):
        self._active_stream_tasks = set()


class _FakeTTS:
    """Stub engine. ``stream_chunks`` (list of (int16_bytes, is_final)) drives
    the streaming path; when None, streaming produces no chunks (BATCH-like)."""

    def __init__(self, app_ref, audio, stream_chunks=None, success=True):
        self._app_ref = app_ref
        self._audio = audio
        self._stream_chunks = stream_chunks
        self._success = success
        self.calls = []

    async def generate_custom_voice(
        self,
        text,
        speaker="Ryan",
        language="Auto",
        instruct=None,
        emotion_preset=None,
        streaming=True,
        session_id=None,
    ):
        self.calls.append(
            {"text": text, "speaker": speaker, "streaming": streaming, "session_id": session_id}
        )
        if streaming and self._stream_chunks is not None:
            for audio_bytes, is_final in self._stream_chunks:
                self._app_ref._stream_hub.publish(session_id, audio_bytes, is_final)
                await asyncio.sleep(0)  # yield so the route's gen() can drain
        return SimpleNamespace(
            success=self._success,
            audio_data=self._audio if self._success else None,
            sample_rate=24000,
            error_message=None if self._success else "boom",
        )


def _build(audio=None, voices=("Ryan", "Vivian"), api_key="", stream_chunks=None, success=True):
    if audio is None:
        audio = (0.3 * np.sin(np.linspace(0, 6.28 * 110, 6000))).astype(np.float32)
    app_ref = _FakeApp()
    tts = _FakeTTS(app_ref, audio, stream_chunks=stream_chunks, success=success)
    controller = _FakeController()
    settings = SimpleNamespace(http_api_key=api_key)
    app = build_app(
        tts_service=tts,
        voice_manager=_FakeVoiceManager(voices),
        app_ref=app_ref,
        settings_provider=lambda: settings,
        controller=controller,
    )
    return app, tts, app_ref


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")


# --- buffered -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_buffered_mp3_default():
    app, *_ = _build()
    async with _client(app) as client:
        resp = await client.post("/v1/audio/speech", json={"input": "hello", "voice": "Ryan"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content[0] == 0xFF  # MPEG frame sync


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt,ctype", [("wav", "audio/wav"), ("pcm", "audio/L16; rate=24000")])
async def test_buffered_formats(fmt, ctype):
    app, *_ = _build()
    async with _client(app) as client:
        resp = await client.post(
            "/v1/audio/speech", json={"input": "hi", "voice": "Ryan", "response_format": fmt}
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == ctype
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_unknown_voice_400():
    app, tts, _ = _build()
    async with _client(app) as client:
        resp = await client.post("/v1/audio/speech", json={"input": "hi", "voice": "Nobody"})
    assert resp.status_code == 400
    assert tts.calls == []  # no generation occurred


@pytest.mark.asyncio
async def test_empty_input_422():
    app, *_ = _build()
    async with _client(app) as client:
        resp = await client.post("/v1/audio/speech", json={"input": "", "voice": "Ryan"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_speed_two_accepted_noop():
    app, *_ = _build()
    async with _client(app) as client:
        resp = await client.post(
            "/v1/audio/speech", json={"input": "hi", "voice": "Ryan", "speed": 2.0}
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_generation_failure_500():
    app, *_ = _build(success=False)
    async with _client(app) as client:
        resp = await client.post("/v1/audio/speech", json={"input": "hi", "voice": "Ryan"})
    assert resp.status_code == 500


# --- metadata -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_voices_list():
    app, *_ = _build(voices=("Ryan", "Vivian", "Serena"))
    async with _client(app) as client:
        resp = await client.get("/v1/voices")
    assert resp.status_code == 200
    names = {v["name"] for v in resp.json()["voices"]}
    assert names == {"Ryan", "Vivian", "Serena"}


@pytest.mark.asyncio
async def test_models_and_health():
    app, *_ = _build()
    async with _client(app) as client:
        models = await client.get("/v1/models")
        health = await client.get("/health")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "myvoice-1"
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}


# --- auth + host ----------------------------------------------------------- #


@pytest.mark.asyncio
async def test_auth_required_when_key_set():
    app, *_ = _build(api_key="secret")
    async with _client(app) as client:
        no_key = await client.post("/v1/audio/speech", json={"input": "hi", "voice": "Ryan"})
        good = await client.post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "Ryan"},
            headers={"Authorization": "Bearer secret"},
        )
    assert no_key.status_code == 401
    assert good.status_code == 200


@pytest.mark.asyncio
async def test_host_guard_rejects_non_loopback():
    app, *_ = _build()
    async with _client(app) as client:
        resp = await client.get("/health", headers={"Host": "evil.com"})
    assert resp.status_code == 400


# --- streaming ------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_streaming_pcm_happy_path():
    chunk0 = np.full(100, 1000, dtype=np.int16).tobytes()
    chunk1 = np.full(50, -1000, dtype=np.int16).tobytes()
    app, *_ = _build(stream_chunks=[(chunk0, False), (chunk1, True)])
    async with _client(app) as client:
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "Ryan", "response_format": "pcm", "stream": True},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/L16; rate=24000"
    # pcm passthrough -> body is the concatenated chunks.
    assert resp.content == chunk0 + chunk1


@pytest.mark.asyncio
async def test_streaming_wav_rejected_400():
    app, *_ = _build(stream_chunks=[])
    async with _client(app) as client:
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "Ryan", "response_format": "wav", "stream": True},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_streaming_zero_chunks_degrades_to_buffered():
    # stream_chunks=None -> engine publishes nothing (BATCH-like); the route
    # must gracefully degrade to the buffered result, not hang (AC11b).
    audio = (0.3 * np.sin(np.linspace(0, 6.28 * 110, 4000))).astype(np.float32)
    app, *_ = _build(audio=audio, stream_chunks=None)
    async with _client(app) as client:
        resp = await asyncio.wait_for(
            client.post(
                "/v1/audio/speech",
                json={"input": "hi", "voice": "Ryan", "response_format": "pcm", "stream": True},
            ),
            timeout=5.0,
        )
    assert resp.status_code == 200
    assert resp.content == float32_to_int16_bytes(audio)
