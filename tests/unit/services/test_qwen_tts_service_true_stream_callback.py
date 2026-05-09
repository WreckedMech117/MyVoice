"""Story 17.3 — TRUE_STREAM ``_audio_chunk_ready_callback`` emission tests.

Drives ``_generate_true_stream`` end-to-end with a mocked talker + decode_fn
so the chunk-emit point's callback wiring is observable under deterministic
token feeds. The callback contract this story adds to TRUE_STREAM mirrors
the SENTENCE_STREAM precedent at ``qwen_tts_service.py:3071-3082``.

Covers AC #1 (Task 1.5):
  - ``test_true_stream_emits_chunk_callback_per_append`` — one
    ``AudioChunk(is_final=False)`` per ``append_chunk`` post.
  - ``test_true_stream_emits_final_chunk_on_finalize`` — exactly one
    terminal ``AudioChunk(is_final=True)`` after the last data chunk.
  - ``test_true_stream_callback_exception_does_not_break_dispatch`` — a
    callback raising ``RuntimeError`` is swallowed; dispatch still
    produces the assembled ``QwenTTSResponse``.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Mirror the integration-suite gate so torch-less / Qt-less envs skip cleanly.
pytest.importorskip("PyQt6")

from myvoice.services.audio_coordinator import AudioCoordinator
from myvoice.services.monitor_audio_service import MonitorAudioService
from myvoice.services.virtual_microphone_service import VirtualMicrophoneService
from myvoice.services.sessions import SessionRegistry
from myvoice.services.qwen_tts_service import (
    AudioChunk,
    QwenModelType,
    QwenTTSRequest,
    QwenTTSService,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_monitor():
    monitor = MagicMock(spec=MonitorAudioService)
    monitor.stop_all_playback = AsyncMock(return_value=0)
    monitor.play_monitor_audio = AsyncMock()
    return monitor


@pytest.fixture
def mock_virtual():
    virtual = MagicMock(spec=VirtualMicrophoneService)
    virtual.stop_all_virtual_microphone_playback = AsyncMock(return_value=0)
    virtual.play_virtual_microphone = AsyncMock()
    return virtual


@pytest.fixture
def coordinator(mock_monitor, mock_virtual) -> AudioCoordinator:
    coord = AudioCoordinator()
    coord._is_initialized = True
    coord.monitor_service = mock_monitor
    coord.virtual_service = mock_virtual
    return coord


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def registry(qapp) -> SessionRegistry:
    return SessionRegistry()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_true_stream_service(registry, coordinator):
    """Construct a ``QwenTTSService`` primed for TRUE_STREAM end-to-end tests
    without loading any real model. Mirrors the integration smoke-test rig
    at ``tests/integration/test_streaming_tts_smoke.py:478`` but is local
    so this unit test stays self-contained.
    """
    from myvoice.models.app_settings import AppSettings
    from myvoice.services.core.base_service import ServiceStatus

    settings = AppSettings(streaming_mode_override="true_stream")
    service = QwenTTSService(
        audio_coordinator=coordinator,
        device="cpu",
        dtype="float32",
        session_registry=registry,
        app_settings=settings,
    )
    service.status = ServiceStatus.RUNNING

    from concurrent.futures import ThreadPoolExecutor

    service._executor = ThreadPoolExecutor(max_workers=1)
    service._request_semaphore = asyncio.Semaphore(1)

    mock_model = MagicMock()
    service._model_registry.ensure_model_loaded = AsyncMock(
        return_value=(True, None)
    )
    service._model_registry.get_loaded_model = MagicMock(return_value=mock_model)
    service._model_registry.device = "cpu"
    return service, mock_model


def _fake_talker_factory(token_count: int):
    """Return a ``_build_true_stream_talker``-compatible builder that feeds
    ``token_count`` single-codebook tokens through ``streamer.put`` then
    calls ``streamer.end()``. Single-codebook is fine for these tests
    because the patched decode_fn handles 1-D chunks (matches the
    ``_build_true_stream_decode_fn`` defensive 1-D path).
    """

    def builder(model, request, streamer):
        def run():
            for tok in range(token_count):
                streamer.put([tok])
                time.sleep(0.001)
            streamer.end()

        return run

    return builder


def _fake_decode_fn(model):
    """Deterministic decode: token id → ``token_id * 0.01`` PCM sample.
    Matches the integration-suite fixture so chunk → audio mapping is
    predictable across both suites.
    """
    return lambda chunk: np.array([t * 0.01 for t in chunk], dtype=np.float32)


def _drive_dispatch(qapp, service, request):
    """Run ``_generate_true_stream`` with a Qt-event drainer so registry
    queued-connection posts reach the main thread. Returns the
    ``QwenTTSResponse``.
    """

    async def runner():
        async def drainer(stop_evt):
            while not stop_evt.is_set():
                qapp.processEvents()
                await asyncio.sleep(0.005)

        stop_evt = asyncio.Event()
        drain_task = asyncio.create_task(drainer(stop_evt))
        try:
            return await service._generate_true_stream(request)
        finally:
            stop_evt.set()
            await drain_task

    response = asyncio.run(runner())
    # Drain anything queued post-finish so the registry settles before
    # the test asserts on session state or hooks.
    for _ in range(50):
        qapp.processEvents()
        time.sleep(0.005)
    return response


def _make_request() -> QwenTTSRequest:
    return QwenTTSRequest(
        text="hello world",
        language="Auto",
        model_type=QwenModelType.CUSTOM_VOICE,
        speaker="Ryan",
        streaming=True,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestTrueStreamChunkCallbackEmission:
    """Story 17.3 AC #1 — TRUE_STREAM emits ``AudioChunk`` callbacks per
    ``append_chunk`` mutation, plus one synthetic terminal chunk on
    ``finalize``. The callback is additive to the existing accumulator
    behavior; producer-side errors must never propagate from the
    consumer's exception.
    """

    def test_true_stream_emits_chunk_callback_per_append(
        self, qapp, registry, coordinator, monkeypatch
    ):
        service, _ = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(
            service, "_build_true_stream_talker", _fake_talker_factory(100)
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", _fake_decode_fn
        )

        captured: list[AudioChunk] = []
        service.set_audio_chunk_ready_callback(captured.append)

        response = _drive_dispatch(qapp, service, _make_request())

        assert response.success is True, (
            f"dispatch failed: {response.error_message}"
        )
        per_chunk = [c for c in captured if not c.is_final]
        assert len(per_chunk) == response.chunks_generated, (
            "Expected one is_final=False AudioChunk per accumulated_chunks "
            f"append; got {len(per_chunk)} callbacks vs "
            f"{response.chunks_generated} chunks_generated"
        )
        assert len(per_chunk) >= 1, (
            "Expected at least one chunk; the talker fed 100 tokens"
        )
        for idx, chunk in enumerate(per_chunk):
            assert isinstance(chunk, AudioChunk)
            assert chunk.chunk_index == idx, (
                f"chunk_index sequence violated at position {idx}: "
                f"got {chunk.chunk_index}"
            )
            assert chunk.sample_rate == 24000
            assert chunk.is_final is False
            assert chunk.audio_data.size > 0

    def test_true_stream_emits_final_chunk_on_finalize(
        self, qapp, registry, coordinator, monkeypatch
    ):
        service, _ = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(
            service, "_build_true_stream_talker", _fake_talker_factory(50)
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", _fake_decode_fn
        )

        captured: list[AudioChunk] = []
        service.set_audio_chunk_ready_callback(captured.append)

        response = _drive_dispatch(qapp, service, _make_request())

        assert response.success is True
        finals = [c for c in captured if c.is_final]
        assert len(finals) == 1, (
            f"Expected exactly one terminal AudioChunk; got {len(finals)}"
        )
        terminal = finals[0]
        # Synthetic terminal chunk is zero-length float32 — the consumer
        # must skip play_audio_chunk for it (Task 2.2).
        assert terminal.audio_data.size == 0
        assert terminal.audio_data.dtype == np.float32
        assert terminal.sample_rate == 24000
        # Terminal callback fires AFTER the last data chunk — finalize
        # is posted by the worker only after END_OF_STREAM is drained.
        assert captured[-1] is terminal, (
            "terminal AudioChunk must be the last callback emitted"
        )
        assert all(not c.is_final for c in captured[:-1])

    def test_true_stream_callback_exception_does_not_break_dispatch(
        self, qapp, registry, coordinator, monkeypatch
    ):
        service, _ = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(
            service, "_build_true_stream_talker", _fake_talker_factory(50)
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", _fake_decode_fn
        )

        invocations: list[int] = []

        def buggy_callback(chunk: AudioChunk) -> None:
            invocations.append(1)
            raise RuntimeError("simulated consumer failure")

        service.set_audio_chunk_ready_callback(buggy_callback)

        response = _drive_dispatch(qapp, service, _make_request())

        # Producer is robust: dispatch still produces the assembled
        # QwenTTSResponse despite the consumer raising on every chunk.
        assert response.success is True
        assert response.audio_data is not None
        assert response.audio_data.size > 0
        # Exception path was actually exercised (proves we tested the
        # right code branch — not just that the dispatch happens to work).
        assert len(invocations) >= 1
