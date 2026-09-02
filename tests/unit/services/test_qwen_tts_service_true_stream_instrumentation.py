"""Story 18.1 Task 1.1 — TRUE_STREAM producer-side ``progressive_chunk_emit_ms``
instrumentation tests.

Drives ``_generate_true_stream`` end-to-end (mocked talker + decode_fn)
exactly like ``test_qwen_tts_service_true_stream_callback.py``, but
asserts on the ``progressive_chunk_emit_ms`` metric stream instead of the
``AudioChunk`` callback stream. Each ``append_chunk`` post must emit
exactly one ``progressive_chunk_emit_ms`` record with ``session_id`` +
``chunk_index`` tags; the emit timestamp is monotonically non-decreasing
across the chunk sequence.

The metric is the producer-side half of the Task 1.4 CSV — paired with
the consumer-side ``progressive_chunk_playback_arrival_ms`` (covered by
``tests/unit/test_app_progressive_playback_instrumentation.py``) it
gives the inter-chunk-emit interval Task 1.4 compares against the
audio-duration to choose Option 1 / 2 / 3 (AC #3).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from myvoice.observability import metrics
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
# Fixtures (mirror test_qwen_tts_service_true_stream_callback.py)
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


@pytest.fixture
def metric_capture():
    captured: list = []
    unsub = metrics.add_listener(captured.append)
    yield captured
    unsub()


# --------------------------------------------------------------------------- #
# Helpers (mirror test_qwen_tts_service_true_stream_callback.py)
# --------------------------------------------------------------------------- #


def _build_true_stream_service(registry, coordinator):
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
    def builder(model, request, streamer):
        def run():
            for tok in range(token_count):
                streamer.put([tok])
                time.sleep(0.001)
            streamer.end()

        return run

    return builder


def _fake_decode_fn(model, *_geometry, **_kwargs):
    return lambda chunk: np.array([t * 0.01 for t in chunk], dtype=np.float32)


def _drive_dispatch(qapp, service, request):
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


class TestTrueStreamProgressiveChunkEmitMetric:
    """Story 18.1 Task 1.1 — every TRUE_STREAM ``append_chunk`` post emits
    exactly one ``progressive_chunk_emit_ms`` metric with the documented
    tag schema (``session_id``, ``chunk_index``).
    """

    def test_emit_metric_fires_per_append_chunk(
        self, qapp, registry, coordinator, monkeypatch, metric_capture
    ):
        service, _ = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(
            service, "_build_true_stream_talker", _fake_talker_factory(100)
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", _fake_decode_fn
        )

        captured_chunks: list[AudioChunk] = []
        service.set_audio_chunk_ready_callback(captured_chunks.append)

        response = _drive_dispatch(qapp, service, _make_request())
        assert response.success is True

        emit_records = [
            r for r in metric_capture if r.name == "progressive_chunk_emit_ms"
        ]
        per_chunk_callbacks = [c for c in captured_chunks if not c.is_final]
        # Exactly one emit metric per data chunk callback; the synthetic
        # terminal chunk (finalize) does NOT emit progressive_chunk_emit_ms.
        assert len(emit_records) == len(per_chunk_callbacks), (
            f"Expected {len(per_chunk_callbacks)} progressive_chunk_emit_ms "
            f"records (one per append_chunk); got {len(emit_records)}"
        )
        assert len(emit_records) >= 1

    def test_emit_metric_tags_have_session_id_and_monotonic_chunk_index(
        self, qapp, registry, coordinator, monkeypatch, metric_capture
    ):
        service, _ = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(
            service, "_build_true_stream_talker", _fake_talker_factory(60)
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", _fake_decode_fn
        )

        service.set_audio_chunk_ready_callback(lambda c: None)
        response = _drive_dispatch(qapp, service, _make_request())
        assert response.success is True

        emit_records = [
            r for r in metric_capture if r.name == "progressive_chunk_emit_ms"
        ]
        assert len(emit_records) >= 1

        # session_id is a non-empty string (registry-issued) for every record
        # AND the same string across all records (single generation).
        session_ids = {r.session_id for r in emit_records}
        assert len(session_ids) == 1, (
            "All emit records from one generation must share the same "
            f"session_id; got {session_ids}"
        )
        sid = session_ids.pop()
        assert isinstance(sid, str) and sid, (
            f"session_id must be a non-empty string; got {sid!r}"
        )

        # chunk_index is monotonically increasing 0..N-1.
        chunk_indices = [r.tags["chunk_index"] for r in emit_records]
        assert chunk_indices == list(range(len(emit_records))), (
            f"chunk_index sequence violated: got {chunk_indices}"
        )

    def test_emit_metric_value_is_monotonically_non_decreasing(
        self, qapp, registry, coordinator, monkeypatch, metric_capture
    ):
        """Wall-clock timestamps from a single thread must be monotonically
        non-decreasing across the chunk sequence. Sleep(0.001) between
        token feeds in ``_fake_talker_factory`` ensures multi-millisecond
        spacing so the assertion is meaningful even on fast hosts.
        """
        service, _ = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(
            service, "_build_true_stream_talker", _fake_talker_factory(80)
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", _fake_decode_fn
        )

        service.set_audio_chunk_ready_callback(lambda c: None)
        response = _drive_dispatch(qapp, service, _make_request())
        assert response.success is True

        emit_records = [
            r for r in metric_capture if r.name == "progressive_chunk_emit_ms"
        ]
        values = [r.value for r in emit_records]
        assert len(values) >= 2, (
            f"Need at least 2 chunks for monotonicity check; got {len(values)}"
        )
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], (
                f"emit_ms regressed at chunk_index={i}: "
                f"values[{i-1}]={values[i-1]}, values[{i}]={values[i]}"
            )

class TestTtfaGenerationStartMetric:
    """Story 20.1 Task 2.1 — ``ttfa_generation_start_ms`` publishes the t0
    that ``first_chunk_latency_ms`` is already measured relative to, in
    absolute wall-clock ms so the CSV segments are a subtraction rather
    than a clock reconciliation.

    Retained as permanent observational surface by architect decision
    2026-08-31 (AC #2b Phase 3 depends on it), hence this test.
    """

    def test_generation_start_metric_fires_once_per_dispatch(
        self, qapp, registry, coordinator, monkeypatch, metric_capture
    ):
        service, _ = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(
            service, "_build_true_stream_talker", _fake_talker_factory(100)
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", _fake_decode_fn
        )

        request = _make_request()
        before_ms = time.time() * 1000.0
        response = _drive_dispatch(qapp, service, request)
        after_ms = time.time() * 1000.0

        assert response.success is True

        starts = [
            r for r in metric_capture if r.name == "ttfa_generation_start_ms"
        ]
        assert len(starts) == 1, (
            "ttfa_generation_start_ms must fire exactly once per TRUE_STREAM "
            f"dispatch; got {len(starts)}"
        )
        rec = starts[0]
        # Absolute wall-clock ms bracketed by the test's own clock reads.
        assert isinstance(rec.value, float)
        assert before_ms <= rec.value <= after_ms, (
            "ttfa_generation_start_ms must be absolute wall-clock ms inside "
            f"the dispatch window [{before_ms}, {after_ms}]; got {rec.value}"
        )
        assert rec.session_id is not None
        assert rec.tags["text_length"] == len(request.text)

        # It is the SAME t0 first_chunk_latency_ms is relative to: the
        # first producer emit minus this value must equal the reported
        # first-chunk latency, within the 0.5 ms clock step on this host.
        emits = [
            r for r in metric_capture if r.name == "progressive_chunk_emit_ms"
        ]
        assert emits, "expected at least one producer emit"
        derived_ms = emits[0].value - rec.value
        assert response.first_chunk_latency is not None
        assert abs(derived_ms - response.first_chunk_latency * 1000.0) < 5.0, (
            "ttfa_generation_start_ms must share t0 with "
            f"first_chunk_latency_ms; derived={derived_ms}ms vs "
            f"reported={response.first_chunk_latency * 1000.0}ms"
        )
