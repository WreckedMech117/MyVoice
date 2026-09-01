"""Story 20.2 AC #2 / AC #3 — compile-priming audio suppression.

Story 18.4 shipped a compile-priming generation that dispatches a **real**
TRUE_STREAM utterance through ``generate_custom_voice``. Its audio safety
rested on two claims in its own docstring — that priming runs before
``set_audio_chunk_ready_callback`` wires a consumer, and that "the generation
is short enough that any audible artifact is bounded". The first is a startup
ordering race (already lost once in production — see the ``tts_compile="off"``
gate's comment, which records audible spurious "Hello world." reaching a
user's speakers); the second is not a safety property at all.

Story 20.2 makes warm-cache launches prime too, i.e. on **every** launch
rather than only the first-ever, so the race would become recurring. This
suite pins the positive mechanism that replaces both claims:

  * ``QwenTTSRequest.suppress_audio_output`` — a **per-request** flag.
  * ``QwenTTSService._audio_chunk_sink(request)`` — the single sanctioned
    resolver every chunk-emit site uses; returns ``None`` for a suppressed
    request.

Rows:

  1. ``test_priming_reaches_no_consumer_wired_before_it`` (AC #2) — the
     consumer is wired **before** priming runs; priming produces zero
     consumer callbacks while the producer-side ``progressive_chunk_emit_ms``
     metric proves chunks were genuinely produced (non-vacuity).
  2. ``test_cold_path_warmup_reaches_no_consumer`` (AC #2, "the existing
     cold-path priming is brought under the same mechanism") — the full
     ``warmup_compile_async`` cold path with a consumer wired first.
  3. ``test_user_generation_during_priming_still_reaches_consumer`` (AC #3) —
     a user generation dispatched while priming is in flight receives all of
     its chunks; suppression is scoped to the priming generation, never to a
     window of time or a service-wide flag.
  4. ``test_unsuppressed_request_still_emits`` — control: the same rig with
     ``suppress_audio_output=False`` emits normally (guards against the
     suppression being accidentally universal).
  5. ``TestSuppressionMechanismIsHardToBypass`` — source invariant: no
     chunk-emit site may read ``_audio_chunk_ready_callback`` directly; the
     only sanctioned reads are the declaration, the setter, and
     ``_audio_chunk_sink``. This is what makes the mechanism hard for a
     future caller to get wrong (AC #2's "pick the one that is hardest to get
     wrong" latitude).
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Mirror the Story 17.3 TRUE_STREAM suite's gate so torch-less / Qt-less envs
# skip cleanly.
pytest.importorskip("PyQt6")

from myvoice.observability import metrics
from myvoice.services.audio_coordinator import AudioCoordinator
from myvoice.services.monitor_audio_service import MonitorAudioService
from myvoice.services.sessions import SessionRegistry
from myvoice.services.virtual_microphone_service import VirtualMicrophoneService
from myvoice.services.qwen_tts_service import (
    AudioChunk,
    QwenModelType,
    QwenTTSRequest,
    QwenTTSService,
)


# --------------------------------------------------------------------------- #
# Fixtures (mirror tests/unit/services/test_qwen_tts_service_true_stream_callback.py)
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
    coord.play_dual_stream = AsyncMock(return_value=MagicMock())
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
def emit_records():
    """Producer-side ``progressive_chunk_emit_ms`` records.

    These fire inside ``_wrapped_post`` *before* the sink is consulted, so a
    non-zero count proves the priming generation genuinely produced audio
    chunks — which is what makes "the consumer saw zero" a real assertion
    rather than a vacuous one.
    """
    captured: List[metrics.MetricRecord] = []

    def listener(record: metrics.MetricRecord) -> None:
        if record.name == "progressive_chunk_emit_ms":
            captured.append(record)

    unsub = metrics.add_listener(listener)
    try:
        yield captured
    finally:
        unsub()


# --------------------------------------------------------------------------- #
# Rig
# --------------------------------------------------------------------------- #


def _build_true_stream_service(registry, coordinator):
    """A ``QwenTTSService`` wired for TRUE_STREAM dispatch with no real model."""
    from myvoice.models.app_settings import AppSettings
    from myvoice.services.core.base_service import ServiceStatus

    settings = AppSettings(
        streaming_mode_override="true_stream", tts_compile="auto"
    )
    service = QwenTTSService(
        audio_coordinator=coordinator,
        device="cpu",
        dtype="float32",
        session_registry=registry,
        app_settings=settings,
    )
    service.status = ServiceStatus.RUNNING

    from concurrent.futures import ThreadPoolExecutor

    service._executor = ThreadPoolExecutor(max_workers=2)
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


def _fake_decode_fn(model):
    return lambda chunk: np.array([t * 0.01 for t in chunk], dtype=np.float32)


def _install_fakes(service, monkeypatch, *, tokens: int = 60, talker=None):
    monkeypatch.setattr(
        service,
        "_build_true_stream_talker",
        talker if talker is not None else _fake_talker_factory(tokens),
    )
    monkeypatch.setattr(service, "_build_true_stream_decode_fn", _fake_decode_fn)


def _run_with_qt_drain(qapp, coro_factory, *, settle_iterations: int = 50):
    """Run ``coro_factory()`` on a fresh loop while pumping the Qt event queue
    so the SessionRegistry's queued-connection posts land on the main thread."""

    async def runner():
        stop_evt = asyncio.Event()

        async def drainer():
            while not stop_evt.is_set():
                qapp.processEvents()
                await asyncio.sleep(0.005)

        drain_task = asyncio.create_task(drainer())
        try:
            return await coro_factory()
        finally:
            stop_evt.set()
            await drain_task

    result = asyncio.run(runner())
    for _ in range(settle_iterations):
        qapp.processEvents()
        time.sleep(0.005)
    return result


def _user_request(session_id: str = "user-session") -> QwenTTSRequest:
    return QwenTTSRequest(
        text="the user's actual utterance",
        language="Auto",
        model_type=QwenModelType.CUSTOM_VOICE,
        speaker="Ryan",
        streaming=True,
        session_id=session_id,
    )


# --------------------------------------------------------------------------- #
# AC #2 — priming reaches no consumer, by explicit mechanism
# --------------------------------------------------------------------------- #


def test_priming_reaches_no_consumer_wired_before_it(
    qapp, registry, coordinator, monkeypatch, emit_records
):
    """AC #2 — the consumer is wired BEFORE priming; it must receive zero
    chunks. The old docstring's "priming runs before the callback is wired"
    ordering claim is exactly the precondition this test violates."""
    service, _ = _build_true_stream_service(registry, coordinator)
    _install_fakes(service, monkeypatch)

    captured: List[AudioChunk] = []
    # Wired FIRST — the race the pre-20.2 code assumed could not happen.
    service.set_audio_chunk_ready_callback(captured.append)

    _run_with_qt_drain(qapp, lambda: service._run_compile_priming())

    assert captured == [], (
        "Compile priming must reach NO audio consumer, even when one is "
        f"already wired; got {len(captured)} chunk callback(s)"
    )
    # Non-vacuity: the producer really did emit chunks, they just went
    # nowhere. Without this, a priming that silently failed would pass.
    assert len(emit_records) >= 1, (
        "Expected the priming generation to actually produce audio chunks "
        "(progressive_chunk_emit_ms); zero would make the assertion above "
        "vacuous"
    )


def test_cold_path_warmup_reaches_no_consumer(
    qapp, registry, coordinator, monkeypatch, emit_records
):
    """AC #2 — "the existing cold-path priming is brought under the same
    mechanism". Drives the full ``warmup_compile_async`` cold path
    (``is_warm`` False → prime → ``mark_warm``) with a consumer wired first.
    """
    import torch

    service, mock_model = _build_true_stream_service(registry, coordinator)
    # Give the mock model the attributes ``compute_key`` reads.
    mock_model.model.name_or_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    mock_model.model.dtype = torch.bfloat16
    _install_fakes(service, monkeypatch)

    monkeypatch.delenv("MYVOICE_DISABLE_COMPILE_WARMUP", raising=False)
    monkeypatch.delenv("MYVOICE_DISABLE_WARM_COMPILE_PRIMING", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer", lambda: True
    )
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm", lambda key: False
    )
    mark_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.mark_warm", mark_warm_mock
    )

    captured: List[AudioChunk] = []
    service.set_audio_chunk_ready_callback(captured.append)

    _run_with_qt_drain(qapp, lambda: service.warmup_compile_async())

    assert captured == [], (
        "Cold-path compile warmup must reach NO audio consumer; got "
        f"{len(captured)} chunk callback(s)"
    )
    assert len(emit_records) >= 1, "cold-path priming produced no chunks at all"
    mark_warm_mock.assert_called_once()


def test_unsuppressed_request_still_emits(
    qapp, registry, coordinator, monkeypatch
):
    """Control — the same rig without suppression emits normally, so the
    zero-chunk assertions above are about suppression, not about the rig."""
    service, _ = _build_true_stream_service(registry, coordinator)
    _install_fakes(service, monkeypatch)

    captured: List[AudioChunk] = []
    service.set_audio_chunk_ready_callback(captured.append)

    response = _run_with_qt_drain(
        qapp, lambda: service._generate_true_stream(_user_request())
    )

    assert response.success is True, f"dispatch failed: {response.error_message}"
    assert len([c for c in captured if not c.is_final]) >= 1
    assert len([c for c in captured if c.is_final]) == 1


# --------------------------------------------------------------------------- #
# AC #3 — a user generation during priming is never suppressed
# --------------------------------------------------------------------------- #


def test_user_generation_during_priming_still_reaches_consumer(
    qapp, registry, coordinator, monkeypatch
):
    """AC #3 — priming is in flight when the user's generation is dispatched.

    The user's chunks must arrive. This is the row that fails for any
    suppression scoped by time or by a service-wide boolean: those would
    silence the user's first utterance, converting a latency fix into a
    "no audio" bug.
    """
    service, _ = _build_true_stream_service(registry, coordinator)

    priming_talker_started = threading.Event()
    release_priming = threading.Event()

    def talker_builder(model, request, streamer):
        is_priming = getattr(request, "suppress_audio_output", False)

        def run():
            if is_priming:
                priming_talker_started.set()
                # Hold the priming generation open until the test has
                # dispatched the user's generation behind it.
                release_priming.wait(timeout=10.0)
            for tok in range(40):
                streamer.put([tok])
                time.sleep(0.001)
            streamer.end()

        return run

    _install_fakes(service, monkeypatch, talker=talker_builder)

    captured: List[AudioChunk] = []
    service.set_audio_chunk_ready_callback(captured.append)

    async def scenario():
        priming_task = asyncio.create_task(service._run_compile_priming())

        # Wait until priming is genuinely in flight (its talker thread ran).
        deadline = time.monotonic() + 10.0
        while not priming_talker_started.is_set():
            assert time.monotonic() < deadline, "priming talker never started"
            await asyncio.sleep(0.01)

        user_task = asyncio.create_task(
            service._generate_true_stream(_user_request())
        )
        # Let the user dispatch reach the request semaphore behind priming.
        await asyncio.sleep(0.05)
        assert not user_task.done()

        release_priming.set()
        return await asyncio.gather(priming_task, user_task)

    _priming_result, user_response = _run_with_qt_drain(qapp, scenario)

    assert user_response.success is True, (
        f"user dispatch failed: {user_response.error_message}"
    )
    data_chunks = [c for c in captured if not c.is_final]
    final_chunks = [c for c in captured if c.is_final]
    assert len(data_chunks) >= 1, (
        "The user's generation was dispatched while priming was in flight and "
        "its audio never reached the consumer — the suppression mechanism is "
        "not scoped to the priming generation"
    )
    assert len(final_chunks) == 1
    # Every delivered chunk belongs to the USER's session, never priming's.
    assert {c.session_id for c in captured} == {"user-session"}, (
        "A chunk from a session other than the user's reached the consumer: "
        f"{sorted({str(c.session_id) for c in captured})}"
    )


# --------------------------------------------------------------------------- #
# Source invariant — the mechanism is hard to bypass
# --------------------------------------------------------------------------- #


class TestSuppressionMechanismIsHardToBypass:
    """AC #2's latitude asks for the mechanism "hardest to get wrong from a
    future caller's perspective". A per-request flag is only as strong as the
    discipline that every emit site resolves its sink through
    ``_audio_chunk_sink``. This test makes that discipline mechanical.
    """

    SERVICE_PATH = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "myvoice"
        / "services"
        / "qwen_tts_service.py"
    )

    def test_no_emit_site_reads_the_raw_callback(self):
        source = self.SERVICE_PATH.read_text(encoding="utf-8")
        lines = source.splitlines()

        offenders = []
        for lineno, line in enumerate(lines, start=1):
            code = line.split("#", 1)[0]
            if "self._audio_chunk_ready_callback" not in code:
                continue
            stripped = code.strip()
            # Three sanctioned occurrences only:
            #   1. the attribute declaration in __init__
            #   2. the assignment inside set_audio_chunk_ready_callback
            #   3. the single read inside _audio_chunk_sink
            sanctioned = (
                stripped.startswith("self._audio_chunk_ready_callback:")
                or stripped == "self._audio_chunk_ready_callback = callback"
                or stripped == "return self._audio_chunk_ready_callback"
            )
            if not sanctioned:
                offenders.append((lineno, stripped))

        assert offenders == [], (
            "Every audio-chunk emit site must resolve its consumer through "
            "QwenTTSService._audio_chunk_sink(request) so a compile-priming "
            "request (suppress_audio_output=True) can never reach the user's "
            "speakers (Story 20.2 AC #2). Direct reads found at: "
            + "; ".join(f"line {n}: {t}" for n, t in offenders)
        )

    def test_audio_chunk_sink_returns_none_for_suppressed_request(
        self, registry, coordinator
    ):
        service, _ = _build_true_stream_service(registry, coordinator)
        sentinel = MagicMock()
        service.set_audio_chunk_ready_callback(sentinel)

        normal = _user_request()
        suppressed = QwenTTSRequest(
            text="Hello world.", suppress_audio_output=True
        )

        assert service._audio_chunk_sink(normal) is sentinel
        assert service._audio_chunk_sink(suppressed) is None

    def test_priming_request_carries_the_flag(self, registry, coordinator, monkeypatch):
        """The priming dispatch must actually set ``suppress_audio_output``."""
        service, _ = _build_true_stream_service(registry, coordinator)
        seen: List[QwenTTSRequest] = []

        async def _capture(request, mode):
            seen.append(request)
            return MagicMock(success=True)

        monkeypatch.setattr(service, "_dispatch_by_streaming_mode", _capture)
        asyncio.run(service._run_compile_priming())

        assert len(seen) == 1
        assert seen[0].suppress_audio_output is True
        assert seen[0].text == QwenTTSService._COMPILE_PRIMING_TEXT

    def test_suppress_flag_defaults_to_false(self):
        """No ordinary request is ever born suppressed."""
        assert QwenTTSRequest(text="hi").suppress_audio_output is False


def test_docstring_no_longer_claims_ordering_safety():
    """Task 1.3 — the ``_run_compile_priming`` docstring must not assert the
    guarantee the code does not provide."""
    source = TestSuppressionMechanismIsHardToBypass.SERVICE_PATH.read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"async def _run_compile_priming\(self\).*?\n\s{8}\"\"\"(.*?)\"\"\"",
        source,
        re.DOTALL,
    )
    assert match is not None, "could not locate _run_compile_priming docstring"
    doc = match.group(1)
    assert "No audio output reaches the user (the priming runs before" not in doc
    assert "suppress_audio_output" in doc, (
        "the docstring must describe the positive mechanism it now relies on"
    )
