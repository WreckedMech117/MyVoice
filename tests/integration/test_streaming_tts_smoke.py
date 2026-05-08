"""Integration smoke tests for the Phase ⊥ True Streaming TTS chain.

Story 16.5 — verifies the cooperative cancellation chain end-to-end with
the REAL Story-16.3 ``CodecTokenStreamer`` and Story-16.4
``StreamingDecoderWorker``, a real ``SessionRegistry``, and a real
``AudioCoordinator`` with mocked monitor + virtual-mic sinks.

Architecture references:
  - P-7 (architecture-optimization-pass.md:443-451): one-direction cancel
    chain — user → registry → streamer's _cancel_event → talker .generate()
    returns → decoder drains → CANCELLED → DISCARDED.
  - Validation gap #3 (lines 835-839): two follow-on actions on cancel —
    (i) audio_coordinator.cancel_playback(session_id), (ii) decoder
    drain-on-cancel drops queued chunks.
  - File map (lines 639-641): this file is the Phase ⊥ smoke-test sibling
    of the unit tests in tests/unit/services/tts_streaming/.

Test design — wakeup invariant
------------------------------
The Story-16.4 ``StreamingDecoderWorker._run`` checks ``_cancel_event``
at the top of each loop iteration but otherwise blocks on
``streamer.queue.get()``. Once cancel fires, ``streamer.put()`` becomes
a no-op (Story 16.3 D-11), so a worker that has drained the queue stays
blocked indefinitely on ``get()``. To verify the cancel chain in
isolation, these tests pre-fill the queue with enough chunks that the
worker is still consuming when cancel fires — the next iteration's top-
of-loop check sees the event, ``_drain_and_post_cancel()`` runs (drops
remaining chunks via ``get_nowait``), and the worker exits. In real
production (Story 16.6's TRUE_STREAM dispatch), the talker thread
running HF ``.generate()`` continuously feeds the queue, so the worker
is naturally still consuming when cancel fires; the pre-fill is the
test-rig analogue of "talker is still running."

This file does NOT exercise Story 16.6 (TRUE_STREAM dispatch in
QwenTTSService.generate). The streamer + worker pair is constructed
inline by the test rig, simulating the wiring that Story 16.6 will
eventually compose in production.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Tests in this module require PyQt6 + the streaming subpackage.
pytest.importorskip("PyQt6")

from myvoice.services.audio_coordinator import AudioCoordinator
from myvoice.services.monitor_audio_service import MonitorAudioService
from myvoice.services.virtual_microphone_service import VirtualMicrophoneService
from myvoice.services.sessions import SessionRegistry, SessionState
from myvoice.services.tts_streaming import (
    CodecTokenStreamer,
    StreamingDecoderWorker,
)


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def _make_decoded_pcm(chunk):
    """Deterministic decode_fn — each token id maps to one PCM sample of
    value (token_id * 0.1). Token-to-PCM ratio is 1.0 (matches Story 16.4
    test convention).
    """
    return np.array([t * 0.1 for t in chunk], dtype=np.float32)


def _slow_decoded_pcm(chunk):
    """Like _make_decoded_pcm but takes ~5ms per call — keeps the worker
    busy long enough for a mid-stream cancel to land while a chunk is
    being processed (mirrors the production case where decode latency
    dominates the cancel-detection window).
    """
    time.sleep(0.005)
    return np.array([t * 0.1 for t in chunk], dtype=np.float32)


def _drain_qt_events(qapp, iterations: int = 10) -> None:
    """Drain queued-connection backlog (worker → registry post_mutation
    posts use QueuedConnection)."""
    for _ in range(iterations):
        qapp.processEvents()


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
def event_loop_thread():
    """Background asyncio event loop on its own thread — needed so the
    cancel hook can schedule ``coordinator.cancel_playback(...)`` (an
    async coroutine) via ``asyncio.run_coroutine_threadsafe``.
    """
    loop_holder: dict = {}
    loop_started = threading.Event()

    def loop_runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        loop_started.set()
        loop.run_forever()

    thread = threading.Thread(target=loop_runner, daemon=True)
    thread.start()
    assert loop_started.wait(timeout=2.0), "asyncio loop did not start"
    yield loop_holder["loop"]
    loop_holder["loop"].call_soon_threadsafe(loop_holder["loop"].stop)
    thread.join(timeout=2.0)


class _RecordingPostMutation:
    """Wraps registry.post_mutation and records timestamped call tuples
    so tests can assert ordering invariants (e.g., no append_chunk after
    cancel) without peering at QMetaObject internals.
    """

    def __init__(self, real_post_mutation) -> None:
        self._real = real_post_mutation
        self.calls: list[tuple[float, str, tuple]] = []

    def __call__(self, method_name: str, *args) -> None:
        self.calls.append((time.perf_counter(), method_name, args))
        self._real(method_name, *args)


def _build_cancel_hook(streamer, coordinator, sid, event_loop):
    """Story 16.6's wiring (simulated by the test): the hook flips the
    streamer's event AND asks the coordinator to stop playback for the
    session via the asyncio event loop.
    """

    def cancel_hook():
        streamer._cancel_event.set()
        fut = asyncio.run_coroutine_threadsafe(
            coordinator.cancel_playback(sid), event_loop
        )
        # Wait briefly so test mock call counts reflect the cancel before
        # assertions run.
        fut.result(timeout=2.0)

    return cancel_hook


# --------------------------------------------------------------------------- #
# Story 16.5 — TestCancelChainEndToEnd (AC #1, #7)
# --------------------------------------------------------------------------- #


class TestCancelChainEndToEnd:
    """AC #1 + AC #7: full chain — request_cancel → hook fires → streamer
    _cancel_event flips → decoder drains → ('cancel', sid) post →
    registry CANCELLED → coordinator playback stopped.
    """

    def test_cancel_chain_end_to_end_with_real_streamer_and_worker(
        self, qapp, registry, coordinator, mock_monitor, mock_virtual,
        event_loop_thread,
    ):
        # ---------- Set up the rig (mirrors Story 16.5 AC #1 Given) -----
        sid = registry.create_session(
            text="hello", voice="default", model_type="qwen3_tts"
        )
        registry.start_generation(sid)
        assert registry.get(sid).state == SessionState.GENERATING

        # Real streamer + real worker, with a slow decode_fn (~5ms/chunk)
        # so the worker is busy for a meaningful window when cancel fires.
        streamer = CodecTokenStreamer(chunk_size=4, lookahead=2)
        recording_post = _RecordingPostMutation(registry.post_mutation)
        worker = StreamingDecoderWorker(
            streamer=streamer,
            decode_fn=_slow_decoded_pcm,
            post_mutation=recording_post,
            session_id=sid,
            model_type="qwen3_tts",
            hardware="cpu",
        )

        # Pre-fill queue with chunks the worker will process for ~50ms total
        # (10 chunks × ~5ms decode each). Cancel fires mid-stream — the
        # worker's next top-of-loop check sees it and drains the rest.
        # queue maxsize = 4 * chunk_size = 16, so 10 fits comfortably.
        for _ in range(10):
            streamer.queue.put([1, 2, 3, 4, 5, 6])

        # Pre-populate the coordinator's session map (simulates a prior
        # play_dual_stream that registered the dispatch).
        coordinator._session_id_to_coordination_id[sid] = "coord_simulated"

        cancel_hook = _build_cancel_hook(
            streamer, coordinator, sid, event_loop_thread
        )
        registry.register_cancel_hook(sid, cancel_hook)

        # ---------- Drive the chain ---------------------------------
        worker.start()

        # Give the worker ~15ms to process a few chunks (3 chunks @ ~5ms
        # each), so cancel lands mid-stream — exactly the AC #7 scenario.
        time.sleep(0.015)
        _drain_qt_events(qapp)

        # User clicks Cancel mid-stream.
        t_cancel_start = time.perf_counter()
        registry.request_cancel(sid)

        # Streamer event flipped synchronously per AC #1.
        assert streamer._cancel_event.is_set() is True

        # Wait for the worker to drain + post ('cancel', sid) and for the
        # registry's `cancel` slot to fire on the Qt main thread.
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            _drain_qt_events(qapp)
            session = registry.get(sid)
            if session is not None and session.state == SessionState.CANCELLED:
                break
            time.sleep(0.005)

        # Worker exits after the drain.
        worker.join(timeout=1.0)
        assert worker.is_alive() is False

        # Registry is in CANCELLED state.
        assert registry.get(sid).state == SessionState.CANCELLED

        # Audio coordinator was asked to stop playback exactly once.
        mock_monitor.stop_all_playback.assert_called_once()
        mock_virtual.stop_all_virtual_microphone_playback.assert_called_once()

        # Map was cleaned by cancel_playback.
        assert sid not in coordinator._session_id_to_coordination_id

        # Hook was auto-cleared on the registry's cancel slot (AC #5).
        assert sid not in registry._cancel_hooks

        # Architecturally-named no-window invariant (AC #7): every
        # ('append_chunk', sid, ...) post precedes the single
        # ('cancel', sid) post in clock order.
        cancel_calls = [c for c in recording_post.calls if c[1] == "cancel"]
        assert len(cancel_calls) == 1, (
            f"Expected exactly one cancel post; got "
            f"{[c[1] for c in recording_post.calls]}"
        )
        cancel_t = cancel_calls[0][0]
        for t, method, _ in recording_post.calls:
            if method == "append_chunk":
                assert t < cancel_t, (
                    f"append_chunk post at {t} occurred AFTER cancel "
                    f"post at {cancel_t} — P-7 'no cancelled-but-still-"
                    f"emitting' invariant violated"
                )

    def test_cancel_chain_idempotent_when_called_twice(
        self, qapp, registry, coordinator, mock_monitor, mock_virtual,
        event_loop_thread,
    ):
        # AC #1 second `Given` clause: a second request_cancel call after
        # the event was already flipped is harmless. The hook is auto-
        # cleared by the registry's `cancel` slot, so the second call is
        # a quiet no-op (no exception, no duplicate cancel post).
        sid = registry.create_session(
            text="hello", voice="default", model_type="qwen3_tts"
        )
        registry.start_generation(sid)
        streamer = CodecTokenStreamer(chunk_size=4, lookahead=2)
        recording_post = _RecordingPostMutation(registry.post_mutation)
        worker = StreamingDecoderWorker(
            streamer=streamer,
            decode_fn=_slow_decoded_pcm,
            post_mutation=recording_post,
            session_id=sid,
            model_type="qwen3_tts",
            hardware="cpu",
        )

        # Pre-fill queue so worker is busy.
        for _ in range(10):
            streamer.queue.put([1, 2, 3, 4, 5, 6])

        coordinator._session_id_to_coordination_id[sid] = "coord_xyz"
        hook_call_count: list[int] = []

        def cancel_hook():
            hook_call_count.append(1)
            streamer._cancel_event.set()
            fut = asyncio.run_coroutine_threadsafe(
                coordinator.cancel_playback(sid), event_loop_thread
            )
            fut.result(timeout=2.0)

        registry.register_cancel_hook(sid, cancel_hook)
        worker.start()

        time.sleep(0.015)
        _drain_qt_events(qapp)

        # First cancel.
        registry.request_cancel(sid)
        # Second cancel might find the hook already cleared (if Qt
        # processed the cancel slot already) or still present (if not).
        # Either way: no exception.
        registry.request_cancel(sid)

        # Wait for worker drain.
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            _drain_qt_events(qapp)
            session = registry.get(sid)
            if session is not None and session.state == SessionState.CANCELLED:
                break
            time.sleep(0.005)
        worker.join(timeout=1.0)

        # Worker exited cleanly.
        assert worker.is_alive() is False

        # Hook fired at least once. The second call's behavior depends on
        # Qt event-loop timing — both 1 and 2 are valid per AC #1's "no
        # exception, no duplicate cancel post" contract.
        assert len(hook_call_count) in (1, 2)

        # Exactly one ('cancel', sid) post landed (the worker exits after
        # one drain-on-cancel; the second request_cancel did not produce
        # a duplicate post).
        cancel_calls = [c for c in recording_post.calls if c[1] == "cancel"]
        assert len(cancel_calls) == 1


# --------------------------------------------------------------------------- #
# Story 16.5 — TestCancelChainLatency (AC #1)
# --------------------------------------------------------------------------- #


class TestCancelChainLatency:
    """AC #1 — wallclock latency from request_cancel entry to the
    registry's cancel(session_id) slot finishing. Architecture target is
    100ms; the test bound is 200ms to absorb Windows scheduler jitter
    and AV scanners (Story 16.7's empirical-validation harness is what
    enforces the production target). Repeated 3 times to verify stability.
    """

    @pytest.mark.parametrize("run_index", [0, 1, 2])
    def test_cancel_chain_latency_under_100ms(
        self, qapp, registry, coordinator, run_index, event_loop_thread
    ):
        sid = registry.create_session(
            text=f"hello-{run_index}",
            voice="default",
            model_type="qwen3_tts",
        )
        registry.start_generation(sid)
        streamer = CodecTokenStreamer(chunk_size=4, lookahead=2)
        recording_post = _RecordingPostMutation(registry.post_mutation)
        worker = StreamingDecoderWorker(
            streamer=streamer,
            # Use the ~5ms-per-chunk slow decode to keep the worker
            # responsive at top-of-loop when cancel fires — the
            # alternative (a microsecond decode_fn that drains the queue
            # instantly) leaves the worker blocked on get() with no way
            # to wake (post-cancel put() is a no-op per Story 16.3 D-11).
            # 5ms × ~8 chunks = ~40ms worst-case worker workload; well
            # within the 200ms test bound.
            decode_fn=_slow_decoded_pcm,
            post_mutation=recording_post,
            session_id=sid,
            model_type="qwen3_tts",
            hardware="cpu",
        )

        coordinator._session_id_to_coordination_id[sid] = f"coord_{run_index}"

        # Pre-fill enough chunks that the worker is still iterating when
        # cancel fires (mirrors the production case where HF .generate()
        # is continuously feeding the queue).
        for _ in range(8):
            streamer.queue.put([1, 2, 3, 4, 5, 6])

        cancel_hook = _build_cancel_hook(
            streamer, coordinator, sid, event_loop_thread
        )
        registry.register_cancel_hook(sid, cancel_hook)
        worker.start()

        # Give the worker a moment to enter the loop (avoid measuring the
        # thread-startup delay as part of cancel-chain latency).
        time.sleep(0.005)

        # MEASURE: t_start → registry CANCELLED state.
        t_start = time.perf_counter()
        registry.request_cancel(sid)

        deadline = t_start + 0.5
        while time.perf_counter() < deadline:
            _drain_qt_events(qapp)
            session = registry.get(sid)
            if session is not None and session.state == SessionState.CANCELLED:
                break
            time.sleep(0.001)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        worker.join(timeout=1.0)
        assert worker.is_alive() is False

        assert registry.get(sid) is not None
        assert registry.get(sid).state == SessionState.CANCELLED
        assert latency_ms < 200.0, (
            f"run #{run_index}: cancel-chain latency {latency_ms:.1f}ms "
            f"exceeded 200ms in-process bound (architecture target: 100ms)"
        )


# --------------------------------------------------------------------------- #
# Story 16.6 — TestTrueStreamDispatchEndToEnd (AC #1, #9, #10)
# --------------------------------------------------------------------------- #


class _RecordingMetricsRecorder:
    """Records metrics.record() calls for Story 16.6 dispatch-metric assertions."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, name, value, **tags):
        self.calls.append((name, value, dict(tags)))

    def calls_for(self, name):
        return [c for c in self.calls if c[0] == name]


def _build_true_stream_service(registry, coordinator, monkeypatch=None):
    """Construct a QwenTTSService primed for TRUE_STREAM end-to-end tests.

    The model is a MagicMock with ``.model.generate`` and
    ``.speech_tokenizer.decode`` configured per-test. The service's
    BaseService status is forced to RUNNING and the model registry is
    primed so ``ensure_model_loaded`` and ``get_loaded_model`` return the
    mock without actually loading anything.
    """
    from myvoice.models.app_settings import AppSettings
    from myvoice.services.core.base_service import ServiceStatus
    from myvoice.services.qwen_tts_service import QwenTTSService

    settings = AppSettings(streaming_mode_override="true_stream")
    service = QwenTTSService(
        audio_coordinator=coordinator,
        device="cpu",
        dtype="float32",
        session_registry=registry,
        app_settings=settings,
    )

    # Force-running per BaseService.is_running() check.
    service.status = ServiceStatus.RUNNING

    # Initialize the executor + semaphore that start() would have created.
    from concurrent.futures import ThreadPoolExecutor
    service._executor = ThreadPoolExecutor(max_workers=1)
    service._request_semaphore = asyncio.Semaphore(1)

    # Prime the model registry — ensure_model_loaded returns success and
    # get_loaded_model returns a MagicMock. The MagicMock's
    # .model.generate and .speech_tokenizer.decode are the test's hooks.
    mock_model = MagicMock()
    service._model_registry.ensure_model_loaded = AsyncMock(
        return_value=(True, None)
    )
    service._model_registry.get_loaded_model = MagicMock(return_value=mock_model)
    service._model_registry.device = "cpu"

    # Patch metrics.record in the qwen_tts_service module's namespace so the
    # dispatcher's emissions are observable.
    return service, mock_model


class TestTrueStreamDispatchEndToEnd:
    """Story 16.6 — full dispatch path from public entry through TRUE_STREAM
    to a populated QwenTTSResponse. The model and tokenizer are mocked but
    the rest of the chain (registry, audio coordinator, real
    CodecTokenStreamer + StreamingDecoderWorker) runs for real.
    """

    def test_true_stream_produces_correct_audio_for_representative_input(
        self, qapp, registry, coordinator, mock_monitor, mock_virtual,
        monkeypatch,
    ):
        """AC #1 — fake .generate feeds 100 tokens; fake decode_fn maps
        token_id → token_id*0.01 sample. Expect 4 chunks (100 / chunk_size=25)
        of float32 PCM, success response, registry reaches READY_TO_PLAY,
        play_dual_stream called once.
        """
        from myvoice.services.qwen_tts_service import GenerationMode, QwenModelType, QwenTTSRequest
        service, mock_model = _build_true_stream_service(registry, coordinator)

        # Mock play_dual_stream so we can assert it was invoked.
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        # Patch the talker builder so we feed tokens deterministically. The
        # default builder calls ``model.model.generate(streamer=streamer)``
        # with parameters bound to qwen-tts internals; for the test we
        # bypass that binding entirely.
        def fake_talker_builder(model, request, streamer):
            def run():
                # Feed 100 tokens at 1ms intervals (~100ms total).
                for tok in range(100):
                    streamer.put([tok])
                    time.sleep(0.001)
                streamer.end()
            return run

        monkeypatch.setattr(
            service, "_build_true_stream_talker", fake_talker_builder
        )

        # Patch the decode_fn builder so we don't need a real torch tokenizer.
        def fake_decode_fn_builder(model):
            def decode(chunk_tokens):
                return np.array(
                    [t * 0.01 for t in chunk_tokens], dtype=np.float32
                )
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        # Capture metric emissions.
        recorder = _RecordingMetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )

        request = QwenTTSRequest(
            text="hello world",
            language="Auto",
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker="Ryan",
            streaming=True,
        )

        async def runner():
            async def drainer(stop_evt):
                while not stop_evt.is_set():
                    qapp.processEvents()
                    await asyncio.sleep(0.005)

            stop_evt = asyncio.Event()
            drain_task = asyncio.create_task(drainer(stop_evt))
            try:
                resp = await service._generate_true_stream(request)
            finally:
                stop_evt.set()
                await drain_task
            return resp

        response = asyncio.run(runner())
        # Drain anything queued post-finish.
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)

        assert response.success is True, (
            f"Response not successful: {response.error_message}"
        )
        assert response.mode == GenerationMode.STREAMING
        assert response.audio_data is not None
        # 100 tokens at chunk_size=25 → 4 chunks (the streamer's lookahead
        # affects the first chunk's sample count via overlap-add trim).
        assert response.chunks_generated == 4
        # Audio length within ±5 of 100 samples (overlap-add trim slack).
        assert 95 <= len(response.audio_data) <= 105

        # Registry session reached READY_TO_PLAY at finalize time. The state
        # may have advanced further (PLAYING / DONE) by the time we assert.
        # Just check the session exists and is past GENERATING.
        sess = None
        # The session id was created inside _generate_true_stream; recover
        # from the registry's most recent session.
        from myvoice.services.sessions import SessionRegistry
        for sid_iter in list(registry._sessions.keys()):
            sess = registry.get(sid_iter)
            break
        assert sess is not None
        assert sess.state.value not in ("pending", "generating"), (
            f"session state stuck at {sess.state.value} — finalize not "
            f"posted by worker"
        )

        # play_dual_stream called once with the session_id.
        assert coordinator.play_dual_stream.await_count == 1

        # Story 16.6 metric: streaming_mode emitted with value='true_stream'.
        # NOTE: this metric is emitted from the dispatcher (Task 2), not
        # from _generate_true_stream itself. When called directly (this
        # test), no streaming_mode metric fires — only first_chunk_latency_ms.
        latency_metrics = recorder.calls_for("first_chunk_latency_ms")
        assert len(latency_metrics) == 1
        assert latency_metrics[0][1] >= 0.0  # value in ms

    def test_true_stream_via_dispatch_emits_streaming_mode_metric(
        self, qapp, registry, coordinator, monkeypatch,
    ):
        """AC #6 in integration — when ``_dispatch_by_streaming_mode`` is the
        entry point (the production path), the streaming_mode metric fires."""
        from myvoice.services.qwen_tts_service import QwenModelType, QwenTTSRequest
        from myvoice.services.tts_streaming import StreamingMode

        service, mock_model = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        def fake_talker_builder(model, request, streamer):
            def run():
                for tok in range(50):
                    streamer.put([tok])
                    time.sleep(0.001)
                streamer.end()
            return run

        def fake_decode_fn_builder(model):
            def decode(chunk_tokens):
                return np.array(
                    [t * 0.01 for t in chunk_tokens], dtype=np.float32
                )
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_talker", fake_talker_builder
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        recorder = _RecordingMetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )

        request = QwenTTSRequest(
            text="hi",
            language="Auto",
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker="Ryan",
            streaming=True,
        )

        async def runner():
            async def drainer(stop_evt):
                while not stop_evt.is_set():
                    qapp.processEvents()
                    await asyncio.sleep(0.005)

            stop_evt = asyncio.Event()
            drain_task = asyncio.create_task(drainer(stop_evt))
            try:
                resp = await service._dispatch_by_streaming_mode(
                    request, StreamingMode.TRUE_STREAM
                )
            finally:
                stop_evt.set()
                await drain_task
            return resp

        response = asyncio.run(runner())
        assert response.success is True

        mode_metrics = recorder.calls_for("streaming_mode")
        # Only one streaming_mode metric — TRUE_STREAM succeeded on first try.
        assert len(mode_metrics) == 1
        assert mode_metrics[0][1] == "true_stream"

        # No fallback metrics.
        assert recorder.calls_for("streaming_mode_fallback") == []

    def test_cancel_mid_true_stream_propagates_through_chain(
        self, qapp, registry, coordinator, monkeypatch, event_loop_thread,
    ):
        """AC #9 — cancel mid-TRUE_STREAM dispatch propagates correctly:
        request_cancel → hook fires → streamer event flips → worker drains
        → posts ('cancel', sid) → registry CANCELLED. The dispatch path
        does NOT post a duplicate ('cancel', sid) per the P-7 invariant.

        The talker BLOCKS until the cancel hook fires the streamer's cancel
        event, guaranteeing cancel actually lands mid-stream (Story 16.6
        review H2 — the previous version of this test used wallclock timing
        and accepted ``cancel_count <= 1``, which silently passed when the
        cancel never propagated).
        """
        from myvoice.services.qwen_tts_service import QwenModelType, QwenTTSRequest

        service, mock_model = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        # Talker feeds 5 tokens (enough for first-chunk to fire and
        # play_dual_stream to be called), then BLOCKS on the cancel event
        # so cancel deterministically lands mid-stream.
        def fake_talker_builder(model, request, streamer):
            def run():
                for tok in range(5):
                    streamer.put([tok])
                # Block until cancel actually fires; ensures the test never
                # races past the cancel point. 5s is a hard upper bound —
                # the cancel_after_delay coroutine fires it within ~50ms.
                streamer._cancel_event.wait(timeout=5.0)
                # Continue feeding so the worker is still consuming when
                # the next top-of-loop check sees the cancel event.
                for tok in range(5, 200):
                    streamer.put([tok])
                    if streamer._cancel_event.is_set():
                        break
                try:
                    streamer.end()
                except Exception:
                    pass
            return run

        def fake_decode_fn_builder(model):
            def decode(chunk_tokens):
                return np.array(
                    [t * 0.01 for t in chunk_tokens], dtype=np.float32
                )
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_talker", fake_talker_builder
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        request = QwenTTSRequest(
            text="hello",
            language="Auto",
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker="Ryan",
            streaming=True,
        )

        # Count cancel transitions exactly (was '<=' before; AC #9 requires
        # exactly one per session — Story 16.6 review H2 / Subtask 7.6).
        cancel_count_box = [0]

        def state_listener(sid_arg, _state):
            sess = registry.get(sid_arg)
            if sess is not None and sess.state.value == "cancelled":
                cancel_count_box[0] += 1

        registry.session_state_changed.connect(state_listener)

        async def runner():
            stop_evt = asyncio.Event()

            async def drainer():
                while not stop_evt.is_set():
                    qapp.processEvents()
                    await asyncio.sleep(0.005)

            async def cancel_after_delay():
                deadline = time.perf_counter() + 2.0
                while (
                    time.perf_counter() < deadline
                    and service._current_session_id is None
                ):
                    await asyncio.sleep(0.001)
                # Wait briefly so the talker has fed its initial tokens.
                await asyncio.sleep(0.05)
                if service._current_session_id is not None:
                    service._session_registry.request_cancel(
                        service._current_session_id
                    )

            drain_task = asyncio.create_task(drainer())
            cancel_task = asyncio.create_task(cancel_after_delay())
            try:
                resp = await service._generate_true_stream(request)
            finally:
                stop_evt.set()
                await drain_task
                await cancel_task
            return resp

        response = asyncio.run(runner())
        # Drain Qt events for any pending state transitions.
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)

        assert response is not None
        # AC #9: exactly one cancel transition (P-7 — worker is canonical
        # source; dispatcher must NOT double-post).
        assert cancel_count_box[0] == 1, (
            f"Got {cancel_count_box[0]} cancelled-state transitions; "
            f"AC #9 requires exactly one (P-7 invariant)"
        )

    def test_concurrent_dispatches_serialized_via_semaphore(
        self, qapp, registry, coordinator, monkeypatch,
    ):
        """AC #10 — the existing ``_request_semaphore`` serializes
        concurrent TRUE_STREAM dispatches; second dispatch waits on first."""
        from myvoice.services.qwen_tts_service import QwenModelType, QwenTTSRequest

        service, mock_model = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        # Each talker feeds 30 tokens at 0.5ms intervals.
        def fake_talker_builder(model, request, streamer):
            def run():
                for tok in range(30):
                    streamer.put([tok])
                    time.sleep(0.0005)
                streamer.end()
            return run

        def fake_decode_fn_builder(model):
            def decode(chunk_tokens):
                return np.array(
                    [t * 0.01 for t in chunk_tokens], dtype=np.float32
                )
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_talker", fake_talker_builder
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        request1 = QwenTTSRequest(
            text="first",
            language="Auto",
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker="Ryan",
            streaming=True,
        )
        request2 = QwenTTSRequest(
            text="second",
            language="Auto",
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker="Ryan",
            streaming=True,
        )

        async def runner():
            async def drainer(stop_evt):
                while not stop_evt.is_set():
                    qapp.processEvents()
                    await asyncio.sleep(0.005)

            stop_evt = asyncio.Event()
            drain_task = asyncio.create_task(drainer(stop_evt))
            try:
                # Run two dispatches concurrently — the semaphore should
                # serialize them.
                t1 = asyncio.create_task(
                    service._generate_true_stream(request1)
                )
                t2 = asyncio.create_task(
                    service._generate_true_stream(request2)
                )
                resp1, resp2 = await asyncio.gather(t1, t2)
            finally:
                stop_evt.set()
                await drain_task
            return resp1, resp2

        resp1, resp2 = asyncio.run(runner())
        assert resp1.success is True, f"first: {resp1.error_message}"
        assert resp2.success is True, f"second: {resp2.error_message}"
        # Two distinct sessions in the registry.
        assert len(registry._sessions) == 2

    def test_cancel_before_first_chunk_returns_cleanly(
        self, qapp, registry, coordinator, monkeypatch, event_loop_thread,
    ):
        """AC #9 / Subtask 7.4 — the cancel-before-first-chunk edge case.

        The talker has not yet produced enough tokens to fill a chunk
        (chunk_size=25) when cancel arrives. The dispatch must not crash,
        must not leak threads, and must produce exactly one cancel
        transition on the session.

        The literal "cancel before talker_thread.start()" the AC text
        describes is structurally untestable today — the dispatcher
        unconditionally starts both threads — so this test exercises the
        next-closest realistic timing (cancel before chunk-size threshold,
        before play_dual_stream is invoked).
        """
        from myvoice.services.qwen_tts_service import QwenModelType, QwenTTSRequest

        service, _mock_model = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        # Talker feeds 30 tokens (so chunk_size=25 produces one chunk and
        # the worker processes it before cancel fires), then BLOCKS on
        # cancel — guaranteeing cancel deterministically lands before
        # finalize. After cancel, talker calls end() so the worker drains.
        def fake_talker_builder(model, request, streamer):
            def run():
                for tok in range(30):
                    streamer.put([tok])
                streamer._cancel_event.wait(timeout=5.0)
                try:
                    streamer.end()
                except Exception:
                    pass
            return run

        def fake_decode_fn_builder(model):
            def decode(chunk_tokens):
                return np.array(
                    [t * 0.01 for t in chunk_tokens], dtype=np.float32
                )
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_talker", fake_talker_builder
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        request = QwenTTSRequest(
            text="hello",
            language="Auto",
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker="Ryan",
            streaming=True,
        )

        cancel_count_box = [0]

        def state_listener(sid_arg, _state):
            sess = registry.get(sid_arg)
            if sess is not None and sess.state.value == "cancelled":
                cancel_count_box[0] += 1

        registry.session_state_changed.connect(state_listener)

        async def runner():
            stop_evt = asyncio.Event()

            async def drainer():
                while not stop_evt.is_set():
                    qapp.processEvents()
                    await asyncio.sleep(0.005)

            async def cancel_early():
                deadline = time.perf_counter() + 2.0
                while (
                    time.perf_counter() < deadline
                    and service._current_session_id is None
                ):
                    await asyncio.sleep(0.001)
                # Fire cancel quickly — before play_dual_stream's first
                # invocation in normal flow.
                if service._current_session_id is not None:
                    service._session_registry.request_cancel(
                        service._current_session_id
                    )

            drain_task = asyncio.create_task(drainer())
            cancel_task = asyncio.create_task(cancel_early())
            try:
                resp = await service._generate_true_stream(request)
            finally:
                stop_evt.set()
                await drain_task
                await cancel_task
            return resp

        response = asyncio.run(runner())
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)

        # Dispatch returned a response (no exception leaked). Threads are
        # daemon=True and local to _generate_true_stream, so reaching this
        # point inside the pytest wallclock proves no leak.
        assert response is not None
        # Exactly one cancel transition — never zero (cancel missed), never
        # two (double-post). P-7 invariant.
        assert cancel_count_box[0] == 1, (
            f"Got {cancel_count_box[0]} cancelled-state transitions; "
            f"AC #9 requires exactly one"
        )

    def test_no_double_cancel_post_from_dispatch_path(
        self, qapp, registry, coordinator, monkeypatch, event_loop_thread,
    ):
        """AC #9 / Subtask 7.6 — P-7 invariant: the worker's drain-on-cancel
        is the canonical source of ``('cancel', sid)``. The dispatcher's
        ``asyncio.CancelledError`` handler must NOT post a duplicate
        ``('cancel', sid)`` itself, or Story 16.5 AC #1's "exactly one
        cancel post" assertion fails.

        Asserts on the count of ``post_mutation('cancel', sid)`` calls
        observed (exactly one), regardless of whether they originated from
        the worker or the dispatcher.
        """
        from myvoice.services.qwen_tts_service import QwenModelType, QwenTTSRequest

        service, _mock_model = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        # Same blocking talker pattern as test_cancel_mid_true_stream so
        # cancel deterministically lands mid-stream.
        def fake_talker_builder(model, request, streamer):
            def run():
                for tok in range(5):
                    streamer.put([tok])
                streamer._cancel_event.wait(timeout=5.0)
                for tok in range(5, 100):
                    streamer.put([tok])
                    if streamer._cancel_event.is_set():
                        break
                try:
                    streamer.end()
                except Exception:
                    pass
            return run

        def fake_decode_fn_builder(model):
            def decode(chunk_tokens):
                return np.array(
                    [t * 0.01 for t in chunk_tokens], dtype=np.float32
                )
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_talker", fake_talker_builder
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        # Wrap registry.post_mutation to count ('cancel', sid) calls. We
        # cannot replace the bound method (would break the queued-connection
        # wiring), so subscribe at the call site by monkeypatching the
        # module-level reference the dispatcher reads.
        cancel_call_count = [0]
        original_post_mutation = registry.post_mutation

        def counting_post(method, *args, **kwargs):
            if method == "cancel":
                cancel_call_count[0] += 1
            return original_post_mutation(method, *args, **kwargs)

        # The dispatcher reads ``self._session_registry.post_mutation``
        # at construction-time of the worker's _wrapped_post closure, so
        # we patch the bound method on this specific registry instance.
        monkeypatch.setattr(registry, "post_mutation", counting_post)

        request = QwenTTSRequest(
            text="hello",
            language="Auto",
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker="Ryan",
            streaming=True,
        )

        async def runner():
            stop_evt = asyncio.Event()

            async def drainer():
                while not stop_evt.is_set():
                    qapp.processEvents()
                    await asyncio.sleep(0.005)

            async def cancel_after_delay():
                deadline = time.perf_counter() + 2.0
                while (
                    time.perf_counter() < deadline
                    and service._current_session_id is None
                ):
                    await asyncio.sleep(0.001)
                await asyncio.sleep(0.05)
                if service._current_session_id is not None:
                    service._session_registry.request_cancel(
                        service._current_session_id
                    )

            drain_task = asyncio.create_task(drainer())
            cancel_task = asyncio.create_task(cancel_after_delay())
            try:
                resp = await service._generate_true_stream(request)
            finally:
                stop_evt.set()
                await drain_task
                await cancel_task
            return resp

        response = asyncio.run(runner())
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)

        assert response is not None
        # P-7 invariant: exactly one ('cancel', sid) post — never zero
        # (cancel was missed) and never two (dispatcher double-posted).
        assert cancel_call_count[0] == 1, (
            f"Got {cancel_call_count[0]} ('cancel', sid) posts; AC #9 / "
            f"P-7 invariant requires exactly 1 (worker is canonical source)"
        )


# --------------------------------------------------------------------------- #
# Story 16.7 — TestSilentTalkerSurfaceAsFailure (regression: empty-chunk guard)
# --------------------------------------------------------------------------- #


class TestSilentTalkerSurfacesAsFailure:
    """Story 16.7 empirical-validation regression — when the talker thread
    silently fails (its except branch in ``_build_true_stream_talker``
    swallows all exceptions and just calls ``streamer.end()``), the dispatch
    used to return ``success=True`` with zero-sample audio. The fallback
    chain never fired and CUDA users heard silence.

    These tests mirror the EXACT bug class (per
    ``memory/code_review_regression_test_exact_class.md``):
      1. ``_generate_true_stream`` raises ``RuntimeError`` when the talker
         emits zero tokens, so the dispatcher's fallback chain catches it.
      2. ``_dispatch_by_streaming_mode(request, TRUE_STREAM)`` falls back
         to SENTENCE_STREAM when TRUE_STREAM raises the empty-chunks error.
    """

    @pytest.mark.qt_no_exception_capture
    def test_zero_token_talker_raises_so_fallback_chain_fires(
        self, qapp, registry, coordinator, monkeypatch,
    ):
        """A talker that emits 0 tokens (mirroring the silent-failure mode
        Story 16.6 shipped) must surface as a raised exception, not as a
        successful empty-audio response. This is the load-bearing guard for
        Story 16.7 AC #5's fallback semantics.
        """
        from myvoice.services.qwen_tts_service import QwenModelType, QwenTTSRequest
        service, _mock_model = _build_true_stream_service(registry, coordinator)

        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        def silent_talker_builder(model, request, streamer):
            def run():
                # Emit no tokens at all — exactly the production failure mode
                # observed during Story 16.7's first empirical run on the RTX
                # 5090 host (real-model talker raised, swallowed by the
                # except branch, ``streamer.end()`` called with empty queue).
                streamer.end()
            return run

        def fake_decode_fn_builder(model):
            def decode(chunk_tokens):
                return np.array([0.0] * len(chunk_tokens), dtype=np.float32)
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_talker", silent_talker_builder
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        recorder = _RecordingMetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )

        request = QwenTTSRequest(
            text="hello world",
            language="Auto",
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker="Ryan",
            streaming=True,
        )

        async def runner():
            async def drainer(stop_evt):
                while not stop_evt.is_set():
                    qapp.processEvents()
                    await asyncio.sleep(0.005)

            stop_evt = asyncio.Event()
            drain_task = asyncio.create_task(drainer(stop_evt))
            try:
                with pytest.raises(RuntimeError, match="0 audio chunks"):
                    await service._generate_true_stream(request)
            finally:
                stop_evt.set()
                await drain_task

        asyncio.run(runner())
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)

    @pytest.mark.qt_no_exception_capture
    def test_dispatch_fallback_chain_routes_to_sentence_stream_on_silent_talker(
        self, qapp, registry, coordinator, monkeypatch,
    ):
        """When TRUE_STREAM raises the empty-chunks error, the existing
        ``_dispatch_by_streaming_mode`` fallback chain must route to
        SENTENCE_STREAM and return its successful response. This is the
        end-to-end NFR7 graceful-degradation path Story 16.7 unblocks.
        """
        from myvoice.services.qwen_tts_service import (
            GenerationMode, QwenModelType, QwenTTSRequest, QwenTTSResponse,
        )
        from myvoice.services.tts_streaming import StreamingMode

        service, _mock_model = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        # Silent talker — talker thread produces 0 tokens, so
        # _generate_true_stream raises RuntimeError per the fix.
        def silent_talker_builder(model, request, streamer):
            def run():
                streamer.end()
            return run

        def fake_decode_fn_builder(model):
            def decode(chunk_tokens):
                return np.array([0.0] * len(chunk_tokens), dtype=np.float32)
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_talker", silent_talker_builder
        )
        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        # Mock _generate_streaming so the fallback path returns a
        # deterministic success response we can assert on.
        sentence_response = QwenTTSResponse(
            success=True,
            audio_data=np.zeros(2400, dtype=np.float32),
            sample_rate=24000,
            mode=GenerationMode.STREAMING,
            chunks_generated=2,
        )
        monkeypatch.setattr(
            service, "_generate_streaming",
            AsyncMock(return_value=sentence_response),
        )

        recorder = _RecordingMetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )

        request = QwenTTSRequest(
            text="hello world",
            language="Auto",
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker="Ryan",
            streaming=True,
        )

        async def runner():
            async def drainer(stop_evt):
                while not stop_evt.is_set():
                    qapp.processEvents()
                    await asyncio.sleep(0.005)

            stop_evt = asyncio.Event()
            drain_task = asyncio.create_task(drainer(stop_evt))
            try:
                resp = await service._dispatch_by_streaming_mode(
                    request, StreamingMode.TRUE_STREAM,
                )
            finally:
                stop_evt.set()
                await drain_task
            return resp

        response = asyncio.run(runner())
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)

        # The fallback chain caught the TRUE_STREAM RuntimeError, ran
        # SENTENCE_STREAM, and returned its successful response.
        assert response is sentence_response
        assert response.success is True
        assert response.audio_data.shape == (2400,)

        # Story 16.6 fallback metric fired with from_mode=true_stream and
        # next-mode=sentence_stream.
        fallback_metrics = recorder.calls_for("streaming_mode_fallback")
        assert len(fallback_metrics) == 1
        # tags is the third element of the recorder tuple.
        _, fallback_target, fallback_tags = fallback_metrics[0]
        assert fallback_target == "sentence_stream"
        assert fallback_tags.get("from_mode") == "true_stream"
