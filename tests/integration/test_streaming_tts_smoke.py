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
from myvoice.services.tts_streaming import (
    codec_token_streamer as _codec_token_streamer,
)


# --------------------------------------------------------------------------- #
# Story 20.4 — the streamer geometry is DERIVED here, once, and never restated
# in an individual test.
#
# Before Story 20.4 this file carried ``25``, ``30`` and ``20`` as literals in
# three assertions. The retune (chunk_size 25 -> 10) turned all three red at
# once, which is the right behaviour for a smoke suite -- but the fix is to
# derive, not to re-type the new numbers, because the next retune would
# otherwise repeat the whole exercise.
# --------------------------------------------------------------------------- #

_STREAMER_CHUNK_SIZE = _codec_token_streamer.DEFAULT_CHUNK_SIZE
_STREAMER_LOOKAHEAD = _codec_token_streamer.DEFAULT_LOOKAHEAD
_STREAMER_WINDOW = _STREAMER_CHUNK_SIZE + _STREAMER_LOOKAHEAD


def _expected_chunk_count(n_tokens: int) -> int:
    """Chunks a stream of ``n_tokens`` yields at the committed geometry.

    Mirrors ``CodecTokenStreamer``'s arithmetic rather than calling it: the
    buffer pushes once it holds ``chunk_size + lookahead`` tokens and then
    slides forward by ``chunk_size``, keeping the lookahead tail as the next
    chunk's left context. So after ``n`` in-loop pushes the stream has
    consumed ``lookahead + n * chunk_size`` tokens, and ``end()`` flushes
    whatever remains.

    At the pre-20.4 geometry (25 + 5) and n_tokens=100 this returns 4, which
    is the number the assertions in this file used to hard-code.
    """
    pushes = max(0, (n_tokens - _STREAMER_LOOKAHEAD) // _STREAMER_CHUNK_SIZE)
    residual = n_tokens - pushes * _STREAMER_CHUNK_SIZE
    return pushes + (1 if residual > 0 else 0)


# A step count guaranteed to fall SHORT of the first-emit threshold, so the
# only chunk the generation produces is the terminal residual flush. This is
# the Story 20.1 short-utterance / Clear Comms regime.
_SUB_THRESHOLD_STEPS = max(1, _STREAMER_WINDOW - 1)


def test_expected_chunk_count_helper_matches_both_shipped_geometries():
    """Self-check for the helper the assertions above depend on.

    A test helper that silently computes the wrong expectation is worse
    than a hard-coded literal, because it looks principled. These two rows
    pin it against the only two geometries this project has shipped:
    25 + 5 (through Story 20.3) and 10 + 5 (Story 20.4). The 25 + 5 answer,
    4, is the literal this file carried before the retune.
    """
    global _STREAMER_CHUNK_SIZE, _STREAMER_LOOKAHEAD
    saved = (_STREAMER_CHUNK_SIZE, _STREAMER_LOOKAHEAD)
    try:
        _STREAMER_CHUNK_SIZE, _STREAMER_LOOKAHEAD = 25, 5
        assert _expected_chunk_count(100) == 4
        _STREAMER_CHUNK_SIZE, _STREAMER_LOOKAHEAD = 10, 5
        assert _expected_chunk_count(100) == 10
    finally:
        _STREAMER_CHUNK_SIZE, _STREAMER_LOOKAHEAD = saved


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
        token_id → token_id*0.01 sample. Expect ``_expected_chunk_count(100)``
        chunks of float32 PCM (4 at the pre-20.4 25+5 geometry, 10 at the
        committed 10+5), success response, registry reaches READY_TO_PLAY,
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
        # Chunk count is a function of the committed streamer geometry (the
        # lookahead also affects the first chunk's sample count via the
        # overlap-add trim). Derived, not restated -- see
        # ``_expected_chunk_count``.
        assert response.chunks_generated == _expected_chunk_count(100), (
            f"100 tokens at chunk_size={_STREAMER_CHUNK_SIZE}/"
            f"lookahead={_STREAMER_LOOKAHEAD} must yield "
            f"{_expected_chunk_count(100)} chunks; got "
            f"{response.chunks_generated}"
        )
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
        when cancel arrives. The dispatch must not crash,
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

        # Talker feeds 30 tokens (enough to fill at least one window at any
        # shipped geometry, so the worker processes a chunk before cancel
        # fires), then BLOCKS on
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


# --------------------------------------------------------------------------- #
# Story 16.8 — TestTrueStreamWireUpEndToEnd
#
# The Story 16.6 / 16.7 smoke tests above ALL monkey-patch
# ``service._build_true_stream_talker`` to inject a controlled token-feeding
# closure. That patching is what allowed Story 16.6's structural wire-up
# bug — ``model.model.generate(streamer=streamer)`` with no conditioning
# args — to ship: every test bypassed the real builder. This class is the
# regression guard against another silent wire-up failure ever recurring:
# it exercises the REAL ``_build_true_stream_talker`` body (the talker-patch
# variant of Path A from Story 16.8) with a fake-model fixture whose
# attribute graph satisfies the wire-up's runtime contract — a reachable
# ``model.model.talker.generate`` call that the patch can interpose on,
# and a ``model.generate_custom_voice`` that drives the wrapper through to
# that call site.
# --------------------------------------------------------------------------- #


def _make_streamer_aware_fake_model(
    step_count: int = 100, num_code_groups: int = 4,
):
    """Build a MagicMock whose attribute graph satisfies Story 16.8's
    forward-hook wire-up.

    Story 16.8 finding (2026-05-07): HF ``GenerationMixin._sample``'s
    standard ``streamer.put(next_tokens)`` protocol fires only with the
    main-codebook token; the qwen-tts talker is multi-codebook and the
    12Hz speech-tokenizer's ``decode`` requires
    ``(N_steps, num_code_groups)`` per sample. The production fix
    captures ``codec_ids`` via a ``talker.forward`` hook (each forward
    call returns ``Qwen3TTSTalkerOutputWithPast.hidden_states[1] = codec_ids``
    of shape ``(batch=1, num_code_groups)``).

    This fixture mimics that contract:
      - ``talker.forward`` is called step_count + 1 times (1 prefill +
        step_count generation steps). Prefill returns ``codec_ids=None``;
        each generation step returns a deterministic
        ``(1, num_code_groups)`` tensor.
      - ``talker.generate`` simulates HF ``_sample``'s loop by invoking
        ``talker.forward`` step_count + 1 times.
      - ``generate_custom_voice`` (and the two siblings) simulate the
        public wrapper by invoking ``talker.generate``.

    The fixture does NOT replace ``_build_true_stream_talker`` itself —
    that is the body under test. The forward-hook installed by
    ``_build_true_stream_talker`` interposes on ``talker.forward``,
    captures codec_ids, accumulates into step_buffer, and pushes chunks
    to ``streamer.queue`` when ``chunk_size + lookahead`` steps land.

    The chunk count follows the committed geometry: at the pre-Story-20.4
    defaults (chunk_size=25, lookahead=5) ``step_count=100`` produced 3
    chunks during generation (push points at 30 / 55 / 80 steps; slide
    forward by 25 each time keeps the last 5 as overlap) plus 1 residual
    chunk on flush. At the committed 10 + 5 the push points are denser.
    ``_expected_chunk_count`` is the arithmetic; no test in this file
    restates the number.

    Returns ``(mock_model, hits)`` where ``hits`` is a dict captured by
    the fake methods so tests can assert on call counts.
    """
    import torch

    mock_model = MagicMock()
    hits = {
        "talker_generate_calls": 0,
        "talker_forward_calls": 0,
        "custom_voice_calls": 0,
        "voice_clone_calls": 0,
        "voice_design_calls": 0,
    }

    def fake_talker_forward(*args, **kwargs):
        hits["talker_forward_calls"] += 1
        # First call: prefill (inputs_embeds.shape[1] > 1) — codec_ids
        # is None per modeling_qwen3_tts.py:1665-1667.
        if hits["talker_forward_calls"] == 1:
            codec_ids = None
        else:
            # Deterministic per-step codec_ids; values are
            # (step_index * 7) % 1024 across the num_code_groups
            # codebooks. Just needs to be a valid LongTensor of the
            # right shape for the decode_fn override to consume.
            step_idx = hits["talker_forward_calls"] - 1
            codec_ids = torch.tensor(
                [[(step_idx * 7 + g) % 1024 for g in range(num_code_groups)]],
                dtype=torch.long,
            )
        result = MagicMock()
        # ``Qwen3TTSTalkerOutputWithPast.hidden_states = (real_hidden, codec_ids)``
        # per modeling_qwen3_tts.py:1738.
        result.hidden_states = (None, codec_ids)
        return result

    def fake_talker_generate(*args, **kwargs):
        hits["talker_generate_calls"] += 1
        # Simulate HF GenerationMixin._sample's loop: 1 prefill forward
        # call + step_count generation forward calls. Each subsequent
        # forward returns one step's codec_ids; the production
        # forward-hook captures and accumulates.
        for _ in range(step_count + 1):
            mock_model.model.talker.forward(
                inputs_embeds=None, attention_mask=None,
            )
        return MagicMock()

    def fake_generate_custom_voice(
        text, speaker, language, instruct=None, non_streaming_mode=False,
    ):
        hits["custom_voice_calls"] += 1
        # Production wrapper at qwen3_tts_model.py:829 calls
        # ``self.model.generate(...)`` which internally calls
        # ``self.talker.generate(...)`` at modeling_qwen3_tts.py:2272.
        # For the test, we shortcut to the talker.
        return mock_model.model.talker.generate(
            inputs_embeds=None,
            attention_mask=None,
            trailing_text_hidden=None,
            tts_pad_embed=None,
        )

    def fake_generate_voice_design(
        text, instruct, language, non_streaming_mode=False,
    ):
        hits["voice_design_calls"] += 1
        return mock_model.model.talker.generate(
            inputs_embeds=None, attention_mask=None,
        )

    def fake_generate_voice_clone(
        text, language, voice_clone_prompt, non_streaming_mode=False,
    ):
        hits["voice_clone_calls"] += 1
        return mock_model.model.talker.generate(
            inputs_embeds=None, attention_mask=None,
        )

    mock_model.model.talker.forward = fake_talker_forward
    mock_model.model.talker.generate = fake_talker_generate
    mock_model.generate_custom_voice = fake_generate_custom_voice
    mock_model.generate_voice_design = fake_generate_voice_design
    mock_model.generate_voice_clone = fake_generate_voice_clone
    return mock_model, hits


class TestTrueStreamWireUpEndToEnd:
    """Story 16.8 — regression guard for the real ``_build_true_stream_talker``
    body. These tests do NOT monkey-patch ``_build_true_stream_talker``
    itself; they install a streamer-shaped fake model and exercise the
    production wire-up. A future regression that re-introduces the
    Story 16.6 silent-failure shape (e.g., literal ``model.model.generate(
    streamer=streamer)`` without conditioning, or a path that bypasses
    ``model.generate_custom_voice``) will produce zero chunks here and
    fail the empty-chunks assertion.
    """

    def test_real_wire_up_fires_streamer_for_custom_voice_request(
        self, qapp, registry, coordinator, monkeypatch,
    ):
        """AC #2 happy path — the talker-patch installs a forward-hook on
        ``model.model.talker.forward``, the wrapper's
        ``self.talker.generate(...)`` runs HF ``_sample`` which calls
        forward repeatedly, the hook captures multi-codebook codec_ids per
        step, chunks land on ``streamer.queue``, the worker decodes them,
        and the dispatch returns ``success=True`` with non-empty audio.
        """
        import torch
        from myvoice.services.qwen_tts_service import (
            GenerationMode, QwenModelType, QwenTTSRequest,
        )

        service, _placeholder_model = _build_true_stream_service(
            registry, coordinator,
        )
        # Replace the placeholder mock with our forward-hook-aware fake.
        # step_count=100 crosses the first-emit threshold repeatedly at
        # every shipped geometry; the exact chunk count follows
        # ``_expected_chunk_count`` and is not asserted here.
        fake_model, hits = _make_streamer_aware_fake_model(
            step_count=100, num_code_groups=4,
        )
        service._model_registry.get_loaded_model = MagicMock(
            return_value=fake_model
        )
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        # Override the decode_fn builder so we don't need a real
        # speech_tokenizer. The fake decode_fn receives a (N_steps, Q)
        # tensor and returns a deterministic float32 PCM array of length
        # ``samples_per_step * N_steps`` so the worker's overlap-add trim
        # has well-defined geometry.
        SAMPLES_PER_STEP = 100

        def fake_decode_fn_builder(model):
            def decode(chunk_tensor):
                # chunk_tensor: torch.Tensor of shape (N_steps, num_code_groups)
                if isinstance(chunk_tensor, torch.Tensor):
                    n_steps = chunk_tensor.shape[0]
                else:
                    n_steps = len(chunk_tensor)
                return np.full(
                    n_steps * SAMPLES_PER_STEP, 0.01, dtype=np.float32
                )
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        request = QwenTTSRequest(
            text="hello world",
            language="English",
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
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)

        # The wrapper fired exactly once.
        assert hits["custom_voice_calls"] == 1, (
            f"Expected 1 generate_custom_voice call, got "
            f"{hits['custom_voice_calls']}; the talker-patch wrapper must "
            f"route through the public entrypoint matching request.model_type"
        )
        # talker.generate fired exactly once (Story 16.8 invariant).
        assert hits["talker_generate_calls"] == 1, (
            f"Expected 1 talker.generate call, got "
            f"{hits['talker_generate_calls']}"
        )
        # Forward-hook fired step_count + 1 times (1 prefill + 100 gen).
        assert hits["talker_forward_calls"] == 101, (
            f"Expected 101 talker.forward calls (1 prefill + 100 gen), got "
            f"{hits['talker_forward_calls']}; the forward-hook may not be "
            f"installed, OR fake_talker_generate's loop count drifted"
        )

        # Empty-chunks guard was NOT triggered: dispatch returned success.
        assert response is not None
        assert response.success is True, (
            f"TRUE_STREAM dispatch returned failure: {response.error_message}"
        )
        assert response.mode == GenerationMode.STREAMING
        assert response.audio_data is not None
        assert response.audio_data.size > 0
        assert response.chunks_generated >= 1, (
            f"Expected >=1 chunks, got {response.chunks_generated}; the "
            f"forward-hook may have produced 0 codec_id captures"
        )

    def test_patch_is_restored_after_dispatch_completes(
        self, qapp, registry, coordinator, monkeypatch,
    ):
        """AC #2 invariant — Story 16.8's talker-patch installs a
        streamer-injecting wrapper for the duration of one dispatch and
        MUST restore the original ``model.model.talker.generate`` on exit
        (success, exception, or sentinel). Otherwise a subsequent
        non-streaming call (e.g., a SENTENCE_STREAM fallback re-using the
        same model instance) would still have the patch installed and
        fire the streamer in non-streaming context.
        """
        from myvoice.services.qwen_tts_service import (
            QwenModelType, QwenTTSRequest,
        )

        import torch

        service, _placeholder = _build_true_stream_service(
            registry, coordinator,
        )
        fake_model, _hits = _make_streamer_aware_fake_model(
            step_count=50, num_code_groups=4,
        )
        service._model_registry.get_loaded_model = MagicMock(
            return_value=fake_model
        )
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        # Snapshot both originals — Story 16.8 patches BOTH .generate
        # (for the _TalkerStreamComplete short-circuit) and .forward
        # (for the multi-codebook codec_ids capture).
        original_talker_generate = fake_model.model.talker.generate
        original_talker_forward = fake_model.model.talker.forward

        def fake_decode_fn_builder(model):
            def decode(chunk_tensor):
                n_steps = (
                    chunk_tensor.shape[0]
                    if isinstance(chunk_tensor, torch.Tensor)
                    else len(chunk_tensor)
                )
                return np.full(n_steps * 50, 0.01, dtype=np.float32)
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        request = QwenTTSRequest(
            text="hello",
            language="English",
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
                await service._generate_true_stream(request)
            finally:
                stop_evt.set()
                await drain_task

        asyncio.run(runner())
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)

        # After the dispatch, BOTH patches must be restored.
        assert fake_model.model.talker.generate is original_talker_generate, (
            "Story 16.8's talker.generate patch did not restore the "
            "original after dispatch — the patch leaked past its scope "
            "and would break a subsequent SENTENCE_STREAM fallback re-using "
            "the same model instance."
        )
        assert fake_model.model.talker.forward is original_talker_forward, (
            "Story 16.8's talker.forward patch did not restore the "
            "original after dispatch — the forward-hook leak would break "
            "a subsequent non-streaming call by intercepting its forward "
            "outputs and pushing to a stale streamer."
        )

    def test_real_wire_up_cooperative_cancel_does_not_raise(
        self, qapp, registry, coordinator, monkeypatch,
    ):
        """AC #2 cancel-mid-generation — the forward-hook checks
        ``streamer._cancel_event.is_set()`` before appending each
        captured ``codec_ids``; once cancel fires, accumulation stops and
        no further chunks are pushed. The fake talker.generate returns
        cleanly (no exception), the talker.generate patch's sentinel
        raises, ``_run_talker``'s ``except _TalkerStreamComplete`` catches
        it, residual is flushed (already-cancelled buffer may be empty),
        END_OF_STREAM is pushed, the talker thread exits without
        escalating to D-11's "no exceptions through HF internals"
        violation.
        """
        import torch
        from myvoice.services.qwen_tts_service import (
            QwenModelType, QwenTTSRequest,
        )

        service, _placeholder = _build_true_stream_service(
            registry, coordinator,
        )
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())

        # Fake talker that runs ~30 forward calls (1 prefill + 29 gen
        # steps, enough to push 1 chunk before cancel), then BLOCKS on
        # the cancel event mid-loop so cancel deterministically lands
        # before completion. After cancel fires, continues a few more
        # forward calls (mirroring HF's behavior of iterating a few more
        # times after streamer goes silent).
        cancel_observed = [False]
        post_cancel_forward_calls = [0]
        forward_call_count = [0]
        # Capture the streamer reference passed into _build_true_stream_talker
        # so the talker thread can directly assert on
        # streamer._cancel_event.is_set() rather than just trusting that
        # time.sleep(0.25) was long enough for cancel to propagate.
        # Without this capture, a regression that breaks the cancel hook
        # (e.g., register_cancel_hook never called, or _cancel_event.set()
        # removed from the hook) would silently pass this test.
        captured_streamer: list = []
        real_builder = service._build_true_stream_talker

        def spy_builder(model, request, streamer):
            captured_streamer.append(streamer)
            return real_builder(model, request, streamer)

        monkeypatch.setattr(
            service, "_build_true_stream_talker", spy_builder
        )

        def fake_talker_forward(*args, **kwargs):
            forward_call_count[0] += 1
            # Prefill = call #1, codec_ids=None.
            if forward_call_count[0] == 1:
                codec_ids = None
            else:
                step_idx = forward_call_count[0] - 1
                codec_ids = torch.tensor(
                    [[step_idx, step_idx + 1, step_idx + 2, step_idx + 3]],
                    dtype=torch.long,
                )
            result = MagicMock()
            result.hidden_states = (None, codec_ids)
            return result

        fake_model = MagicMock()
        fake_model.model.talker.forward = fake_talker_forward

        # The fake_talker_generate is called by the wrapper. It simulates
        # HF _sample by calling forward in a loop, sleeping mid-loop to
        # give the cancel coroutine time to fire ``request_cancel`` (which
        # flips the streamer's cancel_event via the cancel hook), then
        # continuing forward calls — the forward-hook in production will
        # skip codec_id accumulation once cancel fires.
        def fake_talker_generate(*args, **kwargs):
            # Initial prefill + 29 generation steps populates step_buffer
            # to 29 entries (one short of the 30-step chunk threshold,
            # so no chunk is pushed yet).
            for _ in range(30):
                fake_model.model.talker.forward(*args, **kwargs)
            # Wait for the cancel coroutine to run + request_cancel to
            # propagate through the registry's cancel hook all the way to
            # streamer._cancel_event. Poll the captured streamer's
            # _cancel_event directly rather than trusting a fixed sleep —
            # a sleep alone could pass even if the hook never fired.
            cancel_deadline = time.perf_counter() + 1.0
            while time.perf_counter() < cancel_deadline:
                if (
                    captured_streamer
                    and captured_streamer[0]._cancel_event.is_set()
                ):
                    cancel_observed[0] = True
                    break
                time.sleep(0.005)
            # Continue feeding forward calls — the production
            # forward-hook's cancel check will skip codec_id accumulation
            # per D-11.
            for _ in range(5):
                fake_model.model.talker.forward(*args, **kwargs)
                post_cancel_forward_calls[0] += 1
            return MagicMock()

        fake_model.model.talker.generate = fake_talker_generate

        cv_call_count = [0]

        def fake_generate_custom_voice(
            text, speaker, language, instruct=None, non_streaming_mode=False,
        ):
            cv_call_count[0] += 1
            return fake_model.model.talker.generate(
                inputs_embeds=None,
                attention_mask=None,
            )

        fake_model.generate_custom_voice = fake_generate_custom_voice
        service._model_registry.get_loaded_model = MagicMock(
            return_value=fake_model
        )

        def fake_decode_fn_builder(model):
            def decode(chunk_tensor):
                n_steps = (
                    chunk_tensor.shape[0]
                    if isinstance(chunk_tensor, torch.Tensor)
                    else len(chunk_tensor)
                )
                return np.full(n_steps * 50, 0.01, dtype=np.float32)
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )

        request = QwenTTSRequest(
            text="hello world",
            language="English",
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

            async def cancel_after_first_tokens():
                deadline = time.perf_counter() + 2.0
                while (
                    time.perf_counter() < deadline
                    and service._current_session_id is None
                ):
                    await asyncio.sleep(0.001)
                # Brief delay so the talker has fed the first 5 tokens
                # and reached the cancel-event wait.
                await asyncio.sleep(0.05)
                if service._current_session_id is not None:
                    service._session_registry.request_cancel(
                        service._current_session_id
                    )

            cancel_task = asyncio.create_task(cancel_after_first_tokens())
            try:
                response = await service._generate_true_stream(request)
            except Exception as exc:
                stop_evt.set()
                await drain_task
                await cancel_task
                raise AssertionError(
                    f"Talker thread raised through HF internals "
                    f"(D-11 invariant violated): {exc!r}"
                ) from exc
            finally:
                stop_evt.set()
                await drain_task
                await cancel_task
            return response

        response = asyncio.run(runner())
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)

        # Cancel actually propagated through the registry → cancel hook →
        # streamer._cancel_event (deterministic, not racy). Without this,
        # the post-cancel forward-call assertion below is meaningless —
        # the production forward-hook only skips codec_id accumulation
        # when streamer._cancel_event.is_set() returns True.
        assert captured_streamer, (
            "_build_true_stream_talker was never invoked — the spy never "
            "captured a streamer reference, so the cancel-event assertion "
            "below cannot run"
        )
        assert cancel_observed[0], (
            "Cancel never propagated to streamer._cancel_event within 1s. "
            "The cancel hook chain (request_cancel → registry hook → "
            "_cancel_event.set) is broken; D-11's cooperative cancellation "
            "invariant cannot hold and the post-cancel forward-call "
            "assertion below is meaningless"
        )
        # Wrapper fired exactly once.
        assert cv_call_count[0] == 1
        # Dispatch returned without raising; D-11 invariant preserved.
        assert response is not None
        # post-cancel forward calls went through but the forward-hook's
        # cancel check skipped codec_id accumulation (D-11). The dispatch
        # did not crash.
        assert post_cancel_forward_calls[0] == 5

class TestTtfaTalkerBoundaryInstrumentation:
    """Story 20.1 Task 2.1 — the three talker-side first-audio boundaries,
    exercised against the REAL ``_build_true_stream_talker`` body (this
    class does not monkeypatch it).

    These metrics are retained product surface by architect decision
    2026-08-31 because AC #2b Phase 3 (the deferred RTX 3060 confirmation)
    reads them out of the shipped CSV capture.
    """

    @staticmethod
    def _run_dispatch(qapp, service, monkeypatch, step_count, recorder):
        import torch
        from myvoice.services.qwen_tts_service import (
            QwenModelType, QwenTTSRequest,
        )

        fake_model, hits = _make_streamer_aware_fake_model(
            step_count=step_count, num_code_groups=4,
        )
        service._model_registry.get_loaded_model = MagicMock(
            return_value=fake_model
        )

        SAMPLES_PER_STEP = 100

        def fake_decode_fn_builder(model):
            def decode(chunk_tensor):
                if isinstance(chunk_tensor, torch.Tensor):
                    n_steps = chunk_tensor.shape[0]
                else:
                    n_steps = len(chunk_tensor)
                return np.full(
                    n_steps * SAMPLES_PER_STEP, 0.01, dtype=np.float32
                )
            return decode

        monkeypatch.setattr(
            service, "_build_true_stream_decode_fn", fake_decode_fn_builder
        )
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )

        request = QwenTTSRequest(
            text="hello world",
            language="English",
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
                return await service._generate_true_stream(request)
            finally:
                stop_evt.set()
                await drain_task

        response = asyncio.run(runner())
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.005)
        return response, hits

    def test_talker_boundaries_fire_once_each_in_order_on_threshold_path(
        self, qapp, registry, coordinator, monkeypatch,
    ):
        """step_count=100 crosses the first-emit threshold, so
        ``ttfa_first_chunk_emit_ms`` comes from the in-loop flush and is
        tagged ``path="threshold"``.

        100 steps clears the window at every geometry this project has
        shipped (30 pre-Story-20.4, 15 after), so the fixture stays valid
        across the retune; only the reported ``frames`` moves.
        """
        service, _ = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())
        recorder = _RecordingMetricsRecorder()

        response, _hits = self._run_dispatch(
            qapp, service, monkeypatch, 100, recorder,
        )
        assert response.success is True

        thread_start = recorder.calls_for("ttfa_talker_thread_start_ms")
        decode_step = recorder.calls_for("ttfa_first_decode_step_ms")
        chunk_emit = recorder.calls_for("ttfa_first_chunk_emit_ms")

        assert len(thread_start) == 1, (
            f"ttfa_talker_thread_start_ms must be one-shot; got "
            f"{len(thread_start)}"
        )
        assert len(decode_step) == 1, (
            "ttfa_first_decode_step_ms must fire once, on the first forward "
            f"that produced codec_ids; got {len(decode_step)}"
        )
        assert len(chunk_emit) == 1, (
            "ttfa_first_chunk_emit_ms must be one-shot even though the "
            "threshold is crossed repeatedly at step_count=100 "
            f"(window={_STREAMER_WINDOW}); got {len(chunk_emit)}"
        )

        # Prefill is exactly one forward call before codec_ids appear.
        assert decode_step[0][2]["prefill_forward_calls"] == 1

        # The in-loop flush path, and the window geometry it flushed.
        assert chunk_emit[0][2]["path"] == "threshold"
        # The window the in-loop flush actually emitted. Derived from the
        # streamer constants: a retune must move this number, and a retune
        # that does NOT move it means the dispatch path stopped reading the
        # streamer's geometry.
        assert chunk_emit[0][2]["frames"] == _STREAMER_WINDOW

        # Monotonic ordering: t0 <= thread start <= first decode step <=
        # first chunk emit. These are the segment boundaries; if they can
        # invert, every segment in the decomposition can go negative.
        gen_start = recorder.calls_for("ttfa_generation_start_ms")
        assert len(gen_start) == 1
        ordered = [
            gen_start[0][1],
            thread_start[0][1],
            decode_step[0][1],
            chunk_emit[0][1],
        ]
        assert ordered == sorted(ordered), (
            f"TTFA boundaries must be non-decreasing in wall-clock; got "
            f"{ordered}"
        )

    def test_residual_flush_path_emits_first_chunk_emit_boundary(
        self, qapp, registry, coordinator, monkeypatch,
    ):
        """A sub-threshold step count never reaches ``chunk_size +
        lookahead``, so the ONLY token chunk the generation ever produces is
        the terminal residual flush.

        This is the Story 20.1 short-utterance / Clear-Comms regime, and it
        is why the boundary is emitted from ``_flush_residual_and_eos`` as
        well as from the in-loop flush: without it, 6 of 11 short-utterance
        runs produced no measurable first-audio interval at all. Every
        other real-talker fixture in this file uses step_count > the window,
        so this branch is otherwise unexercised.

        Story 20.4: the step count is ``_STREAMER_WINDOW - 1``, derived. It
        was the literal 20, which stopped being sub-threshold the moment the
        window dropped from 30 to 15 -- the test then silently changed
        meaning (it would have exercised the THRESHOLD path while asserting
        residual-flush behaviour). Note that shrinking the window is exactly
        what moves REAL short utterances off this path: Story 20.1 SS5.3
        measured 11 of 20 short runs on residual_flush at chunk_size 25
        versus 0 of 5 at chunk_size 10. This row keeps the path covered
        anyway, because it is still reachable by anything shorter than
        1.25 s of speech.
        """
        service, _ = _build_true_stream_service(registry, coordinator)
        coordinator.play_dual_stream = AsyncMock(return_value=MagicMock())
        recorder = _RecordingMetricsRecorder()

        response, _hits = self._run_dispatch(
            qapp, service, monkeypatch, _SUB_THRESHOLD_STEPS, recorder,
        )
        assert response.success is True, (
            f"short-utterance dispatch failed: {response.error_message}"
        )
        assert response.chunks_generated == 1, (
            f"step_count={_SUB_THRESHOLD_STEPS} is one frame short of the "
            f"{_STREAMER_WINDOW}-frame window, so it must produce exactly "
            f"one chunk (the residual flush); got "
            f"{response.chunks_generated}"
        )

        chunk_emit = recorder.calls_for("ttfa_first_chunk_emit_ms")
        assert len(chunk_emit) == 1, (
            "ttfa_first_chunk_emit_ms must still fire exactly once when the "
            f"in-loop threshold is never reached; got {len(chunk_emit)}"
        )
        assert chunk_emit[0][2]["path"] == "residual_flush", (
            "the residual-flush emission must be distinguishable from the "
            "threshold emission by the ``path`` tag, or the short-utterance "
            "degeneration is invisible in the captured data"
        )
        assert chunk_emit[0][2]["frames"] == _SUB_THRESHOLD_STEPS, (
            "``frames`` must report the residual buffer depth so the "
            "evidence can tell how far short of the threshold the "
            f"utterance fell; got {chunk_emit[0][2]['frames']}"
        )
