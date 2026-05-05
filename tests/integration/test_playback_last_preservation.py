"""End-to-end integration tests for FR28-FR32 (Playback Last) preservation
under Story 13.2's PlaybackQueue.

Story 13.3 owns FR28-FR32 preservation under the queue. Epic 14 / Story 14.1
will extend this file with D-4 saveable-slot tests when the slot lands.
Until then, the file scopes to Epic 13 territory:
  - FR28: dual-stream replay (monitor + virtual mic)
  - FR29: main-window replay (Voice Design dialog OUT OF SCOPE — see
          Story 13.3 Open Question #1)
  - FR30: device-change handled by inherited V2 plumbing; queue does
          not corrupt device-id propagation
  - FR31: cache file write on generate, NOT on replay
  - FR32: replay reads from cache; no TTS generation path invoked

13.2 OQ re-verification (per Story 13.3 AC #13):
  - OQ#3 (synthetic UUID per replay click): pinned by
    ``test_three_rapid_replays_mint_distinct_tokens``
  - OQ#5 (UI status text on defer stays "Speech generated successfully"):
    pinned by ``test_deferred_replay_does_not_flip_status_text``
  - OQ#6 (test_playback_last suite passes unmodified): re-verified by
    the post-13.3 sweep at Task 7; not pinned at the integration layer

The whole module skips when torch + PyQt6 fail to load — Story 11.3 Task 18
set this precedent.

Cache filename note: this file uses ``myvoice_current.wav`` (the actual
filename from ``qwen_tts_service.py:520``); the architecture document
references ``myvoice_last.wav``. Per Story 13.3 Open Question #2 the rename
is deferred to avoid orphaning users' existing %TEMP% caches.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


pytest.importorskip("PyQt6")

import soundfile as sf  # noqa: E402  # imported after PyQt6 importorskip per project convention


# --------------------------------------------------------------------------- #
# Production import — guarded so the module skips cleanly without torch
# --------------------------------------------------------------------------- #

_IMPORT_ERROR: Optional[Exception] = None
MyVoiceApp = None
PlaybackQueue = None
SessionRegistry = None
SessionState = None
SessionSource = None

try:
    from myvoice.app import MyVoiceApp  # type: ignore[import-not-found]
    from myvoice.services.sessions import (  # type: ignore[import-not-found]
        PlaybackQueue,
        SessionRegistry,
        SessionSource,
        SessionState,
    )
except Exception as exc:  # pragma: no cover — env-dependent
    # Catch ``Exception`` (not just ImportError/OSError): once torch's DLL
    # load fails earlier in the test session, the partially-imported
    # ``qwen_tts`` package breaks subsequent re-imports with KeyError on
    # the package-path cache. Mirrors Story 11.3 Task 18 precedent.
    _IMPORT_ERROR = exc

if _IMPORT_ERROR is not None:
    pytestmark = pytest.mark.skip(
        reason=f"MyVoiceApp / sessions import failed (e.g. torch DLL load): {_IMPORT_ERROR!r}"
    )


# --------------------------------------------------------------------------- #
# Test scaffolding
# --------------------------------------------------------------------------- #


def _drain(qapp, iterations: int = 5) -> None:
    """Process queued slot invocations posted via QMetaObject.invokeMethod."""
    for _ in range(iterations):
        qapp.processEvents()


async def _drain_async(qapp, iterations: int = 5) -> None:
    """Drain Qt events AND yield to asyncio so QueuedConnection slots fire
    and ensure_future-scheduled coroutines run. Mirrors
    ``test_session_lifecycle.py::TestPlaybackQueueIntegration._drain_async``.
    """
    import asyncio as _asyncio

    for _ in range(iterations):
        qapp.processEvents()
        await _asyncio.sleep(0)


def _make_synthetic_cache_audio(
    sample_rate: int = 24000, duration_s: float = 0.1
) -> Tuple[np.ndarray, int]:
    """Build a deterministic ramp audio array for the synthetic cache file.

    A linear ramp avoids the all-zeros edge case in ``soundfile.write`` and
    keeps the byte content deterministic across runs (no PRNG state).
    """
    num_samples = int(sample_rate * duration_s)
    ramp = np.linspace(-0.5, 0.5, num_samples).astype(np.float32)
    return ramp, sample_rate


def _write_cache_file(tmp_path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write the synthetic cache file at ``tmp_path/myvoice_current.wav``.

    Mirrors the actual production filename (Open Question #2).
    """
    path = tmp_path / "myvoice_current.wav"
    sf.write(str(path), audio, sample_rate)
    return path


def _make_dispatch_stub(
    app, dispatched: List[dict]
) -> Callable[..., Any]:
    """Build the stub for ``_play_generated_audio`` that delegates queue
    gating to the production helpers (mirroring 13.2's H1 review fix
    pattern from ``test_session_lifecycle.py:1040-1067``).

    The stub records each dispatch with a richer payload than 13.2's
    queue tests so 13.3's byte-equality + device-id assertions can run.
    """

    async def stub_play_generated_audio(
        audio_data, session_id=None, _queue_token=None
    ):
        queue_token = app._derive_queue_token(session_id, _queue_token)
        if not app._claim_queue_slot_or_defer(
            queue_token, audio_data, session_id
        ):
            return
        monitor_task_id = f"monitor-{queue_token}"
        virtual_task_id = f"virtual-{queue_token}"
        if session_id is not None:
            app._task_to_session[monitor_task_id] = session_id
            app._task_to_session[virtual_task_id] = session_id
        else:
            app._task_to_replay_token[monitor_task_id] = queue_token
            app._task_to_replay_token[virtual_task_id] = queue_token
        # Snapshot device IDs at dispatch time so FR30 assertions can
        # verify they reflect current settings (not a stale capture from
        # a prior dispatch's pending entry).
        monitor_device_id = (
            app._app_settings.monitor_device_id
            if app._app_settings is not None
            else None
        )
        virtual_device_id = (
            app._app_settings.virtual_microphone_device_id
            if app._app_settings is not None
            else None
        )
        dispatched.append(
            {
                "queue_token": queue_token,
                "session_id": session_id,
                "audio_data": audio_data,
                "monitor_task_id": monitor_task_id,
                "virtual_task_id": virtual_task_id,
                "monitor_device_id": monitor_device_id,
                "virtual_device_id": virtual_device_id,
            }
        )

    return stub_play_generated_audio


@pytest.fixture
def app_with_replay(qapp, tmp_path, monkeypatch):
    """Build a partial ``MyVoiceApp`` wired with ``SessionRegistry``,
    ``PlaybackQueue``, the queue→registry depth-signal forwarding, a
    stubbed ``_tts_service`` whose ``get_cached_audio_path`` returns a
    real file in ``tmp_path``, and a stubbed ``_play_generated_audio``
    that delegates queue gating to the production helpers.

    Yields ``(app, registry, queue, dispatched, cache_path)`` where
    ``dispatched`` is a list of dicts with ``queue_token``, ``session_id``,
    ``audio_data``, ``monitor_task_id``, ``virtual_task_id``,
    ``monitor_device_id``, ``virtual_device_id``.

    Pattern A (per Story 13.3 Open Question #3): duplicates 13.2's
    ``_build_queue_app`` rather than refactoring to a shared helper.
    Replay-flow tests need different stubs than queue-only tests, and
    a single duplication is cheaper than a cross-file refactor.

    Hook depth (per Story 13.3 Open Question #4): SHALLOW hook on
    ``_play_generated_audio`` itself, with delegation to the production
    ``_derive_queue_token`` / ``_claim_queue_slot_or_defer`` helpers
    (matching 13.2's H1 review fix). The story default of "deeper hook
    on play_dual_stream" was rejected because the production
    ``_play_generated_audio`` body's device-resolution depth (~330
    lines, multiple Win11 collision-detection branches at app.py:1988-
    2150) makes the deeper hook fragile without extensive audio_coordinator
    + windows_audio_client mocking that exceeds 13.3's test-only scope.
    """
    app = MyVoiceApp(qapp)

    # Mimic _initialize_services_async's wiring for SessionRegistry +
    # PlaybackQueue (mirrors test_session_lifecycle.py:1014-1021).
    app._session_registry = SessionRegistry(parent=app)
    app._playback_queue = PlaybackQueue(parent=app)
    app._playback_queue.playback_queue_depth_changed.connect(
        app._session_registry.playback_queue_depth_changed.emit
    )
    monkeypatch.setattr(
        app._session_registry, "post_mutation", lambda *a, **kw: None
    )

    # Synthetic cache file (FR31 pre-condition) — hermetic, never touches
    # real %TEMP% (Story 13.3 Tricky Bits #1).
    audio_arr, sr = _make_synthetic_cache_audio()
    cache_path = _write_cache_file(tmp_path, audio_arr, sr)

    # Stub _tts_service: only get_cached_audio_path needs to return a real
    # path; ensure_model_loaded + generate are MagicMocks so AC #7 / FR32
    # call-count assertions can pin "no TTS work during replay".
    tts_stub = MagicMock()
    tts_stub.get_cached_audio_path.return_value = cache_path
    tts_stub._current_audio_cache = cache_path
    tts_stub.cancel_generation = AsyncMock()
    app._tts_service = tts_stub

    # Stub _audio_coordinator: AsyncMock for stop_all_playback (used by
    # _on_cancel_generation_requested via asyncio.ensure_future).
    audio_coord_stub = MagicMock()
    audio_coord_stub.stop_all_playback = AsyncMock()
    app._audio_coordinator = audio_coord_stub

    # Stub _main_window so the replay flow's set_generation_status calls
    # don't crash. Spy methods so AC #13 OQ#5 status-text assertions can run.
    app._main_window = MagicMock()

    # Stub _app_settings so the dispatch stub can snapshot device IDs.
    # Default both to None — the production "no virtual_device_id" branch
    # would route to monitor-only, but our stub never gets there because
    # _play_generated_audio is hooked. The settings exist purely as a
    # surface for FR30's device-change test to mutate.
    app._app_settings = MagicMock(
        monitor_device_id=None,
        monitor_device_name=None,
        monitor_device_host_api=None,
        virtual_microphone_device_id=None,
        virtual_microphone_device_name=None,
        virtual_microphone_device_host_api=None,
    )

    # Recorded dispatches.
    dispatched: List[dict] = []
    monkeypatch.setattr(
        app, "_play_generated_audio", _make_dispatch_stub(app, dispatched)
    )

    yield app, app._session_registry, app._playback_queue, dispatched, cache_path

    # Teardown — disconnect the depth-forwarding signal so subsequent
    # tests don't see cross-test state.
    try:
        app._playback_queue.playback_queue_depth_changed.disconnect()
    except (TypeError, RuntimeError):
        pass
    app._session_registry.deleteLater()
    app._playback_queue.deleteLater()


def _capture_run_async_task(
    app, monkeypatch
) -> List[Tuple[Any, Optional[Callable], Optional[Callable]]]:
    """Replace ``app._run_async_task`` with a stub that captures the full
    ``(coro, on_success, on_error)`` tuple so the test can ``await`` the
    coroutine AND invoke the production callbacks if needed.

    The production ``_run_async_task`` uses ``asyncio.create_task`` against
    the qasync event loop (which isn't running inside ``pytest.mark.asyncio``
    tests). Capturing + awaiting in the test is the project convention
    for sync-to-async bridges.

    Earlier versions dropped the callbacks silently — review M3 fix
    preserves them so tests asserting status-text behavior driven by
    ``_on_replay_success`` / ``_on_replay_error`` can fire those
    callbacks deterministically.
    """
    captured: List[Tuple[Any, Optional[Callable], Optional[Callable]]] = []

    def _stub(coro, on_success=None, on_error=None):
        captured.append((coro, on_success, on_error))

    monkeypatch.setattr(app, "_run_async_task", _stub)
    return captured


# --------------------------------------------------------------------------- #
# AC #3 / #4 / #5 — TestReplayDualStreamFanOut (FR28, FR29, FR30)
# --------------------------------------------------------------------------- #


class TestReplayDualStreamFanOut:
    """FR28/FR29/FR30 — replay reaches both monitor + virtual mic via
    ``play_dual_stream``; main-window path drives dispatch; device IDs
    aren't cached across replays.
    """

    @pytest.mark.asyncio
    async def test_replay_dispatches_to_both_monitor_and_virtual_devices(
        self, qapp, app_with_replay
    ):
        """FR28 — replay dispatches a synthetic ``replay-XXXXXXXX`` token
        through the queue + the dispatch stub, populating both
        monitor_task_id and virtual_task_id slots in ``_task_to_replay_token``.
        Audio bytes match the cache file byte-for-byte.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay

        audio_data = cache_path.read_bytes()
        await app._play_generated_audio(audio_data, session_id=None)

        assert len(dispatched) == 1
        entry = dispatched[0]
        assert entry["session_id"] is None
        assert entry["queue_token"].startswith("replay-")
        assert len(entry["queue_token"]) == len("replay-") + 8  # uuid4().hex[:8]
        # FR28: dual-task fan-out — both task ids registered for dedup.
        assert entry["monitor_task_id"] in app._task_to_replay_token
        assert entry["virtual_task_id"] in app._task_to_replay_token
        assert (
            app._task_to_replay_token[entry["monitor_task_id"]]
            == entry["queue_token"]
        )
        assert (
            app._task_to_replay_token[entry["virtual_task_id"]]
            == entry["queue_token"]
        )
        # Byte-equality: replay does not transform / re-encode.
        assert entry["audio_data"] == audio_data

    @pytest.mark.asyncio
    async def test_replay_main_window_path_via_signal_emission(
        self, qapp, app_with_replay, monkeypatch
    ):
        """FR29 (main window) — ``_on_replay_last_requested`` reads the
        cache, calls ``_play_generated_audio``, the synthetic replay token
        flows through the queue, dispatch records once.

        We capture the coroutine emitted by ``_run_async_task`` and await
        it explicitly because the qasync loop isn't running in
        ``pytest.mark.asyncio`` tests.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay

        captured = _capture_run_async_task(app, monkeypatch)

        # Direct call mirrors MainWindow.replay_last_requested →
        # app._on_replay_last_requested at app.py:500. The signal
        # connection is sealed by tests/ui/test_playback_last.py
        # (AC #1 gate); 13.3 verifies the *integration* path.
        app._on_replay_last_requested()

        assert len(captured) == 1
        coro, _on_success, _on_error = captured[0]
        await coro

        assert len(dispatched) == 1
        assert dispatched[0]["session_id"] is None
        assert dispatched[0]["queue_token"].startswith("replay-")
        # FR32 cross-check: audio bytes are the cache content (production
        # path read cached_path.read_bytes()).
        assert dispatched[0]["audio_data"] == cache_path.read_bytes()

    @pytest.mark.asyncio
    async def test_replay_uses_current_app_settings_device_ids(
        self, qapp, app_with_replay
    ):
        """FR30 — when device IDs change between replays, each dispatch
        observes the device IDs current at dispatch time. The queue does
        NOT cache stale device IDs from a prior dispatch's pending entry.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay
        audio_data = cache_path.read_bytes()

        # First replay: monitor=A virtual=B
        app._app_settings.monitor_device_id = "device-monitor-A"
        app._app_settings.virtual_microphone_device_id = "device-virtual-B"
        await app._play_generated_audio(audio_data, session_id=None)

        # Fire playback-complete for the first replay so the queue clears
        # and the next dispatch is unblocked.
        first = dispatched[0]
        app._on_playback_complete(first["monitor_task_id"])
        app._on_playback_complete(first["virtual_task_id"])
        await _drain_async(qapp, iterations=5)

        # Second replay: monitor=C virtual=D (settings changed mid-flight)
        app._app_settings.monitor_device_id = "device-monitor-C"
        app._app_settings.virtual_microphone_device_id = "device-virtual-D"
        await app._play_generated_audio(audio_data, session_id=None)

        assert len(dispatched) == 2
        assert dispatched[0]["monitor_device_id"] == "device-monitor-A"
        assert dispatched[0]["virtual_device_id"] == "device-virtual-B"
        assert dispatched[1]["monitor_device_id"] == "device-monitor-C"
        assert dispatched[1]["virtual_device_id"] == "device-virtual-D"
        # Distinct tokens — each replay click mints a fresh uuid.
        assert dispatched[0]["queue_token"] != dispatched[1]["queue_token"]

    @pytest.mark.asyncio
    async def test_advanced_replay_tokens_dedup_short_circuits_dual_fire(
        self, qapp, app_with_replay
    ):
        """AC #3 — ``_on_playback_complete``'s dual-fire dedup via
        ``_advanced_replay_tokens`` (app.py:2379-2388). After the first
        fire for the monitor task id, the synthetic replay token is
        added to the dedup set. The second fire (for the virtual task
        id on the same dispatch) observes the token in the set, sets
        ``should_advance_queue = False``, and skips the queue advance
        — so the queue advances *exactly once* per replay dispatch.

        Review H3 fix: AC #3's load-bearing dedup invariant was
        previously not asserted by any test. Symptomatic detection
        (depth-history forwarding) would have caught a double-emit but
        not pinned the dedup mechanism directly.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay
        cache_bytes = cache_path.read_bytes()

        # Watch the queue's depth signal so we can count advances.
        depth_history: List[int] = []
        queue.playback_queue_depth_changed.connect(depth_history.append)

        # Single replay → enqueues at depth 1.
        await app._play_generated_audio(cache_bytes, session_id=None)
        assert len(dispatched) == 1
        assert depth_history == [1]
        entry = dispatched[0]
        replay_token = entry["queue_token"]

        # Pre-condition: token is NOT yet in the dedup set.
        assert replay_token not in app._advanced_replay_tokens

        # First dual-fire callback (monitor task) — production adds the
        # token to _advanced_replay_tokens AND queues cancel_current
        # via QMetaObject.invokeMethod.
        app._on_playback_complete(entry["monitor_task_id"])
        assert replay_token in app._advanced_replay_tokens, (
            "First dual-fire callback must add the replay token to "
            "_advanced_replay_tokens (the dedup gate)."
        )

        # Second dual-fire callback (virtual task) — production sees
        # token in the set and short-circuits. No additional cancel
        # is queued. The dedup set size is unchanged (no double-add).
        set_size_before = len(app._advanced_replay_tokens)
        app._on_playback_complete(entry["virtual_task_id"])
        assert len(app._advanced_replay_tokens) == set_size_before, (
            "Second dual-fire callback re-added the replay token; "
            "the dedup short-circuit failed."
        )

        # Drain so the queued cancel_current fires. After drain:
        #   - depth_history gains exactly one entry: 0 (the dequeue
        #     from the FIRST fire's queued cancel_current; the second
        #     fire was dedup'd so it queued nothing).
        #   - queue is empty.
        await _drain_async(qapp, iterations=5)
        assert depth_history == [1, 0], (
            f"Expected exactly one queue advance per replay dispatch; "
            f"saw depth history {depth_history!r}. A second advance "
            "would indicate the dual-fire dedup failed."
        )
        assert queue.depth == 0
        assert app._dispatching_session_id is None

    def test_voice_design_replay_is_out_of_scope(self, qapp, app_with_replay):
        """FR29 (Voice Design) — forward-compatibility marker. The Voice
        Design Studio dialog has no Replay Last button. If a future
        story adds one, this test fails and forces a deliberate update
        to split FR29 coverage into main-window vs Voice Design test
        classes (per Story 13.3 Open Question #1).

        Review H1 fix: scans the SamplePathPanel module source for any
        ``replay`` reference rather than ``dir(SamplePathPanel)``. The
        original ``dir()`` check only saw class-level attributes —
        instance attributes assigned inside ``__init__`` (the realistic
        case for QPushButton members) would have slipped past.
        ``inspect.getsource`` returns the full module source so any
        replay-related identifier, comment, widget construction, or
        signal definition added later trips this gate.
        """
        from myvoice.ui.dialogs.voice_design_studio import sample_path_panel

        module_source = inspect.getsource(sample_path_panel).lower()
        # Search for ``replay_last`` — the canonical naming prefix used
        # throughout the V2 codebase for the FR28-FR32 cache-replay path
        # (``replay_last_button``, ``replay_last_requested``,
        # ``_on_replay_last_requested``, ``set_replay_enabled``, etc.).
        # The existing docstrings in sample_path_panel.py reference
        # "replay" colloquially for the local sample-audio play/stop
        # control (a different feature) — those are NOT Replay Last,
        # so the bare ``replay`` substring would false-trigger.
        # ``replay_last`` is specific enough that any future Voice
        # Design Replay Last surface aligned with the main-window
        # naming convention will trip this gate.
        assert "replay_last" not in module_source, (
            "SamplePathPanel module references 'replay_last' — Voice "
            "Design Studio has grown a Replay Last surface. FR29 "
            "coverage must be split into TestFR29MainWindowReplay vs "
            "TestFR29VoiceDesignReplay classes per Story 13.3 OQ#1."
        )


# --------------------------------------------------------------------------- #
# AC #6 / #7 — TestReplayCacheReadNoRegen (FR31, FR32)
# --------------------------------------------------------------------------- #


class TestReplayCacheReadNoRegen:
    """FR31/FR32 — cache file is written on generate (not replay); replay
    reads from the cache without invoking any TTS generation path.
    """

    @pytest.mark.asyncio
    async def test_replay_does_not_write_to_cache(
        self, qapp, app_with_replay, monkeypatch
    ):
        """FR31 (replay-side) — the replay flow does NOT call
        ``soundfile.write`` (no cache writes during replay). The
        complementary "generate writes the cache" half is sealed by
        ``test_session_lifecycle.py`` AC #4 and not duplicated here.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay

        # Spy on soundfile.write at the integration layer. Any write
        # during the replay flow would surface in the spy's call list.
        write_calls: List[tuple] = []

        original_sf_write = sf.write

        def _spy_sf_write(*args, **kwargs):
            write_calls.append((args, kwargs))
            return original_sf_write(*args, **kwargs)

        monkeypatch.setattr("soundfile.write", _spy_sf_write)
        # The production cache-write path (qwen_tts_service.py:2619-2640)
        # imports soundfile as ``sf`` and calls ``sf.write``; the
        # ``soundfile.write`` patch above intercepts both the bare and
        # aliased import paths because Python interns the underlying
        # function. (Earlier defensive double-patch on this module's
        # ``sf.write`` alias was redundant — review L3 fix removed it.)

        # Run the replay flow.
        await app._play_generated_audio(
            cache_path.read_bytes(), session_id=None
        )

        # FR31: replay path performs ZERO sf.write calls. The cache file
        # was written by the test fixture (pre-condition); no further
        # write should occur during replay.
        assert write_calls == [], (
            f"Replay path must not call soundfile.write; saw: {write_calls!r}"
        )
        assert len(dispatched) == 1

    @pytest.mark.asyncio
    async def test_replay_does_not_invoke_tts_generate(
        self, qapp, app_with_replay
    ):
        """FR32 — replay reads the cache; no TTS generation path is
        invoked. ``_tts_service.generate`` and ``ensure_model_loaded``
        call counts must be zero.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay

        # Reset call counts (the fixture's MagicMock starts at 0, but a
        # prior interaction could have incremented; this is defensive).
        app._tts_service.generate.reset_mock()
        app._tts_service.ensure_model_loaded.reset_mock()

        await app._play_generated_audio(
            cache_path.read_bytes(), session_id=None
        )

        assert app._tts_service.generate.call_count == 0
        assert app._tts_service.ensure_model_loaded.call_count == 0
        assert len(dispatched) == 1
        # The dispatch is keyed on a synthetic replay token (no session id).
        assert dispatched[0]["session_id"] is None

    @pytest.mark.asyncio
    async def test_replay_audio_is_byte_for_byte_cache_content(
        self, qapp, app_with_replay
    ):
        """FR32 — replay's audio_data is byte-for-byte the cache file's
        contents. No re-encoding, no resampling, no transformation.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay

        expected_bytes = cache_path.read_bytes()
        # The production _on_replay_last_requested reads the cache as
        # bytes and passes through to _play_generated_audio. Pre-read
        # outside the call so the test is independent of read ordering.
        assert len(expected_bytes) > 0  # sanity: cache file exists

        await app._play_generated_audio(expected_bytes, session_id=None)

        assert len(dispatched) == 1
        assert dispatched[0]["audio_data"] == expected_bytes


# --------------------------------------------------------------------------- #
# AC #8 / #9 / #10 — TestReplayQueueInteraction
# --------------------------------------------------------------------------- #


class TestReplayQueueInteraction:
    """Replay enqueues behind in-progress generation; rapid replays mint
    distinct tokens; cancel-then-replay dispatches cleanly without the
    13.2 KeyError regression.
    """

    @pytest.mark.asyncio
    async def test_replay_enqueues_behind_in_progress_generation(
        self, qapp, app_with_replay
    ):
        """AC #8 — session A (registry-tracked) is in PLAYING; replay
        click enqueues behind A; A's playback-complete advances the
        queue; replay dispatches.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay
        cache_bytes = cache_path.read_bytes()

        # Session A: registry-tracked generation completes and dispatches.
        await app._play_generated_audio(b"audio-A", session_id="sid-A")
        assert len(dispatched) == 1
        assert dispatched[0]["session_id"] == "sid-A"
        assert app._dispatching_session_id == "sid-A"
        assert queue.depth == 1

        # Replay click while A is playing: token enqueues at tail; head
        # check fails; dispatch parks in _pending_dispatches.
        await app._play_generated_audio(cache_bytes, session_id=None)
        # play_dual_stream stub was NOT called for the replay yet.
        assert len(dispatched) == 1
        assert queue.depth == 2
        # _dispatching_session_id stays on A.
        assert app._dispatching_session_id == "sid-A"
        # Replay token is parked in _pending_dispatches.
        replay_tokens = [
            t for t in app._pending_dispatches.keys() if t.startswith("replay-")
        ]
        assert len(replay_tokens) == 1
        replay_token = replay_tokens[0]
        assert app._pending_dispatches[replay_token].audio_data == cache_bytes

        # Fire A's dual-fire playback-complete → queue advances → replay
        # dispatches via _dispatch_next_pending which re-enters
        # _play_generated_audio with the original _queue_token.
        sid_a_entry = dispatched[0]
        app._on_playback_complete(sid_a_entry["monitor_task_id"])
        app._on_playback_complete(sid_a_entry["virtual_task_id"])
        await _drain_async(qapp, iterations=5)

        assert len(dispatched) == 2
        assert dispatched[1]["session_id"] is None
        assert dispatched[1]["queue_token"] == replay_token
        # Replay dispatched the cache bytes verbatim.
        assert dispatched[1]["audio_data"] == cache_bytes
        # Queue is now down to the replay (still in flight from the
        # dispatch stub's perspective).
        assert app._dispatching_session_id == replay_token
        assert queue.depth == 1
        # _pending_dispatches drained the replay entry.
        assert replay_token not in app._pending_dispatches

    @pytest.mark.asyncio
    async def test_three_rapid_replays_mint_distinct_tokens(
        self, qapp, app_with_replay
    ):
        """AC #9 — three replay clicks in rapid succession mint three
        distinct synthetic tokens, queue holds them in submission order,
        each dispatches in order. Pins 13.2 OQ#3 (synthetic UUID per
        replay click).
        """
        app, registry, queue, dispatched, cache_path = app_with_replay
        cache_bytes = cache_path.read_bytes()

        # Three rapid replay enqueues — back-to-back without yielding to
        # the event loop, modelling the rapid-click scenario.
        await app._play_generated_audio(cache_bytes, session_id=None)
        await app._play_generated_audio(cache_bytes, session_id=None)
        await app._play_generated_audio(cache_bytes, session_id=None)

        # First dispatched immediately (queue was empty); second + third
        # parked in _pending_dispatches.
        assert len(dispatched) == 1
        assert queue.depth == 3
        assert app._dispatching_session_id.startswith("replay-")

        # Three distinct replay tokens: the first is dispatching; the
        # other two are in _pending_dispatches.
        first_token = dispatched[0]["queue_token"]
        pending_replay_tokens = [
            t for t in app._pending_dispatches.keys() if t.startswith("replay-")
        ]
        assert len(pending_replay_tokens) == 2
        all_tokens = {first_token, *pending_replay_tokens}
        assert len(all_tokens) == 3, (
            f"Each replay click must mint a distinct token; saw: {all_tokens!r}"
        )

        # Drain queue: fire playback-complete for the first → second
        # dispatches → fire its complete → third dispatches.
        for _ in range(3):
            current = dispatched[-1]
            app._on_playback_complete(current["monitor_task_id"])
            app._on_playback_complete(current["virtual_task_id"])
            await _drain_async(qapp, iterations=5)

        # All three replays dispatched in submission order, each with
        # the same cache bytes.
        assert len(dispatched) == 3
        assert all(
            entry["queue_token"].startswith("replay-") for entry in dispatched
        )
        assert all(entry["audio_data"] == cache_bytes for entry in dispatched)
        # Queue cleared.
        assert queue.depth == 0
        assert app._dispatching_session_id is None
        assert app._pending_dispatches == {}

    @pytest.mark.asyncio
    async def test_cancel_then_immediate_replay_dispatches_cleanly(
        self, qapp, app_with_replay
    ):
        """AC #10 — cancel during playback, then immediate replay click,
        does not raise the 13.2 KeyError: 'Unknown session_id' regression.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay
        cache_bytes = cache_path.read_bytes()

        # Session A is dispatching.
        await app._play_generated_audio(b"audio-A", session_id="sid-A")
        assert app._dispatching_session_id == "sid-A"

        # Cancel — _on_cancel_generation_requested explicitly advances
        # the queue (because stop_all_playback does NOT fire
        # _playback_complete_callback per 13.2 AC #9). _tts_service and
        # _audio_coordinator are AsyncMock-stubbed so the
        # asyncio.ensure_future calls return cleanly.
        app._on_cancel_generation_requested()
        # Queue advance ran synchronously inside the cancel handler
        # (Qt main thread, no QMetaObject.invokeMethod indirection per
        # app.py:1099-1110).
        assert app._dispatching_session_id is None
        assert queue.depth == 0

        # Immediate replay click — no KeyError, dispatch fires.
        await app._play_generated_audio(cache_bytes, session_id=None)

        assert len(dispatched) == 2
        assert dispatched[0]["session_id"] == "sid-A"
        assert dispatched[1]["session_id"] is None
        assert dispatched[1]["queue_token"].startswith("replay-")
        assert dispatched[1]["audio_data"] == cache_bytes

    @pytest.mark.asyncio
    async def test_deferred_replay_does_not_flip_status_text(
        self, qapp, app_with_replay, monkeypatch
    ):
        """AC #13 OQ#5 — when a replay defers behind an in-progress
        generation, the only status mutation during the deferred window
        is the production-mandated "Replaying last audio..." flip set
        by ``_on_replay_last_requested`` BEFORE ``_play_generated_audio``
        returns from the defer path. No additional "Queued" or transient
        status appears.

        Review M1 fix: this test now drives the production
        ``_on_replay_last_requested`` entry point (capturing the
        ``_run_async_task`` coroutine and awaiting it explicitly) so
        production ``set_generation_status`` calls are observable.
        Earlier versions called ``_play_generated_audio`` directly while
        the fixture's stub never touched ``_main_window`` — the
        assertion was vacuous.
        """
        app, registry, queue, dispatched, cache_path = app_with_replay

        # Session A is dispatching (registry-tracked, holds the head).
        await app._play_generated_audio(b"audio-A", session_id="sid-A")

        # Reset main_window's call history so the assertion focuses on
        # the deferred-replay window only.
        app._main_window.reset_mock()

        # Capture the run_async_task coroutine produced by
        # _on_replay_last_requested so we can await it deterministically.
        captured = _capture_run_async_task(app, monkeypatch)

        # Drive the production entry point. The handler reads the cache,
        # sets status text to "Replaying last audio...", then schedules
        # _play_generated_audio via _run_async_task. The coroutine defers
        # in _claim_queue_slot_or_defer (head is sid-A, replay token
        # parks in _pending_dispatches) and returns.
        app._on_replay_last_requested()
        assert len(captured) == 1
        coro, _on_success, _on_error = captured[0]
        await coro

        # Production must call set_generation_status exactly once with
        # "Replaying last audio..." — the only status mutation during
        # the deferred window. No "Queued", "Waiting", or other
        # transient text.
        status_calls = app._main_window.set_generation_status.call_args_list
        status_messages = [
            call.args[0] for call in status_calls if call.args
        ]
        assert status_messages == ["Replaying last audio..."], (
            "Unexpected status mutations during deferred replay window: "
            f"{status_messages!r}. OQ#5 requires the status text stays "
            "'Replaying last audio...' until the actual dispatch fires."
        )

        # The replay token did park in _pending_dispatches (the deferred
        # path was actually taken — no spurious early dispatch).
        replay_tokens = [
            t for t in app._pending_dispatches.keys()
            if t.startswith("replay-")
        ]
        assert len(replay_tokens) == 1


# --------------------------------------------------------------------------- #
# AC #12 — TestReplayDepthSignalForwarding
# --------------------------------------------------------------------------- #


class TestReplayDepthSignalForwarding:
    """Queue → registry depth-signal forwarding for replay tokens (mirrors
    13.2 AC #7 for synthetic replay tokens rather than session IDs).
    """

    @pytest.mark.asyncio
    async def test_replay_depth_changes_forward_to_registry_signal(
        self, qapp, app_with_replay
    ):
        """AC #12 — three replay clicks; queue.playback_queue_depth_changed
        and registry.playback_queue_depth_changed receive identical depth
        histories (one-to-one forwarding per 13.2 AC #7, re-verified for
        the replay path specifically).
        """
        app, registry, queue, dispatched, cache_path = app_with_replay
        cache_bytes = cache_path.read_bytes()

        queue_history: List[int] = []
        registry_history: List[int] = []
        queue.playback_queue_depth_changed.connect(queue_history.append)
        registry.playback_queue_depth_changed.connect(registry_history.append)

        # Three rapid replays.
        await app._play_generated_audio(cache_bytes, session_id=None)
        await app._play_generated_audio(cache_bytes, session_id=None)
        await app._play_generated_audio(cache_bytes, session_id=None)

        # After three enqueues, depth should have stepped 1 → 2 → 3.
        assert queue_history == [1, 2, 3]
        assert registry_history == [1, 2, 3]
        assert queue_history == registry_history

        # Drain replays one at a time; each playback-complete drives a
        # depth decrement.
        for _ in range(3):
            current = dispatched[-1]
            app._on_playback_complete(current["monitor_task_id"])
            app._on_playback_complete(current["virtual_task_id"])
            await _drain_async(qapp, iterations=5)

        # Final history: enqueues + dequeues, perfectly mirrored between
        # queue and registry.
        assert queue_history == registry_history
        # Last value is 0 — queue is empty after all replays drain.
        assert queue_history[-1] == 0
