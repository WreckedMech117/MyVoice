"""End-to-end integration tests for Story 15.2 (Clear Comms dispatch).

Exercises the click → app slot → dispatch chain for the Clear Comms
button. Covers AC #7 (resolution + dispatch), AC #8 (interrupt path),
AC #9 (queue path), AC #10 (D-5 invariant), AC #11 (file-source error
paths). Pattern A from Story 13.3 (Open Question #3) — duplicates the
``_make_dispatch_stub`` shape rather than refactoring to a shared
helper, because Clear Comms's resolve-then-dispatch chain is different
enough from Replay Last's cache-read-then-dispatch chain that a unified
fixture would cost more than it saves.

The whole module skips when torch + PyQt6 fail to load — Story 11.3
Task 18 set this precedent.
"""

from __future__ import annotations

import io
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
_ClearCommsResolveError = None

try:
    from myvoice.app import MyVoiceApp, _ClearCommsResolveError  # type: ignore[import-not-found]
    from myvoice.services.sessions import (  # type: ignore[import-not-found]
        PlaybackQueue,
        SessionRegistry,
        SessionSource,
        SessionState,
    )
except Exception as exc:  # pragma: no cover — env-dependent
    _IMPORT_ERROR = exc

if _IMPORT_ERROR is not None:
    pytestmark = pytest.mark.skip(
        reason=f"MyVoiceApp / sessions import failed (e.g. torch DLL load): {_IMPORT_ERROR!r}"
    )


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #


def _drain(qapp, iterations: int = 5) -> None:
    for _ in range(iterations):
        qapp.processEvents()


def _make_synthetic_saveable_audio(
    duration_s: float = 0.05, sample_rate: int = 24000
) -> Tuple[np.ndarray, int]:
    n = int(sample_rate * duration_s)
    ramp = np.linspace(-0.5, 0.5, n).astype(np.float32)
    return ramp, sample_rate


def _make_dispatch_stub(app, dispatched: List[dict]) -> Callable[..., Any]:
    """Mirror of test_playback_last_preservation.py's stub — delegates
    queue gating to the production helpers and records each dispatch.
    """

    async def stub_play_generated_audio(
        audio_data, session_id=None, _queue_token=None
    ):
        queue_token = app._derive_queue_token(session_id, _queue_token)
        if not app._claim_queue_slot_or_defer(
            queue_token, audio_data, session_id
        ):
            return
        dispatched.append(
            {
                "queue_token": queue_token,
                "session_id": session_id,
                "audio_data": audio_data,
            }
        )

    return stub_play_generated_audio


@pytest.fixture
def app_with_clear_comms(qapp, tmp_path, monkeypatch):
    """Build a partial ``MyVoiceApp`` wired with ``SessionRegistry``,
    ``PlaybackQueue``, mocked ``AudioCoordinator``, mocked
    ``TTSService``, mocked main_window, and a stubbed
    ``_play_generated_audio`` that delegates queue gating to the
    production helpers.

    Yields ``(app, registry, queue, dispatched)``.
    """
    app = MyVoiceApp(qapp)

    app._session_registry = SessionRegistry(parent=app)
    app._playback_queue = PlaybackQueue(parent=app)
    app._playback_queue.playback_queue_depth_changed.connect(
        app._session_registry.playback_queue_depth_changed.emit
    )

    audio_coord_stub = MagicMock()
    audio_coord_stub.stop_all_playback = AsyncMock()
    app._audio_coordinator = audio_coord_stub

    tts_stub = MagicMock()
    tts_stub.cancel_generation = AsyncMock()
    app._tts_service = tts_stub

    app._main_window = MagicMock()

    # AppSettings stub — set fields directly per test as needed.
    app._app_settings = MagicMock(
        clear_comms_source_kind="last_generation",
        clear_comms_file_path=None,
        clear_comms_queue_mode=False,
    )

    dispatched: List[dict] = []
    monkeypatch.setattr(
        app, "_play_generated_audio", _make_dispatch_stub(app, dispatched)
    )
    # Replace _run_async_task with a synchronous bridge: call the coro to
    # produce the dispatch effect immediately. The stub's coroutine runs
    # to completion under qasync in production; in tests we drive it
    # synchronously since there is no qasync loop.
    def _sync_run(coro, on_success=None, on_error=None):
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(coro)
        except RuntimeError:
            asyncio.new_event_loop().run_until_complete(coro)
    monkeypatch.setattr(app, "_run_async_task", _sync_run)

    yield app, app._session_registry, app._playback_queue, dispatched

    # Teardown
    try:
        app._playback_queue.playback_queue_depth_changed.disconnect()
    except (TypeError, RuntimeError):
        pass
    app._session_registry.deleteLater()
    app._playback_queue.deleteLater()


def _finalize_session_with_audio(
    registry, audio: np.ndarray, sample_rate: int = 24000, text: str = "hello"
) -> str:
    """Drive a session through PENDING → GENERATING → READY_TO_PLAY so
    its saveable slot is promoted by the registry's natural lifecycle.
    """
    sid = registry.create_session(
        text=text, voice="v", model_type="m", source=SessionSource.GENERATED,
    )
    # Override sample_rate by post-init mutation — the registry stores
    # whatever the session reports.
    object.__setattr__(registry.get(sid), "sample_rate", sample_rate)
    registry.start_generation(sid)
    registry.append_chunk(sid, audio)
    registry.finalize(sid)
    return sid


def _decode_wav_bytes(buf: bytes) -> Tuple[np.ndarray, int]:
    """Read int16 WAV bytes back into a float32 array + sample rate."""
    with io.BytesIO(buf) as bio:
        arr, sr = sf.read(bio, dtype="float32", always_2d=False)
    return arr, sr


# --------------------------------------------------------------------------- #
# AC #7 — resolution + dispatch (last_generation branch)
# --------------------------------------------------------------------------- #


class TestClearCommsLastGenerationDispatch:
    """AC #7: ``source_kind == "last_generation"`` → encode saveable to
    WAV bytes and dispatch via the existing replay path."""

    def test_dispatches_saveable_audio_in_interrupt_mode(self, app_with_clear_comms):
        app, registry, _queue, dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        _finalize_session_with_audio(registry, audio, sample_rate=sr)

        app._app_settings.clear_comms_source_kind = "last_generation"
        app._app_settings.clear_comms_queue_mode = False
        app._on_clear_comms_requested()

        assert len(dispatched) == 1
        assert dispatched[0]["session_id"] is None  # registry-less dispatch
        assert dispatched[0]["queue_token"].startswith("replay-")
        # Audio bytes round-trip back to (approximately) the saveable buffer.
        roundtrip, sr_back = _decode_wav_bytes(dispatched[0]["audio_data"])
        assert sr_back == sr
        assert np.allclose(roundtrip, audio, atol=1e-3)

    def test_interrupt_mode_calls_stop_all_playback(self, app_with_clear_comms):
        app, registry, _queue, _dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        _finalize_session_with_audio(registry, audio, sample_rate=sr)

        app._app_settings.clear_comms_queue_mode = False
        app._on_clear_comms_requested()
        # AudioCoordinator.stop_all_playback was scheduled by the
        # interrupt helper. AsyncMock counts both call() and call(...)
        # as 1 invocation.
        assert app._audio_coordinator.stop_all_playback.call_count >= 1

    def test_queue_mode_skips_interrupt_helper(self, app_with_clear_comms):
        """AC #9 — queue mode does NOT call stop_all_playback; the
        existing _claim_queue_slot_or_defer machinery handles deferral.
        """
        app, registry, _queue, dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        _finalize_session_with_audio(registry, audio, sample_rate=sr)

        app._app_settings.clear_comms_queue_mode = True
        app._on_clear_comms_requested()

        # The dispatch still happens (queue is empty), and stop_all_playback
        # was NOT awaited.
        assert len(dispatched) == 1
        assert app._audio_coordinator.stop_all_playback.call_count == 0

    def test_interrupt_mode_clear_comms_claims_freed_slot_not_pending(
        self, app_with_clear_comms
    ):
        """Story 15.2 review fix H3 regression test (added by follow-up
        review). The H3 fix removed ``_dispatch_next_pending()`` from
        ``_interrupt_active_playback_for_clear_comms`` so a parked
        session-B cannot grab the freed queue slot before Step 3's
        Clear Comms dispatch arrives. Without this test, a future
        refactor that re-introduces ``_dispatch_next_pending()`` (the
        symmetric Stop-button helper still calls it, so the asymmetry
        is the load-bearing detail) would silently regress D-18
        "interrupt by default" semantics.

        Setup: pre-occupy the queue head AND park a session-B replay
        token in ``_pending_dispatches``. Then click Clear Comms in
        interrupt mode. Expectation: exactly ONE new dispatch fires
        (the Clear Comms WAV bytes), session-B stays parked, and the
        cancelled occupant token is not re-dispatched.
        """
        app, registry, _queue, dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        _finalize_session_with_audio(registry, audio, sample_rate=sr)

        # Pre-occupy the queue head with a synthetic replay token so the
        # interrupt helper has something to vacate.
        from myvoice.app import _PendingDispatch  # type: ignore[import-not-found]

        app._dispatching_session_id = "replay-occupant"
        # Park a queued session-B behind it. ``audio_data`` is just a
        # sentinel bytestring — _dispatch_next_pending would re-enter
        # _play_generated_audio with it, but the H3 fix prevents that
        # from happening here.
        parked_b_token = "replay-parked-b"
        parked_b_audio = b"PARKED_B_BYTES"
        app._pending_dispatches[parked_b_token] = _PendingDispatch(
            audio_data=parked_b_audio,
            session_id=None,
            queue_token=parked_b_token,
        )

        app._app_settings.clear_comms_source_kind = "last_generation"
        app._app_settings.clear_comms_queue_mode = False  # interrupt
        app._on_clear_comms_requested()

        # Exactly ONE dispatch fired — the Clear Comms one. If the
        # interrupt helper had called _dispatch_next_pending(), the
        # parked session-B would have dispatched first (or also)
        # before/instead of Clear Comms.
        assert len(dispatched) == 1, (
            f"Expected 1 dispatch (Clear Comms only), got {len(dispatched)}: "
            f"{[d['queue_token'] for d in dispatched]}"
        )
        # And it must be Clear Comms, not session-B.
        assert dispatched[0]["audio_data"] != parked_b_audio, (
            "Parked session-B's bytes were dispatched instead of Clear Comms's "
            "— H3 regression: the interrupt helper grabbed _dispatch_next_pending."
        )
        assert dispatched[0]["queue_token"] != parked_b_token
        # Session-B stays parked.
        assert parked_b_token in app._pending_dispatches, (
            "Parked session-B disappeared from _pending_dispatches — "
            "H3 regression: the interrupt helper popped it via "
            "_dispatch_next_pending."
        )
        # The cancelled occupant token was registered for cleanup so a
        # late playback-complete signal doesn't re-dispatch it.
        assert "replay-occupant" in app._advanced_replay_tokens

    def test_queue_mode_parks_behind_occupied_head(self, app_with_clear_comms):
        """AC #9 (Story 15.2 review fix M3) — when the queue head is
        already occupied, queue-mode Clear Comms parks itself in
        ``_pending_dispatches`` instead of dispatching immediately. This
        is the path AC #9 promises ("the new replay token sees the queue
        head is occupied, parks the dispatch in _pending_dispatches");
        without this test it was never exercised.
        """
        app, registry, _queue, dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        _finalize_session_with_audio(registry, audio, sample_rate=sr)

        # Pre-occupy the queue head with a synthetic dispatch so the
        # Clear Comms click hits an occupied slot.
        app._dispatching_session_id = "fake-occupant-token"

        app._app_settings.clear_comms_queue_mode = True
        app._on_clear_comms_requested()

        # The Clear Comms dispatch did NOT fire through the stub (parked).
        assert dispatched == []
        # Exactly one new pending dispatch keyed by the synthetic
        # replay-XXXX token Clear Comms minted.
        new_pending = [
            tok for tok in app._pending_dispatches if tok.startswith("replay-")
        ]
        assert len(new_pending) == 1
        # And the interrupt helper was still skipped (queue mode).
        assert app._audio_coordinator.stop_all_playback.call_count == 0


# --------------------------------------------------------------------------- #
# AC #7 / AC #11 — file-source branch + error surfaces
# --------------------------------------------------------------------------- #


class TestClearCommsFileSource:
    """AC #7 file branch + AC #11 error paths."""

    def test_dispatches_loaded_file_audio(self, app_with_clear_comms, tmp_path):
        app, _registry, _queue, dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        wav_path = tmp_path / "clear_comms.wav"
        sf.write(str(wav_path), audio, sr)

        app._app_settings.clear_comms_source_kind = "file"
        app._app_settings.clear_comms_file_path = str(wav_path)
        app._app_settings.clear_comms_queue_mode = False
        app._on_clear_comms_requested()

        assert len(dispatched) == 1
        roundtrip, sr_back = _decode_wav_bytes(dispatched[0]["audio_data"])
        assert sr_back == sr
        assert np.allclose(roundtrip, audio, atol=1e-3)

    def test_missing_file_surfaces_toast_and_does_not_dispatch(
        self, app_with_clear_comms, tmp_path
    ):
        app, _registry, _queue, dispatched = app_with_clear_comms
        app._app_settings.clear_comms_source_kind = "file"
        app._app_settings.clear_comms_file_path = str(tmp_path / "does_not_exist.wav")
        app._on_clear_comms_requested()

        assert dispatched == []
        # The toast surface received a "File not found" message
        # (loader's contract from Story 15.1).
        app._main_window.set_generation_status.assert_called_once()
        toast_args = app._main_window.set_generation_status.call_args
        assert "File not found" in toast_args[0][0]

    def test_unconfigured_file_path_surfaces_toast(self, app_with_clear_comms):
        app, _registry, _queue, dispatched = app_with_clear_comms
        app._app_settings.clear_comms_source_kind = "file"
        app._app_settings.clear_comms_file_path = None
        app._on_clear_comms_requested()

        assert dispatched == []
        app._main_window.set_generation_status.assert_called_once()
        msg = app._main_window.set_generation_status.call_args[0][0]
        assert "No Clear Comms file" in msg

    def test_no_saveable_for_last_generation_surfaces_toast(self, app_with_clear_comms):
        """AC #11 third case — saveable was cancelled between enable
        check and click."""
        app, _registry, _queue, dispatched = app_with_clear_comms
        # No saveable session has been registered; saveable_audio is None.
        app._app_settings.clear_comms_source_kind = "last_generation"
        app._on_clear_comms_requested()

        assert dispatched == []
        app._main_window.set_generation_status.assert_called_once()
        msg = app._main_window.set_generation_status.call_args[0][0]
        # Wording matches save_dialog._TOAST_NO_SAVEABLE for symmetry.
        assert "No saveable audio" in msg


# --------------------------------------------------------------------------- #
# Story 15.3 H3 — Test Playback shim end-to-end (panel-supplied vs persisted)
# --------------------------------------------------------------------------- #


class TestClearCommsTestPlaybackShim:
    """Story 15.3 review fix H3 — pin the ``_PanelSettingsShim`` contract:
    ``MyVoiceApp._on_clear_comms_test_playback_requested`` must use the
    panel-supplied ``(source_kind, file_path, queue_mode)`` rather than
    ``self._app_settings.*``. Without these tests, a future refactor of
    the slot that reads ``settings.clear_comms_source_kind`` (which the
    shim doesn't expose) would silently break Test Playback.
    """

    def test_test_playback_uses_panel_file_path_not_persisted(
        self, app_with_clear_comms, tmp_path
    ):
        """The shim's getattr contract — ``clear_comms_file_path`` is the
        only attribute the slot reads from the shim. Even if the
        persisted ``_app_settings.clear_comms_file_path`` points elsewhere,
        Test Playback dispatches the *panel-supplied* file's bytes.
        """
        app, _registry, _queue, dispatched = app_with_clear_comms

        # Persisted config: source = file, path = persisted.wav (audio P).
        audio_p = (np.sin(np.linspace(0, 6, 1200)) * 0.3).astype(np.float32)
        persisted_wav = tmp_path / "persisted.wav"
        sf.write(str(persisted_wav), audio_p, 24000)
        app._app_settings.clear_comms_source_kind = "file"
        app._app_settings.clear_comms_file_path = str(persisted_wav)

        # Panel-supplied (preview) config: a *different* file (audio T).
        audio_t = (np.cos(np.linspace(0, 5, 1200)) * 0.3).astype(np.float32)
        preview_wav = tmp_path / "preview.wav"
        sf.write(str(preview_wav), audio_t, 24000)

        app._on_clear_comms_test_playback_requested(
            "file", str(preview_wav), False
        )

        assert len(dispatched) == 1
        roundtrip, _sr = _decode_wav_bytes(dispatched[0]["audio_data"])
        # The dispatched bytes are the preview file's, NOT the persisted one's.
        assert np.allclose(roundtrip, audio_t, atol=1e-3), (
            "Test Playback dispatched persisted bytes instead of panel-supplied — "
            "_PanelSettingsShim contract regression."
        )
        assert not np.allclose(roundtrip, audio_p, atol=1e-2)

    def test_test_playback_never_calls_stop_all_playback(
        self, app_with_clear_comms
    ):
        """AC #5 divergence — Test Playback NEVER interrupts, even with
        ``queue_mode=False``. ``queue_mode`` is plumbed for future
        extension but is informational in v1.
        """
        app, registry, _queue, _dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        _finalize_session_with_audio(registry, audio, sample_rate=sr)

        # queue_mode=False — would interrupt for a real Clear Comms click.
        app._on_clear_comms_test_playback_requested(
            "last_generation", None, False
        )

        assert app._audio_coordinator.stop_all_playback.call_count == 0, (
            "Test Playback called stop_all_playback — it must NEVER "
            "interrupt (it's a preview, not a callout)."
        )

    def test_test_playback_resolve_error_surfaces_toast(
        self, app_with_clear_comms, tmp_path
    ):
        """A bad file path supplied by the panel surfaces the loader's
        message via ``set_generation_status`` — same toast surface as
        the real Clear Comms click."""
        app, _registry, _queue, dispatched = app_with_clear_comms

        bad_path = str(tmp_path / "vanished.wav")
        app._on_clear_comms_test_playback_requested("file", bad_path, False)

        assert dispatched == []
        app._main_window.set_generation_status.assert_called_once()
        msg = app._main_window.set_generation_status.call_args[0][0]
        assert "File not found" in msg


# --------------------------------------------------------------------------- #
# AC #10 — D-5 invariant: Clear Comms NEVER advances the saveable slot
# --------------------------------------------------------------------------- #


class TestClearCommsDoesNotAdvanceSaveable:
    """AC #10 — saveable_session_id remains stable across a Clear Comms
    click (interrupt or queue mode); saveable_session_changed is NOT
    emitted as a side effect of the click."""

    def test_saveable_session_id_unchanged_after_clear_comms_interrupt(
        self, qapp, app_with_clear_comms
    ):
        app, registry, _queue, dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        sid = _finalize_session_with_audio(registry, audio, sample_rate=sr)
        _drain(qapp)
        assert registry.saveable_session_id == sid

        # Capture saveable_session_changed emissions during the click.
        emissions: List[Any] = []
        registry.saveable_session_changed.connect(
            lambda payload: emissions.append(payload)
        )

        app._app_settings.clear_comms_queue_mode = False
        app._on_clear_comms_requested()
        _drain(qapp)

        assert registry.saveable_session_id == sid  # unchanged
        assert dispatched and dispatched[0]["session_id"] is None
        assert emissions == []

    def test_saveable_session_id_unchanged_after_clear_comms_queue(
        self, qapp, app_with_clear_comms
    ):
        app, registry, _queue, dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        sid = _finalize_session_with_audio(registry, audio, sample_rate=sr)
        _drain(qapp)

        emissions: List[Any] = []
        registry.saveable_session_changed.connect(
            lambda payload: emissions.append(payload)
        )

        app._app_settings.clear_comms_queue_mode = True
        app._on_clear_comms_requested()
        _drain(qapp)

        assert registry.saveable_session_id == sid
        assert dispatched and dispatched[0]["session_id"] is None
        assert emissions == []


# --------------------------------------------------------------------------- #
# AC #8 — interrupt helper does NOT cancel generation
# --------------------------------------------------------------------------- #


class TestClearCommsDoesNotCancelGeneration:
    """AC #8 — Clear Comms is a *playback* interrupt, not a *generation*
    cancel. The mocked ``tts_service.cancel_generation`` must NEVER be
    called as a side effect of a Clear Comms click."""

    def test_interrupt_helper_does_not_call_tts_cancel(self, app_with_clear_comms):
        app, registry, _queue, _dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        _finalize_session_with_audio(registry, audio, sample_rate=sr)

        app._app_settings.clear_comms_queue_mode = False
        app._on_clear_comms_requested()

        assert app._tts_service.cancel_generation.call_count == 0

    def test_interrupt_helper_safe_when_idle(self, app_with_clear_comms):
        """The helper is safe to call when nothing is currently playing —
        every branch is a defensive no-op (per AC #8 last paragraph)."""
        app, registry, _queue, dispatched = app_with_clear_comms
        audio, sr = _make_synthetic_saveable_audio()
        _finalize_session_with_audio(registry, audio, sample_rate=sr)

        # No playback in progress — _dispatching_session_id is None,
        # focal is not in PLAYING state.
        assert app._dispatching_session_id is None

        app._app_settings.clear_comms_queue_mode = False
        # Should not raise:
        app._on_clear_comms_requested()

        # The dispatch ran.
        assert len(dispatched) == 1


# --------------------------------------------------------------------------- #
# Resolve-helper unit-style tests (testable in isolation)
# --------------------------------------------------------------------------- #


class TestResolveClearCommsWavBytes:
    """Direct unit tests of ``_resolve_clear_comms_wav_bytes`` without
    going through the full slot — keeps the failure modes pinned to the
    helper rather than the dispatch chain."""

    def test_no_settings_at_all_resolves_via_slot_short_circuit(
        self, app_with_clear_comms
    ):
        """Slot-level guard: ``_app_settings is None`` → no-op (logs +
        returns) so the helper isn't even called."""
        app, _registry, _queue, dispatched = app_with_clear_comms
        app._app_settings = None
        app._on_clear_comms_requested()
        assert dispatched == []
        # No toast — there's no main_window method to call without settings,
        # but the slot also doesn't crash.

    def test_unknown_source_kind_raises_resolve_error(self, app_with_clear_comms):
        app, _registry, _queue, _dispatched = app_with_clear_comms
        with pytest.raises(_ClearCommsResolveError) as excinfo:
            app._resolve_clear_comms_wav_bytes(
                "bogus_source_kind", app._app_settings
            )
        assert "bogus_source_kind" in excinfo.value.user_message
