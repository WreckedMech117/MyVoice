"""Story 15.3 — AC #11: D-5 invariant under repeated Clear Comms clicks.

The architecture's D-5 invariant ("PRELOADED-clone exclusion") promises
that clicking Clear Comms NEVER advances the saveable slot and NEVER
fires ``saveable_session_changed``. Story 15.2 already pinned this for
a single click in
``tests/integration/test_clear_comms_dispatch.py::TestClearCommsDoesNotAdvanceSaveable``.

This story extends the assertion to a *click sequence* (N=3) under
both source kinds (last_generation, file) and to a Save round-trip
that confirms the bytes the user would actually save are A's bytes —
not any of the Clear Comms playback's bytes. This is the architecture
file's "verified explicitly in Story 15.3's final acceptance criterion"
promise (epic file line 824) made empirical.

Pattern A from Story 13.3 (Open Question #3) — duplicates the
``_make_dispatch_stub`` shape from ``test_clear_comms_dispatch.py``
rather than refactoring to a shared helper. The two test files are
kept independent so a refactor of one doesn't churn the other.
"""

from __future__ import annotations

import hashlib
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

try:
    from myvoice.app import MyVoiceApp  # type: ignore[import-not-found]
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
        reason=(
            "MyVoiceApp / sessions import failed (e.g. torch DLL load): "
            f"{_IMPORT_ERROR!r}"
        )
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _drain(qapp, iterations: int = 5) -> None:
    for _ in range(iterations):
        qapp.processEvents()


def _make_synthetic_audio(
    duration_s: float = 0.05,
    sample_rate: int = 24000,
    seed: int = 1,
) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_s)
    audio = (rng.uniform(-0.5, 0.5, size=n)).astype(np.float32)
    return audio, sample_rate


def _hash_audio(audio: np.ndarray) -> str:
    return hashlib.sha256(audio.tobytes()).hexdigest()


def _make_dispatch_stub(app, dispatched: List[dict]) -> Callable[..., Any]:
    """Mirror of test_clear_comms_dispatch.py's stub — claims the queue
    slot via the production helpers and records each dispatch.
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


def _decode_wav_bytes(buf: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(buf) as bio:
        arr, sr = sf.read(bio, dtype="float32", always_2d=False)
    return arr, sr


def _finalize_session_with_audio(
    registry, audio: np.ndarray, sample_rate: int = 24000, text: str = "hello"
) -> str:
    """Drive a session through the registry's normal lifecycle so its
    saveable slot is promoted naturally.
    """
    sid = registry.create_session(
        text=text, voice="v", model_type="m", source=SessionSource.GENERATED,
    )
    object.__setattr__(registry.get(sid), "sample_rate", sample_rate)
    registry.start_generation(sid)
    registry.append_chunk(sid, audio)
    registry.finalize(sid)
    return sid


def _encode_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Mirror of MyVoiceApp._encode_wav_bytes; needed for the Save
    round-trip assertion (the saveable's bytes are stored as float32 in
    the registry; the would-be-saved file is int16 PCM16 WAV)."""
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_int16, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def app_d5(qapp, monkeypatch):
    """Build a partial MyVoiceApp for D-5 invariant testing.

    Mirrors ``app_with_clear_comms`` in test_clear_comms_dispatch.py but
    re-built locally so a refactor of one fixture doesn't churn the
    other (Pattern A — Story 13.3 Open Question #3).
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

    def _sync_run(coro, on_success=None, on_error=None):
        import asyncio

        try:
            asyncio.get_event_loop().run_until_complete(coro)
        except RuntimeError:
            asyncio.new_event_loop().run_until_complete(coro)

    monkeypatch.setattr(app, "_run_async_task", _sync_run)

    yield app, app._session_registry, dispatched

    try:
        app._playback_queue.playback_queue_depth_changed.disconnect()
    except (TypeError, RuntimeError):
        pass
    app._session_registry.deleteLater()
    app._playback_queue.deleteLater()


# --------------------------------------------------------------------------- #
# AC #11 — D-5 across N=3 clicks (parametrized over source_kind)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source_kind", ["last_generation", "file"])
def test_d5_invariant_holds_across_three_clear_comms_clicks(
    qapp, app_d5, tmp_path, source_kind
):
    """AC #11 — clicking Clear Comms N=3 times never advances the
    saveable slot or fires saveable_session_changed.
    """
    app, registry, dispatched = app_d5
    audio_a, sr = _make_synthetic_audio(seed=11)
    sid_a = _finalize_session_with_audio(registry, audio_a, sample_rate=sr)
    _drain(qapp)
    assert registry.saveable_session_id == sid_a

    # Capture pre-click hash of A's complete buffer.
    pre_click_hash = _hash_audio(registry.saveable_audio.complete_audio)

    # Configure source per parametrization.
    if source_kind == "last_generation":
        app._app_settings.clear_comms_source_kind = "last_generation"
        app._app_settings.clear_comms_file_path = None
    else:
        # Build a *different* WAV file (audio B) so the file dispatch
        # is empirically distinct from the saveable; this lets us
        # confirm the file branch dispatches B (not A) yet leaves A
        # untouched in the registry.
        audio_b, sr_b = _make_synthetic_audio(seed=22, sample_rate=sr)
        wav_path = tmp_path / "clear_comms.wav"
        sf.write(str(wav_path), audio_b, sr_b)
        app._app_settings.clear_comms_source_kind = "file"
        app._app_settings.clear_comms_file_path = str(wav_path)
    app._app_settings.clear_comms_queue_mode = False  # interrupt mode

    # Watch saveable_session_changed for any spurious emission.
    saveable_emissions: List[Any] = []
    registry.saveable_session_changed.connect(
        lambda payload: saveable_emissions.append(payload)
    )

    # Click 3 times. Interrupt mode resets _dispatching_session_id at
    # the start of each click so the next dispatch can claim the slot.
    for _ in range(3):
        app._on_clear_comms_requested()
        _drain(qapp)

    # D-5 invariant assertions.
    assert registry.saveable_session_id == sid_a, (
        "saveable_session_id changed during Clear Comms click sequence"
    )
    assert _hash_audio(registry.saveable_audio.complete_audio) == pre_click_hash, (
        "saveable audio buffer mutated during Clear Comms click sequence"
    )
    assert saveable_emissions == [], (
        f"saveable_session_changed emitted {len(saveable_emissions)} times "
        f"during Clear Comms click sequence — D-5 invariant violated"
    )

    # Each click produced exactly one dispatch.
    assert len(dispatched) == 3
    for d in dispatched:
        assert d["session_id"] is None, (
            "Clear Comms dispatch was registered with a session_id; "
            "it must use the registry-less replay path (D-5 mechanism)"
        )
        assert d["queue_token"].startswith("replay-"), (
            "Clear Comms dispatch did not use a synthetic replay token"
        )

    # Source-kind-specific dispatch content assertions.
    if source_kind == "last_generation":
        # All three dispatches contain A's bytes.
        for d in dispatched:
            roundtrip, sr_back = _decode_wav_bytes(d["audio_data"])
            assert sr_back == sr
            assert np.allclose(roundtrip, audio_a, atol=1e-3)
    else:
        # All three dispatches contain B's bytes — and B != A.
        for d in dispatched:
            roundtrip, sr_back = _decode_wav_bytes(d["audio_data"])
            assert sr_back == sr
            # B is distinct from A by construction (different rng seed).
            assert not np.allclose(roundtrip, audio_a, atol=1e-2)


# Review fix M1: removed test_save_round_trips_to_a_after_clear_comms_click_sequence
# because it duplicated the hash assertion already in
# test_d5_invariant_holds_across_three_clear_comms_clicks (which pins
# registry.saveable_audio.complete_audio unchanged after N=3 clicks)
# and the encode/decode round-trip already pinned by
# test_clear_comms_dispatch.py::test_dispatches_saveable_audio_in_interrupt_mode.
# The deleted test exercised no Save dialog code despite its name.


# --------------------------------------------------------------------------- #
# Story 15.3 review fix H1 — AC #11 verbatim: real MainWindow + qtbot.mouseClick
# --------------------------------------------------------------------------- #


@pytest.fixture
def app_with_real_main_window(qapp, qtbot, monkeypatch):
    """Build a MyVoiceApp wired to a real MyVoiceMainWindow + ClearCommsButton.

    This is the AC #11 fixture review fix H1 demanded — the prior
    ``app_d5`` fixture used ``MagicMock()`` for ``_main_window`` and
    drove the slot directly, bypassing the toolbar button entirely.
    """
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import Qt as _Qt  # noqa: F401  # used by qtbot.mouseClick at call site

    from myvoice.ui.main_window import MainWindow

    # Real registry + queue (the D-5 invariant lives on the registry,
    # not the app, so this must be real).
    registry = SessionRegistry()
    queue = PlaybackQueue()
    queue.playback_queue_depth_changed.connect(
        registry.playback_queue_depth_changed.emit
    )

    # Real MainWindow with a real ClearCommsButton wired to a real
    # clear_comms_requested signal.
    main_window = MainWindow(session_registry=registry)
    qtbot.addWidget(main_window)

    # Partial app (no GUI parent — we attach the real main_window post-init).
    app = MyVoiceApp(qapp)
    app._session_registry = registry
    app._playback_queue = queue

    audio_coord_stub = MagicMock()
    audio_coord_stub.stop_all_playback = AsyncMock()
    app._audio_coordinator = audio_coord_stub

    tts_stub = MagicMock()
    tts_stub.cancel_generation = AsyncMock()
    app._tts_service = tts_stub

    app._main_window = main_window  # REAL — review fix H1.

    app._app_settings = MagicMock(
        clear_comms_source_kind="last_generation",
        clear_comms_file_path=None,
        clear_comms_queue_mode=False,
    )

    # Wire the click signal exactly as MyVoiceApp.run() would.
    main_window.clear_comms_requested.connect(app._on_clear_comms_requested)

    dispatched: List[dict] = []

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

    monkeypatch.setattr(app, "_play_generated_audio", stub_play_generated_audio)

    def _sync_run(coro, on_success=None, on_error=None):
        import asyncio

        try:
            asyncio.get_event_loop().run_until_complete(coro)
        except RuntimeError:
            asyncio.new_event_loop().run_until_complete(coro)

    monkeypatch.setattr(app, "_run_async_task", _sync_run)

    yield app, main_window, registry, dispatched

    try:
        queue.playback_queue_depth_changed.disconnect()
    except (TypeError, RuntimeError):
        pass
    registry.deleteLater()
    queue.deleteLater()


def test_d5_invariant_via_real_clear_comms_button_click(
    qapp, qtbot, app_with_real_main_window
):
    """AC #11 verbatim — drive ``qtbot.mouseClick`` against the real
    ``main_window.clear_comms_button`` and assert D-5 holds across the
    UI signal chain (button → MainWindow.clear_comms_requested →
    MyVoiceApp._on_clear_comms_requested → dispatch).

    Review fix H1 for the prior ``test_d5_invariant_holds_across_three_clear_comms_clicks``
    test, which mocked ``_main_window`` and called the slot directly —
    bypassing the entire UI layer the AC promises to verify.
    """
    from PyQt6.QtCore import Qt

    app, main_window, registry, dispatched = app_with_real_main_window

    # Finalize session A so the button enables (last_generation gate).
    audio_a, sr = _make_synthetic_audio(seed=51)
    sid_a = _finalize_session_with_audio(registry, audio_a, sample_rate=sr)
    _drain(qapp)
    assert registry.saveable_session_id == sid_a

    # Seed the Clear Comms snapshot so the button's enablement matrix
    # resolves to the enabled-interrupt-last-gen state. This is the same
    # call the SettingsDialog "OK" path makes (see Story 15.3 AC #8).
    main_window.set_clear_comms_config_snapshot(
        source_kind="last_generation",
        file_path=None,
        file_path_valid=False,
        queue_mode=False,
    )
    _drain(qapp)
    assert main_window.clear_comms_button.isEnabled() is True, (
        "clear_comms_button is not enabled — fixture wiring is wrong; "
        "the H1 test cannot exercise the real click path."
    )

    pre_click_hash = _hash_audio(registry.saveable_audio.complete_audio)

    saveable_emissions: List[Any] = []
    registry.saveable_session_changed.connect(
        lambda payload: saveable_emissions.append(payload)
    )

    # AC #11 verbatim: ``qtbot.mouseClick(main_window.clear_comms_button,
    # Qt.MouseButton.LeftButton)`` — three times, draining between each
    # so the dispatch + queue-vacate cycle completes before the next.
    for _ in range(3):
        qtbot.mouseClick(
            main_window.clear_comms_button, Qt.MouseButton.LeftButton
        )
        _drain(qapp)

    # D-5 assertions, identical to the slot-level parametrized test —
    # but now they're empirically pinned through the real UI path.
    assert registry.saveable_session_id == sid_a
    assert _hash_audio(registry.saveable_audio.complete_audio) == pre_click_hash
    assert saveable_emissions == [], (
        f"saveable_session_changed emitted {len(saveable_emissions)} times "
        "during Clear Comms click sequence (real UI path) — D-5 violated"
    )
    assert len(dispatched) == 3
    for d in dispatched:
        assert d["session_id"] is None
        assert d["queue_token"].startswith("replay-")
