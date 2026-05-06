"""Integration tests for Story 14.3: Save Dialog with WAV Writer and
Save-During-Streaming Flow.

Drives the real ``SessionRegistry`` through ``finalize`` / ``cancel``
slots to exercise the saveable lifecycle, and runs ``SaveAudioDialog``
against the real registry surface (no slot mocking). The OS file
dialog is the single component still patched — ``QFileDialog.getSaveFileName``
is mocked to return ``tmp_path``-based targets so tests are hermetic.

Branch C tests (review fix H2): the streaming-wait completion and
abort tests now use the **real** ``saveable_session_changed.emit(...)``
signal followed by ``qtbot.wait(...)`` to drain the
``QueuedConnection`` — exercising the Qt signal-delivery path that the
unit tests in ``tests/ui/test_save_dialog.py`` cannot reach. Direct
slot invocation is no longer used here; that pattern lives only in the
unit-test file where the deferred-delivery contract is documented.

Test scope (per Story 14.3 AC #15):

  - ``test_save_finalized_session_writes_correct_wav`` — Branch B end-
    to-end through the registry. AC #2 / #5 / #11 / #14.
  - ``test_save_during_streaming_waits_for_finalize`` — Branch C with a
    real registry: drive a streaming saveable, click Save, emit
    ``saveable_session_changed`` via the real Qt signal, drain the
    queue, assert WAV exists and content matches the post-finalize
    buffer. Validation gap #2 in concrete form.
  - ``test_save_during_streaming_aborts_on_cancel`` — Branch C abort:
    emit ``saveable_session_changed(None)`` via the real Qt signal;
    assert no WAV on disk and the abort toast.

The whole module skips when torch + PyQt6 fail to load — mirrors
``test_playback_last_preservation.py``'s precedent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("PyQt6")

import soundfile as sf  # noqa: E402

# --------------------------------------------------------------------------- #
# Production import — guarded so the module skips cleanly without torch
# --------------------------------------------------------------------------- #

_IMPORT_ERROR: Optional[Exception] = None
SaveAudioDialog = None
SaveableAudio = None
SessionRegistry = None
SessionSource = None
SessionState = None

try:
    from myvoice.services.sessions import (  # type: ignore[import-not-found]
        SaveableAudio,
        SessionRegistry,
        SessionSource,
        SessionState,
    )
    from myvoice.services.sessions.generation_session import (  # type: ignore[import-not-found]
        GenerationSession,
    )
    from myvoice.ui.dialogs.save_dialog import (  # type: ignore[import-not-found]
        SaveAudioDialog,
        _TOAST_ABORT_CANCELLED,
    )
except Exception as exc:  # pragma: no cover — env-dependent
    _IMPORT_ERROR = exc

if _IMPORT_ERROR is not None:
    pytestmark = pytest.mark.skip(
        reason=(
            "Sessions / SaveAudioDialog import failed "
            f"(e.g. torch DLL load): {_IMPORT_ERROR!r}"
        )
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def registry(qapp):
    """Real SessionRegistry parented to qapp."""
    reg = SessionRegistry(parent=qapp)
    yield reg
    reg.deleteLater()


@pytest.fixture
def parent_widget(qapp):
    """Parent QWidget exposing ``status_bar`` (mirrors MainWindow)."""
    from PyQt6.QtWidgets import QStatusBar, QWidget

    parent = QWidget()
    parent.status_bar = QStatusBar(parent)
    yield parent
    parent.deleteLater()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _drive_to_finalized(
    registry: "SessionRegistry",
    *,
    text: str = "integration test",
    voice: str = "test_voice",
    audio: Optional[np.ndarray] = None,
) -> str:
    """Drive a session through PENDING → GENERATING → READY_TO_PLAY
    via the registry's slot path so the saveable lifecycle promotes
    naturally.
    """
    if audio is None:
        audio = np.array([0.0, 0.25, 0.5, -0.25], dtype=np.float32)
    sid = registry.create_session(
        text=text, voice=voice, model_type="test_model",
        source=SessionSource.GENERATED,
    )
    registry.start_generation(sid)
    registry.append_chunk(sid, audio)
    registry.finalize(sid)
    return sid


def _drive_to_streaming_with_saveable(
    registry: "SessionRegistry",
    *,
    text: str = "integration test",
    voice: str = "test_voice",
    audio: Optional[np.ndarray] = None,
) -> str:
    """Inject a SaveableAudio with ``is_streaming=True`` into the
    registry's slot directly so Branch C activates without a real
    streaming TTS path. Mirrors the Story 14.3 AC #15 helper guidance.
    """
    if audio is None:
        audio = np.array([0.0, 0.1], dtype=np.float32)
    sid = registry.create_session(
        text=text, voice=voice, model_type="test_model",
        source=SessionSource.GENERATED,
    )
    registry.start_generation(sid)
    registry.append_chunk(sid, audio)
    # Inject a streaming saveable directly — bypasses normal finalize so
    # is_streaming=True. The dialog reads from registry.saveable_audio /
    # saveable_session_id (and listens for saveable_session_changed),
    # which reflect the slot.
    saveable = SaveableAudio(
        session_id=sid,
        complete_audio=audio,
        sample_rate=24000,
        voice=voice,
        text=text,
        is_streaming=True,
        created_at=0.0,
    )
    registry._saveable = saveable
    return sid


# --------------------------------------------------------------------------- #
# Branch B integration — finalized session write
# --------------------------------------------------------------------------- #


class TestSaveFinalizedSession:
    def test_save_finalized_session_writes_correct_wav(
        self, qapp, registry, parent_widget, tmp_path
    ):
        ramp = np.linspace(-0.5, 0.5, 240, dtype=np.float32)
        sid = _drive_to_finalized(
            registry, voice="testvoice", audio=ramp
        )
        # Saveable promotion fires on finalize per Story 14.1.
        assert registry.saveable_session_id == sid
        assert registry.saveable_audio is not None
        assert registry.saveable_audio.is_streaming is False

        target = tmp_path / "out.wav"
        dialog = SaveAudioDialog(registry=registry, parent=parent_widget)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), "WAV files (*.wav)"),
        ):
            dialog.run()

        assert target.exists()
        # AC #5 D-16 invariants: int16, sample rate matches session.
        data, sr = sf.read(str(target), dtype="int16")
        assert sr == 24000
        assert data.dtype == np.int16
        # Content matches the D-16 conversion exactly.
        expected = (ramp * 32767).clip(-32768, 32767).astype(np.int16)
        np.testing.assert_array_equal(data, expected)


# --------------------------------------------------------------------------- #
# Branch C integration — save during streaming, finalize completes
# --------------------------------------------------------------------------- #


class TestSaveDuringStreaming:
    def test_save_during_streaming_waits_for_finalize(
        self, qapp, registry, parent_widget, qtbot, tmp_path
    ):
        """Review fix H2: drain the real ``QueuedConnection`` rather
        than direct-invoking the dialog's wait slot.

        We still inject ``_saveable`` directly because in V2 today no
        production code path produces a saveable with ``is_streaming
        =True`` (Epic 16 territory; see Story 14.3 Dev Notes). What
        the previous version skipped — and this version restores — is
        the actual Qt signal flow: ``saveable_session_changed.emit
        (sid)`` fires through the real connection, the QueuedConnection
        defers the slot to the next event-loop iteration, and
        ``qtbot.wait(...)`` drains it.
        """
        # Click-time: short partial buffer; user clicks Save while streaming.
        partial = np.array([0.0, 0.1], dtype=np.float32)
        sid = _drive_to_streaming_with_saveable(
            registry, voice="streamvoice", audio=partial
        )

        target = tmp_path / "out.wav"
        dialog = SaveAudioDialog(registry=registry, parent=parent_widget)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), "WAV files (*.wav)"),
        ):
            dialog.run()
            # Branch C deferred: no file yet.
            assert not target.exists()

            # Promote a "post-finalize" snapshot into the slot — this is
            # what Story 14.1's `_maybe_promote_saveable` would write
            # after a real finalize. We do this directly because in V2
            # today no production code finalizes a session with
            # is_streaming=True (Epic 16).
            full_audio = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32
            )
            registry._saveable = SaveableAudio(
                session_id=sid,
                complete_audio=full_audio,
                sample_rate=24000,
                voice="streamvoice",
                text="integration test",
                is_streaming=False,
                created_at=0.0,
            )
            # Real Qt signal emission — exercises QueuedConnection
            # delivery through the dialog's connect() call. The slot
            # runs on the next event-loop iteration.
            registry.saveable_session_changed.emit(sid)
            qtbot.wait(100)

        assert target.exists()
        data, sr = sf.read(str(target), dtype="int16")
        assert sr == 24000
        assert data.dtype == np.int16
        # The post-finalize 5-sample buffer was written, NOT the
        # click-time 2-sample partial.
        expected = (full_audio * 32767).clip(-32768, 32767).astype(np.int16)
        np.testing.assert_array_equal(data, expected)


# --------------------------------------------------------------------------- #
# Branch C integration — cancel before finalize aborts the save
# --------------------------------------------------------------------------- #


class TestSaveDuringStreamingAbort:
    def test_save_during_streaming_aborts_on_cancel(
        self, qapp, registry, parent_widget, qtbot, tmp_path
    ):
        """Review fix H2: emit cancellation through the real Qt signal
        rather than direct-invoking the slot.
        """
        sid = _drive_to_streaming_with_saveable(
            registry, voice="cancelvoice"
        )
        target = tmp_path / "out.wav"
        dialog = SaveAudioDialog(registry=registry, parent=parent_widget)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), "WAV files (*.wav)"),
        ):
            dialog.run()
            assert not target.exists()
            # Cancellation: vacate slot and emit None via the real
            # Qt signal — the dialog's QueuedConnection slot fires
            # on the next event-loop iteration.
            registry._saveable = None
            registry.saveable_session_changed.emit(None)
            qtbot.wait(100)

        assert not target.exists()
        # AC #9 abort toast — None payload triggers the
        # "cancelled" copy (review fix H1).
        assert (
            parent_widget.status_bar.currentMessage()
            == _TOAST_ABORT_CANCELLED
        )
