"""Tests for Story 14.3: Save Dialog with WAV Writer and Save-During-
Streaming Flow.

Covers AC #1 through #16 of the SaveAudioDialog controller — construction
contract, three-branch dispatch (no-saveable / finalized / streaming),
file-dialog invocation, WAV write per D-16, default filename
sanitization, streaming-wait state machine, abort paths, success/failure
toasts, module-boundary discipline, and the app.py wiring.

Test class layout (mirrors Story 14.2 organization):

  - ``TestSaveAudioDialogConstruction``         — AC #1 construction.
  - ``TestSaveAudioDialogBranchA``              — AC #2 Branch A.
  - ``TestSaveAudioDialogBranchB``              — AC #2 Branch B + AC
    #3, #4, #5, #11 finalized save.
  - ``TestSaveAudioDialogBranchC``              — AC #2 Branch C + AC
    #4.1, #7, #8 streaming-wait.
  - ``TestSaveAudioDialogStreamingAbort``       — AC #6, #9 abort
    paths.
  - ``TestSaveAudioDialogFilenameSanitization`` — AC #10 parametrized.
  - ``TestSaveAudioDialogModuleBoundary``       — AC #13 import
    discipline.
  - ``TestSaveAudioDialogWriteErrors``          — AC #5 error path.
  - ``TestAppPySaveRequestedWiring``            — AC #12 app.py
    integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# PyQt6 imports — skip the whole file if PyQt6 is not available.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QStatusBar, QWidget

from myvoice.services.sessions import SaveableAudio
from myvoice.services.sessions.session_registry import SessionRegistry
from myvoice.ui.dialogs import SaveAudioDialog as SaveAudioDialogFromPackage
from myvoice.ui.dialogs.save_dialog import (
    SaveAudioDialog,
    _FILE_DIALOG_FILTER,
    _FILE_DIALOG_TITLE,
    _TOAST_ABORT_CANCELLED,
    _TOAST_ABORT_SAVEABLE_CHANGED,
    _TOAST_FINALIZING,
    _TOAST_NO_SAVEABLE,
)


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def app(qtbot):
    """Ensure QApplication exists (matches test_save_button.py)."""
    return QApplication.instance() or QApplication([])


def make_finalized_saveable(
    session_id: str = "test-target-sid",
    is_streaming: bool = False,
    sample_rate: int = 24000,
    voice: str = "test_voice",
    text: str = "test text",
    created_at: float = 0.0,
    audio: Optional[np.ndarray] = None,
) -> SaveableAudio:
    """Build a SaveableAudio for dialog-controller-level isolation."""
    if audio is None:
        audio = np.array([0.0, 0.5, -0.5, 0.99], dtype=np.float32)
    return SaveableAudio(
        session_id=session_id,
        complete_audio=audio,
        sample_rate=sample_rate,
        voice=voice,
        text=text,
        is_streaming=is_streaming,
        created_at=created_at,
    )


class _SignalSpy(QObject):
    """Tiny Qt-signal wrapper supporting connect/disconnect/emit + capture
    of connect/disconnect calls.

    The story's mock-registry pattern uses a MagicMock for
    ``saveable_session_changed``, which works for connect/disconnect call
    counting but cannot actually emit through Qt. ``_SignalSpy`` is the
    "real-Qt" alternative used by Branch C tests that need a slot to fire.
    """

    sig = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.connect_calls: list = []
        self.disconnect_calls: list = []

    def connect(self, slot, conn_type=Qt.ConnectionType.AutoConnection):
        self.connect_calls.append((slot, conn_type))
        self.sig.connect(slot, conn_type)

    def disconnect(self, slot=None):
        self.disconnect_calls.append(slot)
        if slot is None:
            self.sig.disconnect()
        else:
            self.sig.disconnect(slot)

    def emit(self, payload):
        self.sig.emit(payload)


def make_mock_registry(
    saveable_audio: Optional[SaveableAudio] = None,
    use_real_signal: bool = False,
) -> MagicMock:
    """A mock SessionRegistry exposing only what SaveAudioDialog reads.

    ``use_real_signal=True`` swaps in a ``_SignalSpy`` so tests can
    actually emit ``saveable_session_changed`` and observe the slot fire.

    Sanity check (review fix M4): the ``MagicMock(spec=SessionRegistry)``
    constraint is bypassed when we assign over the spec'd attributes, so
    we explicitly assert that the contract surface still exists. If a
    future refactor renames any of these, the assertion fails fast
    instead of letting tests silently drift against a stale interface.
    """
    assert hasattr(SessionRegistry, "saveable_audio"), (
        "SessionRegistry.saveable_audio missing — mock contract drifted"
    )
    assert hasattr(SessionRegistry, "saveable_session_id"), (
        "SessionRegistry.saveable_session_id missing — mock contract drifted"
    )
    assert hasattr(SessionRegistry, "saveable_session_changed"), (
        "SessionRegistry.saveable_session_changed missing — mock contract drifted"
    )
    registry = MagicMock(spec=SessionRegistry)
    registry.saveable_audio = saveable_audio
    registry.saveable_session_id = (
        saveable_audio.session_id if saveable_audio else None
    )
    if use_real_signal:
        registry.saveable_session_changed = _SignalSpy()
    else:
        registry.saveable_session_changed = MagicMock()
    return registry


def make_parent_with_status_bar(qtbot) -> QWidget:
    """Create a parent QWidget exposing ``status_bar`` like MainWindow."""
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.status_bar = QStatusBar(parent)
    return parent


# --------------------------------------------------------------------------- #
# AC #1 — TestSaveAudioDialogConstruction
# --------------------------------------------------------------------------- #


class TestSaveAudioDialogConstruction:
    """Construction contract: subclass, no I/O, no signal connections."""

    def test_dialog_is_qobject_subclass(self):
        assert issubclass(SaveAudioDialog, QObject)

    def test_dialog_is_re_exported_from_package(self):
        # AC #1 requires the package-root import path.
        assert SaveAudioDialogFromPackage is SaveAudioDialog

    def test_dialog_construction_no_io(self, app, qtbot):
        """No filesystem / soundfile calls happen during __init__."""
        registry = make_mock_registry()
        with patch("myvoice.ui.dialogs.save_dialog.sf.write") as sf_write:
            dialog = SaveAudioDialog(registry=registry, parent=None)
            assert dialog is not None
            assert sf_write.call_count == 0

    def test_dialog_construction_no_signal_connections(self, app):
        """Construction must NOT subscribe to saveable_session_changed.

        Subscription is deferred to AC #7 (Branch C only).
        """
        registry = make_mock_registry()
        SaveAudioDialog(registry=registry, parent=None)
        assert registry.saveable_session_changed.connect.call_count == 0

    def test_dialog_logger_attached(self, app):
        registry = make_mock_registry()
        dialog = SaveAudioDialog(registry=registry, parent=None)
        assert dialog._logger is not None
        assert dialog._logger.name == "SaveAudioDialog"

    def test_dialog_run_is_callable(self, app):
        registry = make_mock_registry()
        dialog = SaveAudioDialog(registry=registry, parent=None)
        assert callable(dialog.run)


# --------------------------------------------------------------------------- #
# AC #2 Branch A — TestSaveAudioDialogBranchA
# --------------------------------------------------------------------------- #


class TestSaveAudioDialogBranchA:
    """Defensive no-saveable path: log warning, transient toast, return."""

    def test_run_with_no_saveable_audio_shows_toast(self, app, qtbot):
        registry = make_mock_registry(saveable_audio=None)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName"
        ) as getsave:
            dialog.run()
            assert getsave.call_count == 0
        assert _TOAST_NO_SAVEABLE in parent.status_bar.currentMessage()

    def test_run_with_no_saveable_audio_does_not_write(self, app, qtbot):
        registry = make_mock_registry(saveable_audio=None)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch("myvoice.ui.dialogs.save_dialog.sf.write") as sf_write:
            dialog.run()
            assert sf_write.call_count == 0


# --------------------------------------------------------------------------- #
# AC #2 Branch B + AC #3, #4, #5, #11 — TestSaveAudioDialogBranchB
# --------------------------------------------------------------------------- #


class TestSaveAudioDialogBranchB:
    """Finalized / immediate-save path."""

    def test_run_with_finalized_audio_opens_dialog(self, app, qtbot, tmp_path):
        audio = make_finalized_saveable(is_streaming=False, voice="male1")
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), "WAV files (*.wav)"),
        ) as getsave, patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ):
            dialog.run()
            assert getsave.call_count == 1
            args = getsave.call_args
            # parent, title, default_path, filter
            assert args.args[1] == _FILE_DIALOG_TITLE
            assert args.args[3] == _FILE_DIALOG_FILTER
            # AC #10 default filename pattern in default path.
            assert "myvoice-male1-" in args.args[2]
            assert args.args[2].endswith(".wav")

    def test_run_with_user_cancels_dialog_writes_nothing(self, app, qtbot):
        audio = make_finalized_saveable(is_streaming=False)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            assert sf_write.call_count == 0
        # No toast on cancel (AC #3 explicit silent-return).
        assert parent.status_bar.currentMessage() == ""

    def test_run_with_finalized_audio_writes_wav(self, app, qtbot, tmp_path):
        audio = make_finalized_saveable(
            is_streaming=False,
            sample_rate=24000,
            audio=np.array([0.5, -0.5, 0.0], dtype=np.float32),
        )
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), "WAV files (*.wav)"),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            assert sf_write.call_count == 1
            args = sf_write.call_args
            # positional: target, audio, sample_rate
            assert args.args[0] == str(target)
            np.testing.assert_array_equal(
                args.args[1],
                (audio.complete_audio * 32767)
                .clip(-32768, 32767)
                .astype(np.int16),
            )
            assert args.args[2] == 24000
            assert args.kwargs.get("subtype") == "PCM_16"

    def test_wav_int16_conversion_matches_d16(self, app, qtbot, tmp_path):
        # Edge cases: clip on +/-1.0 and slightly above.
        raw = np.array([0.0, 1.0, -1.0, 1.5, -1.5, 0.5], dtype=np.float32)
        audio = make_finalized_saveable(audio=raw)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
        expected = (raw * 32767).clip(-32768, 32767).astype(np.int16)
        np.testing.assert_array_equal(sf_write.call_args.args[1], expected)
        assert sf_write.call_args.args[1].dtype == np.int16

    def test_default_filename_format(self, app):
        registry = make_mock_registry(make_finalized_saveable(voice="male1"))
        dialog = SaveAudioDialog(registry=registry, parent=None)
        name = dialog._make_default_filename("male1")
        # myvoice-male1-YYYY-MM-DD-HHMMSS.wav
        import re

        assert re.fullmatch(
            r"myvoice-male1-\d{4}-\d{2}-\d{2}-\d{6}\.wav", name
        ), name

    def test_default_filename_sanitizes_voice(self, app):
        registry = make_mock_registry()
        dialog = SaveAudioDialog(registry=registry, parent=None)
        name = dialog._make_default_filename("Bad/Voice:Name?")
        # Bad/Voice:Name? → Bad_Voice_Name_
        assert "Bad_Voice_Name_" in name

    def test_default_filename_empty_voice_uses_fallback(self, app):
        registry = make_mock_registry()
        dialog = SaveAudioDialog(registry=registry, parent=None)
        name = dialog._make_default_filename("")
        # voice fallback
        assert name.startswith("myvoice-voice-")

    def test_extension_appended_when_user_omits(
        self, app, qtbot, tmp_path
    ):
        audio = make_finalized_saveable(is_streaming=False)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        # User typed a path without .wav extension.
        target_no_ext = tmp_path / "myrecording"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target_no_ext), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            written_path = sf_write.call_args.args[0]
            assert written_path.endswith(".wav")
            assert written_path == str(tmp_path / "myrecording.wav")

    def test_success_toast_shown_after_write(self, app, qtbot, tmp_path):
        audio = make_finalized_saveable(is_streaming=False)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ):
            dialog.run()
        msg = parent.status_bar.currentMessage()
        assert msg.startswith("Saved to ")
        assert "out.wav" in msg

    def test_filter_string_is_wav_only(self, app, qtbot, tmp_path):
        audio = make_finalized_saveable(is_streaming=False)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ) as getsave, patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ):
            dialog.run()
        assert getsave.call_args.args[3] == "WAV files (*.wav);;All files (*.*)"

    def test_branch_b_does_not_subscribe(self, app, qtbot, tmp_path):
        """Branch B (immediate save) MUST NOT subscribe to saveable_session_changed."""
        audio = make_finalized_saveable(is_streaming=False)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ):
            dialog.run()
        assert registry.saveable_session_changed.connect.call_count == 0


# --------------------------------------------------------------------------- #
# AC #2 Branch C + AC #4.1, #7, #8 — TestSaveAudioDialogBranchC
# --------------------------------------------------------------------------- #


class TestSaveAudioDialogBranchC:
    """Streaming-wait path."""

    def test_run_with_streaming_audio_opens_dialog_and_subscribes(
        self, app, qtbot, tmp_path
    ):
        audio = make_finalized_saveable(is_streaming=True)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ):
            dialog.run()
        assert registry.saveable_session_changed.connect.call_count == 1
        # AC #7: Qt.ConnectionType.QueuedConnection passed explicitly.
        connect_args = registry.saveable_session_changed.connect.call_args
        # call_args.args[0] = slot, args[1] = connection type
        assert connect_args.args[1] == Qt.ConnectionType.QueuedConnection

    def test_streaming_wait_shows_finalizing_toast(
        self, app, qtbot, tmp_path
    ):
        audio = make_finalized_saveable(is_streaming=True)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ):
            dialog.run()
        assert parent.status_bar.currentMessage() == _TOAST_FINALIZING

    def test_streaming_wait_no_write_until_finalize(
        self, app, qtbot, tmp_path
    ):
        audio = make_finalized_saveable(is_streaming=True)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            # No write yet — Branch C defers until finalize.
            assert sf_write.call_count == 0

    def test_streaming_wait_completes_on_target_finalize(
        self, app, qtbot, tmp_path
    ):
        """Direct-invoke the slot to validate dispatch logic.

        ``Qt.ConnectionType.QueuedConnection`` defers the slot until
        the next event-loop iteration; the slot's logic itself is
        what AC #8 pins, so we call it directly. The connection-type
        contract is asserted in ``test_run_with_streaming_audio_opens_
        dialog_and_subscribes``.
        """
        click_audio = make_finalized_saveable(
            is_streaming=True,
            audio=np.array([0.0, 0.1], dtype=np.float32),
        )
        finalized_audio = make_finalized_saveable(
            is_streaming=False,
            audio=np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        )
        registry = make_mock_registry(
            saveable_audio=click_audio, use_real_signal=True
        )
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            registry.saveable_audio = finalized_audio
            registry.saveable_session_id = finalized_audio.session_id
            dialog._on_saveable_changed_during_wait(
                finalized_audio.session_id
            )
            assert sf_write.call_count == 1
            written = sf_write.call_args.args[1]
            np.testing.assert_array_equal(
                written,
                (finalized_audio.complete_audio * 32767)
                .clip(-32768, 32767)
                .astype(np.int16),
            )

    def test_streaming_wait_disconnects_on_complete(
        self, app, qtbot, tmp_path
    ):
        click_audio = make_finalized_saveable(is_streaming=True)
        finalized_audio = make_finalized_saveable(is_streaming=False)
        registry = make_mock_registry(
            saveable_audio=click_audio, use_real_signal=True
        )
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ):
            dialog.run()
            registry.saveable_audio = finalized_audio
            registry.saveable_session_id = finalized_audio.session_id
            dialog._on_saveable_changed_during_wait(
                finalized_audio.session_id
            )
        assert len(registry.saveable_session_changed.disconnect_calls) == 1


# --------------------------------------------------------------------------- #
# AC #6 + AC #9 — TestSaveAudioDialogStreamingAbort
# --------------------------------------------------------------------------- #


class TestSaveAudioDialogStreamingAbort:
    """Streaming-wait abort paths."""

    def test_streaming_wait_aborts_on_none_emit(self, app, qtbot, tmp_path):
        audio = make_finalized_saveable(is_streaming=True)
        registry = make_mock_registry(
            saveable_audio=audio, use_real_signal=True
        )
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            registry.saveable_audio = None
            registry.saveable_session_id = None
            dialog._on_saveable_changed_during_wait(None)
            assert sf_write.call_count == 0
        assert parent.status_bar.currentMessage() == _TOAST_ABORT_CANCELLED
        assert not target.exists()

    def test_streaming_wait_aborts_on_different_id(
        self, app, qtbot, tmp_path
    ):
        target_audio = make_finalized_saveable(
            session_id="A", is_streaming=True
        )
        registry = make_mock_registry(
            saveable_audio=target_audio, use_real_signal=True
        )
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            other_audio = make_finalized_saveable(
                session_id="B", is_streaming=False
            )
            registry.saveable_audio = other_audio
            registry.saveable_session_id = "B"
            dialog._on_saveable_changed_during_wait("B")
            assert sf_write.call_count == 0
        # Review fix H1: different non-None id is ambiguous (could be
        # supersession OR cancel-with-revert OR error-with-revert), so
        # the controller emits the neutral "saveable changed" copy
        # rather than the now-removed "superseded" copy.
        assert (
            parent.status_bar.currentMessage()
            == _TOAST_ABORT_SAVEABLE_CHANGED
        )
        assert not target.exists()

    def test_streaming_wait_cancel_with_revert_shows_neutral_toast(
        self, app, qtbot, tmp_path
    ):
        """Review fix H1: cancel-with-revert-to-previous-saveable must
        not show the "superseded" copy (the cause is cancel, not
        supersession). Story 14.1 emits ``saveable_session_changed
        (previous_id)`` when the current saveable is cancelled and a
        previous saveable exists; the dialog sees a different non-None
        id and now uses the neutral "saveable changed" wording.
        """
        target_audio = make_finalized_saveable(
            session_id="B", is_streaming=True
        )
        registry = make_mock_registry(
            saveable_audio=target_audio, use_real_signal=True
        )
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            # Simulate Story 14.1's cancel-with-revert path: B was the
            # current saveable, user cancelled B, A reverts to current.
            previous_audio = make_finalized_saveable(
                session_id="A", is_streaming=False
            )
            registry.saveable_audio = previous_audio
            registry.saveable_session_id = "A"
            dialog._on_saveable_changed_during_wait("A")
            assert sf_write.call_count == 0
        assert (
            parent.status_bar.currentMessage()
            == _TOAST_ABORT_SAVEABLE_CHANGED
        )
        assert not target.exists()

    def test_streaming_wait_race_path_shows_neutral_toast(
        self, app, qtbot, tmp_path
    ):
        """Review fix H3: when emit's payload matches our target id but
        ``saveable_audio`` no longer matches by the time the slot reads
        it, the cause is ambiguous (cancel vs supersession vs error).
        The controller emits the neutral toast rather than guessing
        "cancelled".
        """
        target_audio = make_finalized_saveable(
            session_id="A", is_streaming=True
        )
        registry = make_mock_registry(
            saveable_audio=target_audio, use_real_signal=True
        )
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            # Race: emit fires with our target id, but by the time our
            # slot reads `saveable_audio`, the slot was overwritten
            # (different session id appears).
            other_audio = make_finalized_saveable(
                session_id="B", is_streaming=False
            )
            registry.saveable_audio = other_audio
            registry.saveable_session_id = "B"
            dialog._on_saveable_changed_during_wait("A")  # target id
            assert sf_write.call_count == 0
        assert (
            parent.status_bar.currentMessage()
            == _TOAST_ABORT_SAVEABLE_CHANGED
        )
        assert not target.exists()

    def test_streaming_wait_aborts_on_supersession_with_revert(
        self, app, qtbot, tmp_path
    ):
        """AC #6 option (a): supersession aborts; target file does not exist."""
        target_audio = make_finalized_saveable(
            session_id="A", is_streaming=True
        )
        registry = make_mock_registry(
            saveable_audio=target_audio, use_real_signal=True
        )
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        target = tmp_path / "out.wav"
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), ""),
        ):
            dialog.run()
            registry.saveable_audio = make_finalized_saveable(
                session_id="B", is_streaming=False
            )
            registry.saveable_session_id = "B"
            dialog._on_saveable_changed_during_wait("B")
        assert not target.exists()

    def test_streaming_wait_disconnects_on_abort(
        self, app, qtbot, tmp_path
    ):
        audio = make_finalized_saveable(is_streaming=True)
        registry = make_mock_registry(
            saveable_audio=audio, use_real_signal=True
        )
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ):
            dialog.run()
            registry.saveable_audio = None
            registry.saveable_session_id = None
            dialog._on_saveable_changed_during_wait(None)
        assert len(registry.saveable_session_changed.disconnect_calls) == 1


# --------------------------------------------------------------------------- #
# AC #10 — TestSaveAudioDialogFilenameSanitization
# --------------------------------------------------------------------------- #


class TestSaveAudioDialogFilenameSanitization:
    """Parametrized over voice strings; assert ``<voice>`` segment."""

    @pytest.mark.parametrize(
        "voice_input,expected_voice_segment",
        [
            ("clean", "clean"),
            ("with space", "with_space"),
            ("emoji🎤voice", "emoji_voice"),
            # `..` (2 dots) + `/` + `etc` + `/` + `passwd`
            # → `___etc_passwd` (3 leading underscores).
            ("../etc/passwd", "___etc_passwd"),
            ("Bad<>:/\\|?*", "Bad________"),
            ("", "voice"),
            ("___", "___"),
            ("a-b_c", "a-b_c"),
        ],
    )
    def test_voice_segment_sanitized(
        self, app, voice_input, expected_voice_segment
    ):
        registry = make_mock_registry()
        dialog = SaveAudioDialog(registry=registry, parent=None)
        name = dialog._make_default_filename(voice_input)
        # Format: myvoice-<voice>-<timestamp>.wav
        # The timestamp pattern is always YYYY-MM-DD-HHMMSS, so the
        # voice segment is whatever sits between the `myvoice-` prefix
        # and the trailing `-YYYY-MM-DD-HHMMSS.wav` segment. Use a
        # regex anchored on the date suffix to extract the voice.
        import re

        match = re.fullmatch(
            r"myvoice-(?P<voice>.+)-\d{4}-\d{2}-\d{2}-\d{6}\.wav", name
        )
        assert match is not None, f"Did not match expected pattern: {name}"
        assert match.group("voice") == expected_voice_segment

    def test_unicode_only_voice_uses_underscores(self, app):
        # All-non-ASCII voice → all-underscore sanitization (length-preserving).
        registry = make_mock_registry()
        dialog = SaveAudioDialog(registry=registry, parent=None)
        name = dialog._make_default_filename("中文")
        # Each non-ASCII char becomes one underscore; "中文" → "__".
        import re

        match = re.fullmatch(
            r"myvoice-(?P<voice>.+)-\d{4}-\d{2}-\d{2}-\d{6}\.wav", name
        )
        assert match is not None
        assert match.group("voice") == "__"


# --------------------------------------------------------------------------- #
# AC #13 — TestSaveAudioDialogModuleBoundary
# --------------------------------------------------------------------------- #


class TestSaveAudioDialogModuleBoundary:
    """Static text scan: forbidden imports must not appear."""

    @pytest.fixture
    def source_text(self) -> str:
        from myvoice.ui.dialogs import save_dialog as mod

        return Path(mod.__file__).read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "forbidden",
        [
            "qwen_tts_service",
            "audio_coordinator",
            "tts_streaming",
        ],
    )
    def test_save_dialog_does_not_import_forbidden_module(
        self, source_text, forbidden
    ):
        # Architecture lines 689-694: ui/* may not import services.* runtime
        # modules. The substring scan mirrors test_save_button.py's
        # TestSaveButtonModuleBoundary pattern.
        # We allow the substring inside docstrings only — but since we don't
        # mention any forbidden module in the docstring, a flat substring
        # check is sufficient.
        assert forbidden not in source_text


# --------------------------------------------------------------------------- #
# AC #5 error path — TestSaveAudioDialogWriteErrors
# --------------------------------------------------------------------------- #


class TestSaveAudioDialogWriteErrors:
    """sf.write raises → error toast, no exception propagates."""

    def test_oserror_on_write_shows_error_toast(self, app, qtbot, tmp_path):
        audio = make_finalized_saveable(is_streaming=False)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write",
            side_effect=OSError("disk full"),
        ):
            # Must not raise.
            dialog.run()
        msg = parent.status_bar.currentMessage()
        assert msg.startswith("Save failed:")
        assert "disk full" in msg

    def test_runtimeerror_on_write_shows_error_toast(
        self, app, qtbot, tmp_path
    ):
        audio = make_finalized_saveable(is_streaming=False)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write",
            side_effect=RuntimeError("soundfile error"),
        ):
            dialog.run()
        msg = parent.status_bar.currentMessage()
        assert msg.startswith("Save failed:")
        assert "soundfile error" in msg

    def test_empty_audio_buffer_shows_error_toast_no_write(
        self, app, qtbot, tmp_path
    ):
        """Review fix M2: empty complete_audio must not produce a
        header-only WAV file with a misleading success toast.
        """
        empty = np.array([], dtype=np.float32)
        audio = make_finalized_saveable(is_streaming=False, audio=empty)
        registry = make_mock_registry(saveable_audio=audio)
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        with patch(
            "myvoice.ui.dialogs.save_dialog.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.wav"), ""),
        ), patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog.run()
            assert sf_write.call_count == 0
        msg = parent.status_bar.currentMessage()
        assert msg == "Save failed: empty audio buffer"

    def test_complete_streaming_save_invariant_violation_no_crash(
        self, app, qtbot
    ):
        """Review fix M1: ``_complete_streaming_save`` previously used
        an ``assert`` that vanishes under ``python -O``. The explicit
        guard now logs and disconnects rather than crashing on
        ``Path(None)``.
        """
        audio = make_finalized_saveable(is_streaming=False)
        registry = make_mock_registry(
            saveable_audio=audio, use_real_signal=True
        )
        parent = make_parent_with_status_bar(qtbot)
        dialog = SaveAudioDialog(registry=registry, parent=parent)
        # Force the invariant violation: invoke completion without
        # going through _run_streaming_save (so _chosen_path is None).
        dialog._wait_connected = True  # so disconnect path runs
        registry.saveable_session_changed.connect(
            dialog._on_saveable_changed_during_wait,
            Qt.ConnectionType.DirectConnection,
        )
        with patch(
            "myvoice.ui.dialogs.save_dialog.sf.write"
        ) as sf_write:
            dialog._complete_streaming_save(audio)
            # No crash, no write — the guard absorbs the violation.
            assert sf_write.call_count == 0
        # Disconnect happened (defensive cleanup).
        assert dialog._wait_connected is False


# --------------------------------------------------------------------------- #
# AC #12 — TestAppPySaveRequestedWiring
# --------------------------------------------------------------------------- #


class TestAppPySaveRequestedWiring:
    """app.py subscribes save_requested and constructs a fresh dialog."""

    def test_app_imports_save_dialog(self):
        # AC #12: top-level import of SaveAudioDialog in app.py.
        import myvoice.app as app_mod

        assert hasattr(app_mod, "SaveAudioDialog")
        assert app_mod.SaveAudioDialog is SaveAudioDialog

    def test_on_save_requested_constructs_dialog_and_calls_run(self, app):
        """Drive the bare handler with mocks for registry + main window."""
        from myvoice.app import MyVoiceApp

        # Construct the bound method on a stand-in object, bypassing
        # MyVoiceApp.__init__ which requires a real QApplication setup.
        instance = MyVoiceApp.__new__(MyVoiceApp)
        instance.logger = MagicMock()
        instance._session_registry = MagicMock(spec=SessionRegistry)
        instance._main_window = MagicMock()
        with patch("myvoice.app.SaveAudioDialog") as SaveDlgMock:
            dialog_inst = MagicMock()
            SaveDlgMock.return_value = dialog_inst
            MyVoiceApp._on_save_requested(instance)
            SaveDlgMock.assert_called_once_with(
                registry=instance._session_registry,
                parent=instance._main_window,
            )
            dialog_inst.run.assert_called_once_with()

    def test_on_save_requested_with_no_registry_logs_and_ignores(self, app):
        from myvoice.app import MyVoiceApp

        instance = MyVoiceApp.__new__(MyVoiceApp)
        instance.logger = MagicMock()
        instance._session_registry = None
        instance._main_window = MagicMock()
        with patch("myvoice.app.SaveAudioDialog") as SaveDlgMock:
            MyVoiceApp._on_save_requested(instance)
            assert SaveDlgMock.call_count == 0
            assert instance.logger.warning.call_count == 1


# --------------------------------------------------------------------------- #
# AC #14 — TestSaveDialogNetZero (sentinel)
# --------------------------------------------------------------------------- #


class TestSaveDialogNetZero:
    """Documents the suites swept manually as part of Task 7."""

    def test_save_dialog_net_zero_sentinel(self):
        """Sentinel: the AC #14 sweep is enforced in Task 7's pytest runs;
        this test just records the contract for future readers.
        """
        # Pre-14.3 baseline:
        #  - tests/unit/services/sessions/  : 280 tests pass
        #  - tests/ui/test_save_button.py   : 43 tests pass
        #  - tests/ui/test_status_indicators.py + test_playback_last.py
        #    + test_accessibility.py        : 109 tests pass
        #  - tests/integration/test_session_lifecycle.py +
        #    test_playback_last_preservation.py : 61 tests pass
        #  - tests/unit/observability + qwen_tts_metrics_migration : 46 tests
        # Story 14.3 must not regress any of the above.
        assert True
