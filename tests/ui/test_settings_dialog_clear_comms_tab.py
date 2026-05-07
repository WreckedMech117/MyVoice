"""Integration tests for the Clear Comms tab inside SettingsDialog.

Story 15.3 — covers AC #2 (tab presence + ordering + signal forward),
AC #5 (signal-chain re-emission), AC #7 (OK persistence + Cancel
losslessness), AC #9 (load/save current_settings), AC #10 (reset to
defaults).

The D-5 invariant (AC #11) lives in
``tests/integration/test_clear_comms_d5_invariant.py``; the panel
widget tests live in
``tests/ui/dialogs/settings/test_clear_comms_settings_panel.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("PyQt6")

import soundfile as sf  # noqa: E402

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox, QTabWidget  # noqa: E402

from myvoice.models.app_settings import AppSettings  # noqa: E402
from myvoice.ui.components.settings_dialog import SettingsDialog  # noqa: E402
from myvoice.ui.dialogs.settings.clear_comms_settings_panel import (  # noqa: E402
    SOURCE_FILE,
    SOURCE_LAST_GENERATION,
    ClearCommsSettingsPanel,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def default_settings(tmp_path):
    """Real AppSettings instance pointing at a temp config dir.

    Real (not MagicMock) because the dialog calls
    ``AppSettings.from_dict(self.app_settings.to_dict())`` to make the
    current_settings snapshot, and the panel needs the three
    ``clear_comms_*`` fields to round-trip through ``to_dict``/``from_dict``.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    settings = AppSettings(
        config_directory=str(config_dir),
        voice_files_directory=str(voice_dir),
    )
    return settings


@pytest.fixture
def valid_wav(tmp_path):
    path = tmp_path / "callout.wav"
    sample_rate = 24000
    samples = 12000
    t = np.linspace(0, samples / sample_rate, samples, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(path), audio, sample_rate, format="WAV", subtype="PCM_16")
    return path


@pytest.fixture
def dialog(qapp, default_settings, qtbot):
    quick_speak_stub = MagicMock()
    quick_speak_stub.load_entries = MagicMock()
    dlg = SettingsDialog(
        default_settings,
        parent=None,
        quick_speak_service=quick_speak_stub,
    )
    qtbot.addWidget(dlg)
    return dlg


# --------------------------------------------------------------------------- #
# AC #2 — tab presence + ordering + panel instance
# --------------------------------------------------------------------------- #


class TestClearCommsTabPresence:
    def test_clear_comms_tab_is_last(self, dialog):
        tab_widget: QTabWidget = dialog.tab_widget
        assert tab_widget.count() >= 6  # Audio, Voices, TTS, Interface, Quick Speak, Clear Comms
        last_idx = tab_widget.count() - 1
        assert tab_widget.tabText(last_idx) == "Clear Comms"

    def test_clear_comms_tab_widget_is_panel_instance(self, dialog):
        tab_widget: QTabWidget = dialog.tab_widget
        last_idx = tab_widget.count() - 1
        widget = tab_widget.widget(last_idx)
        assert isinstance(widget, ClearCommsSettingsPanel)
        assert widget is dialog.clear_comms_panel

    def test_dialog_exposes_clear_comms_test_playback_signal(self, dialog):
        # AC #2 — class-level signal exists at runtime.
        assert hasattr(dialog, "clear_comms_test_playback_requested")


# --------------------------------------------------------------------------- #
# AC #5 — Test Playback signal chain (panel → dialog re-emit)
# --------------------------------------------------------------------------- #


class TestSignalForward:
    def test_panel_signal_re_emits_through_dialog(self, dialog):
        emissions: List[Tuple[str, Optional[str], bool]] = []
        dialog.clear_comms_test_playback_requested.connect(
            lambda sk, fp, qm: emissions.append((sk, fp, qm))
        )
        dialog.clear_comms_panel.test_playback_requested.emit(
            SOURCE_LAST_GENERATION, None, False
        )
        assert emissions == [(SOURCE_LAST_GENERATION, None, False)]

    def test_panel_signal_with_file_payload(self, dialog, valid_wav):
        emissions: List[Tuple[str, Optional[str], bool]] = []
        dialog.clear_comms_test_playback_requested.connect(
            lambda sk, fp, qm: emissions.append((sk, fp, qm))
        )
        dialog.clear_comms_panel.test_playback_requested.emit(
            SOURCE_FILE, str(valid_wav), True
        )
        assert emissions == [(SOURCE_FILE, str(valid_wav), True)]


# --------------------------------------------------------------------------- #
# AC #7 — OK persists panel state; Cancel does not
# --------------------------------------------------------------------------- #


class TestOkPersistence:
    def test_save_current_settings_pushes_panel_state_into_current_settings(
        self, dialog, valid_wav
    ):
        dialog.clear_comms_panel.load_state(
            source_kind=SOURCE_FILE,
            file_path=str(valid_wav),
            queue_mode=True,
        )
        dialog._save_current_settings()
        assert dialog.current_settings.clear_comms_source_kind == SOURCE_FILE
        assert dialog.current_settings.clear_comms_file_path == str(valid_wav)
        assert dialog.current_settings.clear_comms_queue_mode is True

    def test_load_current_settings_hydrates_panel_from_current_settings(
        self, dialog, valid_wav
    ):
        # Mutate current_settings (the snapshot the dialog edits) and
        # re-hydrate the panel from it.
        dialog.current_settings.clear_comms_source_kind = SOURCE_FILE
        dialog.current_settings.clear_comms_file_path = str(valid_wav)
        dialog.current_settings.clear_comms_queue_mode = True
        dialog._load_current_settings()
        sk, fp, qm = dialog.clear_comms_panel.save_state()
        assert sk == SOURCE_FILE
        assert fp == str(valid_wav)
        assert qm is True


# --------------------------------------------------------------------------- #
# Cancel ergonomics — Cancel does not mutate persisted state
# --------------------------------------------------------------------------- #


class TestCancelLosslessness:
    def test_cancel_button_does_not_emit_settings_changed(
        self, dialog
    ):
        emissions: List[AppSettings] = []
        dialog.settings_changed.connect(lambda s: emissions.append(s))
        # Mutate the panel without calling _save_current_settings.
        dialog.clear_comms_panel.load_state(
            source_kind=SOURCE_FILE,
            file_path="C:/elsewhere/x.wav",
            queue_mode=True,
        )
        # Click Cancel (calls reject()).
        dialog.cancel_button.click()
        assert emissions == []

    def test_persisted_settings_unchanged_after_panel_mutation_then_cancel(
        self, dialog, default_settings
    ):
        # The dialog edits its own current_settings snapshot, not
        # default_settings — that's the mechanism that makes Cancel
        # lossless. Mutate the panel and Cancel; assert the upstream
        # default_settings is untouched.
        original_source = default_settings.clear_comms_source_kind
        original_path = default_settings.clear_comms_file_path
        original_queue = default_settings.clear_comms_queue_mode

        dialog.clear_comms_panel.load_state(
            source_kind=SOURCE_FILE,
            file_path="C:/elsewhere/y.wav",
            queue_mode=True,
        )
        dialog.cancel_button.click()

        assert default_settings.clear_comms_source_kind == original_source
        assert default_settings.clear_comms_file_path == original_path
        assert default_settings.clear_comms_queue_mode == original_queue


# --------------------------------------------------------------------------- #
# AC #10 — Reset to Defaults restores Clear Comms fields
# --------------------------------------------------------------------------- #


class TestResetToDefaults:
    def test_reset_restores_panel_to_defaults(
        self, dialog, valid_wav, monkeypatch
    ):
        # Mutate panel to non-default state.
        dialog.clear_comms_panel.load_state(
            source_kind=SOURCE_FILE,
            file_path=str(valid_wav),
            queue_mode=True,
        )
        # Confirm it took.
        sk, fp, qm = dialog.clear_comms_panel.save_state()
        assert (sk, fp, qm) == (SOURCE_FILE, str(valid_wav), True)

        # Stub the confirmation dialog to auto-accept (Yes) AND avoid
        # the checkbox path (Quick Speak reset).
        def fake_exec(self_msg):
            return QMessageBox.StandardButton.Yes

        monkeypatch.setattr(QMessageBox, "exec", fake_exec)
        # Stub the voice selector so reset_to_defaults doesn't blow up
        # if the dialog wasn't fully populated by the parent.
        if dialog.voice_selector is None:
            pass  # Path: no voice selector, code already guards it.
        # Stub the device combos to avoid a None dereference.
        if not hasattr(dialog, "device_combo") or dialog.device_combo is None:
            pass

        dialog._on_reset_defaults()

        sk, fp, qm = dialog.clear_comms_panel.save_state()
        assert sk == SOURCE_LAST_GENERATION
        assert fp is None
        assert qm is False
