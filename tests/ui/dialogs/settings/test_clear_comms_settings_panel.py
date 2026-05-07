"""Widget unit tests for Story 15.3: ClearCommsSettingsPanel.

Covers AC #1, #3, #4, #5, #6, plus accessibility audit (AC #12).
The settings-dialog integration tests live in
``tests/ui/test_settings_dialog_clear_comms_tab.py``; the D-5
invariant integration test lives in
``tests/integration/test_clear_comms_d5_invariant.py``.

The tests use ``qtbot`` (pytest-qt) for widget cleanup and avoid
calling slot methods directly when an event-driven equivalent exists.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest

pytest.importorskip("PyQt6")

import soundfile as sf  # noqa: E402

from PyQt6.QtCore import pyqtBoundSignal  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QWidget,
)

from myvoice.ui.dialogs.settings import ClearCommsSettingsPanel as PanelFromPkg  # noqa: E402
from myvoice.ui.dialogs.settings.clear_comms_settings_panel import (  # noqa: E402
    SOURCE_FILE,
    SOURCE_LAST_GENERATION,
    WAV_FILE_DIALOG_FILTER,
    ClearCommsSettingsPanel,
    PreloadedAudioLoadError,
    _PLACEHOLDER_NO_FILE,
    _TOOLTIP_TEST_PLAYBACK_ENABLED,
    _TOOLTIP_TEST_PLAYBACK_INVALID_FILE,
    _TOOLTIP_TEST_PLAYBACK_NO_SOURCE,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def app(qtbot):
    """Ensure QApplication exists (mirrors test_save_button.py)."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qtbot):
    """Fresh ClearCommsSettingsPanel registered with qtbot for cleanup."""
    widget = ClearCommsSettingsPanel()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def valid_wav_factory(tmp_path):
    """Factory: ``valid_wav_factory(sample_rate=24000, samples=24000)``
    writes a sine WAV to ``tmp_path`` and returns its absolute path.
    """
    counter = {"n": 0}

    def _make(sample_rate: int = 24000, samples: int = 24000) -> Path:
        counter["n"] += 1
        path = tmp_path / f"valid_{counter['n']}_{sample_rate}.wav"
        t = np.linspace(0, samples / sample_rate, samples, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(str(path), audio, sample_rate, format="WAV", subtype="PCM_16")
        return path

    return _make


@pytest.fixture
def stereo_wav(tmp_path):
    # Filename intentionally avoids "stereo"/"channel"/"mix" substrings —
    # the AC #4 regression check searches the rendered status label for
    # those words, and the loader echoes Path(file_path).name verbatim.
    path = tmp_path / "two_track_clip.wav"
    sample_rate = 44100
    samples = 22050
    t = np.linspace(0, samples / sample_rate, samples, endpoint=False)
    left = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 660.0 * t)).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    sf.write(str(path), stereo, sample_rate, format="WAV", subtype="PCM_16")
    return path


@pytest.fixture
def corrupt_wav(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"NOT A REAL WAV FILE - just plain text bytes.")
    return path


@pytest.fixture
def spoofed_mp3(tmp_path):
    """A non-WAV file renamed with a .wav extension. Loader must reject."""
    real_mp3_bytes = b"\xff\xfb\x90\x00" + b"\x00" * 1024  # MPEG header + filler
    path = tmp_path / "spoofed.wav"
    path.write_bytes(real_mp3_bytes)
    return path


# --------------------------------------------------------------------------- #
# AC #1 — Construction and public API
# --------------------------------------------------------------------------- #


class TestPanelConstruction:
    """AC #1: panel exposes the documented public API and is a QWidget."""

    def test_panel_is_qwidget_subclass(self):
        assert issubclass(ClearCommsSettingsPanel, QWidget)

    def test_panel_constructs_with_no_state(self, panel):
        # The panel renders without any caller invoking load_state — the
        # constructor's defaults match AppSettings v1 defaults so the UI
        # is coherent on first paint.
        assert panel.findChild(QButtonGroup, "clear_comms_source_group") is not None
        assert (
            panel.findChild(QRadioButton, "clear_comms_source_last_generation")
            is not None
        )
        assert panel.findChild(QRadioButton, "clear_comms_source_file") is not None
        assert (
            panel.findChild(QLineEdit, "clear_comms_file_path_display") is not None
        )
        assert panel.findChild(QPushButton, "clear_comms_browse_button") is not None
        assert panel.findChild(QLabel, "clear_comms_file_status") is not None
        assert (
            panel.findChild(QCheckBox, "clear_comms_queue_mode_checkbox") is not None
        )
        assert (
            panel.findChild(QPushButton, "clear_comms_test_playback_button")
            is not None
        )

    def test_panel_test_playback_signal_is_pyqt_signal(self, panel):
        sig = panel.test_playback_requested
        # When accessed on an instance, pyqtSignal returns a bound signal.
        assert isinstance(sig, pyqtBoundSignal)

    def test_panel_default_internal_state_matches_appsettings_defaults(self, panel):
        source_kind, file_path, queue_mode = panel.save_state()
        assert source_kind == SOURCE_LAST_GENERATION
        assert file_path is None
        assert queue_mode is False


class TestPanelReexport:
    def test_clear_comms_settings_panel_reexport(self):
        # AC #1 — the package re-export resolves to the same class.
        assert PanelFromPkg is ClearCommsSettingsPanel


# --------------------------------------------------------------------------- #
# AC #1 — load_state / save_state round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source_kind, file_path, queue_mode",
    [
        (SOURCE_LAST_GENERATION, None, False),
        (SOURCE_LAST_GENERATION, None, True),
        (SOURCE_FILE, None, False),
        (SOURCE_FILE, None, True),
        (SOURCE_LAST_GENERATION, "C:/sounds/legacy.wav", False),
        (SOURCE_FILE, "C:/sounds/legacy.wav", True),
    ],
)
def test_panel_load_state_save_state_round_trip(
    panel, source_kind, file_path, queue_mode
):
    """AC #1 — load_state(...) → save_state() returns the same tuple."""
    panel.load_state(
        source_kind=source_kind,
        file_path=file_path,
        queue_mode=queue_mode,
    )
    out = panel.save_state()
    assert out == (source_kind, file_path, queue_mode)


# --------------------------------------------------------------------------- #
# AC #3 — source-selection radio group
# --------------------------------------------------------------------------- #


class TestSourceRadios:
    def test_default_radio_is_last_generation(self, panel):
        radio_lg = panel.findChild(
            QRadioButton, "clear_comms_source_last_generation"
        )
        radio_file = panel.findChild(QRadioButton, "clear_comms_source_file")
        assert radio_lg.isChecked() is True
        assert radio_file.isChecked() is False

    def test_radio_toggle_hides_picker_row(self, panel):
        radio_file = panel.findChild(QRadioButton, "clear_comms_source_file")
        radio_lg = panel.findChild(
            QRadioButton, "clear_comms_source_last_generation"
        )
        # Initial: last_generation → picker hidden.
        assert panel._file_picker_row.isVisibleTo(panel) is False
        # Switch to file → visible.
        radio_file.setChecked(True)
        assert panel._file_picker_row.isVisibleTo(panel) is True
        # Switch back → hidden.
        radio_lg.setChecked(True)
        assert panel._file_picker_row.isVisibleTo(panel) is False

    def test_radio_toggle_does_not_emit_signal(self, panel):
        emissions: List[Tuple[str, Optional[str], bool]] = []
        panel.test_playback_requested.connect(
            lambda sk, fp, qm: emissions.append((sk, fp, qm))
        )
        radio_file = panel.findChild(QRadioButton, "clear_comms_source_file")
        radio_lg = panel.findChild(
            QRadioButton, "clear_comms_source_last_generation"
        )
        radio_file.setChecked(True)
        radio_lg.setChecked(True)
        radio_file.setChecked(True)
        assert emissions == []

    def test_load_state_file_source_with_path_shows_picker_and_path(
        self, panel, valid_wav_factory
    ):
        path = valid_wav_factory()
        panel.load_state(
            source_kind=SOURCE_FILE, file_path=str(path), queue_mode=False
        )
        radio_file = panel.findChild(QRadioButton, "clear_comms_source_file")
        radio_lg = panel.findChild(
            QRadioButton, "clear_comms_source_last_generation"
        )
        display = panel.findChild(QLineEdit, "clear_comms_file_path_display")
        assert radio_file.isChecked() is True
        assert radio_lg.isChecked() is False
        assert panel._file_picker_row.isVisibleTo(panel) is True
        assert display.text() == str(path)

    def test_path_display_placeholder_when_no_file(self, panel):
        panel.load_state(
            source_kind=SOURCE_FILE, file_path=None, queue_mode=False
        )
        display = panel.findChild(QLineEdit, "clear_comms_file_path_display")
        assert display.text() == ""
        assert display.placeholderText() == _PLACEHOLDER_NO_FILE


# --------------------------------------------------------------------------- #
# AC #4 — file-picker row + live validity probe + status label
# --------------------------------------------------------------------------- #


class TestFilePicker:
    def test_browse_button_opens_file_dialog_with_wav_filter(
        self, panel, valid_wav_factory, monkeypatch
    ):
        path = valid_wav_factory()
        captured = {}

        def fake_get_open_file_name(parent, caption, directory, filter_):
            captured["parent"] = parent
            captured["caption"] = caption
            captured["directory"] = directory
            captured["filter"] = filter_
            return str(path), filter_

        monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_get_open_file_name)
        panel._browse_button.click()

        assert captured["filter"] == WAV_FILE_DIALOG_FILTER
        assert captured["caption"] == "Choose Clear Comms audio file"
        assert captured["parent"] is panel
        # Path is recorded in panel state and display.
        sk, fp, _qm = panel.save_state()
        assert fp == str(path)
        display = panel.findChild(QLineEdit, "clear_comms_file_path_display")
        assert display.text() == str(path)

    def test_browse_cancel_does_not_change_state(self, panel, monkeypatch):
        # User-cancel returns ("", "") from QFileDialog.
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            lambda *a, **k: ("", ""),
        )
        # Pre-populate state to assert non-mutation on cancel.
        panel.load_state(
            source_kind=SOURCE_FILE,
            file_path="C:/already/configured.wav",
            queue_mode=True,
        )
        before = panel.save_state()
        panel._browse_button.click()
        after = panel.save_state()
        assert after == before

    @pytest.mark.parametrize("sample_rate", [8000, 22050, 44100, 48000])
    def test_status_label_shows_ready_message_for_valid_wav(
        self, panel, valid_wav_factory, sample_rate
    ):
        path = valid_wav_factory(sample_rate=sample_rate, samples=sample_rate)
        panel.load_state(
            source_kind=SOURCE_FILE, file_path=str(path), queue_mode=False
        )
        label = panel.findChild(QLabel, "clear_comms_file_status")
        text = label.text()
        # Full match with thousands-separator formatting per AC #4.
        pattern = re.compile(
            r"^Ready: \w+\.wav \(\d{1,3}(,\d{3})* samples, \d+ Hz\)$"
        )
        assert pattern.match(text), f"Unexpected status text: {text!r}"
        assert label.property("class") == "status-ok"

    def test_status_label_shows_loader_error_for_corrupt_file(
        self, panel, corrupt_wav
    ):
        panel.load_state(
            source_kind=SOURCE_FILE, file_path=str(corrupt_wav), queue_mode=False
        )
        label = panel.findChild(QLabel, "clear_comms_file_status")
        # Loader's user-facing message for a corrupt WAV.
        assert "corrupt" in label.text().lower() or "could not read" in label.text().lower()
        assert label.property("class") == "status-error"

    def test_status_label_shows_loader_error_for_missing_file(
        self, panel, tmp_path
    ):
        missing_path = tmp_path / "does_not_exist.wav"
        panel.load_state(
            source_kind=SOURCE_FILE,
            file_path=str(missing_path),
            queue_mode=False,
        )
        label = panel.findChild(QLabel, "clear_comms_file_status")
        # Loader emits "File not found: <name>".
        assert "not found" in label.text().lower()
        assert label.property("class") == "status-error"

    def test_status_label_shows_loader_error_for_spoofed_mp3(
        self, panel, spoofed_mp3
    ):
        panel.load_state(
            source_kind=SOURCE_FILE,
            file_path=str(spoofed_mp3),
            queue_mode=False,
        )
        label = panel.findChild(QLabel, "clear_comms_file_status")
        assert label.property("class") == "status-error"
        # Spoofed file passes the .wav extension check; soundfile rejects it.
        # Loader wraps as "Could not read WAV file ... corrupt or not a valid WAV"
        text_lower = label.text().lower()
        assert "could not read" in text_lower or "corrupt" in text_lower

    def test_status_label_does_not_mention_stereo_downmix(
        self, panel, stereo_wav
    ):
        panel.load_state(
            source_kind=SOURCE_FILE, file_path=str(stereo_wav), queue_mode=False
        )
        label = panel.findChild(QLabel, "clear_comms_file_status")
        text_lower = label.text().lower()
        assert text_lower.startswith("ready:")
        # Regression guard against future "helpfulness" leaks.
        for forbidden in ("stereo", "channel", "mix", "downmix"):
            assert forbidden not in text_lower, (
                f"Status text leaked stereo info: {label.text()!r} contained {forbidden!r}"
            )

    def test_status_label_clears_on_no_file(self, panel):
        panel.load_state(
            source_kind=SOURCE_FILE, file_path=None, queue_mode=False
        )
        label = panel.findChild(QLabel, "clear_comms_file_status")
        assert label.text() == ""

    def test_path_display_is_read_only(self, panel):
        display = panel.findChild(QLineEdit, "clear_comms_file_path_display")
        assert display.isReadOnly() is True


# --------------------------------------------------------------------------- #
# AC #5 — Test Playback button
# --------------------------------------------------------------------------- #


class TestTestPlaybackButton:
    def test_test_playback_button_emits_signal_with_panel_state(
        self, panel
    ):
        emissions: List[Tuple[str, Optional[str], bool]] = []
        panel.test_playback_requested.connect(
            lambda sk, fp, qm: emissions.append((sk, fp, qm))
        )
        panel.load_state(
            source_kind=SOURCE_LAST_GENERATION,
            file_path=None,
            queue_mode=False,
        )
        panel._test_playback_button.click()
        assert emissions == [(SOURCE_LAST_GENERATION, None, False)]

    def test_test_playback_button_uses_panel_state_not_persisted(
        self, panel, valid_wav_factory
    ):
        """The button payload is the *current widget state*, not the
        AppSettings — this is what makes the Cancel-button losslessness
        contract hold (AC #5).
        """
        path = valid_wav_factory()
        emissions: List[Tuple[str, Optional[str], bool]] = []
        panel.test_playback_requested.connect(
            lambda sk, fp, qm: emissions.append((sk, fp, qm))
        )
        panel.load_state(
            source_kind=SOURCE_FILE,
            file_path=str(path),
            queue_mode=True,
        )
        panel._test_playback_button.click()
        assert emissions == [(SOURCE_FILE, str(path), True)]

    def test_test_playback_button_enabled_for_last_generation(self, panel):
        panel.load_state(
            source_kind=SOURCE_LAST_GENERATION,
            file_path=None,
            queue_mode=False,
        )
        assert panel._test_playback_button.isEnabled() is True
        assert panel._test_playback_button.toolTip() == _TOOLTIP_TEST_PLAYBACK_ENABLED

    def test_test_playback_button_disabled_for_file_without_valid_path(
        self, panel
    ):
        panel.load_state(
            source_kind=SOURCE_FILE, file_path=None, queue_mode=False
        )
        assert panel._test_playback_button.isEnabled() is False
        assert (
            panel._test_playback_button.toolTip()
            == _TOOLTIP_TEST_PLAYBACK_INVALID_FILE
        )

    def test_test_playback_button_enabled_for_file_with_valid_path(
        self, panel, valid_wav_factory
    ):
        path = valid_wav_factory()
        panel.load_state(
            source_kind=SOURCE_FILE, file_path=str(path), queue_mode=False
        )
        assert panel._test_playback_button.isEnabled() is True
        assert panel._test_playback_button.toolTip() == _TOOLTIP_TEST_PLAYBACK_ENABLED

    def test_test_playback_button_disabled_for_invalid_file(
        self, panel, corrupt_wav
    ):
        panel.load_state(
            source_kind=SOURCE_FILE,
            file_path=str(corrupt_wav),
            queue_mode=False,
        )
        assert panel._test_playback_button.isEnabled() is False
        assert (
            panel._test_playback_button.toolTip()
            == _TOOLTIP_TEST_PLAYBACK_INVALID_FILE
        )

    def test_test_playback_button_disabled_for_unrecognized_source_kind(
        self, panel
    ):
        """Defensive default branch in _compute_test_playback_enabled
        (line 606) — guards against in-memory mutation between the
        AppSettings validator's auto-correct and a subsequent click.
        Coverage gap surfaced by review fix M2.
        """
        # Bypass load_state's coercion so the bogus kind survives.
        panel._source_kind = "bogus_kind_no_validator_caught_this"
        panel._file_path_valid = False
        panel._refresh_test_playback_enablement()
        assert panel._test_playback_button.isEnabled() is False
        assert (
            panel._test_playback_button.toolTip()
            == _TOOLTIP_TEST_PLAYBACK_NO_SOURCE
        )


# --------------------------------------------------------------------------- #
# AC #6 — queue-mode checkbox
# --------------------------------------------------------------------------- #


class TestQueueModeCheckbox:
    def test_queue_mode_default_unchecked(self, panel):
        # Fresh panel, no load_state — D-18 default = interrupt = unchecked.
        cb = panel.findChild(QCheckBox, "clear_comms_queue_mode_checkbox")
        assert cb.isChecked() is False

    def test_queue_mode_load_state_round_trip(self, panel):
        panel.load_state(
            source_kind=SOURCE_LAST_GENERATION, file_path=None, queue_mode=True
        )
        cb = panel.findChild(QCheckBox, "clear_comms_queue_mode_checkbox")
        assert cb.isChecked() is True
        cb.setChecked(False)
        _sk, _fp, qm = panel.save_state()
        assert qm is False

    def test_queue_mode_toggle_does_not_emit_signal(self, panel):
        emissions: List[Tuple[str, Optional[str], bool]] = []
        panel.test_playback_requested.connect(
            lambda sk, fp, qm: emissions.append((sk, fp, qm))
        )
        cb = panel.findChild(QCheckBox, "clear_comms_queue_mode_checkbox")
        cb.setChecked(True)
        cb.setChecked(False)
        cb.setChecked(True)
        assert emissions == []


# --------------------------------------------------------------------------- #
# Accessibility audit (AC #12 / #14)
# --------------------------------------------------------------------------- #


class TestAccessibility:
    def test_each_interactive_widget_has_accessible_metadata(self, panel):
        widgets_to_check = [
            panel.findChild(QRadioButton, "clear_comms_source_last_generation"),
            panel.findChild(QRadioButton, "clear_comms_source_file"),
            panel.findChild(QPushButton, "clear_comms_browse_button"),
            panel.findChild(QLineEdit, "clear_comms_file_path_display"),
            panel.findChild(QCheckBox, "clear_comms_queue_mode_checkbox"),
            panel.findChild(QPushButton, "clear_comms_test_playback_button"),
        ]
        for widget in widgets_to_check:
            assert widget is not None
            assert widget.accessibleName(), (
                f"{widget.objectName()} missing accessibleName()"
            )

    def test_radios_have_accessible_descriptions(self, panel):
        # AC #3 specifies verbatim accessible-description strings; pin
        # them exactly so a wording change anywhere is caught.
        radio_lg = panel.findChild(
            QRadioButton, "clear_comms_source_last_generation"
        )
        radio_file = panel.findChild(QRadioButton, "clear_comms_source_file")
        assert radio_lg.accessibleDescription() == (
            "Replay the most recently generated audio when Clear Comms is clicked."
        )
        assert radio_file.accessibleDescription() == (
            "Replay a chosen WAV file when Clear Comms is clicked."
        )

    def test_queue_checkbox_has_tooltip_explaining_default(self, panel):
        cb = panel.findChild(QCheckBox, "clear_comms_queue_mode_checkbox")
        # AC #6 spec: tooltip explains both states. The exact wording
        # ("Unchecked: ... interrupts ... (default). Checked: ... plays
        # after ...") does not contain the literal word "queue", but the
        # checkbox label does.
        assert "interrupt" in cb.toolTip().lower()
        assert "default" in cb.toolTip().lower()
        assert "after" in cb.toolTip().lower()


# --------------------------------------------------------------------------- #
# Probe behavior — reload after path becomes invalid
# --------------------------------------------------------------------------- #


class TestProbeReload:
    def test_load_state_with_stale_path_surfaces_error(self, panel, tmp_path):
        # File exists at construction; user deletes it; panel re-load
        # surfaces the loader error in the status label.
        path = tmp_path / "transient.wav"
        sample_rate = 24000
        samples = 24000
        t = np.linspace(0, samples / sample_rate, samples, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(str(path), audio, sample_rate, format="WAV", subtype="PCM_16")

        panel.load_state(
            source_kind=SOURCE_FILE, file_path=str(path), queue_mode=False
        )
        label = panel.findChild(QLabel, "clear_comms_file_status")
        assert label.text().startswith("Ready:")

        # Now delete the file and reload.
        path.unlink()
        panel.load_state(
            source_kind=SOURCE_FILE, file_path=str(path), queue_mode=False
        )
        assert "not found" in label.text().lower()
        assert panel._test_playback_button.isEnabled() is False
