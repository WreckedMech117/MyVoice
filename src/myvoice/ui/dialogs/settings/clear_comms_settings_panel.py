"""PRELOADED audio source loader + Clear Comms settings panel (Stories 15.1 + 15.3).

OFR-B Clear Comms — Epic 15. The loader (added by Story 15.1) is the
WAV-loading primitive that both the Clear Comms button (Story 15.2)
and the Settings panel (Story 15.3) consume; the
``ClearCommsSettingsPanel`` ``QWidget`` class (added by Story 15.3) is
the configuration surface where the user picks the source kind, file
path, and queue-vs-interrupt mode that Story 15.2's click handler then
reads from ``AppSettings``.

Story 15.3 added the ``ClearCommsSettingsPanel`` ``QWidget`` class
beneath the loader. The class consumes the same module's
``WAV_FILE_DIALOG_FILTER``, ``PreloadedAudioLoadError``, and
``load_preloaded_audio_source`` as the loader's only direct callers —
no other module needs to know the loader and the panel are
co-located.

Architecture decisions activated here:

  - **D-17** (line 282-283): "Audio loader for Clear Comms (OFR-B).
    ``soundfile.read()``, WAV-only in v1." → AC #1, #5, #6.
  - **D-5** (line 247-248): PRELOADED-clone exclusion is enforced
    upstream by Story 14.1's saveable-lifecycle policy. This module
    has no responsibility to assert it; the integration test in the
    test file verifies the loader's output composes correctly with
    the existing ``clone_for_replay(source=PRELOADED)`` zero-copy
    path (D-6).
  - **D-6** (line 249-250): zero-copy clone semantics — the audio
    array returned by this loader is the source-of-truth buffer that
    later replay-clones share by reference.
  - **Architecture file map line 704:** "PRELOADED-source loader inline
    in the panel." Interpreted here as "in the same module as the
    panel class" — top-level function rather than a method, so Story
    15.2's button click path can call it without instantiating a
    panel and so unit tests do not need a ``QApplication``.

Module-boundary discipline (architecture lines 689-694): ``ui/*`` may
import ``sessions.session_registry`` and ``sessions.generation_session``
only. For 15.1 specifically, neither is needed — the loader returns a
plain ``(audio, sample_rate)`` tuple. **No** PyQt6 imports in this
story; the panel UI lands in Story 15.3.

Sample-rate policy (resolves the architecture's "consistent rate must
be guaranteed — decision deferred to tech-spec"): **preserve the
file's native sample rate.** The downstream playback pipeline is
already rate-parameterized end-to-end —
``AudioCoordinator.start_streaming_sessions(sample_rate=...)``
propagates an arbitrary rate, the virtual-mic service auto-resamples
to its 48 kHz comms target, and the monitor service opens its stream
at the source rate. Adding resampling in the loader would duplicate
work the pipeline already does. If a future integration test surfaces
a playback issue at an unusual rate, a follow-up story adds
resampling — gated on empirical evidence, not preemptive complexity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

WAV_FILE_DIALOG_FILTER = "WAV files (*.wav)"

_logger = logging.getLogger(__name__)


class PreloadedAudioLoadError(Exception):
    """Raised when ``load_preloaded_audio_source`` cannot produce a
    valid ``(audio, sample_rate)`` tuple.

    Carries a user-facing ``message`` attribute (suitable for direct
    display in a Settings panel error label) plus an optional
    ``cause`` chained via ``__cause__`` so the loader's logger
    preserves the original traceback.
    """

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if cause is not None:
            # Own the chain end-to-end: setting both __cause__ and
            # __suppress_context__ matches what `raise X from Y` does, so
            # call sites don't need to repeat `from exc` (which would set
            # __cause__ a second time and risk divergence).
            self.__cause__ = cause
            self.__suppress_context__ = True

    def __str__(self) -> str:
        return self.message


def load_preloaded_audio_source(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file into a ``(mono float32 audio array, sample_rate)``
    tuple compatible with ``GenerationSession.complete_audio`` and
    ``GenerationSession.sample_rate``.

    Reentrant; safe to call concurrently with distinct paths.

    Raises:
        PreloadedAudioLoadError: when the file is missing, unreadable,
            corrupt, non-WAV, or contains audio that cannot be coerced
            into the expected (1-D float32) shape. The error's
            ``message`` is short, user-facing, and free of stack-trace
            artifacts.
    """
    # Probe stat once; wrap so OS-level failures (PermissionError on a
    # protected path) surface as PreloadedAudioLoadError, not raw OSError
    # the panel won't catch.
    try:
        is_file = path.is_file()
        file_exists = True if is_file else path.exists()
    except OSError as exc:
        _logger.error(
            "could not stat preloaded audio source: path=%s, error=%s",
            path,
            exc,
        )
        raise PreloadedAudioLoadError(
            f"Could not access file: {path.name}",
            cause=exc,
        )

    if not is_file:
        if not file_exists:
            # Missing file: chain a synthetic FileNotFoundError per AC #5
            # so the logger preserves the canonical exception type.
            notfound = FileNotFoundError(str(path))
            _logger.error(
                "preloaded audio source not found: path=%s",
                path,
            )
            raise PreloadedAudioLoadError(
                f"File not found: {path.name}",
                cause=notfound,
            )
        # Exists but not a regular file (directory, special). No log,
        # no cause — the loader's own validation surfaced this.
        raise PreloadedAudioLoadError(f"File not found: {path.name}")

    if path.suffix.lower() != ".wav":
        raise PreloadedAudioLoadError(
            f"Only WAV files are supported in this version. "
            f"Cannot load: {path.name}"
        )
    try:
        audio, sr = sf.read(str(path), always_2d=False, dtype="float32")
    except (FileNotFoundError, PermissionError) as exc:
        # TOCTOU: file vanished or became inaccessible between the stat
        # probe and sf.read. Surface as access failure so the user is not
        # misdirected toward "corrupt file" troubleshooting.
        _logger.error(
            "preloaded audio source vanished between check and read: path=%s, error=%s",
            path,
            exc,
        )
        raise PreloadedAudioLoadError(
            f"Could not access file: {path.name}",
            cause=exc,
        )
    except Exception as exc:
        _logger.error(
            "failed to read preloaded audio source: path=%s, error=%s",
            path,
            exc,
        )
        raise PreloadedAudioLoadError(
            f"Could not read WAV file: {path.name} "
            f"- file may be corrupt or not a valid WAV",
            cause=exc,
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    _logger.debug(
        "loaded preloaded audio source: path=%s, samples=%d, sr=%d, dtype=%s",
        path.name,
        len(audio),
        sr,
        audio.dtype,
    )
    return audio, int(sr)


# --------------------------------------------------------------------------- #
# Story 15.3: ClearCommsSettingsPanel
# --------------------------------------------------------------------------- #

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# Source-kind constants — single source of truth shared across the panel,
# the radio object names, and the AppSettings field. Story 15.2's
# AppSettings.validate uses the same string literals (see
# app_settings.py:347-359), but importing AppSettings here would couple
# the panel to the model layer, which AC #1 forbids — so the strings are
# duplicated as constants and the AppSettings validator's auto-correct
# rule covers any drift from a direct JSON edit.
SOURCE_LAST_GENERATION = "last_generation"
SOURCE_FILE = "file"

# Tooltip constants — single source of truth so tests can assert against
# the exact string and a wording change here propagates symmetrically.
_TOOLTIP_TEST_PLAYBACK_ENABLED = "Preview Clear Comms with the configured source"
_TOOLTIP_TEST_PLAYBACK_INVALID_FILE = "Pick a valid WAV file first"
_TOOLTIP_TEST_PLAYBACK_NO_SOURCE = "Configure a Clear Comms source first"

_TOOLTIP_QUEUE_MODE = (
    "Unchecked: Clear Comms interrupts the current playback (default). "
    "Checked: Clear Comms plays after the current playback finishes."
)

_PLACEHOLDER_NO_FILE = "(no file selected)"


class ClearCommsSettingsPanel(QWidget):
    """Settings tab body for Clear Comms (Story 15.3, OFR-B).

    Configures the three Clear Comms fields persisted in
    ``AppSettings``: source kind ("use last generation" or "use a file"),
    optional file path, and queue-vs-interrupt mode toggle.

    The panel is stateless w.r.t. the registry — the user can
    configure "use last generation" before any audio has been
    generated; the toolbar button's enablement (handled by Story 15.2's
    ``ClearCommsButton.set_state``) is what gates click activation.

    Signals:
        test_playback_requested(str, object, bool):
            Emitted only on Test Playback click; payload is
            ``(source_kind, file_path, queue_mode)`` where ``file_path``
            is ``Optional[str]`` typed as ``object`` because PyQt6's
            ``pyqtSignal`` cannot natively express a nullable ``str``.
            Routed through ``SettingsDialog`` →
            ``MyVoiceMainWindow`` → ``MyVoiceApp`` per AC #5.
    """

    test_playback_requested = pyqtSignal(str, object, bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)

        # Internal state — initialized to AppSettings v1 defaults so the
        # panel renders cleanly before any load_state call.
        self._source_kind: str = SOURCE_LAST_GENERATION
        self._file_path: Optional[str] = None
        self._file_path_valid: bool = False  # Last validity-probe verdict.

        self.setProperty("class", "settings-panel")

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(10)

        # --- Source-selection group (AC #3) ---------------------------- #
        self._source_group_box = QGroupBox("Clear Comms Source")
        source_layout = QVBoxLayout(self._source_group_box)

        self._source_button_group = QButtonGroup(self)
        self._source_button_group.setObjectName("clear_comms_source_group")
        self._source_button_group.setExclusive(True)

        self._radio_last_generation = QRadioButton("Use last generation")
        self._radio_last_generation.setObjectName(
            "clear_comms_source_last_generation"
        )
        self._radio_last_generation.setAccessibleName("Use last generation")
        self._radio_last_generation.setAccessibleDescription(
            "Replay the most recently generated audio when Clear Comms is clicked."
        )
        self._radio_last_generation.setChecked(True)
        self._source_button_group.addButton(self._radio_last_generation)
        source_layout.addWidget(self._radio_last_generation)

        self._radio_file = QRadioButton("Use a file")
        self._radio_file.setObjectName("clear_comms_source_file")
        self._radio_file.setAccessibleName("Use a file")
        self._radio_file.setAccessibleDescription(
            "Replay a chosen WAV file when Clear Comms is clicked."
        )
        self._source_button_group.addButton(self._radio_file)
        source_layout.addWidget(self._radio_file)

        outer_layout.addWidget(self._source_group_box)

        # --- File-picker row (AC #4) ----------------------------------- #
        # Hosted in a thin QWidget so the entire row (label + line edit +
        # button) toggles visibility together via setVisible.
        self._file_picker_row = QWidget()
        picker_layout = QHBoxLayout(self._file_picker_row)
        picker_layout.setContentsMargins(0, 0, 0, 0)

        picker_layout.addWidget(QLabel("Audio file:"))

        self._file_path_display = QLineEdit()
        self._file_path_display.setObjectName("clear_comms_file_path_display")
        self._file_path_display.setReadOnly(True)
        self._file_path_display.setPlaceholderText(_PLACEHOLDER_NO_FILE)
        self._file_path_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._file_path_display.setAccessibleName("Clear Comms audio file path")
        self._file_path_display.setAccessibleDescription(
            "Absolute path of the WAV file used by Clear Comms when the file source is selected."
        )
        picker_layout.addWidget(self._file_path_display)

        self._browse_button = QPushButton("Browse…")
        self._browse_button.setObjectName("clear_comms_browse_button")
        self._browse_button.setAccessibleName(
            "Browse for Clear Comms audio file"
        )
        picker_layout.addWidget(self._browse_button)

        outer_layout.addWidget(self._file_picker_row)

        # File-status label sits beneath the picker row.
        self._file_status_label = QLabel("")
        self._file_status_label.setObjectName("clear_comms_file_status")
        self._file_status_label.setWordWrap(True)
        self._file_status_label.setAccessibleName("Clear Comms file status")
        outer_layout.addWidget(self._file_status_label)

        # --- Queue-mode checkbox (AC #6) ------------------------------- #
        self._queue_mode_checkbox = QCheckBox(
            "Queue behind current playback (default: interrupt)"
        )
        self._queue_mode_checkbox.setObjectName(
            "clear_comms_queue_mode_checkbox"
        )
        self._queue_mode_checkbox.setAccessibleName("Clear Comms queue mode")
        self._queue_mode_checkbox.setAccessibleDescription(
            "When unchecked, Clear Comms interrupts the current playback. "
            "When checked, Clear Comms plays after the current playback finishes."
        )
        self._queue_mode_checkbox.setToolTip(_TOOLTIP_QUEUE_MODE)
        outer_layout.addWidget(self._queue_mode_checkbox)

        # --- Test Playback button (AC #5) ------------------------------ #
        test_row = QHBoxLayout()
        test_row.addStretch()
        self._test_playback_button = QPushButton("Test Playback")
        self._test_playback_button.setObjectName(
            "clear_comms_test_playback_button"
        )
        self._test_playback_button.setAccessibleName("Test Clear Comms Playback")
        test_row.addWidget(self._test_playback_button)
        outer_layout.addLayout(test_row)

        outer_layout.addStretch()

        # --- Wire signals after construction --------------------------- #
        self._radio_last_generation.toggled.connect(
            self._on_source_radio_toggled
        )
        self._radio_file.toggled.connect(self._on_source_radio_toggled)
        self._browse_button.clicked.connect(self._on_browse_clicked)
        self._queue_mode_checkbox.stateChanged.connect(
            self._on_queue_mode_toggled
        )
        self._test_playback_button.clicked.connect(
            self._on_test_playback_clicked
        )

        # Apply initial visibility + enablement from defaults.
        self._apply_source_visibility()
        self._refresh_test_playback_enablement()

    # ------------------------------------------------------------------ #
    # Caller-driven state I/O (AC #1)
    # ------------------------------------------------------------------ #

    def load_state(
        self,
        *,
        source_kind: str,
        file_path: Optional[str],
        queue_mode: bool,
    ) -> None:
        """Populate the panel from caller-supplied scalars.

        Mirrors how every other tab in ``SettingsDialog`` works: the
        dialog's ``current_settings: AppSettings`` snapshot is the
        source of truth; ``load_state`` pushes its three Clear Comms
        fields into the panel without coupling the panel to the
        ``AppSettings`` type.
        """
        self._source_kind = (
            SOURCE_FILE if source_kind == SOURCE_FILE else SOURCE_LAST_GENERATION
        )
        self._file_path = file_path

        # Block toggled signals while applying — otherwise checking a
        # radio fires _on_source_radio_toggled which would re-run the
        # validity probe twice (once here, once via the slot).
        self._radio_last_generation.blockSignals(True)
        self._radio_file.blockSignals(True)
        try:
            if self._source_kind == SOURCE_FILE:
                self._radio_file.setChecked(True)
            else:
                self._radio_last_generation.setChecked(True)
        finally:
            self._radio_last_generation.blockSignals(False)
            self._radio_file.blockSignals(False)

        self._queue_mode_checkbox.blockSignals(True)
        try:
            self._queue_mode_checkbox.setChecked(bool(queue_mode))
        finally:
            self._queue_mode_checkbox.blockSignals(False)

        self._update_file_display_text()
        self._apply_source_visibility()
        self._run_live_validity_probe(self._file_path)
        self._refresh_test_playback_enablement()

    def save_state(self) -> tuple[str, Optional[str], bool]:
        """Read panel state into ``(source_kind, file_path, queue_mode)``.

        Pure read; does NOT mutate ``AppSettings``. Idempotent.
        """
        return (
            self._source_kind,
            self._file_path,
            self._queue_mode_checkbox.isChecked(),
        )

    # ------------------------------------------------------------------ #
    # Internal slots / helpers
    # ------------------------------------------------------------------ #

    def _on_source_radio_toggled(self, checked: bool) -> None:
        """Qt fires ``toggled`` for both the newly-checked AND the
        newly-unchecked radio when a button group's selection moves;
        only react on the ``checked is True`` edge to avoid double work.
        """
        if not checked:
            return
        self._source_kind = (
            SOURCE_FILE
            if self._radio_file.isChecked()
            else SOURCE_LAST_GENERATION
        )
        self._apply_source_visibility()
        # Recompute validity for the file branch on switch-in so a stale
        # path that became invalid since the last visit surfaces the
        # error immediately rather than after the user re-clicks Browse.
        self._run_live_validity_probe(self._file_path)
        self._refresh_test_playback_enablement()

    def _on_browse_clicked(self) -> None:
        """Open the WAV-only file dialog (AC #4) and apply the result."""
        starting_dir = ""
        if self._file_path:
            try:
                starting_dir = str(Path(self._file_path).parent)
            except (OSError, ValueError):
                starting_dir = ""

        chosen_path, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose Clear Comms audio file",
            starting_dir,
            WAV_FILE_DIALOG_FILTER,
        )
        if not chosen_path:
            # User cancelled — no state change per AC #4.
            return

        self._file_path = chosen_path
        self._update_file_display_text()
        self._run_live_validity_probe(self._file_path)
        self._refresh_test_playback_enablement()

    def _on_queue_mode_toggled(self, _state: int) -> None:
        """Per AC #6, queue-mode toggling does NOT emit any signal
        upward. Internal state is maintained on the checkbox itself;
        ``save_state`` reads it on demand.
        """
        # No upward signal; no internal mirroring needed (the checkbox
        # IS the state). Recompute Test Playback enablement so the
        # panel can react if a future AC's enablement matrix grows to
        # depend on queue_mode.
        self._refresh_test_playback_enablement()

    def _on_test_playback_clicked(self) -> None:
        """Emit ``test_playback_requested`` with the current widget
        state (AC #5). NOT the persisted AppSettings — this is what
        makes the Cancel-button losslessness contract hold.
        """
        self.test_playback_requested.emit(
            self._source_kind,
            self._file_path,
            self._queue_mode_checkbox.isChecked(),
        )

    def _apply_source_visibility(self) -> None:
        """Hide the picker row + status label when the source is
        ``"last_generation"``; show them when it is ``"file"``.
        """
        is_file = self._source_kind == SOURCE_FILE
        self._file_picker_row.setVisible(is_file)
        self._file_status_label.setVisible(is_file)

    def _update_file_display_text(self) -> None:
        """Mirror ``self._file_path`` into the read-only display field.
        Empty path renders as the placeholder set at construction time.
        """
        self._file_path_display.setText(self._file_path or "")

    def _run_live_validity_probe(self, file_path: Optional[str]) -> None:
        """Synchronously call the 15.1 loader and update the status
        label with either a ``"Ready: …"`` message or the loader's
        verbatim error message (AC #4).

        Idempotent: with ``file_path is None`` the label is cleared.
        Re-applies the appropriate ``status-ok`` / ``status-error`` /
        no-class property so the theme manager renders the right color.
        """
        if not file_path:
            self._file_status_label.setText("")
            self._set_status_class(None)
            self._file_path_valid = False
            return

        try:
            audio, sample_rate = load_preloaded_audio_source(Path(file_path))
        except PreloadedAudioLoadError as exc:
            # Display the loader's user-facing message verbatim so the
            # panel does not editorialize. Tests guard against drift.
            self._file_status_label.setText(exc.message)
            self._set_status_class("status-error")
            self._file_path_valid = False
            return
        except Exception as exc:
            # Defensive: any unexpected exception (loader contract is
            # PreloadedAudioLoadError, but a future loader bug should
            # not crash the dialog). Surface a generic message.
            self._logger.exception(
                "Unexpected error during Clear Comms validity probe: %s", exc
            )
            self._file_status_label.setText("Could not validate file.")
            self._set_status_class("status-error")
            self._file_path_valid = False
            return

        samples = len(audio)
        name = Path(file_path).name
        self._file_status_label.setText(
            f"Ready: {name} ({samples:,} samples, {sample_rate} Hz)"
        )
        self._set_status_class("status-ok")
        self._file_path_valid = True

    def _set_status_class(self, css_class: Optional[str]) -> None:
        """Apply ``setProperty('class', ...)`` so the theme manager's
        QSS picks up the right color (mirrors how other settings panels
        do status coloring). Forces a style repolish so the change
        takes effect mid-paint.
        """
        # Clear prior class first; PyQt's property is sticky otherwise.
        self._file_status_label.setProperty("class", css_class or "")
        style = self._file_status_label.style()
        if style is not None:
            style.unpolish(self._file_status_label)
            style.polish(self._file_status_label)

    def _compute_test_playback_enabled(self) -> tuple[bool, str]:
        """Return ``(enabled, tooltip)`` for the Test Playback button
        per AC #5.
        """
        if self._source_kind == SOURCE_LAST_GENERATION:
            # The slot resolves last-generation against the registry;
            # the panel cannot know whether a saveable exists, so the
            # button stays enabled and the slot surfaces the
            # "no saveable" toast on click. Same UX as a real Clear
            # Comms toolbar click on a stale state.
            return True, _TOOLTIP_TEST_PLAYBACK_ENABLED
        if self._source_kind == SOURCE_FILE:
            if self._file_path_valid:
                return True, _TOOLTIP_TEST_PLAYBACK_ENABLED
            return False, _TOOLTIP_TEST_PLAYBACK_INVALID_FILE
        return False, _TOOLTIP_TEST_PLAYBACK_NO_SOURCE

    def _refresh_test_playback_enablement(self) -> None:
        enabled, tooltip = self._compute_test_playback_enabled()
        self._test_playback_button.setEnabled(enabled)
        self._test_playback_button.setToolTip(tooltip)


__all__ = [
    "WAV_FILE_DIALOG_FILTER",
    "PreloadedAudioLoadError",
    "load_preloaded_audio_source",
    "ClearCommsSettingsPanel",
    "SOURCE_LAST_GENERATION",
    "SOURCE_FILE",
]
