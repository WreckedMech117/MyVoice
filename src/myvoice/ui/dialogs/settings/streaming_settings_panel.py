"""Streaming Mode settings panel (Story 16.6, Phase ⊥, D-9).

Exposes the streaming-mode override field of ``AppSettings`` as a
four-option ``QComboBox`` so the user can pick ``Auto`` / ``True Stream`` /
``Sentence Stream`` / ``Batch``. ``Auto`` maps to ``None`` (the
``AppSettings.streaming_mode_override`` default — meaning "let the
hardware probe decide" per ``effective_streaming_mode(None)``); the three
explicit modes map 1:1 to the string values the
``QwenTTSService._dispatch_by_streaming_mode`` resolver consumes.

Story 16.6 ships the minimum-viable surface — a dropdown + persistence
wire-up. Visual placement (tab vs. inline section, accelerator key,
help-tooltip wording) is a tech-spec / UX review concern; placement
inside the existing ``SettingsDialog`` is the integrating module's
choice. AC #5 of Story 16.6 is satisfied here when:

  - The dropdown's selection round-trips through ``app_settings``
    (``Auto`` ↔ ``None``, others ↔ strings) per
    ``test_streaming_mode.py:107-127`` (Story 16.2's already-shipped
    serializer test).
  - Changing the selection writes to
    ``app_settings.streaming_mode_override`` immediately.
  - Reading the panel's state reflects the current
    ``streaming_mode_override`` value.

Module-boundary discipline (architecture lines 689-694): ``ui/*`` may
import ``models.app_settings`` and ``services.tts_streaming``
(``StreamingMode`` for the enum-to-label mapping); may NOT import
``services.qwen_tts_service`` (would couple UI to the dispatcher).
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from myvoice.models.app_settings import AppSettings


_logger = logging.getLogger(__name__)


# Display labels (user-facing) ↔ AppSettings.streaming_mode_override values.
# ``Auto`` is a UI-only label that maps to None; the three other labels map
# to the canonical string values the AppSettings validator accepts at
# ``app_settings.py:375-389``.
_LABEL_AUTO = "Auto (recommended)"
_LABEL_TRUE_STREAM = "True Stream (lowest latency, GPU only)"
_LABEL_SENTENCE_STREAM = "Sentence Stream"
_LABEL_BATCH = "Batch (legacy)"

_LABEL_TO_VALUE: dict[str, Optional[str]] = {
    _LABEL_AUTO: None,
    _LABEL_TRUE_STREAM: "true_stream",
    _LABEL_SENTENCE_STREAM: "sentence_stream",
    _LABEL_BATCH: "batch",
}

_VALUE_TO_LABEL: dict[Optional[str], str] = {
    None: _LABEL_AUTO,
    "true_stream": _LABEL_TRUE_STREAM,
    "sentence_stream": _LABEL_SENTENCE_STREAM,
    "batch": _LABEL_BATCH,
}


class StreamingSettingsPanel(QWidget):
    """Settings widget for the Streaming Mode dropdown (Story 16.6, AC #5).

    Stateful w.r.t. an injected ``AppSettings`` reference: changing the
    dropdown writes to the settings object immediately. Persistence (the
    actual write to disk) is the parent dialog's responsibility, matching
    the precedent set by ``ClearCommsSettingsPanel``.

    Signals:
        streaming_mode_changed(object): Emitted when the user picks a new
            mode. Payload is ``Optional[str]`` (the new override value);
            ``None`` indicates ``Auto``. Typed as ``object`` because
            PyQt6's ``pyqtSignal`` cannot natively express a nullable
            ``str``.
    """

    streaming_mode_changed = pyqtSignal(object)

    def __init__(
        self,
        app_settings: AppSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._app_settings = app_settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        group = QGroupBox("Streaming Mode")
        form = QFormLayout(group)

        self._mode_combo = QComboBox()
        self._mode_combo.setObjectName("streaming_mode_dropdown")
        self._mode_combo.setAccessibleName("Streaming mode selector")
        self._mode_combo.setAccessibleDescription(
            "Selects how generated audio is streamed. Auto picks the best "
            "mode for your hardware. True Stream produces audio with the "
            "lowest latency on GPU. Sentence Stream emits one chunk per "
            "sentence. Batch waits for the full audio before playback."
        )
        # Order mirrors the user-mental-model "best first."
        self._mode_combo.addItems(
            [
                _LABEL_AUTO,
                _LABEL_TRUE_STREAM,
                _LABEL_SENTENCE_STREAM,
                _LABEL_BATCH,
            ]
        )
        form.addRow(QLabel("Mode:"), self._mode_combo)

        layout.addWidget(group)
        layout.addStretch()

        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)

        # Initialize from the injected settings so the dropdown reflects
        # the persisted state on first render.
        self.load_state(self._app_settings)

    def load_state(self, app_settings: AppSettings) -> None:
        """Populate the dropdown from ``app_settings.streaming_mode_override``.

        ``None`` maps to the ``Auto`` label; the three explicit string
        values map 1:1. An unrecognized value (which AppSettings' own
        validator already coerces to None) falls back to ``Auto``
        defensively so a future schema drift can never produce a panel
        that fails to render.
        """
        self._app_settings = app_settings
        current_value = app_settings.streaming_mode_override
        label = _VALUE_TO_LABEL.get(current_value, _LABEL_AUTO)
        idx = self._mode_combo.findText(label)
        if idx < 0:
            self._logger.warning(
                f"streaming mode label not found: {label!r}; defaulting "
                f"to {_LABEL_AUTO!r}"
            )
            idx = self._mode_combo.findText(_LABEL_AUTO)
        # blockSignals so load_state does not emit streaming_mode_changed
        # — the listener should fire only on explicit user action.
        self._mode_combo.blockSignals(True)
        try:
            self._mode_combo.setCurrentIndex(idx)
        finally:
            self._mode_combo.blockSignals(False)

    def save_state(self, app_settings: AppSettings) -> None:
        """Write the dropdown's current selection to
        ``app_settings.streaming_mode_override``.

        Idempotent: writing the same value twice is harmless. The
        AppSettings validator at ``app_settings.py:375-389`` re-checks the
        string against the allowed set on the next ``__post_init__``-
        equivalent; this panel always writes a known-good value (None or
        one of the three documented strings) so the validator never
        rejects.
        """
        label = self._mode_combo.currentText()
        value = _LABEL_TO_VALUE.get(label)
        app_settings.streaming_mode_override = value

    @property
    def current_value(self) -> Optional[str]:
        """The currently-selected override value (``None`` or a mode str)."""
        return _LABEL_TO_VALUE.get(self._mode_combo.currentText())

    def _on_mode_changed(self, label: str) -> None:
        """Slot for the dropdown's currentTextChanged signal."""
        value = _LABEL_TO_VALUE.get(label)
        if value == self._app_settings.streaming_mode_override:
            return
        self._app_settings.streaming_mode_override = value
        self._logger.info(f"Streaming mode override set to: {value!r}")
        self.streaming_mode_changed.emit(value)


__all__ = ["StreamingSettingsPanel"]
