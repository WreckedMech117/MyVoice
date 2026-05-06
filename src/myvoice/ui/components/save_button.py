"""Save button (Story 14.2, Phase 4 of D-20).

Toolbar icon button that consumes the registry's saveable-slot lifecycle
(Story 14.1) to gate user-initiated save of the most-recent generation.

Visual contract:
  - Initial state: disabled with tooltip "No generation to save yet".
  - ``set_saveable(audio)`` with ``audio is None`` → disabled, disabled
    tooltip.
  - ``set_saveable(audio)`` with a non-None payload and
    ``audio.is_streaming is False`` → enabled, finalized tooltip.
  - ``set_saveable(audio)`` with a non-None payload and
    ``audio.is_streaming is True`` → enabled, finalized tooltip with
    ``(streaming…)`` suffix appended (architecture nice-to-have #1).

Source rule (mirrors ``queue_depth_badge.py``):
  Sink, not a source. ``clicked`` is exposed unchanged from
  ``QToolButton``; consumers (``MainWindow._on_save_clicked``) connect
  to it. ``set_saveable`` is the only state-mutating entry point — call
  it from the single integration site (``MainWindow._on_saveable_session_changed``).

Architecture references:
  - D-15 (``is_audible`` / saveable lifecycle) — architecture
    optimization-pass line 278.
  - D-16 (Save format / OFR-A) — architecture line 280; consumed by
    Story 14.3, referenced here so the contract chain is visible.
  - Nice-to-have #1 (streaming tooltip suffix) — architecture line 849.
  - P-4 (primitive signal payloads) — architecture lines 399-414.

Story chain:
  Wired by ``MainWindow._connect_registry_to_indicator``. Story 14.1
  made ``saveable_session_changed`` an active signal and
  ``saveable_audio`` a property. Story 14.3 will subscribe to
  ``MainWindow.save_requested`` to open the file dialog; until then the
  click-emitted signal lands in the void.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QToolButton, QWidget

from myvoice.services.sessions import SaveableAudio


# Module-level tooltip constants — single source of truth for the three
# observable button states. Tests assert against these constants so a
# wording change here propagates symmetrically.
_TOOLTIP_DISABLED = "No generation to save yet"
_TOOLTIP_ENABLED = "Save last generation"
# U+2026 HORIZONTAL ELLIPSIS — matches architecture nice-to-have #1's
# wording at line 849. NOT three ASCII dots.
_TOOLTIP_ENABLED_STREAMING = "Save last generation (streaming…)"


class SaveButton(QToolButton):
    """Compact toolbar icon button gating user-initiated save.

    See module docstring for the visual and source rules. Construction
    is cheap: no timers, no signal connections, no I/O. ``set_saveable``
    is the only state-mutating entry point — wire it from a single
    integration site (``MainWindow._on_saveable_session_changed``) per
    the architecture's "one signal source" rule (D-13).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)

        self.setObjectName("save_button")
        save_icon = self.style().standardIcon(
            self.style().StandardPixmap.SP_DialogSaveButton
        )
        self.setIcon(save_icon)
        self.setFixedSize(QSize(24, 24))
        self.setToolTip(_TOOLTIP_DISABLED)
        self.setEnabled(False)

    def set_saveable(self, audio: Optional[SaveableAudio]) -> None:
        """Update the button to reflect the current saveable session.

        - ``audio is None`` → disable, set disabled tooltip.
        - ``audio.is_streaming is False`` → enable, set finalized
          tooltip.
        - ``audio.is_streaming is True`` → enable, set finalized tooltip
          with ``(streaming…)`` suffix.

        Idempotent: Qt's ``setEnabled``/``setToolTip`` are no-ops when
        the value matches, so re-calling with the same payload does not
        flicker or repaint. No internal caching needed.
        """
        if audio is None:
            self.setEnabled(False)
            self.setToolTip(_TOOLTIP_DISABLED)
            self._logger.debug("set_saveable: disabled (no saveable session)")
            return
        self.setEnabled(True)
        if audio.is_streaming:
            self.setToolTip(_TOOLTIP_ENABLED_STREAMING)
        else:
            self.setToolTip(_TOOLTIP_ENABLED)
        self._logger.debug(
            "set_saveable: enabled (session_id=%s, is_streaming=%s)",
            audio.session_id, audio.is_streaming,
        )


__all__ = ["SaveButton"]
