"""Clear Comms button (Story 15.2, Phase 5 of D-20).

Toolbar icon button that interrupts the currently playing audio and
replays the user's configured Clear Comms source — typically a
short "I lost my voice, hold on" WAV file or the most-recent
generation. OFR-B.

Visual contract:
  - Initial state: disabled with tooltip
    ``"Generate audio first to use Clear Comms"``.
  - ``set_state(...)`` follows the enablement matrix in AC #3 of the
    story spec; six observable states (three enabled tooltips for the
    interrupt/queue × last_generation/file matrix, one for
    last_generation queue, one for file queue, plus two disabled
    tooltips).
  - Unrecognized ``source_kind`` defaults to disabled with the
    "no saveable" tooltip — defensive default; the AppSettings
    validator should have caught this upstream.

Source rule (mirrors ``save_button.py``):
  Sink, not a source. ``clicked`` is exposed unchanged from
  ``QPushButton``; the parent (``MainWindow``) wires it. ``set_state``
  is the only state-mutating entry point — call it from a single
  integration site (``MainWindow._on_clear_comms_state_changed``).

Architecture decisions activated:
  - **D-5** (line 247-248): PRELOADED-clone exclusion. Enforced
    redundantly: the click-handler in ``app.py`` does not register a
    ``GenerationSession`` for the Clear Comms playback, so the
    saveable slot is never advanced. The button is unaware of this
    mechanism.
  - **D-17** (line 282-283): Clear Comms loader (consumed by the
    click handler in ``app.py`` for the file-source branch). The
    button only tracks whether the resolved file path is currently
    valid; it does not call the loader itself.
  - **D-18** (line 284): "Interrupt by default." The ``queue_mode``
    state input flips the tooltip between interrupt and queue
    wording; the actual interrupt vs queue dispatch lives in
    ``app.py``.
  - **P-4** (lines 399-414): primitive signal payloads. The button is
    a sink; ``MainWindow.clear_comms_requested`` is the void
    pyqtSignal.

Module-boundary discipline (architecture lines 689-694):
  ``ui/*`` may import ``sessions.session_registry`` and
  ``sessions.generation_session`` only. This module needs neither —
  ``set_state`` takes plain booleans/strings; the parent supplies
  state.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtWidgets import QPushButton, QWidget


# Module-level tooltip constants — single source of truth for the six
# observable button states. Tests assert against these constants so a
# wording change here propagates symmetrically. Numbered to match the
# enablement matrix in the story's AC #2/#3 (1=disabled-no-saveable,
# 2=disabled-file-missing, 3=interrupt-last-gen, 4=interrupt-file,
# 5=queue-last-gen, 6=queue-file).
_TOOLTIP_DISABLED_NO_SAVEABLE: str = "Generate audio first to use Clear Comms"
_TOOLTIP_DISABLED_FILE_MISSING: str = "Pick a Clear Comms file in Settings"
_TOOLTIP_INTERRUPT_LAST_GEN: str = (
    "Clear Comms (interrupt) — replay last generation"
)
_TOOLTIP_INTERRUPT_FILE: str = (
    "Clear Comms (interrupt) — play configured file"
)
_TOOLTIP_QUEUE_LAST_GEN: str = (
    "Clear Comms (queue) — replay last generation after current"
)
_TOOLTIP_QUEUE_FILE: str = (
    "Clear Comms (queue) — play configured file after current"
)


class ClearCommsButton(QPushButton):
    """Compact toolbar icon button for the Clear Comms feature.

    See module docstring for the visual and source rules.
    Construction is cheap: no timers, no signal connections, no I/O.
    ``set_state`` is the only state-mutating entry point.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)

        self.setObjectName("clear_comms_button")
        # SP_MediaPlay is the only stock pixmap that reads as
        # "play this thing" without ambiguity. v1 placeholder; UX
        # review may swap for a custom icon in a follow-up pass
        # (architecture line 320 — visual styling is per-feature
        # tech-spec / UX review territory).
        play_icon = self.style().standardIcon(
            self.style().StandardPixmap.SP_MediaPlay
        )
        self.setIcon(play_icon)
        # 24×24 fixed size matches save_button / replay_button /
        # clear_button at main_window.py:294-313.
        self.setFixedSize(24, 24)
        self.setToolTip(_TOOLTIP_DISABLED_NO_SAVEABLE)
        self.setEnabled(False)

    def set_state(
        self,
        *,
        source_kind: str,
        has_saveable: bool,
        file_path_valid: bool,
        queue_mode: bool,
    ) -> None:
        """Update the button's enabled state and tooltip per AC #3.

        Args:
            source_kind: ``"last_generation"`` or ``"file"``. Anything
                else falls back to the disabled-no-saveable tooltip
                (defensive default; the AppSettings validator should
                have caught the bad value upstream).
            has_saveable: True iff
                ``SessionRegistry.saveable_session_id is not None``.
                Ignored when ``source_kind == "file"``.
            file_path_valid: True iff
                ``source_kind == "file"`` and the configured file
                resolves to an existing readable WAV. Ignored when
                ``source_kind == "last_generation"``.
            queue_mode: True iff
                ``AppSettings.clear_comms_queue_mode`` is True (queue
                instead of interrupt; D-18 default is False).

        Idempotent: Qt's ``setEnabled``/``setToolTip`` are no-ops when
        the value matches, so re-calling with the same payload does
        not flicker or repaint.
        """
        enabled, tooltip = _compute_enabled_and_tooltip(
            source_kind=source_kind,
            has_saveable=has_saveable,
            file_path_valid=file_path_valid,
            queue_mode=queue_mode,
        )
        self.setEnabled(enabled)
        self.setToolTip(tooltip)
        self._logger.debug(
            "set_state: source_kind=%s has_saveable=%s file_path_valid=%s "
            "queue_mode=%s -> enabled=%s",
            source_kind,
            has_saveable,
            file_path_valid,
            queue_mode,
            enabled,
        )


def _compute_enabled_and_tooltip(
    *,
    source_kind: str,
    has_saveable: bool,
    file_path_valid: bool,
    queue_mode: bool,
) -> tuple[bool, str]:
    """Pure helper for AC #3's enablement matrix; testable in isolation."""
    if source_kind == "last_generation":
        if not has_saveable:
            return False, _TOOLTIP_DISABLED_NO_SAVEABLE
        if queue_mode:
            return True, _TOOLTIP_QUEUE_LAST_GEN
        return True, _TOOLTIP_INTERRUPT_LAST_GEN
    if source_kind == "file":
        if not file_path_valid:
            return False, _TOOLTIP_DISABLED_FILE_MISSING
        if queue_mode:
            return True, _TOOLTIP_QUEUE_FILE
        return True, _TOOLTIP_INTERRUPT_FILE
    # Defensive default for unrecognized source_kind values.
    return False, _TOOLTIP_DISABLED_NO_SAVEABLE


__all__ = ["ClearCommsButton"]
