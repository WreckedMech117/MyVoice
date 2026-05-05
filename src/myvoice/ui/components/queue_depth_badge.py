"""Queue-depth indicator badge (Story 13.4, Phase 3 of D-20).

Visual indicator showing how many generations are deferred in the
PlaybackQueue. Fed by ``MainWindow._on_playback_queue_depth_changed``
which subscribes to ``SessionRegistry.playback_queue_depth_changed``
(the queue→registry forwarding connection lives in ``app.py:338-340``).

Visual contract:
  - depth == 0  → hidden (``isVisible() == False``)
  - depth >= 1  → shown with text containing the digit
  - depth <  0  → defensively clamped to 0 (logs a warning)

The badge is a sink, not a source — no signals emitted; ``set_depth``
is the only public entry point.

Architecture references:
  - D-13 signal contract (architecture-optimization-pass.md:267-274):
    ``playback_queue_depth_changed (depth: int) — Drives the indicator
    badge``
  - Architecture line 137: badge promoted into scope (party-mode
    decision, 2026-04-27)
  - P-4 (architecture-optimization-pass.md:399-414): primitive
    signal payloads only — the badge consumes ``int``, never emits
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QWidget


class QueueDepthBadge(QLabel):
    """Compact label showing the deferred PlaybackQueue depth.

    See module docstring for the visual contract. Construction is cheap:
    no timers, no signals, no I/O. ``set_depth`` is the only state-mutating
    entry point — call it from a single integration site (the registry
    slot in MainWindow) per the architecture's "one signal source" rule
    (D-13).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._depth: int = 0
        self._logger = logging.getLogger(self.__class__.__name__)

        # Match service_status_indicator height (16px) so the badge sits
        # visually flush with the existing TTS/Mic indicators. Width is
        # left to auto-adjust — variable digit count makes a fixed width
        # ugly in practice.
        font = QFont()
        font.setPointSize(9)
        self.setFont(font)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(16)
        # CSS hook for future theme work; no current theme rule references it.
        self.setProperty("class", "queue-depth-badge")
        self.setToolTip("Queue depth — generations waiting to play")

        # Initial state is hidden (matches ``set_depth(0)``). Calling
        # ``hide()`` from ``__init__`` is safe — the widget has no
        # parent in ``show()`` chain yet, so no flicker.
        self.hide()

    @property
    def depth(self) -> int:
        """Current queue depth as last passed to ``set_depth``."""
        return self._depth

    def set_depth(self, depth: int) -> None:
        """Update the badge to reflect ``depth``.

        - ``depth < 0``  → clamped to 0 + WARNING log; badge hides.
        - ``depth == 0`` → badge hides.
        - ``depth >= 1`` → text set to ``_format_text(depth)``; badge shows.

        Calling ``show()`` on an already-visible widget is a no-op (no
        flicker); calling ``hide()`` on an already-hidden widget is a
        no-op as well.
        """
        if depth < 0:
            self._logger.warning(
                "set_depth received negative value %d; clamping to 0", depth
            )
            depth = 0
        self._depth = depth
        if depth == 0:
            self.hide()
            return
        self.setText(self._format_text(depth))
        self.show()

    def _format_text(self, depth: int) -> str:
        """Format the badge text for a positive depth value.

        Default format ``⏳N`` (hourglass + digit) echoes the emoji-friendly
        visual language of ``STATUS_EMOJI`` in ``service_status_indicator.py``
        and conveys "waiting" semantics. If the hourglass renders poorly on a
        future deployment surface, a future maintainer can swap to ``f"{depth}"``
        without breaking tests (which assert "the digit appears in the text").
        """
        return f"⏳{depth}"


__all__ = ["QueueDepthBadge"]
