"""Save dialog + WAV writer (Story 14.3, Phase 4 of D-20).

OFR-A user-visible save flow. Owns the OS file dialog, the int16 WAV
writer per architecture D-16, and the "Finalizing…" save-during-
streaming wait state per architecture validation gap #2 (line 832-833).

``run()`` reads ``registry.saveable_audio`` once and dispatches to one
of three branches per AC #2:

  - **Branch A — no saveable** (``audio is None``): defensive toast,
    return. Should not normally fire because the Save button is gated
    on ``saveable_session_id is not None`` per Story 14.2 AC #5.
  - **Branch B — finalized** (``audio.is_streaming is False``): open
    the OS file dialog, capture the path, write the WAV, success
    toast.
  - **Branch C — streaming** (``audio.is_streaming is True``): open
    the dialog, capture the path, show "Finalizing…" toast, subscribe
    to ``saveable_session_changed`` and wait for the target session's
    finalize re-emit (post-finalize ``is_streaming=False``); on
    finalize, write with the post-finalize complete buffer; on
    supersession or cancel (option (a) per AC #6.2), abort with a
    transient toast and write nothing.

D-16 WAV format pinning (AC #5):

    audio_int16 = (audio.complete_audio * 32767).clip(-32768, 32767).astype(np.int16)
    sf.write(str(target_path), audio_int16, audio.sample_rate, subtype='PCM_16')

Module-boundary discipline (AC #13, architecture lines 689-694): the
UI layer reads the read-only ``SessionRegistry`` slot/property surface
established in Story 14.1 only. No imports from forbidden services
modules; the test in ``tests/ui/test_save_dialog.py`` enforces this
via a static substring scan.

Story chain: Story 14.1 made ``saveable_session_changed`` an active
signal. Story 14.2 added the visible Save button + ``MainWindow.save_
requested`` void-emit. This story (14.3) wires the void to a working
dialog + WAV writer per OFR-A.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QObject, Qt
from PyQt6.QtWidgets import QFileDialog, QWidget

from myvoice.services.sessions import SaveableAudio

if TYPE_CHECKING:
    from myvoice.services.sessions.session_registry import SessionRegistry


# Module-level toast text constants — single source of truth so tests
# can assert against the same strings the controller emits. A wording
# change here propagates symmetrically to the test suite.
_TOAST_NO_SAVEABLE = "No saveable audio — generation may have been cancelled"
_TOAST_FINALIZING = "Finalizing — will save when generation completes"
# Used only when the registry emits ``saveable_session_changed(None)`` —
# the saveable was cancelled or errored AND no previous saveable existed
# to revert to. Guaranteed-cancel semantics.
_TOAST_ABORT_CANCELLED = "Generation cancelled — nothing saved"
# Used for every other Branch C abort case: a different-id emit (could be
# real supersession by a fresh generation, or a cancel-with-revert-to-
# previous-saveable, or an error-with-revert), or a same-id race where
# ``saveable_audio`` was mutated between emit dispatch and slot read.
# Neutral wording so we never lie about the cause.
_TOAST_ABORT_SAVEABLE_CHANGED = (
    "Save aborted — saveable session changed before save could complete"
)
_FILE_DIALOG_TITLE = "Save Generation as WAV"
_FILE_DIALOG_FILTER = "WAV files (*.wav);;All files (*.*)"

# AC #11 toast timeouts (milliseconds).
_TOAST_TIMEOUT_INFO_MS = 3000
_TOAST_TIMEOUT_SUCCESS_MS = 5000
_TOAST_TIMEOUT_ERROR_MS = 5000


class SaveAudioDialog(QObject):
    """Story 14.3 controller for the Save flow.

    Subclass of ``QObject`` (not ``QDialog``) — the OS native dialog
    (``QFileDialog.getSaveFileName``) is what the user sees; this class
    is a controller, not a widget.

    Single-shot per ``MainWindow.save_requested`` emit: ``app.py``
    constructs a fresh instance per click, calls ``run()``, and lets
    the instance live only for the duration of the save flow. Branch B
    completes synchronously; Branch C is held alive across the wait by
    the Qt signal-slot connection (Qt holds a strong reference to the
    bound slot's ``self``).
    """

    def __init__(
        self,
        *,
        registry: "SessionRegistry",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._parent = parent
        self._logger = logging.getLogger(self.__class__.__name__)

        # Branch C streaming-wait state. Populated only when Branch C
        # activates; Branch B leaves these untouched.
        self._target_session_id: Optional[str] = None
        self._chosen_path: Optional[Path] = None
        self._wait_connected: bool = False

    # ----- AC #2 — public entry point ---------------------------------- #

    def run(self) -> None:
        """Open the save flow.

        Reads ``registry.saveable_audio`` exactly once at the top
        (per AC #2's "snapshot is the canonical save target" rule) and
        dispatches to Branch A, B, or C.
        """
        audio = self._registry.saveable_audio

        if audio is None:
            # Branch A — defensive. Button gating should make this
            # unreachable, but a click-vs-cancel race could expose it.
            self._logger.warning(
                "Save requested but no saveable audio (defensive)"
            )
            self._show_toast(_TOAST_NO_SAVEABLE, _TOAST_TIMEOUT_INFO_MS)
            return

        if audio.is_streaming:
            # Branch C — save during streaming (validation-gap-#2 flow).
            self._run_streaming_save(audio)
            return

        # Branch B — finalized; immediate save.
        self._run_immediate_save(audio)

    # ----- Branch B — immediate save (finalized) ----------------------- #

    def _run_immediate_save(self, audio: SaveableAudio) -> None:
        target_path = self._open_file_dialog(audio.voice)
        if target_path is None:
            # AC #3: empty chosen_path → silent return, no toast.
            return
        self._write_wav(audio, target_path)

    # ----- Branch C — save during streaming ---------------------------- #

    def _run_streaming_save(self, audio: SaveableAudio) -> None:
        target_path = self._open_file_dialog(audio.voice)
        if target_path is None:
            return
        # Pin the captured target — subsequent saveable promotions during
        # the wait do NOT redirect the save (AC #6).
        self._target_session_id = audio.session_id
        self._chosen_path = target_path
        # AC #4.1: "Finalizing…" toast (no timeout — replaced on
        # success/abort).
        self._show_toast(_TOAST_FINALIZING, timeout_ms=0)
        # AC #7: subscribe to saveable_session_changed with explicit
        # QueuedConnection (matches Story 14.2 AC #3 pinning).
        self._registry.saveable_session_changed.connect(
            self._on_saveable_changed_during_wait,
            Qt.ConnectionType.QueuedConnection,
        )
        self._wait_connected = True

    # ----- AC #7 — streaming-wait slot --------------------------------- #

    def _on_saveable_changed_during_wait(self, session_id: object) -> None:
        """Wait for our target session's finalize, OR abort on any
        disturbance.

        Dispatch table (AC #6 / #8 / #9):
          - Same id, ``is_streaming=False`` → write per AC #5.
          - Same id, ``is_streaming=True`` (defensive; should not happen) → stay subscribed.
          - Same id but ``saveable_audio`` no longer matches → race;
            abort with neutral copy (we cannot tell whether the cause
            was cancel, error, or supersession).
          - ``None`` payload → guaranteed cancel/error with no previous
            saveable to revert to → cancel toast.
          - Different non-None id → could be (a) real supersession,
            (b) cancel-with-revert-to-previous, or (c) error-with-
            revert-to-previous. We cannot disambiguate from the
            payload alone, so use the neutral "saveable changed" copy
            rather than guessing. Implements **option (a)** of AC #6.2:
            do not consult any ``previous_saveable_audio`` getter
            (forbidden in this story).
        """
        if session_id == self._target_session_id:
            post = self._registry.saveable_audio
            if post is None or post.session_id != self._target_session_id:
                # Race between emit and our slot: target was already
                # vacated by cancel/error/supersession before we could
                # read. Cause is ambiguous — use neutral copy.
                self._abort_streaming_wait(
                    "target session changed before finalize completed",
                    kind="changed",
                )
                return
            if post.is_streaming:
                # Defensive: saveable_session_changed re-emits with the
                # same id only after finalize sets is_streaming=False
                # in the GenerationSession contract. Stay subscribed.
                return
            self._complete_streaming_save(post)
            return
        # Different id or None payload — target lost its saveable slot.
        # ``None`` is a guaranteed cancel/error (no previous saveable).
        # A non-None different id could be supersession OR cancel-with-
        # revert OR error-with-revert; we don't guess.
        kind = "cancelled" if session_id is None else "changed"
        self._abort_streaming_wait(
            "target superseded or cancelled while save was pending",
            kind=kind,
        )

    # ----- AC #8 — streaming-wait completion --------------------------- #

    def _complete_streaming_save(self, post_audio: SaveableAudio) -> None:
        """Write the WAV using the post-finalize snapshot, then clean up.

        ``self._chosen_path`` is set by ``_run_streaming_save`` before the
        wait subscriber connects, so reaching this method without it is
        an internal invariant violation rather than a user-facing error.
        Use an explicit check (not ``assert``) so ``python -O`` builds
        still surface the bug rather than crashing on ``Path(None)``
        downstream.
        """
        if self._chosen_path is None:
            self._logger.error(
                "_complete_streaming_save reached without _chosen_path; "
                "this is an internal invariant violation (Branch C path "
                "must capture the path before subscribing)"
            )
            self._disconnect_streaming_wait()
            return
        self._write_wav(post_audio, self._chosen_path)
        self._disconnect_streaming_wait()

    # ----- AC #9 — streaming-wait abort -------------------------------- #

    def _abort_streaming_wait(self, reason: str, *, kind: str) -> None:
        """Show abort toast, disconnect, leave destination file untouched.

        ``kind`` selects the user-facing copy:
          - ``"cancelled"`` — guaranteed cancel/error path (registry
            emitted ``saveable_session_changed(None)``).
          - ``"changed"`` — ambiguous path (supersession, cancel-with-
            revert, error-with-revert, or same-id race). Neutral copy
            so we don't lie about the cause.
        """
        toast = (
            _TOAST_ABORT_CANCELLED
            if kind == "cancelled"
            else _TOAST_ABORT_SAVEABLE_CHANGED
        )
        self._show_toast(toast, _TOAST_TIMEOUT_ERROR_MS)
        self._logger.warning("Save aborted (kind=%s): %s", kind, reason)
        self._disconnect_streaming_wait()

    # ----- AC #7.1 — disconnect helper --------------------------------- #

    def _disconnect_streaming_wait(self) -> None:
        if not self._wait_connected:
            return
        try:
            self._registry.saveable_session_changed.disconnect(
                self._on_saveable_changed_during_wait
            )
        except (TypeError, RuntimeError):
            # Qt raises TypeError if not connected; RuntimeError if the
            # underlying QObject was already deleted. Either is benign
            # here — defense against double-disconnect.
            pass
        self._wait_connected = False

    # ----- AC #3 / AC #4 — OS file dialog ------------------------------ #

    def _open_file_dialog(self, voice: str) -> Optional[Path]:
        """Open ``QFileDialog.getSaveFileName`` and return the chosen
        path with a guaranteed ``.wav`` suffix, or ``None`` on cancel.
        """
        default_filename = self._make_default_filename(voice)
        default_dir = self._default_save_directory()
        default_path = default_dir / default_filename

        chosen_path, _filter = QFileDialog.getSaveFileName(
            self._parent,
            _FILE_DIALOG_TITLE,
            str(default_path),
            _FILE_DIALOG_FILTER,
        )
        if not chosen_path:
            # AC #3: empty string → user cancelled, silent return.
            return None
        result = Path(chosen_path)
        # AC #4: enforce .wav extension regardless of OS dialog
        # behavior. Cross-platform guarantee.
        if result.suffix.lower() != ".wav":
            result = result.with_suffix(".wav")
        return result

    # ----- AC #5 — WAV write ------------------------------------------- #

    def _write_wav(self, audio: SaveableAudio, target_path: Path) -> None:
        """D-16 conversion + ``soundfile.write`` with explicit PCM_16.

        Defensive empty-buffer guard (review fix M2): an empty
        ``complete_audio`` would produce a header-only ~44-byte WAV file
        and a misleading success toast. Story 14.1's contract makes this
        unreachable in production today (finalize requires non-empty
        chunks), but Epic 16's streaming paths could surface it through
        a slot-injection bug.
        """
        if audio.complete_audio.size == 0:
            self._logger.error(
                "Save failed for %s: complete_audio is empty (size=0)",
                target_path,
            )
            self._show_toast(
                "Save failed: empty audio buffer",
                _TOAST_TIMEOUT_ERROR_MS,
            )
            return
        try:
            audio_int16 = (
                (audio.complete_audio * 32767)
                .clip(-32768, 32767)
                .astype(np.int16)
            )
            sf.write(
                str(target_path),
                audio_int16,
                audio.sample_rate,
                subtype="PCM_16",
            )
        except (OSError, RuntimeError) as exc:
            self._logger.error(
                "Save failed for %s: %s", target_path, exc, exc_info=True
            )
            self._show_toast(
                f"Save failed: {exc}", _TOAST_TIMEOUT_ERROR_MS
            )
            return
        # AC #11: success toast with resolved absolute path.
        resolved = target_path.resolve()
        self._show_toast(
            f"Saved to {resolved}", _TOAST_TIMEOUT_SUCCESS_MS
        )

    # ----- AC #10 — default filename ----------------------------------- #

    def _make_default_filename(self, voice: str) -> str:
        """D-16: ``myvoice-<voice>-<yyyy-mm-dd-hhmmss>.wav`` (sanitized).

        Sanitization rule (AC #10): keep ASCII alphanumeric, underscore,
        and hyphen; replace everything else with underscore. Empty /
        all-illegal voice falls back to literal ``"voice"``.
        """
        sanitized_voice = re.sub(r"[^A-Za-z0-9_-]", "_", voice)
        if not sanitized_voice:
            sanitized_voice = "voice"
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        return f"myvoice-{sanitized_voice}-{timestamp}.wav"

    # ----- AC #3.1 — default directory --------------------------------- #

    def _default_save_directory(self) -> Path:
        """Return ``Music`` if it exists, else ``Downloads``, else home.

        Does NOT create directories. Does NOT consult any persisted
        preference — relies on ``QFileDialog``'s native last-directory
        memory across calls.
        """
        home = Path.home()
        for candidate in (home / "Music", home / "Downloads"):
            if candidate.is_dir():
                return candidate
        return home

    # ----- AC #11 — status-bar toast helper ---------------------------- #

    def _show_toast(self, text: str, timeout_ms: int) -> None:
        """Show ``text`` on the parent's status bar, falling back to log
        if no status bar is reachable.

        ``timeout_ms=0`` means "show until replaced" — used by the
        Branch C "Finalizing…" toast which is replaced on success/abort.
        """
        status_bar = getattr(self._parent, "status_bar", None)
        if status_bar is not None and hasattr(status_bar, "showMessage"):
            status_bar.showMessage(text, timeout_ms)
            return
        # Defensive fallback: log at INFO so headless tests / missing
        # status bar don't lose the message entirely.
        self._logger.info("[toast] %s", text)


__all__ = ["SaveAudioDialog"]
