"""
FIFO playback queue for session ids (Phase 3 of D-20, P-8 invariant).

Story 13.1 — mechanism only. This module ships the queue, its FIFO
semantics, its `playback_queue_depth_changed(int)` signal, the cross-
thread mutation guard, and a fully isolated unit-test suite. It does
NOT instantiate the queue in production code; that wiring is Story
13.2's responsibility (`AudioCoordinator` + `app.py`).

End-user behavior is therefore unchanged after 13.1 lands — the queue
is constructed only in tests, and the existing inert
`SessionRegistry.playback_queue_depth_changed` signal at
`session_registry.py:152` stays inert until 13.2 connects them.

Responsibilities (per architecture P-8, lines 453-461):
  * Hold session ids in FIFO order with O(1) enqueue/dequeue
    (`collections.deque`, NOT `list`).
  * Emit `playback_queue_depth_changed(int)` once per state-changing
    operation, with the post-mutation depth as payload.
  * Reject cross-thread mutation with the same `RuntimeError` pattern
    `SessionRegistry` uses (D-2 threading discipline).
  * Expose `cancel_current()` for the "drop the head" operation Story
    13.2 wires into the playback-complete callback chain.

Design notes that future readers (especially Story 13.2) MUST know:

  * The queue holds session-id strings, NEVER `GenerationSession`
    references. P-4 forbids object payloads on signal/queue boundaries
    because they decouple from the registry's lifecycle. The integration
    layer (Story 13.2) looks the session up via
    `SessionRegistry.get(session_id)` at dispatch time.

  * Signal emission rule: `playback_queue_depth_changed` fires ONLY
    when the depth actually changes. An empty-queue `dequeue()` returns
    `None` and emits NOTHING; an empty-queue `cancel_current()` is a
    silent no-op. Emitting `0 → 0` would be a spurious-emission
    anti-pattern that would cause Story 13.4's badge widget to
    repaint pointlessly. Mirrors the "only emit on change" rule from
    `session_registry.py:437-441`.

  * Module boundary (architecture lines 665-667): this module may
    import `PyQt6` and `myvoice.services.sessions.generation_session`.
    It may NOT import `audio_coordinator`, `audio_service`,
    `qwen_tts_service`, `session_registry`, or anything under
    `myvoice.ui`. The integration with the coordinator is Story 13.2's
    job — *the coordinator depends on the queue, not vice versa*. An
    AST-scan test in
    `tests/unit/services/sessions/test_playback_queue.py` enforces
    this contract.

  * Cross-thread mutation is rejected with `RuntimeError`. The guard
    pattern is identical to `SessionRegistry._guard_and_lookup`
    (`session_registry.py:391-402`). Read-only accessors (`depth`,
    `peek`) are exempt from the explicit guard. Individual deque ops
    (`len()`, `[0]`) are atomic in CPython under the GIL, but the
    `peek` *composite* of "if non-empty then [0]" is NOT atomic — a
    concurrent `popleft` between the two ops would raise
    `IndexError`. `peek` therefore uses `try/except IndexError` to
    stay correct under any read pattern. Story 13.2's integration
    code should still call these on the Qt main thread by convention;
    the try/except is belt-and-braces, not a license to read from
    worker threads.

  * Signal name matches the inert
    `SessionRegistry.playback_queue_depth_changed` signal at
    `session_registry.py:152` so Story 13.2's wiring layer can
    connect them without renaming.

Cross-references:
  * `_bmad-output/planning-artifacts/architecture-optimization-pass.md`
    P-8 (lines 453-461), D-13 (lines 267-274), D-2 (line 239),
    Module Boundaries (lines 665-667), file map (lines 569-570).
  * `tests/unit/services/sessions/test_playback_queue.py` —
    contract pin: `TestFifoOrdering`, `TestDepthSignalEmission`,
    `TestQtThreadAffinity`, `TestModuleBoundary`, `TestCancelCurrent`.
"""

from collections import deque
from typing import Optional

from PyQt6.QtCore import (
    QObject,
    QThread,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtWidgets import QApplication


class PlaybackQueue(QObject):
    """FIFO playback queue for session ids (Story 13.1, P-8 invariant).

    The queue is a `QObject` so it can declare `pyqtSignal` at class
    scope (PyQt6 requirement) and live on the Qt main thread per D-2.
    Public API: `enqueue`, `dequeue`, `cancel_current`, `peek`, `depth`,
    plus the `playback_queue_depth_changed(int)` signal.

    See module docstring for the architectural contract this class
    fulfills (P-8, D-2, D-13, P-4, module boundary).
    """

    # D-13 / P-8 — depth signal. Class-level declaration is required by
    # PyQt6 for `pyqtSignal` to bind correctly. The name matches the
    # inert `SessionRegistry.playback_queue_depth_changed` signal at
    # `session_registry.py:152` so Story 13.2's wiring layer can
    # connect them without renaming. P-4: payload is a primitive `int`.
    playback_queue_depth_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Construct on the Qt main thread.

        Raises `RuntimeError` if `QApplication` has not been
        instantiated, or if construction is attempted from a worker
        thread. The thread check uses `QThread.currentThread()`
        (the executing thread), NOT `self.thread()` (the QObject's
        affinity), because `super().__init__(parent)` adopts the
        parent's thread affinity — so a main-thread `parent` would
        otherwise let a worker-thread caller bypass the guard.
        """
        super().__init__(parent)

        app = QApplication.instance()
        if app is None:
            raise RuntimeError(
                "PlaybackQueue requires a QApplication; construct "
                "QApplication([]) before instantiating the queue."
            )
        if QThread.currentThread() is not app.thread():
            raise RuntimeError(
                "PlaybackQueue must be constructed on the Qt main thread; "
                "use QMetaObject.invokeMethod or signals to dispatch from "
                "worker threads."
            )

        self._deque: deque[str] = deque()

    # ----- internal helpers --------------------------------------------- #

    def _assert_owner_thread(self, method_name: str) -> None:
        # Mirrors `session_registry.py:391-398`. Only mutating slots call
        # this; read-only accessors (`depth`, `peek`) tolerate cross-
        # thread reads (per-op atomic; `peek` uses try/except to make
        # its composite atomic too).
        #
        # Naming note: the check compares `QThread.currentThread()` to
        # `self.thread()` — i.e., the QObject's *owner/affinity* thread,
        # not "main thread" specifically. By construction these are the
        # same (the `__init__` pre-flight rejects construction off the
        # main thread), so in practice `owner == main`. If the queue
        # were ever `moveToThread`'d, the guard would shift with it,
        # which is the correct semantics — Qt slot dispatch is owner-
        # thread, not main-thread.
        if QThread.currentThread() is not self.thread():
            raise RuntimeError(
                f"Cross-thread mutation of PlaybackQueue.{method_name}; "
                f"use QMetaObject.invokeMethod(queue, '{method_name}', "
                f"Qt.ConnectionType.QueuedConnection, ...) instead."
            )

    # ----- mutation slots (P-8 / D-2) ----------------------------------- #

    @pyqtSlot(str)
    def enqueue(self, session_id: str) -> None:
        """Append a session id to the right (FIFO tail).

        Emits `playback_queue_depth_changed` with the post-append depth.
        Raises `TypeError` if `session_id` is not a `str` — P-4 forbids
        object payloads on queue boundaries; the runtime check defends
        against direct Python callers (Qt's `@pyqtSlot(str)` machinery
        only enforces type when invoked through the metaobject system).
        Idempotency / duplicate-detection is the caller's policy
        (Story 13.2 owns "session reaches READY_TO_PLAY → enqueue once").
        """
        self._assert_owner_thread("enqueue")
        if not isinstance(session_id, str):
            raise TypeError(
                f"PlaybackQueue.enqueue requires a str session id "
                f"(P-4: no object payloads); got {type(session_id).__name__}."
            )
        self._deque.append(session_id)
        self.playback_queue_depth_changed.emit(len(self._deque))

    # TODO(13.2): `@pyqtSlot()` declares this slot as void to Qt's
    # metaobject system, so a cross-thread `QMetaObject.invokeMethod`
    # invocation cannot retrieve the popped session id — the return
    # value is silently dropped. The Python-level call site is
    # unaffected. If 13.2 (or later) dispatches `dequeue` cross-thread,
    # this decoration must change (likely `@pyqtSlot(result="QString")`
    # or split into a void slot + a separate read-side accessor) AND
    # `Optional[str]` handling under PyQt6's QVariant marshalling must
    # be verified. See Story 13.1 Open Question #1.
    @pyqtSlot()
    def dequeue(self) -> Optional[str]:
        """Pop the leftmost session id (FIFO head) and return it.

        Returns `None` if the queue is empty; emits NOTHING in that
        case (an empty-queue dequeue is a no-op per AC #4 / P-8 — the
        depth does not change so no signal fires). Otherwise emits
        `playback_queue_depth_changed` with the post-pop depth.
        """
        self._assert_owner_thread("dequeue")
        if not self._deque:
            return None
        sid = self._deque.popleft()
        self.playback_queue_depth_changed.emit(len(self._deque))
        return sid

    @pyqtSlot()
    def cancel_current(self) -> None:
        """Drop the head of the queue (the session currently 'playing'
        under P-8). Emits `playback_queue_depth_changed` with the
        post-pop depth, or no signal if the queue was empty (the empty
        case is a silent no-op — no exception, no signal).
        """
        self._assert_owner_thread("cancel_current")
        if not self._deque:
            return
        self._deque.popleft()
        self.playback_queue_depth_changed.emit(len(self._deque))

    # ----- read-only accessors (no mutation guard) ---------------------- #

    @property
    def depth(self) -> int:
        """Current queue depth. Recomputed on every read (no cached
        integer) — `len(deque)` is O(1).
        """
        return len(self._deque)

    def peek(self) -> Optional[str]:
        """Return the leftmost session id without mutating, or `None`
        if the queue is empty. Used by Story 13.2's "is the head
        currently dispatched?" check.

        Implementation note: the `try/except IndexError` form is
        atomic where the equivalent `if self._deque: return self._deque[0]`
        is NOT — the bool check + index access can be split by a
        concurrent `popleft` (the GIL releases between bytecode ops),
        causing `IndexError` on the bare composite. The single
        `[0]` is atomic; catching `IndexError` makes `peek`
        robust under any read pattern.
        """
        try:
            return self._deque[0]
        except IndexError:
            return None

    def __repr__(self) -> str:
        return f"PlaybackQueue(depth={len(self._deque)})"
