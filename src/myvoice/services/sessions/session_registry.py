"""
Qt-main-thread singleton that mediates all generation-session mutations.

Story 11.2 — Phase 1 of D-20 (architecture-optimization-pass.md). Net-zero
behavior change: this story creates the registry plus tests; no existing
module imports it yet (Story 11.4 wires it into `qwen_tts_service`).

Responsibilities (per architecture):
  * Owns the in-flight `GenerationSession` collection (`_sessions`).
  * Mediates every state mutation through Qt-thread-affined slot methods.
  * Provides `post_mutation()` for cross-thread (worker → main) dispatch.
  * Emits `session_state_changed` and `current_session_changed`. The other
    two D-13 signals (`saveable_session_changed`, `playback_queue_depth_changed`)
    are declared as inert ports for Stories 14.1 and 13.1 to wire later.
  * Computes `focal_session_id` from session lifecycle state with the
    four-tier priority defined in validation gap #1.

Design notes that future readers (especially Story 11.4) MUST know:

  * `mark_audible` is a substate flip, not a transition. The registry still
    re-emits `session_state_changed(id, current_state)` so subscribers (the
    Story 12.1 indicator) can re-read both `state` and `is_audible` from
    `registry.get(session_id)`. This is the chosen substate-observation
    pattern (D-15); a fifth signal was rejected to keep the D-13 contract
    at four signals.

  * `start_generation` is an explicit mutation rather than auto-promoting
    on the first `append_chunk`. Story 11.4 needs to fire
    `session_state_changed(GENERATING)` and the legacy `generation_started`
    signal at the same moment — explicit transition makes that wiring
    obvious and keeps `append_chunk` semantically pure (data-only).

  * The registry is the *sole* signal emitter for session state (P-2). The
    underlying `GenerationSession` is signal-free (Story 11.1) and never
    crosses a signal boundary as a payload (P-4).

  * `model_type` is accepted by `create_session` (validation gap #5) but
    `GenerationSession` does not currently carry it. The registry stores
    it in a parallel `_session_model_types` mapping until Story 11.4
    decides whether to migrate the field into the session itself.

Focal Session Contract (Story 12.2 lock-down)
=============================================

  * **Four-tier priority** (validation gap #1 / `focal_session_id`):
      (a) most recently transitioned PLAYING session;
      (b) else most recently transitioned GENERATING/READY_TO_PLAY session;
      (c) else most recently transitioned terminal (DONE/CANCELLED/ERROR)
          session within ``_FOCAL_DECAY_SECONDS`` of now;
      (d) else None.
    Tier-(b) MUST skip terminal-within-decay candidates even when those are
    more recent. PENDING is not focal-eligible at any tier.

  * **Inclusive decay boundary**: tier-(c)'s comparison is
    ``(now - s._last_transition_at) <= _FOCAL_DECAY_SECONDS`` — a terminal
    session at exactly ``_FOCAL_DECAY_SECONDS`` ago is still focal.

  * **Time source**: ``GenerationSession._last_transition_at`` and the
    registry's ``now`` both come from ``time.perf_counter()``. Rationale
    is tie-avoidance under tight time clustering; on Python 3.10 + Windows,
    both ``time.time()`` and ``time.monotonic()`` use ``GetTickCount64``
    with ~15.625 ms resolution, which ties under back-to-back transitions
    in the same Qt event tick. ``perf_counter`` uses
    ``QueryPerformanceCounter`` (sub-microsecond) and is documented
    monotonic on this platform (see ``time.get_clock_info('perf_counter')``).
    AC #4 of Story 12.2 specifies ``time.monotonic()`` literally; the
    deviation is necessary for Python 3.10 and is documented in that
    story's Change Log. Python 3.13+ aligns ``time.monotonic`` with QPC
    on Windows, so a future runtime upgrade may revisit this.

  * **No spurious emissions**: ``current_session_changed`` fires exactly
    when the recomputed focal id differs from ``_last_focal_id``. State-
    preserving mutations (``append_chunk``, idempotent ``set_error``, the
    ``mark_audible`` substate flip) and tier-internal transitions
    (``GENERATING`` → ``READY_TO_PLAY`` is tier-(b) → tier-(b)) do NOT
    emit. The asymmetry vs ``session_state_changed``'s D-15 substate
    re-emit is intentional and load-bearing for the indicator's redraw
    suppression (Story 12.1).

  * **No eager emit on construct**: ``__init__`` does not fire
    ``current_session_changed``. ``_last_focal_id`` starts at ``None``;
    a subscriber attached immediately after construction observes only
    the registry's first real focal change. This pins the contract so a
    future "synthetic initialization frame" optimization is rejected.

  * Cross-references: see
    ``tests/unit/services/sessions/test_session_registry.py``,
    classes ``TestFocalPriorityExplicit`` and
    ``TestCurrentSessionChangedContract``.
"""

import time
from typing import Optional

import numpy as np

from PyQt6.QtCore import (
    QObject,
    pyqtSignal,
    pyqtSlot,
    QMetaObject,
    Qt,
    Q_ARG,
    QThread,
)
from PyQt6.QtWidgets import QApplication

from myvoice.services.sessions.generation_session import (
    GenerationSession,
    InvalidSessionStateError,
    SessionSource,
    SessionState,
)


# Terminal sessions remain focal-eligible for this many seconds after their
# last transition (validation gap #1, tier (c)).
_FOCAL_DECAY_SECONDS: float = 5.0

# Names of `@pyqtSlot`-decorated mutation methods that `post_mutation` may
# dispatch. Pre-validation against this set catches typos and mis-targeted
# calls upfront — `QMetaObject.invokeMethod`'s own behavior on unknown
# methods (raises `RuntimeError` in PyQt6, return value unreliable) is too
# brittle to rely on as the sole guard.
_MUTATION_SLOT_NAMES: frozenset[str] = frozenset({
    "start_generation",
    "append_chunk",
    "finalize",
    "mark_playing",
    "mark_audible",
    "mark_done",
    "cancel",
    "discard",
    "set_error",
})


class SessionRegistry(QObject):
    """Qt-main-thread mediator for `GenerationSession` lifecycle (D-1, D-2).

    See module docstring for the architectural contract this class fulfills.
    """

    # D-13 signals (P-4: payloads are primitives or `Optional[str]` via `object`).
    # `session_state_changed` is also re-emitted by `mark_audible` with the
    # unchanged state so subscribers can observe the `is_audible` substate flip.
    session_state_changed = pyqtSignal(str, object)
    # `Optional[str]` cannot be encoded by PyQt6's signal type system; use
    # `object` and emit either `None` or the focal session id.
    current_session_changed = pyqtSignal(object)
    # Inert in 11.2 — Story 13.1 will emit from PlaybackQueue.
    playback_queue_depth_changed = pyqtSignal(int)
    # Inert in 11.2 — Story 14.1 will emit from saveable-slot lifecycle.
    saveable_session_changed = pyqtSignal(object)

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
                "SessionRegistry requires a QApplication; construct "
                "QApplication([]) before instantiating the registry."
            )
        if QThread.currentThread() is not app.thread():
            raise RuntimeError(
                "SessionRegistry must be constructed on the Qt main thread; "
                "use QMetaObject.invokeMethod or signals to dispatch from "
                "worker threads."
            )

        self._sessions: dict[str, GenerationSession] = {}
        self._session_model_types: dict[str, str] = {}
        self._last_focal_id: Optional[str] = None

    # ----- AC #4 / #5 — creation gateway and read-only access ----------- #

    def create_session(
        self,
        text: str,
        voice: str,
        model_type: str,
        source: SessionSource = SessionSource.GENERATED,
    ) -> str:
        """Instantiate a `GenerationSession` (always in PENDING per 11.1
        `__post_init__`), store it, and return the new session id.

        `model_type` is held in a registry-side mapping until Story 11.4
        decides whether to add the field to `GenerationSession` itself.
        """
        session = GenerationSession(text=text, voice=voice, source=source)
        self._sessions[session.session_id] = session
        self._session_model_types[session.session_id] = model_type
        # PENDING is not focal-eligible, so this is usually a no-op until the
        # first transition; call it anyway for symmetry with mutation slots.
        self._recompute_focal_and_maybe_emit()
        return session.session_id

    def get(self, session_id: str) -> Optional[GenerationSession]:
        """Return the stored session (same reference every call) or `None`."""
        return self._sessions.get(session_id)

    # ----- AC #6 / #7 / #9 — mutation slots ----------------------------- #

    @pyqtSlot(str)
    def start_generation(self, session_id: str) -> None:
        session = self._guard_and_lookup("start_generation", session_id)
        old_state = session.state
        self._transition_for_slot(session, "start_generation", SessionState.GENERATING)
        self._maybe_emit_state_changed(session_id, session, old_state)
        self._recompute_focal_and_maybe_emit()

    @pyqtSlot(str, object)
    def append_chunk(self, session_id: str, audio: np.ndarray) -> None:
        session = self._guard_and_lookup("append_chunk", session_id)
        old_state = session.state
        session.append_chunk(audio)
        self._maybe_emit_state_changed(session_id, session, old_state)
        self._recompute_focal_and_maybe_emit()

    @pyqtSlot(str)
    def finalize(self, session_id: str) -> None:
        session = self._guard_and_lookup("finalize", session_id)
        old_state = session.state
        session.finalize()
        self._maybe_emit_state_changed(session_id, session, old_state)
        self._recompute_focal_and_maybe_emit()

    @pyqtSlot(str)
    def mark_playing(self, session_id: str) -> None:
        session = self._guard_and_lookup("mark_playing", session_id)
        old_state = session.state
        session.mark_playing()
        self._maybe_emit_state_changed(session_id, session, old_state)
        self._recompute_focal_and_maybe_emit()

    @pyqtSlot(str)
    def mark_audible(self, session_id: str) -> None:
        # Substate flip — state does not change, but subscribers (Story 12.1
        # indicator) need to observe the `is_audible` flag. Re-emit
        # `session_state_changed` with the unchanged state per D-15 ONLY
        # when the substate actually flips False→True; duplicate calls when
        # already audible are silently absorbed so subscribers don't repaint
        # on noise.
        session = self._guard_and_lookup("mark_audible", session_id)
        was_audible = session.is_audible
        session.mark_audible()
        if not was_audible:
            self.session_state_changed.emit(session_id, session.state)
        self._recompute_focal_and_maybe_emit()

    @pyqtSlot(str)
    def mark_done(self, session_id: str) -> None:
        session = self._guard_and_lookup("mark_done", session_id)
        old_state = session.state
        session.mark_done()
        self._maybe_emit_state_changed(session_id, session, old_state)
        self._recompute_focal_and_maybe_emit()

    @pyqtSlot(str)
    def cancel(self, session_id: str) -> None:
        session = self._guard_and_lookup("cancel", session_id)
        old_state = session.state
        session.cancel()
        self._maybe_emit_state_changed(session_id, session, old_state)
        self._recompute_focal_and_maybe_emit()

    @pyqtSlot(str)
    def discard(self, session_id: str) -> None:
        # After the underlying transition completes, the session is no
        # longer in-flight. Emit the state change first (subscribers
        # connected via DirectConnection see the registry entry intact at
        # the moment of emission), then drop the registry entries to keep
        # the in-flight collection bounded — long-running apps can
        # otherwise accumulate DISCARDED sessions indefinitely.
        session = self._guard_and_lookup("discard", session_id)
        old_state = session.state
        session.discard()
        self._maybe_emit_state_changed(session_id, session, old_state)
        del self._sessions[session_id]
        self._session_model_types.pop(session_id, None)
        self._recompute_focal_and_maybe_emit()

    @pyqtSlot(str)
    def set_error(self, session_id: str) -> None:
        # Forces `<current> → ERROR`. Required by Story 11.4's exception
        # paths, which can call this from multiple unwinding handlers on
        # the same session — so the slot is idempotent on already-errored
        # sessions (the second call is silently absorbed). Calls from other
        # terminal states (DONE, CANCELLED, DISCARDED) still raise per the
        # transition graph; once a session has settled into a non-ERROR
        # terminal it is not reclassifiable as an error.
        session = self._guard_and_lookup("set_error", session_id)
        if session.state == SessionState.ERROR:
            return
        old_state = session.state
        self._transition_for_slot(session, "set_error", SessionState.ERROR)
        self._maybe_emit_state_changed(session_id, session, old_state)
        self._recompute_focal_and_maybe_emit()

    # ----- AC #8 — cross-thread helper (P-3) ---------------------------- #

    def post_mutation(self, method_name: str, *args) -> None:
        """Queue a mutation onto the registry's (Qt main) thread.

        Returns immediately; the actual mutation runs the next time the Qt
        event loop processes events. Each positional argument is wrapped in
        the appropriate `Q_ARG` based on its runtime type:

        =================  ============================
        Python type        ``Q_ARG`` form
        =================  ============================
        ``str``            ``Q_ARG(str, value)``
        anything else      ``Q_ARG('PyQt_PyObject', value)``
        =================  ============================

        Example (worker thread):
            registry.post_mutation('start_generation', sid)
            registry.post_mutation('append_chunk', sid, np_audio)
        """
        # P-1 protection: typos and non-slot method names are rejected
        # upfront with a clear RuntimeError, so 11.4's exception paths
        # cannot silently drop a mutation through a misspelled string.
        # `QMetaObject.invokeMethod`'s own bool return is unreliable in
        # PyQt6 (it can raise RuntimeError for unknown methods, and the
        # return value does not correlate cleanly with success even when
        # the queue post succeeds), so we do not depend on it.
        if method_name not in _MUTATION_SLOT_NAMES:
            raise RuntimeError(
                f"post_mutation('{method_name}', ...) rejected: not a "
                f"recognized mutation slot on SessionRegistry. Valid "
                f"slots: {sorted(_MUTATION_SLOT_NAMES)}. Check spelling."
            )

        q_args = []
        for arg in args:
            if isinstance(arg, str):
                q_args.append(Q_ARG(str, arg))
            else:
                q_args.append(Q_ARG("PyQt_PyObject", arg))

        QMetaObject.invokeMethod(
            self,
            method_name,
            Qt.ConnectionType.QueuedConnection,
            *q_args,
        )

    # ----- AC #10 / #11 — focal session ---------------------------------- #

    @property
    def focal_session_id(self) -> Optional[str]:
        """Recompute the focal session every access (no cached state read).

        Priority (validation gap #1):
          (a) most recently transitioned PLAYING session
          (b) else most recently transitioned GENERATING/READY_TO_PLAY session
          (c) else most recently transitioned terminal session within
              ``_FOCAL_DECAY_SECONDS`` of now
          (d) else None
        """
        # Source must match GenerationSession._last_transition_at
        # (Story 12.2: both use time.perf_counter — see that field's comment
        # for the Python 3.10 / Windows GetTickCount64 rationale).
        now = time.perf_counter()
        sessions = list(self._sessions.values())

        playing = [s for s in sessions if s.state == SessionState.PLAYING]
        if playing:
            return max(playing, key=lambda s: s._last_transition_at).session_id

        active = [
            s for s in sessions
            if s.state in (SessionState.GENERATING, SessionState.READY_TO_PLAY)
        ]
        if active:
            return max(active, key=lambda s: s._last_transition_at).session_id

        terminal = [
            s for s in sessions
            if s.state in (SessionState.DONE, SessionState.CANCELLED, SessionState.ERROR)
            and (now - s._last_transition_at) <= _FOCAL_DECAY_SECONDS
        ]
        if terminal:
            return max(terminal, key=lambda s: s._last_transition_at).session_id

        return None

    # ----- internal helpers --------------------------------------------- #

    def _guard_and_lookup(self, method_name: str, session_id: str) -> GenerationSession:
        # Anti-pattern catalog row 3: cross-thread mutation must use
        # `post_mutation` instead of calling slots directly.
        if QThread.currentThread() is not self.thread():
            raise RuntimeError(
                f"Cross-thread mutation; use registry.post_mutation"
                f"('{method_name}', ...) instead of calling {method_name}() directly."
            )
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session

    def _transition_for_slot(
        self,
        session: GenerationSession,
        slot_method_name: str,
        new_state: SessionState,
    ) -> None:
        """Forward a state transition to `_transition_to`, but re-raise any
        `InvalidSessionStateError` with the public slot's name attached.

        `start_generation` and `set_error` are the only slots that delegate
        directly to `_transition_to` (no public session method to call).
        Without this helper, an invalid-state error from those slots would
        surface as `_transition_to() called in state X; ...`, blaming an
        internal helper for a bug the caller introduced via the public API.
        """
        try:
            session._transition_to(new_state)
        except InvalidSessionStateError as exc:
            raise InvalidSessionStateError(
                current_state=exc.current_state,
                method=slot_method_name,
                expected_states=exc.expected_states,
            ) from exc

    def _maybe_emit_state_changed(
        self,
        session_id: str,
        session: GenerationSession,
        old_state: SessionState,
    ) -> None:
        if session.state != old_state:
            self.session_state_changed.emit(session_id, session.state)

    def _recompute_focal_and_maybe_emit(self) -> None:
        new_focal = self.focal_session_id
        if new_focal != self._last_focal_id:
            self._last_focal_id = new_focal
            self.current_session_changed.emit(new_focal)
