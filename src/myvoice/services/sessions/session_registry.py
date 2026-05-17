"""
Qt-main-thread singleton that mediates all generation-session mutations.

Story 11.2 — Phase 1 of D-20 (architecture-optimization-pass.md). Net-zero
behavior change: this story creates the registry plus tests; no existing
module imports it yet (Story 11.4 wires it into `qwen_tts_service`).

Responsibilities (per architecture):
  * Owns the in-flight `GenerationSession` collection (`_sessions`).
  * Mediates every state mutation through Qt-thread-affined slot methods.
  * Provides `post_mutation()` for cross-thread (worker → main) dispatch.
  * Emits `session_state_changed` and `current_session_changed`. The
    `playback_queue_depth_changed` signal is declared as an inert port for
    Story 13.1 to wire later. The `saveable_session_changed` signal is wired
    by Story 14.1 (this story) — see Saveable Slot Lifecycle Contract below.
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

Saveable Slot Lifecycle Contract (Story 14.1 lock-down)
========================================================

  * **Two slots, never more** (D-3, D-4): the registry holds at most two
    ``SaveableAudio`` records — ``_saveable`` (the most recently finalized
    GENERATED session, drives ``saveable_session_id`` and Save button
    enablement) and ``_previous_saveable`` (the prior saveable, lingering
    in grace-period memory so its buffer is preserved). Slot capture happens
    at ``finalize`` time; the dataclass holds a strong reference to
    ``complete_audio`` so the buffer survives the originating session's
    later ``discard()`` (which sets ``session.complete_audio = None`` per
    ``generation_session.py:197``).

  * **Promotion on finalize** (D-3, AC #2/#3/#4): on every
    ``finalize(session_id)`` for a GENERATED-source session, the prior
    ``_saveable`` demotes to ``_previous_saveable`` (overwriting and
    releasing whatever was there), the new session's snapshot becomes the
    current ``_saveable``, and ``saveable_session_changed.emit(session_id)``
    fires exactly once.

  * **Release on next mark_done** (D-4, AC #5): when the current saveable
    reaches DONE (its ``mark_done`` slot fires), ``_previous_saveable`` is
    cleared (set to ``None``); ``_saveable`` is unchanged;
    ``saveable_session_changed`` does NOT fire (the saveable_session_id is
    unchanged). Idempotent if ``_previous_saveable`` is already ``None``.

  * **Invalidation on cancel / set_error** (P-7, AC #7 + review fix): a
    GENERATED session reaching ``CANCELLED`` or ``ERROR`` must vacate
    whichever slot it occupies, so neither slot ever references a
    non-successful session. Two cases:
      - Session is the current ``_saveable`` → revert: ``_saveable`` becomes
        ``_previous_saveable`` (or ``None``), ``_previous_saveable``
        clears, and ``saveable_session_changed.emit(<new_id_or_None>)``
        fires exactly once.
      - Session is the ``_previous_saveable`` → clear: ``_previous_saveable``
        becomes ``None``; ``_saveable`` is unchanged and no signal fires
        (saveable_session_id is unchanged).
    If the session is in neither slot, both slots are unchanged and no
    signal fires (silent no-op — mirrors
    ``_recompute_focal_and_maybe_emit``'s no-spurious-emission discipline).
    The architecture's P-7 "Cancelled sessions never become saveable" is
    enforced by the invariant that no slot ever holds a CANCELLED/ERROR
    session — this is what makes the revert path safe to promote
    ``_previous_saveable`` without re-checking its state.

  * **PRELOADED exclusion** (D-5, AC #6): sessions with
    ``source == SessionSource.PRELOADED`` (Clear Comms playback clones,
    Epic 15 territory) NEVER participate in saveable lifecycle. Promotion,
    release, and revert paths all gate on
    ``session.source == SessionSource.GENERATED``.

  * **Discard does NOT modify slots** (AC #8): the ``discard`` slot drops
    the session record from ``_sessions`` but leaves ``_saveable`` and
    ``_previous_saveable`` untouched. Slot lifecycle is driven by
    ``mark_done`` (release) and ``cancel`` (revert), not by ``discard``.

  * **Emission ordering on finalize** (AC #2): the three signals fire in
    this fixed order — ``session_state_changed(id, READY_TO_PLAY)`` then
    ``current_session_changed(id)`` (if focal changed) then
    ``saveable_session_changed(id)``. DirectConnection subscribers observe
    them in this order; QueuedConnection subscribers see them queued in
    this order.

  * Cross-references: see
    ``tests/unit/services/sessions/test_session_registry.py``,
    classes ``TestSaveableSlotPromotion``, ``TestSaveableSlotDemotion``,
    ``TestPreloadedSourceExclusion``,
    ``TestPreviousSaveableReleaseOnMarkDone``,
    ``TestSaveableCancelRevert``, ``TestDiscardDoesNotClearSlot``,
    ``TestSaveableSessionIdProperty``, ``TestSaveableAudioProperty``,
    ``TestSaveableEmissionOrdering``, ``TestSaveableModuleBoundary``,
    ``TestNetZeroFromPreviousStories``, ``TestDebugDumpHelper``.

Cooperative Cancellation Chain Contract (Story 16.5 lock-down)
==============================================================

  * **Two-endpoint API split** (P-7, architecture-optimization-pass.md:443-451):
    cancellation flows in one direction —
    ``user → registry → streamer's _cancel_event → talker .generate()
    returns → decoder drains → session transitions CANCELLED → DISCARDED``.
    The registry exposes two endpoints that map to the two halves of this
    chain:

      - ``request_cancel(session_id)`` — synchronous, runs on the calling
        thread, fires the registered cancel hook (if any). The hook
        flips ``streamer._cancel_event`` and asks the audio coordinator to
        stop playback for the session. **Does NOT transition session
        state** — that's the decoder worker's job (it posts
        ``('cancel', sid)`` via ``post_mutation`` after draining its queue;
        the existing ``cancel`` slot then runs on Qt main thread and does
        the actual ``session.cancel()``).
      - The existing ``cancel(session_id)`` slot — Qt-main-thread only,
        transitions state via ``session.cancel()``, emits
        ``session_state_changed``, invokes saveable-slot invalidation
        (Story 14.1), auto-clears the cancel hook (Story 16.5).

    The split exists so the architecture's P-7 invariant — "session.cancel()
    sets the event; it does not immediately transition state. The decoder
    worker's drain-on-cancel posts the actual CANCELLED transition (so we
    never have a 'cancelled but still emitting chunks' window)" — is
    honored verbatim. **Do NOT** extend ``request_cancel`` to also call
    ``session.cancel()`` directly; that would re-introduce the
    "cancelled but still emitting" window.

  * **Hook lifecycle**: Story 16.6's TRUE_STREAM dispatcher registers a
    hook at session-create time via ``register_cancel_hook(sid, fn)``;
    re-registration replaces the prior hook silently (so the dispatcher
    can re-construct the streamer + worker pair across retries within a
    single session_id without leaking stale hooks). The hook is
    auto-cleared by ``_maybe_clear_cancel_hook(sid)`` at the bottom of
    every terminal-state slot — ``cancel``, ``mark_done``, ``set_error``,
    ``discard``. Cleanup is idempotent (``dict.pop(sid, None)``); the
    redundant ``discard`` call provides defense-in-depth in case a future
    code path adds an unobserved terminal transition.

  * **Race tolerance**: ``register_cancel_hook`` does NOT require the
    session_id to be present in ``_sessions`` — Story 16.6's wiring may
    happen in a slightly different tick from ``create_session``. A hook
    registered for a never-existing session lives until the registry is
    GC'd, but that path is not reachable under normal flow (registry
    callers always create the session first).

  * **Defensive but quiet**: ``request_cancel`` for an unregistered
    session_id is a quiet no-op (no exception, no metric, no WARN-log).
    Today's batch + sentence-stream callers in
    ``QwenTTSService.cancel_generation`` invoke this unconditionally for
    every session, and those flows correctly carry no hook. A hook that
    raises is caught and recorded as a ``cancel_hook_error`` metric (per
    Story 16.4's ``decode_error`` precedent — numeric value 1.0, error
    repr in tags); ``request_cancel`` returns normally and the hook is
    NOT auto-cleared on exception (Story 16.6's wiring may want to retry
    against the same surface in failure-recovery scenarios).

  * **Cross-thread invocation**: ``request_cancel`` runs on the calling
    thread synchronously — it does NOT hop to the Qt main thread. The
    hook's job is to flip a thread-safe ``threading.Event`` and schedule
    an async coroutine on the audio coordinator's event loop; both are
    thread-safe operations and forcing a Qt-main hop would introduce
    latency that violates the P-7 100ms cancel-chain target.

  * **Module boundary preserved**: the new code adds at most one
    ``Callable`` import from ``typing``; it does NOT import
    ``myvoice.services.tts_streaming.*`` or
    ``myvoice.services.audio_coordinator`` (architecture line 662). The
    cancel-hook callable carries any tts_streaming / audio_coordinator
    reference indirectly via its closure (Story 16.6's wiring composes
    the closure).

  * Cross-references: see
    ``tests/unit/services/sessions/test_session_registry.py``,
    classes ``TestCancelHookRegistration``, ``TestRequestCancelInvocation``.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

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

from myvoice.observability import metrics
from myvoice.services.sessions.generation_session import (
    GenerationSession,
    InvalidSessionStateError,
    SessionSource,
    SessionState,
)


# `eq=False` is load-bearing: the dataclass holds a numpy buffer, and the
# auto-generated `__eq__` would compare arrays with `==` (which returns an
# array, not a bool, so tuple-equality dispatch raises ValueError) and the
# auto-generated `__hash__` would attempt to hash an unhashable np.ndarray.
# We don't need value equality on snapshots — identity (`is`) is the contract,
# since the registry holds at most one slot reference at a time.
@dataclass(frozen=True, eq=False)
class SaveableAudio:
    """Snapshot of a saveable session's audio + metadata (Story 14.1).

    Captured at finalize() time so the buffer survives the originating
    session's later discard() (which clears session.complete_audio per
    generation_session.py:197). Held by SessionRegistry in either the
    current `_saveable` or the lingering `_previous_saveable` slot per
    D-3 / D-4 / D-5 lifecycle.

    Treat as immutable per D-6 convention — sessions never mutate
    complete_audio after finalize, and consumers (Story 14.3 WAV
    writer) read-only.
    """
    session_id: str
    complete_audio: np.ndarray
    sample_rate: int
    voice: str
    text: str
    is_streaming: bool
    created_at: float


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
    "try_set_error",
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
    # Wired by Story 14.1 — emitted by `_maybe_promote_saveable` (finalize)
    # and `_invalidate_saveable_slot_for` (cancel / set_error) per the
    # "Saveable Slot Lifecycle Contract" section above.
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
        # Story 14.1: saveable-slot lifecycle (D-3, D-4, D-5).
        # Captured at finalize-time so the audio buffer survives session.discard().
        self._saveable: Optional[SaveableAudio] = None
        self._previous_saveable: Optional[SaveableAudio] = None
        # Story 16.5: per-session cooperative-cancel hooks. Story 16.6's
        # TRUE_STREAM dispatcher registers a closure here that flips the
        # streamer's `_cancel_event` and asks the audio coordinator to stop
        # playback. Auto-cleared at the bottom of every terminal-state slot
        # (cancel / mark_done / set_error / discard) per
        # `_maybe_clear_cancel_hook`.
        self._cancel_hooks: dict[str, Callable[[], None]] = {}

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
        # Story 14.1: saveable promotion — D-3 (most-recent finalize wins),
        # D-4 (demote prior to _previous_saveable), D-5 (PRELOADED clones
        # do not participate). The slot capture is BEFORE any later discard
        # could clear session.complete_audio (per AC #1 / AC #8).
        self._maybe_promote_saveable(session)

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
        # Story 14.1: D-4 release — the next GENERATED session reaching
        # DONE is the canonical trigger to free the previous saveable's
        # buffer hold. See AC #5 / module docstring "Saveable Slot
        # Lifecycle Contract".
        self._maybe_release_previous_saveable_on_done(session)
        # Story 16.5: terminal-state cleanup — idempotent.
        self._maybe_clear_cancel_hook(session_id)

    @pyqtSlot(str)
    def cancel(self, session_id: str) -> None:
        session = self._guard_and_lookup("cancel", session_id)
        old_state = session.state
        session.cancel()
        self._maybe_emit_state_changed(session_id, session, old_state)
        self._recompute_focal_and_maybe_emit()
        # Story 14.1 + review fix: vacate whichever slot the cancelled
        # session occupies (both _saveable and _previous_saveable are in
        # scope), so the invariant "no slot ever holds a CANCELLED/ERROR
        # session" holds across subsequent revert paths.
        self._invalidate_saveable_slot_for(session)
        # Story 16.5: terminal-state cleanup — idempotent.
        self._maybe_clear_cancel_hook(session_id)

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
        # Story 14.1: do NOT modify _saveable / _previous_saveable here. The
        # slot's strong reference to complete_audio survives session.discard()
        # (which set session.complete_audio to None at generation_session.py:197).
        # Slot lifecycle is driven by mark_done (release per D-4) and cancel
        # (revert per AC #7), not by discard.
        self._recompute_focal_and_maybe_emit()
        # Story 16.5: defense-in-depth — the prior CANCELLED/DONE/ERROR
        # transition should already have cleared the hook, but DISCARDED is
        # the architecturally-correct latest cleanup point. Idempotent.
        self._maybe_clear_cancel_hook(session_id)

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
        # Story 14.1 + review fix: ERROR is symmetric with CANCELLED for
        # saveable-slot purposes — neither outcome should leave the session
        # selectable for Save. The idempotency guard above prevents
        # double-invalidation on repeated set_error calls.
        self._invalidate_saveable_slot_for(session)
        # Story 16.5: terminal-state cleanup — idempotent.
        self._maybe_clear_cancel_hook(session_id)

    @pyqtSlot(str)
    def try_set_error(self, session_id: str) -> None:
        """Race-safe variant of `set_error` for error-cleanup paths.

        Behaves identically to `set_error` for active sessions (PENDING,
        GENERATING, READY_TO_PLAY, PLAYING) but silently absorbs sessions
        that have already settled into any terminal state — ERROR,
        CANCELLED, DONE, or DISCARDED. The strict `set_error` contract is
        preserved separately (see `test_set_error_propagates_from_other_terminals`).

        Why this exists: when TRUE_STREAM raises mid-dispatch
        (`qwen_tts_service._generate_true_stream`'s 0-chunks-on-cold-compile
        path), the cleanup posts `cancel_event.set()` → joins the worker,
        which in turn posts ('cancel', sid) per the worker's drain-on-cancel
        contract. By the time the dispatch's own ('set_error', sid) reaches
        the Qt event loop, the session is already CANCELLED — and strict
        `set_error` raises InvalidSessionStateError, which the global
        exception handler surfaces as an "Unexpected Error" dialog. The
        fallback chain has already succeeded by then, so the dialog is
        purely cosmetic noise. `try_set_error` absorbs the race.

        Side effects (saveable slot invalidation, cancel hook cleanup) are
        unchanged from `set_error`'s contract — they only run when an
        actual transition occurs. Terminal-state absorption is a true
        no-op.
        """
        session = self._guard_and_lookup("try_set_error", session_id)
        if session.state in (
            SessionState.ERROR,
            SessionState.CANCELLED,
            SessionState.DONE,
            SessionState.DISCARDED,
        ):
            return
        old_state = session.state
        self._transition_for_slot(session, "try_set_error", SessionState.ERROR)
        self._maybe_emit_state_changed(session_id, session, old_state)
        self._recompute_focal_and_maybe_emit()
        self._invalidate_saveable_slot_for(session)
        self._maybe_clear_cancel_hook(session_id)

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

    # ----- Story 16.5 — cooperative cancellation chain ------------------- #

    def register_cancel_hook(
        self, session_id: str, hook: Callable[[], None]
    ) -> None:
        """Story 16.5: register a per-session cancel hook.

        The hook is invoked synchronously by ``request_cancel(session_id)``
        on whatever thread that method was called from. Story 16.6's
        TRUE_STREAM dispatcher composes a closure that flips the streamer's
        ``_cancel_event`` (thread-safe per ``threading.Event`` semantics)
        and schedules ``audio_coordinator.cancel_playback(session_id)`` on
        the coordinator's event loop.

        Re-registration replaces the prior hook silently — Story 16.6's
        dispatcher may re-construct the streamer + worker pair across
        retries within a single session_id without leaking stale hooks.
        Story 16.5's wiring contract.

        The session_id is NOT required to be present in ``_sessions``;
        race-tolerance for Story 16.6's wiring (where hook registration
        may happen in a slightly different tick from session creation).
        A hook registered for a never-existing session lives until the
        registry is GC'd — acceptable at Story 16.5's runtime scale.

        Raises:
            ValueError: if ``session_id`` is empty/None or ``hook`` is None.
        """
        if not session_id:
            raise ValueError(
                "register_cancel_hook: session_id must be a non-empty string"
            )
        if hook is None:
            raise ValueError("register_cancel_hook: hook must not be None")
        self._cancel_hooks[session_id] = hook

    def request_cancel(self, session_id: str) -> None:
        """Story 16.5: synchronously fire the registered cancel hook.

        Quiet no-op for unregistered session_ids; rationale: today's batch
        + sentence-stream callers in ``QwenTTSService.cancel_generation``
        invoke this unconditionally for every session, and the existing
        flows correctly carry no hook. Story 16.6's TRUE_STREAM dispatch
        path registers the hook at session-create-time for streaming
        sessions only.

        Hook exceptions are caught and recorded as a ``cancel_hook_error``
        metric (numeric value 1.0 per Story 16.4's ``decode_error``
        precedent — the real ``metrics.record`` validates value to
        ``int|float``). The hook is NOT auto-cleared on exception;
        Story 16.6's wiring may want to retry against the same surface.

        Runs on the calling thread (NOT Qt main); the hook's job is to
        flip a thread-safe ``threading.Event`` and schedule an async
        coroutine — both thread-safe operations. Forcing a Qt-main hop
        would introduce latency that violates the P-7 cancel-chain
        budget.

        **Anti-pattern**: Do NOT extend this method to also call
        ``session.cancel()`` directly — that would re-introduce the
        "cancelled but still emitting" window for streaming sessions,
        where the worker's in-flight ``append_chunk`` post would land
        AFTER the registry's CANCELLED transition. The two endpoints
        (request_cancel + cancel) are intentionally split. Non-streaming
        callers (today's batch + sentence-stream paths) ALSO call
        ``post_mutation('cancel', sid)`` from their own ``CancelledError``
        handlers — that path is unchanged.
        """
        hook = self._cancel_hooks.get(session_id)
        if hook is None:
            return
        try:
            hook()
        except Exception as exc:  # noqa: BLE001 — keep cancel surface resilient
            metrics.record(
                "cancel_hook_error",
                1.0,
                session_id=session_id,
                error_repr=repr(exc),
            )

    def _maybe_clear_cancel_hook(self, session_id: str) -> None:
        """Story 16.5: idempotent removal of a per-session cancel hook.

        Invoked at the bottom of every terminal-state slot (cancel /
        mark_done / set_error / discard) so the hook map cannot leak
        across session lifecycles. ``dict.pop(_, None)`` is the standard
        idempotent removal idiom; double-clearing across multiple
        terminal transitions (e.g., cancel → discard) is a silent no-op.
        """
        self._cancel_hooks.pop(session_id, None)

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

    # ----- Story 14.1 — saveable session properties --------------------- #

    @property
    def saveable_session_id(self) -> Optional[str]:
        """Story 14.1: id of the current saveable session, or None.

        Driven by the `finalize` slot (promotes new saveables per D-3) and
        the `cancel` slot (reverts to previous saveable per AC #7). See
        module docstring's "Saveable Slot Lifecycle Contract" section.
        """
        return self._saveable.session_id if self._saveable is not None else None

    @property
    def saveable_audio(self) -> Optional[SaveableAudio]:
        """Story 14.1: snapshot of the current saveable's audio + metadata.

        Returned dataclass is frozen and shares the complete_audio buffer
        zero-copy with the slot — consumers (Story 14.3 WAV writer) MUST
        treat as immutable per D-6 convention.
        """
        return self._saveable

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

    # ----- Story 14.1 — saveable slot lifecycle helpers ----------------- #

    def _maybe_promote_saveable(self, session: GenerationSession) -> None:
        """Story 14.1: promote a freshly-finalized GENERATED session to the
        saveable slot per D-3, demote the prior saveable to the previous-
        saveable slot per D-4. PRELOADED clones do not participate (D-5).
        """
        if session.source != SessionSource.GENERATED:
            return  # D-5: PRELOADED clones never become saveable
        new_slot = SaveableAudio(
            session_id=session.session_id,
            complete_audio=session.complete_audio,
            sample_rate=session.sample_rate,
            voice=session.voice,
            text=session.text,
            is_streaming=session.is_streaming,
            created_at=session.created_at,
        )
        # D-4: prior saveable demotes to previous; the third-back drops
        # entirely (the assignment overwrites _previous_saveable, releasing
        # any earlier reference for GC).
        self._previous_saveable = self._saveable
        self._saveable = new_slot
        self.saveable_session_changed.emit(session.session_id)

    def _maybe_release_previous_saveable_on_done(
        self, session: GenerationSession
    ) -> None:
        """Story 14.1: release the previous-saveable hold per D-4. Triggers
        only when the session reaching DONE is GENERATED-source AND is the
        current saveable. The release is idempotent (None = None is a
        no-op) and emits NO signal — the saveable_session_id is unchanged.
        """
        if session.source != SessionSource.GENERATED:
            return  # D-5: PRELOADED clones don't drive lifecycle
        if self._saveable is None or self._saveable.session_id != session.session_id:
            return  # session reaching DONE is not the current saveable
        # Drop the previous-saveable hold; the buffer is GC-eligible if no
        # other reference exists (typically the session was already through
        # discard() so this slot held the last reference).
        self._previous_saveable = None

    def _invalidate_saveable_slot_for(
        self, session: GenerationSession
    ) -> None:
        """Story 14.1 + review fix: a GENERATED session reaching CANCELLED
        or ERROR must vacate any saveable slot it occupies.

        Called from both the `cancel` and `set_error` slots after the
        underlying transition has been emitted. Maintains the invariant
        that neither `_saveable` nor `_previous_saveable` holds a
        non-successful session — this is what makes the revert path safe
        to promote `_previous_saveable` without re-checking its state.

        Two cases:
          (1) Session is the current ``_saveable`` → revert to
              ``_previous_saveable`` (or ``None``), clear
              ``_previous_saveable``, emit ``saveable_session_changed``.
          (2) Session is the ``_previous_saveable`` → clear it; do not
              emit (saveable_session_id is unchanged).

        PRELOADED clones never enter either slot (D-5), so the source
        guard is defense-in-depth against future bypass paths; under
        normal flow it is unreachable.
        """
        if session.source != SessionSource.GENERATED:
            return
        sid = session.session_id
        if self._saveable is not None and self._saveable.session_id == sid:
            new_current = self._previous_saveable
            self._saveable = new_current
            self._previous_saveable = None
            new_id: Optional[str] = (
                new_current.session_id if new_current is not None else None
            )
            self.saveable_session_changed.emit(new_id)
            return
        if (
            self._previous_saveable is not None
            and self._previous_saveable.session_id == sid
        ):
            self._previous_saveable = None

    def _DEBUG_dump_saveable_state(self) -> dict:
        """Story 14.1: test/debug-only inspection of the saveable slots.

        NOT public API. Intended for integration tests that need a stable
        inspection surface without poking at private fields directly. May
        be removed if Stories 14.2 / 14.3 don't need it.
        """
        def _slot_view(slot: Optional[SaveableAudio]) -> dict:
            if slot is None:
                return {"session_id": None, "audio_id": None, "audio_size_bytes": None}
            return {
                "session_id": slot.session_id,
                "audio_id": id(slot.complete_audio),
                "audio_size_bytes": slot.complete_audio.nbytes,
            }
        cur = _slot_view(self._saveable)
        prev = _slot_view(self._previous_saveable)
        return {
            "saveable_session_id": cur["session_id"],
            "previous_saveable_session_id": prev["session_id"],
            "saveable_audio_id": cur["audio_id"],
            "previous_saveable_audio_id": prev["audio_id"],
            "saveable_audio_size_bytes": cur["audio_size_bytes"],
            "previous_saveable_audio_size_bytes": prev["audio_size_bytes"],
        }
