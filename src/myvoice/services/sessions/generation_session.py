"""
Per-utterance generation session with strict state machine.

Story 11.1 — Phase 1 of D-20 (architecture-optimization-pass.md).

A `GenerationSession` represents one TTS utterance from request to playback to
disposal. The state machine (P-1) prevents the silent state-transition bugs
that plagued the V2 baseline, and the single `_transition_to` chokepoint (P-2)
gives downstream features (Save, Clear Comms, Queue, Indicator) one place to
attach lifecycle-aware behaviors.

This module is signal-free by design — state-change emission is the registry's
job (Story 11.2). Module boundary: stdlib + numpy only; no PyQt, no peer
service imports.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SessionState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    READY_TO_PLAY = "ready_to_play"
    PLAYING = "playing"
    DONE = "done"
    CANCELLED = "cancelled"
    ERROR = "error"
    DISCARDED = "discarded"


class SessionSource(Enum):
    GENERATED = "generated"
    PRELOADED = "preloaded"


_VALID_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.PENDING: frozenset({
        SessionState.GENERATING, SessionState.CANCELLED, SessionState.ERROR,
    }),
    SessionState.GENERATING: frozenset({
        SessionState.READY_TO_PLAY, SessionState.CANCELLED, SessionState.ERROR,
    }),
    SessionState.READY_TO_PLAY: frozenset({
        SessionState.PLAYING, SessionState.CANCELLED, SessionState.ERROR,
    }),
    SessionState.PLAYING: frozenset({
        SessionState.DONE, SessionState.CANCELLED, SessionState.ERROR,
    }),
    SessionState.DONE: frozenset({SessionState.DISCARDED}),
    SessionState.CANCELLED: frozenset({SessionState.DISCARDED}),
    SessionState.ERROR: frozenset({SessionState.DISCARDED}),
    SessionState.DISCARDED: frozenset(),
}


def _states_that_can_reach(target: SessionState) -> frozenset[SessionState]:
    """Set of source states from which `target` is a valid successor."""
    return frozenset(s for s, succ in _VALID_TRANSITIONS.items() if target in succ)


class InvalidSessionStateError(Exception):
    """Raised when a lifecycle method is called in a disallowed state."""

    def __init__(
        self,
        current_state: SessionState,
        method: str,
        expected_states: frozenset[SessionState],
    ) -> None:
        self.current_state = current_state
        self.method = method
        self.expected_states = expected_states
        expected_names = sorted(s.name for s in expected_states) or ["(none)"]
        super().__init__(
            f"{method}() called in state {current_state.name}; "
            f"valid only in: {', '.join(expected_names)}"
        )


@dataclass
class GenerationSession:
    text: str
    voice: str
    source: SessionSource
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.PENDING
    chunks: list[np.ndarray] = field(default_factory=list)
    complete_audio: Optional[np.ndarray] = None
    sample_rate: int = 24000
    is_audible: bool = False
    is_streaming: bool = False
    created_at: float = field(default_factory=time.time)
    # Per-state wall-clock durations (seconds), populated by `_transition_to`
    # when each state is *exited*. Story 11.3 (`metrics.record`) reads this
    # directly for the `session_state_duration_ms` metric.
    #
    # Limitation: terminal-predecessor states (DONE, CANCELLED, ERROR) are
    # only recorded when the session is later DISCARDED; sessions that are
    # never discarded leave those entries missing. DISCARDED itself is never
    # recorded — there is no further transition to mark its exit. Story 11.3
    # must tolerate missing entries.
    _state_durations: dict[SessionState, float] = field(default_factory=dict)
    # time.perf_counter() (Story 12.2): sub-microsecond QueryPerformanceCounter
    # source. Avoids tier-(b)/(c) ties; Python 3.10's time.monotonic() on Windows
    # uses GetTickCount64 with the SAME ~15.625ms resolution as time.time(),
    # so it does not actually solve the tight-clustering tie problem despite
    # AC #4's wording. perf_counter is documented monotonic on this platform.
    _last_transition_at: float = field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        # P-2 entry-state guard. The constructor blesses sessions in PENDING
        # (the normal entry point — all later state changes flow through
        # `_transition_to`), OR in READY_TO_PLAY with `complete_audio`
        # already attached (the replay-clone signature, used by
        # `clone_for_replay` so the clone never needs an out-of-band state
        # mutation). Every other entry-state is rejected.
        if self.state == SessionState.PENDING:
            return
        if (
            self.state == SessionState.READY_TO_PLAY
            and self.complete_audio is not None
        ):
            return
        raise ValueError(
            f"GenerationSession must be constructed in PENDING (or in "
            f"READY_TO_PLAY with complete_audio set, for replay clones); "
            f"got state={self.state.name}, complete_audio="
            f"{'set' if self.complete_audio is not None else 'None'}."
        )

    # ----- P-2: sole state-mutation chokepoint --------------------------- #

    def _transition_to(self, new_state: SessionState) -> None:
        if new_state not in _VALID_TRANSITIONS[self.state]:
            raise InvalidSessionStateError(
                current_state=self.state,
                method="_transition_to",
                expected_states=_VALID_TRANSITIONS[self.state],
            )
        # Source must match SessionRegistry.focal_session_id's `now`
        # (Story 12.2: both use time.perf_counter for tie-avoidance).
        now = time.perf_counter()
        # Duration spent in the state being left (now - entry-into-state).
        self._state_durations[self.state] = now - self._last_transition_at
        self._last_transition_at = now
        self.state = new_state

    # ----- P-1: state-bound lifecycle methods ---------------------------- #

    def append_chunk(self, audio: np.ndarray) -> None:
        self._require_state("append_chunk", frozenset({SessionState.GENERATING}))
        self.chunks.append(audio)

    def finalize(self) -> None:
        self._require_state("finalize", frozenset({SessionState.GENERATING}))
        if not self.chunks:
            raise ValueError(
                "finalize() called with no chunks; append_chunk() must be "
                "called at least once before finalize()."
            )
        # D-7 memory hygiene: concat → clear → transition, in that order.
        self.complete_audio = np.concatenate(self.chunks)
        self.chunks.clear()
        self._transition_to(SessionState.READY_TO_PLAY)

    def mark_playing(self) -> None:
        self._require_state("mark_playing", frozenset({SessionState.READY_TO_PLAY}))
        self._transition_to(SessionState.PLAYING)

    def mark_audible(self) -> None:
        # Substate flag — does NOT transition.
        self._require_state("mark_audible", frozenset({SessionState.PLAYING}))
        self.is_audible = True

    def mark_done(self) -> None:
        self._require_state("mark_done", frozenset({SessionState.PLAYING}))
        self._transition_to(SessionState.DONE)

    def cancel(self) -> None:
        # Method-validity reflects the transition graph: cancel() is valid
        # exactly in those states from which CANCELLED is reachable.
        self._require_state("cancel", _states_that_can_reach(SessionState.CANCELLED))
        self._transition_to(SessionState.CANCELLED)

    def discard(self) -> None:
        self._require_state(
            "discard",
            frozenset({SessionState.DONE, SessionState.CANCELLED, SessionState.ERROR}),
        )
        self._transition_to(SessionState.DISCARDED)
        self.complete_audio = None

    def clone_for_replay(self, source: SessionSource = SessionSource.PRELOADED) -> "GenerationSession":
        if self.complete_audio is None:
            raise ValueError(
                "clone_for_replay() requires complete_audio to be set; "
                "call finalize() on the source session first."
            )
        # D-6: zero-copy — share the buffer by reference. The clone is
        # constructed directly in READY_TO_PLAY via the replay-clone
        # signature recognized by `__post_init__` (state=READY_TO_PLAY +
        # complete_audio set). No out-of-band state mutation is required;
        # the P-2 chokepoint guarantee is preserved at the syntactic level
        # (no `self.state =` or `object.__setattr__(..., "state", ...)`
        # outside `_transition_to`).
        return GenerationSession(
            text=self.text,
            voice=self.voice,
            source=source,
            state=SessionState.READY_TO_PLAY,
            complete_audio=self.complete_audio,
            sample_rate=self.sample_rate,
        )

    # ----- internal helpers ---------------------------------------------- #

    def _require_state(self, method: str, expected: frozenset[SessionState]) -> None:
        if self.state not in expected:
            raise InvalidSessionStateError(
                current_state=self.state,
                method=method,
                expected_states=expected,
            )
