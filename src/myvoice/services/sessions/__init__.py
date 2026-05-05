"""Sessions subpackage — per-utterance lifecycle types (Phase 1, Stories 11.1, 11.2; Phase 3 mechanism, Story 13.1)."""

from myvoice.services.sessions.generation_session import (
    GenerationSession,
    InvalidSessionStateError,
    SessionSource,
    SessionState,
    _VALID_TRANSITIONS,
)
from myvoice.services.sessions.session_registry import SessionRegistry
from myvoice.services.sessions.playback_queue import PlaybackQueue

__all__ = [
    "GenerationSession",
    "InvalidSessionStateError",
    "PlaybackQueue",
    "SessionSource",
    "SessionState",
    "SessionRegistry",
    "_VALID_TRANSITIONS",
]
