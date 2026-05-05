"""
Tests for GenerationSession (Story 11.1)

Verifies the per-utterance state machine specified by architecture P-1/P-2,
D-6 (clone_for_replay zero-copy), D-7 (memory hygiene), and the module
boundary discipline (no PyQt, no peer-service imports).
"""

import inspect
import re
import time
from pathlib import Path

import numpy as np
import pytest

from myvoice.services.sessions import (
    GenerationSession,
    InvalidSessionStateError,
    SessionSource,
    SessionState,
    _VALID_TRANSITIONS,
)
from myvoice.services.sessions import generation_session as gs_module


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_session(state: SessionState = SessionState.PENDING) -> GenerationSession:
    """Build a minimal session and force-position it into `state` for tests.

    Tests need to exercise methods in many starting states. We deliberately
    bypass `_transition_to` here (the only place this is allowed in the
    codebase — production code must never do this; the static-scan test below
    enforces that for the production module).
    """
    session = GenerationSession(text="hello", voice="default", source=SessionSource.GENERATED)
    object.__setattr__(session, "state", state)
    return session


# --------------------------------------------------------------------------- #
# AC #2 / #3 — enum membership
# --------------------------------------------------------------------------- #

class TestEnums:
    def test_session_state_has_exactly_eight_members(self):
        expected = {
            "PENDING", "GENERATING", "READY_TO_PLAY", "PLAYING",
            "DONE", "CANCELLED", "ERROR", "DISCARDED",
        }
        assert {m.name for m in SessionState} == expected

    def test_session_state_values_are_lowercase_strings(self):
        for member in SessionState:
            assert isinstance(member.value, str)
            assert member.value == member.name.lower()

    def test_session_source_has_exactly_two_members(self):
        assert {m.name for m in SessionSource} == {"GENERATED", "PRELOADED"}


# --------------------------------------------------------------------------- #
# AC #5 — _VALID_TRANSITIONS table
# --------------------------------------------------------------------------- #

class TestValidTransitions:
    def test_all_states_are_keys(self):
        assert set(_VALID_TRANSITIONS.keys()) == set(SessionState)

    def test_pending_successors(self):
        assert _VALID_TRANSITIONS[SessionState.PENDING] == frozenset({
            SessionState.GENERATING, SessionState.CANCELLED, SessionState.ERROR,
        })

    def test_generating_successors(self):
        assert _VALID_TRANSITIONS[SessionState.GENERATING] == frozenset({
            SessionState.READY_TO_PLAY, SessionState.CANCELLED, SessionState.ERROR,
        })

    def test_ready_to_play_successors(self):
        assert _VALID_TRANSITIONS[SessionState.READY_TO_PLAY] == frozenset({
            SessionState.PLAYING, SessionState.CANCELLED, SessionState.ERROR,
        })

    def test_playing_successors(self):
        assert _VALID_TRANSITIONS[SessionState.PLAYING] == frozenset({
            SessionState.DONE, SessionState.CANCELLED, SessionState.ERROR,
        })

    def test_terminal_chain_successors(self):
        assert _VALID_TRANSITIONS[SessionState.DONE] == frozenset({SessionState.DISCARDED})
        assert _VALID_TRANSITIONS[SessionState.CANCELLED] == frozenset({SessionState.DISCARDED})
        assert _VALID_TRANSITIONS[SessionState.ERROR] == frozenset({SessionState.DISCARDED})

    def test_discarded_is_terminal(self):
        assert _VALID_TRANSITIONS[SessionState.DISCARDED] == frozenset()

    def test_every_non_terminal_state_has_at_least_one_successor(self):
        for state, successors in _VALID_TRANSITIONS.items():
            if state == SessionState.DISCARDED:
                continue
            assert successors, f"state {state} must have at least one successor"

    def test_every_state_except_pending_is_reachable(self):
        reachable = set()
        for successors in _VALID_TRANSITIONS.values():
            reachable.update(successors)
        # PENDING is the entry point, every other state must appear as a
        # successor of some other state.
        for state in SessionState:
            if state == SessionState.PENDING:
                continue
            assert state in reachable, f"state {state} is not reachable"


# --------------------------------------------------------------------------- #
# AC #4 — state-bound method validity
# --------------------------------------------------------------------------- #

class TestAppendChunk:
    def test_valid_in_generating(self):
        session = make_session(SessionState.GENERATING)
        chunk = np.array([0.1, 0.2], dtype=np.float32)
        session.append_chunk(chunk)
        assert session.chunks == [chunk]

    @pytest.mark.parametrize("state", [
        s for s in SessionState if s is not SessionState.GENERATING
    ])
    def test_invalid_outside_generating(self, state):
        session = make_session(state)
        with pytest.raises(InvalidSessionStateError) as excinfo:
            session.append_chunk(np.array([0.0]))
        assert excinfo.value.current_state == state
        assert excinfo.value.method == "append_chunk"
        assert excinfo.value.expected_states == frozenset({SessionState.GENERATING})


class TestFinalize:
    def test_valid_in_generating_concatenates_and_clears(self):
        session = make_session(SessionState.GENERATING)
        c1 = np.array([0.1, 0.2], dtype=np.float32)
        c2 = np.array([0.3, 0.4], dtype=np.float32)
        session.append_chunk(c1)
        session.append_chunk(c2)
        session.finalize()
        assert session.state == SessionState.READY_TO_PLAY
        assert session.complete_audio is not None
        assert np.array_equal(session.complete_audio, np.concatenate([c1, c2]))
        # AC #6 — chunks emptied immediately after concat
        assert session.chunks == []

    def test_chunks_empty_after_finalize_returns(self):
        session = make_session(SessionState.GENERATING)
        session.append_chunk(np.array([1.0], dtype=np.float32))
        session.finalize()
        assert len(session.chunks) == 0

    @pytest.mark.parametrize("state", [
        s for s in SessionState if s is not SessionState.GENERATING
    ])
    def test_invalid_outside_generating(self, state):
        session = make_session(state)
        with pytest.raises(InvalidSessionStateError):
            session.finalize()

    def test_finalize_empty_chunks_raises_value_error(self):
        # Upstream race: GENERATING state reached but no chunks ever arrived.
        # Must surface as a clean ValueError, not an opaque numpy crash.
        session = make_session(SessionState.GENERATING)
        with pytest.raises(ValueError, match="no chunks"):
            session.finalize()
        # State must not have advanced.
        assert session.state == SessionState.GENERATING


class TestMarkPlaying:
    def test_valid_in_ready_to_play(self):
        session = make_session(SessionState.READY_TO_PLAY)
        session.mark_playing()
        assert session.state == SessionState.PLAYING

    @pytest.mark.parametrize("state", [
        s for s in SessionState if s is not SessionState.READY_TO_PLAY
    ])
    def test_invalid_outside_ready_to_play(self, state):
        session = make_session(state)
        with pytest.raises(InvalidSessionStateError):
            session.mark_playing()


class TestMarkAudible:
    def test_valid_in_playing_does_not_transition(self):
        session = make_session(SessionState.PLAYING)
        session.mark_audible()
        assert session.is_audible is True
        assert session.state == SessionState.PLAYING

    @pytest.mark.parametrize("state", [
        s for s in SessionState if s is not SessionState.PLAYING
    ])
    def test_invalid_outside_playing(self, state):
        session = make_session(state)
        with pytest.raises(InvalidSessionStateError):
            session.mark_audible()


class TestMarkDone:
    def test_valid_in_playing(self):
        session = make_session(SessionState.PLAYING)
        session.mark_done()
        assert session.state == SessionState.DONE

    @pytest.mark.parametrize("state", [
        s for s in SessionState if s is not SessionState.PLAYING
    ])
    def test_invalid_outside_playing(self, state):
        session = make_session(state)
        with pytest.raises(InvalidSessionStateError):
            session.mark_done()


class TestCancel:
    @pytest.mark.parametrize("state", [
        SessionState.PENDING, SessionState.GENERATING,
        SessionState.READY_TO_PLAY, SessionState.PLAYING,
    ])
    def test_valid_in_active_states(self, state):
        session = make_session(state)
        session.cancel()
        assert session.state == SessionState.CANCELLED

    @pytest.mark.parametrize("state", [
        SessionState.DONE, SessionState.DISCARDED,
        SessionState.CANCELLED, SessionState.ERROR,
    ])
    def test_invalid_in_terminal_or_already_settled_states(self, state):
        session = make_session(state)
        with pytest.raises(InvalidSessionStateError):
            session.cancel()


class TestDiscard:
    @pytest.mark.parametrize("state", [
        SessionState.DONE, SessionState.CANCELLED, SessionState.ERROR,
    ])
    def test_valid_in_terminal_predecessors(self, state):
        session = make_session(state)
        session.complete_audio = np.array([0.5], dtype=np.float32)
        session.discard()
        assert session.state == SessionState.DISCARDED
        # AC #6/D-7 — discard frees the buffer
        assert session.complete_audio is None

    @pytest.mark.parametrize("state", [
        SessionState.PENDING, SessionState.GENERATING,
        SessionState.READY_TO_PLAY, SessionState.PLAYING,
        SessionState.DISCARDED,
    ])
    def test_invalid_outside_terminal_predecessors(self, state):
        session = make_session(state)
        with pytest.raises(InvalidSessionStateError):
            session.discard()


# --------------------------------------------------------------------------- #
# AC #7 — clone_for_replay zero-copy semantics
# --------------------------------------------------------------------------- #

class TestCloneForReplay:
    def test_requires_complete_audio(self):
        session = make_session(SessionState.READY_TO_PLAY)
        # complete_audio is None at construction time
        assert session.complete_audio is None
        # Missing audio is a precondition violation, not a state-machine
        # violation — must raise ValueError with an actionable message.
        with pytest.raises(ValueError, match="complete_audio"):
            session.clone_for_replay()

    def test_shares_buffer_by_reference(self):
        original = make_session(SessionState.READY_TO_PLAY)
        original.complete_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        clone = original.clone_for_replay()
        # D-6: zero-copy
        assert clone.complete_audio is original.complete_audio

    def test_clone_has_distinct_session_id(self):
        original = make_session(SessionState.READY_TO_PLAY)
        original.complete_audio = np.array([0.0], dtype=np.float32)
        clone = original.clone_for_replay()
        assert clone.session_id != original.session_id

    def test_clone_starts_in_ready_to_play(self):
        original = make_session(SessionState.READY_TO_PLAY)
        original.complete_audio = np.array([0.0], dtype=np.float32)
        clone = original.clone_for_replay()
        assert clone.state == SessionState.READY_TO_PLAY

    def test_clone_default_source_is_preloaded(self):
        original = make_session(SessionState.READY_TO_PLAY)
        original.complete_audio = np.array([0.0], dtype=np.float32)
        clone = original.clone_for_replay()
        assert clone.source == SessionSource.PRELOADED

    def test_clone_source_can_be_overridden(self):
        original = make_session(SessionState.READY_TO_PLAY)
        original.complete_audio = np.array([0.0], dtype=np.float32)
        clone = original.clone_for_replay(source=SessionSource.GENERATED)
        assert clone.source == SessionSource.GENERATED

    def test_clone_preserves_text_and_voice(self):
        original = make_session(SessionState.READY_TO_PLAY)
        original.complete_audio = np.array([0.0], dtype=np.float32)
        clone = original.clone_for_replay()
        assert clone.text == original.text
        assert clone.voice == original.voice


# --------------------------------------------------------------------------- #
# AC #5 / #8 — _transition_to behavior
# --------------------------------------------------------------------------- #

class TestTransitionTo:
    def test_valid_transition_records_duration(self):
        session = make_session(SessionState.PENDING)
        session._transition_to(SessionState.GENERATING)
        assert session.state == SessionState.GENERATING
        assert SessionState.PENDING in session._state_durations

    def test_invalid_transition_raises(self):
        session = make_session(SessionState.PENDING)
        with pytest.raises(InvalidSessionStateError):
            session._transition_to(SessionState.DONE)

    def test_transition_does_not_emit_signal(self):
        # Defensive: there is no signal mechanism on the session at all.
        session = make_session(SessionState.PENDING)
        assert not hasattr(session, "emit")
        assert not hasattr(session, "stateChanged")

    def test_state_durations_are_per_state_not_cumulative(self):
        # Story 11.3 reads `_state_durations` for the
        # `session_state_duration_ms` metric. Each entry must be the time
        # spent in that state, NOT cumulative wall time since session
        # creation.
        session = GenerationSession(
            text="t", voice="v", source=SessionSource.GENERATED,
        )
        time.sleep(0.05)
        session._transition_to(SessionState.GENERATING)
        time.sleep(0.05)
        session._transition_to(SessionState.READY_TO_PLAY)
        pending_dur = session._state_durations[SessionState.PENDING]
        generating_dur = session._state_durations[SessionState.GENERATING]
        # Each interval was ~50ms; allow generous slack for timer jitter.
        assert 0.04 <= pending_dur <= 0.20, pending_dur
        assert 0.04 <= generating_dur <= 0.20, generating_dur
        # If durations were cumulative, generating_dur would be ~100ms
        # (50ms pending + 50ms generating). Guard explicitly.
        assert generating_dur < pending_dur + 0.04, (
            f"generating_dur ({generating_dur}) looks cumulative — "
            f"pending was {pending_dur}"
        )


# --------------------------------------------------------------------------- #
# AC #9 — module boundary discipline (static analysis)
# --------------------------------------------------------------------------- #

class TestModuleBoundaries:
    def setup_method(self):
        self.module_path = Path(inspect.getfile(gs_module))
        self.source = self.module_path.read_text(encoding="utf-8")

    def test_no_pyqt_in_source(self):
        forbidden = ["PyQt6", "PyQt5", "pyqtSignal", "QObject"]
        for needle in forbidden:
            assert needle not in self.source, (
                f"{needle!r} must not appear in generation_session.py "
                f"— signal emission belongs to the registry (Story 11.2)"
            )

    def test_no_peer_service_imports(self):
        # The module may import from `myvoice.services.sessions` (its own
        # package), but not from any other `myvoice.services.*` peer.
        peer_imports = re.findall(
            r"^\s*from\s+myvoice\.services\.([a-zA-Z_]\w*)",
            self.source,
            re.MULTILINE,
        )
        peer_imports += re.findall(
            r"^\s*import\s+myvoice\.services\.([a-zA-Z_]\w*)",
            self.source,
            re.MULTILINE,
        )
        for module_name in peer_imports:
            assert module_name == "sessions", (
                f"generation_session.py may not import from peer service "
                f"'myvoice.services.{module_name}'"
            )

    def test_no_ui_imports(self):
        assert "myvoice.ui" not in self.source

    def test_imports_only_from_allowlist(self):
        # Collect top-level imports and verify they are in the allowed set.
        allowed = {
            "enum", "dataclasses", "typing", "uuid", "time", "numpy",
        }
        # Match `import X` and `from X import ...` (top-level X only).
        import_pattern = re.compile(
            r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))",
            re.MULTILINE,
        )
        for match in import_pattern.finditer(self.source):
            module = (match.group(1) or match.group(2)).split(".")[0]
            # `__future__` is benign and not in the spec list, allow it.
            if module == "__future__":
                continue
            assert module in allowed, (
                f"generation_session.py imports disallowed module {module!r}; "
                f"allowed: {sorted(allowed)}"
            )


# --------------------------------------------------------------------------- #
# AC #5 — direct-state-assignment guard
# --------------------------------------------------------------------------- #

class TestDirectAssignmentGuard:
    def test_self_state_assigned_only_inside_transition_to(self):
        source = Path(inspect.getfile(gs_module)).read_text(encoding="utf-8")
        # Find every occurrence of `self.state = X` (assignment) — NOT
        # `self.state == X` (comparison). The negative lookahead `(?!=)`
        # excludes `==`. We also strip comments so docstring/comment
        # references like `self.state =` in a comment don't count.
        lines_no_comments = [
            re.sub(r"#.*$", "", line)
            for line in source.splitlines()
        ]
        occurrences = [
            (idx, line) for idx, line in enumerate(lines_no_comments, start=1)
            if re.search(r"\bself\.state\s*=(?!=)", line)
        ]
        assert len(occurrences) == 1, (
            f"Expected exactly one `self.state = ...` assignment "
            f"(inside _transition_to); found {len(occurrences)}: {occurrences}"
        )
        # Walk backwards from the match line to find the enclosing `def`.
        match_line_no = occurrences[0][0]
        lines = source.splitlines()
        enclosing_def = None
        for i in range(match_line_no - 1, -1, -1):
            stripped = lines[i].lstrip()
            if stripped.startswith("def "):
                enclosing_def = stripped
                break
        assert enclosing_def is not None and "_transition_to" in enclosing_def, (
            f"`self.state = ...` must live inside `_transition_to`; "
            f"found inside: {enclosing_def}"
        )

    def test_no_object_setattr_against_state_in_production(self):
        # AC #5 intent: state mutations live exclusively in `_transition_to`.
        # The literal `\bself\.state\s*=` regex above misses backdoor
        # patterns like `object.__setattr__(x, "state", ...)` and
        # `setattr(x, "state", ...)`. This test plugs that hole — production
        # code must never bypass the chokepoint via the descriptor protocol.
        source = Path(inspect.getfile(gs_module)).read_text(encoding="utf-8")
        # Strip comments so docstring/comment references to the pattern
        # don't trip the scan.
        code_only = re.sub(r"#.*$", "", source, flags=re.MULTILINE)
        forbidden_patterns = [
            r"object\.__setattr__\([^,]+,\s*[\"']state[\"']",
            r"\bsetattr\([^,]+,\s*[\"']state[\"']",
            r"\.__setattr__\([\"']state[\"']",
        ]
        for pattern in forbidden_patterns:
            match = re.search(pattern, code_only)
            assert match is None, (
                f"Production code uses {pattern!r} to mutate `state` "
                f"— forbidden by P-2; route through `_transition_to`. "
                f"Match: {match.group(0)!r}"
            )


# --------------------------------------------------------------------------- #
# AC #1 / __init__.py — public exports
# --------------------------------------------------------------------------- #

class TestPostInit:
    """P-2 entry-state guard — sessions enter in PENDING (normal path) or in
    READY_TO_PLAY with `complete_audio` set (the replay-clone signature)."""

    def test_default_construction_succeeds(self):
        session = GenerationSession(
            text="t", voice="v", source=SessionSource.GENERATED,
        )
        assert session.state == SessionState.PENDING

    @pytest.mark.parametrize("state", [
        s for s in SessionState
        if s not in (SessionState.PENDING, SessionState.READY_TO_PLAY)
    ])
    def test_explicit_non_entry_state_raises(self, state):
        with pytest.raises(ValueError, match="PENDING"):
            GenerationSession(
                text="t", voice="v", source=SessionSource.GENERATED,
                state=state,
            )

    def test_ready_to_play_without_complete_audio_raises(self):
        # READY_TO_PLAY entry is only valid when complete_audio is already
        # attached — the replay-clone signature.
        with pytest.raises(ValueError, match="PENDING"):
            GenerationSession(
                text="t", voice="v", source=SessionSource.GENERATED,
                state=SessionState.READY_TO_PLAY,
            )

    def test_ready_to_play_with_complete_audio_succeeds(self):
        # The replay-clone construction signature: explicit READY_TO_PLAY
        # with complete_audio attached. No out-of-band mutation needed.
        session = GenerationSession(
            text="t", voice="v", source=SessionSource.PRELOADED,
            state=SessionState.READY_TO_PLAY,
            complete_audio=np.array([0.0], dtype=np.float32),
        )
        assert session.state == SessionState.READY_TO_PLAY
        assert session.complete_audio is not None

    def test_clone_for_replay_still_starts_in_ready_to_play(self):
        # Regression: the new entry-state guard must accept legitimate
        # clones, which now use the direct READY_TO_PLAY construction
        # signature instead of an out-of-band mutation.
        original = make_session(SessionState.READY_TO_PLAY)
        original.complete_audio = np.array([0.0], dtype=np.float32)
        clone = original.clone_for_replay()
        assert clone.state == SessionState.READY_TO_PLAY


class TestPublicExports:
    def test_all_required_names_exported(self):
        from myvoice.services import sessions
        for name in (
            "GenerationSession", "SessionState", "SessionSource",
            "InvalidSessionStateError", "_VALID_TRANSITIONS",
        ):
            assert hasattr(sessions, name), f"sessions package must export {name}"


# --------------------------------------------------------------------------- #
# InvalidSessionStateError payload shape (AC #4)
# --------------------------------------------------------------------------- #

class TestInvalidSessionStateError:
    def test_attributes_are_set(self):
        err = InvalidSessionStateError(
            current_state=SessionState.PENDING,
            method="finalize",
            expected_states=frozenset({SessionState.GENERATING}),
        )
        assert err.current_state == SessionState.PENDING
        assert err.method == "finalize"
        assert err.expected_states == frozenset({SessionState.GENERATING})

    def test_str_mentions_method_and_states(self):
        err = InvalidSessionStateError(
            current_state=SessionState.PENDING,
            method="finalize",
            expected_states=frozenset({SessionState.GENERATING}),
        )
        message = str(err)
        assert "finalize" in message
        assert "PENDING" in message
        assert "GENERATING" in message
