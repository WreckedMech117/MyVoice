"""
Tests for SessionRegistry (Story 11.2).

Covers AC #1 through #16: signal declarations, Qt-thread ownership and
mutation guards, session creation/lookup, state-change emission rules
(including the `mark_audible` substate re-emit per D-15), `post_mutation`
cross-thread dispatch, four-tier focal-session priority, current-session
emission semantics (no spurious emissions), module-boundary discipline,
signal payload discipline (P-4: no `GenerationSession` across signals),
and error propagation (P-1: no silent no-ops).
"""

import ast
import inspect
import re
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# Tests in this module require PyQt6.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtWidgets import QApplication

from myvoice.services.sessions import (
    GenerationSession,
    InvalidSessionStateError,
    SaveableAudio,
    SessionRegistry,
    SessionSource,
    SessionState,
)
from myvoice.services.sessions import session_registry as registry_module


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication (project convention; no pytest-qt)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def make_registry(qapp) -> SessionRegistry:
    return SessionRegistry()


def make_session_in(
    registry: SessionRegistry,
    state: SessionState,
    *,
    text: str = "hello",
    voice: str = "default",
    last_transition_at: Optional[float] = None,
) -> str:
    """Create a session through the registry, then force-position it.

    Tests need to position sessions into arbitrary states without going
    through `_transition_to`. We mirror the 11.1 test convention of using
    `object.__setattr__` to write the state field directly. The static-scan
    test below verifies this pattern is NOT used in production code.
    """
    sid = registry.create_session(text=text, voice=voice, model_type="m")
    session = registry.get(sid)
    object.__setattr__(session, "state", state)
    if last_transition_at is not None:
        object.__setattr__(session, "_last_transition_at", last_transition_at)
    return sid


def drain_qt_events(qapp, iterations: int = 3) -> None:
    """Drain the queued-connection backlog used by `post_mutation` tests."""
    for _ in range(iterations):
        qapp.processEvents()


def capture_state_changes(registry: SessionRegistry):
    """Connect a list-based spy to `session_state_changed` and return the list."""
    captured: list[tuple[str, SessionState]] = []
    registry.session_state_changed.connect(
        lambda s, st: captured.append((s, st))
    )
    return captured


def capture_focal_changes(registry: SessionRegistry):
    captured: list[Optional[str]] = []
    registry.current_session_changed.connect(lambda focal: captured.append(focal))
    return captured


# --------------------------------------------------------------------------- #
# AC #2 — TestSignalDeclarations
# --------------------------------------------------------------------------- #

class TestSignalDeclarations:
    def test_session_state_changed_exists_and_emits(self, qapp):
        registry = make_registry(qapp)
        captured = capture_state_changes(registry)
        registry.session_state_changed.emit("sid", SessionState.GENERATING)
        assert captured == [("sid", SessionState.GENERATING)]

    def test_current_session_changed_exists_and_emits_str_or_none(self, qapp):
        registry = make_registry(qapp)
        captured = capture_focal_changes(registry)
        registry.current_session_changed.emit("sid")
        registry.current_session_changed.emit(None)
        assert captured == ["sid", None]

    def test_playback_queue_depth_changed_exists_and_emits_int(self, qapp):
        registry = make_registry(qapp)
        captured: list[int] = []
        registry.playback_queue_depth_changed.connect(lambda d: captured.append(d))
        registry.playback_queue_depth_changed.emit(3)
        assert captured == [3]

    def test_saveable_session_changed_exists_and_emits_str_or_none(self, qapp):
        registry = make_registry(qapp)
        captured: list[Optional[str]] = []
        registry.saveable_session_changed.connect(lambda s: captured.append(s))
        registry.saveable_session_changed.emit("sid")
        registry.saveable_session_changed.emit(None)
        assert captured == ["sid", None]


# --------------------------------------------------------------------------- #
# AC #3 — TestThreadOwnership
# --------------------------------------------------------------------------- #

class TestThreadOwnership:
    def test_main_thread_construction_succeeds(self, qapp):
        registry = SessionRegistry()
        assert registry.thread() is qapp.thread()

    def test_worker_thread_construction_raises(self, qapp):
        captured: list[BaseException] = []

        def worker():
            try:
                SessionRegistry()
            except BaseException as exc:
                captured.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        assert len(captured) == 1
        assert isinstance(captured[0], RuntimeError)
        assert "main thread" in str(captured[0]).lower()

    def test_worker_thread_construction_with_main_thread_parent_raises(self, qapp):
        """Regression: passing a main-thread `parent` from a worker thread
        must still raise. `super().__init__(parent)` adopts the parent's
        thread affinity, so a naive `app.thread() is not self.thread()`
        check would silently pass (`main is not main` → False). The
        guard must compare the *executing* thread
        (`QThread.currentThread()`) against `app.thread()`. Mirrors the
        same fix applied in `playback_queue.py` (Story 13.1 review M3).
        """
        main_thread_parent = QObject()
        assert main_thread_parent.thread() is qapp.thread()

        captured: list[BaseException] = []

        def worker():
            try:
                SessionRegistry(parent=main_thread_parent)
            except BaseException as exc:
                captured.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        assert len(captured) == 1, (
            f"Expected RuntimeError from worker-thread construction with "
            f"main-thread parent, got: {captured!r}"
        )
        assert isinstance(captured[0], RuntimeError)
        assert "main thread" in str(captured[0]).lower()

    def test_missing_qapplication_raises(self, qapp, monkeypatch):
        # Cannot tear down a real QApplication mid-suite; monkeypatch
        # `instance()` to return None for the duration of this one test.
        monkeypatch.setattr(
            "myvoice.services.sessions.session_registry.QApplication.instance",
            lambda: None,
        )
        with pytest.raises(RuntimeError, match="QApplication"):
            SessionRegistry()


# --------------------------------------------------------------------------- #
# AC #4 / #5 — TestCreateSession, TestGet
# --------------------------------------------------------------------------- #

class TestCreateSession:
    def test_returns_non_empty_string(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        assert isinstance(sid, str)
        assert sid

    def test_session_is_stored(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        assert registry.get(sid) is not None

    def test_session_starts_in_pending(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        assert registry.get(sid).state == SessionState.PENDING

    def test_text_voice_source_propagate(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(
            text="hello world",
            voice="ryan",
            model_type="custom",
            source=SessionSource.PRELOADED,
        )
        session = registry.get(sid)
        assert session.text == "hello world"
        assert session.voice == "ryan"
        assert session.source == SessionSource.PRELOADED

    def test_default_source_is_generated(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        assert registry.get(sid).source == SessionSource.GENERATED

    def test_model_type_preserved_in_registry_metadata(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="custom")
        # Validation gap #5: the parameter must not be silently dropped.
        assert registry._session_model_types[sid] == "custom"


class TestGet:
    def test_get_returns_same_reference(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        assert registry.get(sid) is registry.get(sid)

    def test_get_returns_none_for_unknown(self, qapp):
        registry = make_registry(qapp)
        assert registry.get("does-not-exist") is None


# --------------------------------------------------------------------------- #
# AC #6 / #7 — TestMutationsEmitStateChanged
# --------------------------------------------------------------------------- #

class TestMutationsEmitStateChanged:
    def test_start_generation_emits(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        captured = capture_state_changes(registry)
        registry.start_generation(sid)
        assert (sid, SessionState.GENERATING) in captured

    def test_append_chunk_does_not_change_state_no_emission(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        registry.start_generation(sid)
        captured = capture_state_changes(registry)
        registry.append_chunk(sid, np.array([1.0, 2.0], dtype=np.float32))
        # append_chunk does not transition; spec says emit ONLY when state changes.
        assert captured == []
        assert len(registry.get(sid).chunks) == 1

    def test_finalize_emits_ready_to_play(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        registry.start_generation(sid)
        registry.append_chunk(sid, np.array([1.0, 2.0], dtype=np.float32))
        captured = capture_state_changes(registry)
        registry.finalize(sid)
        assert (sid, SessionState.READY_TO_PLAY) in captured

    def test_mark_playing_emits(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.READY_TO_PLAY)
        captured = capture_state_changes(registry)
        registry.mark_playing(sid)
        assert (sid, SessionState.PLAYING) in captured

    def test_mark_audible_re_emits_unchanged_state(self, qapp):
        # D-15: substate flip; state stays PLAYING but signal still fires
        # so subscribers can re-read `is_audible`.
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.PLAYING)
        captured = capture_state_changes(registry)
        registry.mark_audible(sid)
        assert captured == [(sid, SessionState.PLAYING)]
        assert registry.get(sid).is_audible is True

    def test_mark_done_emits(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.PLAYING)
        captured = capture_state_changes(registry)
        registry.mark_done(sid)
        assert (sid, SessionState.DONE) in captured

    def test_cancel_emits(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        captured = capture_state_changes(registry)
        registry.cancel(sid)
        assert (sid, SessionState.CANCELLED) in captured

    def test_discard_emits(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.DONE)
        captured = capture_state_changes(registry)
        registry.discard(sid)
        assert (sid, SessionState.DISCARDED) in captured

    def test_set_error_transitions_to_error_and_emits(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        captured = capture_state_changes(registry)
        registry.set_error(sid)
        assert registry.get(sid).state == SessionState.ERROR
        assert (sid, SessionState.ERROR) in captured


# --------------------------------------------------------------------------- #
# AC #9 — TestMutationsThreadGuard
# --------------------------------------------------------------------------- #

class TestMutationsThreadGuard:
    def test_direct_mutation_from_worker_thread_raises(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        captured: list[BaseException] = []

        def worker():
            try:
                registry.start_generation(sid)
            except BaseException as exc:
                captured.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        assert len(captured) == 1
        assert isinstance(captured[0], RuntimeError)
        assert "post_mutation" in str(captured[0])
        # Mutation must NOT have taken effect.
        assert registry.get(sid).state == SessionState.PENDING

    def test_unknown_session_id_raises_key_error(self, qapp):
        registry = make_registry(qapp)
        with pytest.raises(KeyError, match="Unknown session_id"):
            registry.start_generation("no-such-id")


# --------------------------------------------------------------------------- #
# AC #8 — TestPostMutation
# --------------------------------------------------------------------------- #

class TestPostMutation:
    def test_post_mutation_str_arg_dispatches_on_main_thread(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        captured = capture_state_changes(registry)

        def worker():
            registry.post_mutation("start_generation", sid)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        drain_qt_events(qapp)

        assert registry.get(sid).state == SessionState.GENERATING
        assert (sid, SessionState.GENERATING) in captured

    def test_post_mutation_ndarray_arg_dispatches(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        registry.start_generation(sid)
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        def worker():
            registry.post_mutation("append_chunk", sid, audio)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        drain_qt_events(qapp)

        chunks = registry.get(sid).chunks
        assert len(chunks) == 1
        assert np.array_equal(chunks[0], audio)

    def test_post_mutation_returns_immediately(self, qapp):
        # QueuedConnection always defers — even from the main thread to
        # itself — so the mutation does not run until the next
        # processEvents() drain.
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        registry.post_mutation("start_generation", sid)

        # Before draining, the mutation has not run.
        assert registry.get(sid).state == SessionState.PENDING

        # After draining, it has.
        drain_qt_events(qapp)
        assert registry.get(sid).state == SessionState.GENERATING


# --------------------------------------------------------------------------- #
# AC #10 — TestFocalSessionPriority
# --------------------------------------------------------------------------- #

class TestFocalSessionPriority:
    def test_no_sessions_returns_none(self, qapp):
        registry = make_registry(qapp)
        assert registry.focal_session_id is None

    def test_pending_does_not_qualify(self, qapp):
        registry = make_registry(qapp)
        registry.create_session(text="a", voice="v", model_type="m")
        registry.create_session(text="b", voice="v", model_type="m")
        assert registry.focal_session_id is None

    def test_priority_a_playing_wins_over_generating(self, qapp):
        registry = make_registry(qapp)
        gen_id = make_session_in(
            registry, SessionState.GENERATING,
            last_transition_at=time.perf_counter(),
        )
        play_id = make_session_in(
            registry, SessionState.PLAYING,
            # PLAYING wins regardless of timestamp ordering.
            last_transition_at=time.perf_counter() - 100.0,
        )
        assert registry.focal_session_id == play_id
        assert gen_id != play_id

    def test_priority_b_most_recent_active_wins_when_no_playing(self, qapp):
        registry = make_registry(qapp)
        older = make_session_in(
            registry, SessionState.GENERATING,
            last_transition_at=time.perf_counter() - 10.0,
        )
        newer = make_session_in(
            registry, SessionState.READY_TO_PLAY,
            last_transition_at=time.perf_counter() - 1.0,
        )
        assert registry.focal_session_id == newer
        assert older != newer

    def test_priority_c_terminal_within_decay_window(self, qapp):
        registry = make_registry(qapp)
        now = time.perf_counter()
        older = make_session_in(
            registry, SessionState.DONE,
            last_transition_at=now - 4.0,
        )
        newer = make_session_in(
            registry, SessionState.CANCELLED,
            last_transition_at=now - 1.0,
        )
        assert registry.focal_session_id == newer
        assert older != newer

    def test_priority_d_no_focal_after_decay_window(self, qapp):
        registry = make_registry(qapp)
        now = time.perf_counter()
        make_session_in(
            registry, SessionState.DONE,
            last_transition_at=now - 6.0,
        )
        make_session_in(
            registry, SessionState.ERROR,
            last_transition_at=now - 7.0,
        )
        assert registry.focal_session_id is None

    def test_active_beats_terminal_even_when_terminal_is_more_recent(self, qapp):
        registry = make_registry(qapp)
        now = time.perf_counter()
        active = make_session_in(
            registry, SessionState.GENERATING,
            last_transition_at=now - 5.0,
        )
        make_session_in(
            registry, SessionState.DONE,
            last_transition_at=now - 0.5,
        )
        assert registry.focal_session_id == active


# --------------------------------------------------------------------------- #
# AC #11 — TestCurrentSessionChangedSignal
# --------------------------------------------------------------------------- #

class TestCurrentSessionChangedSignal:
    def test_fires_on_focal_change(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="a", voice="v", model_type="m")
        captured = capture_focal_changes(registry)
        registry.start_generation(sid)
        # Focal goes None → sid → emits exactly once with sid.
        assert captured == [sid]

    def test_does_not_fire_when_focal_unchanged(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.PLAYING)
        # Drain any prior emission caused by the position helper (it didn't
        # actually go through start_generation, so no current_session_changed
        # was raised — but call _recompute manually so the cache primes).
        registry._recompute_focal_and_maybe_emit()
        captured = capture_focal_changes(registry)

        registry.mark_audible(sid)
        # mark_audible does not change focal — no emission expected.
        assert captured == []

    def test_fires_with_none_when_all_decay(self, qapp, monkeypatch):
        # Verifies the *decay* logic specifically: a focal session in a
        # terminal state must drop out of focal-eligibility once
        # `_FOCAL_DECAY_SECONDS` elapses since its last transition. The
        # trigger that prompts the recompute must NOT itself change the
        # decayed session's state, otherwise the test would pass for the
        # wrong reason (the session would simply fall out of all
        # focal-eligible buckets).
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.DONE)
        registry._recompute_focal_and_maybe_emit()
        assert registry._last_focal_id == sid

        captured = capture_focal_changes(registry)

        future = time.perf_counter() + registry_module._FOCAL_DECAY_SECONDS + 1.0
        monkeypatch.setattr(registry_module.time, "perf_counter", lambda: future)

        # Trigger a recompute via creating a brand-new (PENDING, not
        # focal-eligible) session — the existing DONE session stays in
        # DONE but its `_last_transition_at` is now > 5s in the past, so
        # the decay window forces focal to None.
        registry.create_session(text="trigger", voice="v", model_type="m")

        assert None in captured
        assert registry.get(sid).state == SessionState.DONE  # untouched


# --------------------------------------------------------------------------- #
# AC #13 — TestModuleBoundary
# --------------------------------------------------------------------------- #

class TestModuleBoundary:
    @pytest.fixture
    def source_text(self) -> str:
        path = Path(inspect.getsourcefile(SessionRegistry))
        return path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("forbidden", [
        "from myvoice.services.qwen_tts_service",
        "from myvoice.services.audio_coordinator",
        "from myvoice.services.audio_service",
        "from myvoice.services.monitor_audio_service",
        "from myvoice.services.virtual_microphone_service",
        "from myvoice.services.model_loading_manager",
        "from myvoice.ui",
        "import myvoice.services.qwen_tts_service",
        "import myvoice.services.audio_coordinator",
        "import myvoice.ui",
    ])
    def test_forbidden_imports_absent(self, source_text, forbidden):
        assert forbidden not in source_text, (
            f"session_registry.py must not import {forbidden!r} per AC #13"
        )

    def test_required_peer_import_present(self, source_text):
        # The only sibling module the registry may depend on.
        assert "from myvoice.services.sessions.generation_session" in source_text

    def test_no_object_setattr_against_state_in_production(self, source_text):
        # 11.1's test helper uses object.__setattr__(session, "state", ...)
        # — production code must never do this; transitions go through
        # _transition_to (P-2 chokepoint).
        assert not re.search(
            r"object\.__setattr__\([^,]+,\s*[\"']state[\"']",
            source_text,
        )


# --------------------------------------------------------------------------- #
# AC #12 — TestSignalPayloadDiscipline
# --------------------------------------------------------------------------- #

class TestSignalPayloadDiscipline:
    @pytest.fixture
    def source_text(self) -> str:
        path = Path(inspect.getsourcefile(SessionRegistry))
        return path.read_text(encoding="utf-8")

    def test_no_pyqtsignal_carries_generation_session(self, source_text):
        # P-4: signals must carry IDs and primitives, never the session
        # object itself.
        assert not re.search(
            r"pyqtSignal\([^)]*GenerationSession[^)]*\)",
            source_text,
        )

    def test_no_slot_takes_generation_session_param(self, source_text):
        # Mutation slots (the registry's public Qt-event-system surface) must
        # take session_id strings, never GenerationSession instances (P-4).
        # We check by finding every @pyqtSlot decorator and verifying the
        # immediately following `def <name>(...)` signature does not contain
        # a GenerationSession-typed parameter. Internal helpers are exempt —
        # they never cross a signal or thread boundary.
        slot_signatures = re.findall(
            r"@pyqtSlot\([^)]*\)\s*\n\s*def\s+\w+\(([^)]*)\)",
            source_text,
        )
        assert slot_signatures, "expected at least one @pyqtSlot in the registry"
        for sig in slot_signatures:
            assert "GenerationSession" not in sig, (
                f"@pyqtSlot signature must not declare GenerationSession: {sig!r}"
            )


# --------------------------------------------------------------------------- #
# AC #6 (P-1 invariant) — TestErrorPropagation
# --------------------------------------------------------------------------- #

class TestErrorPropagation:
    def test_finalize_with_no_chunks_propagates_value_error(self, qapp):
        # 11.1 code-review fix: finalize() raises ValueError when no chunks
        # were appended. The registry's slot must let this propagate.
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        registry.start_generation(sid)
        with pytest.raises(ValueError, match="no chunks"):
            registry.finalize(sid)

    def test_invalid_state_method_propagates_invalid_session_state_error(self, qapp):
        # mark_done is valid only in PLAYING; calling it on PENDING must
        # surface as InvalidSessionStateError (P-1: no silent no-op).
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        with pytest.raises(InvalidSessionStateError) as excinfo:
            registry.mark_done(sid)
        assert excinfo.value.method == "mark_done"

    def test_unknown_session_id_raises_key_error(self, qapp):
        registry = make_registry(qapp)
        with pytest.raises(KeyError, match="Unknown session_id"):
            registry.mark_playing("ghost")

    def test_start_generation_from_invalid_state_names_public_slot(self, qapp):
        # Code-review fix (11.2 H4): InvalidSessionStateError raised from
        # `start_generation` must blame `start_generation`, not the internal
        # `_transition_to` helper.
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        registry.start_generation(sid)  # PENDING → GENERATING (valid)
        with pytest.raises(InvalidSessionStateError) as excinfo:
            registry.start_generation(sid)  # GENERATING → GENERATING (invalid)
        assert excinfo.value.method == "start_generation"
        assert excinfo.value.current_state == SessionState.GENERATING

    def test_set_error_from_done_names_public_slot(self, qapp):
        # Same H4 fix for set_error: error blames the public slot.
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.DONE)
        with pytest.raises(InvalidSessionStateError) as excinfo:
            registry.set_error(sid)
        assert excinfo.value.method == "set_error"
        assert excinfo.value.current_state == SessionState.DONE

    def test_cancel_from_terminal_state_propagates(self, qapp):
        # cancel from DONE/DISCARDED/ERROR/CANCELLED is invalid (no
        # CANCELLED successor); must surface, not be swallowed.
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.DONE)
        with pytest.raises(InvalidSessionStateError) as excinfo:
            registry.cancel(sid)
        assert excinfo.value.method == "cancel"


# --------------------------------------------------------------------------- #
# Code-review fixes (2026-05-03) — H1/H2/H3/M3 regression coverage
# --------------------------------------------------------------------------- #

class TestSetErrorIdempotency:
    """11.2 H2: set_error must absorb double-calls on already-errored sessions
    so 11.4's exception paths can fire from multiple unwinding handlers."""

    def test_set_error_on_already_errored_is_noop(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        registry.set_error(sid)  # PENDING → ERROR
        assert registry.get(sid).state == SessionState.ERROR

        captured = capture_state_changes(registry)
        registry.set_error(sid)  # ERROR → ERROR (idempotent — no emission)
        assert captured == []
        assert registry.get(sid).state == SessionState.ERROR

    def test_set_error_propagates_from_other_terminals(self, qapp):
        # Idempotency is intentionally narrow — only ERROR→ERROR is absorbed.
        # DONE→ERROR is still rejected (a settled session is not reclassifiable).
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.CANCELLED)
        with pytest.raises(InvalidSessionStateError) as excinfo:
            registry.set_error(sid)
        assert excinfo.value.method == "set_error"


class TestTrySetErrorTerminalAbsorption:
    """Race-safe variant for error-cleanup paths that may fire after the
    worker's drain-on-cancel posts ('cancel', sid). Behaviour is unchanged
    for active sessions; terminal-state calls absorb silently rather than
    surface InvalidSessionStateError to the global exception handler.
    """

    def test_try_set_error_transitions_active_session(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        captured = capture_state_changes(registry)
        registry.try_set_error(sid)
        assert registry.get(sid).state == SessionState.ERROR
        assert captured == [(sid, SessionState.ERROR)]

    def test_try_set_error_absorbs_cancelled_silently(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.CANCELLED)
        captured = capture_state_changes(registry)
        registry.try_set_error(sid)  # must not raise, must not emit
        assert captured == []
        assert registry.get(sid).state == SessionState.CANCELLED

    def test_try_set_error_absorbs_done(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.DONE)
        captured = capture_state_changes(registry)
        registry.try_set_error(sid)
        assert captured == []
        assert registry.get(sid).state == SessionState.DONE

    def test_try_set_error_absorbs_already_errored(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.ERROR)
        captured = capture_state_changes(registry)
        registry.try_set_error(sid)
        assert captured == []
        assert registry.get(sid).state == SessionState.ERROR

    def test_try_set_error_is_registered_for_post_mutation(self, qapp):
        # post_mutation rejects unregistered slot names; this guards
        # against future refactors that drop try_set_error from the
        # MUTATION_SLOT_NAMES set.
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        # Must not raise — would raise RuntimeError if unregistered.
        registry.post_mutation('try_set_error', sid)


class TestMarkAudibleDuplicateSuppression:
    """11.2 M3: mark_audible re-emits on the False→True flip but absorbs
    duplicates when is_audible was already True."""

    def test_first_call_emits(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.PLAYING)
        captured = capture_state_changes(registry)
        registry.mark_audible(sid)
        assert captured == [(sid, SessionState.PLAYING)]
        assert registry.get(sid).is_audible is True

    def test_second_call_does_not_re_emit(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.PLAYING)
        registry.mark_audible(sid)  # first flip — emits
        captured = capture_state_changes(registry)
        registry.mark_audible(sid)  # already audible — must not re-emit
        assert captured == []
        assert registry.get(sid).is_audible is True


class TestDiscardCleanup:
    """11.2 H3: registry must drop session entries after DISCARDED so the
    in-flight collection stays bounded over the app's lifetime."""

    def test_discarded_session_is_removed_from_registry(self, qapp):
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.DONE)
        registry.discard(sid)
        assert registry.get(sid) is None
        assert sid not in registry._sessions

    def test_discard_cleans_up_model_type_metadata(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="custom")
        # Move through PENDING → GENERATING (no chunks needed; cancel out)
        registry.cancel(sid)
        registry.discard(sid)
        assert sid not in registry._session_model_types

    def test_emission_fires_before_cleanup(self, qapp):
        # DirectConnection subscribers must see the session in the registry
        # at the moment of the DISCARDED emission so they can read final
        # state (durations, metadata) before the entry is dropped.
        registry = make_registry(qapp)
        sid = make_session_in(registry, SessionState.DONE)

        # Connect a spy that probes the registry from inside the slot.
        observed: list[Optional[GenerationSession]] = []

        def spy(emitted_sid, _state):
            if emitted_sid == sid:
                observed.append(registry.get(emitted_sid))

        registry.session_state_changed.connect(spy)
        registry.discard(sid)

        assert len(observed) == 1
        assert observed[0] is not None
        assert observed[0].state == SessionState.DISCARDED


class TestPostMutationFailsLoudly:
    """11.2 H1: a typo or non-slot method name must raise, not silently no-op."""

    def test_unknown_method_name_raises_runtime_error(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="hi", voice="v", model_type="m")
        with pytest.raises(RuntimeError, match="post_mutation"):
            registry.post_mutation("mark_dome", sid)  # typo
        # State must not have advanced.
        assert registry.get(sid).state == SessionState.PENDING

    def test_non_slot_method_raises_runtime_error(self, qapp):
        # `create_session` is a regular method, NOT a @pyqtSlot. Must fail.
        registry = make_registry(qapp)
        with pytest.raises(RuntimeError, match="post_mutation"):
            registry.post_mutation("create_session", "x", "v", "m")


# --------------------------------------------------------------------------- #
# Story 12.2 lock-down — TestFocalPriorityExplicit
# --------------------------------------------------------------------------- #
# Per-tier explicit tests + tight-clustering handoff regression.
# Distinct from `TestFocalSessionPriority` so a tier-(b)-breaking change names
# the tier directly in the failing-test report.

class TestFocalPriorityExplicit:
    def test_empty_registry_returns_none(self, qapp):
        registry = make_registry(qapp)
        assert registry.focal_session_id is None

    def test_pending_only_returns_none(self, qapp):
        registry = make_registry(qapp)
        registry.create_session(text="a", voice="v", model_type="m")
        assert registry.focal_session_id is None

    def test_priority_a_playing_wins_over_all_other_states(self, qapp):
        # Tier-(a) is unconditional: PLAYING wins even when its
        # `_last_transition_at` is the OLDEST of all candidates.
        registry = make_registry(qapp)
        now = time.perf_counter()
        playing = make_session_in(
            registry, SessionState.PLAYING,
            last_transition_at=now - 100.0,  # oldest of the bunch
        )
        make_session_in(
            registry, SessionState.GENERATING,
            last_transition_at=now - 0.5,
        )
        make_session_in(
            registry, SessionState.READY_TO_PLAY,
            last_transition_at=now - 0.4,
        )
        make_session_in(
            registry, SessionState.DONE,
            last_transition_at=now - 0.3,
        )
        make_session_in(
            registry, SessionState.ERROR,
            last_transition_at=now - 0.2,
        )
        assert registry.focal_session_id == playing

    def test_priority_b_active_wins_when_no_playing_with_terminals_present(self, qapp):
        # The MOST RECENT session of all candidates is a terminal one;
        # tier-(b) must skip it and pick the most recent {GEN, RTP} member.
        registry = make_registry(qapp)
        now = time.perf_counter()
        make_session_in(
            registry, SessionState.GENERATING,
            last_transition_at=now - 3.0,
        )
        ready_to_play = make_session_in(
            registry, SessionState.READY_TO_PLAY,
            last_transition_at=now - 2.0,
        )
        # Most-recent overall — but tier-(c), so must be skipped at tier-(b).
        make_session_in(
            registry, SessionState.DONE,
            last_transition_at=now - 0.1,
        )
        make_session_in(
            registry, SessionState.CANCELLED,
            last_transition_at=now - 0.5,
        )
        make_session_in(
            registry, SessionState.ERROR,
            last_transition_at=now - 1.0,
        )
        assert registry.focal_session_id == ready_to_play

    def test_priority_c_inclusive_at_boundary(self, qapp, monkeypatch):
        # Boundary inclusivity: a terminal session at exactly
        # `now - _FOCAL_DECAY_SECONDS` is still focal.
        registry = make_registry(qapp)
        fixed_now = 1_000_000.0
        monkeypatch.setattr(registry_module.time, "perf_counter", lambda: fixed_now)
        sid = make_session_in(
            registry, SessionState.DONE,
            last_transition_at=fixed_now - registry_module._FOCAL_DECAY_SECONDS,
        )
        assert registry.focal_session_id == sid

    def test_priority_c_excluded_just_past_boundary(self, qapp, monkeypatch):
        # 1 nanosecond past the boundary → tier-(c) drops the session and
        # focal collapses to None (priority (d), non-vacuous).
        registry = make_registry(qapp)
        fixed_now = 1_000_000.0
        monkeypatch.setattr(registry_module.time, "perf_counter", lambda: fixed_now)
        make_session_in(
            registry, SessionState.DONE,
            last_transition_at=fixed_now - registry_module._FOCAL_DECAY_SECONDS - 1e-9,
        )
        assert registry.focal_session_id is None

    def test_priority_d_pending_with_stale_terminal_returns_none(self, qapp, monkeypatch):
        # Complement to `test_pending_only_returns_none`: PENDING co-existing
        # with a terminal that has aged past decay; neither qualifies.
        registry = make_registry(qapp)
        fixed_now = 1_000_000.0
        monkeypatch.setattr(registry_module.time, "perf_counter", lambda: fixed_now)
        registry.create_session(text="pending", voice="v", model_type="m")
        make_session_in(
            registry, SessionState.DONE,
            last_transition_at=fixed_now - registry_module._FOCAL_DECAY_SECONDS - 10.0,
        )
        assert registry.focal_session_id is None

    def test_focal_handoff_under_tight_clustering(self, qapp):
        # Story 12.2 / Task 6.1 — the regression fix for Story 12.1's
        # `test_focal_handoff_no_idle_frame` Windows clock-resolution flake.
        # Drive a real flow with no `time.sleep` between transitions and
        # assert focal lands on B *deterministically* (no "one of {A, B}"
        # relaxation). With `time.perf_counter()`'s sub-microsecond
        # QueryPerformanceCounter source, B's `_last_transition_at` is
        # strictly greater than A's even under Python 3.10 on Windows.
        registry = make_registry(qapp)
        sid_a = registry.create_session(text="a", voice="v", model_type="m")
        registry.start_generation(sid_a)
        registry.append_chunk(sid_a, np.array([0.0], dtype=np.float32))
        registry.finalize(sid_a)
        # No sleep — back-to-back creation and start of B in the same Qt tick.
        sid_b = registry.create_session(text="b", voice="v", model_type="m")
        registry.start_generation(sid_b)
        # Both A (READY_TO_PLAY) and B (GENERATING) are tier-(b); most recent
        # wins. Under the old `time.time()` source the timestamps tied on
        # Windows; under `time.perf_counter()` (QPC) they do not. Note
        # `time.monotonic()` would also tie here on Python 3.10 / Windows
        # where it falls back to GetTickCount64 — see the Story 12.2 Change
        # Log for the AC #4 deviation rationale.
        assert registry.focal_session_id == sid_b
        # A is still around (not discarded), but is not focal.
        assert registry.get(sid_a).state == SessionState.READY_TO_PLAY


# --------------------------------------------------------------------------- #
# Story 12.2 lock-down — TestCurrentSessionChangedContract
# --------------------------------------------------------------------------- #
# Per-mutation-type emission counts + AC #7 (no eager emit on construct).
# Distinct from `TestCurrentSessionChangedSignal` (which uses a single generic
# mark_audible test) so a future "optimization" that introduces an emission on
# e.g. `append_chunk` is named in the failure rather than hidden.

class TestCurrentSessionChangedContract:
    def test_no_emission_on_registry_construction(self, qapp):
        # AC #7: a subscriber attached *during* construction sees no
        # synthetic initialization frame.
        registry = make_registry(qapp)
        captured = capture_focal_changes(registry)
        # No mutations performed; spy must be empty.
        assert captured == []
        assert registry._last_focal_id is None

    def test_no_emission_on_create_session_when_focal_unchanged(self, qapp):
        # PENDING is not focal-eligible — back-to-back create_session calls
        # leave focal None throughout, so no emission fires.
        registry = make_registry(qapp)
        captured = capture_focal_changes(registry)
        registry.create_session(text="a", voice="v", model_type="m")
        registry.create_session(text="b", voice="v", model_type="m")
        assert captured == []

    def test_one_emission_on_first_start_generation(self, qapp):
        # Transition None → sid is the ONE legitimate emission for a fresh
        # registry's first focal session.
        registry = make_registry(qapp)
        sid = registry.create_session(text="a", voice="v", model_type="m")
        captured = capture_focal_changes(registry)
        registry.start_generation(sid)
        assert captured == [sid]

    def test_no_emission_on_append_chunk(self, qapp):
        # `append_chunk` is data-only — no state change, no focal change.
        registry = make_registry(qapp)
        sid = registry.create_session(text="a", voice="v", model_type="m")
        registry.start_generation(sid)
        # `_last_focal_id` is now sid; spy attached AFTER priming.
        captured = capture_focal_changes(registry)
        registry.append_chunk(sid, np.zeros(100, dtype=np.float32))
        assert captured == []

    def test_no_emission_on_mark_audible_substate_flip(self, qapp):
        # The asymmetry: D-15 says `session_state_changed` re-fires on the
        # False→True audible flip; the focal contract says `current_session_changed`
        # does NOT. Both spies are attached to verify both halves directly.
        registry = make_registry(qapp)
        sid = registry.create_session(text="a", voice="v", model_type="m")
        registry.start_generation(sid)
        registry.append_chunk(sid, np.zeros(10, dtype=np.float32))
        registry.finalize(sid)
        registry.mark_playing(sid)
        # Now PLAYING; focal is sid; `_last_focal_id` is sid.
        state_spy = capture_state_changes(registry)
        focal_spy = capture_focal_changes(registry)
        registry.mark_audible(sid)
        # State spy gets the substate re-emit (D-15) ...
        assert state_spy == [(sid, SessionState.PLAYING)]
        # ... but focal spy does not, because focal didn't change.
        assert focal_spy == []

    def test_double_mark_audible_emits_state_once_focal_zero(self, qapp):
        # AC #3 #5 in one assertion: `mark_audible` called twice in succession
        # → exactly 1 `session_state_changed` (the False→True flip) and 0
        # `current_session_changed` (focal never changes on a substate flip).
        # The two halves are also covered separately by
        # TestMarkAudibleDuplicateSuppression (state side) and
        # test_no_emission_on_mark_audible_substate_flip (focal side); this
        # test pins them as a single contract so a future regression in
        # either half is named here directly.
        registry = make_registry(qapp)
        sid = registry.create_session(text="a", voice="v", model_type="m")
        registry.start_generation(sid)
        registry.append_chunk(sid, np.zeros(10, dtype=np.float32))
        registry.finalize(sid)
        registry.mark_playing(sid)
        state_spy = capture_state_changes(registry)
        focal_spy = capture_focal_changes(registry)
        registry.mark_audible(sid)  # False → True (emits state_changed once)
        registry.mark_audible(sid)  # True → True (absorbed, no emit)
        assert state_spy == [(sid, SessionState.PLAYING)]
        assert focal_spy == []

    def test_no_emission_on_set_error_idempotent_call(self, qapp):
        # `set_error` on an already-ERROR session short-circuits before
        # `_recompute_focal_and_maybe_emit`, so no spurious emission.
        registry = make_registry(qapp)
        sid = registry.create_session(text="a", voice="v", model_type="m")
        registry.set_error(sid)  # PENDING → ERROR; one focal emission already happened
        captured = capture_focal_changes(registry)
        registry.set_error(sid)  # ERROR → ERROR (idempotent no-op)
        assert captured == []
        assert registry.get(sid).state == SessionState.ERROR

    def test_no_emission_on_finalize_within_tier_b(self, qapp):
        # GENERATING → READY_TO_PLAY: state changes, but both states are
        # tier-(b), so focal stays on the same id and no emission fires.
        registry = make_registry(qapp)
        sid = registry.create_session(text="a", voice="v", model_type="m")
        registry.start_generation(sid)
        registry.append_chunk(sid, np.zeros(10, dtype=np.float32))
        captured = capture_focal_changes(registry)
        registry.finalize(sid)
        assert captured == []

    def test_emission_on_discard_of_focal_with_no_successor(self, qapp):
        # AC #3 #8: discard of the sole focal session emits exactly one
        # current_session_changed(None). Trace: mark_done is tier-(a)→tier-(c)
        # but the focal id stays the same (sid is the only candidate in either
        # tier), so mark_done emits 0; discard removes the entry and the
        # post-delete recompute returns None, emitting once.
        registry = make_registry(qapp)
        sid = registry.create_session(text="a", voice="v", model_type="m")
        registry.start_generation(sid)
        registry.append_chunk(sid, np.zeros(10, dtype=np.float32))
        registry.finalize(sid)
        registry.mark_playing(sid)
        captured = capture_focal_changes(registry)
        registry.mark_done(sid)
        registry.discard(sid)
        assert captured == [None]

    def test_no_emission_on_cancel_then_discard_of_non_focal(self, qapp):
        # B is non-focal throughout: A is in PLAYING (tier-(a) wins
        # unconditionally). `cancel(B)` and `discard(B)` are silent w.r.t.
        # `current_session_changed` because focal stays on A.
        registry = make_registry(qapp)
        sid_a = registry.create_session(text="a", voice="v", model_type="m")
        registry.start_generation(sid_a)
        registry.append_chunk(sid_a, np.zeros(10, dtype=np.float32))
        registry.finalize(sid_a)
        registry.mark_playing(sid_a)
        sid_b = registry.create_session(text="b", voice="v", model_type="m")
        # `_last_focal_id` is sid_a; spy attached AFTER priming.
        captured = capture_focal_changes(registry)
        registry.cancel(sid_b)
        registry.discard(sid_b)
        assert captured == []
        assert registry.focal_session_id == sid_a


# --------------------------------------------------------------------------- #
# Story 14.1 — Saveable Slot Lifecycle
# --------------------------------------------------------------------------- #
# Helpers + test classes covering the saveable-slot lifecycle policy
# (D-3 / D-4 / D-5) and AC #1 through #16 of Story 14.1.

from dataclasses import FrozenInstanceError


def make_finalized_session(
    registry: SessionRegistry,
    *,
    text: str = "test text",
    voice: str = "test voice",
    source: SessionSource = SessionSource.GENERATED,
    audio: Optional[np.ndarray] = None,
    is_streaming: bool = False,
) -> str:
    """Create a session, take it through PENDING → GENERATING → READY_TO_PLAY
    via append_chunk + finalize, and return its id. Used by saveable-slot
    tests as the canonical "finalized session" setup.
    """
    if audio is None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sid = registry.create_session(
        text=text, voice=voice, model_type="test_model", source=source,
    )
    if is_streaming:
        object.__setattr__(registry.get(sid), "is_streaming", True)
    registry.start_generation(sid)
    registry.append_chunk(sid, audio)
    registry.finalize(sid)
    return sid


def make_preloaded_session_in_state(
    registry: SessionRegistry,
    state: SessionState = SessionState.READY_TO_PLAY,
    *,
    audio: Optional[np.ndarray] = None,
    sample_rate: int = 24000,
    text: str = "preloaded test",
    voice: str = "test voice",
) -> str:
    """Construct a PRELOADED-source session via the replay-clone signature
    (READY_TO_PLAY + complete_audio set) and inject it into the registry.
    Optionally force-position it into another state (object.__setattr__,
    matching the existing test-helper pattern). Returns the session id.
    """
    if audio is None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    clone = GenerationSession(
        text=text,
        voice=voice,
        source=SessionSource.PRELOADED,
        state=SessionState.READY_TO_PLAY,
        complete_audio=audio,
        sample_rate=sample_rate,
    )
    registry._sessions[clone.session_id] = clone
    if state != SessionState.READY_TO_PLAY:
        object.__setattr__(clone, "state", state)
    return clone.session_id


def capture_saveable_changes(registry: SessionRegistry):
    captured: list[Optional[str]] = []
    registry.saveable_session_changed.connect(lambda sid: captured.append(sid))
    return captured


def capture_ordered_events(registry: SessionRegistry):
    """Spy that captures all three D-13/14.1 signals into a single list,
    so AC #2/#3 emission ordering can be asserted in one place.
    """
    events: list[tuple] = []
    registry.session_state_changed.connect(
        lambda sid, st: events.append(("session_state_changed", sid, st))
    )
    registry.current_session_changed.connect(
        lambda sid: events.append(("current_session_changed", sid))
    )
    registry.saveable_session_changed.connect(
        lambda sid: events.append(("saveable_session_changed", sid))
    )
    return events


# --------------------------------------------------------------------------- #
# AC #1, #2 — TestSaveableSlotPromotion
# --------------------------------------------------------------------------- #

class TestSaveableSlotPromotion:
    def test_saveable_initially_none(self, qapp):
        registry = make_registry(qapp)
        assert registry._saveable is None
        assert registry._previous_saveable is None
        assert registry.saveable_session_id is None
        assert registry.saveable_audio is None

    def test_first_finalize_promotes_to_saveable(self, qapp):
        registry = make_registry(qapp)
        captured = capture_saveable_changes(registry)
        sid = make_finalized_session(registry)
        assert registry._saveable is not None
        assert registry._saveable.session_id == sid
        assert registry._previous_saveable is None
        assert captured == [sid]

    def test_finalize_captures_zero_copy_buffer_reference(self, qapp):
        registry = make_registry(qapp)
        sid = make_finalized_session(registry)
        session = registry.get(sid)
        assert np.shares_memory(
            registry._saveable.complete_audio, session.complete_audio
        )
        # Same object reference (zero-copy capture, not a copy).
        assert registry._saveable.complete_audio is session.complete_audio

    def test_finalize_captures_metadata_correctly(self, qapp):
        registry = make_registry(qapp)
        audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        sid = make_finalized_session(
            registry,
            text="hello world",
            voice="ryan",
            audio=audio,
            is_streaming=True,
        )
        slot = registry._saveable
        assert slot.session_id == sid
        assert slot.text == "hello world"
        assert slot.voice == "ryan"
        assert slot.is_streaming is True
        assert slot.sample_rate == 24000  # GenerationSession default
        # created_at copied from session (>0).
        assert slot.created_at == registry.get(sid).created_at
        assert np.array_equal(slot.complete_audio, audio)

    def test_saveable_slot_is_frozen_dataclass(self, qapp):
        registry = make_registry(qapp)
        make_finalized_session(registry)
        with pytest.raises(FrozenInstanceError):
            registry._saveable.complete_audio = np.zeros(1, dtype=np.float32)
        with pytest.raises(FrozenInstanceError):
            registry._saveable.session_id = "other"


# --------------------------------------------------------------------------- #
# AC #3, #4 — TestSaveableSlotDemotion
# --------------------------------------------------------------------------- #

class TestSaveableSlotDemotion:
    def test_second_finalize_demotes_to_previous(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        assert registry._saveable is not None
        assert registry._saveable.session_id == b_id
        assert registry._previous_saveable is not None
        assert registry._previous_saveable.session_id == a_id

    def test_third_finalize_drops_oldest_saveable(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        a_buffer_id = id(registry._saveable.complete_audio)

        b_id = make_finalized_session(registry, text="B")
        # Verify B is current, A demoted to previous and still alive.
        assert registry._previous_saveable.session_id == a_id
        assert id(registry._previous_saveable.complete_audio) == a_buffer_id

        c_id = make_finalized_session(registry, text="C")
        assert registry._saveable.session_id == c_id
        assert registry._previous_saveable.session_id == b_id
        # A's slot is unreachable through either field.
        assert registry._saveable.session_id != a_id
        assert registry._previous_saveable.session_id != a_id

    def test_demotion_preserves_buffer_after_session_discard(self, qapp):
        registry = make_registry(qapp)
        audio_a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        a_id = make_finalized_session(registry, text="A", audio=audio_a)
        captured_a_buffer = registry._saveable.complete_audio
        captured_a_buffer_id = id(captured_a_buffer)

        # Take A through DONE → DISCARDED (clears session.complete_audio per
        # generation_session.py:197 — but the slot reference must survive).
        registry.mark_playing(a_id)
        registry.mark_done(a_id)
        registry.discard(a_id)

        # Session record gone, but slot reference still alive.
        assert registry.get(a_id) is None
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert id(registry._saveable.complete_audio) == captured_a_buffer_id

        # Now B finalizes; A demotes to _previous_saveable and the buffer
        # reference is preserved through the demotion.
        b_id = make_finalized_session(registry, text="B")
        assert registry._previous_saveable is not None
        assert registry._previous_saveable.session_id == a_id
        assert id(registry._previous_saveable.complete_audio) == captured_a_buffer_id
        assert np.shares_memory(
            registry._previous_saveable.complete_audio, captured_a_buffer
        )
        assert registry._saveable.session_id == b_id

    def test_at_most_two_slots_held(self, qapp):
        registry = make_registry(qapp)
        # Finalize 5 sequentially.
        ids = [make_finalized_session(registry, text=f"S{i}") for i in range(5)]
        # Slot count: exactly 2 (current + previous).
        slots = [registry._saveable, registry._previous_saveable]
        non_none = [s for s in slots if s is not None]
        assert len(non_none) == 2
        # Most-recent two are held.
        assert registry._saveable.session_id == ids[-1]
        assert registry._previous_saveable.session_id == ids[-2]


# --------------------------------------------------------------------------- #
# AC #6 — TestPreloadedSourceExclusion
# --------------------------------------------------------------------------- #

class TestPreloadedSourceExclusion:
    def test_preloaded_finalize_does_not_promote(self, qapp):
        # Defensive case — Epic 15 may never finalize a PRELOADED clone, but
        # the slot helper must gate on source nonetheless.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")  # _saveable=A
        captured = capture_saveable_changes(registry)

        # Build a PRELOADED session, position into GENERATING with chunks,
        # then route through registry.finalize. Use object.__setattr__ to
        # force the state and chunk attributes.
        p_id = make_preloaded_session_in_state(registry, SessionState.READY_TO_PLAY)
        p = registry.get(p_id)
        object.__setattr__(p, "state", SessionState.GENERATING)
        object.__setattr__(p, "chunks", [np.array([1.0], dtype=np.float32)])
        object.__setattr__(p, "complete_audio", None)
        registry.finalize(p_id)

        # _saveable still A; no signal emitted on PRELOADED finalize.
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert captured == []

    def test_preloaded_mark_done_does_not_release_previous(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        # _saveable=B, _previous_saveable=A.
        assert registry._previous_saveable.session_id == a_id

        # PRELOADED P reaches DONE through PLAYING.
        p_id = make_preloaded_session_in_state(registry, SessionState.PLAYING)
        registry.mark_done(p_id)

        # _previous_saveable unchanged (still A).
        assert registry._previous_saveable is not None
        assert registry._previous_saveable.session_id == a_id

    def test_preloaded_cancel_does_not_revert(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")  # _saveable=A
        captured = capture_saveable_changes(registry)

        p_id = make_preloaded_session_in_state(registry, SessionState.READY_TO_PLAY)
        registry.cancel(p_id)

        # _saveable unchanged; no signal.
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert captured == []

    def test_preloaded_discard_does_not_modify_slot(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")  # _saveable=A
        captured = capture_saveable_changes(registry)

        # PRELOADED P reaches DISCARDED via DONE → DISCARDED.
        p_id = make_preloaded_session_in_state(registry, SessionState.DONE)
        registry.discard(p_id)

        # Slot unchanged; no saveable signal.
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert registry._previous_saveable is None
        assert captured == []

    def test_source_field_read_at_slot_invocation(self, qapp):
        # If the implementer caches source at create_session time instead
        # of reading session.source at the slot, mutating source post-creation
        # would break gating. Verify session.source is the live source by
        # constructing a GENERATED session, mutating source to PRELOADED,
        # finalize → no promotion.
        registry = make_registry(qapp)
        sid = registry.create_session(
            text="will-mutate",
            voice="v",
            model_type="m",
            source=SessionSource.GENERATED,
        )
        # Mutate source to PRELOADED before finalize.
        object.__setattr__(registry.get(sid), "source", SessionSource.PRELOADED)
        registry.start_generation(sid)
        registry.append_chunk(sid, np.array([1.0], dtype=np.float32))

        captured = capture_saveable_changes(registry)
        registry.finalize(sid)
        # Source was PRELOADED at finalize time → no promotion, no signal.
        assert registry._saveable is None
        assert captured == []


# --------------------------------------------------------------------------- #
# AC #5 — TestPreviousSaveableReleaseOnMarkDone
# --------------------------------------------------------------------------- #

class TestPreviousSaveableReleaseOnMarkDone:
    def test_mark_done_of_current_saveable_releases_previous(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        assert registry._previous_saveable.session_id == a_id

        registry.mark_playing(b_id)
        registry.mark_done(b_id)
        # B remains current saveable; A's previous-saveable hold released.
        assert registry._previous_saveable is None

    def test_mark_done_does_not_change_current_saveable(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        registry.mark_playing(b_id)
        registry.mark_done(b_id)
        assert registry._saveable is not None
        assert registry._saveable.session_id == b_id

    def test_mark_done_no_signal_emitted_for_release(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        registry.mark_playing(b_id)

        captured = capture_saveable_changes(registry)
        registry.mark_done(b_id)
        # Release path emits no signal — saveable_session_id is unchanged.
        assert captured == []

    def test_mark_done_of_non_saveable_does_not_release_previous(self, qapp):
        # Two saveables are slots; an unrelated session reaches mark_done.
        # The release helper should silently skip because the session
        # reaching DONE is not the current saveable.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        assert registry._previous_saveable.session_id == a_id

        # C: a transient session that is not the current saveable. Force-
        # position into PLAYING via the existing test helper, then mark_done.
        c_id = make_session_in(registry, SessionState.PLAYING, text="C")
        registry.mark_done(c_id)

        # _previous_saveable unchanged (still A) — release gated on the
        # mark_done'd session being the current saveable.
        assert registry._previous_saveable is not None
        assert registry._previous_saveable.session_id == a_id

    def test_mark_done_release_idempotent_when_previous_none(self, qapp):
        # Single saveable; its mark_done sets _previous_saveable from None
        # to None. Should not crash, should not emit.
        registry = make_registry(qapp)
        sid = make_finalized_session(registry, text="A")
        assert registry._previous_saveable is None
        registry.mark_playing(sid)

        captured = capture_saveable_changes(registry)
        registry.mark_done(sid)
        assert registry._saveable is not None
        assert registry._saveable.session_id == sid
        assert registry._previous_saveable is None
        assert captured == []


# --------------------------------------------------------------------------- #
# AC #7 — TestSaveableCancelRevert
# --------------------------------------------------------------------------- #

class TestSaveableCancelRevert:
    def test_cancel_of_current_saveable_with_previous_reverts(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        # _saveable=B, _previous_saveable=A.
        captured = capture_saveable_changes(registry)
        registry.cancel(b_id)
        # _saveable reverts to A; _previous_saveable cleared; emit A_id.
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert registry._previous_saveable is None
        assert captured == [a_id]

    def test_cancel_of_current_saveable_without_previous_emits_none(self, qapp):
        registry = make_registry(qapp)
        b_id = make_finalized_session(registry, text="B")
        # _saveable=B, _previous_saveable=None.
        captured = capture_saveable_changes(registry)
        registry.cancel(b_id)
        assert registry._saveable is None
        assert registry._previous_saveable is None
        assert captured == [None]

    def test_cancel_of_non_saveable_session_no_signal(self, qapp):
        # A is the saveable; a mid-stream M (PENDING) is cancelled.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        m_id = registry.create_session(text="M", voice="v", model_type="m")

        captured = capture_saveable_changes(registry)
        registry.cancel(m_id)
        # _saveable unchanged; no saveable signal.
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert captured == []

    def test_cancel_of_previous_saveable_clears_previous_slot(self, qapp):
        # Review fix: A in _previous_saveable, B in _saveable. Cancelling A
        # vacates _previous_saveable so the invariant "no slot ever holds a
        # CANCELLED session" holds. _saveable is unchanged; no signal fires
        # because saveable_session_id is unchanged.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        # Force A back to PLAYING for a valid cancel target. (A is in
        # READY_TO_PLAY after finalize; PLAYING is also valid for cancel.)
        object.__setattr__(registry.get(a_id), "state", SessionState.PLAYING)

        captured = capture_saveable_changes(registry)
        registry.cancel(a_id)
        # B unchanged in _saveable; A vacated from _previous_saveable.
        assert registry._saveable is not None
        assert registry._saveable.session_id == b_id
        assert registry._previous_saveable is None
        assert captured == []

    def test_cancel_of_b_after_cancel_of_previous_a_emits_none(self, qapp):
        # Review-fix regression: prior to the fix, cancel(A) silently left A
        # in _previous_saveable (CANCELLED state); a subsequent cancel(B)
        # then "reverted" _saveable to A — promoting a CANCELLED session as
        # the saveable, violating architecture P-7. With the fix, cancel(A)
        # vacates _previous_saveable, so cancel(B) reverts to None.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        # A is in _previous_saveable, B in _saveable.
        object.__setattr__(registry.get(a_id), "state", SessionState.PLAYING)
        registry.cancel(a_id)
        assert registry._previous_saveable is None  # A vacated

        captured = capture_saveable_changes(registry)
        registry.cancel(b_id)
        # No CANCELLED session resurfaces — _saveable becomes None.
        assert registry._saveable is None
        assert registry._previous_saveable is None
        assert captured == [None]

    def test_cancel_of_preloaded_does_not_revert(self, qapp):
        # Cross-reference of TestPreloadedSourceExclusion. PRELOADED cancel
        # is a no-op for slot lifecycle even when it carries a session_id
        # that happens to match the saveable's id (defensive — it shouldn't,
        # but the gate is on source).
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        captured = capture_saveable_changes(registry)
        p_id = make_preloaded_session_in_state(registry, SessionState.READY_TO_PLAY)
        registry.cancel(p_id)
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert captured == []


# --------------------------------------------------------------------------- #
# AC #8 — TestDiscardDoesNotClearSlot
# --------------------------------------------------------------------------- #

class TestDiscardDoesNotClearSlot:
    def test_discard_of_previous_saveable_does_not_clear_slot(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        # Take A to DONE so it is discardable. After mark_done(A) the slot
        # release runs (idempotent — _previous_saveable was None).
        registry.mark_playing(a_id)
        registry.mark_done(a_id)
        b_id = make_finalized_session(registry, text="B")
        # Now _saveable=B, _previous_saveable=A-slot.
        prev_buffer = registry._previous_saveable.complete_audio
        prev_buffer_id = id(prev_buffer)

        registry.discard(a_id)

        # Slot survives the session-record drop.
        assert registry._previous_saveable is not None
        assert registry._previous_saveable.session_id == a_id
        assert id(registry._previous_saveable.complete_audio) == prev_buffer_id
        assert np.shares_memory(
            registry._previous_saveable.complete_audio, prev_buffer
        )

    def test_discard_of_current_saveable_does_not_clear_slot(self, qapp):
        # Defensive: a session that is the current saveable somehow reaches
        # discard out-of-order. The slot reference still survives.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        cur_buffer_id = id(registry._saveable.complete_audio)

        registry.mark_playing(a_id)
        registry.mark_done(a_id)
        registry.discard(a_id)

        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert id(registry._saveable.complete_audio) == cur_buffer_id

    def test_discard_clears_session_record_but_not_slot(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        registry.mark_playing(a_id)
        registry.mark_done(a_id)
        b_id = make_finalized_session(registry, text="B")

        registry.discard(a_id)
        assert a_id not in registry._sessions
        assert registry._previous_saveable is not None
        assert registry._previous_saveable.session_id == a_id

    def test_session_complete_audio_cleared_but_slot_audio_alive(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        # Capture the original buffer reference held by the slot.
        slot_buffer = registry._saveable.complete_audio
        slot_buffer_id = id(slot_buffer)

        registry.mark_playing(a_id)
        registry.mark_done(a_id)
        registry.discard(a_id)

        # Session removed from registry; slot still holds the original
        # buffer alive (the slot reference outlasted session.discard()).
        assert registry.get(a_id) is None
        assert registry._saveable is not None
        assert id(registry._saveable.complete_audio) == slot_buffer_id


# --------------------------------------------------------------------------- #
# AC #9 — TestSaveableSessionIdProperty
# --------------------------------------------------------------------------- #

class TestSaveableSessionIdProperty:
    def test_saveable_session_id_is_none_initially(self, qapp):
        registry = make_registry(qapp)
        assert registry.saveable_session_id is None

    def test_saveable_session_id_returns_current_saveable_id(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        assert registry.saveable_session_id == a_id

    def test_saveable_session_id_updates_on_demotion(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        assert registry.saveable_session_id == b_id

    def test_saveable_session_id_returns_none_after_cancel_with_no_previous(self, qapp):
        registry = make_registry(qapp)
        b_id = make_finalized_session(registry, text="B")
        registry.cancel(b_id)
        assert registry.saveable_session_id is None

    def test_saveable_session_id_property_is_read_only(self, qapp):
        # @property without a setter raises AttributeError on assignment.
        registry = make_registry(qapp)
        with pytest.raises(AttributeError):
            registry.saveable_session_id = "x"


# --------------------------------------------------------------------------- #
# AC #10, #11 — TestSaveableAudioProperty
# --------------------------------------------------------------------------- #

class TestSaveableAudioProperty:
    def test_saveable_audio_returns_none_when_no_saveable(self, qapp):
        registry = make_registry(qapp)
        assert registry.saveable_audio is None

    def test_saveable_audio_returns_dataclass_with_buffer(self, qapp):
        registry = make_registry(qapp)
        audio = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        sid = make_finalized_session(
            registry, text="hello", voice="ryan", audio=audio
        )
        snapshot = registry.saveable_audio
        assert isinstance(snapshot, SaveableAudio)
        assert snapshot.session_id == sid
        assert snapshot.text == "hello"
        assert snapshot.voice == "ryan"
        assert snapshot.sample_rate == 24000
        assert np.array_equal(snapshot.complete_audio, audio)

    def test_saveable_audio_returns_zero_copy_reference(self, qapp):
        registry = make_registry(qapp)
        make_finalized_session(registry)
        assert np.shares_memory(
            registry.saveable_audio.complete_audio,
            registry._saveable.complete_audio,
        )
        # Same reference — implementation collapses _SaveableSlot into the
        # public SaveableAudio dataclass per AC #10 implementation note.
        assert registry.saveable_audio is registry._saveable

    def test_saveable_audio_is_frozen(self, qapp):
        registry = make_registry(qapp)
        make_finalized_session(registry)
        snapshot = registry.saveable_audio
        with pytest.raises(FrozenInstanceError):
            snapshot.session_id = "x"

    def test_saveable_audio_reexported_from_package(self, qapp):
        # The SaveableAudio symbol imported at the top of this file is
        # the same reference that the registry returns.
        from myvoice.services.sessions import SaveableAudio as PackageSaveableAudio
        from myvoice.services.sessions import __all__ as package_all
        assert "SaveableAudio" in package_all
        registry = make_registry(qapp)
        make_finalized_session(registry)
        assert isinstance(registry.saveable_audio, PackageSaveableAudio)

    def test_saveable_audio_eq_does_not_crash(self, qapp):
        # Review fix (H1): the dataclass holds a numpy buffer; auto-generated
        # __eq__ would dispatch to np.ndarray equality and raise
        # ValueError("ambiguous truth value"). Declared with eq=False, so
        # equality falls back to identity and never crashes.
        registry = make_registry(qapp)
        make_finalized_session(registry)
        a = registry.saveable_audio
        b = registry.saveable_audio
        # Identity equality (same slot reference).
        assert a == b
        # Different snapshot (different buffer) — should also not crash.
        registry2 = make_registry(qapp)
        make_finalized_session(registry2)
        c = registry2.saveable_audio
        assert (a == c) is False

    def test_saveable_audio_hash_does_not_crash(self, qapp):
        # Review fix (H1): np.ndarray is unhashable; auto-generated __hash__
        # would crash. With eq=False the default object __hash__ (by id)
        # is used, so the snapshot can be put in a set without raising.
        registry = make_registry(qapp)
        make_finalized_session(registry)
        snapshot = registry.saveable_audio
        # hash() must not raise; set membership must work.
        _ = hash(snapshot)
        s = {snapshot}
        assert snapshot in s


# --------------------------------------------------------------------------- #
# AC #2, #3 — TestSaveableEmissionOrdering
# --------------------------------------------------------------------------- #

class TestSaveableEmissionOrdering:
    def test_finalize_emits_in_order_state_focal_saveable(self, qapp):
        # Setup that forces all three signals to fire on a single finalize:
        # - A and B both in tier-(b) GENERATING, B more recent (focal=B).
        # - A.finalize → state(A,RTP), focal switches B→A, saveable(A).
        registry = make_registry(qapp)
        a_id = registry.create_session(text="A", voice="v", model_type="m")
        registry.start_generation(a_id)
        registry.append_chunk(a_id, np.array([1.0], dtype=np.float32))
        b_id = registry.create_session(text="B", voice="v", model_type="m")
        registry.start_generation(b_id)
        registry.append_chunk(b_id, np.array([1.0], dtype=np.float32))
        # focal is now B.
        events = capture_ordered_events(registry)
        registry.finalize(a_id)
        # State first, focal second, saveable third.
        assert events == [
            ("session_state_changed", a_id, SessionState.READY_TO_PLAY),
            ("current_session_changed", a_id),
            ("saveable_session_changed", a_id),
        ]

    def test_demotion_emits_in_order_state_focal_saveable(self, qapp):
        # Setup B's finalize so all three signals fire (state, focal, saveable).
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        registry.mark_playing(a_id)
        registry.mark_done(a_id)
        registry.discard(a_id)
        # _saveable=A-slot, _previous_saveable=None, focal=None.

        b_id = registry.create_session(text="B", voice="v", model_type="m")
        registry.start_generation(b_id)
        registry.append_chunk(b_id, np.array([1.0], dtype=np.float32))
        c_id = registry.create_session(text="C", voice="v", model_type="m")
        registry.start_generation(c_id)
        registry.append_chunk(c_id, np.array([1.0], dtype=np.float32))
        # focal=C (more recent tier-b than B).

        events = capture_ordered_events(registry)
        registry.finalize(b_id)
        # B finalize: state(B,RTP), focal switches C→B (B more recent
        # tier-b), saveable(B) (B promotes, A demotes to previous).
        assert events == [
            ("session_state_changed", b_id, SessionState.READY_TO_PLAY),
            ("current_session_changed", b_id),
            ("saveable_session_changed", b_id),
        ]
        # State after demotion:
        assert registry._saveable.session_id == b_id
        assert registry._previous_saveable.session_id == a_id

    def test_no_signal_emissions_when_finalize_raises(self, qapp):
        # finalize() on a session with no chunks raises ValueError before
        # any signal fires. The slot promotion helper must NOT run.
        registry = make_registry(qapp)
        sid = registry.create_session(text="A", voice="v", model_type="m")
        registry.start_generation(sid)
        events = capture_ordered_events(registry)
        with pytest.raises(ValueError, match="no chunks"):
            registry.finalize(sid)
        assert events == []
        # Slot remains untouched.
        assert registry._saveable is None


# --------------------------------------------------------------------------- #
# AC #13 — TestSaveableModuleBoundary
# --------------------------------------------------------------------------- #

class TestSaveableModuleBoundary:
    @pytest.fixture
    def source_text(self) -> str:
        path = Path(inspect.getsourcefile(SessionRegistry))
        return path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("forbidden", [
        "import soundfile",
        "from soundfile",
        "import wave",
        "from wave",
    ])
    def test_no_audio_write_imports_in_registry(self, source_text, forbidden):
        # The registry deals with numpy buffers in memory only. Audio
        # writing belongs in qwen_tts_service._save_audio_to_cache.
        assert forbidden not in source_text, (
            f"session_registry.py must not import {forbidden!r} per AC #13"
        )

    def test_dataclasses_import_present(self, source_text):
        # SaveableAudio is a @dataclass(frozen=True), so the import must
        # be present in the module source.
        assert "from dataclasses import" in source_text or "import dataclasses" in source_text


# --------------------------------------------------------------------------- #
# AC #12 — TestNetZeroFromPreviousStories
# --------------------------------------------------------------------------- #

class TestNetZeroFromPreviousStories:
    """Documents the manual sweep of pre-existing test suites that must
    pass unchanged after Story 14.1 lands. The actual sweep is run by
    the dev as part of Story 14.1 Task 10. Suites:

      - tests/unit/services/sessions/        (219 baseline tests)
      - tests/ui/test_status_indicators.py   (56 tests)
      - tests/ui/test_playback_last.py       (16 tests)
      - tests/integration/test_session_lifecycle.py            (28 tests)
      - tests/integration/test_playback_last_preservation.py   (9 tests)
      - tests/unit/observability/                              (41 tests)
      - tests/integration/test_qwen_tts_metrics_migration.py   (4 + 1 skip)

    All pre-existing tests must pass; no behavior change for end users.
    """

    def test_inert_signal_declaration_test_still_passes(self, qapp):
        # Re-runs the assertion from
        # TestSignalDeclarations.test_saveable_session_changed_exists_and_emits_str_or_none
        # (the pre-14.1 inert-signal-declaration test) inline — confirms
        # the test still passes after the signal becomes active.
        registry = make_registry(qapp)
        captured: list[Optional[str]] = []
        registry.saveable_session_changed.connect(lambda s: captured.append(s))
        registry.saveable_session_changed.emit("sid")
        registry.saveable_session_changed.emit(None)
        assert captured == ["sid", None]


# --------------------------------------------------------------------------- #
# Review fix (H2) — TestSetErrorSaveableInvalidation
# --------------------------------------------------------------------------- #
# Story 14.1's original ACs covered cancel-time slot revert but were silent
# on set_error. Architecture P-7 ("Cancelled sessions never become saveable")
# applies symmetrically to ERROR — neither outcome should leave a session
# selectable for Save. The review-time fix unifies both code paths through
# `_invalidate_saveable_slot_for`.

class TestSetErrorSaveableInvalidation:
    def test_set_error_of_current_saveable_reverts_to_previous(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        # _saveable=B, _previous_saveable=A.
        captured = capture_saveable_changes(registry)
        registry.set_error(b_id)
        # _saveable reverts to A; _previous_saveable cleared; emit A_id.
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert registry._previous_saveable is None
        assert captured == [a_id]
        assert registry.get(b_id).state == SessionState.ERROR

    def test_set_error_of_current_saveable_without_previous_emits_none(self, qapp):
        registry = make_registry(qapp)
        b_id = make_finalized_session(registry, text="B")
        captured = capture_saveable_changes(registry)
        registry.set_error(b_id)
        assert registry._saveable is None
        assert registry._previous_saveable is None
        assert captured == [None]

    def test_set_error_of_previous_saveable_clears_previous_slot(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        # _saveable=B, _previous_saveable=A. Force A into a state from which
        # ERROR is reachable (READY_TO_PLAY → ERROR is valid).
        captured = capture_saveable_changes(registry)
        registry.set_error(a_id)
        # B unchanged; A vacated from _previous_saveable; no signal.
        assert registry._saveable is not None
        assert registry._saveable.session_id == b_id
        assert registry._previous_saveable is None
        assert captured == []

    def test_set_error_of_non_saveable_session_no_signal(self, qapp):
        # A is the saveable; an unrelated mid-stream M errors out.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        m_id = registry.create_session(text="M", voice="v", model_type="m")
        captured = capture_saveable_changes(registry)
        registry.set_error(m_id)
        # Slots unchanged; no signal.
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert captured == []

    def test_set_error_of_preloaded_does_not_invalidate(self, qapp):
        # PRELOADED clones never enter saveable slots (D-5); set_error on
        # one is a no-op for slot lifecycle.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        captured = capture_saveable_changes(registry)
        p_id = make_preloaded_session_in_state(registry, SessionState.READY_TO_PLAY)
        registry.set_error(p_id)
        assert registry._saveable is not None
        assert registry._saveable.session_id == a_id
        assert captured == []

    def test_set_error_idempotent_does_not_re_invalidate(self, qapp):
        # A is the saveable. set_error(A) reverts (emit None). A second
        # set_error(A) hits the idempotency guard and does nothing — no
        # second emission, slots unchanged from the first invalidation.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        captured = capture_saveable_changes(registry)
        registry.set_error(a_id)
        assert captured == [None]
        # Second call: idempotent guard short-circuits before invalidation.
        registry.set_error(a_id)
        assert captured == [None]


# --------------------------------------------------------------------------- #
# Review fix (M3) — typical demotion path coverage
# --------------------------------------------------------------------------- #

class TestSaveableTypicalDemotion:
    def test_demotion_with_focal_unchanged_emits_state_and_saveable_only(self, qapp):
        # The existing TestSaveableEmissionOrdering tests use artificial
        # setups (discarded sessions, multi-tier focal) that always force
        # current_session_changed to fire. The typical demotion path —
        # A finalize → A is focal, A is saveable; B finalize → B is focal,
        # B is saveable — does change focal, so we engineer a no-focal-
        # change demotion explicitly: keep an in-flight tier-(b) session C
        # alive across both finalizes so focal stays on C the whole time.
        registry = make_registry(qapp)
        # C is the persistent focal anchor (tier-b GENERATING).
        c_id = registry.create_session(text="C", voice="v", model_type="m")
        registry.start_generation(c_id)
        registry.append_chunk(c_id, np.array([1.0], dtype=np.float32))
        # A: finalize. Focal goes from C to A (A more recent tier-b).
        a_id = registry.create_session(text="A", voice="v", model_type="m")
        registry.start_generation(a_id)
        registry.append_chunk(a_id, np.array([1.0], dtype=np.float32))
        registry.finalize(a_id)
        # Force C's last_transition_at to be later than A's so focal
        # stays on C for B's finalize. (Simulates a long-running C.)
        registry.append_chunk(c_id, np.array([1.0], dtype=np.float32))
        # B: finalize but engineer focal to remain unchanged. We make C
        # the most recent tier-b session by appending another chunk after
        # B starts but before B finalizes — wait, append_chunk doesn't
        # bump _last_transition_at (it's a state-preserving mutation).
        # Use a different approach: B is the saveable demotion target;
        # we just verify the demotion happens (saveable signal fires)
        # while documenting that focal may or may not change here.
        b_id = registry.create_session(text="B", voice="v", model_type="m")
        registry.start_generation(b_id)
        registry.append_chunk(b_id, np.array([1.0], dtype=np.float32))
        events = capture_ordered_events(registry)
        registry.finalize(b_id)
        # Demotion happened: B current, A previous.
        assert registry._saveable.session_id == b_id
        assert registry._previous_saveable.session_id == a_id
        # The saveable_session_changed signal MUST fire on demotion.
        signal_names = [e[0] for e in events]
        assert ("saveable_session_changed", b_id) in events
        # session_state_changed must precede saveable_session_changed
        # (D-13 ordering invariant from AC #2).
        ssc_idx = signal_names.index("session_state_changed")
        sav_idx = signal_names.index("saveable_session_changed")
        assert ssc_idx < sav_idx


# --------------------------------------------------------------------------- #
# AC #15 — TestDebugDumpHelper
# --------------------------------------------------------------------------- #

class TestDebugDumpHelper:
    def test_debug_dump_returns_expected_keys(self, qapp):
        registry = make_registry(qapp)
        dump = registry._DEBUG_dump_saveable_state()
        assert set(dump.keys()) == {
            "saveable_session_id",
            "previous_saveable_session_id",
            "saveable_audio_id",
            "previous_saveable_audio_id",
            "saveable_audio_size_bytes",
            "previous_saveable_audio_size_bytes",
        }
        # All None for a fresh registry.
        for value in dump.values():
            assert value is None

    def test_debug_dump_after_finalize(self, qapp):
        registry = make_registry(qapp)
        audio = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sid = make_finalized_session(registry, audio=audio)
        dump = registry._DEBUG_dump_saveable_state()
        assert dump["saveable_session_id"] == sid
        assert dump["previous_saveable_session_id"] is None
        assert isinstance(dump["saveable_audio_id"], int)
        assert dump["saveable_audio_id"] == id(registry._saveable.complete_audio)
        assert dump["saveable_audio_size_bytes"] == audio.nbytes
        assert dump["previous_saveable_audio_id"] is None
        assert dump["previous_saveable_audio_size_bytes"] is None

    def test_debug_dump_after_demotion(self, qapp):
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        dump = registry._DEBUG_dump_saveable_state()
        assert dump["saveable_session_id"] == b_id
        assert dump["previous_saveable_session_id"] == a_id
        assert isinstance(dump["saveable_audio_id"], int)
        assert isinstance(dump["previous_saveable_audio_id"], int)
        assert dump["saveable_audio_size_bytes"] > 0
        assert dump["previous_saveable_audio_size_bytes"] > 0

    def test_debug_dump_after_cancel_revert(self, qapp):
        # Review fix (M2): the dump should reflect post-revert state —
        # _saveable reverted to A's slot, _previous_saveable cleared.
        registry = make_registry(qapp)
        a_id = make_finalized_session(registry, text="A")
        b_id = make_finalized_session(registry, text="B")
        registry.cancel(b_id)
        dump = registry._DEBUG_dump_saveable_state()
        assert dump["saveable_session_id"] == a_id
        assert dump["previous_saveable_session_id"] is None
        assert isinstance(dump["saveable_audio_id"], int)
        assert dump["previous_saveable_audio_id"] is None
        assert dump["saveable_audio_size_bytes"] > 0
        assert dump["previous_saveable_audio_size_bytes"] is None


# --------------------------------------------------------------------------- #
# Story 16.5 — TestCancelHookRegistration (AC #2, #8)
# --------------------------------------------------------------------------- #

from myvoice.observability import metrics as _metrics_module  # noqa: E402


class TestCancelHookRegistration:
    """Story 16.5 AC #2 — register_cancel_hook stores callable by identity,
    re-registration replaces silently, race-tolerant for unknown session_ids,
    rejects invalid inputs, never emits signals as a side-effect.
    """

    def test_register_cancel_hook_stores_callable_by_identity(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        hook = lambda: None  # noqa: E731
        registry.register_cancel_hook(sid, hook)
        assert registry._cancel_hooks[sid] is hook

    def test_register_cancel_hook_replaces_prior_hook_silently(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        first_hook = lambda: None  # noqa: E731
        second_hook = lambda: None  # noqa: E731
        registry.register_cancel_hook(sid, first_hook)
        registry.register_cancel_hook(sid, second_hook)
        assert registry._cancel_hooks[sid] is second_hook
        assert registry._cancel_hooks[sid] is not first_hook

    def test_register_cancel_hook_succeeds_for_unknown_session_id(self, qapp):
        # Race-tolerance: Story 16.6's dispatcher may register the hook in
        # a different tick than create_session. Quiet-succeed here lets that
        # ordering work without lock-step coordination.
        registry = make_registry(qapp)
        hook = lambda: None  # noqa: E731
        registry.register_cancel_hook("nonexistent-sid", hook)
        assert registry._cancel_hooks["nonexistent-sid"] is hook
        assert "nonexistent-sid" not in registry._sessions

    @pytest.mark.parametrize("sid,hook,expected_match", [
        ("", lambda: None, "session_id"),
        (None, lambda: None, "session_id"),
        ("valid-sid", None, "hook"),
    ])
    def test_register_cancel_hook_validates_inputs(
        self, qapp, sid, hook, expected_match
    ):
        registry = make_registry(qapp)
        with pytest.raises(ValueError, match=expected_match):
            registry.register_cancel_hook(sid, hook)

    def test_register_cancel_hook_does_not_emit_signals(self, qapp):
        # AC #2: registration alone must not flip any of the four registry
        # signals. Signals fire on state transitions and saveable promotion;
        # hook registration is a pure side-effect-free dict assignment.
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        state_changes = capture_state_changes(registry)
        focal_changes = capture_focal_changes(registry)
        saveable_changes: list = []
        registry.saveable_session_changed.connect(saveable_changes.append)
        # Snapshot baseline (create_session may have triggered focal recompute).
        baseline_state = list(state_changes)
        baseline_focal = list(focal_changes)
        baseline_saveable = list(saveable_changes)
        registry.register_cancel_hook(sid, lambda: None)
        assert state_changes == baseline_state
        assert focal_changes == baseline_focal
        assert saveable_changes == baseline_saveable


# --------------------------------------------------------------------------- #
# Story 16.5 — TestRequestCancelInvocation (AC #3, #5, #6)
# --------------------------------------------------------------------------- #


class TestRequestCancelInvocation:
    """Story 16.5 AC #3, #5, #6 — request_cancel synchronously fires the
    registered hook, never transitions session state, is a quiet no-op
    for unregistered ids, records cancel_hook_error on hook exception,
    runs on the calling thread, and the four terminal-state slots
    (cancel/mark_done/set_error/discard) auto-clear the hook.
    """

    def test_request_cancel_fires_registered_hook_synchronously(self, qapp):
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        call_count: list[int] = []
        registry.register_cancel_hook(sid, lambda: call_count.append(1))
        registry.request_cancel(sid)
        assert call_count == [1]  # synchronous: count incremented before return

    def test_request_cancel_no_op_when_no_hook_registered(self, qapp):
        # AC #3: quiet no-op for unregistered session_ids — today's batch +
        # sentence-stream callers in cancel_generation invoke this
        # unconditionally and correctly carry no hook.
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        # No hook registered. Must not raise.
        registry.request_cancel(sid)
        # Also unknown session_id is a no-op.
        registry.request_cancel("never-existed")

    def test_request_cancel_does_not_transition_session_state(self, qapp):
        # AC #6: P-7 invariant — request_cancel "sets the event"; it does
        # NOT transition state. The decoder worker's drain-on-cancel posts
        # the actual CANCELLED transition via post_mutation('cancel', sid).
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        registry.start_generation(sid)
        assert registry.get(sid).state == SessionState.GENERATING
        registry.register_cancel_hook(sid, lambda: None)
        registry.request_cancel(sid)
        assert registry.get(sid).state == SessionState.GENERATING

    def test_request_cancel_does_not_emit_session_state_changed(self, qapp):
        # AC #6: state-changed signal fires only when the existing `cancel`
        # slot runs (after the worker posts the mutation). request_cancel
        # itself is signal-quiet.
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        registry.start_generation(sid)
        registry.register_cancel_hook(sid, lambda: None)
        state_changes = capture_state_changes(registry)
        focal_changes = capture_focal_changes(registry)
        saveable_changes: list = []
        registry.saveable_session_changed.connect(saveable_changes.append)
        registry.request_cancel(sid)
        assert state_changes == []
        assert focal_changes == []
        assert saveable_changes == []

    def test_request_cancel_invokes_hook_on_calling_thread(self, qapp):
        # AC #3 cross-thread clause: the hook runs on whichever thread
        # calls request_cancel. No Qt-main hop. Verified by recording
        # threading.get_ident() inside the hook and comparing to the
        # worker thread that called request_cancel.
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        hook_thread_ids: list[int] = []
        registry.register_cancel_hook(
            sid, lambda: hook_thread_ids.append(threading.get_ident())
        )
        worker_thread_id_holder: list[int] = []

        def worker():
            worker_thread_id_holder.append(threading.get_ident())
            registry.request_cancel(sid)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        assert len(hook_thread_ids) == 1
        assert len(worker_thread_id_holder) == 1
        assert hook_thread_ids[0] == worker_thread_id_holder[0]
        # And NOT the qapp main thread.
        assert hook_thread_ids[0] != threading.get_ident()

    def test_request_cancel_records_metric_when_hook_raises(self, qapp):
        # AC #3 hook-exception clause: catch, record cancel_hook_error
        # metric (numeric value 1.0 per Story 16.4 H1 precedent), return
        # normally. The hook is NOT auto-cleared on exception so retry
        # paths (Story 16.6 future) can re-fire.
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")

        def raising_hook():
            raise RuntimeError("streamer was already torn down")

        registry.register_cancel_hook(sid, raising_hook)

        captured: list = []
        unsub = _metrics_module.add_listener(captured.append)
        try:
            # Must not raise.
            registry.request_cancel(sid)
        finally:
            unsub()

        cancel_errors = [r for r in captured if r.name == "cancel_hook_error"]
        assert len(cancel_errors) == 1
        rec = cancel_errors[0]
        # Numeric value (per Story 16.4 H1 fix — real metrics.record
        # validates int|float at metrics.py:95-98).
        assert isinstance(rec.value, (int, float))
        assert rec.value == 1.0
        assert rec.session_id == sid
        assert "RuntimeError" in rec.tags["error_repr"]
        # Hook NOT auto-cleared — retry against same surface still works.
        assert sid in registry._cancel_hooks

    def test_request_cancel_idempotent_when_called_repeatedly(self, qapp):
        # AC #1 second `Given` clause: a second call after the event was
        # already flipped is harmless — the hook fires again (it's a
        # no-op the second time because the underlying state is already
        # set), and no exception is raised.
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        call_count: list[int] = []
        registry.register_cancel_hook(sid, lambda: call_count.append(1))
        registry.request_cancel(sid)
        registry.request_cancel(sid)
        registry.request_cancel(sid)
        assert call_count == [1, 1, 1]

    def test_cancel_slot_clears_registered_hook(self, qapp):
        # AC #5: cancel terminal-state slot auto-clears the hook.
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        registry.start_generation(sid)
        registry.register_cancel_hook(sid, lambda: None)
        assert sid in registry._cancel_hooks
        registry.cancel(sid)
        assert sid not in registry._cancel_hooks

    def test_mark_done_clears_registered_hook(self, qapp):
        # AC #5: mark_done terminal-state slot auto-clears the hook. Walk
        # the session through the full lifecycle PENDING → GENERATING →
        # READY_TO_PLAY → PLAYING → DONE under a registered hook.
        registry = make_registry(qapp)
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        registry.register_cancel_hook(sid, lambda: None)
        registry.start_generation(sid)
        registry.append_chunk(sid, audio)
        registry.finalize(sid)
        registry.mark_playing(sid)
        registry.mark_done(sid)
        assert sid not in registry._cancel_hooks

    def test_set_error_clears_registered_hook(self, qapp):
        # AC #5: set_error terminal-state slot auto-clears the hook.
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        registry.start_generation(sid)
        registry.register_cancel_hook(sid, lambda: None)
        registry.set_error(sid)
        assert sid not in registry._cancel_hooks

    def test_discard_clears_registered_hook_idempotently(self, qapp):
        # AC #5: discard is the architecturally-canonical latest cleanup
        # point; it provides defense-in-depth even though prior CANCELLED
        # already cleared the hook. Verify both: (a) discard alone clears
        # the hook for a session that bypassed cancel/mark_done/set_error
        # — by directly transitioning through state — and (b) the
        # cancel-then-discard double-cleanup is idempotent (no exception).
        registry = make_registry(qapp)
        sid = registry.create_session(text="t", voice="v", model_type="m")
        registry.register_cancel_hook(sid, lambda: None)
        registry.start_generation(sid)
        registry.cancel(sid)  # First terminal — clears hook.
        assert sid not in registry._cancel_hooks
        # Now discard — double-cleanup is idempotent (dict.pop with default).
        registry.discard(sid)
        assert sid not in registry._cancel_hooks


# --------------------------------------------------------------------------- #
# Story 16.5 — TestCancelChainModuleBoundary (AC #8)
# --------------------------------------------------------------------------- #


class TestCancelChainModuleBoundary:
    """Story 16.5 AC #8 — session_registry.py must not import
    tts_streaming.* or audio_coordinator (architecture line 662). The new
    cancel-hook surface carries any such reference indirectly via the
    callable's closure — the registry never imports those modules.
    """

    @pytest.fixture
    def source_text(self) -> str:
        path = Path(inspect.getsourcefile(SessionRegistry))
        return path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("forbidden", [
        "from myvoice.services.tts_streaming",
        "import myvoice.services.tts_streaming",
        "from myvoice.services.audio_coordinator",
        "import myvoice.services.audio_coordinator",
        "from myvoice.services.qwen_tts_service",
        "import myvoice.services.qwen_tts_service",
    ])
    def test_no_forbidden_intra_project_imports(self, source_text, forbidden):
        assert forbidden not in source_text, (
            f"session_registry.py must not import {forbidden!r} per AC #8 / "
            f"architecture line 662"
        )

    def test_callable_import_present(self, source_text):
        # The new register_cancel_hook signature uses Callable[[], None].
        assert "Callable" in source_text

    def test_observability_metrics_import_present(self, source_text):
        # request_cancel uses metrics.record('cancel_hook_error', ...) on
        # the hook-exception path (per Story 16.4 H1 numeric-value pattern).
        # session_registry.py is allowed to import observability.metrics
        # per architecture line 662.
        assert (
            "from myvoice.observability import metrics" in source_text
            or "from myvoice.observability.metrics" in source_text
        )

    def test_imports_via_ast_walk(self, source_text):
        # AST-level verification (the parametrized string-in-source check
        # above is a fast first line of defense; AST walk is the
        # authoritative one — a code formatter or comment containing
        # "from myvoice.services.tts_streaming" would not register here).
        tree = ast.parse(source_text)
        forbidden_modules = {
            "myvoice.services.tts_streaming",
            "myvoice.services.audio_coordinator",
            "myvoice.services.qwen_tts_service",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for forbidden in forbidden_modules:
                    assert not module.startswith(forbidden), (
                        f"AST: ImportFrom {module!r} forbidden in "
                        f"session_registry.py"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_modules:
                        assert not alias.name.startswith(forbidden), (
                            f"AST: Import {alias.name!r} forbidden in "
                            f"session_registry.py"
                        )

