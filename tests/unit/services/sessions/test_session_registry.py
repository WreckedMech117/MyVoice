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
