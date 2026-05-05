"""
Tests for PlaybackQueue (Story 13.1).

Covers AC #1 through #8: module exports + boundary, public API surface,
FIFO ordering, depth-signal emission rules (post-mutation payload, no
spurious emissions on empty-queue dequeue/cancel), signal payload type
(P-4: primitive `int`), Qt-thread affinity (cross-thread mutation
rejected with `RuntimeError`), and the `cancel_current` edge cases.

The queue is signal-bearing but synchronous: every mutation emits its
`playback_queue_depth_changed(int)` signal inline on the calling thread,
so tests use a plain `list.append` spy (no `qtbot.waitSignal`). This
mirrors the convention from `tests/unit/services/sessions/test_session_registry.py`.
"""

import ast
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import pytest

# Tests in this module require PyQt6.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from myvoice.services.sessions import PlaybackQueue
from myvoice.services.sessions import playback_queue as queue_module


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


@pytest.fixture
def queue(qapp) -> PlaybackQueue:
    """Function-scoped fresh queue per test for isolation."""
    return PlaybackQueue()


def make_spy(signal) -> list:
    """Connect a list-based spy to ``signal`` and return the list.

    The spy receives every emitted depth integer in order. Callers can
    `history.clear()` between sub-scenarios. Reference-counting keeps the
    closure alive for the test's duration (mirrors
    `test_session_registry.py::capture_state_changes`).
    """
    history: list = []
    signal.connect(history.append)
    return history


# --------------------------------------------------------------------------- #
# AC #1 / #2 / #5 — TestPlaybackQueueAPI
# --------------------------------------------------------------------------- #

class TestPlaybackQueueAPI:
    """Module exports, public API surface, signal class declaration."""

    def test_playback_queue_is_qobject_subclass(self, qapp):
        assert issubclass(PlaybackQueue, QObject)

    def test_playback_queue_depth_changed_is_class_signal(self, queue):
        # Class-level attribute exists (PyQt6 binds `pyqtSignal` as a
        # descriptor; it is NOT an instance attribute).
        assert hasattr(PlaybackQueue, "playback_queue_depth_changed")
        # Bound signal on an instance has `.emit` and `.connect`.
        bound = queue.playback_queue_depth_changed
        assert hasattr(bound, "emit")
        assert hasattr(bound, "connect")

    def test_public_api_surface(self, queue):
        # Methods are callable, `depth` is a property (not a method).
        assert callable(queue.enqueue)
        assert callable(queue.dequeue)
        assert callable(queue.cancel_current)
        assert callable(queue.peek)
        # `depth` resolves to an `int` via the property descriptor on
        # the class — getting it on an instance returns the value, not
        # a method object.
        assert isinstance(type(queue).depth, property)

    def test_depth_property_returns_int(self, queue):
        assert isinstance(queue.depth, int)
        assert queue.depth == 0

    def test_repr_includes_depth(self, queue):
        # `__repr__` is a debug aid for grep-able log output during
        # Story 13.2's integration debugging (Task 1.11). Pin the format
        # so future log scrapes don't silently break.
        assert repr(queue) == "PlaybackQueue(depth=0)"
        queue.enqueue("A")
        queue.enqueue("B")
        assert repr(queue) == "PlaybackQueue(depth=2)"

    def test_init_requires_qapplication(self, qapp, monkeypatch):
        """Mirrors `test_session_registry.py::test_missing_qapplication_raises`.

        Cannot tear down a real QApplication mid-suite; monkeypatch
        `QApplication.instance` to return `None` for the duration of
        this one test. The `qapp` fixture is preserved for other tests
        because monkeypatch auto-rolls-back at function exit.
        """
        monkeypatch.setattr(
            "myvoice.services.sessions.playback_queue.QApplication.instance",
            lambda: None,
        )
        with pytest.raises(RuntimeError, match="QApplication"):
            PlaybackQueue()

    def test_init_rejects_non_main_thread(self, qapp):
        """Constructing the queue on a worker thread must raise.

        Pattern from `test_session_registry.py::test_worker_thread_construction_raises`.
        """
        captured: list[BaseException] = []

        def worker():
            try:
                PlaybackQueue()
            except BaseException as exc:
                captured.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        assert len(captured) == 1
        assert isinstance(captured[0], RuntimeError)
        assert "main thread" in str(captured[0]).lower()

    def test_init_rejects_worker_thread_even_with_main_thread_parent(self, qapp):
        """Regression: passing a main-thread `parent` from a worker thread
        must still raise. `super().__init__(parent)` adopts the parent's
        thread affinity, so a naive `app.thread() is not self.thread()`
        check would silently pass (`main is not main` → False). The
        guard must compare the *executing* thread
        (`QThread.currentThread()`) against `app.thread()`.
        """
        # Construct a main-thread QObject to use as parent.
        main_thread_parent = QObject()
        assert main_thread_parent.thread() is qapp.thread()

        captured: list[BaseException] = []

        def worker():
            try:
                PlaybackQueue(parent=main_thread_parent)
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


# --------------------------------------------------------------------------- #
# AC #3 — TestFifoOrdering
# --------------------------------------------------------------------------- #

class TestFifoOrdering:
    """FIFO semantics, mixed enqueue/dequeue, peek non-mutation."""

    def test_enqueue_three_then_dequeue_returns_in_order(self, queue):
        queue.enqueue("A")
        queue.enqueue("B")
        queue.enqueue("C")

        assert queue.depth == 3
        assert queue.peek() == "A"
        assert queue.dequeue() == "A"  # left/oldest
        assert queue.dequeue() == "B"  # middle
        assert queue.dequeue() == "C"  # right/newest
        assert queue.dequeue() is None  # drained
        assert queue.depth == 0

    def test_mixed_enqueue_dequeue_preserves_fifo(self, queue):
        queue.enqueue("X")
        assert queue.dequeue() == "X"
        queue.enqueue("Y")
        queue.enqueue("Z")
        # FIFO survives interleaving — second dequeue returns the older
        # of the two remaining items, which is "Y".
        assert queue.dequeue() == "Y"
        assert queue.peek() == "Z"
        assert queue.depth == 1

    def test_dequeue_on_empty_returns_none(self, queue):
        assert queue.dequeue() is None
        assert queue.depth == 0

    def test_peek_on_empty_returns_none(self, queue):
        assert queue.peek() is None

    def test_peek_does_not_mutate(self, queue):
        queue.enqueue("A")
        queue.enqueue("B")
        # Peek twice — depth must stay at 2, head stays "A".
        assert queue.peek() == "A"
        assert queue.peek() == "A"
        assert queue.depth == 2
        assert queue.dequeue() == "A"

    def test_internal_storage_is_deque(self, queue):
        """Pin the perf invariant from AC #3.

        Deliberate breach of the public-API line: `_deque` is
        implementation-private, but the AC requires that the type
        is `collections.deque` (NOT `list`) so `popleft()` is O(1)
        on the OFR-C hot path.
        """
        # `getattr` with the underscore name to make the breach
        # explicit; if the implementation renames the attribute, this
        # test will surface the rename and force an AC update.
        assert isinstance(queue._deque, deque)

    def test_enqueue_rejects_non_string(self, queue):
        """P-4 defends against object payloads on queue boundaries.

        Direct Python callers bypass Qt's `@pyqtSlot(str)` type
        marshalling, so the queue carries a runtime `isinstance(str)`
        guard. The deque must be unchanged after the failed enqueue.
        """
        with pytest.raises(TypeError, match="P-4"):
            queue.enqueue(42)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            queue.enqueue(None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            queue.enqueue(["not", "a", "string"])  # type: ignore[arg-type]
        # No partial state — the failed enqueues must NOT have stored
        # anything, and the depth signal must NOT have fired.
        assert queue.depth == 0


# --------------------------------------------------------------------------- #
# AC #4 / #5 — TestDepthSignalEmission
# --------------------------------------------------------------------------- #

class TestDepthSignalEmission:
    """Depth signal emits exactly once per state-changing operation,
    with the post-mutation depth as payload. Empty-queue no-ops emit
    nothing. Payload is a primitive `int` (P-4)."""

    def test_enqueue_emits_post_mutation_depth(self, queue):
        history = make_spy(queue.playback_queue_depth_changed)
        queue.enqueue("A")
        queue.enqueue("B")
        queue.enqueue("C")
        assert history == [1, 2, 3]

    def test_dequeue_emits_post_mutation_depth(self, queue):
        # Preload three (emissions occur but we discard them).
        queue.enqueue("A")
        queue.enqueue("B")
        queue.enqueue("C")
        history = make_spy(queue.playback_queue_depth_changed)
        assert queue.dequeue() == "A"
        assert queue.dequeue() == "B"
        assert queue.dequeue() == "C"
        assert history == [2, 1, 0]

    def test_dequeue_on_empty_does_not_emit(self, queue):
        history = make_spy(queue.playback_queue_depth_changed)
        assert queue.dequeue() is None
        # Mirrors `_recompute_focal_and_maybe_emit` "only emit on
        # change" rule — emitting `0 → 0` would be spurious.
        assert history == []

    def test_cancel_current_on_empty_does_not_emit(self, queue):
        history = make_spy(queue.playback_queue_depth_changed)
        queue.cancel_current()
        assert history == []

    def test_cancel_current_emits_post_mutation_depth(self, queue):
        queue.enqueue("A")
        queue.enqueue("B")
        history = make_spy(queue.playback_queue_depth_changed)
        queue.cancel_current()
        assert history == [1]

    def test_signal_payload_is_int_and_nonnegative(self, queue):
        history = make_spy(queue.playback_queue_depth_changed)
        # Mixed sequence covers enqueue, dequeue (with-pop), and
        # cancel_current (with-pop).
        queue.enqueue("A")
        queue.enqueue("B")
        queue.dequeue()
        queue.cancel_current()
        assert history  # non-empty sanity check
        for depth in history:
            assert isinstance(depth, int)
            assert depth >= 0

    def test_emit_history_with_mixed_operations(self, queue):
        """Single-pass regression test catching off-by-one or double-emit
        bugs in any single mutation method.

        Sequence: enqueue, enqueue, enqueue, dequeue, enqueue, dequeue,
        dequeue, cancel_current.
            depth after step 1 (enq A):           1
            depth after step 2 (enq B):           2
            depth after step 3 (enq C):           3
            depth after step 4 (deq → A):         2
            depth after step 5 (enq D):           3
            depth after step 6 (deq → B):         2
            depth after step 7 (deq → C):         1
            depth after step 8 (cancel → D popped): 0
        Final cancel is NOT a no-op: queue still has D when it runs.
        """
        history = make_spy(queue.playback_queue_depth_changed)
        queue.enqueue("A")
        queue.enqueue("B")
        queue.enqueue("C")
        queue.dequeue()
        queue.enqueue("D")
        queue.dequeue()
        queue.dequeue()
        queue.cancel_current()
        assert history == [1, 2, 3, 2, 3, 2, 1, 0]


# --------------------------------------------------------------------------- #
# AC #6 — TestQtThreadAffinity
# --------------------------------------------------------------------------- #

class TestQtThreadAffinity:
    """Cross-thread mutation must raise; main-thread mutation must
    succeed; read-only accessors are not guarded."""

    def test_main_thread_mutation_succeeds(self, queue):
        # All three mutation methods on the main thread — no exception.
        queue.enqueue("A")
        queue.enqueue("B")
        assert queue.dequeue() == "A"
        queue.cancel_current()
        assert queue.depth == 0

    def _capture_worker_exception(self, target_callable):
        captured: list[BaseException] = []

        def worker():
            try:
                target_callable()
            except BaseException as exc:
                captured.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        return captured

    def test_worker_thread_enqueue_raises(self, queue, qapp):
        captured = self._capture_worker_exception(lambda: queue.enqueue("X"))
        assert len(captured) == 1
        assert isinstance(captured[0], RuntimeError)
        msg = str(captured[0]).lower()
        assert "cross-thread" in msg or "main thread" in msg
        # Mutation must NOT have taken effect.
        assert queue.depth == 0

    def test_worker_thread_dequeue_raises(self, queue, qapp):
        # Preload one item so we can prove no pop occurred.
        queue.enqueue("A")
        captured = self._capture_worker_exception(lambda: queue.dequeue())
        assert len(captured) == 1
        assert isinstance(captured[0], RuntimeError)
        assert "cross-thread" in str(captured[0]).lower()
        # Item must still be there — pop must NOT have happened.
        assert queue.depth == 1
        assert queue.peek() == "A"

    def test_worker_thread_cancel_current_raises(self, queue, qapp):
        queue.enqueue("A")
        captured = self._capture_worker_exception(lambda: queue.cancel_current())
        assert len(captured) == 1
        assert isinstance(captured[0], RuntimeError)
        assert "cross-thread" in str(captured[0]).lower()
        # Cancel must NOT have taken effect.
        assert queue.depth == 1

    def test_main_thread_read_only_accessors_succeed(self, queue):
        queue.enqueue("A")
        # `depth` and `peek` are read-only — no thread guard. Calling
        # them on the main thread is the happy path.
        assert queue.depth == 1
        assert queue.peek() == "A"


# --------------------------------------------------------------------------- #
# AC #1 / #7 — TestModuleBoundary
# --------------------------------------------------------------------------- #

class TestModuleBoundary:
    """AST-scan static analysis. Mirrors the convention from Story
    11.2's `TestModuleBoundary` and Story 11.4's static-scan tests."""

    @pytest.fixture
    def imports(self) -> list[str]:
        """Return every module name imported by `playback_queue.py`.

        For `import X` and `import X.Y`: returns the alias.name (full
        dotted form).
        For `from X.Y import Z`: returns the module attribute (`X.Y`).
        For relative imports (`from . import ...`): node.module is
        `None` — surfaced as a failure so the project's
        absolute-import convention is enforced.
        """
        module_path = Path(queue_module.__file__)
        tree = ast.parse(module_path.read_text(encoding="utf-8"))

        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    # Relative import — project convention is absolute
                    # imports throughout `services/sessions/*`.
                    pytest.fail(
                        "playback_queue.py uses a relative import; "
                        "project convention is absolute imports."
                    )
                names.append(node.module)
        return names

    def test_playback_queue_does_not_import_forbidden_modules(self, imports):
        forbidden = {
            "myvoice.services.audio_coordinator",
            "myvoice.services.audio_service",
            "myvoice.services.qwen_tts_service",
            "myvoice.services.sessions.session_registry",
        }
        forbidden_prefixes = ("myvoice.ui",)
        for name in imports:
            assert name not in forbidden, (
                f"playback_queue.py must not import {name!r} per AC #1 / "
                f"architecture lines 665-667"
            )
            for prefix in forbidden_prefixes:
                assert not name.startswith(prefix), (
                    f"playback_queue.py must not import from {prefix!r} "
                    f"(found: {name!r}) per AC #1"
                )

    def test_playback_queue_only_imports_allowed_modules(self, imports):
        # Per AC #1: stdlib (collections, typing, logging) + PyQt6.QtCore
        # are mandatory; PyQt6.QtWidgets is required for the
        # `QApplication.instance()` check (Task 1.2 explicit). The
        # `generation_session` import is OPTIONAL and only present
        # under `TYPE_CHECKING` if needed.
        allowed = {
            "collections",
            "logging",
            "typing",
            "PyQt6.QtCore",
            "PyQt6.QtWidgets",
        }
        allowed_prefixes = ("myvoice.services.sessions.generation_session",)
        for name in imports:
            if name in allowed:
                continue
            if any(name.startswith(p) for p in allowed_prefixes):
                continue
            pytest.fail(
                f"playback_queue.py imported unexpected module {name!r}; "
                f"allowed set is {sorted(allowed)} (plus optional "
                f"{allowed_prefixes})"
            )


# --------------------------------------------------------------------------- #
# AC #2 / #4 — TestCancelCurrent
# --------------------------------------------------------------------------- #

class TestCancelCurrent:
    """Cancel-specific edge cases: empty no-op, drops head only,
    repeated cancel drains."""

    def test_cancel_current_drops_head(self, queue):
        queue.enqueue("A")
        queue.enqueue("B")
        queue.enqueue("C")
        queue.cancel_current()  # drops "A"
        # FIFO contract: next dequeue returns "B" (the new head),
        # NOT "A" (which was cancelled).
        assert queue.dequeue() == "B"
        assert queue.depth == 1
        assert queue.peek() == "C"

    def test_cancel_current_returns_none(self, queue):
        queue.enqueue("A")
        result = queue.cancel_current()
        # Signature contract: returns None explicitly. The cancelled
        # session id is not exposed (Open Question #2 — left as-is).
        assert result is None

    def test_cancel_current_repeated_drains_queue(self, queue):
        queue.enqueue("A")
        queue.enqueue("B")
        queue.enqueue("C")
        history = make_spy(queue.playback_queue_depth_changed)

        queue.cancel_current()
        queue.cancel_current()
        queue.cancel_current()
        assert queue.depth == 0
        # Three real pops, one signal each, depths 2 → 1 → 0.
        assert history == [2, 1, 0]

        # Fourth cancel against empty queue: no-op, no exception, no
        # extra signal.
        queue.cancel_current()
        assert queue.depth == 0
        assert history == [2, 1, 0]
