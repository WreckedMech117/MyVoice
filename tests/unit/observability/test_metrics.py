"""
Tests for myvoice.observability.metrics (Story 11.3).

Covers AC #1 through #11, #17, and the static-scan boundary discipline (AC #3).
The aggregator-migration tests (AC #12-#16) live in:
- tests/unit/services/test_qwen_tts_metric_migration_static.py (static scan)
- tests/integration/test_qwen_tts_metrics_migration.py (numerical equivalence)
"""

import ast
import dataclasses
import inspect
import logging
import threading
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from myvoice.observability import (
    METRIC_LOGGER_NAME,
    MetricRecord,
    add_listener,
    record,
    remove_listener,
)
from myvoice.observability import metrics as metrics_module


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def clean_listeners():
    """Clear the module-level listener list around every test (AC #19)."""
    # Pre-test: ensure clean slate (a previous test crashing mid-registration
    # could leave residue).
    metrics_module._listeners.clear()
    yield
    # Post-test: drop any listener the test forgot to unsubscribe.
    metrics_module._listeners.clear()


class _ListHandler(logging.Handler):
    """Logging handler that captures every emitted ``LogRecord``."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level=level)
        self.records: "list[logging.LogRecord]" = []

    def emit(self, log_record: logging.LogRecord) -> None:
        self.records.append(log_record)


@pytest.fixture
def captured_logs(request):
    """Attach a list-handler to ``myvoice.metrics``; auto-detach after test."""
    logger = logging.getLogger(METRIC_LOGGER_NAME)
    handler = _ListHandler(level=logging.DEBUG)
    # ``propagate`` may be True (inherits from root) — leave it alone so the
    # test mirrors the production environment. Setting our own level means we
    # capture INFO and WARNING records regardless of root config.
    prior_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    def _detach():
        logger.removeHandler(handler)
        logger.setLevel(prior_level)

    request.addfinalizer(_detach)
    return handler


# --------------------------------------------------------------------------- #
# AC #1, #2 — module exports
# --------------------------------------------------------------------------- #


class TestModuleExports:
    """All five public names importable from the package and the module."""

    def test_package_exports_record(self):
        from myvoice.observability import record as r

        assert callable(r)

    def test_package_exports_add_remove_listener(self):
        from myvoice.observability import add_listener as a, remove_listener as rm

        assert callable(a)
        assert callable(rm)

    def test_package_exports_metric_record(self):
        from myvoice.observability import MetricRecord as MR

        assert MR is MetricRecord
        assert dataclasses.is_dataclass(MR)

    def test_package_exports_logger_name_constant(self):
        from myvoice.observability import METRIC_LOGGER_NAME as constant

        assert constant == "myvoice.metrics"

    def test_module_exports_match_package_exports(self):
        # All five names must also be importable from myvoice.observability.metrics.
        from myvoice.observability.metrics import (
            METRIC_LOGGER_NAME as a,
            MetricRecord as b,
            add_listener as c,
            record as d,
            remove_listener as e,
        )

        assert a == "myvoice.metrics"
        assert b is MetricRecord
        assert c is add_listener
        assert d is record
        assert e is remove_listener


# --------------------------------------------------------------------------- #
# AC #4, #5 — record schema
# --------------------------------------------------------------------------- #


class TestRecordSchema:
    """LogRecord shape conforms to architecture P-9."""

    def test_emits_exactly_one_record_with_correct_fields(self, captured_logs):
        record(
            "test_metric",
            1.5,
            session_id="s1",
            model_type="m",
            hardware="gpu",
        )

        assert len(captured_logs.records) == 1
        log_rec = captured_logs.records[0]
        assert log_rec.name == "myvoice.metrics"
        assert log_rec.levelno == logging.INFO
        assert log_rec.msg == "metric"
        assert log_rec.metric_name == "test_metric"
        assert log_rec.value == 1.5
        assert log_rec.session_id == "s1"
        assert log_rec.tags == {"model_type": "m", "hardware": "gpu"}

    def test_empty_tags_still_present_as_empty_dict(self, captured_logs):
        record("queue_depth", 3)

        assert len(captured_logs.records) == 1
        log_rec = captured_logs.records[0]
        assert log_rec.metric_name == "queue_depth"
        assert log_rec.value == 3
        assert log_rec.session_id is None
        # AC #5: "tags" key always present, even when empty.
        assert log_rec.tags == {}

    def test_session_id_separate_from_tags(self, captured_logs):
        record("m", 1, session_id="abc", model_type="x")

        log_rec = captured_logs.records[0]
        assert log_rec.session_id == "abc"
        assert "session_id" not in log_rec.tags
        assert log_rec.tags == {"model_type": "x"}


# --------------------------------------------------------------------------- #
# AC #6 — value validation
# --------------------------------------------------------------------------- #


class TestRecordValueValidation:
    """``value`` must be int/float/bool/str; everything else raises TypeError.

    Story 16.6 widens the accepted types to include ``str`` to match
    architecture P-9 (``"value": <number_or_string>``); the
    ``streaming_mode`` / ``streaming_mode_fallback`` metrics emitted from
    ``QwenTTSService._dispatch_by_streaming_mode`` are the first
    string-valued consumers.
    """

    @pytest.mark.parametrize("value", [1, 1.5, True, False, 0, -3, 3.14e10])
    def test_accepts_python_numerics_and_bool(self, value, captured_logs):
        record("x", value)
        # No exception, log emitted.
        assert len(captured_logs.records) == 1

    @pytest.mark.parametrize(
        "value",
        ["true_stream", "sentence_stream", "batch", "unrecoverable", ""],
    )
    def test_accepts_string_values_per_p9(self, value, captured_logs):
        """Story 16.6 — string values accepted per P-9 schema."""
        record("streaming_mode", value)
        assert len(captured_logs.records) == 1
        assert captured_logs.records[0].value == value

    @pytest.mark.parametrize(
        "bad_value",
        [None, [1], {"k": 1}, (1, 2)],
    )
    def test_rejects_non_numeric_and_non_string(self, bad_value, captured_logs):
        with pytest.raises(TypeError) as exc:
            record("x", bad_value)

        # AC #6: error message names the offending type.
        assert type(bad_value).__name__ in str(exc.value)
        # No log emitted on failure — validation runs BEFORE emission.
        assert captured_logs.records == []

    def test_numpy_scalars_accepted_when_numpy_available(self, captured_logs):
        # numpy is a project dep; ``np.float32`` is a ``numbers.Real`` subclass
        # but ``isinstance(np.float32(1.0), (int, float))`` is ALSO True
        # (np.float32 is a float subclass on most platforms). This is
        # documented as expected behavior — the intent is "numeric scalar".
        np = pytest.importorskip("numpy")

        # np.float32 may or may not be a float subclass depending on numpy
        # version; the test asserts whichever behavior the runtime has, with
        # a clear message either way.
        is_float_subclass = isinstance(np.float32(1.0), (int, float))
        if is_float_subclass:
            record("np_metric", np.float32(1.0))
            record("np_metric", np.int64(2))
            assert len(captured_logs.records) == 2
        else:
            with pytest.raises(TypeError):
                record("np_metric", np.float32(1.0))


# --------------------------------------------------------------------------- #
# AC #7 — name validation
# --------------------------------------------------------------------------- #


class TestRecordNameValidation:
    """Names must be non-empty post-strip strings."""

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            record("", 1)

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError):
            record("   ", 1)

    def test_none_raises(self):
        # Implementation uses isinstance check before .strip() — so ``None``
        # produces a ValueError (NOT TypeError). AC #7 documents either as
        # acceptable; this test pins the chosen behavior.
        with pytest.raises(ValueError):
            record(None, 1)  # type: ignore[arg-type]

    def test_tabs_and_newlines_raise(self):
        with pytest.raises(ValueError):
            record("\t\n", 1)


# --------------------------------------------------------------------------- #
# AC #8 — listener registration
# --------------------------------------------------------------------------- #


class TestListenerRegistration:
    """add_listener semantics: invocation, instance type, ordering."""

    def test_listener_invoked_with_metric_record(self, captured_logs):
        seen: "list[MetricRecord]" = []
        add_listener(seen.append)

        record("m", 1.5, session_id="s", k="v")

        assert len(seen) == 1
        assert isinstance(seen[0], MetricRecord)
        assert seen[0].name == "m"
        assert seen[0].value == 1.5
        assert seen[0].session_id == "s"
        assert seen[0].tags == {"k": "v"}

    def test_listeners_invoked_in_registration_order(self, captured_logs):
        order: "list[str]" = []
        add_listener(lambda r: order.append("A"))
        add_listener(lambda r: order.append("B"))

        record("m", 1)

        assert order == ["A", "B"]

    def test_listener_invoked_after_log_emission(self, captured_logs):
        # AC ordering: log first, listener second. A listener that snapshots
        # ``len(captured_logs.records)`` should see >= 1 because the log
        # emission has already completed.
        snapshots: "list[int]" = []

        def snapshot(r):
            snapshots.append(len(captured_logs.records))

        add_listener(snapshot)

        record("m", 1)

        assert snapshots == [1]


# --------------------------------------------------------------------------- #
# AC #8 — unsubscribe
# --------------------------------------------------------------------------- #


class TestListenerUnsubscribe:
    """add_listener returns a closure; remove_listener is identity-based."""

    def test_unsubscribe_removes_listener(self, captured_logs):
        seen: "list[MetricRecord]" = []
        unsub = add_listener(seen.append)

        record("first", 1)
        unsub()
        record("second", 2)

        assert len(seen) == 1
        assert seen[0].name == "first"

    def test_unsubscribe_idempotent(self, captured_logs):
        seen: "list[MetricRecord]" = []
        unsub = add_listener(seen.append)

        unsub()
        # Second call must not raise.
        unsub()

        record("after", 1)
        assert seen == []

    def test_remove_listener_by_identity(self, captured_logs):
        seen: "list[MetricRecord]" = []

        def cb(r):
            seen.append(r)

        add_listener(cb)
        remove_listener(cb)

        record("after", 1)
        assert seen == []

    def test_remove_listener_unregistered_silent_noop(self):
        def never_registered(r):
            pass

        # No registration; remove must be silent.
        remove_listener(never_registered)


# --------------------------------------------------------------------------- #
# AC #9 — listener exception isolation
# --------------------------------------------------------------------------- #


class TestListenerExceptionIsolation:
    """A raising listener does not prevent later listeners from running."""

    def test_raising_listener_does_not_break_chain(self, captured_logs):
        counter = [0]

        def bad(r):
            raise RuntimeError("boom")

        def counting(r):
            counter[0] += 1

        add_listener(bad)
        add_listener(counting)

        # record() must return without raising.
        record("m", 1)

        assert counter[0] == 1

    def test_exception_logged_at_warning_on_metrics_logger(self, captured_logs):
        def bad(r):
            raise ValueError("listener failed")

        add_listener(bad)
        record("m", 1)

        # captured_logs has the original INFO record AND the WARNING.
        warnings = [r for r in captured_logs.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].name == "myvoice.metrics"
        # exc_info should carry the exception
        assert warnings[0].exc_info is not None
        assert warnings[0].exc_info[0] is ValueError


# --------------------------------------------------------------------------- #
# AC #10 — thread safety
# --------------------------------------------------------------------------- #


class TestThreadSafety:
    """Concurrent record() calls do not lose records or corrupt state."""

    def test_stress_no_lost_records(self):
        counter_lock = threading.Lock()
        seen = [0]

        def counting(r):
            with counter_lock:
                seen[0] += 1

        add_listener(counting)

        n_threads = 10
        n_per_thread = 100

        def worker():
            for i in range(n_per_thread):
                record("stress", i)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert seen[0] == n_threads * n_per_thread

    def test_no_runtime_error_when_listeners_mutated_during_record(self):
        # A listener that adds/removes other listeners must not corrupt the
        # iteration in progress. The snapshot-under-lock pattern guarantees
        # this — we exercise it with a self-removing listener.
        seen = [0]

        def self_removing(r):
            seen[0] += 1
            remove_listener(self_removing)

        add_listener(self_removing)

        # Two records: first one fires the listener (which removes itself);
        # second one finds the listener gone.
        record("a", 1)
        record("b", 2)

        assert seen[0] == 1


# --------------------------------------------------------------------------- #
# AC #3 — module boundary (static AST scan)
# --------------------------------------------------------------------------- #


class TestModuleBoundary:
    """metrics.py must only import from stdlib."""

    ALLOWED_TOP_LEVEL_MODULES = {"dataclasses", "logging", "threading", "typing"}
    FORBIDDEN_PATTERNS = (
        "from myvoice",
        "import numpy",
        "import torch",
        "import PyQt6",
        "import qwen_tts",
        "from numpy",
        "from PyQt6",
        "from torch",
        "from qwen_tts",
    )

    def _read_source(self) -> str:
        path = inspect.getsourcefile(metrics_module)
        assert path is not None
        return Path(path).read_text(encoding="utf-8")

    def test_no_forbidden_import_substrings(self):
        source = self._read_source()
        # We compare against the module's source after stripping docstrings,
        # because the module docstring legitimately contains the architecture
        # rule which has the substring "may import metrics". The AST-based
        # check below is the authoritative one; the substring check is a
        # belt-and-braces guard against accidental top-level imports.
        tree = ast.parse(source)
        # Drop the module docstring before serializing back.
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            tree.body.pop(0)
        non_doc_source = ast.unparse(tree)
        for pattern in self.FORBIDDEN_PATTERNS:
            assert pattern not in non_doc_source, (
                f"Forbidden import pattern {pattern!r} found in metrics.py"
            )

    def test_only_allowed_top_level_imports(self):
        source = self._read_source()
        tree = ast.parse(source)
        top_level_modules = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_level_modules.add(node.module.split(".")[0])
        assert top_level_modules <= self.ALLOWED_TOP_LEVEL_MODULES, (
            f"metrics.py imports beyond stdlib whitelist: "
            f"{top_level_modules - self.ALLOWED_TOP_LEVEL_MODULES}"
        )


# --------------------------------------------------------------------------- #
# AC #11 — MetricRecord immutability
# --------------------------------------------------------------------------- #


class TestMetricRecordImmutability:
    """MetricRecord is frozen; tags dict is intentionally NOT copied."""

    def test_field_assignment_raises_frozen_instance_error(self):
        rec = MetricRecord(name="m", value=1.0, session_id=None, tags={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.value = 99  # type: ignore[misc]

    def test_tag_dict_mutation_succeeds_intentionally(self):
        # AC #11 explicitly: ``tags`` dict is the same object passed in, NOT
        # defensively copied. Mutating it is allowed (and is a documented
        # contract — listeners that need to mutate must copy first).
        tags: "dict[str, Any]" = {"k": 1}
        rec = MetricRecord(name="m", value=1.0, session_id=None, tags=tags)

        rec.tags["new_tag"] = "x"

        assert rec.tags["new_tag"] == "x"
        # The original dict was mutated through the record (same object).
        assert tags["new_tag"] == "x"


# --------------------------------------------------------------------------- #
# AC #17 — opt-in logger configuration
# --------------------------------------------------------------------------- #


class TestLoggerConfigurationIsOptIn:
    """metrics.py must not configure logging itself."""

    def test_no_handlers_attached_by_module(self):
        # The module must not pre-attach any handler. Tests attach their own.
        # We check the captured handler list excluding the test's own handler.
        logger = logging.getLogger(METRIC_LOGGER_NAME)
        # Allowed: handlers added by tests (via captured_logs fixture). Here
        # we did NOT request that fixture, so the handler list should be empty.
        assert logger.handlers == []
