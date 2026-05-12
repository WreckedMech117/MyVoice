"""
Single-chokepoint metric helper (Story 11.3, architecture P-9).

Architecture rule (line 678-679 of architecture-optimization-pass.md):
    "EVERYTHING may import metrics; metrics imports nothing."

Imports are restricted to the Python standard library. Anything in this module
that needs ``myvoice.*`` types is a bug — module boundary discipline is
enforced by a static AST scan in ``tests/unit/observability/test_metrics.py``.

Public surface
--------------
``record(name, value, *, session_id=None, **tags)``
    Emit a structured ``LogRecord`` on the ``myvoice.metrics`` logger and
    invoke every registered in-process listener with a ``MetricRecord``.

``add_listener(callback)`` / ``remove_listener(callback)``
    Subscribe / unsubscribe in-process consumers (e.g., the
    ``_FirstChunkLatencyAggregator`` that derives ``_avg_first_chunk_latency``
    from the metric stream).

``MetricRecord``
    Frozen dataclass passed to listeners; decouples consumers from
    ``logging.LogRecord`` internals.

``METRIC_LOGGER_NAME``
    The logger name (``"myvoice.metrics"``); exported so tests and
    application-level handlers can attach without hard-coding the literal.
"""

from dataclasses import dataclass, field
import logging
import threading
from typing import Any, Callable, Optional


METRIC_LOGGER_NAME = "myvoice.metrics"

_logger = logging.getLogger(METRIC_LOGGER_NAME)

_listeners: "list[Callable[[MetricRecord], None]]" = []
_listeners_lock = threading.Lock()


@dataclass(frozen=True)
class MetricRecord:
    """
    Immutable record passed to listeners (AC #11).

    The ``tags`` dict is the same object that was attached to the underlying
    ``LogRecord``'s ``extra`` payload — it is **not** defensively copied.
    Listeners that intend to mutate ``tags`` (e.g., to enrich with derived
    fields) MUST copy via ``dict(record.tags)`` first; otherwise the mutation
    is observed by every later listener AND by any structured-log handler that
    reads ``log_record.tags`` after emission.

    Field assignment is frozen (``dataclasses.FrozenInstanceError`` on write),
    but the ``tags`` dict itself is mutable — only the binding is immutable.

    Story 16.6 widens ``value`` to ``float | int | str`` to match architecture
    P-9's ``"value": <number_or_string>`` schema (line 470). The original
    Story 11.3 implementation narrowed this to numeric-only; the
    ``streaming_mode`` and ``streaming_mode_fallback`` metrics emitted from
    ``QwenTTSService._dispatch_by_streaming_mode`` are the first string-valued
    consumers. Numeric listeners (e.g., ``_FirstChunkLatencyAggregator``) are
    expected to type-check ``isinstance(record.value, (int, float))`` before
    arithmetic; the project's existing aggregator already filters by metric
    name so unrelated string-valued metrics never reach its averaging code.
    """

    name: str
    value: "float | int | str"
    session_id: Optional[str]
    tags: "dict[str, Any]" = field(default_factory=dict)


def record(
    name: str,
    value: "float | int | str",
    *,
    session_id: Optional[str] = None,
    **tags: Any,
) -> None:
    """
    Emit one structured metric record (AC #4, #5, #6, #7).

    Order of operations (matters for AC #9 / AC #10):
        1. Validate ``name`` and ``value``; raise BEFORE any side effect.
        2. Emit ``LogRecord`` on ``myvoice.metrics`` at INFO with the P-9 schema.
        3. Snapshot the listener list under the lock; release the lock.
        4. Invoke each listener; catch and WARN-log any exception so a buggy
           listener cannot prevent later listeners from running.

    Steps 3-4 release the lock before listener invocation so a listener that
    re-enters ``record()`` (or calls ``add_listener``) cannot deadlock.
    """
    # AC #7: name guard — None / empty / whitespace-only all reject.
    if not isinstance(name, str) or not name.strip():
        raise ValueError("metrics.record() name must be a non-empty string")

    # AC #6 (Story 16.6 widening): value guard — int, float, bool (subclass of
    # int), and str now all accepted; everything else rejected. The string
    # branch is required for the ``streaming_mode`` / ``streaming_mode_fallback``
    # metrics defined in Story 16.6's three-mode dispatch (D-19 / P-9 line 470:
    # ``"value": <number_or_string>``).
    if not isinstance(value, (int, float, str)):
        raise TypeError(
            f"metrics.record() value must be int, float, or str; "
            f"got {type(value).__name__}"
        )

    # AC #4 / AC #5: extra payload — tags always present (even when empty),
    # nested under "tags" so downstream parsers iterate ``record.tags.items()``
    # without colliding against ``metric_name`` / ``value`` / ``session_id``.
    # ``dict(tags)`` materializes a fresh dict from the kwargs map (kwargs are
    # already a fresh dict per call, but this avoids any aliasing surprise and
    # is documented as the "tags dict the listener sees" object).
    tag_dict: "dict[str, Any]" = dict(tags)
    extra = {
        "metric_name": name,
        "value": value,
        "session_id": session_id,
        "tags": tag_dict,
    }

    # AC #4: structured-log emission. Log first, then listeners (Dev Notes
    # "Listener invocation order"). ``_logger.info`` is internally thread-safe.
    _logger.info("metric", extra=extra)

    # AC #10: snapshot under lock, iterate outside. Prevents
    # "list modified during iteration" under concurrent add_listener /
    # remove_listener, and prevents recursion deadlock if a listener calls
    # back into record() or add_listener().
    with _listeners_lock:
        snapshot = list(_listeners)

    # AC #11: pass the SAME tags dict the LogRecord saw — not a copy.
    record_obj = MetricRecord(
        name=name,
        value=value,
        session_id=session_id,
        tags=tag_dict,
    )

    # AC #9: listener exception isolation. A raising listener does not prevent
    # subsequent listeners from running; the exception is logged at WARNING
    # to the metrics logger (not the listener's own logger) and record()
    # returns normally.
    for cb in snapshot:
        try:
            cb(record_obj)
        except Exception:
            _logger.warning("metrics listener raised", exc_info=True)


def add_listener(
    callback: "Callable[[MetricRecord], None]",
) -> "Callable[[], None]":
    """
    Register an in-process listener (AC #2, #8, #10).

    Returns a zero-arg callable that, when invoked, removes ``callback`` from
    the listener list. The unsubscribe is idempotent: calling it twice is a
    silent no-op (matching ``remove_listener``'s identity-based removal).
    """
    with _listeners_lock:
        _listeners.append(callback)

    removed = [False]

    def unsubscribe() -> None:
        # Idempotent: only the first call actually removes the listener.
        # Subsequent calls short-circuit so users can call ``unsub()`` from
        # multiple cleanup paths (e.g., ``stop()`` and ``__del__``) without
        # caring about ordering.
        if removed[0]:
            return
        removed[0] = True
        remove_listener(callback)

    return unsubscribe


def remove_listener(callback: "Callable[[MetricRecord], None]") -> None:
    """
    Remove a previously registered listener by identity (AC #8, #10).

    Silent no-op if ``callback`` is not currently registered — matches the
    semantics of ``set.discard`` (vs. ``set.remove``) so cleanup paths that
    don't track registration state can call this without try/except.
    """
    with _listeners_lock:
        try:
            _listeners.remove(callback)
        except ValueError:
            # Not registered — silent no-op per AC #8.
            pass
