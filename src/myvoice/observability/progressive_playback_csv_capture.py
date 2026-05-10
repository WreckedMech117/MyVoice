"""Story 18.1 Task 1.4 — env-var-gated CSV capture for the three new
progressive-playback instrumentation metrics.

Engages a single ``metrics.add_listener`` callback that filters for:

  * ``progressive_chunk_emit_ms``                 (producer / qwen_tts_service.py)
  * ``progressive_chunk_playback_arrival_ms``     (consumer / app.py)
  * ``progressive_chunk_audio_duration_ms``       (consumer / app.py)
  * ``first_chunk_latency_ms``                    (producer / qwen_tts_service.py)
    — added by Story 18.2 Task 4.1 so the same env-var-gated capture
    surface drives the NFR1 first-chunk-latency before/after measurement
    for Stories 18.2 + 18.3 + 18.4. The chunk-specific tag columns
    (chunk_index / is_final / audio_data_size) are blank for these rows;
    downstream analysis distinguishes by ``metric_name``.

and writes one row per record to a CSV file. Every other metric in the
stream is ignored — the file stays focused on the Story 18.1 ratio
analysis (inter-chunk-emit interval vs chunk audio-duration) so the
post-processing script does not have to filter at parse time.

Activation
----------
Set the environment variable ``MYVOICE_PROGRESSIVE_PLAYBACK_CSV`` before
launching the application. Three accepted forms:

    MYVOICE_PROGRESSIVE_PLAYBACK_CSV=1
        Write to the default path:
        ``<logs_dir>/progressive-playback-instrumentation.csv``
        (story-agnostic filename; the file captures four metrics that
        span Stories 18.1 + 18.2 and will be reused by 18.3 + 18.4).
        Story 18.1's canonical baseline at
        ``_bmad-output/implementation-artifacts/18-1-instrumentation-rtx5090-longform.csv``
        is preserved separately and is NOT overwritten by ``=1``.

    MYVOICE_PROGRESSIVE_PLAYBACK_CSV=/abs/path/to/file.csv
        Write to the given absolute path.

    MYVOICE_PROGRESSIVE_PLAYBACK_CSV=relative/path.csv
        Write to the given path resolved against ``<logs_dir>``.

Unset (or any other value the env-var resolution rejects) → capture is
disabled. There is no production overhead when the env-var is unset:
``enable_csv_capture`` returns ``None`` immediately.

Lifecycle
---------
``enable_csv_capture(path)`` opens the file, writes the CSV header,
registers the listener, and returns a zero-arg ``stop`` callable. The
caller (``MyVoiceApp``) invokes ``stop()`` from the ``aboutToQuit``
cleanup so the file flushes + closes deterministically. Idempotent:
calling ``stop()`` twice is a silent no-op.

Concurrency
-----------
The listener is invoked synchronously from inside ``metrics.record``
(which holds no lock at invocation time — see metrics.py:130-153). The
listener writes to the CSV file under a small ``threading.Lock`` so
producer-thread emissions (``_wrapped_post`` runs on the talker /
worker thread) cannot interleave-corrupt with consumer-thread emissions
(``_handle_progressive_chunk_async`` runs on the qasync event loop).
"""

from __future__ import annotations

import csv
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional

from myvoice.observability.metrics import MetricRecord, add_listener


_CAPTURED_METRIC_NAMES = frozenset(
    {
        "progressive_chunk_emit_ms",
        "progressive_chunk_playback_arrival_ms",
        "progressive_chunk_audio_duration_ms",
        # Story 18.2 Task 4.1: first-chunk-latency capture for the
        # NFR1 before/after measurement. Reused by Stories 18.3 + 18.4.
        "first_chunk_latency_ms",
    }
)

_CSV_HEADER = (
    "metric_name",
    "value",
    "session_id",
    "chunk_index",
    "is_final",
    "audio_data_size",
)

_DEFAULT_FILENAME = "progressive-playback-instrumentation.csv"

_logger = logging.getLogger(__name__)


def _resolve_path(env_value: str, logs_dir: Path) -> Optional[Path]:
    """Resolve the env-var value to an absolute CSV path.

    Returns ``None`` for env values the spec rejects (empty / "0" /
    whitespace-only) so the caller can treat capture as disabled.
    """
    stripped = env_value.strip()
    if not stripped or stripped == "0":
        return None
    if stripped == "1":
        return logs_dir / _DEFAULT_FILENAME
    candidate = Path(stripped)
    if candidate.is_absolute():
        return candidate
    return logs_dir / candidate


def enable_csv_capture(
    path: Path,
) -> Callable[[], None]:
    """Open ``path``, register a metrics listener, return a stop callable.

    The header is written immediately so ``ls -la <path>`` after launch
    always shows a non-empty file (useful for "did the env-var take
    effect" diagnostics). Each record is flushed to disk after writing
    so an abrupt termination (Ctrl-C / crash) preserves the rows
    captured so far.

    Raises any OSError from the ``open(...)`` call upward — the caller
    decides whether a missing/unwritable path is fatal. Listener
    registration only happens after the file is successfully opened
    and the header is written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_obj = path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(file_obj)
    writer.writerow(_CSV_HEADER)
    file_obj.flush()

    write_lock = threading.Lock()
    closed = [False]

    def listener(record: MetricRecord) -> None:
        if record.name not in _CAPTURED_METRIC_NAMES:
            return
        with write_lock:
            if closed[0]:
                return
            try:
                writer.writerow(
                    (
                        record.name,
                        record.value,
                        record.session_id if record.session_id is not None else "",
                        record.tags.get("chunk_index", ""),
                        record.tags.get("is_final", ""),
                        record.tags.get("audio_data_size", ""),
                    )
                )
                file_obj.flush()
            except Exception:
                # Listener exceptions are swallowed by metrics.record (AC #9
                # of Story 11.3 — see metrics.py:149-153). Log loudly here
                # so a silent CSV-corruption mode is visible during the
                # measurement run.
                _logger.exception(
                    "progressive-playback CSV write failed"
                )

    unsub = add_listener(listener)

    def stop() -> None:
        if closed[0]:
            return
        # Order matters: unsubscribe FIRST so a late record cannot reach
        # the writer after the file is closed.
        try:
            unsub()
        finally:
            with write_lock:
                closed[0] = True
                try:
                    file_obj.close()
                except Exception:
                    _logger.exception(
                        "progressive-playback CSV close failed"
                    )

    _logger.info(
        "progressive-playback CSV capture enabled: path=%s metrics=%s",
        path,
        sorted(_CAPTURED_METRIC_NAMES),
    )
    return stop


def maybe_enable_from_env(logs_dir: Path) -> Optional[Callable[[], None]]:
    """Inspect ``MYVOICE_PROGRESSIVE_PLAYBACK_CSV`` and engage capture if set.

    Returns the ``stop`` callable on success, ``None`` if the env-var is
    unset or resolves to "disabled". Any unexpected error during file
    open is logged and ``None`` is returned — the application launch
    must not fail because instrumentation could not engage.
    """
    raw = os.environ.get("MYVOICE_PROGRESSIVE_PLAYBACK_CSV")
    if raw is None:
        return None
    resolved = _resolve_path(raw, logs_dir)
    if resolved is None:
        return None
    try:
        return enable_csv_capture(resolved)
    except Exception:
        _logger.exception(
            "Failed to enable progressive-playback CSV capture at %s",
            resolved,
        )
        return None
