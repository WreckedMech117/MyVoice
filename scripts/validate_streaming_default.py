#!/usr/bin/env python
"""Story 16.7 / 16.9 — Empirical validation harness for the TRUE_STREAM streaming default.

Measures first-audio latency under the production Qwen3-TTS dispatch path on the
maintainer's GPU host (RTX 5090 Blackwell + Win11 + torch 2.10+cu128 per
``memory/hardware_setup.md``) and on a CPU host as the NFR1 inheritance check.

Story 16.9 extension (added via Subtask 1.1): four new flags wire phase-decomposition
profiling and tier/stratified-sample comparisons onto the existing harness:

  - ``--profile-phases`` (Task 1/2): patches ``service._split_text_for_streaming``,
    ``service._generate_sync``, and the registry's ``post_mutation`` so each
    measurement row carries ``split_seconds`` (phase a), ``generate_seconds``
    (phase b — first-chunk only; merges phase c per AC #1 L3 default), and
    ``deliver_seconds`` (phase d — first ``append_chunk`` registry mutation).
  - ``--quality-tier {small,quality}`` (Task 3.2 hypothesis (b) probe): sets the
    registry's persistent quality tier in-process via ``set_quality_tier`` BEFORE
    the measurement loop. Mutates only the in-memory registry; does NOT write to
    AppSettings on disk.
  - ``--stratified-sample N:N:N`` (Task 6 CPU baseline extension): selects the
    first N short + N medium + N long utterances from the input set instead of
    ``[:limit]`` truncation (which defaults to short-class-only on the
    class-ordered Story 16.7 input set).
  - ``--output-csv-name FILENAME`` (any Story 16.9 task): explicit override of the
    derived CSV filename so the harness produces ``16-9-*`` artifacts without
    relying on flag-combination derivation rules.

The harness exists so the streaming-default-flag flip (Story 16.6 already wired
the dispatch chain; ``streaming_mode.py:54-56`` already routes CUDA hosts to
TRUE_STREAM) can be informed by data rather than guesswork. It produces:

  - A measurement CSV (``--output-dir/16-7-gpu-latency-measurements.csv`` for
    TRUE_STREAM on CUDA; ``16-7-cpu-baseline-measurements.csv`` for the CPU
    SENTENCE_STREAM run; ``16-7-gpu-sentence_stream-comparison.csv`` for the
    GPU apples-to-apples comparison).
  - p50/p95/p99 first-chunk-latency aggregates printed to stdout.
  - An explicit ``NFR1 GATE: ... (PASS|FAIL ...)`` line.
  - A fallback-rate readout (rows where the public-dispatch path fell back
    through Story 16.6's three-mode chain are flagged ``error_flag=
    'fallback_occurred'`` and excluded from the aggregate).

Usage::

    # GPU TRUE_STREAM measurement (primary deliverable for AC #1)
    python scripts/validate_streaming_default.py \\
        --input-set _bmad-output/implementation-artifacts/16-7-input-set.csv \\
        --output-dir _bmad-output/implementation-artifacts/ \\
        --mode-override true_stream --utterance-count 50

    # CPU baseline (AC #3)
    set CUDA_VISIBLE_DEVICES=
    python scripts/validate_streaming_default.py \\
        --input-set _bmad-output/implementation-artifacts/16-7-input-set.csv \\
        --output-dir _bmad-output/implementation-artifacts/ \\
        --mode-override sentence_stream --utterance-count 10

    # No-override resolver (AC #3 final clause)
    python scripts/validate_streaming_default.py \\
        --input-set _bmad-output/implementation-artifacts/16-7-input-set.csv \\
        --output-dir _bmad-output/implementation-artifacts/ \\
        --utterance-count 50

The harness consumes Stories 16.1-16.6 unmodified — no edits to ``src/myvoice/``
and no edits to existing tests. It mirrors the standalone-script convention
established by ``scripts/validate_embedding_api.py``.

Architecture references:
  - D-9 / NFR12: hardware-aware streaming default; harness refuses TRUE_STREAM
    on CPU per AC #3.
  - NFR1: first audio < 2s; harness reports p95 against this 2.000s ceiling.
  - D-19 / P-9: ``streaming_mode`` metric is the source of truth for "what
    actually dispatched" when going through the public dispatch entry.
  - Story 16.6 handoff lines 532-565: ``_generate_true_stream(request)``
    direct-call pattern matches the documented Public-contract handoff.
"""

# DLL ordering: torch MUST import before PyQt6 on Windows.
# See memory/torch_pyqt6_dll_ordering.md and src/myvoice/main.py:25-49.
import contextlib
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _torch_lib = _REPO_ROOT / "python310" / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.exists():
        os.add_dll_directory(str(_torch_lib))
    for _cuda_path in (
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp"),
    ):
        if _cuda_path.exists():
            os.add_dll_directory(str(_cuda_path))

import torch  # noqa: E402  — must precede PyQt6 import below

from PyQt6.QtWidgets import QApplication  # noqa: E402

import argparse  # noqa: E402
import asyncio  # noqa: E402
import csv  # noqa: E402
import importlib.metadata  # noqa: E402
import logging  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, asdict, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Callable, Iterator, List, Optional, Tuple  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import numpy as np  # noqa: E402

from myvoice.models.app_settings import AppSettings  # noqa: E402
from myvoice.models.service_enums import QwenModelType  # noqa: E402
from myvoice.observability import metrics  # noqa: E402
from myvoice.observability.metrics import MetricRecord  # noqa: E402
from myvoice.services.audio_coordinator import AudioCoordinator  # noqa: E402
from myvoice.services.monitor_audio_service import MonitorAudioService  # noqa: E402
from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService  # noqa: E402
from myvoice.services.sessions import SessionRegistry  # noqa: E402
from myvoice.services.tts_streaming import codec_token_streamer  # noqa: E402
from myvoice.services.tts_streaming.streaming_mode import (  # noqa: E402
    StreamingMode,
    default_streaming_mode_for_hardware,
)
from myvoice.services.virtual_microphone_service import VirtualMicrophoneService  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scripts.validate_streaming_default")

# NFR1 ceiling per architecture-optimization-pass.md:802.
NFR1_CEILING_SECONDS = 2.000

# Output filename convention (anchored on --output-dir).
GPU_TRUE_STREAM_CSV = "16-7-gpu-latency-measurements.csv"
GPU_SENTENCE_STREAM_CSV = "16-7-gpu-sentence_stream-comparison.csv"
GPU_BATCH_CSV = "16-7-gpu-batch-comparison.csv"
CPU_BASELINE_CSV = "16-7-cpu-baseline-measurements.csv"
NO_OVERRIDE_CSV_GPU = "16-7-gpu-latency-measurements.csv"
NO_OVERRIDE_CSV_CPU = "16-7-cpu-baseline-measurements.csv"

# CSV columns per Story 16.7 AC #1.
CSV_FIELDNAMES = [
    "utterance_id",
    "text_length_chars",
    "text_class",
    "mode_requested",
    "mode_dispatched",
    "first_chunk_latency_seconds",
    "total_audio_seconds",
    "audio_sample_count",
    "error_flag",
    "wallclock_timestamp",
    "qwen_tts_pin",
    "torch_version",
    "gpu_name",
]

# Story 16.9 AC #1: extra columns appended when --profile-phases is set.
# ``decode_seconds`` is merged into ``generate_seconds`` per the L3 default;
# the column is preserved for forward-compat (separate-decode option) and is
# always 0.0 in the merged-mode harness output. ``model_load_seconds`` was
# added after the Subtask 1.5 dry run revealed that ``ensure_model_loaded``
# accounts for ~6.5s of utterance-#1 latency (model load on cold start) but
# ~microseconds of utterance-#2..N latency (cache hit) — without it the
# phase-sum-vs-total sanity check fails on row 1 for every harness run.
PHASE_PROFILE_FIELDNAMES = [
    "split_seconds",
    "model_load_seconds",
    "generate_seconds",
    "decode_seconds",
    "deliver_seconds",
    "split_chunk_count",
    "first_chunk_chars",
]


@dataclass
class MeasurementResult:
    """One row of the measurement CSV (mirrors Story 16.7 AC #1 header exactly).

    Story 16.9 AC #1 appends six phase-profile fields (``split_seconds``,
    ``generate_seconds``, ``decode_seconds``, ``deliver_seconds``,
    ``split_chunk_count``, ``first_chunk_chars``) when ``--profile-phases``
    is set; they default to 0.0 / 0 so the dataclass stays compatible with
    Story 16.7 invocations.
    """
    utterance_id: str
    text_length_chars: int
    text_class: str
    mode_requested: str
    mode_dispatched: str
    first_chunk_latency_seconds: float
    total_audio_seconds: float
    audio_sample_count: int
    error_flag: str
    wallclock_timestamp: str
    qwen_tts_pin: str
    torch_version: str
    gpu_name: str
    # Story 16.9 phase columns (zero unless --profile-phases is set).
    split_seconds: float = 0.0
    model_load_seconds: float = 0.0
    generate_seconds: float = 0.0
    decode_seconds: float = 0.0
    deliver_seconds: float = 0.0
    split_chunk_count: int = 0
    first_chunk_chars: int = 0


@dataclass
class PhaseProfile:
    """Per-utterance phase timings captured via Story 16.9 monkey-patching.

    Records the FIRST chunk's phase timings only — the existing
    ``first_chunk_latency_seconds`` aggregate is the load-bearing metric and
    only the first chunk participates in NFR1. ``decode_seconds`` is merged
    into ``generate_seconds`` per the AC #1 L3 default and stays 0.0 in this
    harness; the field exists so a future ``--separate-decode-phase`` flag
    can populate it without a CSV header change.

    ``model_load_seconds`` was added after Subtask 1.5's dry run: utterance #1
    of any harness session triggers a cold model load inside
    ``ensure_model_loaded`` (~6.5s for the 3B model); utterances #2..N hit
    the cache (~microseconds). Without this column the phase-sum sanity
    check fails on row 1 every run. For Hypothesis (a) qwen-tts version
    drift, ``model_load_seconds`` is informational; the load-bearing column
    remains ``generate_seconds`` (per-chunk inference cost).
    """
    split_seconds: float = 0.0
    model_load_seconds: float = 0.0
    generate_seconds: float = 0.0
    decode_seconds: float = 0.0
    deliver_seconds: float = 0.0
    split_chunk_count: int = 0
    first_chunk_chars: int = 0
    _first_generate_observed: bool = False
    _first_deliver_observed: bool = False
    _first_model_load_observed: bool = False


@dataclass
class InputUtterance:
    """One row of the input-set CSV."""
    utterance_id: str
    text: str
    text_length_chars: int
    text_class: str
    is_perceptual_difficult: bool


@dataclass
class HarnessEnvironment:
    """Captures the run-time environment for every measurement row."""
    qwen_tts_pin: str
    torch_version: str
    gpu_name: str
    cuda_available: bool


@dataclass
class StreamingModeMetricRecorder:
    """Subscribes to ``streaming_mode`` / ``streaming_mode_fallback`` metrics
    and records the LAST value seen for the active session (Story 16.6 emits
    one ``streaming_mode`` metric per attempt — including each fallback
    attempt — so the LAST value is the actually-dispatched mode for that
    session).
    """

    last_streaming_mode: Optional[str] = None
    fallback_observed: bool = False
    unsubscribe_fn: Optional[Callable[[], None]] = field(default=None)

    def __call__(self, record: MetricRecord) -> None:
        if record.name == "streaming_mode":
            self.last_streaming_mode = str(record.value)
        elif record.name == "streaming_mode_fallback":
            self.fallback_observed = True

    def attach(self) -> None:
        self.unsubscribe_fn = metrics.add_listener(self)

    def detach(self) -> None:
        if self.unsubscribe_fn is not None:
            self.unsubscribe_fn()
            self.unsubscribe_fn = None

    def reset(self) -> None:
        self.last_streaming_mode = None
        self.fallback_observed = False


def _resolve_qwen_tts_pin() -> str:
    try:
        return importlib.metadata.version("qwen-tts")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_gpu_name() -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover - defensive
            return f"unknown_cuda_device ({exc!r})"
    return "cpu"


def _capture_environment() -> HarnessEnvironment:
    return HarnessEnvironment(
        qwen_tts_pin=_resolve_qwen_tts_pin(),
        torch_version=torch.__version__,
        gpu_name=_resolve_gpu_name(),
        cuda_available=torch.cuda.is_available(),
    )


def _load_input_set(path: Path, limit: Optional[int]) -> List[InputUtterance]:
    rows: List[InputUtterance] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(InputUtterance(
                utterance_id=raw["utterance_id"].strip(),
                text=raw["text"],
                text_length_chars=int(raw["text_length_chars"]),
                text_class=raw["text_class"].strip(),
                is_perceptual_difficult=(
                    raw.get("is_perceptual_difficult", "false").strip().lower()
                    == "true"
                ),
            ))
    if limit is not None and limit < len(rows):
        rows = rows[:limit]
    return rows


def _build_mock_audio_coordinator() -> AudioCoordinator:
    """Construct a real AudioCoordinator with mocked sinks so the harness
    measures dispatch + decode latency without pumping audio to devices.

    Mirrors the smoke-test rig at
    ``tests/integration/test_streaming_tts_smoke.py:93-115``.
    """
    monitor = MagicMock(spec=MonitorAudioService)
    monitor.stop_all_playback = AsyncMock(return_value=0)
    monitor.play_monitor_audio = AsyncMock()

    virtual = MagicMock(spec=VirtualMicrophoneService)
    virtual.stop_all_virtual_microphone_playback = AsyncMock(return_value=0)
    virtual.play_virtual_microphone = AsyncMock()

    coord = AudioCoordinator()
    coord._is_initialized = True
    coord.monitor_service = monitor
    coord.virtual_service = virtual
    return coord


def _maybe_apply_decoder_constants(
    chunk_size: Optional[int],
    lookahead: Optional[int],
) -> tuple[int, int]:
    """Apply --chunk-size / --lookahead overrides to the module-level
    constants (the streamer reads them at construction time inside
    ``_generate_true_stream``). Returns the (chunk_size, lookahead) the
    streamer will see.
    """
    if chunk_size is not None:
        codec_token_streamer.DEFAULT_CHUNK_SIZE = chunk_size
    if lookahead is not None:
        codec_token_streamer.DEFAULT_LOOKAHEAD = lookahead
    return (
        codec_token_streamer.DEFAULT_CHUNK_SIZE,
        codec_token_streamer.DEFAULT_LOOKAHEAD,
    )


def _resolve_mode_and_csv(
    mode_override: Optional[str],
    cuda_available: bool,
) -> tuple[StreamingMode, str, str]:
    """Return (resolved_mode, mode_requested_label, output_csv_filename).

    AC #3:
      - ``--mode-override true_stream`` on a non-CUDA host: refuse.
      - Default (no override) on CUDA: TRUE_STREAM via hardware probe.
      - Default (no override) on CPU: SENTENCE_STREAM via hardware probe.
    """
    if mode_override is None:
        resolved = default_streaming_mode_for_hardware()
        csv_name = NO_OVERRIDE_CSV_GPU if cuda_available else NO_OVERRIDE_CSV_CPU
        return resolved, resolved.value, csv_name

    requested = StreamingMode(mode_override)
    if requested == StreamingMode.TRUE_STREAM:
        if not cuda_available:
            raise SystemExit(
                "Refusing to run TRUE_STREAM on CPU — D-9 / NFR12 protection. "
                "Use --mode-override sentence_stream for the CPU baseline check."
            )
        return requested, requested.value, GPU_TRUE_STREAM_CSV
    if requested == StreamingMode.SENTENCE_STREAM:
        csv_name = (
            GPU_SENTENCE_STREAM_CSV if cuda_available else CPU_BASELINE_CSV
        )
        return requested, requested.value, csv_name
    if requested == StreamingMode.BATCH:
        return requested, requested.value, GPU_BATCH_CSV
    raise SystemExit(f"Unsupported --mode-override value: {mode_override}")


async def _dispatch_one(
    service: QwenTTSService,
    request: QwenTTSRequest,
    resolved_mode: StreamingMode,
    use_public_dispatch: bool,
) -> tuple[Optional[float], Optional[np.ndarray], int, str, Optional[str]]:
    """Run one measurement dispatch.

    Returns (first_chunk_latency_seconds, audio_data, sample_rate,
    response_mode, error_message_or_none). ``error_message_or_none`` is set
    when the dispatch raised or the response was unsuccessful; the caller
    decides whether to mark ``error_flag``.

    Per Story 16.7 AC #5 Decision (Change Log), the harness uses the
    DIRECT generator method for explicit ``--mode-override`` runs and the
    PUBLIC dispatch entry point for the no-override case (so the resolver
    is exercised and the ``streaming_mode`` metric fires for fallback
    detection).
    """
    try:
        if use_public_dispatch:
            response = await service._dispatch_by_streaming_mode(
                request, resolved_mode,
            )
        elif resolved_mode == StreamingMode.TRUE_STREAM:
            response = await service._generate_true_stream(request)
        elif resolved_mode == StreamingMode.SENTENCE_STREAM:
            response = await service._generate_streaming(request)
        else:  # BATCH
            response = await service._generate(request)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return None, None, 0, resolved_mode.value, repr(exc)

    if not response.success:
        return (
            response.first_chunk_latency,
            response.audio_data,
            response.sample_rate,
            response.mode.value if response.mode is not None else "unknown",
            response.error_message or "response.success=False",
        )

    return (
        response.first_chunk_latency,
        response.audio_data,
        response.sample_rate,
        response.mode.value if response.mode is not None else "unknown",
        None,
    )


def _classify_dispatched_mode(
    requested: str,
    response_mode: str,
    metric_recorder: StreamingModeMetricRecorder,
) -> tuple[str, bool]:
    """Pick the most-trustworthy 'mode_dispatched' string and whether a
    fallback was observed.

    The ``streaming_mode`` and ``streaming_mode_fallback`` metrics are only
    emitted from the public ``_dispatch_by_streaming_mode`` path. When the
    harness calls a private generator (``_generate_true_stream`` /
    ``_generate_streaming`` / ``_generate``) directly, no metric fires and
    a fallback CANNOT occur (no fallback chain is invoked). For those
    direct-call rows the requested mode IS the dispatched mode by
    construction.

    Story 16.7 dev-cycle bug fix (post-first-empirical-run): an earlier
    version inferred fallback whenever ``response_mode != requested``,
    but ``response.mode`` is a ``GenerationMode`` (BATCH / STREAMING)
    while ``requested`` is a ``StreamingMode`` (batch / sentence_stream /
    true_stream). The two enums use different value strings, so direct
    SENTENCE_STREAM calls compared "streaming" vs "sentence_stream" and
    every row got falsely flagged as ``fallback_occurred``. The committed
    CSVs from the first run carry that false flag; the underlying latency
    numbers are valid.
    """
    if metric_recorder.last_streaming_mode is not None:
        dispatched = metric_recorder.last_streaming_mode
    else:
        # Direct-generator call — no metric, no fallback possible. The
        # requested mode IS what dispatched.
        dispatched = requested

    fallback_inferred = metric_recorder.fallback_observed
    return dispatched, fallback_inferred


def _build_request(utterance: InputUtterance) -> QwenTTSRequest:
    """Construct a minimal QwenTTSRequest for measurement.

    Uses CUSTOM_VOICE + the bundled ``Ryan`` speaker so no voice-design
    or voice-clone setup is required; matches the production happy-path
    that an end-user with a default install would hit.
    """
    return QwenTTSRequest(
        text=utterance.text,
        language="English",
        model_type=QwenModelType.CUSTOM_VOICE,
        speaker="Ryan",
        instruct=None,
        streaming=True,
    )


def _percentile(values: List[float], p: float) -> float:
    """Inclusive linear-interpolation percentile (matches numpy default)."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _print_aggregate_summary(
    rows: List[MeasurementResult],
    mode_requested: str,
    cuda_available: bool,
) -> None:
    valid = [
        r.first_chunk_latency_seconds
        for r in rows
        if r.error_flag == "" and r.first_chunk_latency_seconds > 0
    ]
    excluded = [r for r in rows if r.error_flag != ""]
    fallback_rows = [r for r in rows if r.error_flag == "fallback_occurred"]

    print()
    print("=" * 72)
    print(f"VALIDATION SUMMARY — mode_requested={mode_requested}, "
          f"hardware={'gpu' if cuda_available else 'cpu'}")
    print("=" * 72)
    print(f"Total measurements: {len(rows)}")
    print(f"Valid (mode_dispatched matched, no error): {len(valid)}")
    print(f"Excluded: {len(excluded)} "
          f"(of which fallback_occurred: {len(fallback_rows)})")

    if not valid:
        print("No valid measurements — cannot compute percentiles.")
        return

    p50 = _percentile(valid, 50)
    p95 = _percentile(valid, 95)
    p99 = _percentile(valid, 99)
    p_max = max(valid)
    p_min = min(valid)
    p_mean = statistics.fmean(valid)
    print(f"first_chunk_latency_seconds: "
          f"min={p_min:.3f} mean={p_mean:.3f} p50={p50:.3f} "
          f"p95={p95:.3f} p99={p99:.3f} max={p_max:.3f}")

    # AC #1: explicit gate line
    if cuda_available and mode_requested == "true_stream":
        verdict = "PASS" if p95 < NFR1_CEILING_SECONDS else "FAIL"
        ceiling_msg = (
            f"under {NFR1_CEILING_SECONDS:.3f} ceiling"
            if verdict == "PASS"
            else f"exceeds {NFR1_CEILING_SECONDS:.3f} ceiling"
        )
        print(f"NFR1 GATE: p95 first-chunk latency = {p95:.3f} seconds "
              f"({verdict} — {ceiling_msg})")
    elif (not cuda_available) and mode_requested == "sentence_stream":
        # AC #3: CPU NFR1 inheritance check.
        verdict = "PASS" if p95 < NFR1_CEILING_SECONDS else "FAIL"
        suffix = (
            "inherits V2 baseline, satisfies NFR1"
            if verdict == "PASS"
            else "CPU baseline regressed; SEPARATE issue from TRUE_STREAM gate, "
                 "but blocks release"
        )
        print(f"CPU NFR1 INHERITANCE CHECK: p95 first-chunk latency on "
              f"SENTENCE_STREAM = {p95:.3f} seconds for non-trivially-short "
              f"inputs ({verdict} — {suffix})")
    else:
        print(f"NFR1 INFORMATIONAL: p95 first-chunk latency = {p95:.3f} "
              f"seconds (mode={mode_requested}, "
              f"hardware={'gpu' if cuda_available else 'cpu'})")

    if rows and len(fallback_rows) / len(rows) > 0.10:
        pct = 100.0 * len(fallback_rows) / len(rows)
        print(f"WARNING: high fallback rate ({pct:.0f}%) — TRUE_STREAM may "
              f"be structurally unstable on this host; investigate before "
              f"flipping default")
    elif fallback_rows:
        breakdown: dict[str, int] = {}
        for r in fallback_rows:
            key = f"{r.mode_requested} → {r.mode_dispatched}"
            breakdown[key] = breakdown.get(key, 0) + 1
        details = ", ".join(f"{n}× {k}" for k, n in breakdown.items())
        print(f"Excluded {len(fallback_rows)} measurements due to fallback: "
              f"{details}")


def _print_phase_aggregate_summary(rows: List[MeasurementResult]) -> None:
    """Story 16.9 AC #1: per-class per-phase p50/p95/max aggregates.

    Excludes rows with non-empty ``error_flag`` (matches Story 16.7's
    convention). Sums the four phase columns and reports the gap vs.
    ``first_chunk_latency_seconds`` so the AC #1 "phase columns sum to within
    5% of total_first_chunk_latency_seconds" sanity check is checkable from
    stdout.
    """
    valid = [r for r in rows if r.error_flag == "" and r.first_chunk_latency_seconds > 0]
    if not valid:
        print("PHASE-PROFILE: no valid rows; skipping per-phase aggregates.")
        return

    print()
    print("=" * 72)
    print("PHASE-PROFILE SUMMARY (Story 16.9 AC #1)")
    print("=" * 72)
    classes = ["short", "medium", "long"]
    phase_keys = (
        "split_seconds",
        "model_load_seconds",
        "generate_seconds",
        "decode_seconds",
        "deliver_seconds",
    )

    for cls in classes:
        cls_rows = [r for r in valid if r.text_class == cls]
        if not cls_rows:
            print(f"[{cls}] n=0 — skipped")
            continue
        print(f"[{cls}] n={len(cls_rows)}")
        for key in phase_keys:
            vals = [getattr(r, key) for r in cls_rows]
            p50 = _percentile(vals, 50)
            p95 = _percentile(vals, 95)
            p_max = max(vals)
            print(
                f"  {key:20s} p50={p50:.4f}  p95={p95:.4f}  max={p_max:.4f}"
            )
        # Sanity check: phase columns sum vs. first_chunk_latency_seconds.
        sums = [
            (
                r.split_seconds
                + r.model_load_seconds
                + r.generate_seconds
                + r.decode_seconds
                + r.deliver_seconds
            )
            for r in cls_rows
        ]
        firsts = [r.first_chunk_latency_seconds for r in cls_rows]
        gap_pcts = [
            ((s - f) / f * 100.0) if f > 0 else 0.0
            for s, f in zip(sums, firsts)
        ]
        median_gap = _percentile(gap_pcts, 50)
        max_abs_gap = max(abs(g) for g in gap_pcts)
        print(
            f"  phase-sum vs first_chunk_latency: median_gap={median_gap:+.2f}%  "
            f"max_abs_gap={max_abs_gap:.2f}% "
            f"({'within' if max_abs_gap < 5.0 else 'EXCEEDS'} 5% sanity threshold)"
        )

    # Cross-class dominant-phase identification (relative share of the sum).
    all_split = sum(r.split_seconds for r in valid)
    all_load = sum(r.model_load_seconds for r in valid)
    all_generate = sum(r.generate_seconds for r in valid)
    all_decode = sum(r.decode_seconds for r in valid)
    all_deliver = sum(r.deliver_seconds for r in valid)
    total = all_split + all_load + all_generate + all_decode + all_deliver
    if total > 0:
        print()
        print("Aggregate phase share (all classes, all valid rows):")
        for label, val in (
            ("split", all_split),
            ("model_load", all_load),
            ("generate", all_generate),
            ("decode", all_decode),
            ("deliver", all_deliver),
        ):
            pct = 100.0 * val / total
            print(f"  {label:12s} = {pct:5.1f}%  (sum={val:.3f}s)")


def _write_csv(
    rows: List[MeasurementResult],
    path: Path,
    profile_phases: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CSV_FIELDNAMES)
    if profile_phases:
        fieldnames = fieldnames + list(PHASE_PROFILE_FIELDNAMES)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    logger.info("Wrote %d rows to %s", len(rows), path)


@contextlib.contextmanager
def _profile_phases(
    service: QwenTTSService,
    profile: PhaseProfile,
) -> Iterator[None]:
    """Monkey-patch the SENTENCE_STREAM dispatch surface to record phase timings.

    Patches three call sites for the duration of one utterance dispatch:

      - ``service._split_text_for_streaming``  → phase a (``split_seconds``,
        ``split_chunk_count``, ``first_chunk_chars``).
      - ``service._generate_sync``             → phase b (first call only;
        merged with phase c per AC #1 L3 default).
      - ``service._session_registry.post_mutation`` (filtered to mutation_type
        ``'append_chunk'``) → phase d (first call only).

    Patches are bound-method shadows on the instance, so the original methods
    are restored on context exit. ``_generate_sync`` runs in the service's
    thread-pool executor (``loop.run_in_executor(self._executor,
    self._generate_sync, ...)``); the executor resolves ``self._generate_sync``
    at call time, so the instance-level shadow is visible across threads as
    long as the patch is in place when ``run_in_executor`` is invoked.

    Story 16.9 / AC #1 / Task 1 / L3 (decode merge): the harness measures
    ``generate_seconds`` as the WHOLE wallclock of the first ``_generate_sync``
    call (preprocessing + talker.generate + speech_tokenizer.decode). The
    Change Log notes the merge; ``decode_seconds`` stays 0.0 in this harness.

    ``model_load_seconds`` (added in Subtask 1.5) wraps the first
    ``ensure_model_loaded`` call. On utterance #1 it dominates (~6.5s cold
    load for the 3B model); on utterances #2..N it is a cache hit (~µs).
    Captured separately so the AC #1 phase-sum sanity check holds on row 1.
    """
    original_split = service._split_text_for_streaming
    original_generate_sync = service._generate_sync
    registry = service._session_registry  # may be None in legacy harness setups
    original_post_mutation = registry.post_mutation if registry is not None else None
    model_registry = service._model_registry
    original_ensure_model_loaded = model_registry.ensure_model_loaded

    def patched_split(text: str) -> List[str]:
        t0 = time.perf_counter()
        chunks = original_split(text)
        elapsed = time.perf_counter() - t0
        profile.split_seconds = elapsed
        profile.split_chunk_count = len(chunks)
        profile.first_chunk_chars = len(chunks[0]) if chunks else 0
        return chunks

    def patched_generate_sync(req):
        t0 = time.perf_counter()
        try:
            return original_generate_sync(req)
        finally:
            elapsed = time.perf_counter() - t0
            if not profile._first_generate_observed:
                profile.generate_seconds = elapsed
                profile._first_generate_observed = True

    def patched_post_mutation(method_name, *args):
        if (
            method_name == "append_chunk"
            and not profile._first_deliver_observed
            and original_post_mutation is not None
        ):
            t0 = time.perf_counter()
            try:
                return original_post_mutation(method_name, *args)
            finally:
                profile.deliver_seconds = time.perf_counter() - t0
                profile._first_deliver_observed = True
        if original_post_mutation is None:
            return None
        return original_post_mutation(method_name, *args)

    async def patched_ensure_model_loaded(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return await original_ensure_model_loaded(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            if not profile._first_model_load_observed:
                profile.model_load_seconds = elapsed
                profile._first_model_load_observed = True

    service._split_text_for_streaming = patched_split
    service._generate_sync = patched_generate_sync
    if registry is not None:
        registry.post_mutation = patched_post_mutation
    model_registry.ensure_model_loaded = patched_ensure_model_loaded
    try:
        yield
    finally:
        service._split_text_for_streaming = original_split
        service._generate_sync = original_generate_sync
        if registry is not None and original_post_mutation is not None:
            registry.post_mutation = original_post_mutation
        model_registry.ensure_model_loaded = original_ensure_model_loaded


def _parse_stratified_spec(spec: str) -> Tuple[int, int, int]:
    """Parse ``--stratified-sample`` value of the form ``SHORT:MEDIUM:LONG``.

    Story 16.9 AC #4 / Task 6: replaces ``[:limit]`` truncation with a
    class-aware selection so the CPU baseline run picks ≥4 short / ≥4
    medium / ≥2 long utterances from the class-ordered Story 16.7 input set.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(
            f"--stratified-sample expects SHORT:MEDIUM:LONG, got {spec!r}"
        )
    try:
        n_short, n_medium, n_long = (int(p) for p in parts)
    except ValueError as exc:
        raise SystemExit(
            f"--stratified-sample counts must be integers; got {spec!r}: {exc}"
        )
    if min(n_short, n_medium, n_long) < 0:
        raise SystemExit("--stratified-sample counts must be non-negative")
    return n_short, n_medium, n_long


def _apply_stratified_sample(
    utterances: List[InputUtterance],
    spec: Tuple[int, int, int],
) -> List[InputUtterance]:
    """Return the first N_short short + N_medium medium + N_long long rows."""
    n_short, n_medium, n_long = spec
    by_class: dict[str, List[InputUtterance]] = {"short": [], "medium": [], "long": []}
    for u in utterances:
        if u.text_class in by_class:
            by_class[u.text_class].append(u)
    sampled = (
        by_class["short"][:n_short]
        + by_class["medium"][:n_medium]
        + by_class["long"][:n_long]
    )
    return sampled


async def _run_measurements(
    service: QwenTTSService,
    utterances: List[InputUtterance],
    resolved_mode: StreamingMode,
    use_public_dispatch: bool,
    env: HarnessEnvironment,
    profile_phases: bool = False,
) -> List[MeasurementResult]:
    rows: List[MeasurementResult] = []
    recorder = StreamingModeMetricRecorder()
    recorder.attach()
    try:
        for idx, utterance in enumerate(utterances, 1):
            recorder.reset()
            request = _build_request(utterance)
            wallclock_t0 = time.perf_counter()
            wallclock_iso = datetime.now(timezone.utc).isoformat()

            phase_profile: Optional[PhaseProfile] = (
                PhaseProfile() if profile_phases else None
            )
            phase_ctx = (
                _profile_phases(service, phase_profile)
                if phase_profile is not None
                else contextlib.nullcontext()
            )

            with phase_ctx:
                first_chunk, audio, sample_rate, response_mode, error_msg = (
                    await _dispatch_one(
                        service=service,
                        request=request,
                        resolved_mode=resolved_mode,
                        use_public_dispatch=use_public_dispatch,
                    )
                )
            wallclock_dt = time.perf_counter() - wallclock_t0

            mode_dispatched, fallback_inferred = _classify_dispatched_mode(
                requested=resolved_mode.value,
                response_mode=response_mode,
                metric_recorder=recorder,
            )

            error_flag = ""
            if error_msg is not None:
                error_flag = error_msg[:200]
            elif fallback_inferred:
                error_flag = "fallback_occurred"

            audio_sample_count = (
                int(audio.size) if isinstance(audio, np.ndarray) else 0
            )
            total_audio_seconds = (
                audio_sample_count / float(sample_rate)
                if (sample_rate and audio_sample_count)
                else 0.0
            )

            row = MeasurementResult(
                utterance_id=utterance.utterance_id,
                text_length_chars=utterance.text_length_chars,
                text_class=utterance.text_class,
                mode_requested=resolved_mode.value,
                mode_dispatched=mode_dispatched,
                first_chunk_latency_seconds=(
                    float(first_chunk) if first_chunk is not None else 0.0
                ),
                total_audio_seconds=round(total_audio_seconds, 4),
                audio_sample_count=audio_sample_count,
                error_flag=error_flag,
                wallclock_timestamp=wallclock_iso,
                qwen_tts_pin=env.qwen_tts_pin,
                torch_version=env.torch_version,
                gpu_name=env.gpu_name,
                split_seconds=(
                    phase_profile.split_seconds if phase_profile is not None else 0.0
                ),
                model_load_seconds=(
                    phase_profile.model_load_seconds if phase_profile is not None else 0.0
                ),
                generate_seconds=(
                    phase_profile.generate_seconds if phase_profile is not None else 0.0
                ),
                decode_seconds=(
                    phase_profile.decode_seconds if phase_profile is not None else 0.0
                ),
                deliver_seconds=(
                    phase_profile.deliver_seconds if phase_profile is not None else 0.0
                ),
                split_chunk_count=(
                    phase_profile.split_chunk_count if phase_profile is not None else 0
                ),
                first_chunk_chars=(
                    phase_profile.first_chunk_chars if phase_profile is not None else 0
                ),
            )
            rows.append(row)
            phase_suffix = ""
            if phase_profile is not None:
                phase_suffix = (
                    f" phases=split:{phase_profile.split_seconds:.4f}s"
                    f" load:{phase_profile.model_load_seconds:.3f}s"
                    f" gen:{phase_profile.generate_seconds:.3f}s"
                    f" deliver:{phase_profile.deliver_seconds:.4f}s"
                    f" chunks={phase_profile.split_chunk_count}"
                    f" first_chunk_chars={phase_profile.first_chunk_chars}"
                )
            logger.info(
                "[%d/%d] uid=%s class=%s mode=%s first=%.3fs total_audio=%.3fs "
                "wallclock=%.3fs%s%s",
                idx, len(utterances), utterance.utterance_id,
                utterance.text_class, mode_dispatched,
                row.first_chunk_latency_seconds, total_audio_seconds,
                wallclock_dt,
                f" error={error_flag!r}" if error_flag else "",
                phase_suffix,
            )
    finally:
        recorder.detach()
    return rows


async def _amain(args: argparse.Namespace) -> int:
    env = _capture_environment()
    logger.info("Environment: cuda_available=%s gpu=%s torch=%s qwen_tts=%s",
                env.cuda_available, env.gpu_name, env.torch_version,
                env.qwen_tts_pin)

    resolved_mode, mode_label, csv_filename = _resolve_mode_and_csv(
        mode_override=args.mode_override,
        cuda_available=env.cuda_available,
    )
    logger.info("Resolved streaming_mode = %s (%s)",
                resolved_mode.name,
                "CUDA-available" if env.cuda_available else "CPU-only")

    chunk_size, lookahead = _maybe_apply_decoder_constants(
        args.chunk_size, args.lookahead,
    )
    logger.info("Decoder constants: chunk_size=%d lookahead=%d",
                chunk_size, lookahead)

    # Story 16.7: ``[:limit]`` truncation defaults to short-class-only on the
    # class-ordered input set. Story 16.9 AC #4 / Task 6 adds stratified
    # sampling so the CPU baseline run picks ≥4 short / ≥4 medium / ≥2 long.
    if args.stratified_sample is not None:
        all_utterances = _load_input_set(args.input_set, limit=None)
        spec = _parse_stratified_spec(args.stratified_sample)
        utterances = _apply_stratified_sample(all_utterances, spec)
        logger.info(
            "Stratified-sample selection: %d short + %d medium + %d long (total %d)",
            spec[0], spec[1], spec[2], len(utterances),
        )
    else:
        utterances = _load_input_set(args.input_set, args.utterance_count)
    if not utterances:
        logger.error("Input set is empty: %s", args.input_set)
        return 2
    logger.info("Loaded %d utterances from %s", len(utterances), args.input_set)

    coord = _build_mock_audio_coordinator()
    settings = AppSettings()
    registry = SessionRegistry()
    service = QwenTTSService(
        audio_coordinator=coord,
        session_registry=registry,
        app_settings=settings,
    )

    use_public_dispatch = args.mode_override is None

    try:
        ok = await service.start()
        if not ok:
            logger.error("QwenTTSService failed to start; aborting harness.")
            return 3

        # Story 16.9 AC #2 hypothesis (b) probe: apply quality-tier override
        # in-process. ``set_quality_tier`` mutates only the in-memory registry
        # state and unloads any currently-loaded model so the next
        # ``ensure_model_loaded`` call picks up the new tier. Does NOT write
        # to AppSettings on disk (per ``model_registry.py:158``).
        if args.quality_tier is not None:
            changed = await service._model_registry.set_quality_tier(args.quality_tier)
            logger.info(
                "Applied --quality-tier=%s (registry state changed: %s)",
                args.quality_tier, changed,
            )

        rows = await _run_measurements(
            service=service,
            utterances=utterances,
            resolved_mode=resolved_mode,
            use_public_dispatch=use_public_dispatch,
            env=env,
            profile_phases=args.profile_phases,
        )

        # Story 16.9: explicit --output-csv-name overrides the derived filename.
        if args.output_csv_name is not None:
            output_csv_name = args.output_csv_name
        else:
            output_csv_name = csv_filename
        output_path = args.output_dir / output_csv_name
        _write_csv(rows, output_path, profile_phases=args.profile_phases)
        _print_aggregate_summary(rows, mode_label, env.cuda_available)
        if args.profile_phases:
            _print_phase_aggregate_summary(rows)
        return 0
    finally:
        try:
            await service.stop()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.exception("service.stop() raised during teardown")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Story 16.7 — Empirical validation harness for the TRUE_STREAM "
            "streaming default."
        ),
    )
    parser.add_argument(
        "--input-set", type=Path, required=True,
        help="Path to the fixed input-set CSV "
             "(_bmad-output/implementation-artifacts/16-7-input-set.csv).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write the measurements CSV (filename is derived "
             "from --mode-override + cuda_available).",
    )
    parser.add_argument(
        "--mode-override", type=str, default=None,
        choices=["true_stream", "sentence_stream", "batch"],
        help="Explicit streaming mode (skips Story 16.2 hardware probe). "
             "Omit to exercise the production resolver.",
    )
    parser.add_argument(
        "--utterance-count", type=int, default=50,
        help="Maximum utterances to measure (default: 50).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Override codec_token_streamer.DEFAULT_CHUNK_SIZE for parameter "
             "sweeps. Default leaves the production constant in place.",
    )
    parser.add_argument(
        "--lookahead", type=int, default=None,
        help="Override codec_token_streamer.DEFAULT_LOOKAHEAD for parameter "
             "sweeps. Default leaves the production constant in place.",
    )
    # Story 16.9 AC #1 / Task 1: phase-decomposition profiling.
    parser.add_argument(
        "--profile-phases", action="store_true",
        help="Story 16.9: monkey-patch _split_text_for_streaming, "
             "_generate_sync, and registry.post_mutation to record per-phase "
             "timings (split / generate / deliver) into the output CSV. "
             "Adds 6 columns to the CSV header.",
    )
    # Story 16.9 AC #2 hypothesis (b) probe: 0.6B-small-tier comparison.
    parser.add_argument(
        "--quality-tier", type=str, default=None,
        choices=["small", "quality"],
        help="Story 16.9: apply ModelRegistry.set_quality_tier() in-process "
             "before the measurement loop. 'small' = 0.6B model; 'quality' = "
             "3B model. Mutates only the in-memory registry (no AppSettings "
             "disk write). Use to falsify the model-size penalty hypothesis.",
    )
    # Story 16.9 AC #4 / Task 6: stratified-sample CPU baseline.
    parser.add_argument(
        "--stratified-sample", type=str, default=None, metavar="SHORT:MEDIUM:LONG",
        help="Story 16.9: replace --utterance-count [:limit] truncation with "
             "class-aware selection. Picks the first SHORT short + MEDIUM "
             "medium + LONG long utterances from the input set. Example: "
             "'4:4:2' for the AC #4 CPU stratified sample.",
    )
    # Story 16.9: explicit output filename override (Tasks 2, 3.2, 5, 6).
    parser.add_argument(
        "--output-csv-name", type=str, default=None,
        help="Story 16.9: explicit CSV filename (anchored on --output-dir) so "
             "the harness produces 16-9-* artifacts without relying on the "
             "Story 16.7 derivation rules. Example: "
             "'16-9-gpu-sentence_stream-phase-profile.csv'.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Story 11.2 D-2: SessionRegistry lives on the Qt main thread, so a
    # QApplication instance must exist before the registry is constructed.
    # We do NOT spin the Qt event loop — registry post_mutation calls go
    # via QueuedConnection and are fire-and-forget for the harness's
    # latency-only measurement.
    _qapp = QApplication.instance() or QApplication([])
    _ = _qapp  # keep alive for the duration of the run

    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
