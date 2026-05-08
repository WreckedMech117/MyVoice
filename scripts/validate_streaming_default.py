#!/usr/bin/env python
"""Story 16.7 — Empirical validation harness for the TRUE_STREAM streaming default.

Measures first-audio latency under the production Qwen3-TTS dispatch path on the
maintainer's GPU host (RTX 5090 Blackwell + Win11 + torch 2.10+cu128 per
``memory/hardware_setup.md``) and on a CPU host as the NFR1 inheritance check.

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
from typing import Callable, List, Optional  # noqa: E402
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

# CSV columns per AC #1.
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


@dataclass
class MeasurementResult:
    """One row of the measurement CSV (mirrors AC #1 header exactly)."""
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


def _write_csv(rows: List[MeasurementResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    logger.info("Wrote %d rows to %s", len(rows), path)


async def _run_measurements(
    service: QwenTTSService,
    utterances: List[InputUtterance],
    resolved_mode: StreamingMode,
    use_public_dispatch: bool,
    env: HarnessEnvironment,
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
            )
            rows.append(row)
            logger.info(
                "[%d/%d] uid=%s class=%s mode=%s first=%.3fs total_audio=%.3fs "
                "wallclock=%.3fs%s",
                idx, len(utterances), utterance.utterance_id,
                utterance.text_class, mode_dispatched,
                row.first_chunk_latency_seconds, total_audio_seconds,
                wallclock_dt,
                f" error={error_flag!r}" if error_flag else "",
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

        rows = await _run_measurements(
            service=service,
            utterances=utterances,
            resolved_mode=resolved_mode,
            use_public_dispatch=use_public_dispatch,
            env=env,
        )

        output_path = args.output_dir / csv_filename
        _write_csv(rows, output_path)
        _print_aggregate_summary(rows, mode_label, env.cuda_available)
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
