"""Story 20.1 (Epic 20, TTFA spike) - headless four-segment TTFA decomposition
harness + chunk-size sweep driver.

Drives the **production** TRUE_STREAM dispatch path
(``QwenTTSService._generate_true_stream``) with no Qt, no SessionRegistry and
no AudioCoordinator, and reconstructs the four AC #2 segments from the
wall-clock boundary metrics the spike added to the source tree:

    segment 1  prefill / prompt-encode
               ttfa_first_decode_step_ms      - ttfa_generation_start_ms
               (sub-split at ttfa_talker_thread_start_ms: MyVoice dispatch
                overhead vs. the model's own prompt encode)
    segment 2  talker time-to-(chunk_size+lookahead)-frames
               ttfa_first_chunk_emit_ms       - ttfa_first_decode_step_ms
    segment 3  first decode (codec chunk -> PCM)
               ttfa_first_decode_complete_ms  - ttfa_first_chunk_emit_ms
    segment 4  consumer-side cushion (StreamingChunkBuffer watermark)
               ttfa_first_playback_write_ms   - progressive_chunk_emit_ms[0]

Segment 4 caveat (read before quoting a number)
-----------------------------------------------
The harness stands in for ``MyVoiceApp._handle_progressive_chunk_async`` +
``AudioCoordinator.play_audio_chunk``: it marshals every AudioChunk onto the
asyncio loop with ``run_coroutine_threadsafe`` exactly as production does, and
pushes the int16 bytes through a **real** ``StreamingChunkBuffer`` configured
with the production constants. What it does NOT reproduce is (a) the PyAudio
``start_streaming_session`` device-open on chunk 0 and (b) the blocking
``stream.write`` itself. Segment 4 as reported here is therefore the
*cushion-hold* term only; the device-open term (Story 17.3 evidence estimates
~50-100 ms) is named as an unattributed residual in the evidence file rather
than folded in silently.

Usage (portable interpreter is mandatory -
memory/test_interpreter_portable_python310.md)::

    python310\\python.exe tools\\ttfa_spike_harness.py --runs 10 --utterance long --out out.csv
    python310\\python.exe tools\\ttfa_spike_harness.py --runs 5 --utterance short --chunk-size 25 --out sweep-cs25.csv

Spike hygiene (Story 20.1 AC #7): this file lives in ``tools/`` and is not
imported by anything under ``src/myvoice/``. The ``--chunk-size`` override
rebinds ``CodecTokenStreamer.__init__.__defaults__`` in-process, which is
exactly equivalent to the module-constant edit the class docstring documents
as the tuning path, but leaves no source-tree edit to revert.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# torch MUST import before anything that can pull in PyQt6 (Windows DLL-init
# invariant - memory/torch_pyqt6_dll_ordering.md). Nothing in this harness
# imports Qt, but the invariant is cheap to honour and the import graph under
# ``myvoice.services`` is large enough that assuming otherwise is unwise.
# --------------------------------------------------------------------------- #
import torch  # noqa: F401  (import-order side effect)

import argparse
import asyncio
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from myvoice.observability import metrics  # noqa: E402
from myvoice.observability.metrics import MetricRecord  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# Story 20.2 Task 3 - the priming utterance (matches
# ``QwenTTSService._COMPILE_PRIMING_TEXT``). Deliberately short: the point of
# priming is to trigger the first forward pass (inductor reload + CUDA-graph
# record), not to produce audio.
PRIMING_TEXT = "Hello world."

# Long-form: the canonical Story 17.3 section 4.1 step-3 paragraph. Every Epic
# 18 measurement (18.1 / 18.2 / 18.3 / 18.4) used this exact string, so the
# numbers produced here are directly comparable to the 5,929.4 ms Branch-A
# median.
UTTERANCE_LONG = (
    "This is a longer-form test designed to expose the difference between "
    "metric-side first-chunk emission and user-perceived first-audio latency. "
    "On the pre-Story-17.3 build, the user would wait approximately forty "
    "seconds for this utterance to start playing, even though the streaming "
    "pipeline emitted the first chunk internally at around five seconds."
)

# Short: the Clear Comms interjection class (AC #2b). Clear Comms is a
# voice-chat interjection feature (memory/clear_comms_purpose_framing.md), so
# the utterance that matters for perceived TTFA is a one-breath aside, not a
# paragraph. 33 chars / ~2 s of speech.
UTTERANCE_SHORT = "Hold on a second, say that again."

UTTERANCES = {"long": UTTERANCE_LONG, "short": UTTERANCE_SHORT}

VOICE_PROMPT_PATH = REPO_ROOT / "voice_files" / "Sarira-F.quality.pt"
VOICE_REF_TEXT_PATH = REPO_ROOT / "voice_files" / "Sarira-F.txt"

# Production consumer-side constants (audio_coordinator.py:61-62, :89-91).
WATERMARK_MS = 500
CROSSFADE_SAMPLES = 64
SAMPLE_RATE = 24000


# --------------------------------------------------------------------------- #
# Metric collection
# --------------------------------------------------------------------------- #

_TTFA_BOUNDARIES = (
    "ttfa_generation_start_ms",
    "ttfa_talker_thread_start_ms",
    "ttfa_first_decode_step_ms",
    "ttfa_first_chunk_emit_ms",
    "ttfa_first_decode_complete_ms",
    "first_chunk_latency_ms",
)


class RunCollector:
    """Captures one generation's metric stream.

    ``session_id`` is None on this path (no SessionRegistry is wired), so runs
    are separated by the harness driving them strictly sequentially and
    swapping the active collector between generations - the same discipline
    the Story 18.4 harness used with one fresh process per run, minus the
    process cost.
    """

    def __init__(self, run_index: int) -> None:
        self.run_index = run_index
        self.boundaries: Dict[str, float] = {}
        self.chunk_emit_ms: List[float] = []
        self.chunk_audio_ms: List[float] = []
        self.decode_latency_ms: List[float] = []
        self.consumer_first_release_ms: Optional[float] = None
        self.consumer_chunks_held: int = 0
        # "threshold" when the streamer's chunk_size+lookahead first-emit
        # threshold fired; "residual_flush" when the utterance was shorter
        # than one chunk window and the only token chunk was the terminal
        # residual (i.e. TRUE_STREAM degenerated to batch for that run).
        self.first_emit_path: str = "unknown"
        self.first_emit_frames: Optional[int] = None
        # A2 independent bracket. Both stamps are taken OUTSIDE the
        # metric stream: ``bracket_t0_wall_ms`` in the driver immediately
        # before the dispatch is awaited, ``bracket_first_chunk_wall_ms``
        # in the synchronous chunk trampoline on the decoder-worker
        # thread. They therefore straddle the metric-derived interval
        # from both ends and give it a falsifiable outer bound, which the
        # segment sum on its own cannot (the segments telescope).
        # Stamped with perf_counter (sub-microsecond on Win11), NOT
        # time.time (0.5 ms steps on this host): a 0.5 ms-quantised
        # bracket around a boundary pair that is itself sub-0.5 ms apart
        # collapses to exactly 0.000 and checks nothing. A different clock
        # is also what makes this independent rather than a restatement.
        self.bracket_t0_perf: Optional[float] = None
        self.bracket_first_chunk_perf: Optional[float] = None

    def __call__(self, record: MetricRecord) -> None:
        name = record.name
        if name in _TTFA_BOUNDARIES:
            # First writer wins: a boundary is one-shot per generation.
            if name not in self.boundaries:
                self.boundaries[name] = float(record.value)
                if name == "ttfa_first_chunk_emit_ms":
                    self.first_emit_path = record.tags.get("path", "threshold")
                    self.first_emit_frames = record.tags.get("frames")
        elif name == "progressive_chunk_emit_ms":
            self.chunk_emit_ms.append(float(record.value))
        elif name == "progressive_chunk_audio_duration_ms":
            self.chunk_audio_ms.append(float(record.value))
        elif name == "decode_chunk_latency_ms":
            self.decode_latency_ms.append(float(record.value))


# --------------------------------------------------------------------------- #
# Consumer stand-in
# --------------------------------------------------------------------------- #


class ConsumerSim:
    """Mirrors MyVoiceApp's progressive-playback consumer, sans PyAudio.

    Production path (app.py:2794-3090):
        worker thread -> _on_audio_chunk_ready (sync trampoline)
                      -> run_coroutine_threadsafe(_handle_progressive_chunk_async)
                      -> loop -> float32 clip/scale to int16 bytes
                      -> AudioCoordinator.play_audio_chunk
                      -> StreamingChunkBuffer.push  <-- first real work
    Everything up to and including ``push`` is reproduced verbatim here.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        from myvoice.services.streaming_chunk_buffer import StreamingChunkBuffer

        self._loop = loop
        self._buffer = StreamingChunkBuffer(
            watermark_ms=WATERMARK_MS,
            crossfade_samples=CROSSFADE_SAMPLES,
            sample_rate=SAMPLE_RATE,
            channels=1,
            sample_width=2,
        )
        self.collector: Optional[RunCollector] = None
        # Story 20.2 Task 3 - counts EVERY chunk this consumer is handed,
        # regardless of which run it belongs to. Used to prove the startup
        # priming generation reached no consumer (AC #2) on real hardware.
        self.total_chunks_seen = 0

    def reset(self, collector: RunCollector) -> None:
        self._buffer.reset()
        self.collector = collector

    def on_chunk(self, chunk: Any) -> None:
        """Sync trampoline - runs on the StreamingDecoderWorker thread."""
        self.total_chunks_seen += 1
        col = self.collector
        if (
            col is not None
            and col.bracket_first_chunk_perf is None
            and getattr(chunk, "audio_data", None) is not None
            and chunk.audio_data.size > 0
        ):
            # A2: independent observation of "first PCM reached the
            # consumer", taken before anything in this module touches
            # ``metrics``. Stamped here rather than in ``_handle`` so it is
            # not gated on event-loop scheduling.
            col.bracket_first_chunk_perf = time.perf_counter()
        try:
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            asyncio.run_coroutine_threadsafe(self._handle(chunk), loop)
        except Exception:  # pragma: no cover - defensive, mirrors app.py
            logging.getLogger(__name__).exception("chunk trampoline failed")

    async def _handle(self, chunk: Any) -> None:
        col = self.collector
        if col is None:
            return
        if chunk.audio_data is not None and chunk.audio_data.size > 0:
            audio_bytes = (
                np.clip(chunk.audio_data, -1.0, 1.0) * 32767
            ).astype(np.int16).tobytes()
            released = self._buffer.push(audio_bytes, is_final=chunk.is_final)
            if col.consumer_first_release_ms is None:
                col.consumer_chunks_held += 1
                if released:
                    col.consumer_first_release_ms = time.time() * 1000.0
            if chunk.sample_rate > 0:
                metrics.record(
                    "progressive_chunk_audio_duration_ms",
                    (chunk.audio_data.size / chunk.sample_rate) * 1000.0,
                    session_id=chunk.session_id,
                    chunk_index=chunk.chunk_index,
                )


# --------------------------------------------------------------------------- #
# Service construction
# --------------------------------------------------------------------------- #


def _apply_chunk_size(chunk_size: int) -> None:
    """Rebind the CodecTokenStreamer chunk-size default (AC #5 sweep).

    ``_generate_true_stream`` constructs ``CodecTokenStreamer()`` with no
    arguments, so the geometry comes from the ``__init__`` default arguments -
    which Python bound to the module constants at class-definition time.
    Rebinding ``__defaults__`` is therefore the runtime-equivalent of the
    module-constant edit the class docstring documents, and leaves no
    source-tree diff for Task 7.1 to revert.
    """
    from myvoice.services.tts_streaming import codec_token_streamer as cts

    cts.DEFAULT_CHUNK_SIZE = chunk_size
    cts.CodecTokenStreamer.__init__.__defaults__ = (
        chunk_size,
        cts.DEFAULT_LOOKAHEAD,
        cts.DEFAULT_QUEUE_MAX_FACTOR,
        None,
    )


def _build_settings(precision: str, compile_mode: str):
    from myvoice.models.app_settings import AppSettings

    settings = AppSettings()
    settings.tts_precision = precision
    settings.tts_compile = compile_mode
    # Force the TRUE_STREAM branch regardless of what the local settings.json
    # happens to hold; the spike measures TRUE_STREAM only.
    settings.streaming_mode_override = None
    return settings


def _load_voice_clone_prompt(service) -> list:
    """Load the Story 17.2 precomputed Sarira-F prompt and normalise it.

    Mirrors ``QwenTTSService.generate_with_saved_embedding``: normalise to the
    library's ``VoiceClonePromptItem``, move tensors to the registry device,
    and wrap in a LIST (the library folds a list into its internal dict form).
    """
    if not VOICE_PROMPT_PATH.exists():
        raise SystemExit(
            "FATAL: " + str(VOICE_PROMPT_PATH) + " missing. Launch MyVoice once "
            "with Sarira-F selected so Story 17.2's lazy precompute populates it."
        )
    raw = torch.load(str(VOICE_PROMPT_PATH), map_location="cpu", weights_only=False)
    prompt = service._normalize_voice_clone_prompt(raw)
    if not getattr(prompt, "ref_text", None) and VOICE_REF_TEXT_PATH.exists():
        prompt.ref_text = VOICE_REF_TEXT_PATH.read_text(encoding="utf-8").strip()
        prompt.icl_mode = True
    device = service._model_registry.device
    if str(device) != "cpu":
        if getattr(prompt, "ref_code", None) is not None:
            prompt.ref_code = prompt.ref_code.to(device)
        if getattr(prompt, "ref_spk_embedding", None) is not None:
            prompt.ref_spk_embedding = prompt.ref_spk_embedding.to(device)
    return [prompt]


# --------------------------------------------------------------------------- #
# Segment arithmetic
# --------------------------------------------------------------------------- #


def _segments(col: RunCollector) -> Optional[Dict[str, Any]]:
    b = col.boundaries
    need = (
        "ttfa_generation_start_ms",
        "ttfa_talker_thread_start_ms",
        "ttfa_first_decode_step_ms",
        "ttfa_first_chunk_emit_ms",
        "ttfa_first_decode_complete_ms",
    )
    if any(k not in b for k in need):
        return None
    if not col.chunk_emit_ms:
        return None

    t0 = b["ttfa_generation_start_ms"]
    t_thread = b["ttfa_talker_thread_start_ms"]
    t_step = b["ttfa_first_decode_step_ms"]
    t_emit = b["ttfa_first_chunk_emit_ms"]
    t_dec = b["ttfa_first_decode_complete_ms"]
    t_post = col.chunk_emit_ms[0]
    t_rel = col.consumer_first_release_ms

    row: Dict[str, Any] = {
        "run_index": col.run_index,
        "seg1_prefill_ms": t_step - t0,
        "seg1a_dispatch_overhead_ms": t_thread - t0,
        "seg1b_prompt_encode_ms": t_step - t_thread,
        "seg2_talker_to_first_chunk_ms": t_emit - t_step,
        "seg3_first_decode_ms": t_dec - t_emit,
        "seg4_consumer_cushion_ms": (t_rel - t_post) if t_rel is not None else None,
        "residual_post_ms": t_post - t_dec,
        "first_chunk_latency_ms": b.get("first_chunk_latency_ms"),
        "measured_t0_to_post_ms": t_post - t0,
        "chunks": len(col.chunk_emit_ms),
        "consumer_chunks_held": col.consumer_chunks_held,
        "first_emit_path": col.first_emit_path,
        "first_emit_frames": col.first_emit_frames,
    }

    # A2 - the genuinely independent check. Both stamps come from outside
    # the metric stream, so ``independent_ttfa_bracket_ms`` is an upper
    # bound the metric-derived total must fall under; ``bracket_slack_ms``
    # is the unmeasured remainder (loop scheduling + callback dispatch).
    # Unlike the segment sum, this CAN fail.
    if col.bracket_t0_perf is not None and col.bracket_first_chunk_perf is not None:
        row["independent_ttfa_bracket_ms"] = (
            col.bracket_first_chunk_perf - col.bracket_t0_perf
        ) * 1000.0
        row["bracket_slack_ms"] = (
            row["independent_ttfa_bracket_ms"] - row["measured_t0_to_post_ms"]
        )
    else:
        row["independent_ttfa_bracket_ms"] = None
        row["bracket_slack_ms"] = None

    # Steady-state producer emit/drain ratio (Story 18.1 section 4.4
    # methodology): median inter-chunk-emit interval / median chunk audio
    # duration. < 1.0 means the producer outruns playback (the OFR-E target).
    if len(col.chunk_emit_ms) >= 3 and len(col.chunk_audio_ms) >= 2:
        gaps = [
            col.chunk_emit_ms[i] - col.chunk_emit_ms[i - 1]
            for i in range(1, len(col.chunk_emit_ms))
        ]
        # Drop the last audio-duration sample: the residual flush chunk is
        # short by construction and would bias the drain-time median down.
        durations = col.chunk_audio_ms[:-1] or col.chunk_audio_ms
        row["median_inter_emit_ms"] = statistics.median(gaps)
        row["median_chunk_audio_ms"] = statistics.median(durations)
        row["producer_ratio"] = (
            row["median_inter_emit_ms"] / row["median_chunk_audio_ms"]
        )
        # Producer rate P (audio seconds emitted per wall-clock second) - the
        # variable the adaptive cushion at streaming_chunk_buffer.py:148-190
        # observes at runtime. P == 1 / producer_ratio.
        row["producer_rate_P"] = (
            1.0 / row["producer_ratio"] if row["producer_ratio"] else None
        )
    else:
        row["median_inter_emit_ms"] = None
        row["median_chunk_audio_ms"] = None
        row["producer_ratio"] = None
        row["producer_rate_P"] = None

    total = (
        row["seg1_prefill_ms"]
        + row["seg2_talker_to_first_chunk_ms"]
        + row["seg3_first_decode_ms"]
        + row["residual_post_ms"]
    )
    row["segment_sum_to_post_ms"] = total
    row["reconcile_error_pct"] = (
        100.0 * (total - row["measured_t0_to_post_ms"]) / row["measured_t0_to_post_ms"]
        if row["measured_t0_to_post_ms"]
        else None
    )
    # Total audio duration of the generation - T_a for the AC #2b Phase 2
    # break-even solve.
    row["total_audio_ms"] = sum(col.chunk_audio_ms) if col.chunk_audio_ms else None
    return row


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


async def _run(args) -> int:
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    if args.chunk_size is not None:
        _apply_chunk_size(args.chunk_size)
    else:
        from myvoice.services.tts_streaming import codec_token_streamer as _cts
        args.chunk_size = _cts.DEFAULT_CHUNK_SIZE

    settings = _build_settings(args.precision, args.compile)
    service = QwenTTSService(
        audio_coordinator=None,
        device="auto",
        quality_tier="quality",
        session_registry=None,
        app_settings=settings,
    )
    ok = await service.start()
    if not ok:
        print("FATAL: QwenTTSService.start() returned False", file=sys.stderr)
        return 1

    loop = asyncio.get_running_loop()
    consumer = ConsumerSim(loop)
    service.set_audio_chunk_ready_callback(consumer.on_chunk)

    prompt = _load_voice_clone_prompt(service)
    text = UTTERANCES[args.utterance]

    # Story 20.2 Task 3 - startup compile priming. Stands in for
    # ``warmup_compile_async``'s warm-cache branch: one short generation
    # dispatched with ``suppress_audio_output=True`` BEFORE the measured
    # runs, so PyTorch's lazy inductor-cache reload + CUDA-graph record is
    # paid here rather than on the first measured (i.e. user-facing)
    # generation. The BASE model + the same voice_clone_prompt are used so
    # the primed graph is the one the measured run exercises.
    if getattr(args, "prime", False):
        t_prime = time.time()
        prime_req = QwenTTSRequest(
            text=PRIMING_TEXT,
            language="English",
            model_type=QwenModelType.BASE,
            streaming=True,
            voice_clone_prompt=prompt,
            suppress_audio_output=True,
        )
        prime_resp = await service._generate_true_stream(prime_req)
        prime_ms = (time.time() - t_prime) * 1000.0
        await asyncio.sleep(0.15)
        print("  startup priming: {ok} in {ms:.0f}ms; consumer chunks seen "
              "during priming = {n}".format(
                  ok="ok" if prime_resp.success else "FAILED",
                  ms=prime_ms, n=consumer.total_chunks_seen))

    rows: List[Dict[str, Any]] = []
    total_runs = args.runs + args.warmup
    try:
        rows = await _run_cells(service, consumer, prompt, text, args, total_runs)
    finally:
        # C5: ``service.stop()`` must run even if a mid-sweep generation
        # raises, and the rows captured before the failure must survive to
        # the CSV write below.
        await service.stop()

    if not rows:
        print("FATAL: no complete runs captured", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nWrote " + str(len(rows)) + " rows -> " + str(out))

    warm = [r for r in rows if not r["is_warmup"]]
    if warm:
        print(json.dumps(_summarise(warm), indent=2))
    return 0


async def _run_cells(service, consumer, prompt, text, args, total_runs):
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest

    rows: List[Dict[str, Any]] = []
    for i in range(total_runs):
        col = RunCollector(run_index=i - args.warmup)
        consumer.reset(col)
        unsub = metrics.add_listener(col)
        try:
            request = QwenTTSRequest(
                text=text,
                language="English",
                model_type=QwenModelType.BASE,
                streaming=True,
                voice_clone_prompt=prompt,
            )
            t_wall = time.time()
            col.bracket_t0_perf = time.perf_counter()
            resp = await service._generate_true_stream(request)
            wall_ms = (time.time() - t_wall) * 1000.0
            # C5: drain BEFORE unsubscribing. The terminal chunk's
            # ``progressive_chunk_audio_duration_ms`` can still be in flight
            # on the event loop here, and that metric sums into T_a - the
            # input to the AC #2b Phase 2 break-even derivation. Dropping it
            # silently shortened T_a on every run of the first capture pass.
            await asyncio.sleep(0.15)
        finally:
            unsub()

        label = "warmup" if i < args.warmup else "run " + str(i - args.warmup + 1)
        if not resp.success:
            print("  " + label + ": FAILED - " + str(resp.error_message),
                  file=sys.stderr)
            continue
        seg = _segments(col)
        if seg is None:
            print("  " + label + ": incomplete boundary set "
                  + str(sorted(col.boundaries)), file=sys.stderr)
            continue
        seg["generation_wall_ms"] = wall_ms
        seg["is_warmup"] = i < args.warmup
        seg["chunk_size"] = args.chunk_size
        seg["utterance"] = args.utterance
        seg["precision"] = args.precision
        seg["compile"] = args.compile
        rows.append(seg)
        s4 = seg["seg4_consumer_cushion_ms"]
        ratio = seg["producer_ratio"]
        brk = seg["independent_ttfa_bracket_ms"]
        print(
            "  {lab}: TTFA(post)={ttfa:.0f}ms [s1={s1:.0f} s2={s2:.0f} "
            "s3={s3:.0f} s4={s4} res={res:.0f}] bracket={brk} "
            "path={path} ratio={ratio} chunks={n} gen={gen:.0f}ms".format(
                lab=label,
                ttfa=seg["measured_t0_to_post_ms"],
                s1=seg["seg1_prefill_ms"],
                s2=seg["seg2_talker_to_first_chunk_ms"],
                s3=seg["seg3_first_decode_ms"],
                s4="n/a" if s4 is None else round(s4),
                res=seg["residual_post_ms"],
                brk="n/a" if brk is None else round(brk),
                path=seg["first_emit_path"],
                ratio="n/a" if ratio is None else round(ratio, 3),
                n=seg["chunks"],
                gen=wall_ms,
            )
        )
    return rows


def quantile(vals: List[float], q: float) -> Optional[float]:
    """Linear-interpolated quantile (the numpy / R type-7 definition).

    Review finding A1: the original implementation used
    ``sorted(v)[round(q * (n - 1))]``, which lands on ``n - 1`` for every
    n <= 10 at q = 0.95 - i.e. it reported the **maximum** under a p95
    label. This interpolates instead, and callers report ``max`` alongside
    it so the distinction is visible rather than implied.
    """
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def stat(key: str) -> Optional[Dict[str, float]]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        vals_sorted = sorted(vals)
        return {
            "n": len(vals),
            "median": statistics.median(vals),
            "p95_interpolated": quantile(vals, 0.95),
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
        }

    keys = (
        "seg1_prefill_ms",
        "seg1a_dispatch_overhead_ms",
        "seg1b_prompt_encode_ms",
        "seg2_talker_to_first_chunk_ms",
        "seg3_first_decode_ms",
        "seg4_consumer_cushion_ms",
        "residual_post_ms",
        "measured_t0_to_post_ms",
        "first_chunk_latency_ms",
        "reconcile_error_pct",
        "independent_ttfa_bracket_ms",
        "bracket_slack_ms",
        "producer_ratio",
        "producer_rate_P",
        "total_audio_ms",
        "generation_wall_ms",
    )
    return {k: stat(k) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser(description="Story 20.1 TTFA decomposition harness")
    ap.add_argument("--runs", type=int, default=10, help="measured runs (warm)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded leading runs (cold compile / cuDNN autotune)")
    ap.add_argument("--utterance", choices=sorted(UTTERANCES), default="long")
    # Story 20.4: default is now "whatever the streamer module actually
    # commits", not a literal. Before the retune this file's own default of
    # 25 happened to agree with the module constant; after it, a hard-coded
    # default would have silently measured the OLD geometry while claiming
    # to measure current code. ``None`` -> do not rebind anything.
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="override CodecTokenStreamer.DEFAULT_CHUNK_SIZE for "
                         "this run; omit to measure the committed geometry")
    ap.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    ap.add_argument("--compile", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--prime", action="store_true",
                    help="Story 20.2: run one suppressed priming generation "
                         "at startup before the measured runs (stands in for "
                         "warmup_compile_async's warm-cache branch)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    for noisy in ("myvoice", "myvoice.metrics", "transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    print(
        "TTFA harness: utterance={u} chunk_size={c} lookahead=5 precision={p} "
        "compile={k} runs={r} (+{w} warmup)".format(
            u=args.utterance,
            c="committed" if args.chunk_size is None else args.chunk_size,
            p=args.precision, k=args.compile, r=args.runs, w=args.warmup,
        )
    )
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print("  device={n} vram={v:.1f}GiB cap={a}.{b} torch={t}".format(
            n=pr.name, v=pr.total_memory / 1024 ** 3, a=pr.major, b=pr.minor,
            t=torch.__version__,
        ))
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
