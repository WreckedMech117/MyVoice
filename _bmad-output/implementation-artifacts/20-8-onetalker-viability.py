"""Story 20.8 — VIABILITY CHECK for the one-talker-run-per-pair audition design.

Commander approved running this BEFORE any candidate selection or fixture
building, because it decides between a ~20-minute audition and a ~3.5-hour one.

It tests THREE claims, not two. The third is the control without which the
other two cannot be read.

  (b) TOKEN INVARIANCE — the talker's emitted codec-token stream does not
      depend on chunk_size. Tested DIRECTLY: fixed seed, same prompt, same
      text, live at cs25 vs live at cs7 vs live at cs10, compared for
      identity. Not by code-reading.

  (c) SEEDED DETERMINISM (the control for (b)) — the pipeline reproduces its
      own token stream at a fixed seed AT ALL. Two live cs25 runs at the same
      seed. If this fails, a cs25-vs-cs7 mismatch says nothing about chunk
      size, and (b) is unanswerable rather than false. Naming it separately is
      the difference between "chunk size perturbs the talker" and "this
      pipeline is not seed-reproducible".

  (a) RENDER FIDELITY — re-chunking a captured token stream OFFLINE and
      rendering it reproduces what a LIVE run at that geometry produces,
      bit-for-bit, through the REAL CodecTokenStreamer + REAL
      StreamingDecoderWorker + REAL StreamingChunkBuffer. Nothing here is a
      reimplementation: the objects are constructed in the same order, with
      the same arguments, as ``_generate_true_stream`` constructs them
      (qwen_tts_service.py:5281-5331, :5437-5444).

      (a) is tested in its CROSS-GEOMETRY form — offline re-chunk of a cs25
      capture at cs7, against a live cs7 run — because that is the form the
      audition actually needs. The same-geometry form (offline cs25 vs live
      cs25) is also reported: it isolates "offline vs live" from
      "re-chunking", so a failure can be attributed rather than guessed at.

  (a-control) DECODE DETERMINISM — the same offline render run twice. If the
      decoder is not bit-reproducible against itself, bit-exactness against a
      live run is unattainable and the bar has to be restated rather than
      failed. Symmetric with (c).

Why (a) alone is not enough: if (a) holds and (b) does not, re-chunking a cs25
capture at cs7 yields audio no real cs7 run would produce — a fiction that
would audition clean and then not match what ships.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-8-onetalker-viability.py

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttfa_spike_harness as H  # noqa: E402

ART = Path(__file__).resolve().parent
SEEDS = (1234, 99991)
LIVE_POINTS = (25, 25, 7, 10)      # two cs25 runs: the (c) control
OFFLINE_POINTS = (25, 7, 10)
SR = 24000


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------- #
# comparison helpers — quantified, never rounded up to "close enough"
# --------------------------------------------------------------------- #

def cmp_tokens(a, b):
    if a is None or b is None:
        return {"identical": False, "why": "one side missing"}
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        pre = 0
        if n:
            rows_eq = (a[:n] == b[:n]).all(dim=1)
            pre = int(rows_eq.cumprod(0).sum().item())
        return {"identical": False, "why": "shape",
                "shape_a": list(a.shape), "shape_b": list(b.shape),
                "identical_prefix_frames": pre}
    eq = torch.equal(a, b)
    out = {"identical": bool(eq), "shape": list(a.shape)}
    if not eq:
        rows = (a != b).any(dim=1)
        out["differing_frames"] = int(rows.sum().item())
        out["total_frames"] = int(a.shape[0])
        out["first_differing_frame"] = int(rows.nonzero()[0].item())
        out["identical_prefix_frames"] = out["first_differing_frame"]
    return out


def cmp_audio(a, b):
    out = {"len_a": int(a.size), "len_b": int(b.size)}
    if a.size == b.size and a.dtype == b.dtype:
        out["bit_exact"] = bool(np.array_equal(a, b))
        if not out["bit_exact"]:
            fa = a.astype(np.float64)
            fb = b.astype(np.float64)
            d = np.abs(fa - fb)
            nz = np.nonzero(d)[0]
            out["differing_samples"] = int(nz.size)
            out["differing_pct"] = round(100.0 * nz.size / max(a.size, 1), 4)
            out["first_differing_sample"] = int(nz[0]) if nz.size else None
            out["max_abs_diff"] = float(d.max())
            denom = float(np.sqrt((fa ** 2).mean())) or 1.0
            out["nrmse"] = float(np.sqrt((d ** 2).mean()) / denom)
    else:
        out["bit_exact"] = False
        out["why"] = "length or dtype mismatch"
    return out


def to_int16_bytes(pcm):
    """The exact clip/scale/cast app.py applies before the buffer."""
    return (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def through_real_buffer(pcm_chunks, crossfade_samples):
    """Push posted PCM through a REAL StreamingChunkBuffer.

    ``crossfade_samples`` mirrors app.py:3242-3247 rather than the harness's
    64-sample constant: Story 20.5 Phase 4 scoped the consumer cross-fade to
    DISCONTINUOUS producers, so on the state-carrying path production passes
    **0**. Using 64 here would compare two blends nobody ships.
    """
    from myvoice.services.streaming_chunk_buffer import StreamingChunkBuffer
    buf = StreamingChunkBuffer(
        watermark_ms=H.WATERMARK_MS, crossfade_samples=crossfade_samples,
        sample_rate=SR, channels=1, sample_width=2,
    )
    out = []
    for seg in pcm_chunks:
        for released in buf.push(to_int16_bytes(seg), is_final=False):
            out.append(released)
    for released in buf.flush_remaining():
        out.append(released)
    return np.frombuffer(b"".join(out), dtype=np.int16)


# --------------------------------------------------------------------- #
# offline render — REAL streamer, REAL worker, REAL buffer, production order
# --------------------------------------------------------------------- #

def rechunk(flat, n):
    return [flat[i:i + n].clone() for i in range(0, flat.shape[0], n)]


def offline_render(service, model, flat_tokens, chunk_size):
    from myvoice.services.tts_streaming import (
        CodecTokenStreamer, END_OF_STREAM, StreamingDecoderWorker,
    )
    from myvoice.services.tts_streaming import codec_token_streamer as cts

    streamer = CodecTokenStreamer(
        chunk_size=chunk_size, lookahead=cts.DEFAULT_LOOKAHEAD)
    decode_fn = service._build_true_stream_decode_fn(
        model, chunk_size=streamer.chunk_size, lookahead=streamer.lookahead)
    carries = bool(getattr(decode_fn, "carries_codec_state", False))
    if not carries:
        raise SystemExit(
            "FATAL: offline decode_fn is not state-carrying; this is not the "
            "shipping path and no comparison against it is meaningful.")
    streamer.apply_codec_state_geometry(carries)

    posted = []
    done = threading.Event()

    def post(method, session_id, *args):
        if method == "append_chunk":
            posted.append(np.asarray(args[0], dtype=np.float32).copy())
        elif method in ("finalize", "cancel", "discard"):
            done.set()

    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=decode_fn, post_mutation=post,
        session_id="offline-viability", model_type="qwen3_tts", hardware="gpu",
    )
    # Feed from a separate thread AFTER the worker starts, exactly as
    # production does (the worker starts, then the talker thread pushes).
    # Filling first deadlocks: the streamer's queue is bounded at
    # ``queue_max_factor * chunk_size`` = 28 at cs7, and a 244-frame utterance
    # re-chunked at 7 is 35 chunks — ``put`` blocks forever on a worker that
    # has not started. Story 20.5's fixture renderer filled first and never
    # hit it because cs25 gives maxsize 100 against 10 chunks.
    chunks = rechunk(flat_tokens, chunk_size)

    def feed():
        for c in chunks:
            streamer.queue.put(c)
        streamer.queue.put(END_OF_STREAM)

    worker.start()
    feeder = threading.Thread(target=feed, daemon=True, name="offline-feeder")
    feeder.start()
    feeder.join(timeout=600.0)
    worker.join(timeout=600.0)
    done.wait(timeout=10.0)
    if feeder.is_alive() or worker._thread.is_alive():
        raise SystemExit(
            "FATAL: offline render did not drain at cs{} "
            "(feeder alive={}, worker alive={})".format(
                chunk_size, feeder.is_alive(), worker._thread.is_alive()))
    return posted


# --------------------------------------------------------------------- #
# live run — production dispatch, with a token tap that PRESERVES the
# decode_fn's declared attributes
# --------------------------------------------------------------------- #

async def live_run(service, prompt, text, chunk_size, seed):
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest

    H._apply_chunk_size(chunk_size)
    captured = []
    posted = []
    real_builder = service._build_true_stream_decode_fn

    def tapping_builder(model, *a, **kw):
        inner = real_builder(model, *a, **kw)

        def _tap(chunk):
            captured.append(torch.as_tensor(chunk).detach().cpu().clone())
            return inner(chunk)

        # Carry the declarations forward. Without these the worker's geometry
        # guard sees a plain function, the dispatch reads
        # carries_codec_state=False, the lookahead is NOT retired, and the run
        # would silently measure the pre-20.6 geometry.
        for attr in ("carries_codec_state", "_window_frames"):
            if hasattr(inner, attr):
                setattr(_tap, attr, getattr(inner, attr))
        return _tap

    def on_chunk(ch):
        data = getattr(ch, "audio_data", None)
        if data is not None and getattr(data, "size", 0):
            posted.append(np.asarray(data, dtype=np.float32).copy())

    service._build_true_stream_decode_fn = tapping_builder
    service.set_audio_chunk_ready_callback(on_chunk)
    try:
        seed_everything(seed)
        resp = await service._generate_true_stream(QwenTTSRequest(
            text=text, language="English", model_type=QwenModelType.BASE,
            streaming=True, voice_clone_prompt=prompt))
        await asyncio.sleep(0.25)
    finally:
        service._build_true_stream_decode_fn = real_builder
        service.set_audio_chunk_ready_callback(None)
    if not resp.success or not captured:
        raise SystemExit("FATAL: live run failed at cs{} seed {}: {}".format(
            chunk_size, seed, getattr(resp, "error_message", None)))
    return {"chunks": captured, "flat": torch.cat(captured, dim=0),
            "posted": posted}


def cat(chunks):
    return (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32))


async def _run():
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings)
    if not await service.start():
        print("FATAL: service.start() returned False", file=sys.stderr)
        return 1

    report = {"host": {}, "seeds": {}}
    try:
        pr = torch.cuda.get_device_properties(0)
        report["host"] = {
            "device": pr.name, "torch": torch.__version__,
            "capability": "{}.{}".format(pr.major, pr.minor),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        }
        print("  host: " + json.dumps(report["host"]), flush=True)

        prompt = H._load_voice_clone_prompt(service)
        text = H.UTTERANCE_LONG

        await service._generate_true_stream(QwenTTSRequest(
            text=H.PRIMING_TEXT, language="English",
            model_type=QwenModelType.BASE, streaming=True,
            voice_clone_prompt=prompt, suppress_audio_output=True))
        await asyncio.sleep(0.3)
        print("  priming done", flush=True)

        model = service._model_registry.get_loaded_model()

        for seed in SEEDS:
            print("\n=== seed {} ===".format(seed), flush=True)
            live = {}
            for i, cs in enumerate(LIVE_POINTS):
                key = "cs{}#{}".format(cs, i)
                t0 = time.time()
                live[key] = await live_run(service, prompt, text, cs, seed)
                print("  live {:<8} {:>4d} frames  {:>3d} chunks  {:.1f}s"
                      .format(key, live[key]["flat"].shape[0],
                              len(live[key]["chunks"]), time.time() - t0),
                      flush=True)

            a25, b25 = live["cs25#0"], live["cs25#1"]
            r7, r10 = live["cs7#2"], live["cs10#3"]

            s = {"tokens": {}, "audio": {}}
            s["tokens"]["c_seeded_determinism_cs25_vs_cs25"] = cmp_tokens(
                a25["flat"], b25["flat"])
            s["tokens"]["b_invariance_cs25_vs_cs7"] = cmp_tokens(
                a25["flat"], r7["flat"])
            s["tokens"]["b_invariance_cs25_vs_cs10"] = cmp_tokens(
                a25["flat"], r10["flat"])

            resplit = rechunk(a25["flat"], 25)
            s["tokens"]["resplit_reproduces_capture"] = bool(
                len(resplit) == len(a25["chunks"]) and
                all(torch.equal(x, y) for x, y in zip(resplit, a25["chunks"])))

            off = {}
            for cs in OFFLINE_POINTS:
                off[cs] = [offline_render(service, model, a25["flat"], cs),
                           offline_render(service, model, a25["flat"], cs)]
                print("  offline cs{:<3} {:>3d} chunks (x2)".format(
                    cs, len(off[cs][0])), flush=True)

            for cs in OFFLINE_POINTS:
                s["audio"]["a_control_decode_determinism_cs{}".format(cs)] = \
                    cmp_audio(cat(off[cs][0]), cat(off[cs][1]))

            # app.py:3242-3247 — ask the producer, do not assume.
            xfade = (0 if getattr(service, "progressive_stream_is_continuous",
                                  False) else H.CROSSFADE_SAMPLES)
            s["audio"]["consumer_crossfade_samples_used"] = xfade

            pairs = ((25, a25, "same_geometry"), (7, r7, "cross_geometry"),
                     (10, r10, "cross_geometry"))
            for cs, liverun, kind in pairs:
                s["audio"]["a_{}_pcm_cs{}".format(kind, cs)] = cmp_audio(
                    cat(off[cs][0]), cat(liverun["posted"]))
                s["audio"]["a_{}_buffer_bytes_cs{}".format(kind, cs)] = \
                    cmp_audio(through_real_buffer(off[cs][0], xfade),
                              through_real_buffer(liverun["posted"], xfade))
            report["seeds"][str(seed)] = s
            print(json.dumps(s, indent=2), flush=True)
    finally:
        await service.stop()

    out = ART / "20-8-onetalker-viability.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote " + out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
