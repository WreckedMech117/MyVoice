"""Story 20.6 Task 4 (AC #3, partial) — the producer-side half, headless.

WHAT THIS MEASURES, AND WHY IT IS NOT THE GUI CAPTURE
-----------------------------------------------------
AC #3's TTFA number has to come through the shipped GUI, because that is the
only place the consumer-side cushion is real. This bench measures the two
things that do NOT need a human at the keyboard and that the GUI capture
cannot attribute on its own:

  1. **Per-chunk decode time, single-pass vs two-pass.** Story 20.5 measured
     the lookahead's snapshot/restore second pass at +7-10 ms/chunk. Retiring
     the lookahead makes ``window_frames == commit_frames``, so
     ``StatefulCodecDecoder.__call__`` takes the ``commit = n_frames`` branch
     and never snapshots. This scores that recovery on the REAL decoder, on
     the shipping precision, over the real token sequences captured for Story
     20.5 Phase 1 — the same tokens, so the number is not confounded by a new
     talker draw.

  2. **The producer's first-emit threshold.** The streamer's first chunk needs
     ``chunk_size + lookahead`` frames; retiring the lookahead makes that
     ``chunk_size``. That is five fewer talker steps before ANY audio can be
     decoded, and it is the mechanism behind the TTFA claim the GUI capture
     tests. The saving is reported here in talker steps and in measured
     decode-side terms; converting it to milliseconds is the GUI capture's job.

It also re-checks, on the real model rather than the CPU reference decoder,
that the two geometries produce the SAME posted stream — the claim that makes
the AC #4 audition a test of a specific prediction rather than of an unknown.

NOT MEASURED HERE: the emit/drain ratio and the OFR-E gate. Those are
consumer-side and come out of the same CSV capture as the GUI TTFA run; see
the operator hand-off.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-6-lookahead-bench.py

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TOK_DIR = SCRIPT_DIR / "20-5-tokens"
OUT_JSON = SCRIPT_DIR / "20-6-lookahead-bench.json"

sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttfa_spike_harness as H  # noqa: E402

from myvoice.services.tts_streaming import codec_state_cache as csc  # noqa: E402
from myvoice.services.tts_streaming import codec_token_streamer  # noqa: E402
from myvoice.services.tts_streaming.streaming_decoder import (  # noqa: E402
    _CODEC_EDGE_LOSS_SAMPLES as EDGE,
    _CODEC_SAMPLES_PER_FRAME as SPF,
)

CS = codec_token_streamer.DEFAULT_CHUNK_SIZE
LA = codec_token_streamer.DEFAULT_LOOKAHEAD
REPS = 5


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


async def _load():
    """Model + real decoder, reusing Story 20.5's captured tokens so the two
    arms score the same token sequence and no talker draw enters the number."""
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSService

    cached = sorted(TOK_DIR.glob("*.npz"))
    if not cached:
        raise SystemExit(
            "FATAL: no captured tokens in {}. Run 20-5-state-cache-bench.py "
            "first — this bench deliberately does not draw new tokens."
            .format(TOK_DIR))

    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings,
    )
    if not await service.start():
        raise SystemExit("FATAL: service.start() returned False")
    ok, err = await service._model_registry.ensure_model_loaded(QwenModelType.BASE)
    if not ok:
        raise SystemExit("model load failed: {}".format(err))
    model = service._model_registry.get_loaded_model()
    decoder = model.model.speech_tokenizer.model.decoder
    tokens = {
        p.stem: torch.from_numpy(np.load(p)["tokens"]).long() for p in cached
    }
    return service, decoder, tokens


def _slice(frames, window, stride):
    """The streamer's own chunking: fixed windows at a fixed stride, then the
    residual flush. ``window == stride`` is the retired geometry."""
    chunks = []
    index = 0
    n = frames.shape[0]
    while index + window <= n:
        chunks.append(frames[index:index + window])
        index += stride
    if index < n:
        chunks.append(frames[index:])
    return chunks


def _run_arm(decoder, geometry, frames, window, device):
    """Decode one utterance at one geometry, timing every chunk.

    Returns ``(posted_stream, per_chunk_ms, n_chunks)`` where ``posted_stream``
    is what ``StreamingDecoderWorker`` would have posted: the full decode when
    the lookahead is retired, the splice-trimmed head when it is not.
    """
    fn = csc.StatefulCodecDecoder(
        decoder=decoder, geometry=geometry,
        commit_frames=CS, window_frames=window, device=device,
    )
    chunks = _slice(frames, window, CS)
    posted = []
    per_chunk_ms = []
    for i, chunk in enumerate(chunks):
        _sync()
        t0 = time.perf_counter()
        pcm = fn(chunk)
        _sync()
        per_chunk_ms.append((time.perf_counter() - t0) * 1000.0)

        full_window = chunk.shape[0] >= window and window > CS
        if full_window:
            splice = CS * SPF - (EDGE if i == 0 else 0)
            posted.append(pcm[:splice])
        else:
            posted.append(pcm)
    return np.concatenate(posted), per_chunk_ms, len(chunks)


def _nrmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    denom = float(np.linalg.norm(a[:n]))
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(a[:n] - b[:n]) / denom)


def main() -> int:
    print("=" * 78)
    print("Story 20.6 Task 4 — retiring the lookahead: producer-side bench")
    print("=" * 78)
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print("device={} vram={:.1f}GiB torch={}".format(
            pr.name, pr.total_memory / 1024 ** 3, torch.__version__))

    service, decoder, tokens = asyncio.run(_load())
    try:
        geometry = csc.probe_decoder(decoder)
        device = next(decoder.parameters()).device
        dtype = next(decoder.parameters()).dtype
        print("decoder: device={} dtype={} samples/frame={} edge_loss={}"
              .format(device, dtype, geometry.samples_per_frame,
                      geometry.edge_loss_samples))
        print("geometry: chunk_size={}  reference lookahead={}  candidate "
              "lookahead={} (retired)".format(CS, LA, 0))
        print("first-emit threshold: {} frames -> {} frames "
              "({} fewer talker steps)".format(CS + LA, CS, LA))

        results = {
            "_meta": {
                "story": "20.6",
                "chunk_size": CS,
                "reference_lookahead": LA,
                "candidate_lookahead": 0,
                "reps": REPS,
                "device": str(device),
                "dtype": str(dtype),
                "tokens": "Story 20.5 Phase 1 capture (reused, not redrawn)",
            },
            "utterances": {},
        }

        for utt_id, frames in sorted(tokens.items()):
            print("\n--- {} ({} frames) ---".format(utt_id, frames.shape[0]))
            ref_ms, cand_ms = [], []
            ref_stream = cand_stream = None
            ref_chunks = cand_chunks = 0
            for rep in range(REPS + 1):  # rep 0 is a discarded warm-up
                r_stream, r_ms, r_n = _run_arm(
                    decoder, geometry, frames, CS + LA, device)
                c_stream, c_ms, c_n = _run_arm(
                    decoder, geometry, frames, CS, device)
                if rep == 0:
                    continue
                ref_ms.extend(r_ms)
                cand_ms.extend(c_ms)
                ref_stream, cand_stream = r_stream, c_stream
                ref_chunks, cand_chunks = r_n, c_n

            ref_med = statistics.median(ref_ms)
            cand_med = statistics.median(cand_ms)
            row = {
                "frames": int(frames.shape[0]),
                "reference_chunks": ref_chunks,
                "candidate_chunks": cand_chunks,
                "reference_per_chunk_ms_median": round(ref_med, 3),
                "candidate_per_chunk_ms_median": round(cand_med, 3),
                "per_chunk_saving_ms": round(ref_med - cand_med, 3),
                "reference_total_decode_ms": round(sum(ref_ms) / REPS, 2),
                "candidate_total_decode_ms": round(sum(cand_ms) / REPS, 2),
                "posted_samples_reference": int(ref_stream.size),
                "posted_samples_candidate": int(cand_stream.size),
                "posted_nrmse": round(_nrmse(ref_stream, cand_stream), 8),
                "whole_sequence_samples": int(
                    SPF * frames.shape[0] - EDGE),
            }
            results["utterances"][utt_id] = row
            print("  per-chunk median   reference {:.3f} ms   candidate "
                  "{:.3f} ms   saving {:+.3f} ms".format(
                      ref_med, cand_med, ref_med - cand_med))
            print("  chunks             reference {}   candidate {}".format(
                ref_chunks, cand_chunks))
            print("  posted samples     reference {}   candidate {}   "
                  "(whole-sequence decode {})".format(
                      ref_stream.size, cand_stream.size,
                      row["whole_sequence_samples"]))
            print("  posted NRMSE (candidate vs reference): {:.3e}".format(
                row["posted_nrmse"]))
            if ref_stream.size != cand_stream.size:
                print("  WARNING: the two arms posted different lengths. The "
                      "retirement is supposed to change the WORK, not the "
                      "AUDIO — investigate before the audition.")

        savings = [r["per_chunk_saving_ms"]
                   for r in results["utterances"].values()]
        results["_meta"]["per_chunk_saving_ms_median"] = round(
            statistics.median(savings), 3)
        print("\n" + "=" * 78)
        print("median per-chunk saving across utterances: {:+.3f} ms".format(
            statistics.median(savings)))
        print("Story 20.5 measured the two-pass tax at +7-10 ms/chunk; that is "
              "the figure this recovers.")
        print("=" * 78)
        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("wrote {}".format(OUT_JSON))
    finally:
        asyncio.run(service.stop())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
