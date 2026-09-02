"""Story 20.5 Phase 2 (AC #3) — verify codec state caching on the REAL model.

Phase 1's bench proved the mechanism on a hand-rolled traversal. This script
verifies the SHIPPED implementation — ``codec_state_cache.py`` reached through
``QwenTTSService._build_true_stream_decode_fn``, driven by a real
``StreamingDecoderWorker`` with the real splice and the real Story 20.4
overlap-add, and consumed by a real ``StreamingChunkBuffer``.

It answers four questions with numbers:

  1. **Does the shipped path reconstruct the whole-sequence decode?**
     Ground truth is ``decoder(codes)`` over the entire captured token
     sequence. The stitched worker output must match it in length exactly and
     in content to bf16 rounding.

  2. **Is the Story 20.4 1,024-sample seam blend still doing work?** (AC #3
     requires this re-evaluated on evidence, not assumed.) Measured as the
     difference between the tail a chunk retains past its splice and the head
     of the chunk that follows — the two signals the blend cross-fades. Under
     carried state they are the same audio, so the blend should be an
     identity.

  3. **What about the 64-sample consumer crossfade in
     ``StreamingChunkBuffer``?** That one blends the *last* 64 samples of one
     chunk with the *first* 64 of the next — different moments in time — so
     on continuous audio it is a 2.7 ms comb, not an identity. Measured as
     the difference between the buffer's output and the same chunks
     concatenated with no crossfade.

  4. **Cost.** Per-session carried state in bytes and tensors, and per-chunk
     decode time against the stateless arm.

BOTH ARMS COME FROM ONE TOKEN CAPTURE. The talker runs once per utterance and
its chunks are replayed through both decode paths, so the two arms are the
SAME take and every difference is attributable to the decode. That is not
available to a chunk-size retune (Story 20.4's arms necessarily differed in
what the talker produced) and it is what makes this comparison sharp.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-5-phase2-verify.py

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TOK_DIR = SCRIPT_DIR / "20-5-tokens"

sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttfa_spike_harness as H  # noqa: E402

from myvoice.services.streaming_chunk_buffer import (  # noqa: E402
    StreamingChunkBuffer,
)
from myvoice.services.tts_streaming import codec_token_streamer  # noqa: E402
from myvoice.services.tts_streaming import streaming_decoder  # noqa: E402
from myvoice.services.tts_streaming.codec_token_streamer import (  # noqa: E402
    CodecTokenStreamer,
    END_OF_STREAM,
)
from myvoice.services.tts_streaming.streaming_decoder import (  # noqa: E402
    StreamingDecoderWorker,
)

SR = 24000
SPF = streaming_decoder._CODEC_SAMPLES_PER_FRAME
EDGE = streaming_decoder._CODEC_EDGE_LOSS_SAMPLES
OLA = streaming_decoder._OVERLAP_ADD_SAMPLES
CS = codec_token_streamer.DEFAULT_CHUNK_SIZE
LA = codec_token_streamer.DEFAULT_LOOKAHEAD
HEAD = 1024

UTTERANCES = {
    "m-020": "She sells seashells by the seashore on a still summer morning.",
    "l-020": (
        "This is a longer-form test designed to expose the difference "
        "between metric-side first-chunk emission and user-perceived "
        "first-audio latency. On the pre-Story-17.3 build, the user would "
        "wait approximately forty seconds for this utterance to start "
        "playing, even though the streaming pipeline emitted the first "
        "chunk internally at around five seconds."
    ),
    "l-021": (
        "Six slick slim sycamore saplings stood swaying silently as the "
        "soft summer storm slowly swirled across the steep slopes south of "
        "the silver stream below us, and the sound of it carried further "
        "than anyone standing there expected it to."
    ),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def nrmse(ref, test):
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    test = np.asarray(test, dtype=np.float64).reshape(-1)
    n = min(ref.size, test.size)
    denom = np.linalg.norm(ref[:n])
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm(ref[:n] - test[:n]) / denom)


def best_lag(gt, probe, centre, search=1200):
    n = probe.size
    nb = np.linalg.norm(probe)
    best, blag = -2.0, None
    lo = max(0, centre - search)
    hi = min(gt.size - n, centre + search)
    if hi < lo or nb == 0:
        return None, float("nan")
    for lag in range(lo, hi + 1):
        seg = gt[lag:lag + n]
        na = np.linalg.norm(seg)
        if na == 0:
            continue
        c = float(np.dot(seg, probe) / (na * nb))
        if c > best:
            best, blag = c, lag
    return blag, best


def run_worker(chunks, decode_fn, cs=CS, la=LA):
    """Drive a REAL StreamingDecoderWorker over the captured token chunks.

    Returns (posted_segments, retained_tails, per_chunk_decode_seconds).
    The retained tails are read straight off the worker instance after each
    post, which is what lets question 2 measure the blend's two inputs.
    """
    streamer = CodecTokenStreamer(chunk_size=cs, lookahead=la)
    posted = []
    tails = []
    times = []
    done = threading.Event()

    holder = {}

    def post(method, session_id, *args):
        if method == "append_chunk":
            posted.append(np.asarray(args[0], dtype=np.float32).copy())
            pending = holder["worker"]._pending_overlap
            tails.append(None if pending is None else pending.copy())
        elif method in ("finalize", "cancel", "discard"):
            done.set()

    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=decode_fn, post_mutation=post,
        session_id="verify", model_type="qwen3_tts", hardware="gpu",
    )
    holder["worker"] = worker

    for chunk in chunks:
        streamer.queue.put(chunk)
    streamer.queue.put(END_OF_STREAM)

    t0 = time.perf_counter()
    worker.start()
    worker.join(timeout=300.0)
    total = time.perf_counter() - t0
    done.wait(timeout=5.0)
    times.append(total)
    return posted, tails, total


def through_consumer_buffer(segments, crossfade):
    """Push posted segments through a real StreamingChunkBuffer and return the
    released int16 stream."""
    buf = StreamingChunkBuffer(
        watermark_ms=H.WATERMARK_MS, crossfade_samples=crossfade,
        sample_rate=SR, channels=1, sample_width=2,
    )
    out = []
    for seg in segments:
        payload = (np.clip(seg, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        for released in buf.push(payload, is_final=False):
            out.append(released)
    for released in buf.flush_remaining():
        out.append(released)
    return np.frombuffer(b"".join(out), dtype=np.int16)


def error_by_position(gt, posted, bins=(0, 256, 512, 1024, 2048, 4096, 8192)):
    """Phase 1 SS2.3's discriminator: RMS error / ground-truth RMS by distance
    into the chunk. A COLD START is head-weighted and decays over ~4,000
    samples; ROUNDING is flat. The shape says which regime we are in, and it
    says it independently of the absolute level, which is what makes it
    robust to a quiet utterance inflating whole-signal NRMSE.
    """
    offset = 0
    rows = {}
    for i, seg in enumerate(posted):
        if i > 0:
            for lo, hi in zip(bins, bins[1:]):
                a = offset + lo
                b = min(offset + hi, gt.size, offset + seg.size)
                if b <= a:
                    continue
                ref = gt[a:b].astype(np.float64)
                got = seg[lo:lo + (b - a)].astype(np.float64)
                denom = np.sqrt(np.mean(ref ** 2))
                if denom == 0:
                    continue
                rows.setdefault((lo, hi), []).append(
                    float(np.sqrt(np.mean((ref - got) ** 2)) / denom))
        offset += seg.size
    return {"{}-{}".format(lo, hi): float(np.median(v))
            for (lo, hi), v in sorted(rows.items())}


def per_chunk_nrmse(gt, posted):
    offset = 0
    out = []
    for seg in posted:
        end = min(offset + seg.size, gt.size)
        out.append(round(nrmse(gt[offset:end], seg[:end - offset]), 6))
        offset += seg.size
    return out


def as_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


# --------------------------------------------------------------------------- #
# token capture (reuses the Phase 1 captures so runs are comparable)
# --------------------------------------------------------------------------- #


async def load_service_and_tokens():
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    TOK_DIR.mkdir(parents=True, exist_ok=True)
    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings,
    )
    if not await service.start():
        raise SystemExit("FATAL: service.start() returned False")

    cached = {u: TOK_DIR / "{}.npz".format(u) for u in UTTERANCES}
    if all(p.exists() for p in cached.values()):
        ok, err = await service._model_registry.ensure_model_loaded(
            QwenModelType.BASE
        )
        if not ok:
            raise SystemExit("model load failed: {}".format(err))
        tokens = {
            u: torch.from_numpy(np.load(p)["tokens"]).long()
            for u, p in cached.items()
        }
        print("reused Phase 1 token captures: " + ", ".join(
            "{}={}f".format(u, t.shape[0]) for u, t in sorted(tokens.items())))
        return service, tokens

    grabbed = []
    real_builder = service._build_true_stream_decode_fn

    def wrapped(model, *a, **kw):
        inner = real_builder(model, *a, **kw)

        def _decode(chunk):
            grabbed.append(torch.as_tensor(chunk).detach().cpu().clone())
            return inner(chunk)
        return _decode

    service._build_true_stream_decode_fn = wrapped
    prompt = H._load_voice_clone_prompt(service)
    warm = QwenTTSRequest(
        text=H.PRIMING_TEXT, language="English", model_type=QwenModelType.BASE,
        streaming=True, voice_clone_prompt=prompt, suppress_audio_output=True,
    )
    await service._generate_true_stream(warm)
    await asyncio.sleep(0.2)

    tokens = {}
    for utt_id, text in UTTERANCES.items():
        grabbed.clear()
        req = QwenTTSRequest(
            text=text, language="English", model_type=QwenModelType.BASE,
            streaming=True, voice_clone_prompt=prompt,
            suppress_audio_output=True,
        )
        resp = await service._generate_true_stream(req)
        await asyncio.sleep(0.2)
        if not resp.success or not grabbed:
            print("  SKIP {}".format(utt_id))
            continue
        frames = [grabbed[0]] + [g[LA:] for g in grabbed[1:]]
        seq = torch.cat(frames, dim=0)
        np.savez(TOK_DIR / "{}.npz".format(utt_id), tokens=seq.numpy())
        tokens[utt_id] = seq
    service._build_true_stream_decode_fn = real_builder
    return service, tokens


def rechunk(seq):
    """Reproduce the talker forward-hook's chunking: 30-frame windows sliding
    by 25, then whatever is left as the residual flush."""
    out = []
    i = 0
    n = seq.shape[0]
    while i + CS + LA <= n:
        out.append(seq[i:i + CS + LA])
        i += CS
    if i < n:
        out.append(seq[i:])
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


async def _run():
    if (CS, LA) != (25, 5):
        raise SystemExit(
            "FATAL: committed geometry is ({}, {}); Story 20.5 Phase 2 holds "
            "it at (25, 5).".format(CS, LA)
        )

    service, tokens = await load_service_and_tokens()
    model = service._model_registry.get_loaded_model()
    decoder = model.model.speech_tokenizer.model.decoder
    device = model.model.speech_tokenizer.device

    report = {"geometry": {"chunk_size": CS, "lookahead": LA,
                           "samples_per_frame": SPF, "edge_loss": EDGE,
                           "overlap_add": OLA},
              "utterances": {}}

    # Build both arms from the SAME service, one model load.
    os.environ["MYVOICE_CODEC_STATE_CACHE"] = "0"
    stateless = service._build_true_stream_decode_fn(model, CS, LA)
    os.environ.pop("MYVOICE_CODEC_STATE_CACHE", None)
    stateful = service._build_true_stream_decode_fn(model, CS, LA)

    print()
    print("stateless arm : carries_codec_state={}".format(
        getattr(stateless, "carries_codec_state", False)))
    print("stateful  arm : carries_codec_state={}".format(
        getattr(stateful, "carries_codec_state", False)))
    if not getattr(stateful, "carries_codec_state", False):
        from myvoice.services.tts_streaming import codec_state_cache
        try:
            _, why = codec_state_cache.build_stateful_decode_fn(
                decoder, chunk_size=CS, lookahead=LA, device=device
            )
        except Exception as exc:  # noqa: BLE001
            why = "raised {!r}".format(exc)
        raise SystemExit(
            "FATAL: the state-cached decoder declined to engage on this "
            "model.\n  reason: {}\nThere is nothing to verify until it "
            "does.".format(why)
        )
    report["geometry"]["stateful_engaged"] = True

    for utt_id, seq in sorted(tokens.items()):
        chunks = rechunk(seq)
        n_frames = seq.shape[0]
        print("\n=== {} : {} frames, {} chunks ===".format(
            utt_id, n_frames, len(chunks)))

        codes = seq.unsqueeze(0).transpose(1, 2).to(device)
        with torch.inference_mode():
            gt = decoder(codes).squeeze(0).squeeze(0)
        gt = gt.to(torch.float32).cpu().numpy()
        print("  ground truth (whole-sequence decode): {} samples "
              "(1920*N - 555 = {})".format(gt.size, SPF * n_frames - EDGE))

        # Warm both arms on this shape first (cuDNN autotune), then take the
        # best of two timed passes each. Phase 1 SS3.2 flagged that the first
        # timed case carries autotune cost; not repeating that here.
        stateful.reset()
        run_worker(chunks[:2], stateful)
        run_worker(chunks[:2], stateless)
        stateful.reset()
        s_posted, s_tails, s_time = run_worker(chunks, stateful)
        stateful.reset()
        _, _, s_time2 = run_worker(chunks, stateful)
        l_posted, l_tails, l_time = run_worker(chunks, stateless)
        _, _, l_time2 = run_worker(chunks, stateless)
        s_time = min(s_time, s_time2)
        l_time = min(l_time, l_time2)

        s_stitched = np.concatenate(s_posted)
        l_stitched = np.concatenate(l_posted)

        row = {
            "frames": int(n_frames),
            "chunks": len(chunks),
            "gt_samples": int(gt.size),
            "stateful_samples": int(s_stitched.size),
            "stateless_samples": int(l_stitched.size),
            "stateful_len_delta": int(s_stitched.size - gt.size),
            "stateless_len_delta": int(l_stitched.size - gt.size),
            "stateful_posted_sizes": [int(p.size) for p in s_posted[:4]],
            "stateless_posted_sizes": [int(p.size) for p in l_posted[:4]],
            "worker_seconds_stateful": round(s_time, 4),
            "worker_seconds_stateless": round(l_time, 4),
        }

        print("  Q1 length      : state-cached {:+d} vs ground truth, "
              "stateless {:+d}".format(row["stateful_len_delta"],
                                       row["stateless_len_delta"]))
        row["stateful_full_nrmse"] = nrmse(gt, s_stitched)
        row["stateless_full_nrmse"] = nrmse(gt, l_stitched)
        print("  Q1 whole-signal NRMSE vs ground truth: "
              "state-cached {:.3e}  stateless {:.3e}".format(
                  row["stateful_full_nrmse"], row["stateless_full_nrmse"]))

        # -- Q1b: per-seam head fidelity ------------------------------- #
        for label, posted in (("stateful", s_posted), ("stateless", l_posted)):
            offset = 0
            noms, corrs, lags = [], [], []
            for i, seg in enumerate(posted):
                if i > 0 and seg.size >= HEAD and offset + HEAD <= gt.size:
                    probe = seg[:HEAD].astype(np.float64)
                    ref = gt[offset:offset + HEAD].astype(np.float64)
                    noms.append(nrmse(ref, probe))
                    lag, corr = best_lag(gt.astype(np.float64), probe, offset)
                    corrs.append(corr)
                    lags.append(float("nan") if lag is None else lag - offset)
                offset += seg.size
            if noms:
                row[label + "_head_nrmse_med"] = float(np.median(noms))
                row[label + "_head_nrmse_max"] = float(np.max(noms))
                row[label + "_corr_med"] = float(np.median(corrs))
                row[label + "_corr_min"] = float(np.min(corrs))
                row[label + "_lag_min"] = float(np.nanmin(lags))
                row[label + "_lag_max"] = float(np.nanmax(lags))
                print("  Q1b {:<9} head NRMSE med {:.4f} max {:.4f} | "
                      "corr {:.4f}/{:.4f} | lag {:+.0f}..{:+.0f}".format(
                          label, row[label + "_head_nrmse_med"],
                          row[label + "_head_nrmse_max"],
                          row[label + "_corr_med"], row[label + "_corr_min"],
                          row[label + "_lag_min"], row[label + "_lag_max"]))

        # -- Q2: is the 1024-sample seam blend still doing work? -------- #
        # The blend's two inputs are the tail retained past chunk k's splice
        # and the head of chunk k+1. Under carried state they should be the
        # same audio. Note the posted segments have ALREADY been blended, so
        # comparing tail_k against posted[k+1][:w] measures the blend's own
        # input difference only where the blend is an identity -- which is
        # exactly what we are testing, and a non-identity would show up as a
        # large number here.
        for label, posted, tails in (("stateful", s_posted, s_tails),
                                     ("stateless", l_posted, l_tails)):
            diffs = []
            for i, tail in enumerate(tails[:-1]):
                if tail is None or tail.size == 0:
                    continue
                head = posted[i + 1][:tail.size]
                if head.size != tail.size:
                    continue
                diffs.append(nrmse(head, tail))
            if diffs:
                row[label + "_blend_input_nrmse_med"] = float(np.median(diffs))
                row[label + "_blend_input_nrmse_max"] = float(np.max(diffs))
                print("  Q2  {:<9} blend inputs differ by NRMSE "
                      "med {:.3e} max {:.3e}".format(
                          label, float(np.median(diffs)), float(np.max(diffs))))

        # -- Q1c: error by position into the chunk ---------------------- #
        row["stateful_error_by_position"] = error_by_position(gt, s_posted)
        row["stateless_error_by_position"] = error_by_position(gt, l_posted)
        row["stateful_per_chunk_nrmse"] = per_chunk_nrmse(gt, s_posted)
        row["stateless_per_chunk_nrmse"] = per_chunk_nrmse(gt, l_posted)
        print("  Q1c error by position into chunk (RMS err / gt RMS, median):")
        for label in ("stateful", "stateless"):
            prof = row[label + "_error_by_position"]
            print("      {:<9} {}".format(label, "  ".join(
                "{}:{:.3f}".format(k, v) for k, v in prof.items())))
        print("      per-chunk NRMSE stateful : {}".format(
            row["stateful_per_chunk_nrmse"]))

        # -- Q3: the 64-sample consumer crossfade ----------------------- #
        # Measured against GROUND TRUTH, not against itself. On the stateless
        # arm the crossfade bridges a genuine discontinuity; on the
        # state-cached arm the audio is already continuous, so the same
        # cross-fade mixes sample n+i with sample n-64+i -- a 2.7 ms comb over
        # audio that needed no repair. The question is whether it moves the
        # output toward or away from what the codec actually produced.
        gt16 = as_int16(gt).astype(np.float64)
        for label, posted in (("stateful", s_posted), ("stateless", l_posted)):
            with_xf = through_consumer_buffer(posted, H.CROSSFADE_SAMPLES)
            without_xf = through_consumer_buffer(posted, 0)
            n = min(with_xf.size, without_xf.size, gt16.size)
            delta = (with_xf[:n].astype(np.float64)
                     - without_xf[:n].astype(np.float64))
            touched = int(np.count_nonzero(delta))
            row[label + "_xf_samples_touched"] = touched
            row[label + "_xf_fraction"] = touched / max(1, n)
            row[label + "_xf_max_abs_int16"] = (
                float(np.max(np.abs(delta))) if n else 0.0)
            row[label + "_xf_on_nrmse_vs_gt"] = nrmse(gt16[:n], with_xf[:n])
            row[label + "_xf_off_nrmse_vs_gt"] = nrmse(gt16[:n], without_xf[:n])
            verdict = ("HURTS" if row[label + "_xf_on_nrmse_vs_gt"]
                       > row[label + "_xf_off_nrmse_vs_gt"] else "helps")
            print("  Q3  {:<9} 64-sample consumer crossfade: touches {} of {} "
                  "({:.3%}), max |d| {:.0f} LSB".format(
                      label, touched, n, row[label + "_xf_fraction"],
                      row[label + "_xf_max_abs_int16"]))
            print("      vs ground truth: crossfade ON {:.5e}  OFF {:.5e}  -> "
                  "{}".format(row[label + "_xf_on_nrmse_vs_gt"],
                              row[label + "_xf_off_nrmse_vs_gt"], verdict))

        # -- Q4: cost --------------------------------------------------- #
        # Re-run one pass so the state object is populated at the end.
        stateful.reset()
        run_worker(chunks, stateful)
        report["utterances"][utt_id] = row

    # State size after a full utterance, and after the longest one.
    stateful.reset()
    longest = max(tokens.items(), key=lambda kv: kv[1].shape[0])
    chunks = rechunk(longest[1])
    early = None
    for i, chunk in enumerate(chunks):
        stateful(chunk)
        # sliding_window=72 frames and each chunk commits 25, so the KV cache
        # reaches its cap during chunk 3. Sample after it is full: sampling
        # earlier measures a cache that has simply not filled yet, which would
        # read as unbounded growth.
        if i == 3:
            early = stateful.state_nbytes()
    late = stateful.state_nbytes()
    report["cost"] = {
        "utterance": longest[0],
        "frames": int(longest[1].shape[0]),
        "state_bytes_after_4_chunks": early[0],
        "state_tensors_after_4_chunks": early[1],
        "state_bytes_after_all_chunks": late[0],
        "state_tensors_after_all_chunks": late[1],
        "bounded": early[0] == late[0],
        "chunks": len(chunks),
    }
    print("\n=== cost ({} , {} frames) ===".format(longest[0],
                                                   longest[1].shape[0]))
    print("  after 4 chunks (KV window full): {} tensors, {:.2f} KiB".format(
        early[1], early[0] / 1024))
    print("  after all      : {} tensors, {:.2f} KiB  (bounded={})".format(
        late[1], late[0] / 1024, report["cost"]["bounded"]))

    out = SCRIPT_DIR / "20-5-phase2-verify.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote {}".format(out.name))

    await service.stop()
    return 0


def main():
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print("device={} vram={:.1f}GiB torch={}".format(
            pr.name, pr.total_memory / 1024 ** 3, torch.__version__))
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
