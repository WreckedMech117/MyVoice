"""Story 20.8 AC #3/#3a — NFR3 audition fixture: does the ear notice cs10's seams?

    reference = chunk_size 25   <- what ships today
    candidate = chunk_size 10   <- the committed retune

Both arms carry codec state caching (Story 20.5), the retired lookahead
(Story 20.6) and the gated 0-sample consumer crossfade. **The only variable is
chunk_size.**

WHY THIS ROUND CAN USE ONE TALKER RUN PER PAIR — MEASURED, NOT ASSUMED
----------------------------------------------------------------------
Story 20.4 §17 recorded that a chunk-size story could not reuse tokens, because
"chunk size perturbs the talker". Story 20.8 §7 tested that directly instead of
arguing it:

  * (c) the pipeline is seed-reproducible — two live cs25 runs at one seed give
    bit-identical token streams;
  * (b) live cs25 vs live cs7 vs live cs10 at one seed give **bit-identical**
    token streams, both seeds, 4/4. Within a compiled state chunk size does not
    reach the talker;
  * (a) an offline re-chunk rendered through the real worker + real buffer is
    **bit-for-bit** what the live run at that geometry produces — including in
    the cross-geometry form — wherever the decoder is bit-reproducible against
    itself;
  * (d) FAILED: ``decode_window_frames`` is a ``compile_cache`` key dimension,
    so a build at a different chunk size reads a different inductor cache
    directory and draws a *different but equally valid* stream.

AC #3a is (d)'s consequence, and it dictates this file's shape.

THE FOUR THINGS AC #3a REQUIRES, AND WHERE EACH LIVES
------------------------------------------------------
1. **Capture at the CANDIDATE geometry.** ``_preflight`` refuses to run unless
   the committed geometry is the candidate's, so the capture process is a
   shipped cs10 build. The candidate arm is then bit-for-bit what ships; the
   reference arm is cs25 geometry over the same content, which is the correct
   content-constant control rather than a second draw.
2. **Worker BEFORE queue.** ``render`` starts the worker and feeds from a
   separate thread. The queue is bounded at ``queue_max_factor * chunk_size``;
   filling first deadlocks whenever an utterance exceeds that many chunks.
   Story 20.5's and Story 20.6's fixture generators both fill first and escaped
   only because cs25 gives maxsize 100 against ~10 chunks.
3. **No arm ends on a 1- or 2-frame terminal residual.** §7 found the decoder
   is bit-reproducible against itself EXCEPT on a 2-frame residual decode
   (−66 dBFS on 0.1 % of samples). A take whose frame count would leave either
   arm with a 1- or 2-frame residual is REDRAWN, so the §7 (a) caveat is moot
   rather than argued away. 1 is excluded alongside 2 because it is smaller and
   untested, not because it was observed.
4. **A byte-identical control.** ``ctl-020`` is rendered twice through the SAME
   arm and the two files are asserted byte-identical before the round starts.
   It sets the listener's noise floor and it catches a broken fixture before it
   costs a verdict. Take 1 does this at the reference geometry, take 2 at the
   candidate's, so the control also re-verifies §7's determinism finding on the
   actual fixture content.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-8-regen-audition-fixture.py

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
import wave
import zlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = SCRIPT_DIR / "20-8-perceptual-fixtures"

sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttfa_spike_harness as H  # noqa: E402

from myvoice.services.tts_streaming import (  # noqa: E402
    CodecTokenStreamer, END_OF_STREAM, StreamingDecoderWorker,
    codec_token_streamer, resolve_streamer_geometry,
)
from myvoice.services.tts_streaming import streaming_decoder  # noqa: E402
from myvoice.services.streaming_chunk_buffer import (  # noqa: E402
    StreamingChunkBuffer,
)

SR = 24000
REFERENCE_CS = 25          # what ships today
CANDIDATE_CS = 10          # the committed retune (AC #3b's primary candidate)
TAKES = 2
MAX_REDRAWS = 8
FORBIDDEN_RESIDUALS = (1, 2)
BASE_SEED = 20800

UTTERANCES = {
    "ctl-020": "Say that again.",
    "s-020": "Hold on a second, say that again.",
    "s-021": "Six sticks split, ship shape.",
    "s-022": "Bit, bat, bot, but, bet.",
    "m-020": "She sells seashells by the seashore on a still summer morning.",
    "m-021": "The bell rang clear at noon and echoed across the open field.",
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
CONTROL = "ctl-020"


# --------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------- #

def _preflight():
    cs, la = resolve_streamer_geometry()
    if (cs, la) != (CANDIDATE_CS, 0):
        raise SystemExit(
            "FATAL: committed geometry resolves to ({}, {}), expected ({}, 0).\n"
            "AC #3a requires the capture to run in a process AT THE CANDIDATE "
            "geometry, because decode_window_frames is a compile_cache key "
            "dimension (Story 20.8 §7 claim (d)) — a capture taken at another "
            "chunk size draws a stream no shipped cs{} build would produce."
            .format(cs, la, CANDIDATE_CS, CANDIDATE_CS))
    if codec_token_streamer.DEFAULT_LOOKAHEAD != 5:
        raise SystemExit(
            "FATAL: DEFAULT_LOOKAHEAD is not 5. It is the STATELESS path's "
            "lookahead and this story does not touch it.")
    for attr in ("_CODEC_SAMPLES_PER_FRAME", "_CODEC_EDGE_LOSS_SAMPLES",
                 "_OVERLAP_ADD_SAMPLES"):
        if not hasattr(streaming_decoder, attr):
            raise SystemExit(
                "FATAL: streaming_decoder is missing {} — the Story 20.4 seam "
                "fix must be present on BOTH arms or this round repeats Story "
                "20.4 round 2's two-variable confound.".format(attr))
    # The watermark floor, re-derived here rather than trusted (Story 20.8 §1).
    wm = int(0.5 * SR)
    spf = streaming_decoder._CODEC_SAMPLES_PER_FRAME
    edge = streaming_decoder._CODEC_EDGE_LOSS_SAMPLES
    for name, n in (("reference", REFERENCE_CS), ("candidate", CANDIDATE_CS)):
        first = n * spf - edge
        if first < wm:
            raise SystemExit(
                "FATAL: {} arm chunk_size={} gives a first chunk of {} samples "
                "against a {}-sample watermark — it would hand back a cushion "
                "penalty and the arm would not be a geometry result."
                .format(name, n, first, wm))
    print("preflight OK")
    print("  committed geometry = ({}, {})  [capture runs here]".format(cs, la))
    print("  reference arm chunk_size={}   candidate arm chunk_size={}"
          .format(REFERENCE_CS, CANDIDATE_CS))
    print("  seam fix on both arms: samples/frame={} edge={} ola={}".format(
        spf, edge, streaming_decoder._OVERLAP_ADD_SAMPLES))
    print("  watermark floor cleared by both arms "
          "(first chunk {} / {} samples vs {})".format(
              REFERENCE_CS * spf - edge, CANDIDATE_CS * spf - edge, wm))


# --------------------------------------------------------------------- #
# render — REAL streamer, REAL worker, REAL buffer, worker started FIRST
# --------------------------------------------------------------------- #

def _slice(frames, n):
    return [frames[i:i + n].clone() for i in range(0, frames.shape[0], n)]


def render(service, model, frames, chunk_size):
    streamer = CodecTokenStreamer(
        chunk_size=chunk_size, lookahead=codec_token_streamer.DEFAULT_LOOKAHEAD)
    decode_fn = service._build_true_stream_decode_fn(
        model, chunk_size=streamer.chunk_size, lookahead=streamer.lookahead)
    continuous = bool(getattr(decode_fn, "carries_codec_state", False))
    if not continuous:
        raise SystemExit(
            "FATAL: the decode_fn declined codec state caching. Both arms of "
            "this round assume it; there is no round to run until it engages.")
    streamer.apply_codec_state_geometry(continuous)
    assert streamer.lookahead == 0

    posted = []
    done = threading.Event()

    def post(method, session_id, *args):
        if method == "append_chunk":
            posted.append(np.asarray(args[0], dtype=np.float32).copy())
        elif method in ("finalize", "cancel", "discard"):
            done.set()

    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=decode_fn, post_mutation=post,
        session_id="fixture-20-8", model_type="qwen3_tts", hardware="gpu",
    )

    # AC #3a #2 — worker FIRST, then feed from another thread. The queue is
    # bounded at queue_max_factor * chunk_size.
    chunks = _slice(frames, chunk_size)

    def feed():
        for c in chunks:
            streamer.queue.put(c)
        streamer.queue.put(END_OF_STREAM)

    worker.start()
    feeder = threading.Thread(target=feed, daemon=True, name="fixture-feeder")
    feeder.start()
    feeder.join(timeout=600.0)
    worker.join(timeout=600.0)
    done.wait(timeout=10.0)
    if feeder.is_alive() or worker._thread.is_alive():
        raise SystemExit("FATAL: render did not drain at cs{}".format(chunk_size))

    # Ask the producer, exactly as app.py:3242-3247 does.
    crossfade = 0 if continuous else H.CROSSFADE_SAMPLES
    buf = StreamingChunkBuffer(
        watermark_ms=H.WATERMARK_MS, crossfade_samples=crossfade,
        sample_rate=SR, channels=1, sample_width=2,
    )
    out = []
    for seg in posted:
        payload = (np.clip(seg, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        for released in buf.push(payload, is_final=False):
            out.append(released)
    for released in buf.flush_remaining():
        out.append(released)
    return np.frombuffer(b"".join(out), dtype=np.int16), len(chunks), crossfade


# --------------------------------------------------------------------- #
# io / measurement
# --------------------------------------------------------------------- #

def write_wav(path: Path, samples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(samples.astype(np.int16).tobytes())


def active_rms_db(samples) -> float:
    x = samples.astype(np.float64) / 32768.0
    if x.size == 0:
        return float("-inf")
    frame = 480
    n = (x.size // frame) * frame
    if n == 0:
        return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)
    blocks = x[:n].reshape(-1, frame)
    e = (blocks ** 2).mean(axis=1)
    active = e[e > e.max() * 1e-4]
    if active.size == 0:
        active = e
    return float(20 * np.log10(np.sqrt(active.mean()) + 1e-12))


def waveform_delta_db(a, b) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return float("-inf")
    d = a[:n].astype(np.float64) - b[:n].astype(np.float64)
    ref = np.sqrt((a[:n].astype(np.float64) ** 2).mean()) + 1e-12
    return float(20 * np.log10((np.sqrt((d ** 2).mean()) + 1e-12) / ref))


# --------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------- #

async def capture_take(service, prompt, text, seed):
    """One live generation at the committed (candidate) geometry.

    Returns the flat ``(frames, num_code_groups)`` token tensor the talker
    emitted. The tap preserves the decode_fn's declared attributes; dropping
    them would make the dispatch read ``carries_codec_state=False`` and
    silently capture at the pre-20.6 geometry.
    """
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest

    captured = []
    real_builder = service._build_true_stream_decode_fn

    def tapping_builder(model, *a, **kw):
        inner = real_builder(model, *a, **kw)

        def _tap(chunk):
            captured.append(torch.as_tensor(chunk).detach().cpu().clone())
            return inner(chunk)

        for attr in ("carries_codec_state", "_window_frames"):
            if hasattr(inner, attr):
                setattr(_tap, attr, getattr(inner, attr))
        return _tap

    service._build_true_stream_decode_fn = tapping_builder
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        resp = await service._generate_true_stream(QwenTTSRequest(
            text=text, language="English", model_type=QwenModelType.BASE,
            streaming=True, voice_clone_prompt=prompt,
            suppress_audio_output=True))
        await asyncio.sleep(0.2)
    finally:
        service._build_true_stream_decode_fn = real_builder
    if not resp.success or not captured:
        raise SystemExit("FATAL: capture failed: {}".format(
            getattr(resp, "error_message", None)))
    return torch.cat(captured, dim=0)


def residual_ok(n_frames: int) -> bool:
    """AC #3a #3 — neither arm may end on a 1- or 2-frame residual."""
    for cs in (REFERENCE_CS, CANDIDATE_CS):
        if (n_frames % cs) in FORBIDDEN_RESIDUALS:
            return False
    return True


def _write_truthtable(by_trial, manifest) -> None:
    """Assign presentation slots, BALANCED, from a fixed seed.

    Story 20.6's generator established the rule and the reason: per-trial coin
    flips gave a 12/4 reference-first split on a 16-trial round, and position
    bias is the largest nuisance variable left when the predicted answer is
    "equivalent". The order is shuffled from a fixed seed, so it is
    reproducible from this file and not inferable from listening order.
    """
    rng = random.Random(20080902)
    seams = manifest["summary"]
    table = {
        "_meta": {
            "story": "20.8",
            "round": 1,
            "candidate": "candidate",
            "reference": "reference",
            "candidate_desc": (
                "chunk_size = {} — the committed retune. Codec state caching, "
                "retired lookahead, gated 0-sample consumer crossfade, "
                "unchanged.".format(CANDIDATE_CS)),
            "reference_desc": (
                "chunk_size = {} — what ships today, everything else "
                "identical.".format(REFERENCE_CS)),
            "isolates": (
                "chunk_size ONLY. Both arms are decoded from the SAME talker "
                "run, captured in a process running AT THE CANDIDATE geometry "
                "(Story 20.8 §7 claim (d): decode_window_frames is a "
                "compile-cache key, so a capture taken at another chunk size "
                "would draw a stream no shipped build produces). Wording, "
                "prosody, pauses and duration are identical to the sample; "
                "the candidate arm is bit-for-bit what a shipped cs{} build "
                "emits. The ONLY difference is how many seams the stream is "
                "cut into: {} on the reference against {} on the candidate "
                "across the A/B trials ({}x).".format(
                    CANDIDATE_CS, seams["total_ref_seams"],
                    seams["total_cand_seams"], seams["seam_ratio"])),
            "control_trial": CONTROL,
            "control_desc": (
                "rendered TWICE THROUGH THE SAME ARM and asserted "
                "byte-identical at generation time — take 1 at the reference "
                "geometry, take 2 at the candidate's. Any preference or "
                "defect reported here is a property of the listening, not of "
                "the change, and it sets the round's noise floor."),
            "takes_per_utterance": TAKES,
            "prediction": (
                "P1 equivalent is modal, >= 10 of 16. P2 (BLOCKING) no "
                "chunk-boundary defect on a candidate trial that its paired "
                "reference does not also carry. P3 any difference is on the "
                "long fixtures, which carry the most seams. P4 (the "
                "embarrassing one) if the candidate is preferred on >= 4 "
                "trials the diagnosis is wrong in the opposite direction. "
                "Recorded in full in 20-8-chunk-size-reopen-evidence.md §9 "
                "BEFORE the round."),
            "variance_note": (
                "Within a pair there is no take-to-take variance to average "
                "over — the two files are one take rendered at two "
                "geometries. Story 20.4 §17's warning is sidestepped, not "
                "repealed; the second take samples the CONTENT lottery."),
        },
        "L1": {},
    }
    trials = sorted(by_trial)
    orders = [("reference", "candidate")] * (len(trials) // 2)
    orders += [("candidate", "reference")] * (len(trials) - len(orders))
    rng.shuffle(orders)
    for trial, (first, second) in zip(trials, orders):
        table["L1"][trial] = {
            "trial_A_filename": by_trial[trial][first]["filename"],
            "trial_A_arm": first,
            "trial_B_filename": by_trial[trial][second]["filename"],
            "trial_B_arm": second,
        }
    path = OUT_DIR / "_perlistener_truthtable.json"
    path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    first_ref = sum(1 for v in table["L1"].values()
                    if v["trial_A_arm"] == "reference")
    print("\nTruth table -> {} ({} trials; reference is trial A on {}, "
          "candidate on {})".format(path.name, len(table["L1"]), first_ref,
                                    len(table["L1"]) - first_ref))


async def _run() -> int:
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    _preflight()
    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings)
    if not await service.start():
        print("FATAL: service.start() returned False", file=sys.stderr)
        return 1

    manifest = {"reference_chunk_size": REFERENCE_CS,
                "candidate_chunk_size": CANDIDATE_CS,
                "committed_geometry": list(resolve_streamer_geometry()),
                "forbidden_residuals": list(FORBIDDEN_RESIDUALS),
                "trials": [], "redraws": []}
    # Bound before the try so a failure inside it raises the REAL error rather
    # than a NameError from the truth-table call after the finally.
    by_trial = {}
    try:
        prompt = H._load_voice_clone_prompt(service)
        await service._generate_true_stream(QwenTTSRequest(
            text=H.PRIMING_TEXT, language="English",
            model_type=QwenModelType.BASE, streaming=True,
            voice_clone_prompt=prompt, suppress_audio_output=True))
        await asyncio.sleep(0.3)
        print("  priming done\n")
        model = service._model_registry.get_loaded_model()

        print("=== capture: one talker run per pair, at the committed geometry ===")
        takes = {}
        for utt_id, text in sorted(UTTERANCES.items()):
            for take in range(1, TAKES + 1):
                # zlib.crc32, not hash(): str hashing is salted per
                # process (PYTHONHASHSEED), which would make the
                # fixture unreproducible across runs — and a fixture
                # whose seeds cannot be restated is not evidence.
                seed = (BASE_SEED
                        + (zlib.crc32(utt_id.encode()) % 100000)
                        + take * 7919)
                frames = None
                for attempt in range(MAX_REDRAWS):
                    t0 = time.time()
                    f = await capture_take(service, prompt, text, seed + attempt * 104729)
                    n = int(f.shape[0])
                    ok = residual_ok(n)
                    print("  {:<8} take {}  {:>4d} frames  "
                          "residual ref={:<2d} cand={:<2d}  {:>6.0f} ms  {}"
                          .format(utt_id, take, n, n % REFERENCE_CS,
                                  n % CANDIDATE_CS, (time.time() - t0) * 1000.0,
                                  "ok" if ok else "REDRAW"))
                    if ok:
                        frames = f
                        break
                    manifest["redraws"].append(
                        {"utt": utt_id, "take": take, "frames": n,
                         "residual_ref": n % REFERENCE_CS,
                         "residual_cand": n % CANDIDATE_CS})
                if frames is None:
                    raise SystemExit(
                        "FATAL: {} take {} could not be drawn clear of a "
                        "1/2-frame residual in {} attempts.".format(
                            utt_id, take, MAX_REDRAWS))
                takes[(utt_id, take)] = frames

        print("\n=== render both arms from each captured take ===")
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        for utt_id, _text in sorted(UTTERANCES.items()):
            for take in range(1, TAKES + 1):
                frames = takes[(utt_id, take)]
                if utt_id == CONTROL:
                    # AC #3a #4 — a byte-identical control. Same arm rendered
                    # twice: take 1 at the reference geometry, take 2 at the
                    # candidate's, so the control also re-checks §7's decode
                    # determinism on the fixture's own content.
                    cs = REFERENCE_CS if take == 1 else CANDIDATE_CS
                    a, na, xf = render(service, model, frames, cs)
                    b, nb, _ = render(service, model, frames, cs)
                    if not np.array_equal(a, b):
                        raise SystemExit(
                            "FATAL: the byte-identical control is not "
                            "byte-identical at cs{} ({} vs {} samples). The "
                            "fixture is broken; do not spend a verdict on it."
                            .format(cs, a.size, b.size))
                    ref_pcm, cand_pcm = a, b
                    n_ref = n_cand = na
                    kind = "control_cs{}".format(cs)
                else:
                    ref_pcm, n_ref, xf = render(service, model, frames, REFERENCE_CS)
                    cand_pcm, n_cand, _ = render(service, model, frames, CANDIDATE_CS)
                    kind = "ab"

                stem = "{}-t{}".format(utt_id, take)
                # Files are named by ARM, never by presentation slot. The
                # truth table decides which arm is trial A, balanced 8/8 —
                # Story 20.6's generator learned that per-trial coin flips
                # produced a 12/4 split, and position bias is the largest
                # nuisance variable left on a round whose predicted answer
                # is "equivalent".
                ref_name = "{}-ref-cs{}.wav".format(stem, REFERENCE_CS)
                cand_name = "{}-cand-cs{}.wav".format(stem, CANDIDATE_CS)
                write_wav(OUT_DIR / ref_name, ref_pcm)
                write_wav(OUT_DIR / cand_name, cand_pcm)
                by_trial[stem] = {
                    "reference": {"filename": ref_name},
                    "candidate": {"filename": cand_name},
                }
                row = {
                    "trial": stem, "utterance": utt_id, "take": take,
                    "kind": kind,
                    "A_arm": "reference_cs{}".format(REFERENCE_CS)
                             if kind == "ab" else kind,
                    "B_arm": "candidate_cs{}".format(CANDIDATE_CS)
                             if kind == "ab" else kind,
                    "frames": int(frames.shape[0]),
                    "ref_chunks": n_ref, "cand_chunks": n_cand,
                    "ref_seams": n_ref - 1, "cand_seams": n_cand - 1,
                    "residual_ref": int(frames.shape[0]) % REFERENCE_CS,
                    "residual_cand": int(frames.shape[0]) % CANDIDATE_CS,
                    "len_ref": int(ref_pcm.size), "len_cand": int(cand_pcm.size),
                    "len_equal": bool(ref_pcm.size == cand_pcm.size),
                    "consumer_crossfade": xf,
                    "rms_db_ref": round(active_rms_db(ref_pcm), 3),
                    "rms_db_cand": round(active_rms_db(cand_pcm), 3),
                    "level_delta_db": round(
                        active_rms_db(cand_pcm) - active_rms_db(ref_pcm), 4),
                    "waveform_delta_db": round(
                        waveform_delta_db(ref_pcm, cand_pcm), 2),
                    "byte_identical": bool(np.array_equal(ref_pcm, cand_pcm)),
                }
                manifest["trials"].append(row)
                print("  {:<12} {:<12} frames={:<4d} seams {:>2d} -> {:>2d}  "
                      "len {}  level {:+.3f} dB  waveform {:>7.2f} dB{}".format(
                          stem, kind, row["frames"], row["ref_seams"],
                          row["cand_seams"],
                          "equal" if row["len_equal"] else "DIFFER",
                          row["level_delta_db"], row["waveform_delta_db"],
                          "  BYTE-IDENTICAL" if row["byte_identical"] else ""))
                if abs(row["level_delta_db"]) > 0.2 and kind == "ab":
                    print("      !! level delta above 0.2 dB — investigate "
                          "before the round; levels are NOT normalised here "
                          "because both arms share a take")
    finally:
        await service.stop()

    ab = [r for r in manifest["trials"] if r["kind"] == "ab"]
    manifest["summary"] = {
        "ab_trials": len(ab),
        "control_trials": len(manifest["trials"]) - len(ab),
        "total_ref_seams": sum(r["ref_seams"] for r in ab),
        "total_cand_seams": sum(r["cand_seams"] for r in ab),
        "seam_ratio": round(
            sum(r["cand_seams"] for r in ab)
            / max(sum(r["ref_seams"] for r in ab), 1), 3),
        "worst_level_delta_db": max(abs(r["level_delta_db"]) for r in ab),
        "all_lengths_equal": all(r["len_equal"] for r in ab),
        "redraws": len(manifest["redraws"]),
    }
    _write_truthtable(by_trial, manifest)
    out = SCRIPT_DIR / "20-8-audition-manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + json.dumps(manifest["summary"], indent=2))
    print("wrote {} and {} wav files -> {}".format(
        out.name, 2 * len(manifest["trials"]), OUT_DIR.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
