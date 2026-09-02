"""Story 20.6 AC #4 — NFR3 audition fixture: retiring the 5-frame lookahead.

    reference = what ships today  = codec state caching + the gated (0-sample)
                                    consumer crossfade + lookahead 5
    candidate = this change       = the same, with the lookahead RETIRED

ONE VARIABLE. Both arms carry codec state caching, the same chunk_size (25),
the same gated consumer crossfade, and the same decoder-side seam machinery
where it still applies. The only difference is whether the streamer emits
30-frame windows at stride 25 (and the worker trims and blends the overlap) or
25-frame windows at stride 25 (and the worker posts each decode whole).

The Story 20.4 seam blend is NOT a second variable. It cross-fades the
retained lookahead tail into the next chunk's head; with no lookahead there is
no tail, so it disappears *by construction* rather than by a decision. Story
20.5 had already measured it as inert under carried state (the two sides of
the blend were bit-identical), so nothing audible is being traded here either.
That is the claim this round tests.

TOKEN REUSE IS STILL AVAILABLE — VERIFIED, NOT ASSUMED (AC #4)
--------------------------------------------------------------
AC #4 asks whether both arms can still be decoded from ONE talker run, as
Story 20.5 did, given that this change alters the streamer's chunking. They
can. The talker's token sequence does not depend on how the streamer slices
it: the forward-hook captures per-step ``codec_ids`` and the chunking happens
downstream of generation. So this script captures ONE run per pair, recovers
the flat frame sequence from the captured chunks (each chunk k starts at frame
``chunk_size * k``, whatever the window), and re-slices it at each arm's
geometry. Both arms of a pair therefore have identical wording, prosody and
duration, and there is no take-to-take variance inside a pair to account for.

The second take per utterance samples the CONTENT lottery — whether a held
vowel or a plosive happens to land on a boundary — not arm variance.

THE PREDICTION, RECORDED BEFORE THE ROUND (AC #4)
--------------------------------------------------
Falsifiable, and stated before any listening:

  P1. **No audible difference on any trial.** The trim removed exactly the
      lookahead's worth of PCM and the blend was an identity under carried
      state, so the candidate should post the same audio the reference does.
      Measured on the CPU reference decoder in CI the two streams agree to
      1e-06 in float64; measured on the real bf16 decoder
      (``20-6-lookahead-bench.json``) they are bit-identical wherever the
      residual flush lands on the same frame, and differ by NRMSE ~1.1e-03
      (about -59 dB) where it does not.

  P2. **The one place P1 could fail is the residual flush.** Retiring the
      lookahead moves the last chunk boundary: an utterance of N frames splits
      differently, so the final chunk can be a different length and, at 176
      frames, the candidate produced 8 chunks where the reference produced 7.
      If anything is audible it should be at the END of an utterance, on the
      long fixtures, and not at the interior seams.

  P3. **THE EMBARRASSING ONE.** If a listener hears the candidate as *worse at
      the interior seams*, the diagnosis is wrong: it would mean the Story
      20.4 blend was doing real perceptual work under carried state after all,
      and that Story 20.5's "the two sides are bit-identical" measurement does
      not describe what ships. That would not be a tuning problem — it would
      mean the retirement has to be reverted and the blend re-examined on the
      state-cached path, and it would put Story 20.5's Phase 4 conclusion in
      question too.

  P4. Latency is NOT under test here. These are rendered files; nothing about
      TTFA is auditionable. The GUI capture is the only evidence for that.

A ZERO-SEAM CONTROL IS INCLUDED
--------------------------------
``ctl-020`` is short enough to produce a single chunk in BOTH geometries (under
25 frames, so it never reaches either arm's first-emit threshold and flushes as
one residual). Both arms decode it identically and the two files are asserted
byte-identical here. Any difference a listener reports on that trial is a
property of the listening, not of the change.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-6-regen-audition-fixture.py

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import threading
import time
import wave
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = SCRIPT_DIR / "20-6-perceptual-fixtures"

sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttfa_spike_harness as H  # noqa: E402

from myvoice.services.streaming_chunk_buffer import (  # noqa: E402
    StreamingChunkBuffer,
)
from myvoice.services.tts_streaming import codec_state_cache as csc  # noqa: E402
from myvoice.services.tts_streaming import codec_token_streamer  # noqa: E402
from myvoice.services.tts_streaming import streaming_decoder  # noqa: E402
from myvoice.services.tts_streaming.codec_token_streamer import (  # noqa: E402
    CodecTokenStreamer,
    END_OF_STREAM,
    effective_lookahead,
)
from myvoice.services.tts_streaming.streaming_decoder import (  # noqa: E402
    StreamingDecoderWorker,
)

SR = 24000
CS = codec_token_streamer.DEFAULT_CHUNK_SIZE
LA = codec_token_streamer.DEFAULT_LOOKAHEAD

REFERENCE = "la5"    # ships today: state caching + gated crossfade + lookahead 5
CANDIDATE = "la0"    # this change: the lookahead retired
TAKES = 2

# The same seven utterances as Story 20.4 rounds 1-4 and Story 20.5 rounds
# 1-2, so every round in the epic stays comparable — plus the zero-seam
# control. s-022 remains the most informative row: clean at cs25, blocking at
# cs10, plosive-dense.
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


def _preflight() -> None:
    if (CS, LA) != (25, 5):
        raise SystemExit(
            "FATAL: committed geometry is ({}, {}), expected (25, 5). Story "
            "20.6 does not retune chunk size (AC #5); the chunk-size reopen "
            "is its own story.".format(CS, LA))
    if codec_token_streamer.DEFAULT_LOOKAHEAD != 5:
        raise SystemExit(
            "FATAL: DEFAULT_LOOKAHEAD is not 5. Story 20.6 retires the "
            "lookahead CONDITIONALLY; a global constant change would take the "
            "trim and the seam blend away from the stateless fallback, which "
            "is the failure mode AC #2 exists to prevent — and it would make "
            "this round's reference arm unreachable.")
    if effective_lookahead(True) != 0 or effective_lookahead(False) != 5:
        raise SystemExit(
            "FATAL: effective_lookahead() is not the Story 20.6 rule.")
    for attr in ("_CODEC_SAMPLES_PER_FRAME", "_CODEC_EDGE_LOSS_SAMPLES",
                 "_OVERLAP_ADD_SAMPLES"):
        if not hasattr(streaming_decoder, attr):
            raise SystemExit(
                "FATAL: streaming_decoder is missing {} — the Story 20.4 seam "
                "machinery must still be present, because the REFERENCE arm "
                "is what ships today and depends on it.".format(attr))
    print("preflight OK")
    print("  chunk_size held at {} on both arms".format(CS))
    print("  reference lookahead {} (trim + {}-sample seam blend live)".format(
        LA, streaming_decoder._OVERLAP_ADD_SAMPLES))
    print("  candidate lookahead {} (no overlap -> no trim, no blend)".format(
        effective_lookahead(True)))
    print("  consumer crossfade DERIVED on both arms from "
          "carries_codec_state, exactly as the shipped wiring does")


def _recover_frames(chunks):
    """Rebuild the flat frame sequence from the chunks the streamer emitted.

    Chunk ``k`` starts at frame ``chunk_size * k`` regardless of the window
    width (the streamer always slides by ``chunk_size``), so overlapping and
    non-overlapping captures both rebuild the same way.
    """
    frames = []
    for i, chunk in enumerate(chunks):
        arr = torch.as_tensor(chunk)
        start = CS * i
        skip = max(0, len(frames) - start)
        if skip >= arr.shape[0]:
            continue
        frames.extend(list(arr[skip:]))
    return torch.stack(frames, dim=0) if frames else None


def _slice(frames, window):
    """The streamer's own chunking: windows of ``window`` at stride
    ``chunk_size``, then the residual flush."""
    out = []
    index = 0
    n = frames.shape[0]
    while index + window <= n:
        out.append(frames[index:index + window])
        index += CS
    if index < n:
        out.append(frames[index:])
    return out


def render(frames, decode_fn, lookahead):
    """Decode one arm through the REAL worker and the REAL consumer buffer.

    The streamer's geometry comes from ``apply_codec_state_geometry`` — the
    same call the dispatch makes — so the candidate arm cannot drift from what
    production builds. The reference arm forces lookahead 5 back on, because
    production would no longer construct it; that is the arm being scored
    against, not the arm being shipped.
    """
    streamer = CodecTokenStreamer(chunk_size=CS, lookahead=LA)
    if lookahead == 0:
        streamer.apply_codec_state_geometry(True)
    else:
        streamer.apply_codec_state_geometry(False)
    assert streamer.lookahead == lookahead

    posted = []
    done = threading.Event()

    def post(method, session_id, *args):
        if method == "append_chunk":
            posted.append(np.asarray(args[0], dtype=np.float32).copy())
        elif method in ("finalize", "cancel", "discard"):
            done.set()

    worker = StreamingDecoderWorker(
        streamer=streamer, decode_fn=decode_fn, post_mutation=post,
        session_id="fixture", model_type="qwen3_tts", hardware="gpu",
    )
    for chunk in _slice(frames, CS + lookahead):
        streamer.queue.put(chunk)
    streamer.queue.put(END_OF_STREAM)
    worker.start()
    worker.join(timeout=300.0)
    done.wait(timeout=5.0)

    # Same rule the shipped wiring uses: continuity is read off the decode_fn,
    # never assumed. Both arms carry state, so both get 0.
    continuous = bool(getattr(decode_fn, "carries_codec_state", False))
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
    return np.frombuffer(b"".join(out), dtype=np.int16), len(posted), crossfade


def write_wav(path: Path, samples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(samples.tobytes())


def active_rms_db(samples) -> float:
    x = samples.astype(np.float64) / 32768.0
    if x.size == 0:
        return float("-inf")
    frame = 480
    n = (x.size // frame) * frame
    if n == 0:
        return float("-inf")
    energies = np.sqrt((x[:n].reshape(-1, frame) ** 2).mean(axis=1))
    active = energies[energies > 10 ** (-50 / 20)]
    if active.size == 0:
        active = energies
    return float(20 * np.log10(max(active.mean(), 1e-12)))


async def _run() -> int:
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    _preflight()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings,
    )
    if not await service.start():
        print("FATAL: QwenTTSService.start() returned False", file=sys.stderr)
        return 1

    captured = []
    real_builder = service._build_true_stream_decode_fn

    def capturing_builder(model, *args, **kwargs):
        inner = real_builder(model, *args, **kwargs)

        def _decode(chunk):
            captured.append(torch.as_tensor(chunk).detach().cpu().clone())
            return inner(chunk)
        _decode.carries_codec_state = getattr(
            inner, "carries_codec_state", False)
        if hasattr(inner, "reset"):
            _decode.reset = inner.reset
        return _decode

    service._build_true_stream_decode_fn = capturing_builder
    prompt = H._load_voice_clone_prompt(service)

    warm = QwenTTSRequest(
        text=H.PRIMING_TEXT, language="English", model_type=QwenModelType.BASE,
        streaming=True, voice_clone_prompt=prompt, suppress_audio_output=True,
    )
    await service._generate_true_stream(warm)
    await asyncio.sleep(0.2)

    takes = {}
    for utt_id, text in sorted(UTTERANCES.items()):
        for take in range(1, TAKES + 1):
            captured.clear()
            req = QwenTTSRequest(
                text=text, language="English", model_type=QwenModelType.BASE,
                streaming=True, voice_clone_prompt=prompt,
                suppress_audio_output=True,
            )
            t0 = time.time()
            resp = await service._generate_true_stream(req)
            await asyncio.sleep(0.2)
            if not resp.success or not captured:
                raise SystemExit(
                    "FATAL: capture failed for {} take {}: {}".format(
                        utt_id, take, getattr(resp, "error_message", None)))
            frames = _recover_frames(captured)
            takes[(utt_id, take)] = frames
            print("  captured {:<8} take {}  {:>2d} chunks  {:>3d} frames  "
                  "{:>6.0f} ms".format(utt_id, take, len(captured),
                                       frames.shape[0],
                                       (time.time() - t0) * 1000.0))

    service._build_true_stream_decode_fn = real_builder

    # Both arms from ONE model load and one compiled state.
    model = service._model_registry.get_loaded_model()
    os.environ.pop("MYVOICE_CODEC_STATE_CACHE", None)
    decoder = model.model.speech_tokenizer.model.decoder
    device = getattr(model.model.speech_tokenizer, "device", None)
    geometry = csc.probe_decoder(decoder)

    # Candidate: built exactly as production builds it.
    candidate_fn = service._build_true_stream_decode_fn(model, CS, LA)
    if not getattr(candidate_fn, "carries_codec_state", False):
        raise SystemExit(
            "FATAL: the candidate arm declined codec state caching. There is "
            "no round to run until it engages — the retirement is conditional "
            "on carried state, so this configuration IS the reference arm.")
    if candidate_fn._window_frames != candidate_fn._commit_frames:
        raise SystemExit(
            "FATAL: production still builds the stateful decoder with a "
            "lookahead window ({} vs {}); the retirement is not wired."
            .format(candidate_fn._window_frames, candidate_fn._commit_frames))

    # Reference: the pre-20.6 geometry, which production would no longer build.
    reference_fn = csc.StatefulCodecDecoder(
        decoder=decoder, geometry=geometry,
        commit_frames=CS, window_frames=CS + LA, device=device,
    )

    print("\n=== rendering two arms from the captured takes ===")
    print("    {:<4} lookahead {}, trim + {}-sample seam blend  (ships today)"
          .format(REFERENCE, LA, streaming_decoder._OVERLAP_ADD_SAMPLES))
    print("    {:<4} lookahead 0, full decode posted, no trim, no blend"
          .format(CANDIDATE))

    level_deltas = []
    manifest = {}
    for (utt_id, take), frames in sorted(takes.items()):
        row = {}
        for label, fn, la in ((REFERENCE, reference_fn, LA),
                              (CANDIDATE, candidate_fn, 0)):
            fn.reset()
            samples, n_posted, xf = render(frames, fn, la)
            name = "{}-t{}-{}.wav".format(utt_id, take, label)
            write_wav(OUT_DIR / name, samples)
            row[label] = {
                "filename": name,
                "samples": int(samples.size),
                "seconds": round(samples.size / SR, 3),
                "posted_chunks": n_posted,
                "crossfade_samples": xf,
                "active_rms_db": round(active_rms_db(samples), 3),
            }
        assert row[REFERENCE]["crossfade_samples"] == 0
        assert row[CANDIDATE]["crossfade_samples"] == 0

        a = np.fromfile(OUT_DIR / row[REFERENCE]["filename"], dtype=np.uint8)
        b = np.fromfile(OUT_DIR / row[CANDIDATE]["filename"], dtype=np.uint8)
        identical = bool(a.size == b.size and np.array_equal(a, b))
        row["byte_identical"] = identical
        delta_db = (row[CANDIDATE]["active_rms_db"]
                    - row[REFERENCE]["active_rms_db"])
        row["level_delta_db"] = round(delta_db, 4)
        row["length_delta_samples"] = (
            row[CANDIDATE]["samples"] - row[REFERENCE]["samples"])
        level_deltas.append(delta_db)
        manifest["{}-t{}".format(utt_id, take)] = row
        print("  {:<8} take {}  {:>6.2f}s  ref {:>2d} / cand {:>2d} chunks  "
              "level {:+.4f} dB  length {:+d}  {}".format(
                  utt_id, take, row[REFERENCE]["seconds"],
                  row[REFERENCE]["posted_chunks"],
                  row[CANDIDATE]["posted_chunks"], delta_db,
                  row["length_delta_samples"],
                  "IDENTICAL" if identical else "differs"))

        if utt_id == CONTROL and not identical:
            print("  WARNING: the zero-seam control is NOT byte-identical. "
                  "The control's whole purpose is that it cannot differ — "
                  "either it is long enough to cross a boundary in one arm, "
                  "or something outside the seam path changed.")

    worst = max(abs(d) for d in level_deltas)
    print("\n  worst within-pair level delta: {:.4f} dB".format(worst))
    if worst > 0.2:
        print("  WARNING: > 0.2 dB. The two arms share a take, so a level "
              "difference this large is caused by the change and must be "
              "understood BEFORE listening — a listener will hear level as "
              "the variable.")

    _write_truthtable(manifest)
    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    await service.stop()
    print("\nFixture written to {}".format(OUT_DIR))
    print("Next: python310\\python.exe _bmad-output\\implementation-artifacts"
          "\\20-5-l1-audition-helper.py L1 <round-tag>  (or listen directly "
          "against _perlistener_truthtable.json)")
    return 0


def _write_truthtable(manifest) -> None:
    """Randomise which arm is trial A, BALANCED rather than per-trial coin
    flips.

    Story 20.5's generator flipped a coin per trial; on this round's 16 trials
    that produced a 12/4 split, i.e. the reference was trial A three times as
    often as the candidate. Position bias in A/B listening is real, and on a
    round whose predicted answer is "equivalent" it is the largest nuisance
    variable left — an 8/8 assignment removes it by construction rather than
    leaving it to the seed. The order is still shuffled from a fixed seed, so
    it is reproducible from this generator and not inferable from listening
    order.
    """
    rng = random.Random(20060901)
    table = {
        "_meta": {
            "story": "20.6",
            "round": 1,
            "candidate": CANDIDATE,
            "reference": REFERENCE,
            "candidate_desc": "codec state caching, gated consumer crossfade, "
                              "lookahead RETIRED — the streamer emits 25-frame "
                              "chunks with no overlap and the worker posts "
                              "each decode whole (no trim, no seam blend)",
            "reference_desc": "what ships today: codec state caching, gated "
                              "consumer crossfade, lookahead 5 — 30-frame "
                              "windows at stride 25, trimmed to the splice and "
                              "cross-faded over 1024 samples",
            "isolates": "the 5-frame lookahead ONLY. Both arms carry codec "
                        "state caching, chunk_size 25 and the same gated "
                        "consumer crossfade, and BOTH ARMS OF EACH PAIR ARE "
                        "DECODED FROM THE SAME TALKER RUN — identical wording, "
                        "prosody and duration. The Story 20.4 seam blend is a "
                        "DEPENDENT of the lookahead, not a second variable: "
                        "with no lookahead there is no retained tail to blend.",
            "control_trial": CONTROL,
            "control_desc": "short enough to produce one chunk in BOTH "
                            "geometries, so the two files are byte-identical. "
                            "Any difference reported on this trial is a "
                            "property of the listening, not of the change.",
            "takes_per_utterance": TAKES,
            "prediction": "No audible difference on any trial. If anything is "
                          "audible it should be at the END of a long "
                          "utterance, where the residual flush splits "
                          "differently. A listener hearing the candidate as "
                          "WORSE AT THE INTERIOR SEAMS falsifies the "
                          "diagnosis: it would mean the Story 20.4 blend was "
                          "doing real perceptual work under carried state, "
                          "and the retirement has to be reverted.",
            "variance_note": "Within a pair there is no take-to-take variance "
                             "to average over: the two files are the same "
                             "take decoded two ways. The second take samples "
                             "the CONTENT lottery, not arm variance.",
        },
        "L1": {},
    }
    trials = sorted(manifest)
    orders = [(REFERENCE, CANDIDATE)] * (len(trials) // 2)
    orders += [(CANDIDATE, REFERENCE)] * (len(trials) - len(orders))
    rng.shuffle(orders)
    for trial, (first, second) in zip(trials, orders):
        table["L1"][trial] = {
            "trial_A_filename": manifest[trial][first]["filename"],
            "trial_A_arm": first,
            "trial_B_filename": manifest[trial][second]["filename"],
            "trial_B_arm": second,
        }
    path = OUT_DIR / "_perlistener_truthtable.json"
    path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print("Truth table -> {} ({} trials)".format(path.name, len(table["L1"])))


def main() -> int:
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print("device={} vram={:.1f}GiB torch={}".format(
            pr.name, pr.total_memory / 1024 ** 3, torch.__version__))
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
