"""Story 20.5 AC #4 — Phase 3 **round 2**: neutralise the consumer crossfade.

Round 1 was MIXED. State caching was preferred 5-1 wherever the seam was
exposed and the listener's notes were directional and consistent ("B click was
minor compared to A", "A clicks were very minor"), but two single-seam rows —
`m-020-t2` and `s-020-t2` — flagged a click on the candidate that the reference
did not carry. Blocking.

The cause was named in the Phase 2 evidence BEFORE round 1 ran:

    The 64-sample StreamingChunkBuffer crossfade is now the dominant error
    term. It blends different moments (sample n+i with n-64+i), so on
    continuous audio it is a 2.7 ms comb... 2.6x/3.3x worse against ground
    truth.

The crossfade was never a repair for those rows — it was masking, and what it
masked was the cold start. Remove the cold start and the comb is the loudest
thing left. Round 2 removes the comb.

    reference = cs25 + fix + state caching + 64-sample consumer crossfade
    candidate = cs25 + fix + state caching + NO consumer crossfade

**Both arms carry state caching.** The only variable is the consumer
crossfade. Round 1 already scored state-caching against what ships today; this
round scores the crossfade against itself with state caching held fixed, which
is the one thing that can attribute the two blocking rows.

THE CROSSFADE IS DERIVED, NOT SET
---------------------------------
Each arm's crossfade width comes from the SAME rule production uses —
``0 if decode_fn.carries_codec_state else 64`` — rather than being hard-coded
here. So the fixture cannot drift from the shipped wiring, and the "ships
today" arm picks up 64 automatically.

A THIRD ARM IS RENDERED BUT NOT AUDITIONED
------------------------------------------
Every take is also rendered as `cs25fix` — stateless decode with the 64-sample
crossfade, i.e. exactly what ships today. It is NOT in the round-2 truth table
and costs no listening time. It exists because the talker is stochastic: if the
close-out comparison ("state caching + no crossfade" against "ships today")
is ever wanted, it must come from the SAME takes to be worth anything, and
those takes only exist while this process is running. Rendering it now makes
that a truth-table edit rather than a regeneration.

Everything else is round 1's protocol unchanged: one talker run per pair, the
same seven utterances as Story 20.4 rounds 1-4, two takes each, a
byte-identical zero-seam control, levels not normalised.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-5-regen-audition-fixture-r2.py

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
OUT_DIR = SCRIPT_DIR / "20-5-perceptual-fixtures-r2"

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
CS = codec_token_streamer.DEFAULT_CHUNK_SIZE
LA = codec_token_streamer.DEFAULT_LOOKAHEAD

# Round 2 arms. Both carry state caching; the crossfade is the variable.
REFERENCE = "statexf64"        # state caching + the 64-sample crossfade
CANDIDATE = "statexf0"         # state caching, crossfade neutralised
# Rendered from the same takes but NOT auditioned this round: what ships
# today. Kept so a later close-out comparison can reuse these takes.
SHIPS_TODAY = "cs25fix"
TAKES = 2

# The same seven utterances as Story 20.4 rounds 1-4, so all rounds stay
# comparable. s-022 is the most informative row in the set: clean at cs25,
# blocking at cs10, and plosive-dense — precisely the content the transient-
# doubling mechanism (20.4 evidence SS13.3) predicts is most seam-sensitive.
UTTERANCES = {
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


def _preflight() -> None:
    cs = codec_token_streamer.DEFAULT_CHUNK_SIZE
    la = codec_token_streamer.DEFAULT_LOOKAHEAD
    if (cs, la) != (25, 5):
        raise SystemExit(
            "FATAL: committed geometry is ({}, {}), expected (25, 5). Story "
            "20.5 does not retune chunk size (AC #3); AC #5 reopens it as its "
            "own story.".format(cs, la)
        )
    for attr in ("_CODEC_SAMPLES_PER_FRAME", "_CODEC_EDGE_LOSS_SAMPLES",
                 "_OVERLAP_ADD_SAMPLES"):
        if not hasattr(streaming_decoder, attr):
            raise SystemExit(
                "FATAL: streaming_decoder is missing {} — the Story 20.4 seam "
                "fix must be present in BOTH arms or this round repeats round "
                "2's two-variable confound.".format(attr)
            )
    print("preflight OK")
    print("  geometry held at ({}, {}) on both arms".format(cs, la))
    print("  seam fix present on both arms: samples/frame={} edge={} ola={}"
          .format(streaming_decoder._CODEC_SAMPLES_PER_FRAME,
                  streaming_decoder._CODEC_EDGE_LOSS_SAMPLES,
                  streaming_decoder._OVERLAP_ADD_SAMPLES))
    print("  consumer crossfade present on both arms: {} samples, watermark "
          "{} ms".format(H.CROSSFADE_SAMPLES, H.WATERMARK_MS))
    # Round 2 depends on the shipped wiring choosing 0 for a continuous
    # stream. Pin it here so the fixture cannot silently diverge from it.
    import inspect as _inspect
    from myvoice.services.qwen_tts_service import QwenTTSService
    from myvoice.services.audio_coordinator import AudioCoordinator
    if "crossfade_samples" not in _inspect.signature(
            AudioCoordinator.start_streaming_session).parameters:
        raise SystemExit(
            "FATAL: AudioCoordinator.start_streaming_session has no "
            "crossfade_samples parameter — the Phase 4 wiring is not present, "
            "so the candidate arm this round auditions is not reachable in "
            "production.")
    if not hasattr(QwenTTSService, "progressive_stream_is_continuous"):
        raise SystemExit(
            "FATAL: QwenTTSService does not declare stream continuity — the "
            "consumer has no way to know when to neutralise the crossfade.")
    print("  Phase 4 wiring present: producer declares continuity, consumer "
          "passes crossfade_samples")
    print("  reference={} (crossfade {})  candidate={} (crossfade 0)"
          .format(REFERENCE, H.CROSSFADE_SAMPLES, CANDIDATE))
    print("  -> the ONLY variable is the consumer crossfade; both arms carry "
          "codec state caching")


def render(chunks, decode_fn):
    """Decode captured token chunks through the real worker + real consumer
    buffer, exactly as production does, and return (int16 stream, n_posted,
    crossfade_used).

    The crossfade width is DERIVED from the decode_fn by the same rule the
    shipped wiring uses — ``QwenTTSService._generate_true_stream`` sets
    ``_progressive_stream_continuous`` from ``decode_fn.carries_codec_state``,
    ``MyVoiceApp._handle_progressive_chunk_async`` turns that into
    ``crossfade_samples=0``, and ``AudioCoordinator.start_streaming_session``
    passes it to the buffer. Deriving it here rather than hard-coding it means
    this fixture cannot render an arm production would not produce.
    """
    streamer = CodecTokenStreamer(chunk_size=CS, lookahead=LA)
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
    for chunk in chunks:
        streamer.queue.put(chunk)
    streamer.queue.put(END_OF_STREAM)
    worker.start()
    worker.join(timeout=300.0)
    done.wait(timeout=5.0)

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


def render_with_crossfade(chunks, decode_fn, crossfade):
    """Round 2's reference arm: state caching WITH the 64-sample crossfade —
    i.e. round 1's candidate, the configuration that produced the two blocking
    rows. Production would never build this combination now, so it is the one
    arm whose crossfade must be forced rather than derived."""
    streamer = CodecTokenStreamer(chunk_size=CS, lookahead=LA)
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
    for chunk in chunks:
        streamer.queue.put(chunk)
    streamer.queue.put(END_OF_STREAM)
    worker.start()
    worker.join(timeout=300.0)
    done.wait(timeout=5.0)

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
    """RMS over the frames above -50 dBFS, so leading silence does not
    dominate the level comparison."""
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

    # Capture with the state cache OFF, so the talker runs under exactly
    # today's shipping conditions and the reference arm is what ships today
    # in every respect, not merely in its decode.
    os.environ["MYVOICE_CODEC_STATE_CACHE"] = "0"
    captured = []
    real_builder = service._build_true_stream_decode_fn

    def capturing_builder(model, *args, **kwargs):
        inner = real_builder(model, *args, **kwargs)

        def _decode(chunk):
            captured.append(torch.as_tensor(chunk).detach().cpu().clone())
            return inner(chunk)
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
            takes[(utt_id, take)] = [c.clone() for c in captured]
            frames = sum(c.shape[0] for c in captured) - LA * (len(captured) - 1)
            print("  captured {:<6} take {}  {:>2d} chunks  {:>3d} frames  "
                  "{:>6.0f} ms".format(utt_id, take, len(captured), frames,
                                       (time.time() - t0) * 1000.0))

    service._build_true_stream_decode_fn = real_builder

    # Build both decode paths from ONE model load, one compiled state.
    model = service._model_registry.get_loaded_model()
    os.environ["MYVOICE_CODEC_STATE_CACHE"] = "0"
    stateless = service._build_true_stream_decode_fn(model, CS, LA)
    os.environ.pop("MYVOICE_CODEC_STATE_CACHE", None)
    stateful = service._build_true_stream_decode_fn(model, CS, LA)

    if getattr(stateless, "carries_codec_state", False):
        raise SystemExit("FATAL: the reference arm is carrying codec state.")
    if not getattr(stateful, "carries_codec_state", False):
        raise SystemExit(
            "FATAL: the candidate arm declined to engage codec state caching. "
            "Run 20-5-phase2-verify.py to see why; there is no round to run "
            "until it does.")

    print("\n=== rendering three arms from the captured takes ===")
    print("    {:<12} state caching, crossfade forced to {} (round 1's "
          "candidate)".format(REFERENCE, H.CROSSFADE_SAMPLES))
    print("    {:<12} state caching, crossfade derived -> 0 (the fix)"
          .format(CANDIDATE))
    print("    {:<12} stateless, crossfade derived -> {} (ships today; "
          "rendered, NOT auditioned)".format(SHIPS_TODAY, H.CROSSFADE_SAMPLES))
    level_deltas = []
    manifest = {}
    for (utt_id, take), chunks in sorted(takes.items()):
        row = {}
        arms = (
            (REFERENCE, stateful, H.CROSSFADE_SAMPLES),
            (CANDIDATE, stateful, None),
            (SHIPS_TODAY, stateless, None),
        )
        for label, fn, forced in arms:
            if hasattr(fn, "reset"):
                fn.reset()
            if forced is None:
                samples, n_posted, xf = render(chunks, fn)
            else:
                samples, n_posted, xf = render_with_crossfade(chunks, fn, forced)
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
        # Sanity: the derived rule must have produced what this round claims.
        assert row[CANDIDATE]["crossfade_samples"] == 0
        assert row[SHIPS_TODAY]["crossfade_samples"] == H.CROSSFADE_SAMPLES
        delta_db = (row[CANDIDATE]["active_rms_db"]
                    - row[REFERENCE]["active_rms_db"])
        delta_samples = row[CANDIDATE]["samples"] - row[REFERENCE]["samples"]
        level_deltas.append(delta_db)
        row["level_delta_db"] = round(delta_db, 4)
        row["length_delta_samples"] = delta_samples
        manifest["{}-t{}".format(utt_id, take)] = row
        print("  {:<6} take {}  {:>6.2f}s  {:>2d} chunks  level delta "
              "{:+.4f} dB  length delta {:+d} samples".format(
                  utt_id, take, row[REFERENCE]["seconds"],
                  row[REFERENCE]["posted_chunks"], delta_db, delta_samples))

    worst = max(abs(d) for d in level_deltas)
    print("\n  worst within-pair level delta: {:.4f} dB".format(worst))
    if worst > 0.2:
        print("  WARNING: > 0.2 dB. The two arms share a take, so a level "
              "difference this large is caused by the change and should be "
              "understood BEFORE listening — level is not the variable under "
              "test and a listener will hear it as one.")

    _write_truthtable(manifest)
    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    await service.stop()
    print("\nFixture written to {}".format(OUT_DIR))
    print("Next: python310\\python.exe _bmad-output\\implementation-artifacts"
          "\\20-5-l1-audition-helper.py L1 r2")
    return 0


def _write_truthtable(manifest) -> None:
    """Randomise which arm is trial A per trial, from a fixed seed, so the
    mapping is reproducible from this generator but not inferable from
    listening order."""
    rng = random.Random(20050902)
    table = {
        "_meta": {
            "story": "20.5",
            "round": 2,
            "candidate": CANDIDATE,
            "reference": REFERENCE,
            "candidate_desc": "state caching, consumer crossfade "
                              "NEUTRALISED (0 samples)",
            "reference_desc": "state caching, 64-sample consumer crossfade "
                              "— round 1's candidate, the configuration that "
                              "produced the two blocking rows",
            "isolates": "the 64-sample consumer crossfade ONLY. Both arms "
                        "carry codec state caching, the same geometry (25, 5) "
                        "and the same 1024-sample decoder seam blend, and BOTH "
                        "ARMS OF EACH PAIR ARE DECODED FROM THE SAME TALKER "
                        "RUN — identical wording, prosody and duration.",
            "not_auditioned_arm": SHIPS_TODAY,
            "not_auditioned_desc": "stateless decode + 64-sample crossfade — what ships today. Rendered from the same takes so a later close-out comparison does not need a regeneration; not in this round's trials.",
            "takes_per_utterance": TAKES,
            "variance_note": "Within a pair there is no take-to-take variance "
                             "to average over: the two files are the same take "
                             "decoded two ways. The second take exists to "
                             "sample the CONTENT lottery (whether a held vowel "
                             "or plosive lands on a boundary), not to average "
                             "arm variance.",
        },
        "L1": {},
    }
    for trial in sorted(manifest):
        first, second = ((REFERENCE, CANDIDATE) if rng.random() < 0.5
                         else (CANDIDATE, REFERENCE))
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
