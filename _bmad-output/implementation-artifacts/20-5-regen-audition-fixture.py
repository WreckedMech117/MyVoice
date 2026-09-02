"""Story 20.5 AC #4 — Phase 3 audition fixture: does the ear notice that the
chunk boundaries now carry real codec state?

    reference = cs25 + the Story 20.4 seam fix                <- what ships today
    candidate = cs25 + the Story 20.4 seam fix + state caching

Both arms carry the seam fix and the 64-sample consumer crossfade. The
geometry is `chunk_size=25, lookahead=5` on both, unchanged and unchangeable
by this script. **The only variable is whether the decoder carries codec
state across the chunk boundary.**

THE VARIANCE PROBLEM, AND WHY THIS ROUND DOES NOT HAVE IT
---------------------------------------------------------
Story 20.4 SS17 found the same configuration flagging differently across
takes, which is why AC #4 demands the round either use multiple takes or
admit it can only detect a large effect. This round takes a third option that
Story 20.4 could not:

    **Both arms of every pair are rendered from ONE talker run.**

The talker runs once per take; its codec-token chunks are captured and then
decoded twice — once through the stateless decode path, once through the
state-cached one — each driven by a real ``StreamingDecoderWorker`` and
consumed by a real ``StreamingChunkBuffer``. So within a pair the wording,
the prosody, the pauses and the total duration are *identical to the sample*.
Nothing upstream of the decoder differs, because nothing upstream of the
decoder is touched by this story. Story 20.4's arms could not do this: a
chunk-size change alters what the streamer emits and therefore perturbs the
talker, so its two arms were necessarily different takes and take-to-take
variance was in the signal.

The consequence is that a single pair per utterance is already sufficient for
*attribution* — any difference the listener hears is caused by the decode,
because there is nothing else it could be caused by.

**Two takes per utterance are still generated**, for a different reason: to
sample the CONTENT lottery. Whether a given take happens to put a held vowel
or a plosive across a chunk boundary is luck, and one take might contain no
seam-sensitive material at all. Two takes, each rendered both ways, gives 14
independent trials over 7 utterances.

WHAT IS NOT NORMALISED, AND WHY
--------------------------------
Story 20.4 round 4 loudness-normalised its files because its two arms were
different takes that happened to differ by 8 dB. Here the two arms share a
take, so any level difference between them is CAUSED BY THE CHANGE and would
be a finding. This script measures the within-pair level delta and prints it
rather than normalising it away. Expect ~0.00 dB; anything above 0.2 dB is
worth investigating before the audition runs.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-5-regen-audition-fixture.py

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
OUT_DIR = SCRIPT_DIR / "20-5-perceptual-fixtures"

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

REFERENCE = "cs25fix"          # what ships today: stateless decode
CANDIDATE = "cs25fixstate"     # + codec state caching
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
    print("  reference={}  candidate={}  -> the ONLY variable is codec state"
          .format(REFERENCE, CANDIDATE))


def render(chunks, decode_fn):
    """Decode captured token chunks through the real worker + real consumer
    buffer, exactly as production does, and return the int16 stream."""
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
        watermark_ms=H.WATERMARK_MS, crossfade_samples=H.CROSSFADE_SAMPLES,
        sample_rate=SR, channels=1, sample_width=2,
    )
    out = []
    for seg in posted:
        payload = (np.clip(seg, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        for released in buf.push(payload, is_final=False):
            out.append(released)
    for released in buf.flush_remaining():
        out.append(released)
    return np.frombuffer(b"".join(out), dtype=np.int16), len(posted)


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

    print("\n=== rendering both arms from the captured takes ===")
    level_deltas = []
    manifest = {}
    for (utt_id, take), chunks in sorted(takes.items()):
        row = {}
        for label, fn in ((REFERENCE, stateless), (CANDIDATE, stateful)):
            if hasattr(fn, "reset"):
                fn.reset()
            samples, n_posted = render(chunks, fn)
            name = "{}-t{}-{}.wav".format(utt_id, take, label)
            write_wav(OUT_DIR / name, samples)
            row[label] = {
                "filename": name,
                "samples": int(samples.size),
                "seconds": round(samples.size / SR, 3),
                "posted_chunks": n_posted,
                "active_rms_db": round(active_rms_db(samples), 3),
            }
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
          "\\20-5-l1-audition-helper.py L1")
    return 0


def _write_truthtable(manifest) -> None:
    """Randomise which arm is trial A per trial, from a fixed seed, so the
    mapping is reproducible from this generator but not inferable from
    listening order."""
    rng = random.Random(20050901)
    table = {
        "_meta": {
            "story": "20.5",
            "round": 1,
            "candidate": CANDIDATE,
            "reference": REFERENCE,
            "candidate_desc": "cs25 + Story 20.4 seam fix + codec state "
                              "caching across chunk boundaries",
            "reference_desc": "cs25 + Story 20.4 seam fix — what ships today",
            "isolates": "codec state caching only. Both arms carry the same "
                        "geometry (25, 5), the same 1024-sample decoder seam "
                        "blend and the same 64-sample consumer crossfade, and "
                        "BOTH ARMS OF EACH PAIR ARE DECODED FROM THE SAME "
                        "TALKER RUN — identical wording, prosody and duration.",
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
