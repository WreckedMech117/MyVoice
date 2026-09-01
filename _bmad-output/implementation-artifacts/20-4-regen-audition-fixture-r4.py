"""Story 20.4 AC #5 - round-4 fixture: does chunk_size = 15 survive the ear?

Round 3 settled that the seam fix is good at cs25 and that the geometry was
what failed at cs10. That leaves the interpolation question Commander wants
answered: chunk_size = 15 sits between them - 1.67x the seam density of 25
against cs10's 2.5x - and Story 20.1 measured 1,157 ms TTFA there.

Round 2's flaw was changing geometry AND stitching against a reference that
had neither. This round changes ONE thing:

    reference = cs25 + the seam fix     <- the NEW baseline
    candidate = cs15 + the seam fix

Both arms carry the fix, so anything the listener hears is attributable to
the geometry alone.

**The reference is REGENERATED, not reused.** Rounds 1-3 used round 1's cs25
files as the anchor, but those are PRE-fix and are the wrong baseline now:
comparing cs15+fix against cs25-pre-fix would reintroduce exactly the
two-variable confound round 2 died of. Both arms are therefore generated
here, in one process, from one model load, under one compiled state - so the
only difference between them is the streamer geometry.

Same 7 utterances as rounds 1-3 so all four rounds stay comparable. That
includes **s-022** ("Bit, bat, bot, but, bet"), which is the most
informative row in the set: clean at cs25, blocking at cs10, and
plosive-dense, which is precisely the content the transient-doubling
mechanism (evidence SS13.3) predicts is most sensitive to seam density.

DEFAULT_CHUNK_SIZE is NOT modified. It is committed at 25 and stays there
until the ear says otherwise; both geometries are set in-process via the
harness rebinder, and the preflight refuses to run if the committed
constants have drifted.

Rounds 1-3 are left untouched.

Usage:
    python310/python.exe _bmad-output/implementation-artifacts/20-4-regen-audition-fixture-r4.py

Working file - gitignored under _bmad-output/; force-add per
memory/git_repo_state.md.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
import wave
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
R4_DIR = SCRIPT_DIR / "20-4-perceptual-fixtures-r4"

sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import ttfa_spike_harness as H  # noqa: E402
import numpy as np  # noqa: E402

from myvoice.services.streaming_chunk_buffer import (  # noqa: E402
    StreamingChunkBuffer,
)
from myvoice.services.tts_streaming import codec_token_streamer  # noqa: E402
from myvoice.services.tts_streaming import streaming_decoder  # noqa: E402

SAMPLE_RATE = 24000
REFERENCE_CHUNK_SIZE = 25
CANDIDATE_CHUNK_SIZE = 15

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

CANDIDATE = "cs15fix"
REFERENCE = "cs25fix"


class WavCollector:
    """Production-faithful consumer: marshals to the loop, pushes through a
    real StreamingChunkBuffer, keeps what the buffer releases."""

    def __init__(self, loop) -> None:
        self._loop = loop
        self._buffer = StreamingChunkBuffer(
            watermark_ms=H.WATERMARK_MS,
            crossfade_samples=H.CROSSFADE_SAMPLES,
            sample_rate=SAMPLE_RATE, channels=1, sample_width=2,
        )
        self.released: list = []

    def reset(self) -> None:
        self._buffer.reset()
        self.released = []

    def on_chunk(self, chunk) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._handle(chunk), loop)

    async def _handle(self, chunk) -> None:
        if chunk.audio_data is None or chunk.audio_data.size == 0:
            return
        audio_bytes = (
            np.clip(chunk.audio_data, -1.0, 1.0) * 32767
        ).astype(np.int16).tobytes()
        for out in self._buffer.push(audio_bytes, is_final=chunk.is_final):
            self.released.append(out)

    def drain_tail(self) -> None:
        for out in self._buffer.flush_remaining():
            self.released.append(out)

    def write_wav(self, path: Path) -> int:
        payload = b"".join(self.released)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(SAMPLE_RATE)
            fh.writeframes(payload)
        return len(payload)


def _preflight() -> None:
    cs = codec_token_streamer.DEFAULT_CHUNK_SIZE
    la = codec_token_streamer.DEFAULT_LOOKAHEAD
    if (cs, la) != (25, 5):
        raise SystemExit(
            "FATAL: committed geometry is ({}, {}), expected (25, 5). This "
            "pass must not change it; both arms are set in-process.".format(
                cs, la)
        )
    for attr in ("_CODEC_SAMPLES_PER_FRAME", "_CODEC_EDGE_LOSS_SAMPLES",
                 "_OVERLAP_ADD_SAMPLES"):
        if not hasattr(streaming_decoder, attr):
            raise SystemExit(
                "FATAL: streaming_decoder is missing {} - the seam fix must "
                "be present in BOTH arms or this round repeats round 2's "
                "confound.".format(attr)
            )
    print("preflight OK")
    print("  committed geometry untouched at ({}, {})".format(cs, la))
    print("  seam fix present in both arms: samples/frame={} edge={} ola={}"
          .format(streaming_decoder._CODEC_SAMPLES_PER_FRAME,
                  streaming_decoder._CODEC_EDGE_LOSS_SAMPLES,
                  streaming_decoder._OVERLAP_ADD_SAMPLES))
    print("  reference cs={}  candidate cs={}  -> the ONLY variable is the "
          "geometry".format(REFERENCE_CHUNK_SIZE, CANDIDATE_CHUNK_SIZE))


async def _generate_arm(service, consumer, prompt, chunk_size, label):
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest

    H._apply_chunk_size(chunk_size)
    print("\n=== {} (chunk_size={}) ===".format(label, chunk_size))
    warm = QwenTTSRequest(
        text=H.PRIMING_TEXT, language="English",
        model_type=QwenModelType.BASE, streaming=True,
        voice_clone_prompt=prompt, suppress_audio_output=True,
    )
    await service._generate_true_stream(warm)
    await asyncio.sleep(0.2)

    posted = 25 if chunk_size == 25 else chunk_size
    for utt_id, text in UTTERANCES.items():
        consumer.reset()
        req = QwenTTSRequest(
            text=text, language="English",
            model_type=QwenModelType.BASE, streaming=True,
            voice_clone_prompt=prompt,
        )
        t0 = time.time()
        resp = await service._generate_true_stream(req)
        await asyncio.sleep(0.25)
        consumer.drain_tail()
        if not resp.success:
            raise SystemExit("FATAL: {} {} failed: {}".format(
                utt_id, label, resp.error_message))
        out = R4_DIR / "{}-{}.wav".format(utt_id, label)
        nbytes = consumer.write_wav(out)
        n_samples = nbytes // 2
        seams = max(0, (n_samples - 1) // (posted * 1920))
        print("  {:<6} {:<8} {:>7.0f}ms  {:>7.2f}s audio  {:>2d} seams  -> {}"
              .format(utt_id, label, (time.time() - t0) * 1000.0,
                      n_samples / SAMPLE_RATE, seams, out.name))


async def _run() -> int:
    from myvoice.services.qwen_tts_service import QwenTTSService

    _preflight()
    R4_DIR.mkdir(parents=True, exist_ok=True)

    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings,
    )
    if not await service.start():
        print("FATAL: QwenTTSService.start() returned False", file=sys.stderr)
        return 1

    loop = asyncio.get_running_loop()
    consumer = WavCollector(loop)
    service.set_audio_chunk_ready_callback(consumer.on_chunk)
    prompt = H._load_voice_clone_prompt(service)

    try:
        # Both arms in one process, one model load, one compiled state.
        await _generate_arm(service, consumer, prompt,
                            REFERENCE_CHUNK_SIZE, REFERENCE)
        await _generate_arm(service, consumer, prompt,
                            CANDIDATE_CHUNK_SIZE, CANDIDATE)
    finally:
        await service.stop()

    _write_truthtable()
    print("\nRound-4 fixture written to {}".format(R4_DIR))
    return 0


def _write_truthtable() -> None:
    rng = random.Random(20040904)
    table = {
        "_meta": {
            "round": 4,
            "candidate": CANDIDATE,
            "reference": REFERENCE,
            "candidate_desc": "chunk_size=15 WITH the seam fix",
            "reference_desc": "chunk_size=25 WITH the seam fix (the new "
                              "baseline - regenerated, not round 1's "
                              "pre-fix files)",
            "isolates": "the geometry only - both arms carry the seam fix",
        },
        "L1": {},
    }
    for utt_id in sorted(UTTERANCES):
        first, second = ((REFERENCE, CANDIDATE) if rng.random() < 0.5
                         else (CANDIDATE, REFERENCE))
        table["L1"][utt_id] = {
            "trial_A_filename": "{}-{}.wav".format(utt_id, first),
            "trial_A_geometry": first,
            "trial_B_filename": "{}-{}.wav".format(utt_id, second),
            "trial_B_geometry": second,
        }
    path = R4_DIR / "_perlistener_truthtable.json"
    path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print("Truth table -> {}".format(path.name))


def main() -> int:
    import torch
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print("device={} vram={:.1f}GiB torch={}".format(
            pr.name, pr.total_memory / 1024 ** 3, torch.__version__))
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
