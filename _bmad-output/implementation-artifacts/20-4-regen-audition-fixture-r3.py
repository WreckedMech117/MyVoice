"""Story 20.4 AC #5 - round-3 ISOLATING fixture: one variable, the stitching.

Round 2 changed TWO things against a reference that had neither - the
geometry (chunk_size 25 -> 10) AND the stitching (shipped trim -> exact
splice + 1024-sample overlap-add). Its clicks therefore cannot be
attributed. That matters well beyond the retune, because the seam fix is
geometry-independent and alters the SHIPPING cs25 path too.

Round 3 holds the geometry fixed and moves only the stitching:

    reference = cs25, shipped pre-fix stitching   (round 1 exact files)
    candidate = cs25, WITH the seam fix

Outcomes, agreed before the fixture was built so the result cannot be
rationalised afterwards:

  * candidate clean or better -> the fix is good, chunk_size=10 is the
    problem -> keep the fix, retreat the geometry (cs15 or cs25).
  * candidate shows clicks    -> the fix is harmful at ANY geometry ->
    revert it, and the 19.3 ms deletion needs a different remedy.

The reference arm is round 1 cs25 WAVs copied byte-for-byte: it costs no
GPU, and it is the third time those exact files have been auditioned, so
they keep working as a calibration anchor (rounds 1 and 2 agreed on 6 of 7).

Neither the committed DEFAULT_CHUNK_SIZE nor the seam fix is modified by
this script. The candidate geometry is set in-process via the harness
rebinder - the same mechanism Story 20.1 used for its sweep - so there is
no source-tree edit to revert.

Rounds 1 and 2 are left untouched.

Usage:
    python310/python.exe _bmad-output/implementation-artifacts/20-4-regen-audition-fixture-r3.py

Working file - gitignored under _bmad-output/; force-add per
memory/git_repo_state.md.
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import sys
import time
import wave
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
R1_DIR = SCRIPT_DIR / "20-4-perceptual-fixtures"
R3_DIR = SCRIPT_DIR / "20-4-perceptual-fixtures-r3"

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
CANDIDATE_CHUNK_SIZE = 25

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

CANDIDATE = "cs25fix"
REFERENCE = "cs25"


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
    """Confirm the isolation actually holds before spending GPU time."""
    cs = codec_token_streamer.DEFAULT_CHUNK_SIZE
    la = codec_token_streamer.DEFAULT_LOOKAHEAD
    if (cs, la) != (10, 5):
        raise SystemExit(
            "FATAL: committed geometry is ({}, {}). This pass must not change "
            "it; the candidate geometry is set in-process instead.".format(cs, la)
        )
    for attr in ("_CODEC_SAMPLES_PER_FRAME", "_CODEC_EDGE_LOSS_SAMPLES",
                 "_OVERLAP_ADD_SAMPLES"):
        if not hasattr(streaming_decoder, attr):
            raise SystemExit(
                "FATAL: streaming_decoder is missing {} - the seam fix is not "
                "present, so both arms would be identical and the audition "
                "would answer nothing.".format(attr)
            )
    missing = [u for u in UTTERANCES
               if not (R1_DIR / "{}-cs25.wav".format(u)).exists()]
    if missing:
        raise SystemExit(
            "FATAL: round-1 cs25 WAVs missing for {}.".format(missing)
        )
    print("preflight OK")
    print("  committed geometry untouched at ({}, {})".format(cs, la))
    print("  candidate generated in-process at chunk_size={}".format(
        CANDIDATE_CHUNK_SIZE))
    print("  seam fix present: samples/frame={} edge={} overlap-add={}".format(
        streaming_decoder._CODEC_SAMPLES_PER_FRAME,
        streaming_decoder._CODEC_EDGE_LOSS_SAMPLES,
        streaming_decoder._OVERLAP_ADD_SAMPLES))
    print("  both arms are chunk_size={} - the ONLY difference is the "
          "stitching".format(CANDIDATE_CHUNK_SIZE))


async def _run() -> int:
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    _preflight()
    R3_DIR.mkdir(parents=True, exist_ok=True)

    for utt in UTTERANCES:
        shutil.copyfile(R1_DIR / "{}-cs25.wav".format(utt),
                        R3_DIR / "{}-{}.wav".format(utt, REFERENCE))
    print("copied {} round-1 cs25 WAVs as the reference arm".format(
        len(UTTERANCES)))

    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings,
    )
    if not await service.start():
        print("FATAL: QwenTTSService.start() returned False", file=sys.stderr)
        return 1

    # In-process geometry override. Equivalent to the module-constant edit
    # the streamer docstring documents as the tuning path, but leaves no
    # source-tree diff - the same mechanism Story 20.1 used for its sweep.
    H._apply_chunk_size(CANDIDATE_CHUNK_SIZE)

    loop = asyncio.get_running_loop()
    consumer = WavCollector(loop)
    service.set_audio_chunk_ready_callback(consumer.on_chunk)
    prompt = H._load_voice_clone_prompt(service)

    try:
        warm = QwenTTSRequest(
            text=H.PRIMING_TEXT, language="English",
            model_type=QwenModelType.BASE, streaming=True,
            voice_clone_prompt=prompt, suppress_audio_output=True,
        )
        await service._generate_true_stream(warm)
        await asyncio.sleep(0.2)

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
                print("  {}: FAILED - {}".format(utt_id, resp.error_message),
                      file=sys.stderr)
                return 2
            out = R3_DIR / "{}-{}.wav".format(utt_id, CANDIDATE)
            nbytes = consumer.write_wav(out)
            print("  {:<6} {:<8} {:>7.0f}ms  {:>8.2f}s audio  -> {}".format(
                utt_id, CANDIDATE, (time.time() - t0) * 1000.0,
                nbytes / (SAMPLE_RATE * 2), out.name))
    finally:
        await service.stop()

    _write_truthtable()
    print("\nRound-3 fixture written to {}".format(R3_DIR))
    return 0


def _write_truthtable() -> None:
    rng = random.Random(20040903)
    table = {
        "_meta": {
            "round": 3,
            "candidate": CANDIDATE,
            "reference": REFERENCE,
            "candidate_desc": "chunk_size=25 WITH the Story 20.4 seam fix",
            "reference_desc": "chunk_size=25 shipped pre-fix stitching "
                              "(round 1 exact files)",
            "isolates": "the stitching only - both arms are chunk_size=25",
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
    path = R3_DIR / "_perlistener_truthtable.json"
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
