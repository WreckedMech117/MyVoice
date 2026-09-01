"""Story 20.4 AC #5 - round-2 audition fixture, after the seam fix.

Round 1 FAILED: m-020 was clean at chunk_size=25 and carried
``tonal_distortion`` at chunk_size=10, and l-020/l-021 carried seam defects
on BOTH arms. The follow-up analysis traced that to two defects at every
decoder chunk boundary - a splice-alignment bug deleting 15-19 ms of real
speech per seam, and a codec-state mismatch between the two independent
decodes of the shared lookahead frames. Both are now fixed in
``streaming_decoder.py``.

Round-2 design, and why it is shaped this way
---------------------------------------------
* **The A arm is round 1's cs25 WAVs, copied verbatim.** Not regenerated.
  Commander has already judged those exact files, so his round-1 calls
  (l-020 ``click_or_discontinuity``, l-021 ``tonal_distortion``, everything
  else ``none``) carry over as a calibration anchor: if the same files
  draw the same calls again, the session is internally consistent; if they
  do not, that is worth knowing before reading anything into the B arm.
  It also means round 1's evidence is not disturbed - that round is a
  recorded result.

* **The A arm is deliberately the PRE-FIX stitching.** The question AC #5
  asks is whether what we propose to ship is free of seam artefacts, with
  what ships today as the reference. What ships today is chunk_size=25
  with the old splice. Fixing the reference arm would answer a different,
  more academic question.

* **The B arm is chunk_size=10 with the fix**, i.e. the candidate build.

* Same seven utterances, same voice, same machinery, new randomisation.

Everything is written to ``20-4-perceptual-fixtures-r2/``. Round 1's
directory, truth table and results CSV are left untouched.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-4-regen-audition-fixture-r2.py

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
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
R2_DIR = SCRIPT_DIR / "20-4-perceptual-fixtures-r2"

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

CANDIDATE = "cs10fix"
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
    """Refuse to build a fixture that does not contain the fix."""
    cs = codec_token_streamer.DEFAULT_CHUNK_SIZE
    la = codec_token_streamer.DEFAULT_LOOKAHEAD
    if (cs, la) != (10, 5):
        raise SystemExit(
            "FATAL: committed geometry is ({}, {}), expected (10, 5). The B "
            "arm would not be the candidate build.".format(cs, la)
        )
    for attr in ("_CODEC_SAMPLES_PER_FRAME", "_CODEC_EDGE_LOSS_SAMPLES",
                 "_OVERLAP_ADD_SAMPLES"):
        if not hasattr(streaming_decoder, attr):
            raise SystemExit(
                "FATAL: streaming_decoder is missing {} - the seam fix is "
                "not present, so this fixture would just reproduce round "
                "1.".format(attr)
            )
    missing = [u for u in UTTERANCES
               if not (R1_DIR / "{}-cs25.wav".format(u)).exists()]
    if missing:
        raise SystemExit(
            "FATAL: round-1 cs25 WAVs missing for {}. The A arm is supposed "
            "to be round 1's files verbatim.".format(missing)
        )
    print("preflight OK: geometry (10, 5); seam fix present "
          "(samples/frame={}, edge={}, overlap-add={}); round-1 A arm "
          "available".format(
              streaming_decoder._CODEC_SAMPLES_PER_FRAME,
              streaming_decoder._CODEC_EDGE_LOSS_SAMPLES,
              streaming_decoder._OVERLAP_ADD_SAMPLES))


async def _run() -> int:
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    _preflight()
    R2_DIR.mkdir(parents=True, exist_ok=True)

    # A arm: round 1's files, copied byte-for-byte.
    for utt in UTTERANCES:
        src = R1_DIR / "{}-cs25.wav".format(utt)
        shutil.copyfile(src, R2_DIR / "{}-{}.wav".format(utt, REFERENCE))
    print("copied {} round-1 cs25 WAVs as the A arm".format(len(UTTERANCES)))

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
            out = R2_DIR / "{}-{}.wav".format(utt_id, CANDIDATE)
            nbytes = consumer.write_wav(out)
            print("  {:<6} {:<8} {:>7.0f}ms  {:>8.2f}s audio  -> {}".format(
                utt_id, CANDIDATE, (time.time() - t0) * 1000.0,
                nbytes / (SAMPLE_RATE * 2), out.name))
    finally:
        await service.stop()

    _write_truthtable()
    print("\nRound-2 fixture written to {}".format(R2_DIR))
    return 0


def _write_truthtable() -> None:
    """Blind the audition; record which side is the candidate.

    A different seed from round 1 on purpose: reusing it would put the same
    geometry on the same side for every utterance, and a listener who has
    just done round 1 could carry an ordering expectation into round 2.
    """
    rng = random.Random(20040902)
    table = {
        "_meta": {
            "round": 2,
            "candidate": CANDIDATE,
            "reference": REFERENCE,
            "candidate_desc": "chunk_size=10 WITH the Story 20.4 seam fix",
            "reference_desc": "chunk_size=25 with the shipped pre-fix "
                              "stitching (round 1's exact files)",
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
    path = R2_DIR / "_perlistener_truthtable.json"
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
