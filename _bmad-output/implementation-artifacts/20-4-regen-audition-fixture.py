"""Story 20.4 AC #5 - generate the NFR3 perceptual A/B fixture for the
chunk-size retune.

Why this fixture cannot reuse Story 18.4's
------------------------------------------
Story 18.4's ``18-4-regen-fixture.py`` calls ``generate_voice_clone`` - the
BATCH API. It never touches the streamer, the streaming decoder's
overlap-add, or the consumer-side crossfade. Story 20.4 changes the
streamer's chunk boundaries, and the decoder trims a lookahead-sized tail
per chunk, so the thing under audition is precisely the part the batch API
skips. This generator therefore drives the **production TRUE_STREAM dispatch
path** (``QwenTTSService._generate_true_stream``) and pushes every PCM chunk
through a real ``StreamingChunkBuffer`` configured with the shipped
consumer constants, so the WAV on disk is what a user's speakers receive.

  * A rendition: ``{utt}-cs25.wav``  - chunk_size 25 (the PRE-20.4 geometry)
  * B rendition: ``{utt}-cs10.wav``  - chunk_size 10 (the COMMITTED geometry)

Both use the canonical Sarira-F CLONED voice (AC #5 requires a cloned
voice), the same precomputed Story 17.2 prompt every prior audition used,
and identical settings otherwise. The ONLY difference between A and B is
the streamer geometry.

Both utterance classes are covered (AC #5): the short class changes dispatch
path entirely at chunk_size 10 - Story 20.1 SS5.3 measured it moving off
``residual_flush`` onto ``threshold`` in 5/5 runs - so a short-only or
long-only fixture would miss half the change.

Sampling caveat, stated up front: qwen-tts sampling is stochastic, so A and
B are NOT the same waveform with different seams. They are two renditions.
The audition question is therefore "does B carry chunk-boundary artefacts
that A does not" - clicks, discontinuities, altered prosody at stitch
points - not "are these bit-identical". That is the same standard Story
18.4's audition used.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-4-regen-audition-fixture.py

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
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
FIXTURE_DIR = SCRIPT_DIR / "20-4-perceptual-fixtures"

# The harness owns the torch-before-everything import, the settings builder,
# the voice_clone_prompt loader and the chunk-size rebinder. Importing it
# rather than re-implementing keeps this fixture on exactly the plumbing
# Story 20.1's measurements used.
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import ttfa_spike_harness as H  # noqa: E402  (torch imports inside)

import numpy as np  # noqa: E402

from myvoice.services.streaming_chunk_buffer import (  # noqa: E402
    StreamingChunkBuffer,
)

SAMPLE_RATE = 24000

# Short class = the Clear Comms interjection shape. Long class = the Epic 18
# canonical paragraph. The middle rows are sibilant/plosive-dense on purpose:
# fricatives and stops are where a chunk seam is most audible, and Story
# 18.4's fixture used the same trick.
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

GEOMETRIES = {"cs25": 25, "cs10": 10}


class WavCollector:
    """Production-faithful consumer: marshals to the loop, pushes through a
    real StreamingChunkBuffer, and keeps every byte the buffer RELEASES.

    Mirrors ``ttfa_spike_harness.ConsumerSim`` (which mirrors
    ``MyVoiceApp._handle_progressive_chunk_async``) and additionally
    accumulates output. What lands on disk is post-watermark and
    post-crossfade - i.e. what PyAudio would have been handed.
    """

    def __init__(self, loop) -> None:
        self._loop = loop
        self._buffer = StreamingChunkBuffer(
            watermark_ms=H.WATERMARK_MS,
            crossfade_samples=H.CROSSFADE_SAMPLES,
            sample_rate=SAMPLE_RATE,
            channels=1,
            sample_width=2,
        )
        self.released: list = []
        self._pending = 0

    def reset(self) -> None:
        self._buffer.reset()
        self.released = []

    def on_chunk(self, chunk) -> None:
        """Sync trampoline - runs on the StreamingDecoderWorker thread."""
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
        """Mirror ``stop_streaming_session``: flush anything still held."""
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


async def _run() -> int:
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None,
        device="auto",
        quality_tier="quality",
        session_registry=None,
        app_settings=settings,
    )
    if not await service.start():
        print("FATAL: QwenTTSService.start() returned False", file=sys.stderr)
        return 1

    loop = asyncio.get_running_loop()
    consumer = WavCollector(loop)
    service.set_audio_chunk_ready_callback(consumer.on_chunk)
    prompt = H._load_voice_clone_prompt(service)

    try:
        for geom_label, chunk_size in GEOMETRIES.items():
            H._apply_chunk_size(chunk_size)
            print("\n=== geometry {} (chunk_size={}) ===".format(
                geom_label, chunk_size))
            # One warm-up generation per geometry: the decode window is a
            # compile-cache key dimension, so switching geometry can trigger
            # a cold compile. Never let that land inside a fixture take.
            warm = QwenTTSRequest(
                text=H.PRIMING_TEXT, language="English",
                model_type=QwenModelType.BASE, streaming=True,
                voice_clone_prompt=prompt, suppress_audio_output=True,
            )
            t0 = time.time()
            await service._generate_true_stream(warm)
            await asyncio.sleep(0.2)
            print("  warm-up generation: {:.0f}ms".format(
                (time.time() - t0) * 1000.0))

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
                    print("  {} {}: FAILED - {}".format(
                        utt_id, geom_label, resp.error_message),
                        file=sys.stderr)
                    return 2
                out = FIXTURE_DIR / "{}-{}.wav".format(utt_id, geom_label)
                nbytes = consumer.write_wav(out)
                manifest.setdefault(utt_id, {})[geom_label] = out.name
                print("  {:<6} {:<5} {:>7.0f}ms  {:>8.2f}s audio  -> {}".format(
                    utt_id, geom_label, (time.time() - t0) * 1000.0,
                    nbytes / (SAMPLE_RATE * 2), out.name))
    finally:
        await service.stop()

    _write_truthtable(manifest)
    print("\nFixture written to {}".format(FIXTURE_DIR))
    return 0


def _write_truthtable(manifest) -> None:
    """Blind the audition: randomise which geometry is trial A per utterance.

    Deterministic seed so the mapping is reproducible from this file alone
    (Story 17.1's M1 reproducibility requirement), but not guessable from
    listening order.
    """
    rng = random.Random(20040901)
    table = {"L1": {}}
    for utt_id in sorted(manifest):
        pair = manifest[utt_id]
        if len(pair) != 2:
            continue
        first, second = ("cs25", "cs10") if rng.random() < 0.5 else ("cs10", "cs25")
        table["L1"][utt_id] = {
            "trial_A_filename": pair[first],
            "trial_A_geometry": first,
            "trial_B_filename": pair[second],
            "trial_B_geometry": second,
        }
    path = FIXTURE_DIR / "_perlistener_truthtable.json"
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
