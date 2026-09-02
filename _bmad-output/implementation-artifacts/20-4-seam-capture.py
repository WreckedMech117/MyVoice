"""Story 20.4 AC #5 follow-up - capture RAW decoder chunks, pre-consumer.

The round-1 audition failed with ``tonal_distortion`` on m-020 at
chunk_size=10 only. Two candidate mechanisms:

  (a) the CONSUMER crossfade (StreamingChunkBuffer, 64 samples ~ 2.7 ms) is
      too narrow now that seams are 2.5x more frequent, or
  (b) the DECODER overlap-add posts independently-decoded PCM segments, so
      the codec's internal state differs across every boundary.

The fixture WAVs cannot separate these: they are already crossfaded, so the
64-sample blend has smeared exactly the evidence that distinguishes an
amplitude step from a spectral mismatch. This script captures the float32
PCM arrays EXACTLY as ``StreamingDecoderWorker`` posts them - before the
int16 cast, before the watermark merge, before any crossfade - so the seam
can be measured raw, and so a crossfade sweep can be simulated offline
against a FIXED take rather than re-sampling new speech per width.

One .npz per (utterance, geometry) holding the ordered chunk arrays plus
their lengths, so seam offsets are exact rather than inferred from geometry.

Usage:
    python310\python.exe _bmad-output\implementation-artifacts\20-4-seam-capture.py

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = SCRIPT_DIR / "20-4-seam-raw"

sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import ttfa_spike_harness as H  # noqa: E402  (torch imports inside)
import numpy as np  # noqa: E402

# Same utterance set as the round-1 audition fixture, so the analysis speaks
# to the same material the listener judged.
sys.path.insert(0, str(SCRIPT_DIR))
from importlib import import_module  # noqa: E402

_fixture = import_module("20-4-regen-audition-fixture".replace("-", "_")) \
    if False else None  # (module name is not importable; texts inlined below)

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


class RawChunkCollector:
    """Records every posted chunk verbatim. No buffer, no crossfade, no cast.

    Runs on the StreamingDecoderWorker thread (the same trampoline
    production uses), but does nothing except append - the point is to
    observe the decoder's output, not to reproduce the consumer.
    """

    def __init__(self) -> None:
        self.chunks: list = []
        self.is_final: list = []

    def reset(self) -> None:
        self.chunks = []
        self.is_final = []

    def on_chunk(self, chunk) -> None:
        data = getattr(chunk, "audio_data", None)
        if data is None or data.size == 0:
            return
        self.chunks.append(np.asarray(data, dtype=np.float32).copy())
        self.is_final.append(bool(chunk.is_final))


async def _run() -> int:
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings,
    )
    if not await service.start():
        print("FATAL: QwenTTSService.start() returned False", file=sys.stderr)
        return 1

    collector = RawChunkCollector()
    service.set_audio_chunk_ready_callback(collector.on_chunk)
    prompt = H._load_voice_clone_prompt(service)

    try:
        for geom_label, chunk_size in GEOMETRIES.items():
            H._apply_chunk_size(chunk_size)
            print("\n=== {} (chunk_size={}) ===".format(geom_label, chunk_size))
            warm = QwenTTSRequest(
                text=H.PRIMING_TEXT, language="English",
                model_type=QwenModelType.BASE, streaming=True,
                voice_clone_prompt=prompt, suppress_audio_output=True,
            )
            await service._generate_true_stream(warm)
            await asyncio.sleep(0.2)

            for utt_id, text in UTTERANCES.items():
                collector.reset()
                req = QwenTTSRequest(
                    text=text, language="English",
                    model_type=QwenModelType.BASE, streaming=True,
                    voice_clone_prompt=prompt,
                )
                t0 = time.time()
                resp = await service._generate_true_stream(req)
                await asyncio.sleep(0.3)
                if not resp.success:
                    print("  {} {}: FAILED".format(utt_id, geom_label),
                          file=sys.stderr)
                    return 2
                lens = [c.size for c in collector.chunks]
                out = OUT_DIR / "{}-{}.npz".format(utt_id, geom_label)
                np.savez_compressed(
                    out,
                    pcm=np.concatenate(collector.chunks),
                    lengths=np.array(lens, dtype=np.int64),
                    is_final=np.array(collector.is_final, dtype=bool),
                    chunk_size=chunk_size,
                    lookahead=5,
                )
                print("  {:<6} {:<5} {:>6.0f}ms  {:>2d} chunks  lens={}"
                      .format(utt_id, geom_label, (time.time() - t0) * 1000.0,
                              len(lens), lens[:6] + (["..."] if len(lens) > 6 else [])))
    finally:
        await service.stop()
    print("\nRaw chunks -> {}".format(OUT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
