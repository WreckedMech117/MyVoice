"""Story 20.4 AC #5 follow-up - capture pcm_full (PRE-trim) per decoded chunk.

``20-4-seam-capture.py`` established the decoder's output-length model from
posted lengths alone:

    decode(N frames) -> 1920*N - 555 samples

i.e. the codec runs at 12.5 Hz (not the 12 Hz assumed throughout the
codebase) and loses a FIXED 555 samples per decode call to convolution
edge effects. ``StreamingDecoderWorker._decode_and_post`` computes its trim
as ``round(lookahead * len(pcm_full)/len(chunk))``, which treats that fixed
loss as if it were proportional - so every posted chunk is short by
``555 * chunk_size/(chunk_size+lookahead)`` samples.

The open question that decides the fix: is that deficit DROPPED SPEECH
(a splice error at every seam) or edge padding the codec never produced?

Consecutive chunks overlap by ``lookahead`` frames of tokens, so their
pcm_full arrays contain the same audio twice. Cross-correlating them gives
the true alignment, and from that whether the posted concatenation drops
audio, duplicates it, or lines up. This script captures pcm_full by
wrapping the decode_fn the worker is handed.

Usage:
    python310\python.exe _bmad-output\implementation-artifacts\20-4-seam-capture-full.py

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
OUT_DIR = SCRIPT_DIR / "20-4-seam-rawfull"

sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import ttfa_spike_harness as H  # noqa: E402
import numpy as np  # noqa: E402

# A deliberately small set: the blocking utterance (m-020) plus one long
# form (l-020, defective on BOTH geometries in round 1) plus one clean
# control (m-021). Enough to answer the alignment question at both
# geometries without another full-fixture GPU pass.
UTTERANCES = {
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
}
GEOMETRIES = {"cs25": 25, "cs10": 10}


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
        print("FATAL: start() returned False", file=sys.stderr)
        return 1

    # Wrap the decode-fn builder so every pcm_full and its token count are
    # recorded before the worker trims. The worker is untouched: this
    # observes the exact array it is about to trim.
    full_records: list = []
    real_builder = service._build_true_stream_decode_fn

    def wrapped_builder(model):
        inner = real_builder(model)

        def _decode(chunk):
            pcm = inner(chunk)
            try:
                n_frames = int(chunk.shape[0])
            except Exception:
                n_frames = int(len(chunk))
            full_records.append((n_frames, np.asarray(pcm, np.float32).copy()))
            return pcm
        return _decode

    service._build_true_stream_decode_fn = wrapped_builder

    posted: list = []
    def on_chunk(chunk):
        d = getattr(chunk, "audio_data", None)
        if d is not None and d.size:
            posted.append(np.asarray(d, np.float32).copy())
    service.set_audio_chunk_ready_callback(on_chunk)
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
                full_records.clear(); posted.clear()
                req = QwenTTSRequest(
                    text=text, language="English",
                    model_type=QwenModelType.BASE, streaming=True,
                    voice_clone_prompt=prompt,
                )
                t0 = time.time()
                resp = await service._generate_true_stream(req)
                await asyncio.sleep(0.3)
                if not resp.success:
                    print("  FAILED", utt_id, geom_label, file=sys.stderr)
                    return 2
                frames = np.array([r[0] for r in full_records], dtype=np.int64)
                fulls = [r[1] for r in full_records]
                np.savez_compressed(
                    OUT_DIR / "{}-{}.npz".format(utt_id, geom_label),
                    full_concat=np.concatenate(fulls),
                    full_lengths=np.array([f.size for f in fulls], np.int64),
                    frames=frames,
                    posted_concat=np.concatenate(posted),
                    posted_lengths=np.array([p.size for p in posted], np.int64),
                    chunk_size=chunk_size, lookahead=5,
                )
                print("  {:<6} {:<5} {:>6.0f}ms  frames={}  full_lens={}".format(
                    utt_id, geom_label, (time.time() - t0) * 1000.0,
                    list(frames[:6]), [f.size for f in fulls[:6]]))
    finally:
        await service.stop()
    print("\npcm_full -> {}".format(OUT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
