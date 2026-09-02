"""Story 20.8 — the FOURTH claim: cross-process / cross-compile-key invariance.

``20-8-onetalker-viability.py`` establishes (b) token invariance **within one
process, one model load, one compiled state**. That is the right isolation for
the audition itself — a fixture generator captures one talker run and re-chunks
it in that same process, so nothing else can vary.

It is NOT the right isolation for the claim the audition is ultimately used to
support: *"the geometry we ship will sound like the candidate arm."* A shipped
`cs7` build loads with ``decode_window_frames = 7``, which is one of
``compile_cache``'s seven key dimensions, so it reads a DIFFERENT inductor
cache directory and may run different compiled kernels — which could draw
different tokens for the same seed, independently of chunk size.

This script captures one live run's flat token stream in a FRESH process at a
given chunk size and seed, and writes it out. Comparing two such files across
two processes tests exactly that: same seed, different compile-cache key,
different process. It cannot be done in-process by construction.

Usage:
    python310\\python.exe 20-8-crossprocess-tokens.py <chunk_size> <seed> <out.pt>

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttfa_spike_harness as H  # noqa: E402


async def _run(chunk_size: int, seed: int, out_path: Path) -> int:
    import os
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService

    H._apply_chunk_size(chunk_size)
    from myvoice.services.tts_streaming import resolve_streamer_geometry
    geom = resolve_streamer_geometry()
    print("  resolve_streamer_geometry() = {}".format(geom), flush=True)

    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings)
    if not await service.start():
        print("FATAL: service.start() returned False", file=sys.stderr)
        return 1

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

    try:
        prompt = H._load_voice_clone_prompt(service)
        # settle the compiled state first; this is also what pays a cold key
        await service._generate_true_stream(QwenTTSRequest(
            text=H.PRIMING_TEXT, language="English",
            model_type=QwenModelType.BASE, streaming=True,
            voice_clone_prompt=prompt, suppress_audio_output=True))
        await asyncio.sleep(0.3)
        print("  TORCHINDUCTOR_CACHE_DIR = {}".format(
            os.environ.get("TORCHINDUCTOR_CACHE_DIR")), flush=True)

        service._build_true_stream_decode_fn = tapping_builder
        captured.clear()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        resp = await service._generate_true_stream(QwenTTSRequest(
            text=H.UTTERANCE_LONG, language="English",
            model_type=QwenModelType.BASE, streaming=True,
            voice_clone_prompt=prompt))
        await asyncio.sleep(0.3)
        service._build_true_stream_decode_fn = real_builder
        if not resp.success or not captured:
            print("FATAL: generation failed", file=sys.stderr)
            return 2
        flat = torch.cat(captured, dim=0)
        torch.save({"flat": flat, "chunk_size": chunk_size, "seed": seed,
                    "geometry": geom, "n_chunks": len(captured),
                    "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR")},
                   str(out_path))
        print("  captured {} frames in {} chunks -> {}".format(
            flat.shape[0], len(captured), out_path.name), flush=True)
    finally:
        await service.stop()
    return 0


if __name__ == "__main__":
    cs = int(sys.argv[1])
    sd = int(sys.argv[2])
    op = Path(sys.argv[3])
    raise SystemExit(asyncio.run(_run(cs, sd, op)))
