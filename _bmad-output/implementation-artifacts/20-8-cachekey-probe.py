"""Story 20.8 Phase 1 — which compile-cache key does each sweep point use?

AC #1 requires the cold-compile cost per point to be STATED. The sitting
observed almost none, which is either a finding or a broken assumption, and
the difference is decided by whether the points really do land on distinct
``compile_cache`` keys.

This loads the model ONCE and then computes ``compile_cache.compute_key`` for
each candidate ``decode_window_frames`` from exactly the inputs
``QwenTTSService.warmup_compile_async`` uses (``qwen_tts_service.py:2300-2321``),
printing the 16-char directory prefix each would use, and whether that
directory exists on this host.

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import ttfa_spike_harness as H  # noqa: E402


async def _run() -> int:
    from myvoice.models.service_enums import QwenModelType
    from myvoice.services.qwen_tts_service import QwenTTSRequest, QwenTTSService
    from myvoice.services.tts_streaming import compile_cache

    cs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if cs is not None:
        H._apply_chunk_size(cs)
    from myvoice.services.tts_streaming import resolve_streamer_geometry
    print("  resolve_streamer_geometry() = {}".format(
        resolve_streamer_geometry()))
    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings,
    )
    if not await service.start():
        print("FATAL: service.start() returned False", file=sys.stderr)
        return 1
    try:
        # engage_compile_optimizations only reaches
        # set_torchinductor_cache_dir on the first real dispatch, so one
        # suppressed priming generation is required before the env var
        # names the per-key directory this geometry actually reads.
        prompt = H._load_voice_clone_prompt(service)
        await service._generate_true_stream(QwenTTSRequest(
            text=H.PRIMING_TEXT, language="English",
            model_type=QwenModelType.BASE, streaming=True,
            voice_clone_prompt=prompt, suppress_audio_output=True))
        await asyncio.sleep(0.2)
        model = service._model_registry.get_loaded_model()
        model_id = service._compile_cache_model_id(model)
        dtype = getattr(getattr(model, "model", None), "dtype", None)
        precision_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        cap = torch.cuda.get_device_capability()
        print("  model_id       = {!r}".format(model_id))
        print("  precision_str  = {}".format(precision_str))
        print("  torch          = {}".format(torch.__version__))
        print("  cuda_capability= {}".format(cap))
        print("  pin hash       = {}".format(service._QWEN_TTS_PIN_HASH))
        print("  TORCHINDUCTOR_CACHE_DIR now = {}".format(
            os.environ.get("TORCHINDUCTOR_CACHE_DIR")))
        root = compile_cache.cache_root()
        print("\n  {:<26} {:<20} {:<8} {}".format(
            "decode_window_frames", "key dir (16 hex)", "exists", "entries"))
        for dwf in (30, 25, 15, 10, 7):
            key = compile_cache.compute_key(
                qwen_tts_pin_hash=service._QWEN_TTS_PIN_HASH,
                model_id=model_id,
                precision_str=precision_str,
                torch_version=torch.__version__,
                decode_window_frames=dwf,
                cuda_capability=cap,
                compile_mode="reduce-overhead",
            )
            d = root / key[:16]
            n = len(list(d.iterdir())) if d.is_dir() else 0
            print("  {:<26} {:<20} {:<8} {}".format(
                dwf, key[:16], str(d.is_dir()), n))
    finally:
        await service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
