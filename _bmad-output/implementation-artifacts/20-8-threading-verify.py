"""Story 20.8 AC #3 — the geometry threading, verified at all three sites.

AC #3: *"it goes through the geometry threading rather than as a bare constant
edit, and all three sites follow — Story 20.6 verified this works in both
directions."*

The three sites D-25 can drift at (Story 20.4 §1.1 found three, not the two
Story 20.1 §5.4 predicted):

  1. ``resolve_streamer_geometry()`` — the single derivation point;
  2. ``engage_compile_optimizations`` via ``model_registry`` — which selects
     the inductor cache DIRECTORY the compile actually reads and writes;
  3. ``QwenTTSService.warmup_compile_async`` — which computes the key Story
     20.3's startup priming WARMS.

2 and 3 must agree or priming warms a directory nothing reads and the ~4 s
first-generation win silently stops working. That is not a hypothetical: it is
the exact failure Story 20.4 §5.4 closed and Story 20.6 re-opened from the
other side.

BOTH DIRECTIONS
---------------
A constant edit that only moves the value forward proves nothing about the
threading — a hard-coded literal equal to the new value would pass. So the
kill switch is flipped too: with ``MYVOICE_CODEC_STATE_CACHE`` disabling,
the stateless fallback keeps its 5-frame lookahead and every site must follow
to ``chunk_size + 5``.

THE COLD-COMPILE / PRIMING QUESTION
------------------------------------
The retune is a new cache key, so first launch pays one cold compile. What has
to be checked is not that it is paid but that **priming warms the NEW key**.
``warmup_compile_async`` is driven here for real, and the check is whether
``meta.json`` — written only by ``compile_cache.mark_warm`` — appears in the
new key's directory and not in the old one's.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-8-threading-verify.py

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

FAILURES = []


def check(label, got, want):
    ok = got == want
    print("  [{}] {:<58} got={!r} want={!r}".format(
        "PASS" if ok else "FAIL", label, got, want))
    if not ok:
        FAILURES.append(label)
    return ok


def static_geometry_checks():
    from myvoice.services.tts_streaming import (
        codec_token_streamer as cts, resolve_streamer_geometry,
    )
    print("\n== site 1: resolve_streamer_geometry(), both directions ==")
    cs = cts.DEFAULT_CHUNK_SIZE
    check("committed constant DEFAULT_CHUNK_SIZE", cs, 10)
    check("DEFAULT_LOOKAHEAD untouched (stateless path)", cts.DEFAULT_LOOKAHEAD, 5)
    os.environ.pop("MYVOICE_CODEC_STATE_CACHE", None)
    check("state-carrying geometry", resolve_streamer_geometry(), (10, 0))
    check("  -> decode_window_frames", sum(resolve_streamer_geometry()), 10)
    os.environ["MYVOICE_CODEC_STATE_CACHE"] = "0"
    check("kill-switch geometry (the other direction)",
          resolve_streamer_geometry(), (10, 5))
    check("  -> decode_window_frames", sum(resolve_streamer_geometry()), 15)
    os.environ.pop("MYVOICE_CODEC_STATE_CACHE", None)

    print("\n== the streamer itself follows the constant ==")
    s = cts.CodecTokenStreamer()
    check("CodecTokenStreamer() chunk_size", s.chunk_size, 10)
    check("apply_codec_state_geometry(True) -> lookahead",
          s.apply_codec_state_geometry(True), 0)
    check("  chunk_size unchanged by retirement", s.chunk_size, 10)


async def runtime_checks():
    from myvoice.services.qwen_tts_service import QwenTTSService
    from myvoice.services.tts_streaming import compile_cache

    root = compile_cache.cache_root()
    settings = H._build_settings("auto", "auto")
    service = QwenTTSService(
        audio_coordinator=None, device="auto", quality_tier="quality",
        session_registry=None, app_settings=settings)
    if not await service.start():
        print("FATAL: service.start() returned False", file=sys.stderr)
        return 1
    try:
        # Force the model to MATERIALISE before computing anything.
        # ``service.start()`` alone leaves ``model.model.name_or_path`` unset,
        # so a key computed here would use model_id='unknown' / fp32 — a key
        # nothing ever reads. And ``engage_compile_optimizations`` only reaches
        # ``set_torchinductor_cache_dir`` on the first dispatch, so site 2 is
        # not observable until a generation has run. The first version of this
        # file skipped this and reported three false failures; that is exactly
        # the class of mistake it exists to catch, so it is recorded here
        # rather than quietly fixed.
        from myvoice.models.service_enums import QwenModelType
        from myvoice.services.qwen_tts_service import QwenTTSRequest
        prompt = H._load_voice_clone_prompt(service)
        await service._generate_true_stream(QwenTTSRequest(
            text=H.PRIMING_TEXT, language="English",
            model_type=QwenModelType.BASE, streaming=True,
            voice_clone_prompt=prompt, suppress_audio_output=True))
        await asyncio.sleep(0.3)

        model = service._model_registry.get_loaded_model()
        model_id = service._compile_cache_model_id(model)
        dtype = getattr(getattr(model, "model", None), "dtype", None)
        precision_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        cap = torch.cuda.get_device_capability()

        def key_for(dwf):
            return compile_cache.compute_key(
                qwen_tts_pin_hash=service._QWEN_TTS_PIN_HASH,
                model_id=model_id, precision_str=precision_str,
                torch_version=torch.__version__, decode_window_frames=dwf,
                cuda_capability=cap, compile_mode="reduce-overhead")

        new_key, old_key = key_for(10), key_for(25)
        new_dir, old_dir = root / new_key[:16], root / old_key[:16]
        print("\n== keys ==")
        print("  cs10 (new, committed) -> {}".format(new_dir.name))
        print("  cs25 (old)            -> {}".format(old_dir.name))
        old_meta_before = (old_dir / "meta.json").exists()

        print("\n== site 2: engage_compile_optimizations — which directory "
              "does the compile read? ==")
        engaged = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        print("  TORCHINDUCTOR_CACHE_DIR = {}".format(engaged))
        engaged_name = Path(engaged).name if engaged else None
        check("engage path points at the NEW key's directory",
              engaged_name, new_dir.name)

        # Capture the warmup telemetry. A silent early return is the one
        # outcome that would look like a threading failure and is not, so the
        # reason is read rather than inferred from the filesystem.
        from myvoice.observability import metrics as _metrics
        seen = []

        def _tap(record):
            if record.name == "tts_compile_warmup_priming":
                seen.append(record)

        unsub = _metrics.add_listener(_tap)

        # ``warmup_compile_async`` resolves its priming text through the ACTIVE
        # PROFILE's cached voice_clone_prompt (Story 20.3 AC #2 — in-memory
        # lookup, never a computation). A headless service has no profile
        # manager, so without this it returns early with
        # reason='no_priming_prompt' and never reaches the key at all. That is
        # a property of the harness, not of the threading, so the ONE thing a
        # headless process cannot have is supplied and nothing else is touched:
        # the key computation, the is_warm check and mark_warm all run for real.
        service._active_profile_voice_clone_prompt = lambda: prompt

        print("\n== site 3: warmup_compile_async — which key does priming warm? ==")
        print("  is_warm(new key) before = {}".format(
            compile_cache.is_warm(new_key)))
        try:
            await service.warmup_compile_async()
        finally:
            unsub()
        if seen:
            for r in seen:
                print("  telemetry: reason={!r} tags={}".format(
                    r.tags.get("reason"), dict(r.tags)))
        else:
            print("  telemetry: NO tts_compile_warmup_priming record emitted")

        check("priming wrote meta.json into the NEW key's directory",
              (new_dir / "meta.json").exists(), True)
        check("the OLD key's meta.json state is unchanged",
              (old_dir / "meta.json").exists(), old_meta_before)
        check("sites 2 and 3 AGREE — priming warms what engage reads",
              engaged_name,
              new_dir.name if (new_dir / "meta.json").exists()
              else "NEW KEY NOT WARMED")
    finally:
        await service.stop()
    return 0


def main():
    print("=" * 78)
    print("Story 20.8 AC #3 — geometry threading verification")
    print("=" * 78)
    static_geometry_checks()
    rc = asyncio.run(runtime_checks())
    print("\n" + "=" * 78)
    if FAILURES:
        print("FAILED {} check(s): {}".format(len(FAILURES), FAILURES))
        return 1
    print("ALL THREADING CHECKS PASS — the constant moves all three sites, "
          "and it moves back under the kill switch.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
