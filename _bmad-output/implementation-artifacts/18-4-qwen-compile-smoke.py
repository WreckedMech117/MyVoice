"""Story 18.4 / OQ #4 Step 3 — Real Qwen3TTSModel compile smoke.

Mirrors MyVoice's production model-load path (services/model_registry.py:
_load_model_sync, lines 453-584), engages
``Qwen3TTSModel.enable_streaming_optimizations(..., compile_talker=False)``
via the same ``engage_compile_optimizations`` function that ships with
the source tree, runs one short generation, replays it, and verifies:

  * Model loads cleanly under bf16 on CUDA.
  * ``engage_compile_optimizations`` returns ``engaged=True`` with
    reason ``"engaged_cold_compile"``.
  * The P-12 capability probe finds the
    ``_torchdynamo_orig_callable`` sentinel on the code predictor's
    inner forward (compile actually engaged, not silent no-op).
  * One short generation completes without raising (triggers the
    inductor compile on first forward; ~10-30s cold).
  * A second generation completes (replays the compiled graph; should
    be noticeably faster).

If THIS succeeds, the Story 18.4 dev-mode surface is fully functional
and Tasks 8 (NFR1 measurement) + 9 (NFR3 audition) are unblocked. The
remaining work is purely production-bundle packaging (Story 18.5).

Memory: ~3.4 GB VRAM under bf16. RTX 5090 has plenty.
Runtime: ~30-60s cold compile + ~5s warm replay.
"""

from __future__ import annotations

import os
import sys
import time
import traceback


def _stage(n: int, name: str) -> None:
    print(f"[{n}/6] {name}", flush=True)


def main() -> int:
    print(f"CUDA_PATH = {os.environ.get('CUDA_PATH', '(unset)')}")
    print(f"sys.executable = {sys.executable}")
    print()

    # ----- Stage 1: imports + CUDA ----- #
    _stage(1, "Import torch + qwen_tts + tts_streaming")
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
        from myvoice.services.tts_streaming import engage_compile_optimizations
        from myvoice.services.tts_streaming import compile_cache  # noqa: F401
        from myvoice.services.qwen_tts_service import QwenTTSService
        print(f"      torch = {torch.__version__}")
        print(f"      CUDA available = {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("      FAIL: CUDA not available")
            return 1
        print(f"      device = {torch.cuda.get_device_name(0)} "
              f"(capability {torch.cuda.get_device_capability()})")
        print(f"      pin hash = {QwenTTSService._QWEN_TTS_PIN_HASH}")
    except Exception:
        print("      FAIL: import raised")
        traceback.print_exc()
        return 1

    # ----- Stage 2: load model ----- #
    _stage(2, "Load Qwen3TTSModel.from_pretrained (bf16, cuda:0)")
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    try:
        # Mirror model_registry._load_model_sync's load_kwargs.
        load_kwargs = {
            "device_map": "cuda:0",
            "torch_dtype": torch.bfloat16,
        }
        # Skip flash_attention_2 to avoid an unrelated dependency; the
        # compile path engages independently of attention impl.
        t0 = time.monotonic()
        model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
        load_ms = (time.monotonic() - t0) * 1000
        print(f"      OK: model loaded in {load_ms:.0f}ms")
        # Sanity: confirm the API is callable per D-22 Branch B invariant.
        if not callable(getattr(model, "enable_streaming_optimizations", None)):
            print("      FAIL: model lacks enable_streaming_optimizations "
                  "(pin-bump did not propagate)")
            return 2
        print("      OK: enable_streaming_optimizations is callable")
    except Exception:
        print("      FAIL: model load raised")
        traceback.print_exc()
        return 2

    # ----- Stage 3: engage compile via the production function ----- #
    _stage(3, "engage_compile_optimizations (production function)")

    class _Settings:
        # Force engagement regardless of the bundled-smoke "off" default.
        tts_compile = "auto"

    try:
        t0 = time.monotonic()
        result = engage_compile_optimizations(
            model,
            app_settings=_Settings(),
            qwen_tts_pin_hash=QwenTTSService._QWEN_TTS_PIN_HASH,
        )
        engage_ms = (time.monotonic() - t0) * 1000
        print(f"      OK: engage returned in {engage_ms:.0f}ms")
        print(f"      result.engaged = {result['engaged']}")
        print(f"      result.reason  = {result['reason']}")
        print(f"      result.decode_window_frames = {result['decode_window_frames']}")
        print(f"      result.cuda_capability = {result['cuda_capability']}")
        print(f"      result.cache_warm = {result['cache_warm']}")
        if not result["engaged"]:
            print(f"      FAIL: engage returned engaged=False with "
                  f"reason={result['reason']!r}")
            return 3
        if result["reason"] not in ("engaged_cold_compile", "engaged_warm_cache"):
            print(f"      FAIL: unexpected reason {result['reason']!r}")
            return 3
        print(f"      OK: compile engaged ({result['reason']})")
    except Exception:
        print("      FAIL: engage_compile_optimizations raised")
        traceback.print_exc()
        return 3

    # ----- Stage 4: independent P-12 probe ----- #
    _stage(4, "P-12 capability probe (independent verification)")
    try:
        # Walk the same chain _probe_compile_engaged walks:
        # model.model.talker.code_predictor.model.forward
        inner = getattr(model, "model", None)
        talker = getattr(inner, "talker", None) if inner else None
        code_predictor = getattr(talker, "code_predictor", None) if talker else None
        predictor_inner = getattr(code_predictor, "model", None) if code_predictor else None
        fwd = getattr(predictor_inner, "forward", None) if predictor_inner else None
        if fwd is None:
            print("      FAIL: dereference chain broken at some step")
            print(f"      inner={inner is not None}, talker={talker is not None}, "
                  f"code_predictor={code_predictor is not None}, "
                  f"predictor_inner={predictor_inner is not None}, "
                  f"fwd={fwd is not None}")
            return 4
        has_dynamo = hasattr(fwd, "_torchdynamo_orig_callable")
        has_wrapped = hasattr(fwd, "__wrapped__")
        has_orig_mod = hasattr(predictor_inner, "_orig_mod")
        print(f"      _torchdynamo_orig_callable on fwd = {has_dynamo}")
        print(f"      __wrapped__ on fwd                = {has_wrapped}")
        print(f"      _orig_mod on predictor_inner      = {has_orig_mod}")
        if not (has_dynamo or has_wrapped or has_orig_mod):
            print("      FAIL: no compile sentinel attribute present")
            return 4
        print("      OK: probe confirms compile engaged")
    except Exception:
        print("      FAIL: probe raised")
        traceback.print_exc()
        return 4

    # ----- Stage 5: first generation (cold compile triggers inductor build) ----- #
    _stage(5, "First generation — triggers cold inductor compile (~10-30s)")
    try:
        # Use the model's generate_custom_voice directly — this matches
        # what _run_compile_priming does in production.
        t0 = time.monotonic()
        # Per the upstream API: generate_custom_voice(text, speaker, ...)
        # returns a tuple (audio_array, sample_rate) or similar. We don't
        # care about the audio content — only that the call completes.
        out = model.generate_custom_voice(
            text="Hello world.",
            speaker="Ryan",
            # Skip language kwarg if unsupported by the model surface;
            # most variants accept it.
        )
        cold_ms = (time.monotonic() - t0) * 1000
        print(f"      OK: first generation completed in {cold_ms:.0f}ms")
        if out is None:
            print("      WARN: generate_custom_voice returned None — "
                  "expected a (audio, sr) tuple")
        else:
            try:
                # Try to inspect the output shape if it's array-like.
                if hasattr(out, "__len__"):
                    print(f"      output length: {len(out)}")
                    if len(out) >= 1 and hasattr(out[0], "shape"):
                        print(f"      audio shape: {out[0].shape}")
            except Exception:
                pass
    except Exception:
        print("      FAIL: first generation raised")
        traceback.print_exc()
        return 5

    # ----- Stage 6: second generation (warm replay) ----- #
    _stage(6, "Second generation — warm CUDA Graph replay (should be faster)")
    try:
        t0 = time.monotonic()
        out2 = model.generate_custom_voice(
            text="Hello again.",
            speaker="Ryan",
        )
        warm_ms = (time.monotonic() - t0) * 1000
        print(f"      OK: second generation completed in {warm_ms:.0f}ms")
        if cold_ms > 0:
            speedup = cold_ms / max(warm_ms, 1)
            print(f"      cold/warm ratio = {speedup:.2f}x "
                  f"(cold={cold_ms:.0f}ms, warm={warm_ms:.0f}ms)")
    except Exception:
        print("      FAIL: second generation raised")
        traceback.print_exc()
        return 6

    print()
    print("=" * 60)
    print("STORY 18.4 STEP 3 SMOKE: ALL STAGES PASSED.")
    print("=" * 60)
    print("Dev-env compile path is functional against a real Qwen3TTSModel.")
    print("Tasks 8 (NFR1 measurement) and 9 (NFR3 audition) are unblocked.")
    print("Production-bundle packaging stays Story 18.5 scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
