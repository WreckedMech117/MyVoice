"""Story 18.4 / OQ #4 — triton-on-Windows dev-env smoke.

Minimal probe: can torch.compile(..., mode="reduce-overhead") actually
JIT-compile a CUDA kernel via triton in the portable-Python310
environment, after Steps 1-3 of the smoke playbook are in place?

No Qwen-TTS, no MyVoice surface. If THIS fails, the whole D-22 Branch B
compile commitment is unworkable in the current dev environment and
something more fundamental needs fixing. If this succeeds, the
bundled-smoke failure mode is purely a PyInstaller bundling problem
(Story 18.5 scope).

Reports five probe stages, exit 0 on full success.
"""

from __future__ import annotations

import os
import sys
import traceback


def _stage(n: int, name: str) -> None:
    print(f"[{n}/5] {name}", flush=True)


def main() -> int:
    # Stage 1: env sanity.
    _stage(1, "Environment check")
    cuda_path = os.environ.get("CUDA_PATH", "")
    print(f"      CUDA_PATH = {cuda_path!r}")
    if not cuda_path:
        print("      WARN: CUDA_PATH unset in this process env; triton may "
              "fall back to a default search path. Set it before invoking "
              "if probe fails at compile time.")
    print(f"      sys.executable = {sys.executable!r}")
    print(f"      Python = {sys.version.splitlines()[0]}")

    # Stage 2: torch + CUDA visible.
    _stage(2, "torch + CUDA visibility")
    try:
        import torch
        print(f"      torch = {torch.__version__}")
        if not torch.cuda.is_available():
            print("      FAIL: torch.cuda.is_available() is False")
            return 2
        cap = torch.cuda.get_device_capability()
        print(f"      device = {torch.cuda.get_device_name(0)} "
              f"(capability {cap[0]}.{cap[1]})")
    except Exception:
        print("      FAIL: torch import or CUDA probe raised")
        traceback.print_exc()
        return 2

    # Stage 3: triton importable.
    _stage(3, "triton importable")
    try:
        import triton
        import triton.language as tl  # noqa: F401
        print(f"      triton = {triton.__version__}")
        print(f"      triton at {triton.__file__}")
    except Exception:
        print("      FAIL: triton import raised")
        traceback.print_exc()
        return 3

    # Stage 4: pure torch.compile on a trivial function (no triton path).
    # `mode="default"` typically lowers to ATen/inductor without engaging
    # triton kernel generation for simple ops. This stage verifies the
    # inductor backend is even reachable before stressing the triton path.
    _stage(4, "torch.compile (default mode) on tiny CUDA fn")
    try:
        x = torch.randn(8, device="cuda")
        def _trivial(t):
            return t * 2.0 + 1.0
        compiled = torch.compile(_trivial, mode="default")
        y = compiled(x)
        torch.cuda.synchronize()
        # Sanity-check the math.
        expected = x * 2.0 + 1.0
        if not torch.allclose(y, expected):
            print("      FAIL: compiled output != eager output")
            return 4
        print("      OK: compiled output matches eager")
    except Exception:
        print("      FAIL: torch.compile(mode='default') raised")
        traceback.print_exc()
        return 4

    # Stage 5: THE REAL TEST — reduce-overhead mode. This forces CUDA Graph
    # capture, which is exactly what `enable_streaming_optimizations`
    # invokes for the Qwen-TTS codebook predictor. If this stage fails
    # with the same "Cannot find a working triton installation" the
    # bundled smoke surfaced, the dev-env path is also broken.
    _stage(5, "torch.compile(mode='reduce-overhead') — the Story 18.4 target")
    try:
        x = torch.randn(8, device="cuda")
        def _trivial2(t):
            return t * 3.0 - 1.0
        compiled = torch.compile(_trivial2, mode="reduce-overhead")
        expected = x * 3.0 - 1.0
        # First call: captures the CUDA Graph (cold compile).
        # Clone the output immediately because CUDA Graphs reuse the
        # output buffer on replay — holding a stale reference past the
        # next call would raise.
        y1 = compiled(x).clone()
        torch.cuda.synchronize()
        if not torch.allclose(y1, expected):
            print("      FAIL: cold-compile output != eager output")
            return 5
        print("      OK (5a): cold compile produced correct output")
        # Second call: graph replay (warm path).
        y2 = compiled(x).clone()
        torch.cuda.synchronize()
        if not torch.allclose(y2, expected):
            print("      FAIL: warm-replay output != eager output")
            return 5
        print("      OK (5b): CUDA Graph replay produced correct output")
        # Third call: pin the replay invariant — same input, same output.
        y3 = compiled(x).clone()
        torch.cuda.synchronize()
        if not torch.allclose(y3, expected):
            print("      FAIL: third-call output diverged")
            return 5
        print("      OK (5c): third replay stable")
    except Exception:
        print("      FAIL: torch.compile(mode='reduce-overhead') raised — "
              "this would be the same failure class as the bundled smoke")
        traceback.print_exc()
        return 5

    print()
    print("=" * 60)
    print("ALL FIVE STAGES PASSED. The dev-env triton path is functional.")
    print("Next: flip tts_compile='auto' in settings.json and run the "
          "Task 8 NFR1 N=10 measurement loops.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
