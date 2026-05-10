"""TF32 + cuDNN benchmark autotune enable for Ampere+ CUDA hosts.

Story 18.2 — Phase ⊥-Polish-2 of D-20 (architecture-optimization-pass.md).
Successor to Story 18.1 (instrumentation-only producer-bottleneck pin).

Architecture references:
  - D-9 (architecture-optimization-pass.md:257) — hardware-aware default
    discipline. The Ampere+ guard mirrors `streaming_mode.py:54-56`'s
    `torch.cuda.is_available()` precedent and extends it with a compute-
    capability major check (>= 8) so pre-Ampere CUDA hosts (Turing 7.5,
    Pascal 6.x) stay on the V2 baseline alongside CPU-only hosts.
  - NFR12 (architecture-optimization-pass.md:75) — CPU-only support. The
    cuda-unavailable branch is the AC #2 vehicle for this contract:
    no flag mutation; no behavior change; default value "0.0" telemetry
    breadcrumb so observability of the skip is symmetric with engagement.
  - D-19 telemetry (architecture-optimization-pass.md:286+ + helper
    surface at metrics.py:77). The new `tf32_cudnn_benchmark_enabled`
    metric extends the established pub-sub helper without new
    infrastructure.

Module discipline:
  - Mirrors Story 16.2's `streaming_mode.py` lazy-torch-import pattern so
    `monkeypatch.setattr("torch.cuda.is_available", ...)` is honored by
    tests without import-order gymnastics.
  - No peer imports from `myvoice.*` aside from the metrics helper
    (architecture line 678-679: "EVERYTHING may import metrics; metrics
    imports nothing"). The metrics import is allowed under that rule.
  - **Stateless**: idempotency is enforced by reading the three
    `torch.backends.*` flag values on entry and short-circuiting if all
    three are already True on an Ampere+ host. There is no module-level
    "have we run" boolean — the flag values themselves are the canonical
    signal. This preserves Story 16.2's pure-function discipline; the
    only deliberate departure is the side-effect set itself, kept
    minimal and concentrated in one place.

Public surface (Task 1.4 widens `tts_streaming/__init__.py`):
  - is_ampere_or_newer() -> bool: pure hardware probe.
  - enable_tf32_and_cudnn_benchmark() -> dict: idempotent enable + log
    + telemetry; returns status dict.

Telemetry breadcrumb (the metric Stories 18.3 + 18.4 baseline against):
  - Engaged: metrics.record("tf32_cudnn_benchmark_enabled", 1.0,
                             device_capability="<major>.<minor>")
  - Skipped: metrics.record("tf32_cudnn_benchmark_enabled", 0.0,
                             reason="cuda_unavailable" | "pre_ampere",
                             device_capability="<major>.<minor>" | "none")
  Both branches always include `device_capability` (string) for downstream
  parser uniformity. The cuda-unavailable branch uses the string sentinel
  "none" rather than omitting the tag — committed for schema uniformity
  per Story 18.2 OQ #2.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from myvoice.observability import metrics


_logger = logging.getLogger(__name__)

_METRIC_NAME = "tf32_cudnn_benchmark_enabled"

# Compute-capability major mapping (informational; the test is `major >= 8`):
#   Pascal       = 6.x   (GTX 10xx)        — pre-Ampere, no TF32 tensor cores
#   Volta        = 7.0   (Tesla V100)      — pre-Ampere, no TF32 tensor cores
#   Turing       = 7.5   (RTX 20xx)        — pre-Ampere, no TF32 tensor cores
#   Ampere       = 8.0/8.6/8.9 (A100, RTX 30xx, RTX 40xx) — TF32 OK
#   Hopper       = 9.0   (H100)            — TF32 OK
#   Blackwell DC = 10.0  (B100/B200)       — TF32 OK
#   Blackwell GF = 12.0  (RTX 50xx)        — TF32 OK (verified on RTX 5090
#                                            during Story 18.2 Task 4.5
#                                            smoke run 2026-05-09)
# The `>= 8` major check is forward-compatible: any future architecture
# with major >= 8 satisfies it without code change. Note that NVIDIA's
# major-version assignment is not always strictly chronological — the
# RTX 50xx GeForce variant of Blackwell reports 12.0 even though the
# datacenter B100/B200 Blackwell reports 10.0; both engage TF32 correctly.
_AMPERE_CAPABILITY_MAJOR = 8


def is_ampere_or_newer() -> bool:
    """Pure hardware probe: is this host Ampere-or-newer CUDA?

    Returns False if `torch.cuda.is_available()` is False (CPU-only or
    torch-not-installed). Otherwise returns
    `torch.cuda.get_device_capability()[0] >= 8`.

    Defensive ordering: do NOT call `get_device_capability()` if
    `is_available()` is False — on a CPU-only system, the device may not
    exist and the call may error or warn. The early-out is the contract,
    not a polish.

    Lazy-imports torch at function-call time so test monkeypatching of
    `torch.cuda.is_available` and `torch.cuda.get_device_capability` is
    honored without import-order gymnastics (mirrors Story 16.2's
    `streaming_mode.py` discipline).
    """
    import torch  # lazy: see docstring rationale
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= _AMPERE_CAPABILITY_MAJOR


def _device_capability_str(cap: Optional[Tuple[int, int]]) -> str:
    """Format `(major, minor)` as `"<major>.<minor>"`; None → `"none"`.

    The string sentinel "none" is the committed schema for the
    cuda-unavailable branch (Story 18.2 OQ #2). Downstream parsers
    (Story 18.1's CSV-capture infrastructure stringifies tag values)
    see a uniform schema across both engaged and skipped branches.
    """
    if cap is None:
        return "none"
    return f"{cap[0]}.{cap[1]}"


def _all_three_flags_already_true() -> bool:
    """Idempotency check: are all three target flags already True?

    Reads the live `torch.backends.*` values rather than a module-level
    cache. The flag values themselves are the canonical "have we run"
    signal — there is no shadow state to keep in sync.
    """
    import torch  # lazy: see is_ampere_or_newer rationale
    return (
        torch.backends.cuda.matmul.allow_tf32 is True
        and torch.backends.cudnn.allow_tf32 is True
        and torch.backends.cudnn.benchmark is True
    )


def enable_tf32_and_cudnn_benchmark() -> dict:
    """Idempotent: enable TF32 + cuDNN benchmark on Ampere+ CUDA hosts.

    Returns a status dict so the wire-up at `main.py` can log a single
    line carrying actionable detail without re-probing torch::

        {
            "engaged": bool,                          # True iff flags now set
            "reason":  str | None,                    # None if engaged
            "device_capability": tuple[int, int] | None,
        }

    Behavior (AC #1 + #2 + #3 + idempotency contract):

      * **CPU-only or torch-not-installed**: no flag mutation; one
        DEBUG log; one telemetry record with value=0.0 + reason
        "cuda_unavailable" + device_capability="none".

      * **Pre-Ampere CUDA host** (compute < 8.0 — Turing 7.5, Volta 7.0,
        Pascal 6.x): no flag mutation; one DEBUG log; one telemetry
        record with value=0.0 + reason "pre_ampere" + the actual
        device_capability string.

      * **Ampere+ CUDA host, first call**: set the three
        `torch.backends.*` flags to True; one INFO log; one telemetry
        record with value=1.0 + the actual device_capability string.

      * **Ampere+ CUDA host, subsequent calls (idempotency)**: detect
        "all three flags already True" on entry, log at DEBUG (no second
        INFO), do NOT re-emit the metric, return the engaged-True
        status dict. The function is safe to call any number of times.

    Telemetry: `metrics.record(_METRIC_NAME, ...)`. Listener exceptions
    are swallowed by `metrics.record` per the metrics module's own AC #9
    (metrics.py:144-region) — startup MUST NOT abort because a buggy
    listener raised. The function returns the engaged status dict
    regardless of telemetry-listener health.

    Raises: nothing under normal operation. If torch is genuinely
    uninstallable (rare; the `try import torch` block in `main.py:42-49`
    has already absorbed the typical case), the lazy import inside the
    probe will raise; the caller in `main.py` wraps the whole call in
    `try / except` per Task 2.1 — the speedup is opt-in and absence is
    the V2 baseline.
    """
    # Probe first. is_ampere_or_newer() handles the
    # cuda.is_available() guard; we only need to call
    # get_device_capability() ourselves for the
    # tag string + the pre-Ampere reason branch.
    import torch  # lazy: see is_ampere_or_newer rationale

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        # CPU-only branch (or torch.cuda absent / disabled).
        metrics.record(
            _METRIC_NAME,
            0.0,
            reason="cuda_unavailable",
            device_capability=_device_capability_str(None),
        )
        _logger.debug(
            "TF32 + cuDNN benchmark skipped: cuda_unavailable"
        )
        return {
            "engaged": False,
            "reason": "cuda_unavailable",
            "device_capability": None,
        }

    capability = torch.cuda.get_device_capability()
    cap_str = _device_capability_str(capability)

    if capability[0] < _AMPERE_CAPABILITY_MAJOR:
        # Pre-Ampere CUDA host (Turing 7.5, Volta 7.0, Pascal 6.x).
        metrics.record(
            _METRIC_NAME,
            0.0,
            reason="pre_ampere",
            device_capability=cap_str,
        )
        _logger.debug(
            "TF32 + cuDNN benchmark skipped: pre_ampere "
            "(device_capability=%s)",
            cap_str,
        )
        return {
            "engaged": False,
            "reason": "pre_ampere",
            "device_capability": capability,
        }

    # Ampere+ CUDA host. Idempotency check before mutation.
    if _all_three_flags_already_true():
        # Already engaged — no-op. Single DEBUG line, no second
        # INFO, no second metric record. The flag values
        # themselves are the canonical signal.
        _logger.debug(
            "TF32 + cuDNN already enabled — no-op "
            "(device_capability=%s)",
            cap_str,
        )
        return {
            "engaged": True,
            "reason": None,
            "device_capability": capability,
        }

    # First-call engagement: set the three flags, log INFO, emit metric.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    metrics.record(
        _METRIC_NAME,
        1.0,
        device_capability=cap_str,
    )
    _logger.info(
        "TF32 + cuDNN benchmark enabled (device_capability=%s)",
        cap_str,
    )
    return {
        "engaged": True,
        "reason": None,
        "device_capability": capability,
    }
