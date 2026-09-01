"""TF32 + cuDNN benchmark autotune enable + TTS precision resolver for Ampere+ CUDA hosts.

Story 18.2 — Phase ⊥-Polish-2 of D-20 (architecture-optimization-pass.md).
Successor to Story 18.1 (instrumentation-only producer-bottleneck pin).
Story 18.3 extends this module with ``resolve_tts_precision`` — a sibling
pure-decision function that mirrors the same Ampere+ probe gate
(``is_ampere_or_newer()``) for the precision dimension. See
``epics-optimization-pass.md`` lines 1370–1386 (Story 18.3 stub) and the
mirror-pattern guidance in the Story 18.3 file's "Source tree components
to touch" section.

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
  - resolve_tts_precision(override) -> torch.dtype: pure-decision
    precision resolver (Story 18.3); side-effect-free; user override
    wins; "auto"/None defers to ``is_ampere_or_newer()``.

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
import os
from typing import Any, Optional, Tuple

from myvoice.observability import metrics


_logger = logging.getLogger(__name__)

_METRIC_NAME = "tf32_cudnn_benchmark_enabled"
_METRIC_NAME_TTS_PRECISION = "tts_precision_resolved"
_METRIC_NAME_COMPILE = "tts_compile_engaged"

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

    Returns a status dict primarily for test surface — tests assert on
    ``engaged`` / ``reason`` / ``device_capability`` to verify the four
    hardware truth-table branches without scraping log records. The
    function's own INFO/DEBUG log is the canonical runtime breadcrumb;
    callers do NOT need to log the dict (doing so would duplicate the
    breadcrumb the function already emits)::

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
    # NOTE: is_ampere_or_newer() is the canonical pure probe (used by
    # callers that just want a yes/no), but this function does NOT call
    # it — we re-implement the cuda.is_available() + get_device_capability()
    # check inline so the same `capability` tuple feeds both the Ampere+
    # gate AND the device_capability tag string + pre-Ampere reason
    # branch without two get_device_capability() round-trips.
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


# --------------------------------------------------------------------------- #
# Story 18.3 — TTS precision resolver
# --------------------------------------------------------------------------- #


def resolve_tts_precision(override: Optional[str]) -> "torch.dtype":  # type: ignore[name-defined]  # noqa: F821
    """Pure-decision: resolve the TTS model's torch dtype from a settings override.

    Story 18.3 — adds explicit hardware-aware precision engagement on the
    qwen-tts talker + decoder. Mirrors Story 16.2's
    ``effective_streaming_mode`` and Story 18.2's ``is_ampere_or_newer``
    pure-decision discipline: no logging, no metric emission, no flag
    mutation. The side-effect of the chosen dtype landing on the model
    is at the call site in ``ModelRegistry.__init__`` (which also owns
    the INFO log + the telemetry metric).

    Precedence rule:

      * ``override == "fp32"`` → ``torch.float32`` unconditionally
        (NFR7 fallback path; engages even on Ampere+ when the user has
        observed a perceptual defect in bf16 mode).
      * ``override == "bf16"`` → ``torch.bfloat16`` unconditionally
        (advanced-user opt-in; engages even on CPU / pre-Ampere despite
        the slowdown — the user has explicitly opted in).
      * ``override == "auto"`` or ``override is None`` → defers to the
        hardware probe: ``torch.bfloat16`` on Ampere+ CUDA hosts (where
        bf16 tensor cores accelerate matmul) and ``torch.float32``
        elsewhere (CPU, pre-Ampere CUDA — closes the latent D-9 / NFR12
        violation in the V2 ``dtype="bfloat16"`` default that applied
        unconditionally).

    Side-effect discipline: this function does NOT log, does NOT emit a
    telemetry metric, and does NOT mutate any ``torch.backends.*`` flag.
    Stories 16.2 + 18.2 established the pattern — pure decision functions
    in ``tts_streaming/`` are easier to reason about and test, and the
    decision's runtime visibility is concentrated in one place at the
    call site (``ModelRegistry.__init__`` owns the INFO log line +
    ``tts_precision_resolved`` metric for this resolver).

    Args:
        override: One of ``"auto"``, ``"bf16"``, ``"fp32"``, or ``None``.
            Unknown values are NOT remapped here — the AppSettings
            validator at ``app_settings.py``'s ``__post_init__`` is the
            input-validation layer (warn-and-fallback to ``"auto"``).
            By the time this resolver runs, the value should already be
            one of the three valid strings or ``None``.

    Returns:
        ``torch.dtype``: The resolved dtype for ``Qwen3TTSModel.from_pretrained(..., torch_dtype=...)``.
    """
    import torch  # lazy: same rationale as is_ampere_or_newer

    if override == "fp32":
        return torch.float32
    if override == "bf16":
        return torch.bfloat16
    # override == "auto" OR override is None → defer to hardware probe.
    if is_ampere_or_newer():
        return torch.bfloat16
    return torch.float32


# --------------------------------------------------------------------------- #
# Story 18.4 — torch.compile + CUDA Graph engagement via upstream API
# --------------------------------------------------------------------------- #


def _probe_compile_engaged(model: Any) -> bool:
    """P-12 capability probe — has ``torch.compile`` actually wrapped a target?

    Architecture P-12 (line 674-713 of
    ``architecture-streaming-acceleration-and-lightning-tier.md``) requires
    verifying that ``enable_streaming_optimizations`` actually engaged compile,
    not merely returned without raising. The fork's added 50 lines could be a
    no-op stub; the probe is the canonical signal.

    The fork's ``enable_streaming_optimizations`` (when called with our
    ``compile_talker=False`` argument — see ``engage_compile_optimizations``
    for the rationale) compiles the codebook predictor by default. The
    fork's ``CodePredictor.enable_compile`` executes
    ``self.model.forward = torch.compile(self.model.forward, mode=mode, fullgraph=False)``.
    After compile, the wrapped function carries the
    ``_torchdynamo_orig_callable`` attribute (PyTorch's canonical wrapped-
    function sentinel since 2.0). The probe walks
    ``model.model.talker.code_predictor.model.forward`` and checks for that
    attribute.

    Why NOT the talker forward: we pass ``compile_talker=False`` to preserve
    Story 16.8's TRUE_STREAM forward-hook (the talker's forward-hook captures
    per-step `codec_ids`; torch.compile-wrapping breaks that path). With
    `compile_talker=False`, the talker's inner forward stays eager, so a
    talker-targeted probe would always return False — wrong signal.

    Defensive: returns False if any step of the dereference chain returns
    None or a non-callable. The False return triggers the NFR7 graceful-
    degradation fallback (eager-mode generation stays available).

    Args:
        model: The ``Qwen3TTSModel`` wrapper instance.

    Returns:
        True iff the codebook predictor's inner forward is torch.compile-wrapped.
    """
    # Dereference chain: Qwen3TTSModel → Qwen3TTSForConditionalGeneration →
    # Qwen3TTSTalkerForConditionalGeneration → CodePredictor → inner model → forward.
    inner = getattr(model, "model", None)
    if inner is None:
        return False
    talker = getattr(inner, "talker", None)
    if talker is None:
        return False
    code_predictor = getattr(talker, "code_predictor", None)
    if code_predictor is None:
        return False
    predictor_inner = getattr(code_predictor, "model", None)
    if predictor_inner is None:
        return False
    fwd = getattr(predictor_inner, "forward", None)
    if fwd is None:
        return False
    # torch.compile(fn) returns a wrapper function with this attribute
    # since PyTorch 2.0. Defensive against renames: also accept the
    # ``__wrapped__`` and ``_orig_mod`` attributes (the wrapper variants
    # for module-level and bound-method compile targets).
    return (
        hasattr(fwd, "_torchdynamo_orig_callable")
        or hasattr(fwd, "__wrapped__")
        or hasattr(predictor_inner, "_orig_mod")
    )


# --------------------------------------------------------------------------- #
# Reload-compile diagnostic helpers (compile-disengage-post-generation spec)
# --------------------------------------------------------------------------- #

# Cycle correlation: there is NO module-level counter. The canonical cycle_idx
# is owned by ``ModelRegistry._reload_cycle_counter`` and threaded through to
# this module via the ``reload_cycle_idx`` parameter on
# ``engage_compile_optimizations``. This preserves TD-1's "diff post_unload vs
# reload_gate by cycle_idx" contract: both stages of the same unload→load pair
# share the same integer because both read from the same registry counter.

_DIAGNOSTIC_PROBE_KEYS = (
    "stage",
    "cycle_idx",
    "is_available",
    "device_count",
    "in_bad_fork",
    "initialized",
    "last_cuda_error",
    "cuda_path_present",
    "path_has_cuda_redist",
)


def collect_compile_gate_diagnostic(stage: str, cycle_idx: int) -> dict:
    """Pure-decision probe: capture CUDA + env-var state at a single point in time.

    Called at two emission points per unload/reload cycle:

      * ``stage="post_unload"`` — invoked from ``ModelRegistry._unload_model``
        immediately after ``torch.cuda.empty_cache()``. Captures CUDA state
        right after the allocator pool release.
      * ``stage="reload_gate"`` — invoked from ``engage_compile_optimizations``
        on the ``cuda_unavailable`` branch. Captures the state that produced
        the gate flip.
      * ``stage="reload_engaged"`` — invoked from the engaged branches when
        ``MYVOICE_RELOAD_COMPILE_DIAGNOSTIC=1`` is set. Lets the verdict-table
        driver diff "fix engaged" vs "fix didn't help" emissions.

    The dict's values are all serializable as ``int|float|str|bool`` so the
    ``metrics.record`` validator at ``observability/metrics.py:106-110``
    accepts them as tags.

    Pure-decision discipline: this helper does NOT log, does NOT emit a
    metric. The caller owns observability. Each probe is wrapped in
    ``try/except`` so a single broken probe yields a ``"probe_error:..."``
    sentinel rather than aborting the dict assembly.
    """
    # Lazy import — honors monkeypatch.setattr("torch.cuda.is_available", ...)
    # without import-order gymnastics (mirrors is_ampere_or_newer).
    import torch

    probe: "dict[str, Any]" = {
        "stage": stage,
        "cycle_idx": int(cycle_idx),
    }

    try:
        probe["is_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 — defensive: probe-only
        probe["is_available"] = f"probe_error: {type(exc).__name__}"

    try:
        probe["device_count"] = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        probe["device_count"] = f"probe_error: {type(exc).__name__}"

    try:
        probe["in_bad_fork"] = bool(torch.cuda._is_in_bad_fork())
    except Exception as exc:  # noqa: BLE001
        probe["in_bad_fork"] = f"probe_error: {type(exc).__name__}"

    try:
        probe["initialized"] = bool(torch.cuda.is_initialized())
    except Exception as exc:  # noqa: BLE001
        probe["initialized"] = f"probe_error: {type(exc).__name__}"

    # Last CUDA error — try to surface the exception class name from a
    # ``_cuda_getDeviceCount`` call. On healthy CUDA this returns cleanly
    # (the "none" sentinel matches the convention from
    # ``_device_capability_str``). On a poisoned context, the exception
    # class name reveals the failure mode (RuntimeError, OSError, etc.).
    try:
        _ = torch.cuda._cuda_getDeviceCount()
        probe["last_cuda_error"] = "none"
    except Exception as exc:  # noqa: BLE001
        probe["last_cuda_error"] = type(exc).__name__

    # Environment-state probes — pure os.environ reads, no torch involvement.
    try:
        probe["cuda_path_present"] = bool(os.environ.get("CUDA_PATH"))
    except Exception as exc:  # noqa: BLE001
        probe["cuda_path_present"] = f"probe_error: {type(exc).__name__}"

    try:
        path_value = os.environ.get("PATH", "")
        probe["path_has_cuda_redist"] = "cuda_redist" in path_value
    except Exception as exc:  # noqa: BLE001
        probe["path_has_cuda_redist"] = f"probe_error: {type(exc).__name__}"

    return probe


def engage_compile_optimizations(
    model: Any,
    *,
    app_settings: Optional[Any] = None,
    streamer_chunk_size: Optional[int] = None,
    streamer_lookahead: Optional[int] = None,
    decode_window_frames: Optional[int] = None,
    qwen_tts_pin_hash: Optional[str] = None,
    reload_cycle_idx: int = 0,
) -> dict:
    """Engage torch.compile + CUDA Graph replay on the qwen-tts talker + decoder.

    Story 18.4 — Phase ⊥-Polish-2 of D-20, Cluster E of
    ``architecture-streaming-acceleration-and-lightning-tier.md`` (sealed
    2026-05-10). Executes architecture decisions D-21 (decode_window
    bound to the streamer geometry -- 30 through Story 20.3, **15** from
    Story 20.4's chunk_size retune), D-22 Branch B (calls the
    upstream-blessed API the pin-bump engaged), D-25 (decode-window
    invariant), and P-12 (capability verification probe).

    The function gates engagement on:

      1. **D-9 hardware probe** — CUDA-available + Ampere-or-newer
         (capability major >= 8). CPU and pre-Ampere hosts stay on the V2
         baseline (eager-mode generation).
      2. **AppSettings.tts_compile** — three-valued ("auto" / "on" /
         "off"). "off" forces eager (NFR7 fallback); "on" + pre-Ampere
         emits a WARNING + ``pre_ampere_user_explicit`` telemetry; "auto"
         defers to the hardware probe.
      3. **D-22 Branch A invariant** — ``model.enable_streaming_optimizations``
         must be callable (the pin-bump should have ensured this; the
         assertion catches a hypothetical silent pin reversion per P-11).
      4. **D-25 decode-window invariant** — ``decode_window_frames`` must
         equal ``streamer_chunk_size + streamer_lookahead``, and those two
         default to the **live** ``codec_token_streamer`` module constants
         (today 10 + 5 = 15). Hard-fail on mismatch (not a warning log —
         silent fallthrough produces wrong audio).
      5. **P-12 capability probe** — after the API returns, verify that
         compile actually engaged (the API didn't no-op silently).

    The eight reason-enum values for the returned status dict map to the
    eight architectural branches:

        cuda_unavailable         — D-9 CPU-or-torch-absent
        pre_ampere               — D-9 pre-Ampere CUDA host
        pre_ampere_user_explicit — pre-Ampere + tts_compile="on" (WARNING)
        user_disabled            — tts_compile="off"
        engaged_warm_cache       — engaged; compile_cache.is_warm(key) == True
        engaged_cold_compile     — engaged; compile_cache.is_warm(key) == False
        compile_failed           — API raised; eager fallback returned
        probe_failed             — API returned; probe says compile didn't engage

    Telemetry contract (architecture P-15 + parent D-19 / P-9):
    ``metrics.record("tts_compile_engaged", 1.0 if engaged else 0.0,
                       reason=<enum>, decode_window_frames=<int>,
                       cuda_capability=<"major.minor">, cache_warm=<bool_str>)``

    NFR7 graceful degradation: the dispatch chain at
    ``qwen_tts_service._dispatch_by_streaming_mode`` is untouched by this
    function. Compile is a model-state mutation; the dispatch chain reads
    the same model object regardless of compile state. Any non-engaged
    branch leaves the model in eager mode; the dispatch chain continues
    to route TRUE_STREAM through the existing Story 16.8 talker-patch
    path without modification.

    Args:
        model: The ``Qwen3TTSModel`` instance returned by
            ``Qwen3TTSModel.from_pretrained(...)`` (typically owned by
            ``ModelRegistry._load_model_sync``).
        app_settings: Optional ``AppSettings`` instance; if provided, the
            function reads ``app_settings.tts_compile`` via ``getattr`` for
            defensive access (mirrors Story 18.3's ``tts_precision``
            precedent at ``model_registry.py:137``).
        streamer_chunk_size: The streamer's real chunk size. ``None`` (the
            canonical contract) resolves it from
            ``codec_token_streamer.DEFAULT_CHUNK_SIZE`` at call time, so the
            compile geometry cannot drift from the streamer geometry.
            Story 20.4 replaced a hard-coded ``25`` default here: with the
            sole production call site passing nothing, ``decode_window_frames``
            resolved to 30 regardless of what the streamer actually emitted
            (Story 20.1 §5.4 — the D-25 trap). Tests may pass explicit values
            to exercise the invariant.
        streamer_lookahead: The streamer's real lookahead. ``None`` resolves
            it from ``codec_token_streamer.DEFAULT_LOOKAHEAD`` (= 5 today).
            Same drift-proofing rationale as ``streamer_chunk_size``.
        decode_window_frames: Optional explicit decode-window override. If
            ``None`` (the typical caller contract), the function computes
            ``streamer_chunk_size + streamer_lookahead`` itself. If the
            caller passes an explicit value, the D-25 invariant asserts
            equality between explicit and computed — the architecture's
            "loud at startup" P-11 contract catches a divergent caller
            without permitting silent fallthrough.
        qwen_tts_pin_hash: Optional first-8-hex-chars of the active
            qwen-tts pin. If ``None``, defaults to ``"unknown"`` (cache
            key still computes; the dimension simply becomes a non-
            invalidating constant — acceptable degraded behavior). The
            canonical caller is ``ModelRegistry._load_model_sync``, which
            reads ``QwenTTSService._QWEN_TTS_PIN_HASH`` and passes it
            through, preserving module-boundary discipline (this module
            does NOT reach into ``services.qwen_tts_service``).

    Returns:
        Status dict::

            {
                "engaged": bool,                          # True iff compile engaged
                "reason": str,                            # one of the eight reasons
                "decode_window_frames": int,              # the resolved value (15 today)
                "cuda_capability": tuple | None,          # (major, minor) or None
                "cache_warm": bool | None,                # True/False on engage paths; None on skip
            }
    """
    # Lazy-import: torch (DLL ordering); compile_cache (avoid __init__.py
    # partial-init recursion since __init__.py imports both leaf modules).
    import torch
    from myvoice.services.tts_streaming import codec_token_streamer
    from myvoice.services.tts_streaming import compile_cache

    # ----- Resolve the streamer geometry from its single source of truth --- #
    # Story 20.4 (AC #1): read the LIVE module constants rather than carrying
    # a second copy of the literals here. Before this, the defaults were a
    # hard-coded ``25``/``5`` and ``model_registry`` passed neither, so a
    # streamer retune would silently tell the compile path the wrong window
    # (Story 20.1 §5.4). Resolving here means the two cannot diverge even if
    # a future call site also forgets to pass them.
    if streamer_chunk_size is None:
        streamer_chunk_size = codec_token_streamer.DEFAULT_CHUNK_SIZE
    if streamer_lookahead is None:
        streamer_lookahead = codec_token_streamer.DEFAULT_LOOKAHEAD

    # ----- D-25 invariant (architecture-bound; P-11 "loud at startup") ----- #
    # The streamer's chunk_size + lookahead is the canonical decode-window.
    # If the caller passes an explicit ``decode_window_frames``, it MUST
    # equal the streamer-derived value — a mismatch produces wrong audio
    # under CUDA Graph replay (the compiled graph captures one window
    # shape; emitting under a different shape diverges silently).
    # Architecture P-11 forbids ``log.warning(...)`` fallthrough — this
    # is a hard ``AssertionError`` at startup.
    streamer_window = streamer_chunk_size + streamer_lookahead
    if decode_window_frames is None:
        decode_window_frames = streamer_window
    else:
        assert decode_window_frames == streamer_window, (
            f"Decode-window drift: streamer expects "
            f"{streamer_window} (chunk_size={streamer_chunk_size} + "
            f"lookahead={streamer_lookahead}); caller passed "
            f"decode_window_frames={decode_window_frames}. CUDA Graph "
            f"replay would produce wrong audio. The architecture binds "
            f"D-21 to the streamer-derived value; pass None to inherit "
            f"or re-tune the streamer constants in lockstep with the "
            f"caller."
        )

    # ----- Resolve tts_compile setting ----- #
    tts_compile = getattr(app_settings, "tts_compile", None)
    if tts_compile is None:
        tts_compile = "auto"

    # ----- AppSettings gate: tts_compile == "off" ----- #
    if tts_compile == "off":
        metrics.record(
            _METRIC_NAME_COMPILE,
            0.0,
            reason="user_disabled",
            decode_window_frames=decode_window_frames,
            cuda_capability="none",
            cache_warm="none",
        )
        _logger.info("torch.compile skipped: user_disabled")
        return {
            "engaged": False,
            "reason": "user_disabled",
            "decode_window_frames": decode_window_frames,
            "cuda_capability": None,
            "cache_warm": None,
        }

    # ----- D-9 hardware probe ----- #
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        # Diagnostic emission — always on for the cuda_unavailable branch.
        # Uses the caller-supplied ``reload_cycle_idx`` so this emission
        # shares the same cycle_idx as the matching ``stage="post_unload"``
        # emission from ``_unload_model`` (TD-1 contract).
        try:
            _probe = collect_compile_gate_diagnostic(
                stage="reload_gate",
                cycle_idx=reload_cycle_idx,
            )
            metrics.record("reload_compile_diagnostic", 0.0, **_probe)
        except Exception as _diag_exc:  # noqa: BLE001 — diagnostic must not abort
            _logger.warning(
                "reload_compile_diagnostic emission failed (%s: %s); "
                "continuing with cuda_unavailable gate",
                type(_diag_exc).__name__,
                _diag_exc,
            )
        metrics.record(
            _METRIC_NAME_COMPILE,
            0.0,
            reason="cuda_unavailable",
            decode_window_frames=decode_window_frames,
            cuda_capability="none",
            cache_warm="none",
        )
        _logger.info("torch.compile skipped: cuda_unavailable")
        return {
            "engaged": False,
            "reason": "cuda_unavailable",
            "decode_window_frames": decode_window_frames,
            "cuda_capability": None,
            "cache_warm": None,
        }

    capability = torch.cuda.get_device_capability()
    cap_str = _device_capability_str(capability)

    if capability[0] < _AMPERE_CAPABILITY_MAJOR:
        # Pre-Ampere CUDA host. Two sub-branches based on user setting:
        # tts_compile="on" → WARNING + pre_ampere_user_explicit (the user
        # explicitly opted in; the WARNING surfaces the no-op decision).
        # Other values (auto / unknown) → silent pre_ampere skip.
        if tts_compile == "on":
            reason = "pre_ampere_user_explicit"
            _logger.warning(
                "torch.compile skipped: tts_compile='on' on pre-Ampere host "
                "(device_capability=%s); compile requires Ampere or newer.",
                cap_str,
            )
        else:
            reason = "pre_ampere"
            _logger.info(
                "torch.compile skipped: pre_ampere (device_capability=%s)",
                cap_str,
            )
        metrics.record(
            _METRIC_NAME_COMPILE,
            0.0,
            reason=reason,
            decode_window_frames=decode_window_frames,
            cuda_capability=cap_str,
            cache_warm="none",
        )
        return {
            "engaged": False,
            "reason": reason,
            "decode_window_frames": decode_window_frames,
            "cuda_capability": capability,
            "cache_warm": None,
        }

    # ----- D-22 Branch A invariant assertion ----- #
    # The pin-bump should have made this method callable. The assertion
    # catches a hypothetical silent pin reversion (e.g., a pip install
    # that didn't actually pick up the bumped pin). P-11 "loud at startup".
    assert callable(getattr(model, "enable_streaming_optimizations", None)), (
        "qwen-tts pin lacks enable_streaming_optimizations(); the pin-bump "
        "did not propagate. Re-run `pip install -e .` against the bumped "
        "requirements.txt or rebuild the production bundle."
    )

    # ----- Cache key + cache dir ----- #
    # Resolve the seven cache-key dimensions for compile_cache.compute_key().
    # The model_id and precision_str are best-effort extractions from the
    # loaded model; defaults are conservative (no cache pollution risk if
    # the attribute name shifts).
    model_id = (
        getattr(getattr(model, "model", None), "name_or_path", None)
        or getattr(model, "name_or_path", None)
        or "unknown"
    )
    model_dtype = getattr(getattr(model, "model", None), "dtype", None)
    precision_str = "bf16" if model_dtype == torch.bfloat16 else "fp32"
    # qwen-tts pin hash — caller-provided (ModelRegistry passes the
    # canonical QwenTTSService._QWEN_TTS_PIN_HASH constant through). This
    # module does NOT import services.qwen_tts_service: the architecture's
    # module-boundary rule (line 1157-1166 of
    # architecture-streaming-acceleration-and-lightning-tier.md) forbids
    # tts_streaming/* from reaching into services.*. ``None`` degrades
    # gracefully to ``"unknown"`` — cache still computes a stable key,
    # the pin dimension simply ceases to invalidate on a pin bump
    # (acceptable for callers that don't track pins, e.g. unit tests).
    pin_hash = qwen_tts_pin_hash if qwen_tts_pin_hash is not None else "unknown"
    compile_mode = "reduce-overhead"

    key = compile_cache.compute_key(
        qwen_tts_pin_hash=pin_hash,
        model_id=model_id,
        precision_str=precision_str,
        torch_version=torch.__version__,
        decode_window_frames=decode_window_frames,
        cuda_capability=capability,
        compile_mode=compile_mode,
    )
    is_warm_at_call = compile_cache.is_warm(key)
    compile_cache.set_torchinductor_cache_dir(key)

    # ----- API call (wrapped for compile_failed branch) ----- #
    # NOTE: `compile_talker=False` is REQUIRED for compatibility with
    # Story 16.8's TRUE_STREAM forward-hook. The Story 16.8 dispatch chain
    # (qwen_tts_service._build_true_stream_talker at :3404-:3571) patches
    # `model.model.talker.forward` to capture per-step `codec_ids` from
    # `Qwen3TTSTalkerOutputWithPast.hidden_states[1]`. When the fork's
    # `enable_streaming_optimizations(compile_talker=True)` wraps the
    # talker's inner forward via torch.compile, the codec_id extraction
    # path breaks and the talker emits zero chunks — observed in the
    # Story 18.4 bundled smoke 2026-05-10 (ValueError: finalize() called
    # with no chunks).
    #
    # Disabling talker compile preserves the dispatch chain's audited
    # correctness. The 2.15×-per-frame upstream-blessed gain lives on
    # the codebook predictor + decoder (fork's README); leaving the talker
    # eager forfeits only a fraction of the speedup while preserving
    # generation correctness. If a future architecture revision finds a
    # way to compose torch.compile with the Story 16.8 forward-hook
    # cleanly, this kwarg flips back to True.
    try:
        model.enable_streaming_optimizations(
            decode_window_frames=decode_window_frames,
            use_compile=True,
            compile_mode=compile_mode,
            compile_talker=False,
        )
    except Exception as exc:  # noqa: BLE001 — NFR7 graceful degradation: any compile failure → eager
        # NFR7 graceful degradation: log + record + return eager-fallback.
        # The dispatch chain is untouched; eager-mode generation stays
        # available.
        metrics.record(
            _METRIC_NAME_COMPILE,
            0.0,
            reason="compile_failed",
            decode_window_frames=decode_window_frames,
            cuda_capability=cap_str,
            cache_warm=str(is_warm_at_call),
            error=type(exc).__name__,
        )
        _logger.warning(
            "torch.compile skipped: compile_failed (%s: %s); eager-mode fallback engaged",
            type(exc).__name__,
            exc,
        )
        return {
            "engaged": False,
            "reason": "compile_failed",
            "decode_window_frames": decode_window_frames,
            "cuda_capability": capability,
            "cache_warm": is_warm_at_call,
            "error": str(exc),
        }

    # ----- P-12 capability probe ----- #
    if not _probe_compile_engaged(model):
        # API returned without raising but the model state is unchanged.
        # NFR7 fallback — eager-mode generation stays available.
        metrics.record(
            _METRIC_NAME_COMPILE,
            0.0,
            reason="probe_failed",
            decode_window_frames=decode_window_frames,
            cuda_capability=cap_str,
            cache_warm=str(is_warm_at_call),
        )
        _logger.warning(
            "torch.compile skipped: probe_failed (API returned but model state unchanged); "
            "eager-mode fallback engaged"
        )
        return {
            "engaged": False,
            "reason": "probe_failed",
            "decode_window_frames": decode_window_frames,
            "cuda_capability": capability,
            "cache_warm": is_warm_at_call,
        }

    # ----- Engaged: success path ----- #
    reason = "engaged_warm_cache" if is_warm_at_call else "engaged_cold_compile"
    cache_label = "warm" if is_warm_at_call else "cold"
    # Verbose diagnostic on engaged branches — opt-in via env var so the
    # verdict-table driver can collect per-row "fix engaged" emissions
    # without paying the probe cost in production.
    if os.environ.get("MYVOICE_RELOAD_COMPILE_DIAGNOSTIC") == "1":
        try:
            _probe = collect_compile_gate_diagnostic(
                stage="reload_engaged",
                cycle_idx=reload_cycle_idx,
            )
            metrics.record("reload_compile_diagnostic", 1.0, **_probe)
        except Exception as _diag_exc:  # noqa: BLE001
            _logger.warning(
                "reload_compile_diagnostic emission failed (%s: %s); "
                "continuing with engaged branch",
                type(_diag_exc).__name__,
                _diag_exc,
            )
    metrics.record(
        _METRIC_NAME_COMPILE,
        1.0,
        reason=reason,
        decode_window_frames=decode_window_frames,
        cuda_capability=cap_str,
        cache_warm=str(is_warm_at_call),
    )
    _logger.info(
        "torch.compile + CUDA Graph engaged "
        "(decode_window_frames=%d, cuda_capability=%s, cache=%s)",
        decode_window_frames,
        cap_str,
        cache_label,
    )
    return {
        "engaged": True,
        "reason": reason,
        "decode_window_frames": decode_window_frames,
        "cuda_capability": capability,
        "cache_warm": is_warm_at_call,
    }


# --------------------------------------------------------------------------- #
# Reload-compile fix dispatch (compile-disengage-post-generation spec)
# --------------------------------------------------------------------------- #

_VALID_RELOAD_FIX_VALUES = ("off", "post_gen", "rthook", "triton", "all")
_VALID_RELOAD_FIX_STAGES = ("post_unload", "pre_load")


def apply_reload_compile_fix(stage: str) -> None:
    """Single dispatch symbol for the three candidate fixes.

    Reads ``MYVOICE_RELOAD_COMPILE_FIX`` and dispatches to the appropriate
    helper(s) for the given ``stage``. Called from two points:

      * ``stage="post_unload"`` — from ``ModelRegistry._unload_model`` after
        the unload's ``empty_cache()``. The H1 helper actually lives inline
        in ``_unload_model`` (needs ``self``); this dispatch branch is a
        breadcrumb so the verdict-table driver can confirm execution.
      * ``stage="pre_load"`` — from ``ModelRegistry._load_model_sync`` right
        before ``engage_compile_optimizations``. H2 (rthook re-injection)
        and H3 (dynamo/triton reset) bodies run here.

    Unknown ``stage`` or fix values warn + treat as ``off``. Any helper
    exception is caught, logged at WARNING, and the dispatch returns
    normally — NFR7 graceful degradation: a buggy fix MUST NOT brick the
    app.

    Args:
        stage: ``"post_unload"`` or ``"pre_load"``.

    Returns:
        None. All observability is logging + metric emission inside the
        helpers themselves.
    """
    fix = os.environ.get("MYVOICE_RELOAD_COMPILE_FIX", "off")
    if fix not in _VALID_RELOAD_FIX_VALUES:
        _logger.warning(
            "MYVOICE_RELOAD_COMPILE_FIX=%r is not one of %s; treating as 'off'",
            fix,
            _VALID_RELOAD_FIX_VALUES,
        )
        fix = "off"

    if stage not in _VALID_RELOAD_FIX_STAGES:
        _logger.warning(
            "apply_reload_compile_fix called with unknown stage=%r; expected %s",
            stage,
            _VALID_RELOAD_FIX_STAGES,
        )
        return

    _logger.debug(
        "apply_reload_compile_fix(stage=%s) fix=%s", stage, fix
    )

    if fix == "off":
        return

    # H1 (post_gen) — actual hygiene helper runs inline in
    # ``ModelRegistry._unload_model`` (needs ``self``). This branch is a
    # breadcrumb so the verdict-table driver can confirm dispatch fired.
    if stage == "post_unload" and fix in ("post_gen", "all"):
        _logger.info(
            "H1 fix dispatched via _unload_with_cuda_hygiene (fix=%s)", fix
        )

    # H2 (rthook re-injection) — fires on pre_load.
    if stage == "pre_load" and fix in ("rthook", "all"):
        _apply_h2_rthook_reinject()

    # H3 (dynamo + triton driver reset) — fires on pre_load.
    if stage == "pre_load" and fix in ("triton", "all"):
        _apply_h3_dynamo_triton_reset()


def _apply_h2_rthook_reinject() -> None:
    """H2 helper: re-invoke the rthook's CUDA/triton path injection on reload.

    The body is gated on ``sys.frozen`` inside ``reinject_runtime_paths``
    itself, so a non-frozen (dev-tree pytest) call is a silent no-op that
    returns a status dict with all-False slots. We still call it so unit
    tests can assert the dispatch fired.

    Try/except absorbs ImportError if ``build_tools.hooks.rthook_torch`` is
    not on ``sys.path`` (the dev-tree case before ``tests/conftest.py``
    adjusts the path, or a partial-install scenario).
    """
    try:
        from build_tools.hooks.rthook_torch import reinject_runtime_paths
    except (ImportError, AttributeError) as exc:
        _logger.warning(
            "H2 rthook reinject skipped: %s: %s",
            type(exc).__name__,
            exc,
        )
        return
    try:
        status = reinject_runtime_paths()
        _logger.info("H2 rthook reinject status: %s", status)
    except Exception as exc:  # noqa: BLE001 — NFR7: don't brick the app
        _logger.warning(
            "H2 rthook reinject raised (%s: %s); continuing with eager-mode fallback",
            type(exc).__name__,
            exc,
        )


def _apply_h3_dynamo_triton_reset() -> None:
    """H3 helper: reset torch._dynamo + clear triton's per-process driver cache.

    Both operations are wrapped in ``try/except (AttributeError, ImportError,
    Exception)`` because the torch + triton API surface evolves across pin
    versions. A missing attribute or import is silently skipped; the verdict-
    table driver's INFO breadcrumb confirms whether the body actually ran.
    """
    # torch._dynamo.reset() — clears per-process compile state.
    # F3 fix: gate success log on did_run so the log only fires when the
    # call actually completed. Use `except Exception` (single class, not a
    # redundant tuple with AttributeError).
    dynamo_ran = False
    try:
        import torch
        try:
            torch._dynamo.reset()
            dynamo_ran = True
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "H3 dynamo reset: torch._dynamo.reset() raised (%s: %s); skipping",
                type(exc).__name__,
                exc,
            )
    except ImportError as exc:
        _logger.warning(
            "H3 dynamo reset: torch import failed (%s); skipping",
            exc,
        )
    if dynamo_ran:
        _logger.info("H3 dynamo reset: torch._dynamo.reset() succeeded")

    # triton.runtime.driver.active = None — clears per-process driver cache.
    # F4 fix: probe before+after so we can distinguish "cleared" from
    # "attribute is a property/descriptor and our assignment was a no-op".
    triton_before: Any = "<unread>"
    triton_after: Any = "<unread>"
    triton_assign_ran = False
    try:
        import triton
        try:
            triton_before = getattr(triton.runtime.driver, "active", "<missing>")
            triton.runtime.driver.active = None
            triton_assign_ran = True
            triton_after = getattr(triton.runtime.driver, "active", "<missing>")
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "H3 triton reset: clearing driver.active raised (%s: %s); skipping",
                type(exc).__name__,
                exc,
            )
    except ImportError as exc:
        _logger.warning(
            "H3 triton reset: triton import failed (%s); skipping",
            exc,
        )
    if triton_assign_ran:
        if repr(triton_before) == repr(triton_after):
            _logger.warning(
                "H3 triton reset: driver.active assignment did not change value "
                "(before=%r, after=%r); fix is a no-op on this triton pin",
                triton_before, triton_after,
            )
        else:
            _logger.info(
                "H3 triton reset: cleared triton.runtime.driver.active "
                "(before=%r, after=%r)",
                triton_before, triton_after,
            )
