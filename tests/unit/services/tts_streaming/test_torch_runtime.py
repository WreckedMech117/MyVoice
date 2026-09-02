"""Tests for the TF32 + cuDNN benchmark autotune helper (Story 18.2).

Mirrors the structural pattern of ``test_streaming_mode.py``:

  * Monkeypatches ``torch.cuda.is_available`` and
    ``torch.cuda.get_device_capability`` at the attribute-access level so
    the lazy-import discipline inside ``torch_runtime.py`` is honored
    without import-order gymnastics.
  * No real GPU required — the four hardware truth-table branches
    (Ampere, pre-Ampere Turing, CPU-only, idempotency second-call) are
    exercised purely via monkeypatch.

Critical fixture: ``_snapshot_and_restore_torch_backends`` captures the
three ``torch.backends.*`` flag values before each test and restores them
after. The new module mutates global torch state on the engaged branch;
without the snapshot/restore, test ordering would silently shape the
"flags unchanged" assertions in the skip-branch tests. Per
`memory/code_review_regression_test_exact_class.md`, idempotency contract
drift is the highest-risk regression class for this story — the
snapshot/restore is the load-bearing fixture for that test (Task 3.4).
"""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import patch

import pytest

import torch  # noqa: F401  — conftest already enforced DLL ordering

from myvoice.observability import metrics
from myvoice.services.tts_streaming import (
    apply_reload_compile_fix,
    codec_token_streamer,
    collect_compile_gate_diagnostic,
    compile_cache,
    enable_tf32_and_cudnn_benchmark,
    engage_compile_optimizations,
    is_ampere_or_newer,
    resolve_tts_precision,
)

# Story 20.4 AC #1 — the compile path's decode window is DERIVED from the
# streamer's committed geometry, never restated as a literal in this file.
# Restating it is exactly the drift Story 20.1 SS5.4 found in production.
_STREAMER_WINDOW = (
    codec_token_streamer.DEFAULT_CHUNK_SIZE
    + codec_token_streamer.DEFAULT_LOOKAHEAD
)


# -- Fixtures ---------------------------------------------------------------- #


@pytest.fixture
def _snapshot_and_restore_torch_backends():
    """Snapshot + restore the three target torch.backends flags.

    Story 18.2's idempotency contract is read-the-flag-values, so tests
    must be insulated from cross-test bleed of the engaged branch's
    side effects. This fixture captures the three values on entry and
    restores them on teardown — applied automatically because every test
    in this module declares it as an argument.
    """
    snapshot = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    yield snapshot
    (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    ) = snapshot


@pytest.fixture
def metric_records():
    """Capture every ``metrics.record`` call as a list of MetricRecord."""
    captured: "list[metrics.MetricRecord]" = []

    def listener(record: metrics.MetricRecord) -> None:
        captured.append(record)

    unsub = metrics.add_listener(listener)
    try:
        yield captured
    finally:
        unsub()


# -- AC #1 — Ampere+ engaged branch ----------------------------------------- #


def test_is_ampere_or_newer_returns_true_when_ampere_cuda_available(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    assert is_ampere_or_newer() is True


def test_enable_engages_on_ampere_cuda_and_sets_all_three_flags(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    # Pre-set to known-False values so we can assert the function flipped them.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    status = enable_tf32_and_cudnn_benchmark()

    assert status == {
        "engaged": True,
        "reason": None,
        "device_capability": (8, 9),
    }
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cudnn.benchmark is True

    engaged_records = [
        r for r in metric_records if r.name == "tf32_cudnn_benchmark_enabled"
    ]
    assert len(engaged_records) == 1
    rec = engaged_records[0]
    assert rec.value == 1.0
    assert rec.tags.get("device_capability") == "8.9"
    # AC #3: engaged branch records value=1.0; no `reason` tag on engaged.
    assert "reason" not in rec.tags


def test_enable_logs_info_on_ampere_engagement(
    monkeypatch, _snapshot_and_restore_torch_backends, caplog
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (10, 0))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    with caplog.at_level(logging.INFO, logger="myvoice.services.tts_streaming.torch_runtime"):
        enable_tf32_and_cudnn_benchmark()

    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    matched = [m for m in info_messages if "TF32 + cuDNN benchmark enabled" in m]
    assert matched, (
        f"Expected single INFO breadcrumb on engagement; got {info_messages}"
    )
    assert "10.0" in matched[0]


# -- AC #2 — Pre-Ampere CUDA host (Turing 7.5) ------------------------------ #


def test_is_ampere_or_newer_returns_false_for_turing(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 5))

    assert is_ampere_or_newer() is False


def test_enable_skips_on_pre_ampere_and_does_not_mutate_flags(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 5))
    pre_call_flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )

    status = enable_tf32_and_cudnn_benchmark()

    assert status == {
        "engaged": False,
        "reason": "pre_ampere",
        "device_capability": (7, 5),
    }
    # AC #2: no flag mutation. Capture vs initial — the contract is "no
    # mutation," NOT "specific value preserved."
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    ) == pre_call_flags

    skip_records = [
        r for r in metric_records if r.name == "tf32_cudnn_benchmark_enabled"
    ]
    assert len(skip_records) == 1
    rec = skip_records[0]
    assert rec.value == 0.0
    assert rec.tags.get("reason") == "pre_ampere"
    assert rec.tags.get("device_capability") == "7.5"


# -- AC #2 — CPU-only (cuda unavailable) ------------------------------------ #


def test_is_ampere_or_newer_returns_false_when_cuda_unavailable(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert is_ampere_or_newer() is False


def test_enable_skips_on_cpu_only_and_does_not_call_get_device_capability(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records
):
    """Defensive ordering: CPU-only path MUST short-circuit before
    ``get_device_capability``. On a CPU-only system the device may not
    even exist; the call may error or warn. The test patches
    ``get_device_capability`` to raise so any accidental invocation
    surfaces immediately."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    def _explode(*_a, **_k):
        raise RuntimeError(
            "get_device_capability must NOT be called on cuda-unavailable host"
        )

    monkeypatch.setattr("torch.cuda.get_device_capability", _explode)
    pre_call_flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )

    status = enable_tf32_and_cudnn_benchmark()

    assert status == {
        "engaged": False,
        "reason": "cuda_unavailable",
        "device_capability": None,
    }
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    ) == pre_call_flags

    skip_records = [
        r for r in metric_records if r.name == "tf32_cudnn_benchmark_enabled"
    ]
    assert len(skip_records) == 1
    rec = skip_records[0]
    assert rec.value == 0.0
    assert rec.tags.get("reason") == "cuda_unavailable"
    # AC #3 + Story 18.2 OQ #2: include "none" sentinel for schema uniformity.
    assert rec.tags.get("device_capability") == "none"


# -- Idempotency: second call is a no-op (Task 3.4 — load-bearing) ---------- #


def test_enable_is_idempotent_second_call_does_not_re_emit(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records, caplog
):
    """The most-easily-broken AC. Per
    `memory/code_review_regression_test_exact_class.md`, the regression
    test must mirror the EXACT bug class — here, "second call re-emits
    INFO log AND/OR a new metric record."
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    # Pre-set the three flags to True (simulating a prior call's effect).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    with caplog.at_level(logging.DEBUG, logger="myvoice.services.tts_streaming.torch_runtime"):
        status = enable_tf32_and_cudnn_benchmark()

    assert status == {
        "engaged": True,
        "reason": None,
        "device_capability": (8, 9),
    }
    # No NEW metric record (the flags were True on entry).
    engaged_records = [
        r for r in metric_records if r.name == "tf32_cudnn_benchmark_enabled"
    ]
    assert engaged_records == [], (
        "Idempotency violated: second call MUST NOT re-emit the metric"
    )

    # No INFO log on the second call — only DEBUG.
    info_messages = [
        r.message for r in caplog.records if r.levelno == logging.INFO
    ]
    info_matched = [m for m in info_messages if "TF32 + cuDNN benchmark enabled" in m]
    assert info_matched == [], (
        f"Idempotency violated: second call MUST NOT re-emit INFO breadcrumb; "
        f"got {info_messages}"
    )
    debug_messages = [
        r.message for r in caplog.records if r.levelno == logging.DEBUG
    ]
    assert any("already enabled" in m for m in debug_messages), (
        f"Expected DEBUG no-op breadcrumb on idempotent call; got {debug_messages}"
    )


def test_enable_called_twice_back_to_back_emits_metric_only_once(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records
):
    """End-to-end idempotency: two calls in the same process, starting
    from a clean (False) state. First call sets + emits; second call
    short-circuits.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 6))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    enable_tf32_and_cudnn_benchmark()
    enable_tf32_and_cudnn_benchmark()

    engaged_records = [
        r for r in metric_records if r.name == "tf32_cudnn_benchmark_enabled"
    ]
    assert len(engaged_records) == 1, (
        f"Idempotency violated: two back-to-back calls must emit metric "
        f"exactly once; got {len(engaged_records)} record(s)"
    )


# -- Telemetry tag schema (Task 3.5 — string-formatted device_capability) --- #


@pytest.mark.parametrize(
    "is_cuda,capability,expected_value,expected_reason,expected_cap",
    [
        (True, (8, 9), 1.0, None, "8.9"),
        (True, (10, 0), 1.0, None, "10.0"),  # Blackwell
        (True, (9, 0), 1.0, None, "9.0"),    # Hopper
        (True, (7, 5), 0.0, "pre_ampere", "7.5"),
        (False, None, 0.0, "cuda_unavailable", "none"),
    ],
)
def test_metric_tag_schema_is_string_device_capability(
    monkeypatch,
    _snapshot_and_restore_torch_backends,
    metric_records,
    is_cuda,
    capability,
    expected_value,
    expected_reason,
    expected_cap,
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: is_cuda)
    if capability is not None:
        monkeypatch.setattr(
            "torch.cuda.get_device_capability", lambda *a, **k: capability
        )
    # Pre-set flags to False so engaged path actually emits (not idempotent no-op).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    enable_tf32_and_cudnn_benchmark()

    matching = [
        r for r in metric_records if r.name == "tf32_cudnn_benchmark_enabled"
    ]
    assert len(matching) == 1
    rec = matching[0]
    assert rec.value == expected_value
    assert rec.tags.get("device_capability") == expected_cap
    assert isinstance(rec.tags.get("device_capability"), str), (
        "device_capability tag MUST be a string (Story 18.1 CSV-capture "
        "stringifies tag values; the metric must be CSV-compatible)"
    )
    if expected_reason is None:
        assert "reason" not in rec.tags
    else:
        assert rec.tags.get("reason") == expected_reason


# -- AC #6 / D-19 — listener-exception isolation ---------------------------- #


def test_enable_completes_when_a_metrics_listener_raises(
    monkeypatch, _snapshot_and_restore_torch_backends, caplog
):
    """If a registered metrics listener raises, the function MUST still
    successfully set the flags. Application startup MUST NOT abort
    because telemetry has a buggy consumer.

    Mirrors the metrics module's own AC #9 listener-exception isolation
    (metrics.py:144-region) — Story 18.2 inherits the property by going
    through the same `metrics.record` API, but the test pins the
    contract for the new metric explicitly so a future refactor that
    bypasses `metrics.record` (e.g., direct logging.warning) can be
    caught.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    def _bad_listener(_record):
        raise RuntimeError("simulated listener bug — must not propagate")

    unsub = metrics.add_listener(_bad_listener)
    try:
        with caplog.at_level(logging.WARNING, logger="myvoice.metrics"):
            status = enable_tf32_and_cudnn_benchmark()
    finally:
        unsub()

    assert status["engaged"] is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cudnn.benchmark is True
    # The exception must have been logged (by metrics.record) but not propagated.
    assert any(
        "metrics listener raised" in r.message for r in caplog.records
    ), "Expected metrics module to log the listener exception at WARNING"


# --------------------------------------------------------------------------- #
# Story 18.3 — resolve_tts_precision branch tests
# --------------------------------------------------------------------------- #
#
# Six branches per the story's AC #8:
#   1. override == "fp32" → torch.float32 (regardless of hardware)
#   2. override == "bf16" → torch.bfloat16 (regardless of hardware)
#   3. override == "auto" + Ampere+ (cap-major shapes 8.9 / 10.0 / 9.0 / 12.0)
#      → torch.bfloat16
#   4. override == "auto" + pre-Ampere (Turing 7.5) → torch.float32
#   5. override == "auto" + cuda-unavailable → torch.float32
#   6. override is None → behaves identically to override == "auto"
#
# Side-effect-free contract: tests assert no metric records, no log records,
# no torch.backends.* mutation. Mirrors Story 16.2's effective_streaming_mode
# pure-decision test discipline.


def test_resolve_tts_precision_fp32_override_returns_float32_regardless_of_hardware(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    """User-explicit fp32 override engages on Ampere+ too (NFR7 fallback)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (12, 0))
    assert resolve_tts_precision("fp32") is torch.float32


def test_resolve_tts_precision_fp32_override_returns_float32_on_cpu(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_tts_precision("fp32") is torch.float32


def test_resolve_tts_precision_bf16_override_returns_bfloat16_regardless_of_hardware(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    """User-explicit bf16 override engages on CPU / pre-Ampere too — the
    user has explicitly opted in to the slowdown."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_tts_precision("bf16") is torch.bfloat16


def test_resolve_tts_precision_bf16_override_returns_bfloat16_on_pre_ampere(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 5))
    assert resolve_tts_precision("bf16") is torch.bfloat16


@pytest.mark.parametrize("capability", [(8, 9), (10, 0), (9, 0), (12, 0)])
def test_resolve_tts_precision_auto_returns_bfloat16_on_ampere_plus(
    monkeypatch, _snapshot_and_restore_torch_backends, capability
):
    """Mirrors Story 18.2's parametrization: 8.9 (RTX 40xx), 10.0 (Blackwell DC),
    9.0 (Hopper), 12.0 (Blackwell GeForce / RTX 5090)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: capability)
    assert resolve_tts_precision("auto") is torch.bfloat16


def test_resolve_tts_precision_auto_returns_float32_on_pre_ampere(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    """D-9 / NFR12: pre-Ampere CUDA hosts (Turing 7.5) return fp32 under
    "auto" — closing the latent V2 default that applied bf16 unconditionally."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 5))
    assert resolve_tts_precision("auto") is torch.float32


def test_resolve_tts_precision_auto_returns_float32_on_cpu_only(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    """NFR12: CPU-only hosts return fp32 under "auto"."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_tts_precision("auto") is torch.float32


def test_resolve_tts_precision_none_behaves_identically_to_auto_on_ampere(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    """The None case must mirror "auto" — both delegate to the hardware probe."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    assert resolve_tts_precision(None) is torch.bfloat16
    assert resolve_tts_precision(None) is resolve_tts_precision("auto")


def test_resolve_tts_precision_none_behaves_identically_to_auto_on_cpu(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_tts_precision(None) is torch.float32
    assert resolve_tts_precision(None) is resolve_tts_precision("auto")


def test_resolve_tts_precision_does_not_emit_metrics(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records
):
    """Side-effect-free contract: the resolver does NOT emit any metric.
    The side-effect of the chosen dtype landing on the model is at the
    call site in ModelRegistry.__init__ (which owns the
    ``tts_precision_resolved`` metric); the resolver is pure decision."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    resolve_tts_precision("auto")
    resolve_tts_precision("fp32")
    resolve_tts_precision("bf16")
    resolve_tts_precision(None)

    # No `tts_precision_resolved` records — that metric is owned by the
    # call site, not the resolver.
    assert not [r for r in metric_records if r.name == "tts_precision_resolved"], (
        "resolve_tts_precision must be side-effect-free — no metric emission"
    )


def test_resolve_tts_precision_does_not_mutate_torch_backends_flags(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    """The resolver does NOT touch ``torch.backends.*`` flags. Story 18.2's
    enable function owns the TF32 + cuDNN benchmark side effects; the
    Story 18.3 resolver is purely a dtype decision."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    # Pre-set flags to known-False values; resolver must not flip them.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    resolve_tts_precision("auto")
    resolve_tts_precision("bf16")
    resolve_tts_precision("fp32")

    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False


# --------------------------------------------------------------------------- #
# Story 18.4 — engage_compile_optimizations(): the 8-branch reason enum.
#
# Mirrors Story 18.2/18.3's parametrization + monkeypatch patterns. All
# branches are exercised purely via monkeypatch — no real GPU required.
# The model is a Mock with the talker dereference chain hand-built.
# --------------------------------------------------------------------------- #


class _PlainObj:
    """Plain attribute-only object — used in place of MagicMock for the
    talker dereference chain so ``hasattr`` returns False for unset
    attributes (MagicMock would auto-create them, defeating the
    ``_probe_compile_engaged`` False-branch assertions).
    """
    pass


def _make_fake_model_with_compile_api(
    *,
    api_succeeds: bool = True,
    api_raises: type = None,
    probe_engages: bool = True,
    dtype=None,
):
    """Build a fake that mimics the Qwen3TTSModel wrapper surface.

    The probe at ``_probe_compile_engaged`` walks ``model.model.talker.model.forward``;
    when ``probe_engages`` is True, the talker's inner forward gets a
    ``_torchdynamo_orig_callable`` attribute attached so the probe returns
    True. When False, the forward is a plain function with NO compile
    sentinel attributes — the probe returns False (the silent-no-op
    failure mode P-12 guards against).

    Uses ``_PlainObj`` for the talker chain so attribute presence is
    explicitly controlled. The wrapper itself uses MagicMock so
    ``enable_streaming_optimizations`` can be a tracked callable with
    configurable side_effect / return_value.
    """
    from unittest.mock import MagicMock

    if dtype is None:
        dtype = torch.bfloat16

    # Plain forward function — no compile-sentinel attributes unless we
    # explicitly attach them. Defines a no-op so `callable(fwd)` returns True.
    def _plain_forward(*args, **kwargs):
        return None

    if probe_engages:
        _plain_forward._torchdynamo_orig_callable = _plain_forward  # PyTorch sentinel

    # Build the dereference chain matching _probe_compile_engaged's path:
    # model.model.talker.code_predictor.model.forward (the codebook predictor's
    # inner forward — what torch.compile wraps under compile_talker=False).
    # Story 18.4 bundled-smoke regression fix: compile_talker MUST be False to
    # preserve Story 16.8's forward-hook; probe targets the code_predictor
    # (which is still compiled by default) for the "compile actually engaged"
    # signal.
    predictor_inner = _PlainObj()
    predictor_inner.forward = _plain_forward
    code_predictor = _PlainObj()
    code_predictor.model = predictor_inner
    # The talker.model.forward also exists in the real model but is NOT
    # compiled under compile_talker=False. We give it a plain forward
    # (no sentinel) so test code that walks the older path sees no
    # compile signature there.
    talker_inner = _PlainObj()
    talker_inner.forward = lambda *a, **k: None
    talker = _PlainObj()
    talker.model = talker_inner
    talker.code_predictor = code_predictor
    inner = _PlainObj()
    inner.talker = talker
    inner.dtype = dtype
    inner.name_or_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    wrapper = MagicMock(name="Qwen3TTSModel")
    wrapper.model = inner

    if api_raises is not None:
        wrapper.enable_streaming_optimizations.side_effect = api_raises("simulated compile failure")
    else:
        wrapper.enable_streaming_optimizations.return_value = wrapper if api_succeeds else None
    return wrapper


@pytest.fixture
def _isolated_inductor_env(monkeypatch, tmp_path):
    """Isolate set_torchinductor_cache_dir and cache_root for the test process.

    Force a non-Windows cache root under tmp_path and snapshot/restore
    TORCHINDUCTOR_CACHE_DIR so engage_compile_optimizations() does not
    pollute test isolation across rows.
    """
    monkeypatch.setattr(compile_cache, "sys", _make_inductor_sys_shim("linux"))
    monkeypatch.setattr(
        "pathlib.Path.home",
        classmethod(lambda cls: tmp_path),
    )
    snapshot = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    yield
    if snapshot is None:
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
    else:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = snapshot


def _make_inductor_sys_shim(platform: str):
    class _Shim:
        pass

    s = _Shim()
    s.platform = platform
    return s


# Import os here (used by the isolated inductor env fixture above).
import os  # noqa: E402


def test_engage_compile_cuda_unavailable_returns_eager_fallback(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records
):
    """Reason `cuda_unavailable` — CPU-only host skips compile entirely."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    model = _make_fake_model_with_compile_api()

    result = engage_compile_optimizations(model)

    assert result["engaged"] is False
    assert result["reason"] == "cuda_unavailable"
    model.enable_streaming_optimizations.assert_not_called()
    compile_records = [r for r in metric_records if r.name == "tts_compile_engaged"]
    assert len(compile_records) == 1
    assert compile_records[0].value == 0.0
    assert compile_records[0].tags["reason"] == "cuda_unavailable"


def test_engage_compile_pre_ampere_returns_eager_fallback(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records
):
    """Reason `pre_ampere` — Turing/Pascal CUDA host skips compile entirely."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 5))  # Turing
    model = _make_fake_model_with_compile_api()

    result = engage_compile_optimizations(model)

    assert result["engaged"] is False
    assert result["reason"] == "pre_ampere"
    model.enable_streaming_optimizations.assert_not_called()
    compile_records = [r for r in metric_records if r.name == "tts_compile_engaged"]
    assert len(compile_records) == 1
    assert compile_records[0].value == 0.0
    assert compile_records[0].tags["reason"] == "pre_ampere"


def test_engage_compile_user_disabled_returns_eager_fallback(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records
):
    """Reason `user_disabled` — tts_compile='off' forces eager even on Ampere+."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    model = _make_fake_model_with_compile_api()

    class _Settings:
        tts_compile = "off"

    result = engage_compile_optimizations(model, app_settings=_Settings())

    assert result["engaged"] is False
    assert result["reason"] == "user_disabled"
    model.enable_streaming_optimizations.assert_not_called()
    compile_records = [r for r in metric_records if r.name == "tts_compile_engaged"]
    assert len(compile_records) == 1
    assert compile_records[0].tags["reason"] == "user_disabled"


def test_engage_compile_pre_ampere_user_explicit_warns_and_returns_eager(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records, caplog
):
    """Reason `pre_ampere_user_explicit` — tts_compile='on' + Turing emits WARNING."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 5))  # Turing
    model = _make_fake_model_with_compile_api()

    class _Settings:
        tts_compile = "on"

    with caplog.at_level(logging.WARNING, logger="myvoice.services.tts_streaming.torch_runtime"):
        result = engage_compile_optimizations(model, app_settings=_Settings())

    assert result["engaged"] is False
    assert result["reason"] == "pre_ampere_user_explicit"
    model.enable_streaming_optimizations.assert_not_called()
    compile_records = [r for r in metric_records if r.name == "tts_compile_engaged"]
    assert len(compile_records) == 1
    assert compile_records[0].tags["reason"] == "pre_ampere_user_explicit"
    # WARNING log must be emitted (the user's explicit opt-in deserves
    # visibility about why it didn't engage).
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "myvoice.services.tts_streaming.torch_runtime"
    ]
    assert len(warning_records) >= 1


def test_engage_compile_engaged_cold_compile_populates_cache(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records, _isolated_inductor_env
):
    """Reason `engaged_cold_compile` — Ampere+ host, cache cold, API + probe succeed."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    model = _make_fake_model_with_compile_api(api_succeeds=True, probe_engages=True)

    result = engage_compile_optimizations(model)

    assert result["engaged"] is True
    assert result["reason"] == "engaged_cold_compile"
    assert result["cache_warm"] is False
    model.enable_streaming_optimizations.assert_called_once()
    call_kwargs = model.enable_streaming_optimizations.call_args.kwargs
    assert call_kwargs["decode_window_frames"] == _STREAMER_WINDOW
    assert call_kwargs["use_compile"] is True
    assert call_kwargs["compile_mode"] == "reduce-overhead"
    # Story 18.4 bundled-smoke regression fix: compile_talker MUST be False
    # to preserve Story 16.8's forward-hook (talker compile breaks the
    # codec_id capture path; observed zero-chunks error in production smoke).
    assert call_kwargs["compile_talker"] is False
    compile_records = [r for r in metric_records if r.name == "tts_compile_engaged"]
    assert len(compile_records) == 1
    assert compile_records[0].value == 1.0
    assert compile_records[0].tags["reason"] == "engaged_cold_compile"


def test_engage_compile_engaged_warm_cache_short_circuits_probe_label(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records, _isolated_inductor_env
):
    """Reason `engaged_warm_cache` — Ampere+ host, cache warm (mark_warm called pre-test)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    # Pre-warm the cache by computing the same key engage_compile_optimizations
    # will compute and calling mark_warm. Pin hash is caller-provided per
    # the code-review M1 fix (torch_runtime no longer imports QwenTTSService
    # to read the constant); the test passes a known value through the
    # parameter and pre-warms with the same value.
    pin_hash = "test_pin_hash_18.4"

    key = compile_cache.compute_key(
        qwen_tts_pin_hash=pin_hash,
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        precision_str="bf16",
        torch_version=torch.__version__,
        decode_window_frames=_STREAMER_WINDOW,
        cuda_capability=(8, 9),
        compile_mode="reduce-overhead",
    )
    compile_cache.mark_warm(key)

    model = _make_fake_model_with_compile_api(api_succeeds=True, probe_engages=True)
    result = engage_compile_optimizations(model, qwen_tts_pin_hash=pin_hash)

    assert result["engaged"] is True
    assert result["reason"] == "engaged_warm_cache"
    assert result["cache_warm"] is True
    compile_records = [r for r in metric_records if r.name == "tts_compile_engaged"]
    assert len(compile_records) == 1
    assert compile_records[0].tags["reason"] == "engaged_warm_cache"


def test_engage_compile_compile_failed_returns_eager_fallback(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records, _isolated_inductor_env, caplog
):
    """Reason `compile_failed` — API raises RuntimeError; NFR7 eager fallback engages."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    model = _make_fake_model_with_compile_api(api_raises=RuntimeError)

    with caplog.at_level(logging.WARNING, logger="myvoice.services.tts_streaming.torch_runtime"):
        result = engage_compile_optimizations(model)

    assert result["engaged"] is False
    assert result["reason"] == "compile_failed"
    assert "error" in result
    compile_records = [r for r in metric_records if r.name == "tts_compile_engaged"]
    assert len(compile_records) == 1
    assert compile_records[0].value == 0.0
    assert compile_records[0].tags["reason"] == "compile_failed"


def test_engage_compile_d25_invariant_raises_on_explicit_window_mismatch(
    monkeypatch, _snapshot_and_restore_torch_backends, _isolated_inductor_env
):
    """Code-review H2 (Story 18.4 review pass): D-25 decode-window invariant.

    Per architecture P-11 ("loud at startup"), passing an explicit
    ``decode_window_frames`` that does NOT match
    ``streamer_chunk_size + streamer_lookahead`` MUST raise
    ``AssertionError`` — silent fallthrough under a divergent window
    produces wrong audio under CUDA Graph replay (compiled graph
    captures one shape; emitting under another diverges silently).

    The previous implementation had a tautological assertion that
    re-computed the variable from its own RHS, making the invariant
    dead code. This regression test pins the real contract — a future
    refactor that drops the hard-fail (e.g., demotes to a log.warning)
    fails here. Per `memory/code_review_regression_test_exact_class.md`,
    this row mirrors the EXACT bug class (silent fallthrough on
    invariant violation).
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    model = _make_fake_model_with_compile_api(api_succeeds=True, probe_engages=True)

    # Streamer explicitly declared as 25+5=30; caller passes 80 (the
    # fork's default). Architecture D-21 binds the value to MyVoice's
    # streamer window; any caller passing a divergent value is a bug.
    with pytest.raises(AssertionError, match="Decode-window drift"):
        engage_compile_optimizations(
            model,
            streamer_chunk_size=25,
            streamer_lookahead=5,
            decode_window_frames=80,
        )

    # Sanity: passing None (the canonical contract) does NOT raise —
    # the function derives the window from the streamer params it was given.
    result = engage_compile_optimizations(
        model,
        streamer_chunk_size=25,
        streamer_lookahead=5,
        decode_window_frames=None,
    )
    assert result["decode_window_frames"] == 30

    # Sanity: passing the matching explicit value also does NOT raise.
    result = engage_compile_optimizations(
        model,
        streamer_chunk_size=25,
        streamer_lookahead=5,
        decode_window_frames=30,
    )
    assert result["decode_window_frames"] == 30


def test_engage_compile_resolves_window_from_live_streamer_constants(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records,
    _isolated_inductor_env,
):
    """Story 20.4 AC #1 — the compile path reads the STREAMER's real geometry.

    This is the regression row for the Story 20.1 SS5.4 trap, and it
    mirrors that bug's exact class: the function used to declare
    ``streamer_chunk_size: int = 25, streamer_lookahead: int = 5`` as
    hard-coded defaults while the sole production call site passed
    neither, so ``decode_window_frames`` resolved to 30 no matter what
    the streamer emitted. Retuning ``DEFAULT_CHUNK_SIZE`` alone would
    then have told the compile path 30 while the streamer emitted 15 —
    silently violating the very D-25 invariant the assertion exists to
    protect, and computing a compile-cache key for a window nothing uses.

    Two arms:
      1. With no geometry passed, the resolved window equals the LIVE
         module constants (so the committed retune is reflected).
      2. Monkeypatching the module constants moves the resolved window
         with them — i.e. the value is read at call time, not frozen into
         a default at import time. A regression to hard-coded defaults
         fails this arm.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    model = _make_fake_model_with_compile_api(api_succeeds=True, probe_engages=True)
    result = engage_compile_optimizations(model)
    assert result["decode_window_frames"] == _STREAMER_WINDOW
    assert (
        model.enable_streaming_optimizations.call_args.kwargs[
            "decode_window_frames"
        ]
        == _STREAMER_WINDOW
    )

    # Arm 2 — a retune of the streamer constants must move the compile
    # geometry with it, with no edit anywhere else.
    monkeypatch.setattr(codec_token_streamer, "DEFAULT_CHUNK_SIZE", 17)
    monkeypatch.setattr(codec_token_streamer, "DEFAULT_LOOKAHEAD", 3)
    model2 = _make_fake_model_with_compile_api(api_succeeds=True, probe_engages=True)
    result2 = engage_compile_optimizations(model2)
    assert result2["decode_window_frames"] == 20
    assert (
        model2.enable_streaming_optimizations.call_args.kwargs[
            "decode_window_frames"
        ]
        == 20
    )

    # And the D-25 hard-fail still fires against the RETUNED geometry, so
    # the invariant is live rather than decorative.
    with pytest.raises(AssertionError, match="Decode-window drift"):
        engage_compile_optimizations(model2, decode_window_frames=30)


def test_engage_compile_probe_failed_returns_eager_fallback(
    monkeypatch, _snapshot_and_restore_torch_backends, metric_records, _isolated_inductor_env, caplog
):
    """Reason `probe_failed` — API succeeds but P-12 probe says model state unchanged."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    # API succeeds; probe fails (no _torchdynamo_orig_callable on the
    # talker forward).
    model = _make_fake_model_with_compile_api(api_succeeds=True, probe_engages=False)

    with caplog.at_level(logging.WARNING, logger="myvoice.services.tts_streaming.torch_runtime"):
        result = engage_compile_optimizations(model)

    assert result["engaged"] is False
    assert result["reason"] == "probe_failed"
    # API DID get called (the failure mode is silent no-op, not exception).
    model.enable_streaming_optimizations.assert_called_once()
    compile_records = [r for r in metric_records if r.name == "tts_compile_engaged"]
    assert len(compile_records) == 1
    assert compile_records[0].tags["reason"] == "probe_failed"


# --------------------------------------------------------------------------- #
# Compile-disengage-post-generation spec — diagnostic + dispatch tests
# --------------------------------------------------------------------------- #


_EXPECTED_DIAGNOSTIC_KEYS = {
    "stage",
    "cycle_idx",
    "is_available",
    "device_count",
    "in_bad_fork",
    "initialized",
    "last_cuda_error",
    "cuda_path_present",
    "path_has_cuda_redist",
}


def test_collect_diagnostic_returns_expected_keys_and_types(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    """AC #1 — collect_compile_gate_diagnostic returns a flat dict of probes.

    Monkeypatches the CUDA probes to known values and asserts:
      * the returned dict has the nine expected keys
      * all values serialize as int/float/str/bool (metrics.record contract)
      * stage/cycle_idx round-trip verbatim
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    monkeypatch.setattr("torch.cuda._is_in_bad_fork", lambda: False)
    monkeypatch.setattr("torch.cuda.is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_PATH", "C:/test/path")

    probe = collect_compile_gate_diagnostic(stage="reload_gate", cycle_idx=7)

    assert set(probe.keys()) == _EXPECTED_DIAGNOSTIC_KEYS
    assert probe["stage"] == "reload_gate"
    assert probe["cycle_idx"] == 7
    assert probe["is_available"] is False
    assert probe["device_count"] == 0
    assert probe["cuda_path_present"] is True
    # All values must be of the four allowed types — metrics.record's
    # validator rejects anything else.
    for key, value in probe.items():
        assert isinstance(value, (int, float, str, bool)), (
            f"key {key!r} has disallowed value type {type(value).__name__}"
        )


def test_collect_diagnostic_probe_error_yields_sentinel(
    monkeypatch, _snapshot_and_restore_torch_backends
):
    """A broken probe yields a `probe_error: ...` sentinel, not an exception."""
    def boom():
        raise RuntimeError("simulated probe failure")

    monkeypatch.setattr("torch.cuda.is_available", boom)

    probe = collect_compile_gate_diagnostic(stage="post_unload", cycle_idx=0)
    assert isinstance(probe["is_available"], str)
    assert probe["is_available"].startswith("probe_error: ")


def test_apply_reload_compile_fix_off_is_noop(
    monkeypatch, _snapshot_and_restore_torch_backends, caplog
):
    """AC #4 — fix=off (or unset) is a no-op: no torch state mutation, no exception."""
    monkeypatch.setenv("MYVOICE_RELOAD_COMPILE_FIX", "off")

    import torch
    snap_benchmark = torch.backends.cudnn.benchmark

    with caplog.at_level(
        logging.DEBUG,
        logger="myvoice.services.tts_streaming.torch_runtime",
    ):
        apply_reload_compile_fix("pre_load")
        apply_reload_compile_fix("post_unload")

    # No torch state mutation observable.
    assert torch.backends.cudnn.benchmark == snap_benchmark


def test_apply_reload_compile_fix_unknown_warns_and_skips(
    monkeypatch, _snapshot_and_restore_torch_backends, caplog
):
    """AC #5 — unknown fix value warns + behaves as `off`."""
    monkeypatch.setenv("MYVOICE_RELOAD_COMPILE_FIX", "purple_monkey")

    with caplog.at_level(
        logging.WARNING,
        logger="myvoice.services.tts_streaming.torch_runtime",
    ):
        # Must not raise.
        apply_reload_compile_fix("pre_load")

    warning_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("purple_monkey" in r.getMessage() for r in warning_lines), (
        f"Expected a WARNING mentioning the unknown fix value; got: "
        f"{[r.getMessage() for r in warning_lines]}"
    )


def test_apply_reload_compile_fix_unknown_stage_returns_silently(
    monkeypatch, _snapshot_and_restore_torch_backends, caplog
):
    """Unknown stage logs a warning and returns without raising."""
    monkeypatch.setenv("MYVOICE_RELOAD_COMPILE_FIX", "post_gen")

    with caplog.at_level(
        logging.WARNING,
        logger="myvoice.services.tts_streaming.torch_runtime",
    ):
        apply_reload_compile_fix("nonsense_stage")

    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("nonsense_stage" in m for m in warning_msgs), (
        f"Expected a WARNING mentioning the unknown stage; got: {warning_msgs}"
    )


def test_apply_reload_compile_fix_rthook_non_frozen_noop(
    monkeypatch, _snapshot_and_restore_torch_backends, caplog
):
    """AC #6 — H2 rthook helper is a no-op in non-frozen pytest (sys.frozen=False)."""
    monkeypatch.setenv("MYVOICE_RELOAD_COMPILE_FIX", "rthook")

    # Capture starting CUDA_PATH so we can assert no mutation.
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    original_cuda_path = os.environ.get("CUDA_PATH")

    with caplog.at_level(
        logging.INFO,
        logger="myvoice.services.tts_streaming.torch_runtime",
    ):
        apply_reload_compile_fix("pre_load")

    # CUDA_PATH unchanged (or unset) in dev-tree.
    assert os.environ.get("CUDA_PATH") == original_cuda_path
