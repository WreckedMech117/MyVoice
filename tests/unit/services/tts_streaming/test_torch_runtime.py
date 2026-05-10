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
from unittest.mock import patch

import pytest

import torch  # noqa: F401  — conftest already enforced DLL ordering

from myvoice.observability import metrics
from myvoice.services.tts_streaming import (
    enable_tf32_and_cudnn_benchmark,
    is_ampere_or_newer,
    resolve_tts_precision,
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
