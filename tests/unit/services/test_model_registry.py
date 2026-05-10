"""Story 18.3 — unit tests for ModelRegistry's precision precedence resolver.

Covers the four-branch precedence rule per AC #4:

  1. ``app_settings is None`` → falls back to legacy ``dtype: str`` parameter
     (default "bfloat16" → torch.bfloat16). source = "legacy_constructor_arg".
  2. ``app_settings.tts_precision is None`` → same fallback to legacy
     ``dtype`` parameter. source = "legacy_constructor_arg".
  3. ``app_settings.tts_precision == "fp32"`` → resolver wins;
     self.dtype == torch.float32. source = "app_settings_override".
  4. ``app_settings.tts_precision == "auto"`` + Ampere+ probe (monkeypatched)
     → self.dtype == torch.bfloat16. source = "app_settings_auto_ampere".
  4b. ``app_settings.tts_precision == "auto"`` + cuda-unavailable probe
     → self.dtype == torch.float32. source = "app_settings_auto_fallback".

Plus the AC #6 telemetry assertions: a single ``tts_precision_resolved``
metric record per ModelRegistry construction with the correct
``source`` / ``dtype`` / ``device_capability`` tags.

Per `memory/code_review_regression_test_exact_class.md`, the highest-risk
regression class for this story is "precedence-rule drift in
ModelRegistry.__init__" — the test must exercise each of the four
``source`` values directly, not assume "the obvious branches are obvious."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pytest
import torch

from myvoice.observability import metrics


# --------------------------------------------------------------------------- #
# Test infrastructure
# --------------------------------------------------------------------------- #


@dataclass
class _MetricCapture:
    name: str
    value: object
    tags: dict = field(default_factory=dict)


@pytest.fixture
def metric_records():
    captured: List[metrics.MetricRecord] = []

    def listener(record: metrics.MetricRecord) -> None:
        captured.append(record)

    unsub = metrics.add_listener(listener)
    try:
        yield captured
    finally:
        unsub()


def _make_registry(*, app_settings=None, dtype="bfloat16", device="cpu"):
    """Construct a ModelRegistry without loading any model.

    The constructor is the surface under test for Story 18.3 — it does
    NOT call ``ensure_model_loaded`` (which would require GPU + qwen-tts).
    """
    from myvoice.services.model_registry import ModelRegistry
    return ModelRegistry(
        device=device,
        dtype=dtype,
        app_settings=app_settings,
    )


# --------------------------------------------------------------------------- #
# AC #4 branch 1 — app_settings is None → legacy_constructor_arg
# --------------------------------------------------------------------------- #


def test_app_settings_none_falls_back_to_legacy_dtype_param(monkeypatch, metric_records):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    registry = _make_registry(app_settings=None, dtype="bfloat16", device="cpu")

    assert registry.dtype == torch.bfloat16
    rec = _single_record(metric_records, "tts_precision_resolved")
    assert rec.tags.get("source") == "legacy_constructor_arg"
    assert rec.tags.get("dtype") == "bfloat16"
    assert rec.value == 1.0


def test_app_settings_none_with_float32_dtype_legacy_path(monkeypatch, metric_records):
    """Legacy path with explicit dtype="float32" — used by unit tests that
    construct ``ModelRegistry`` without an AppSettings object."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    registry = _make_registry(app_settings=None, dtype="float32", device="cpu")

    assert registry.dtype == torch.float32
    rec = _single_record(metric_records, "tts_precision_resolved")
    assert rec.tags.get("source") == "legacy_constructor_arg"
    assert rec.tags.get("dtype") == "float32"
    assert rec.value == 0.0


# --------------------------------------------------------------------------- #
# AC #4 branch 2 — app_settings.tts_precision is None → legacy fallback
# --------------------------------------------------------------------------- #


class _StubAppSettings:
    """Minimal stub mirroring AppSettings for precedence testing.

    The ModelRegistry only reads ``tts_precision`` via getattr; we don't
    need the full AppSettings construction surface here.
    """
    def __init__(self, tts_precision=None):
        self.tts_precision = tts_precision


def test_app_settings_with_tts_precision_none_falls_back_to_legacy(monkeypatch, metric_records):
    """If AppSettings is supplied but its ``tts_precision`` is None, the
    precedence rule yields the legacy path. This branch handles the
    edge case of AppSettings missing the new field (e.g., after a
    settings migration)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    settings = _StubAppSettings(tts_precision=None)
    registry = _make_registry(app_settings=settings, dtype="bfloat16", device="cpu")

    assert registry.dtype == torch.bfloat16
    rec = _single_record(metric_records, "tts_precision_resolved")
    assert rec.tags.get("source") == "legacy_constructor_arg"


# --------------------------------------------------------------------------- #
# AC #4 branch 3 — explicit fp32 / bf16 override → app_settings_override
# --------------------------------------------------------------------------- #


def test_app_settings_fp32_override_wins_over_legacy_dtype(monkeypatch, metric_records):
    """User-explicit fp32 override takes precedence over the legacy
    ``dtype="bfloat16"`` parameter. NFR7 fallback path."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (12, 0))

    settings = _StubAppSettings(tts_precision="fp32")
    registry = _make_registry(app_settings=settings, dtype="bfloat16", device="cpu")

    assert registry.dtype == torch.float32
    rec = _single_record(metric_records, "tts_precision_resolved")
    assert rec.tags.get("source") == "app_settings_override"
    assert rec.tags.get("dtype") == "float32"
    assert rec.value == 0.0


def test_app_settings_bf16_override_engages_on_cpu(monkeypatch, metric_records):
    """User-explicit bf16 override engages even on CPU — the user has
    explicitly opted in to the slowdown."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    settings = _StubAppSettings(tts_precision="bf16")
    registry = _make_registry(app_settings=settings, dtype="float32", device="cpu")

    assert registry.dtype == torch.bfloat16
    rec = _single_record(metric_records, "tts_precision_resolved")
    assert rec.tags.get("source") == "app_settings_override"
    assert rec.tags.get("dtype") == "bfloat16"
    assert rec.value == 1.0


# --------------------------------------------------------------------------- #
# AC #4 branch 4 — auto + Ampere+ → app_settings_auto_ampere
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("capability", [(8, 9), (10, 0), (9, 0), (12, 0)])
def test_app_settings_auto_on_ampere_resolves_to_bfloat16(
    monkeypatch, metric_records, capability
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: capability)

    settings = _StubAppSettings(tts_precision="auto")
    registry = _make_registry(app_settings=settings, dtype="bfloat16", device="cpu")

    assert registry.dtype == torch.bfloat16
    rec = _single_record(metric_records, "tts_precision_resolved")
    assert rec.tags.get("source") == "app_settings_auto_ampere"
    assert rec.tags.get("dtype") == "bfloat16"
    assert rec.tags.get("device_capability") == f"{capability[0]}.{capability[1]}"
    assert rec.value == 1.0


def test_app_settings_auto_on_ampere_logs_precision_source_at_info(
    monkeypatch, metric_records, caplog
):
    """The INFO log line must include precision_source='app_settings_auto_ampere'
    so Commander can confirm at runtime which branch engaged.

    Per `memory/code_review_regression_test_exact_class.md`, this is the
    canonical "precedence-rule drift" regression test surface — assert
    on the log message, not just the dtype."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    settings = _StubAppSettings(tts_precision="auto")
    with caplog.at_level(logging.INFO, logger="ModelRegistry"):
        _make_registry(app_settings=settings, dtype="bfloat16", device="cpu")

    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    matched = [m for m in info_messages if "ModelRegistry initialized" in m]
    assert matched, f"Expected ModelRegistry init INFO breadcrumb; got {info_messages}"
    assert "precision_source='app_settings_auto_ampere'" in matched[0], (
        f"Expected precision_source breadcrumb; got {matched[0]}"
    )


# --------------------------------------------------------------------------- #
# AC #4 branch 4b — auto + non-Ampere → app_settings_auto_fallback
# --------------------------------------------------------------------------- #


def test_app_settings_auto_on_cpu_resolves_to_float32(monkeypatch, metric_records):
    """D-9 / NFR12: auto + cuda-unavailable returns fp32 — closes the
    latent V2 bf16-on-CPU default."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    settings = _StubAppSettings(tts_precision="auto")
    registry = _make_registry(app_settings=settings, dtype="bfloat16", device="cpu")

    assert registry.dtype == torch.float32
    rec = _single_record(metric_records, "tts_precision_resolved")
    assert rec.tags.get("source") == "app_settings_auto_fallback"
    assert rec.tags.get("dtype") == "float32"
    assert rec.tags.get("device_capability") == "none"
    assert rec.value == 0.0


def test_app_settings_auto_on_pre_ampere_resolves_to_float32(monkeypatch, metric_records):
    """Pre-Ampere CUDA host (Turing 7.5) under "auto" → fp32."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 5))

    settings = _StubAppSettings(tts_precision="auto")
    registry = _make_registry(app_settings=settings, dtype="bfloat16", device="cpu")

    assert registry.dtype == torch.float32
    rec = _single_record(metric_records, "tts_precision_resolved")
    assert rec.tags.get("source") == "app_settings_auto_fallback"
    assert rec.tags.get("device_capability") == "7.5"


# --------------------------------------------------------------------------- #
# AC #6 — telemetry tag schema (mirrors Story 18.2)
# --------------------------------------------------------------------------- #


def test_telemetry_metric_emitted_exactly_once_per_construction(monkeypatch, metric_records):
    """Single startup-once event per ModelRegistry construction. NOT a
    per-chunk metric."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    _make_registry(app_settings=_StubAppSettings(tts_precision="auto"), dtype="bfloat16", device="cpu")

    matching = [r for r in metric_records if r.name == "tts_precision_resolved"]
    assert len(matching) == 1, (
        f"Expected exactly one tts_precision_resolved record per construction; "
        f"got {len(matching)}"
    )


def test_telemetry_device_capability_is_string(monkeypatch, metric_records):
    """Mirrors Story 18.2 OQ #2: ``device_capability`` is the string form,
    with "none" sentinel for CPU. Story 18.1's CSV-capture infrastructure
    stringifies tag values — the metric must be CSV-compatible."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    _make_registry(app_settings=_StubAppSettings(tts_precision="auto"), dtype="bfloat16", device="cpu")

    rec = _single_record(metric_records, "tts_precision_resolved")
    assert isinstance(rec.tags.get("device_capability"), str)
    assert rec.tags.get("device_capability") == "none"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _single_record(records, name):
    matching = [r for r in records if r.name == name]
    assert len(matching) == 1, (
        f"Expected exactly one {name!r} record; got {len(matching)} ({matching})"
    )
    return matching[0]
