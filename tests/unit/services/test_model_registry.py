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

    The ModelRegistry reads ``tts_precision`` (Story 18.3) and stores
    ``self._app_settings`` so ``_load_model_sync`` can pass it to
    ``engage_compile_optimizations`` for the ``tts_compile`` gate
    (Story 18.4). Both fields are exposed here for completeness.
    """
    def __init__(self, tts_precision=None, tts_compile=None):
        self.tts_precision = tts_precision
        self.tts_compile = tts_compile


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
# Story 18.4 — compile_engaged INFO log breadcrumb + app_settings storage
# --------------------------------------------------------------------------- #


def test_app_settings_stored_on_self_for_load_model_compile_engagement(
    monkeypatch, metric_records
):
    """Story 18.4 — `self._app_settings` must be retained for _load_model_sync's
    `engage_compile_optimizations(model, app_settings=self._app_settings)`
    call. The Story 18.3 wire-up only used the constructor argument inline;
    Story 18.4 widened it to a stored field. Regression test: assert the
    stored attribute matches the passed argument."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    settings = _StubAppSettings(tts_precision="auto", tts_compile="on")
    registry = _make_registry(app_settings=settings, dtype="bfloat16", device="cpu")

    assert registry._app_settings is settings, (
        "ModelRegistry must store the app_settings argument on self._app_settings "
        "so _load_model_sync can pass it to engage_compile_optimizations."
    )


def test_modelregistry_init_info_log_includes_compile_engaged_deferred(
    monkeypatch, metric_records, caplog
):
    """Story 18.4 — the __init__ INFO log line carries `compile_engaged='deferred'`
    because the actual engagement happens during _load_model_sync (after
    from_pretrained returns; the model object doesn't exist at __init__).
    Static log parsers see the tag schema at __init__ time; the resolved
    state lands at the post-load INFO line that _load_model_sync emits.

    Per `memory/code_review_regression_test_exact_class.md`, this is the
    canonical regression test for the tag-schema-drift class — the new tag
    must always appear at __init__ even when no model is loaded."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))

    settings = _StubAppSettings(tts_precision="auto")
    with caplog.at_level(logging.INFO, logger="ModelRegistry"):
        _make_registry(app_settings=settings, dtype="bfloat16", device="cpu")

    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    matched = [m for m in info_messages if "ModelRegistry initialized" in m]
    assert matched, f"Expected ModelRegistry init INFO breadcrumb; got {info_messages}"
    assert "compile_engaged='deferred'" in matched[0], (
        f"Expected compile_engaged='deferred' breadcrumb at __init__; got {matched[0]}"
    )


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


# --------------------------------------------------------------------------- #
# compile-disengage-post-generation spec — _unload_with_cuda_hygiene tests
# --------------------------------------------------------------------------- #


import asyncio
import logging as _logging


def _track_calls(*names: str):
    """Return (recorded_calls list, factory(name) -> stub) for monkeypatching.

    Each stub records its name in ``recorded_calls`` then returns silently.
    Use to verify ordering of multiple monkeypatched callables.
    """
    recorded: List[str] = []

    def factory(name: str):
        def _stub(*a, **kw):
            recorded.append(name)
        return _stub

    return recorded, factory


def test_unload_hygiene_calls_helpers_in_order(monkeypatch, metric_records):
    """AC #8 — _unload_with_cuda_hygiene sequences the five steps in order.

    synchronize → _dynamo.reset → graphs._reset_caches → empty_cache → ipc_collect
    """
    from myvoice.models.service_enums import QwenModelType

    recorded, factory = _track_calls()
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.synchronize", factory("synchronize"))
    monkeypatch.setattr("torch._dynamo.reset", factory("dynamo_reset"))
    monkeypatch.setattr("torch.cuda.empty_cache", factory("empty_cache"))
    monkeypatch.setattr("torch.cuda.ipc_collect", factory("ipc_collect"))
    # graphs._reset_caches — set the attribute via setattr; the hygiene
    # helper hasattr-gates on it. We want to confirm it fires when present.
    import torch
    graphs = getattr(torch.cuda, "graphs", None)
    if graphs is not None:
        monkeypatch.setattr(
            torch.cuda.graphs, "_reset_caches", factory("graphs_reset_caches"), raising=False
        )

    registry = _make_registry(dtype="float32", device="cpu")
    registry._unload_with_cuda_hygiene(QwenModelType.BASE)

    # AC #8 — exact sequence, exactly-once. Verbatim assertion catches both
    # ordering regressions AND double-call regressions (the latter is the
    # exact concern of AC #10 — F7 fix tightens this from pairwise to full
    # sequence). The expected sequence accommodates platforms where
    # `torch.cuda.graphs` is missing (older torch): if absent, the
    # hygiene helper skips that step and the expected sequence drops it.
    expected = ["synchronize", "dynamo_reset"]
    if graphs is not None and hasattr(graphs, "_reset_caches"):
        expected.append("graphs_reset_caches")
    expected.extend(["empty_cache", "ipc_collect"])
    assert recorded == expected, (
        f"_unload_with_cuda_hygiene call sequence drift: expected {expected}, "
        f"got {recorded}. AC #8 (exact ordering + exactly-once) regressed."
    )


def test_unload_hygiene_individual_failure_does_not_abort(monkeypatch, caplog):
    """AC #9 — one helper raising does NOT prevent subsequent helpers from running."""
    from myvoice.models.service_enums import QwenModelType

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    called: List[str] = []

    def _synchronize_ok(*a, **kw):
        called.append("synchronize")

    def _dynamo_reset_raises(*a, **kw):
        called.append("dynamo_reset")
        raise RuntimeError("simulated CUDA bug")

    def _empty_cache_ok(*a, **kw):
        called.append("empty_cache")

    def _ipc_collect_ok(*a, **kw):
        called.append("ipc_collect")

    monkeypatch.setattr("torch.cuda.synchronize", _synchronize_ok)
    monkeypatch.setattr("torch._dynamo.reset", _dynamo_reset_raises)
    monkeypatch.setattr("torch.cuda.empty_cache", _empty_cache_ok)
    monkeypatch.setattr("torch.cuda.ipc_collect", _ipc_collect_ok)

    registry = _make_registry(dtype="float32", device="cpu")

    with caplog.at_level(_logging.WARNING, logger="ModelRegistry"):
        registry._unload_with_cuda_hygiene(QwenModelType.BASE)

    # Despite the simulated error, empty_cache + ipc_collect must still run.
    assert "empty_cache" in called, called
    assert "ipc_collect" in called, called

    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == _logging.WARNING]
    assert any("simulated CUDA bug" in m for m in warning_msgs), (
        f"Expected WARNING log recording the simulated error; got: {warning_msgs}"
    )


def test_unload_model_replaces_legacy_block_when_fix_post_gen(monkeypatch):
    """AC #10 — under fix=post_gen, _unload_with_cuda_hygiene is called and
    the legacy gc.collect()+empty_cache() block is bypassed (no double
    empty_cache).
    """
    from myvoice.models.service_enums import QwenModelType, ModelState
    from myvoice.services import model_registry as mr_module

    monkeypatch.setenv("MYVOICE_RELOAD_COMPILE_FIX", "post_gen")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    empty_cache_calls = {"count": 0}

    def _counting_empty_cache(*a, **kw):
        empty_cache_calls["count"] += 1

    monkeypatch.setattr("torch.cuda.empty_cache", _counting_empty_cache)
    monkeypatch.setattr("torch.cuda.synchronize", lambda *a, **kw: None)
    monkeypatch.setattr("torch._dynamo.reset", lambda *a, **kw: None)
    monkeypatch.setattr("torch.cuda.ipc_collect", lambda *a, **kw: None)

    registry = _make_registry(dtype="float32", device="cpu")

    # Pre-populate model state to simulate a READY model so _unload_model
    # takes the real-unload path (not the no-op short-circuit).
    registry._models[QwenModelType.BASE].state = ModelState.READY
    registry._models[QwenModelType.BASE].model_instance = object()
    registry._current_model_type = QwenModelType.BASE

    hygiene_called = {"count": 0}
    original_hygiene = registry._unload_with_cuda_hygiene

    def _tracking_hygiene(model_type):
        hygiene_called["count"] += 1
        return original_hygiene(model_type)

    registry._unload_with_cuda_hygiene = _tracking_hygiene  # type: ignore[method-assign]

    # Reset just before the action — see note on the fix=off mirror test.
    import gc as _gc
    _gc.collect()
    empty_cache_calls["count"] = 0
    asyncio.run(registry._unload_model(QwenModelType.BASE))

    assert hygiene_called["count"] == 1, (
        f"Expected _unload_with_cuda_hygiene to be called exactly once under "
        f"fix=post_gen; got {hygiene_called['count']}"
    )
    # empty_cache called exactly once via the hygiene helper (not twice).
    assert empty_cache_calls["count"] == 1, (
        f"Expected exactly one empty_cache call under fix=post_gen "
        f"(legacy block must NOT also fire); got {empty_cache_calls['count']}"
    )


def test_unload_model_uses_legacy_block_when_fix_off(monkeypatch):
    """Mirror test — under fix=off, the legacy empty_cache() path runs and
    _unload_with_cuda_hygiene is NOT called."""
    from myvoice.models.service_enums import QwenModelType, ModelState

    monkeypatch.setenv("MYVOICE_RELOAD_COMPILE_FIX", "off")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    empty_cache_calls = {"count": 0}
    monkeypatch.setattr(
        "torch.cuda.empty_cache",
        lambda *a, **kw: empty_cache_calls.__setitem__("count", empty_cache_calls["count"] + 1),
    )

    registry = _make_registry(dtype="float32", device="cpu")

    registry._models[QwenModelType.BASE].state = ModelState.READY
    registry._models[QwenModelType.BASE].model_instance = object()
    registry._current_model_type = QwenModelType.BASE

    hygiene_called = {"count": 0}
    original_hygiene = registry._unload_with_cuda_hygiene

    def _tracking_hygiene(model_type):
        hygiene_called["count"] += 1
        return original_hygiene(model_type)

    registry._unload_with_cuda_hygiene = _tracking_hygiene  # type: ignore[method-assign]

    # Reset the counter just before the action so the assertion is
    # order-independent — prior tests' registry __del__/shutdown can fire
    # empty_cache asynchronously. Force a gc cycle first to flush any
    # pending finalizers, then zero the counter.
    import gc as _gc
    _gc.collect()
    empty_cache_calls["count"] = 0
    asyncio.run(registry._unload_model(QwenModelType.BASE))

    assert hygiene_called["count"] == 0, (
        f"Expected _unload_with_cuda_hygiene NOT to be called under fix=off; "
        f"got {hygiene_called['count']}"
    )
    # Legacy block fires empty_cache once.
    assert empty_cache_calls["count"] == 1
