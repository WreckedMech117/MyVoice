"""Tests for Story 18.4 AC #7 — QwenTTSService.warmup_compile_async lifecycle.

Five paths covered (mirrors the AC #7 test obligations):

  1. Cache-hit path: ``compile_cache.is_warm(key)`` → True. **Story 20.2
     changed this path**: it now runs the same priming generation as the
     cold path (so PyTorch's lazy inductor-cache reload is paid at startup
     rather than on the user's first utterance), records
     ``reason="primed_warm"`` with value=1.0, and does NOT call
     ``mark_warm``. Three rows cover it: the warm prime, a failed warm
     prime (non-fatal, ``priming_failed``), and the AC #6 env-var gate
     ``MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1`` which restores the exact
     pre-20.2 silent ``reason="cache_hit"`` early return.
  2. Cache-miss + priming-success path: ``is_warm`` → False; the priming
     surface succeeds. Assert the indicator callback fires with
     ``"Preparing TTS engine…"`` and clears with None; assert ``mark_warm``
     is called; assert telemetry records ``reason="primed_cold"`` with
     value=1.0.
  3. Cache-miss + priming-failure path: ``is_warm`` → False; the priming
     surface raises. Assert the indicator clears; assert ``mark_warm`` is
     NOT called; assert telemetry records ``reason="priming_failed"``
     with value=0.0; assert a WARNING log is emitted.
  4. Lazy-fallback gate: ``MYVOICE_DISABLE_COMPILE_WARMUP=1`` in env;
     assert ``warmup_compile_async`` returns early without calling
     ``is_warm`` or the priming surface; assert telemetry records the
     ``user_disabled`` skip.
  5. Hardware-gated skip: monkeypatch ``is_ampere_or_newer`` → False;
     assert warmup returns early; assert telemetry records
     ``reason="pre_ampere"``.

Plus two additional defensive rows: model not yet loaded (lazy fallback
per D-23 last sentence) and model_registry not wired (defensive skip).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.observability import metrics
from myvoice.services.qwen_tts_service import QwenTTSService


# -- Fixtures ---------------------------------------------------------------- #


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


@pytest.fixture
def _restore_compile_warmup_env():
    """Ensure the test never leaks MYVOICE_DISABLE_COMPILE_WARMUP."""
    snapshot = os.environ.get("MYVOICE_DISABLE_COMPILE_WARMUP")
    yield
    if snapshot is None:
        os.environ.pop("MYVOICE_DISABLE_COMPILE_WARMUP", None)
    else:
        os.environ["MYVOICE_DISABLE_COMPILE_WARMUP"] = snapshot


def _make_service(*, with_model_registry: bool = True, with_loaded_model: bool = True) -> QwenTTSService:
    """Construct a QwenTTSService with a mock ModelRegistry.

    Defaults: a wired model_registry whose `get_loaded_model()` returns a
    plain object so the warmup proceeds to the cache-key computation. Tests
    that want to exercise the no-model-registry or no-model-loaded paths
    override these flags.

    The injected AppSettings overrides ``tts_compile`` to ``"auto"`` so
    the warmup proceeds past the H1 review-fix gate (the field's
    declared default is ``"off"`` per the Story 18.4 bundled-smoke
    amendment; tests that want to exercise the "off" gate explicitly
    override ``service._app_settings`` after construction).
    """
    service = QwenTTSService(
        device="cpu",
        dtype="float32",
        app_settings=AppSettings(tts_compile="auto"),
    )
    if with_model_registry:
        # Build a fake model that survives the cache-key computation.
        fake_inner = type("FakeInner", (), {})()
        fake_inner.name_or_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        import torch
        fake_inner.dtype = torch.bfloat16
        fake_model = type("FakeModel", (), {})()
        fake_model.model = fake_inner

        registry = MagicMock(name="ModelRegistry")
        registry.get_loaded_model.return_value = fake_model if with_loaded_model else None
        service._model_registry = registry
    else:
        service._model_registry = None
    return service


def _compile_records(records: List[metrics.MetricRecord]) -> List[metrics.MetricRecord]:
    return [r for r in records if r.name == "tts_compile_warmup_priming"]


# -- Path 1: cache-hit (Story 20.2 — now the warm-priming path) ------------- #


def _patch_warm_cache(monkeypatch) -> MagicMock:
    """Common Ampere+CUDA + warm-cache monkeypatch set. Returns the mark_warm
    mock so callers can assert it was never invoked."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer",
        lambda: True,
    )
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm",
        lambda key: True,
    )
    mark_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.mark_warm",
        mark_warm_mock,
    )
    return mark_warm_mock


@pytest.fixture
def _restore_warm_priming_env():
    """Ensure the test never leaks MYVOICE_DISABLE_WARM_COMPILE_PRIMING."""
    name = "MYVOICE_DISABLE_WARM_COMPILE_PRIMING"
    snapshot = os.environ.get(name)
    yield
    if snapshot is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = snapshot


@pytest.mark.asyncio
async def test_warmup_cache_hit_primes_warm_path(
    monkeypatch, metric_records, _restore_compile_warmup_env,
    _restore_warm_priming_env,
):
    """Story 20.2 AC #1 — is_warm → True now RUNS priming.

    Telemetry reason is the distinct ``primed_warm`` (separable from
    ``primed_cold`` and ``cache_hit`` in the metric stream), and
    ``mark_warm`` is NOT called — the key is already warm and the cold
    path's mark-on-success contract is unchanged.
    """
    os.environ.pop("MYVOICE_DISABLE_WARM_COMPILE_PRIMING", None)
    mark_warm_mock = _patch_warm_cache(monkeypatch)
    service = _make_service()
    priming_calls: List[int] = []

    async def _priming_succeeds():
        priming_calls.append(1)

    monkeypatch.setattr(service, "_run_compile_priming", _priming_succeeds)
    indicator_calls = []
    service.set_preparing_voice_callback(lambda msg: indicator_calls.append(msg))

    await service.warmup_compile_async()

    assert priming_calls == [1]
    mark_warm_mock.assert_not_called()
    # Indicator fires and clears, as on the cold path.
    assert indicator_calls == ["Preparing TTS engine…", None]
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].value == 1.0
    assert rec[0].tags["reason"] == "primed_warm"


@pytest.mark.asyncio
async def test_warm_priming_failure_is_not_fatal_and_leaves_cache_warm(
    monkeypatch, metric_records, caplog, _restore_compile_warmup_env,
    _restore_warm_priming_env,
):
    """Story 20.2 AC #1 — a failed warm prime lands ``priming_failed``,
    clears the indicator, never calls ``mark_warm``, never raises."""
    os.environ.pop("MYVOICE_DISABLE_WARM_COMPILE_PRIMING", None)
    mark_warm_mock = _patch_warm_cache(monkeypatch)
    service = _make_service()

    async def _priming_raises():
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_run_compile_priming", _priming_raises)
    indicator_calls = []
    service.set_preparing_voice_callback(lambda msg: indicator_calls.append(msg))

    with caplog.at_level(logging.WARNING):
        await service.warmup_compile_async()  # must not raise

    mark_warm_mock.assert_not_called()
    assert indicator_calls == ["Preparing TTS engine…", None]
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].value == 0.0
    assert rec[0].tags["reason"] == "priming_failed"
    assert rec[0].tags["error"] == "RuntimeError"
    assert any("Warm-path compile priming failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_warm_priming_env_gate_restores_pre_story_cache_hit(
    monkeypatch, metric_records, _restore_compile_warmup_env,
    _restore_warm_priming_env,
):
    """Story 20.2 AC #6 — MYVOICE_DISABLE_WARM_COMPILE_PRIMING=1 restores the
    exact pre-20.2 behavior: no priming, silent indicator, reason=cache_hit."""
    monkeypatch.setenv("MYVOICE_DISABLE_WARM_COMPILE_PRIMING", "1")
    mark_warm_mock = _patch_warm_cache(monkeypatch)
    service = _make_service()
    priming_mock = MagicMock()
    monkeypatch.setattr(service, "_run_compile_priming", priming_mock)
    indicator_calls = []
    service.set_preparing_voice_callback(lambda msg: indicator_calls.append(msg))

    await service.warmup_compile_async()

    priming_mock.assert_not_called()
    mark_warm_mock.assert_not_called()
    # Indicator must NOT fire on the gated cache-hit (silent steady-state).
    assert indicator_calls == []
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].value == 0.0
    assert rec[0].tags["reason"] == "cache_hit"


# -- Path 2: cache-miss + priming-success ----------------------------------- #


@pytest.mark.asyncio
async def test_warmup_cache_miss_primes_and_marks_warm(
    monkeypatch, metric_records, _restore_compile_warmup_env
):
    """is_warm → False; priming succeeds; mark_warm called; indicator fires + clears."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer",
        lambda: True,
    )
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm",
        lambda key: False,
    )
    mark_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.mark_warm",
        mark_warm_mock,
    )
    service = _make_service()

    async def _priming_succeeds():
        return None

    monkeypatch.setattr(service, "_run_compile_priming", _priming_succeeds)
    indicator_calls = []
    service.set_preparing_voice_callback(lambda msg: indicator_calls.append(msg))

    await service.warmup_compile_async()

    # The indicator fires with "Preparing TTS engine…" then clears with None.
    assert indicator_calls == ["Preparing TTS engine…", None]
    mark_warm_mock.assert_called_once()
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].value == 1.0
    assert rec[0].tags["reason"] == "primed_cold"


# -- Path 3: cache-miss + priming-failure ----------------------------------- #


@pytest.mark.asyncio
async def test_warmup_cache_miss_priming_failure_leaves_cache_cold(
    monkeypatch, metric_records, _restore_compile_warmup_env, caplog
):
    """is_warm → False; priming raises; mark_warm NOT called; indicator clears; WARNING."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer",
        lambda: True,
    )
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm",
        lambda key: False,
    )
    mark_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.mark_warm",
        mark_warm_mock,
    )
    service = _make_service()

    async def _priming_raises():
        raise RuntimeError("simulated priming failure")

    monkeypatch.setattr(service, "_run_compile_priming", _priming_raises)
    indicator_calls = []
    service.set_preparing_voice_callback(lambda msg: indicator_calls.append(msg))

    with caplog.at_level(logging.WARNING, logger="QwenTTSService"):
        await service.warmup_compile_async()

    # Indicator must clear even on failure (final block).
    assert indicator_calls == ["Preparing TTS engine…", None]
    mark_warm_mock.assert_not_called()
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].value == 0.0
    assert rec[0].tags["reason"] == "priming_failed"
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("priming failed" in r.message.lower() for r in warning_records)


# -- Path 4: lazy-fallback env-var gate ------------------------------------- #


@pytest.mark.asyncio
async def test_warmup_disabled_by_env_var_short_circuits(
    monkeypatch, metric_records, _restore_compile_warmup_env
):
    """MYVOICE_DISABLE_COMPILE_WARMUP=1 → return early; no is_warm or priming."""
    monkeypatch.setenv("MYVOICE_DISABLE_COMPILE_WARMUP", "1")
    is_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm",
        is_warm_mock,
    )
    service = _make_service()
    priming_mock = MagicMock()
    monkeypatch.setattr(service, "_run_compile_priming", priming_mock)

    await service.warmup_compile_async()

    is_warm_mock.assert_not_called()
    priming_mock.assert_not_called()
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].tags["reason"] == "user_disabled"


# -- Path 5: hardware-gated skip --------------------------------------------- #


@pytest.mark.asyncio
async def test_warmup_pre_ampere_skips_priming(
    monkeypatch, metric_records, _restore_compile_warmup_env
):
    """is_ampere_or_newer → False → return early; no is_warm or priming."""
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer",
        lambda: False,
    )
    is_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm",
        is_warm_mock,
    )
    service = _make_service()
    priming_mock = MagicMock()
    monkeypatch.setattr(service, "_run_compile_priming", priming_mock)

    await service.warmup_compile_async()

    is_warm_mock.assert_not_called()
    priming_mock.assert_not_called()
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].tags["reason"] == "pre_ampere"


# -- H1 regression: tts_compile="off" gate — code-review fix ----------------- #


@pytest.mark.asyncio
async def test_warmup_tts_compile_off_skips_priming_and_gate_log(
    monkeypatch, metric_records, _restore_compile_warmup_env, caplog
):
    """Code-review H1 (Story 18.4 review pass): warmup must not run a
    priming generation when ``app_settings.tts_compile == "off"``.

    Without this gate, a first-launch on Ampere+ CUDA with the bundled-
    smoke Fix #4 default (tts_compile="off") fires _run_compile_priming
    which dispatches a real TRUE_STREAM utterance ("Hello world." in
    Ryan's voice). The audio chunks emit to the wired
    audio_chunk_ready_callback — the user hears the priming at app
    startup. AND mark_warm writes a meaningless meta.json sidecar
    (engage stayed eager so no inductor artifacts exist), poisoning the
    cache for any later opt-in to "auto"/"on".

    The gate fires BEFORE the hardware probe, BEFORE the cache-key
    computation, BEFORE the priming dispatch. Test asserts: is_warm
    never called, mark_warm never called, _run_compile_priming never
    called, indicator never fires, telemetry records reason="user_disabled"
    with value=0.0. Per `memory/code_review_regression_test_exact_class.md`,
    this row mirrors the EXACT bug class — a future regression that
    drops or reorders the gate fails this test loudly.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer",
        lambda: True,
    )
    is_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm",
        is_warm_mock,
    )
    mark_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.mark_warm",
        mark_warm_mock,
    )

    # Build a service with tts_compile explicitly "off" so the gate fires.
    service = _make_service()
    service._app_settings = AppSettings(tts_compile="off")
    priming_mock = MagicMock()
    monkeypatch.setattr(service, "_run_compile_priming", priming_mock)
    indicator_calls = []
    service.set_preparing_voice_callback(lambda msg: indicator_calls.append(msg))

    await service.warmup_compile_async()

    # The gate must short-circuit before ANY downstream call.
    is_warm_mock.assert_not_called()
    mark_warm_mock.assert_not_called()
    priming_mock.assert_not_called()
    # Indicator must NOT fire — gate exits before _emit_preparing_voice.
    assert indicator_calls == []
    # Telemetry records the skip with reason="user_disabled" (matches
    # engage_compile_optimizations' tts_compile="off" → user_disabled
    # branch; consistent vocabulary across engage and warmup).
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].value == 0.0
    assert rec[0].tags["reason"] == "user_disabled"


# -- Defensive row 1: no model loaded yet (D-23 lazy fallback) -------------- #


@pytest.mark.asyncio
async def test_warmup_no_model_loaded_defers_to_first_generation(
    monkeypatch, metric_records, _restore_compile_warmup_env
):
    """get_loaded_model() → None: skip with reason=no_model_loaded.

    The first user-facing generation triggers compile inline; this is
    the lazy fallback path architecture D-23 names explicitly."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer",
        lambda: True,
    )
    is_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm",
        is_warm_mock,
    )
    service = _make_service(with_loaded_model=False)

    await service.warmup_compile_async()

    is_warm_mock.assert_not_called()
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].tags["reason"] == "no_model_loaded"


# -- Defensive row 2: model_registry not wired ------------------------------ #


@pytest.mark.asyncio
async def test_warmup_no_model_registry_skips(
    monkeypatch, metric_records, _restore_compile_warmup_env
):
    """Story 20.2 Task 2.4 — the fifth existing fast-exit still fires.

    ``self._model_registry`` is None: skip with reason=no_model_registry,
    above (and therefore unaffected by) the warm-path priming branch."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer",
        lambda: True,
    )
    is_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm",
        is_warm_mock,
    )
    service = _make_service(with_model_registry=False)
    priming_mock = MagicMock()
    monkeypatch.setattr(service, "_run_compile_priming", priming_mock)

    await service.warmup_compile_async()

    is_warm_mock.assert_not_called()
    priming_mock.assert_not_called()
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].tags["reason"] == "no_model_registry"


# -- Story 20.2 review F7: the indicator is a single shared slot ------------- #


@pytest.mark.asyncio
async def test_warm_priming_does_not_erase_a_concurrent_precompute_indicator(
    monkeypatch, metric_records, _restore_compile_warmup_env,
    _restore_warm_priming_env,
):
    """Review F7 — the warmup's ``finally`` must not clear someone else's
    message.

    ``_preparing_voice_callback`` is a single slot shared with Story 17.2's
    lazy voice_clone_prompt precompute ("Preparing voice for streaming…").
    Both run at startup. An unconditional ``_emit_preparing_voice(None)`` in
    the warmup's finally erases the precompute's message mid-flight, leaving
    the user with no feedback while a cloned voice is still being prepared.
    """
    os.environ.pop("MYVOICE_DISABLE_WARM_COMPILE_PRIMING", None)
    _patch_warm_cache(monkeypatch)
    service = _make_service()
    indicator_calls: List[Optional[str]] = []
    service.set_preparing_voice_callback(lambda msg: indicator_calls.append(msg))

    async def _priming_takes_over_the_slot():
        # Stand in for the Story 17.2 precompute claiming the indicator while
        # the prime is in flight.
        service._emit_preparing_voice("Preparing voice for streaming…")

    monkeypatch.setattr(
        service, "_run_compile_priming", _priming_takes_over_the_slot
    )

    await service.warmup_compile_async()

    assert indicator_calls == [
        "Preparing TTS engine…",
        "Preparing voice for streaming…",
    ], (
        "the warmup cleared an indicator message it did not own; got "
        f"{indicator_calls}"
    )
    assert service._last_preparing_voice_message == "Preparing voice for streaming…"


@pytest.mark.asyncio
async def test_warm_priming_clears_its_own_indicator(
    monkeypatch, metric_records, _restore_compile_warmup_env,
    _restore_warm_priming_env,
):
    """The conditional clear must still clear when the message IS ours —
    otherwise F7's fix would strand a "Preparing TTS engine…" message."""
    os.environ.pop("MYVOICE_DISABLE_WARM_COMPILE_PRIMING", None)
    _patch_warm_cache(monkeypatch)
    service = _make_service()
    indicator_calls: List[Optional[str]] = []
    service.set_preparing_voice_callback(lambda msg: indicator_calls.append(msg))

    async def _priming_succeeds():
        return None

    monkeypatch.setattr(service, "_run_compile_priming", _priming_succeeds)

    await service.warmup_compile_async()

    assert indicator_calls == ["Preparing TTS engine…", None]
    assert service._last_preparing_voice_message is None
