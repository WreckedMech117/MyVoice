"""Story 20.3 — compile priming targets the RESIDENT model, and the cache key
can never diverge from the model that was actually primed.

Two pre-existing defects sat between Story 20.2's measured win and the user:

  * **Defect 1 — the warmup never ran.** Startup-ordering; covered by
    ``tests/unit/test_app_compile_warmup_sequencing.py``.
  * **Defect 2 — priming targeted the wrong model.** ``_run_compile_priming``
    hard-coded ``QwenModelType.CUSTOM_VOICE`` while ``warmup_compile_async``
    computed the compile-cache key from whichever model was *loaded*. On a
    cloned-voice launch (BASE resident) that (a) marked BASE's key warm for a
    compile that only ever touched CUSTOM_VOICE, and (b) once the ordering was
    fixed, would have evicted the user's 3.4 GB model to do it — turning
    Epic 20's win into a multi-second loss on the very first generation.

This file covers Defect 2, in three groups:

  1. **Request shape per resident model** (AC #2) — BASE with the active
     profile's cached ``voice_clone_prompt``, CUSTOM_VOICE as Story 18.4
     shipped, VOICE_DESIGN with a synthetic instruct, and the skip-rather-than-
     switch behaviour when the resident model cannot be primed.
  2. **The no-switch invariant** (AC #2, load-bearing) — driven through the
     REAL ``ModelRegistry``, with a non-vacuity control that proves the same
     rig *does* register a switch when the pre-20.3 hard-coded model type is
     used.
  3. **Key / priming coherence** (AC #3) — ``mark_warm`` is called only when
     the key was computed from the model priming actually exercised.

Audio suppression (AC #5) is covered end-to-end in
``test_compile_priming_audio_suppression.py``; the row here asserts only that
the flag survives on *every* resident-model request shape, including the new
BASE one, which carries the user's own cloned voice and would otherwise speak
"Hello world." in it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.models.service_enums import ModelQualityTier, ModelState, QwenModelType
from myvoice.models.voice_profile import VoiceType
from myvoice.observability import metrics
from myvoice.services.qwen_tts_service import (
    CompilePrimingSkipped,
    QwenTTSService,
)


# --------------------------------------------------------------------------- #
# Rig
# --------------------------------------------------------------------------- #


class _FakeRegistry:
    """The three attributes ``_build_compile_priming_request`` reads, plus the
    ``get_loaded_model`` the AC #3 guard re-reads after priming."""

    def __init__(
        self,
        model_type: Optional[QwenModelType],
        *,
        checkpoint_path: Optional[str] = None,
        loaded_model: Any = None,
    ) -> None:
        self.current_model_type = model_type
        self.current_checkpoint_path = checkpoint_path
        self.quality_tier = ModelQualityTier.QUALITY
        self._loaded_model = loaded_model
        self.device = "cuda"

    def get_loaded_model(self) -> Any:
        return self._loaded_model


def _make_service(registry: Any) -> QwenTTSService:
    service = QwenTTSService(
        device="cpu",
        dtype="float32",
        app_settings=AppSettings(tts_compile="auto"),
    )
    service._model_registry = registry
    return service


def _make_ref_audio(tmp_path: Path, name: str = "my_voice.wav") -> Path:
    ref = tmp_path / name
    ref.write_bytes(b"RIFF----WAVEfmt ")
    return ref


def _wire_active_cloned_profile(
    service: QwenTTSService, ref_audio: Path, *, voice_type=VoiceType.CLONED
) -> None:
    profile = MagicMock()
    profile.name = "MyClonedVoice"
    profile.voice_type = voice_type
    profile.file_path = ref_audio
    manager = MagicMock()
    manager.get_active_profile.return_value = profile
    service.set_voice_profile_manager(manager)


def _hydrate_prompt(service: QwenTTSService, ref_audio: Path, prompt: Any) -> None:
    """Populate the in-memory cache exactly as Story 17.2's hydration does."""
    cache_key = (str(ref_audio.resolve()), ModelQualityTier.QUALITY.value)
    service._cache_store(cache_key, prompt, ref_audio)


@pytest.fixture
def metric_records():
    captured: List[metrics.MetricRecord] = []
    unsub = metrics.add_listener(captured.append)
    try:
        yield captured
    finally:
        unsub()


def _compile_records(records):
    return [r for r in records if r.name == "tts_compile_warmup_priming"]


# --------------------------------------------------------------------------- #
# 1. Request shape per resident model (AC #2)
# --------------------------------------------------------------------------- #


def test_base_resident_is_primed_with_the_active_profiles_cached_prompt(tmp_path):
    """AC #2, the common case: BASE resident + a cloned voice active.

    The request must carry the *hydrated* prompt for the *active* profile —
    same model and same conditioning regime as the user's first generation.
    """
    service = _make_service(_FakeRegistry(QwenModelType.BASE))
    ref_audio = _make_ref_audio(tmp_path)
    _wire_active_cloned_profile(service, ref_audio)
    sentinel_prompt = MagicMock(name="voice_clone_prompt")
    _hydrate_prompt(service, ref_audio, sentinel_prompt)

    request = service._build_compile_priming_request()

    assert request.model_type == QwenModelType.BASE
    # List-wrapped per the qwen-tts library contract: a bare prompt item is
    # passed straight to model.generate(...) and crashes on subscripting.
    assert request.voice_clone_prompt == [sentinel_prompt]
    assert request.x_vector_only_mode is False
    assert request.suppress_audio_output is True
    assert request.text == QwenTTSService._COMPILE_PRIMING_TEXT


def test_base_resident_without_a_cached_prompt_skips_with_a_distinct_reason(
    tmp_path,
):
    """AC #2 — no cached prompt ⇒ SKIP. Never a fallback to another model."""
    service = _make_service(_FakeRegistry(QwenModelType.BASE))
    ref_audio = _make_ref_audio(tmp_path)
    _wire_active_cloned_profile(service, ref_audio)
    # Deliberately NOT hydrated.

    with pytest.raises(CompilePrimingSkipped) as excinfo:
        service._build_compile_priming_request()

    assert excinfo.value.reason == "no_priming_prompt"


@pytest.mark.parametrize(
    "scenario",
    ["no_manager", "no_active_profile", "non_cloned_active_profile", "missing_file"],
)
def test_base_resident_skips_on_every_prompt_miss_shape(tmp_path, scenario):
    """AC #2 — every way the prompt can be unavailable lands on the same skip,
    and none of them reaches for a different model."""
    service = _make_service(_FakeRegistry(QwenModelType.BASE))
    ref_audio = _make_ref_audio(tmp_path)

    if scenario == "no_manager":
        pass  # no voice profile manager wired at all
    elif scenario == "no_active_profile":
        manager = MagicMock()
        manager.get_active_profile.return_value = None
        service.set_voice_profile_manager(manager)
    elif scenario == "non_cloned_active_profile":
        _wire_active_cloned_profile(
            service, ref_audio, voice_type=VoiceType.BUNDLED
        )
        _hydrate_prompt(service, ref_audio, MagicMock())
    else:  # missing_file
        _wire_active_cloned_profile(service, ref_audio)
        _hydrate_prompt(service, ref_audio, MagicMock())
        ref_audio.unlink()

    with pytest.raises(CompilePrimingSkipped) as excinfo:
        service._build_compile_priming_request()
    assert excinfo.value.reason == "no_priming_prompt"


def test_base_resident_without_a_prompt_dispatches_nothing(tmp_path):
    """AC #2 — the skip must abort *before* any dispatch, so no generation
    (and therefore no ``ensure_model_loaded``) can happen."""
    service = _make_service(_FakeRegistry(QwenModelType.BASE))
    _wire_active_cloned_profile(service, _make_ref_audio(tmp_path))
    dispatched: List[Any] = []

    async def _capture(request, mode):
        dispatched.append(request)
        return MagicMock(success=True)

    service._dispatch_by_streaming_mode = _capture

    with pytest.raises(CompilePrimingSkipped):
        asyncio.run(service._run_compile_priming())

    assert dispatched == []


def test_custom_voice_resident_preserves_the_story_18_4_request():
    """AC #2 — CUSTOM_VOICE keeps today's canonical-default-speaker shape."""
    service = _make_service(_FakeRegistry(QwenModelType.CUSTOM_VOICE))

    request = service._build_compile_priming_request()

    assert request.model_type == QwenModelType.CUSTOM_VOICE
    assert request.speaker == "Ryan"
    assert request.voice_clone_prompt is None
    assert request.suppress_audio_output is True


def test_voice_design_resident_is_primed_with_a_synthetic_instruct():
    """AC #2 — VOICE_DESIGN is primed, not skipped (see Dev Notes).

    Both ``instruct`` and ``voice_description`` must be set: TRUE_STREAM reads
    ``request.instruct`` while the BATCH fallback reads
    ``request.voice_description``, so setting only one leaves the fallback
    priming with an empty description.
    """
    service = _make_service(_FakeRegistry(QwenModelType.VOICE_DESIGN))

    request = service._build_compile_priming_request()

    assert request.model_type == QwenModelType.VOICE_DESIGN
    assert request.instruct == QwenTTSService._COMPILE_PRIMING_VOICE_DESCRIPTION
    assert (
        request.voice_description
        == QwenTTSService._COMPILE_PRIMING_VOICE_DESCRIPTION
    )
    assert request.suppress_audio_output is True


def test_no_resident_model_skips_rather_than_guessing():
    """AC #2 — an unknown resident type is never guessed at."""
    service = _make_service(_FakeRegistry(None))

    with pytest.raises(CompilePrimingSkipped) as excinfo:
        service._build_compile_priming_request()
    assert excinfo.value.reason == "no_model_loaded"


def test_no_registry_skips():
    service = _make_service(None)

    with pytest.raises(CompilePrimingSkipped) as excinfo:
        service._build_compile_priming_request()
    assert excinfo.value.reason == "no_model_registry"


@pytest.mark.parametrize(
    "resident",
    [QwenModelType.CUSTOM_VOICE, QwenModelType.VOICE_DESIGN, QwenModelType.BASE],
)
def test_every_resident_request_shape_is_suppressed(tmp_path, resident):
    """AC #5 — suppression survives on every path, BASE included.

    The BASE path is the one that makes this non-cosmetic: it carries the
    user's own cloned-voice prompt, so an unsuppressed prime would say
    "Hello world." in the user's own voice through their speakers and their
    virtual microphone.
    """
    service = _make_service(_FakeRegistry(resident))
    if resident == QwenModelType.BASE:
        ref_audio = _make_ref_audio(tmp_path)
        _wire_active_cloned_profile(service, ref_audio)
        _hydrate_prompt(service, ref_audio, MagicMock())

    request = service._build_compile_priming_request()

    assert request.suppress_audio_output is True
    assert service._is_suppressed(request) is True
    assert service._audio_chunk_sink(request) is None


def test_priming_carries_the_resident_checkpoint_path_verbatim():
    """AC #2 — a fine-tuned resident model must not be reloaded.

    ``ModelRegistry``'s already-loaded fast path requires the checkpoint to
    match by string equality as well as the model type. Dropping it, or
    round-tripping it through ``Path`` (which rewrites separators on Windows),
    makes ``same_checkpoint`` False and triggers unload + reload — the exact
    cost this story exists to remove.
    """
    raw = "I:/models/finetuned/my_checkpoint"
    service = _make_service(
        _FakeRegistry(QwenModelType.CUSTOM_VOICE, checkpoint_path=raw)
    )

    request = service._build_compile_priming_request()

    assert request.checkpoint_path == raw
    assert str(request.checkpoint_path) == raw, (
        "the dispatch chain passes str(request.checkpoint_path) to the "
        "registry, which compares it by equality against the raw string it "
        "stored; any normalisation here defeats the already-loaded fast path"
    )


# --------------------------------------------------------------------------- #
# 2. The no-switch invariant, through the REAL ModelRegistry (AC #2)
# --------------------------------------------------------------------------- #


def _real_registry_with_resident(model_type: QwenModelType):
    """A real ``ModelRegistry`` in the state the app is in after preload, with
    counting spies over the two operations that move the model."""
    from myvoice.services.model_registry import ModelRegistry

    registry = ModelRegistry(device="cpu")
    fake_model = MagicMock(name="resident-model-instance")
    registry._models[model_type].state = ModelState.READY
    registry._models[model_type].model_instance = fake_model
    registry._current_model_type = model_type
    registry._current_checkpoint_path = None

    ops: List[str] = []

    async def _spy_load(mt, checkpoint_path=None, tier_override=None):
        ops.append(f"load:{mt}")
        return True, None

    async def _spy_unload(mt):
        ops.append(f"unload:{mt}")
        registry._current_model_type = None
        return True

    registry._load_model = _spy_load
    registry._unload_model = _spy_unload
    return registry, ops, fake_model


def test_priming_never_loads_or_unloads_the_resident_model():
    """AC #2, the story's central invariant.

    The priming request is fed into the REAL
    ``ModelRegistry.ensure_model_loaded`` exactly as
    ``_dispatch_by_streaming_mode`` does it, and the registry must decide
    "already loaded" — zero loads, zero unloads, ``current_model_type``
    unchanged.
    """
    registry, ops, fake_model = _real_registry_with_resident(
        QwenModelType.CUSTOM_VOICE
    )
    service = _make_service(registry)

    request = service._build_compile_priming_request()
    assert request.model_type == QwenModelType.CUSTOM_VOICE

    ok, err = asyncio.run(
        registry.ensure_model_loaded(
            request.model_type,
            checkpoint_path=(
                str(request.checkpoint_path) if request.checkpoint_path else None
            ),
        )
    )

    assert (ok, err) == (True, None)
    assert ops == [], f"priming moved the model: {ops}"
    assert registry.current_model_type == QwenModelType.CUSTOM_VOICE
    assert registry.get_loaded_model() is fake_model


def test_priming_never_moves_a_resident_base_model(tmp_path):
    """AC #2 — the same invariant for the common cloned-voice launch, which is
    the one the pre-20.3 hard-coded CUSTOM_VOICE would have broken."""
    registry, ops, fake_model = _real_registry_with_resident(QwenModelType.BASE)
    service = _make_service(registry)
    ref_audio = _make_ref_audio(tmp_path)
    _wire_active_cloned_profile(service, ref_audio)
    _hydrate_prompt(service, ref_audio, MagicMock())

    request = service._build_compile_priming_request()
    assert request.model_type == QwenModelType.BASE

    asyncio.run(registry.ensure_model_loaded(request.model_type))

    assert ops == [], f"priming moved the model: {ops}"
    assert registry.current_model_type == QwenModelType.BASE
    assert registry.get_loaded_model() is fake_model


def test_control_the_pre_20_3_hardcoded_model_type_would_have_switched():
    """Non-vacuity control for the two rows above.

    With BASE resident, the pre-20.3 hard-coded ``CUSTOM_VOICE`` priming makes
    the SAME rig register an unload + a load and leaves a different model
    resident. Without this row, "ops == []" could be a property of a registry
    spy that never fires.
    """
    registry, ops, _ = _real_registry_with_resident(QwenModelType.BASE)

    asyncio.run(registry.ensure_model_loaded(QwenModelType.CUSTOM_VOICE))

    assert ops == [
        f"unload:{QwenModelType.BASE}",
        f"load:{QwenModelType.CUSTOM_VOICE}",
    ], ops


# --------------------------------------------------------------------------- #
# 3. Key / priming coherence (AC #3)
# --------------------------------------------------------------------------- #


def _patch_cold_cache(monkeypatch) -> MagicMock:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer", lambda: True
    )
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm", lambda key: False
    )
    mark_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.mark_warm", mark_warm_mock
    )
    return mark_warm_mock


def _fake_loaded_model(model_id: str):
    import torch

    inner = type("FakeInner", (), {})()
    inner.name_or_path = model_id
    inner.dtype = torch.bfloat16
    outer = type("FakeModel", (), {})()
    outer.model = inner
    return outer


def _warmup_service(model_id: str, model_type: QwenModelType) -> QwenTTSService:
    registry = _FakeRegistry(
        model_type, loaded_model=_fake_loaded_model(model_id)
    )
    return _make_service(registry)


@pytest.mark.asyncio
async def test_mark_warm_is_not_called_when_priming_exercised_another_model(
    monkeypatch, metric_records
):
    """AC #3, the exact defect class: key computed from model A while priming
    targets model B ⇒ ``mark_warm`` NOT called.

    Priming a model other than the resident one goes through
    ``ensure_model_loaded``, which evicts and replaces the resident model — so
    the observable signature of the defect is "a different model is resident
    when priming returns". The stub reproduces exactly that.
    """
    mark_warm_mock = _patch_cold_cache(monkeypatch)
    service = _warmup_service(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", QwenModelType.BASE
    )

    async def _priming_swaps_the_model():
        service._model_registry._loaded_model = _fake_loaded_model(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
        service._last_priming_model_type = QwenModelType.CUSTOM_VOICE

    monkeypatch.setattr(service, "_run_compile_priming", _priming_swaps_the_model)

    await service.warmup_compile_async()

    mark_warm_mock.assert_not_called()
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].tags["reason"] == "key_model_mismatch"
    assert rec[0].value == 0.0


@pytest.mark.asyncio
async def test_mark_warm_is_not_called_on_model_identity_alone(
    monkeypatch, metric_records
):
    """AC #3, first signal in isolation.

    The row above records BOTH a swapped model instance and a contradicting
    ``_last_priming_model_type``, so either half of the guard could be
    carrying it. Here priming records NO target type — exactly what a
    stubbed/renamed priming surface leaves behind — and the veto must still
    come from the model IDENTITY re-read after priming. Without this row,
    deleting the model-id check leaves the suite green.
    """
    mark_warm_mock = _patch_cold_cache(monkeypatch)
    service = _warmup_service(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", QwenModelType.BASE
    )

    async def _priming_swaps_the_model_silently():
        service._model_registry._loaded_model = _fake_loaded_model(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
        # deliberately does NOT set _last_priming_model_type

    monkeypatch.setattr(
        service, "_run_compile_priming", _priming_swaps_the_model_silently
    )

    await service.warmup_compile_async()

    assert service._last_priming_model_type is None
    mark_warm_mock.assert_not_called()
    assert _compile_records(metric_records)[0].tags["reason"] == "key_model_mismatch"


@pytest.mark.asyncio
async def test_mark_warm_is_not_called_when_the_model_vanished_during_priming(
    monkeypatch, metric_records
):
    """AC #3 — nothing resident after priming is also a divergence: the key
    describes a model that is no longer there to have been compiled."""
    mark_warm_mock = _patch_cold_cache(monkeypatch)
    service = _warmup_service(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", QwenModelType.BASE
    )

    async def _priming_loses_the_model():
        service._model_registry._loaded_model = None
        service._last_priming_model_type = QwenModelType.BASE

    monkeypatch.setattr(service, "_run_compile_priming", _priming_loses_the_model)

    await service.warmup_compile_async()

    mark_warm_mock.assert_not_called()
    assert _compile_records(metric_records)[0].tags["reason"] == "key_model_mismatch"


@pytest.mark.asyncio
async def test_mark_warm_is_not_called_when_the_primed_type_contradicts_the_key(
    monkeypatch, metric_records
):
    """AC #3, second signal — the model *id* can coincide (a shared
    ``name_or_path``) while the primed model TYPE contradicts the key's."""
    mark_warm_mock = _patch_cold_cache(monkeypatch)
    service = _warmup_service("Qwen/Qwen3-TTS-12Hz-1.7B", QwenModelType.BASE)

    async def _priming_targets_another_type():
        service._last_priming_model_type = QwenModelType.VOICE_DESIGN

    monkeypatch.setattr(
        service, "_run_compile_priming", _priming_targets_another_type
    )

    await service.warmup_compile_async()

    mark_warm_mock.assert_not_called()
    assert _compile_records(metric_records)[0].tags["reason"] == "key_model_mismatch"


@pytest.mark.asyncio
async def test_mark_warm_is_called_when_the_key_and_the_priming_agree(
    monkeypatch, metric_records
):
    """Non-vacuity control for the two rows above."""
    mark_warm_mock = _patch_cold_cache(monkeypatch)
    service = _warmup_service(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", QwenModelType.BASE
    )

    async def _priming_targets_the_resident_model():
        service._last_priming_model_type = QwenModelType.BASE

    monkeypatch.setattr(
        service, "_run_compile_priming", _priming_targets_the_resident_model
    )

    await service.warmup_compile_async()

    mark_warm_mock.assert_called_once()
    assert _compile_records(metric_records)[0].tags["reason"] == "primed_cold"


@pytest.mark.asyncio
async def test_cold_path_skip_leaves_the_cache_cold_and_reports_the_reason(
    monkeypatch, metric_records, tmp_path
):
    """AC #2 + AC #3 — a BASE resident with no cached prompt skips, records
    ``no_priming_prompt``, and does NOT mark the cache warm. A cold cache that
    retries next launch is strictly better than a warm marker for a model that
    was never compiled."""
    mark_warm_mock = _patch_cold_cache(monkeypatch)
    service = _warmup_service(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", QwenModelType.BASE
    )
    _wire_active_cloned_profile(service, _make_ref_audio(tmp_path))
    dispatched: List[Any] = []

    async def _capture(request, mode):
        dispatched.append(request)
        return MagicMock(success=True)

    monkeypatch.setattr(service, "_dispatch_by_streaming_mode", _capture)

    await service.warmup_compile_async()

    mark_warm_mock.assert_not_called()
    assert dispatched == []
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].tags["reason"] == "no_priming_prompt"


@pytest.mark.asyncio
async def test_warm_path_skip_reports_the_reason_and_never_marks(
    monkeypatch, metric_records, tmp_path
):
    """AC #2 — the same skip on the warm path, where ``mark_warm`` was never
    going to be called anyway; the point is the distinct telemetry reason
    rather than a silent ``primed_warm``."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer", lambda: True
    )
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm", lambda key: True
    )
    mark_warm_mock = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.mark_warm", mark_warm_mock
    )
    monkeypatch.delenv("MYVOICE_DISABLE_WARM_COMPILE_PRIMING", raising=False)

    service = _warmup_service(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", QwenModelType.BASE
    )
    _wire_active_cloned_profile(service, _make_ref_audio(tmp_path))
    indicator: List[Optional[str]] = []
    service.set_preparing_voice_callback(indicator.append)

    await service.warmup_compile_async()

    mark_warm_mock.assert_not_called()
    rec = _compile_records(metric_records)
    assert len(rec) == 1
    assert rec[0].tags["reason"] == "no_priming_prompt"
    # The transient indicator must not be left stuck on a skip.
    assert indicator == ["Preparing TTS engine…", None]


# --------------------------------------------------------------------------- #
# The reachability claim itself (AC #1, service half)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_model_loaded_is_not_the_outcome_when_a_model_is_resident(
    monkeypatch, metric_records
):
    """AC #1 — with a model resident, the warmup must reach priming.

    ``no_model_loaded`` was the terminal outcome of EVERY shipped launch
    before Story 20.3 (Story 20.2 §6). It must now appear only when the
    registry genuinely has nothing loaded.
    """
    _patch_cold_cache(monkeypatch)
    service = _warmup_service(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", QwenModelType.CUSTOM_VOICE
    )
    reached: List[int] = []

    async def _priming():
        reached.append(1)
        service._last_priming_model_type = QwenModelType.CUSTOM_VOICE

    monkeypatch.setattr(service, "_run_compile_priming", _priming)

    await service.warmup_compile_async()

    assert reached == [1]
    reasons = [r.tags["reason"] for r in _compile_records(metric_records)]
    assert reasons == ["primed_cold"]
    assert "no_model_loaded" not in reasons
