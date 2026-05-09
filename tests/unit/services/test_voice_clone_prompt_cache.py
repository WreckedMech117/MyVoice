"""Story 17.2 — unit tests for the lazy + persistent voice_clone_prompt cache.

Covers Tasks 1, 2, 3, 4, 5, 6 from
``_bmad-output/implementation-artifacts/17-2-cloned-voice-truestream-prompt-precompute.md``:

  - AC #1 / Task 1: four-condition gate, cache hit/miss, per-voice
    serialization, cache key shape.
  - AC #2 / Task 2: transcription priority chain (in-memory → .txt sidecar
    → Whisper), Whisper retry+backoff, status transitions, lazy-fail-safe
    when WhisperSubprocessService is None.
  - AC #3 / Task 3: persisted .pt + .pt.meta.json roundtrip, cache
    invalidation on mtime/size/pin/tier mismatch, startup hydration.
  - AC #4 / Task 5: preparing-voice indicator emission on miss / clear
    on success+failure / silent on hit.
  - AC #5 / Task 6: NFR7 graceful-degradation regression — successful
    cache HIT + downstream raise still falls back; precompute raise still
    falls back; success path asserts request.voice_clone_prompt is set.

Mocks ``_create_voice_clone_prompt_sync`` (the inner helper at line 1286)
to avoid loading the Base model. The async wrapper
``create_voice_clone_prompt_for_tier`` calls
``ensure_model_loaded`` first; we patch the wrapper directly to short-
circuit the model load entirely.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# CRITICAL: torch-before-PyQt6 DLL ordering preamble for coverage runs.
# tests/conftest.py's preamble does NOT fire when pytest-cov instruments
# this module (see memory/torch_before_coverage_dll_ordering.md). Mirror
# the preamble inline so coverage runs work on the production hardware.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _repo_root = Path(__file__).parent.parent.parent.parent
    _torch_lib = _repo_root / "python310" / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.exists():
        os.add_dll_directory(str(_torch_lib))
    for _cuda_path in (
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp"),
    ):
        if _cuda_path.exists():
            os.add_dll_directory(str(_cuda_path))

try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.models.service_enums import QwenModelType
from myvoice.models.voice_profile import (
    TranscriptionStatus,
    VoiceProfile,
    VoiceType,
)
from myvoice.services.qwen_tts_service import (
    QwenTTSRequest,
    QwenTTSResponse,
    QwenTTSService,
    VoiceClonePromptItem,
)
from myvoice.services.tts_streaming import StreamingMode


# --------------------------------------------------------------------------- #
# Test infrastructure
# --------------------------------------------------------------------------- #


def _make_service(
    app_settings: Optional[AppSettings] = None,
) -> QwenTTSService:
    """Construct a QwenTTSService with no model load — safe for unit tests.

    Mirrors test_qwen_tts_service_dispatch.py::_make_service. The constructor
    builds a ModelRegistry but does NOT load the model.
    """
    return QwenTTSService(
        device="cpu",
        dtype="float32",
        app_settings=app_settings,
    )


def _make_clone_voice_file(tmp_path: Path, name: str = "Sarira-F") -> Path:
    """Create a placeholder .wav file so ref_audio.exists() / .stat() work.

    The bytes are a minimal RIFF/WAVE header — enough that VoiceProfile
    construction won't reject the file. Tests that mock the precompute
    don't actually decode this audio.
    """
    voice_dir = tmp_path / "voices" / name
    voice_dir.mkdir(parents=True, exist_ok=True)
    wav_path = voice_dir / f"{name}.wav"
    # Minimal valid RIFF/WAVE header (44 bytes) + zero-filled body.
    # 1s of mono 22050 Hz silence at int16 = 44100 sample bytes.
    sample_rate = 22050
    duration_seconds = 4  # long enough to clear duration checks
    num_samples = sample_rate * duration_seconds
    data_size = num_samples * 2  # int16 -> 2 bytes/sample
    riff_size = 36 + data_size
    header = (
        b"RIFF"
        + riff_size.to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + (sample_rate * 2).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + data_size.to_bytes(4, "little")
    )
    wav_path.write_bytes(header + b"\x00" * data_size)
    return wav_path


def _make_synthetic_prompt() -> VoiceClonePromptItem:
    """Synthetic VoiceClonePromptItem with small CPU tensors for fast tests.

    The wrapper class is at qwen_tts_service.py:175. The library's
    LibraryVoiceClonePromptItem requires the qwen_tts package; we stub the
    wrapper because _normalize_voice_clone_prompt promotes wrapper-shape
    objects on the way back out of the cache load path.
    """
    import torch
    return VoiceClonePromptItem(
        ref_code=torch.zeros(8, dtype=torch.float32),
        ref_spk_embedding=torch.zeros(16, dtype=torch.float32),
        x_vector_only_mode=False,
        icl_mode=True,
        ref_text="hello",
    )


def _patch_normalize(service: QwenTTSService) -> Any:
    """Stub _normalize_voice_clone_prompt to return its input unchanged.

    The real method requires LibraryVoiceClonePromptItem (qwen_tts), which
    is heavy and tier-specific. Tests assert the cache-management semantics
    layer above normalization; the prompt object's exact shape doesn't matter
    as long as identity is preserved.
    """
    return patch.object(
        service,
        "_normalize_voice_clone_prompt",
        side_effect=lambda x: x,
    )


# --------------------------------------------------------------------------- #
# AC #1 / Task 1 — TestFourConditionGate + TestCacheKeyAndSerialization
# --------------------------------------------------------------------------- #


class TestFourConditionGate:
    """Gate must skip precompute on any of the four conditions failing."""

    @pytest.mark.asyncio
    async def test_streaming_false_skips_precompute(self, tmp_path, monkeypatch):
        """Voice Design Studio path: streaming=False must NOT trigger
        precompute (BATCH-force at qwen_tts_service.py:3344-3346 is
        authoritative)."""
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        with patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ) as mock_dispatch:
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=False
            )
        # The dispatch call's request must NOT have voice_clone_prompt set.
        assert mock_dispatch.call_count == 1
        request_arg = mock_dispatch.call_args.args[0]
        assert request_arg.voice_clone_prompt is None
        # No cache writes either — only writes happen post-precompute.
        assert service._voice_clone_prompts == {}

    @pytest.mark.asyncio
    async def test_cpu_resolve_skips_precompute(self, tmp_path, monkeypatch):
        """CPU-only host resolves to SENTENCE_STREAM (NFR12) and must not
        trigger TRUE_STREAM-only precompute."""
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        ref_audio = _make_clone_voice_file(tmp_path)
        with patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ) as mock_dispatch:
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        request_arg = mock_dispatch.call_args.args[0]
        assert request_arg.voice_clone_prompt is None
        assert service._voice_clone_prompts == {}

    @pytest.mark.asyncio
    async def test_x_vector_only_mode_skips_precompute(self, tmp_path, monkeypatch):
        """x-vector mode skips ICL transcription entirely; calling Whisper
        for an empty ref_text path is meaningless."""
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        with patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ) as mock_dispatch:
            await service.generate_voice_clone(
                text="hi",
                ref_audio=ref_audio,
                ref_text="",
                streaming=True,
                x_vector_only_mode=True,
            )
        request_arg = mock_dispatch.call_args.args[0]
        assert request_arg.voice_clone_prompt is None
        assert service._voice_clone_prompts == {}


class TestCacheKeyAndSerialization:
    """Cache key shape, hit-vs-miss, and per-voice serialization invariants."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_precompute_and_sets_request_prompt(
        self, tmp_path, monkeypatch
    ):
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()
        cache_key = (
            str(ref_audio.resolve()),
            service._model_registry.quality_tier.value,
        )
        service._voice_clone_prompts[cache_key] = prompt

        with patch.object(
            service,
            "_ensure_voice_clone_prompt_for_voice",
            new=AsyncMock(),
        ) as mock_compute, patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(),
        ) as mock_transcribe, patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ) as mock_dispatch:
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        mock_compute.assert_not_called()
        mock_transcribe.assert_not_called()
        request_arg = mock_dispatch.call_args.args[0]
        # Library contract (qwen_tts/inference/qwen3_tts_model.py:584-586):
        # voice_clone_prompt MUST be a list[VoiceClonePromptItem] so the
        # `_prompt_items_to_voice_clone_prompt` conversion path fires; a
        # bare item falls into the else-branch and crashes downstream at
        # `voice_clone_prompt['ref_spk_embedding']`.
        assert request_arg.voice_clone_prompt == [prompt]

    @pytest.mark.asyncio
    async def test_cache_miss_computes_once_and_populates_cache(
        self, tmp_path, monkeypatch
    ):
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()

        with patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(return_value="hello world"),
        ) as mock_transcribe, patch.object(
            service,
            "_ensure_voice_clone_prompt_for_voice",
            new=AsyncMock(return_value=prompt),
        ) as mock_compute, patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ) as mock_dispatch:
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        mock_transcribe.assert_awaited_once()
        mock_compute.assert_awaited_once()
        request_arg = mock_dispatch.call_args.args[0]
        # See library contract note in TestCacheKeyAndSerialization
        # ::test_cache_hit_skips_precompute_and_sets_request_prompt.
        assert request_arg.voice_clone_prompt == [prompt]
        cache_key = (
            str(ref_audio.resolve()),
            service._model_registry.quality_tier.value,
        )
        # The cache itself stores the bare item (single-instance memory
        # footprint); list-wrapping happens at the request-assignment site.
        assert service._voice_clone_prompts[cache_key] is prompt

    @pytest.mark.asyncio
    async def test_concurrent_same_voice_serializes(self, tmp_path, monkeypatch):
        """Per-voice asyncio.Lock ensures concurrent calls for the same
        voice run the precompute exactly once. Different voices proceed
        in parallel (covered by the next test)."""
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()

        compute_started = asyncio.Event()
        compute_release = asyncio.Event()
        call_count = {"compute": 0}

        async def slow_compute(*args, **kwargs):
            call_count["compute"] += 1
            compute_started.set()
            await compute_release.wait()
            return prompt

        async def fake_transcribe(*args, **kwargs):
            return "hello"

        with patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(side_effect=fake_transcribe),
        ), patch.object(
            service,
            "_ensure_voice_clone_prompt_for_voice",
            new=AsyncMock(side_effect=slow_compute),
        ), patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ):
            task1 = asyncio.create_task(
                service.generate_voice_clone(
                    text="a", ref_audio=ref_audio, ref_text="hi", streaming=True
                )
            )
            await compute_started.wait()
            task2 = asyncio.create_task(
                service.generate_voice_clone(
                    text="b", ref_audio=ref_audio, ref_text="hi", streaming=True
                )
            )
            # Give task2 a chance to reach the lock before we release.
            await asyncio.sleep(0.01)
            compute_release.set()
            await asyncio.gather(task1, task2)

        # The precompute must run exactly once across the two concurrent
        # calls — that's the whole point of per-voice serialization.
        assert call_count["compute"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_different_voices_run_in_parallel(
        self, tmp_path, monkeypatch
    ):
        """Different voices must NOT serialize on each other — different
        cache keys -> different locks -> parallel precompute."""
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_a = _make_clone_voice_file(tmp_path, name="VoiceA")
        ref_b = _make_clone_voice_file(tmp_path, name="VoiceB")

        a_started = asyncio.Event()
        b_started = asyncio.Event()
        a_release = asyncio.Event()
        b_release = asyncio.Event()

        async def slow_compute(voice_profile, ref_audio, transcription, tier):
            if "VoiceA" in str(ref_audio):
                a_started.set()
                await a_release.wait()
            else:
                b_started.set()
                await b_release.wait()
            return _make_synthetic_prompt()

        with patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(return_value="hi"),
        ), patch.object(
            service,
            "_ensure_voice_clone_prompt_for_voice",
            new=AsyncMock(side_effect=slow_compute),
        ), patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ):
            t1 = asyncio.create_task(
                service.generate_voice_clone(
                    text="a", ref_audio=ref_a, ref_text="hi", streaming=True
                )
            )
            t2 = asyncio.create_task(
                service.generate_voice_clone(
                    text="b", ref_audio=ref_b, ref_text="hi", streaming=True
                )
            )
            # If serialization were per-service (instead of per-voice),
            # b would not start until a finished — assert both started
            # while the first is still suspended in compute.
            await asyncio.wait_for(a_started.wait(), timeout=1.0)
            await asyncio.wait_for(b_started.wait(), timeout=1.0)
            a_release.set()
            b_release.set()
            await asyncio.gather(t1, t2)


# --------------------------------------------------------------------------- #
# AC #2 / Task 2 — TestEnsureTranscriptionForCloneVoice
# --------------------------------------------------------------------------- #


class TestEnsureTranscriptionForCloneVoice:
    """Transcription priority chain + Whisper retry policy."""

    @pytest.mark.asyncio
    async def test_in_memory_transcription_short_circuits(self, tmp_path):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        profile = MagicMock()
        profile.transcription = "pre-existing transcription"

        # Whisper service is None — would raise if invoked. Short-circuit
        # via in-memory transcription means it isn't.
        text = await service._ensure_transcription_for_clone_voice(
            profile, ref_audio
        )
        assert text == "pre-existing transcription"

    @pytest.mark.asyncio
    async def test_txt_sidecar_short_circuits_whisper(self, tmp_path):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        sidecar = ref_audio.with_suffix(".txt")
        sidecar.write_text("sidecar transcription", encoding="utf-8")

        profile = MagicMock()
        profile.transcription = ""  # empty, falls through

        text = await service._ensure_transcription_for_clone_voice(
            profile, ref_audio
        )
        assert text == "sidecar transcription"
        # Profile got populated from sidecar.
        profile.set_transcription_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_whisper_invoked_and_writes_sidecar(self, tmp_path):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        profile = MagicMock()
        profile.transcription = ""

        whisper_result = MagicMock()
        whisper_result.text = "whisper says hello"
        whisper_result.confidence = 0.95
        service._whisper_service = MagicMock()
        service._whisper_service.transcribe_file = AsyncMock(
            return_value=whisper_result
        )

        text = await service._ensure_transcription_for_clone_voice(
            profile, ref_audio
        )
        assert text == "whisper says hello"
        sidecar = ref_audio.with_suffix(".txt")
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8") == "whisper says hello"
        profile.set_transcription_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_whisper_retry_then_success(self, tmp_path, monkeypatch):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        profile = MagicMock()
        profile.transcription = ""

        # No-op asyncio.sleep so retries don't actually wait.
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        attempts = [
            RuntimeError("transient OOM"),
            RuntimeError("transient OOM"),
            MagicMock(text="finally got it", confidence=0.9),
        ]
        service._whisper_service = MagicMock()
        service._whisper_service.transcribe_file = AsyncMock(
            side_effect=attempts
        )

        text = await service._ensure_transcription_for_clone_voice(
            profile, ref_audio
        )
        assert text == "finally got it"
        assert service._whisper_service.transcribe_file.await_count == 3

    @pytest.mark.asyncio
    async def test_whisper_three_failures_marks_failed_and_raises(
        self, tmp_path, monkeypatch
    ):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        profile = MagicMock()
        profile.transcription = ""

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        service._whisper_service = MagicMock()
        service._whisper_service.transcribe_file = AsyncMock(
            side_effect=RuntimeError("persistent failure")
        )

        with pytest.raises(RuntimeError, match="Whisper transcription failed"):
            await service._ensure_transcription_for_clone_voice(
                profile, ref_audio
            )
        # Three attempts (initial + 2 retries per backoffs (1.0, 3.0)).
        assert service._whisper_service.transcribe_file.await_count == 3
        profile.mark_transcription_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_whisper_service_none_raises_without_attempting(self, tmp_path):
        """Lazy-fail-safe: when _whisper_service is None, raise so the
        dispatch chain falls through. Also exercises the optional init
        callback hook."""
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        profile = MagicMock()
        profile.transcription = ""

        service._whisper_service = None
        callback = MagicMock()
        service.set_whisper_init_callback(callback)

        with pytest.raises(RuntimeError, match="WhisperSubprocessService"):
            await service._ensure_transcription_for_clone_voice(
                profile, ref_audio
            )
        callback.assert_called_once()


# --------------------------------------------------------------------------- #
# AC #3 / Task 3 — TestEnsureVoiceClonePromptForVoice + TestHydration
# --------------------------------------------------------------------------- #


class TestEnsureVoiceClonePromptForVoice:
    """Persist + verify + invalidate semantics for the .pt file."""

    @pytest.mark.asyncio
    async def test_compute_writes_pt_and_meta(self, tmp_path):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()
        tier = "quality"

        with patch.object(
            service,
            "create_voice_clone_prompt_for_tier",
            new=AsyncMock(return_value=prompt),
        ), _patch_normalize(service):
            result = await service._ensure_voice_clone_prompt_for_voice(
                voice_profile=None,
                ref_audio=ref_audio,
                transcription="hello",
                tier=tier,
            )
        assert result is prompt
        pt_path = ref_audio.with_name(f"{ref_audio.stem}.{tier}.pt")
        meta_path = ref_audio.with_name(f"{ref_audio.stem}.{tier}.pt.meta.json")
        assert pt_path.exists()
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["tier"] == tier
        assert meta["qwen_tts_pin"] == service._QWEN_TTS_PIN_HASH
        assert meta["ref_audio_size"] == ref_audio.stat().st_size

    @pytest.mark.asyncio
    async def test_persisted_pt_loads_on_second_call(self, tmp_path):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()
        tier = "quality"

        with patch.object(
            service,
            "create_voice_clone_prompt_for_tier",
            new=AsyncMock(return_value=prompt),
        ) as mock_compute, _patch_normalize(service):
            await service._ensure_voice_clone_prompt_for_voice(
                None, ref_audio, "hello", tier
            )
            mock_compute.assert_awaited_once()

            # Second call must hit the on-disk fast path.
            await service._ensure_voice_clone_prompt_for_voice(
                None, ref_audio, "hello", tier
            )
            # No second compute — load from .pt.
            assert mock_compute.await_count == 1

    @pytest.mark.asyncio
    async def test_meta_size_mismatch_invalidates_pt(self, tmp_path):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()
        tier = "quality"

        with patch.object(
            service,
            "create_voice_clone_prompt_for_tier",
            new=AsyncMock(return_value=prompt),
        ) as mock_compute, _patch_normalize(service):
            await service._ensure_voice_clone_prompt_for_voice(
                None, ref_audio, "hello", tier
            )
            # Mutate ref_audio after caching — size diverges from meta.
            ref_audio.write_bytes(ref_audio.read_bytes() + b"\x00\x00\x00\x00")

            await service._ensure_voice_clone_prompt_for_voice(
                None, ref_audio, "hello", tier
            )
            # Stale meta -> recompute fired.
            assert mock_compute.await_count == 2

    @pytest.mark.asyncio
    async def test_pin_mismatch_invalidates_pt(self, tmp_path, monkeypatch):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()
        tier = "quality"

        with patch.object(
            service,
            "create_voice_clone_prompt_for_tier",
            new=AsyncMock(return_value=prompt),
        ) as mock_compute, _patch_normalize(service):
            await service._ensure_voice_clone_prompt_for_voice(
                None, ref_audio, "hello", tier
            )
            # Bump the pin: existing meta is now stale even though the audio
            # didn't change. Mirrors what a future qwen-tts pin bump would
            # do at startup.
            meta_path = ref_audio.with_name(
                f"{ref_audio.stem}.{tier}.pt.meta.json"
            )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["qwen_tts_pin"] = "different-pin"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

            await service._ensure_voice_clone_prompt_for_voice(
                None, ref_audio, "hello", tier
            )
            assert mock_compute.await_count == 2


class TestHydrateVoiceClonePromptCache:
    """Startup hydration scans the voice library for valid .pt files."""

    @pytest.mark.asyncio
    async def test_hydration_with_no_voice_manager_is_noop(self):
        service = _make_service()
        hits, total = await service.hydrate_voice_clone_prompt_cache()
        assert (hits, total) == (0, 0)

    @pytest.mark.asyncio
    async def test_hydration_loads_valid_pt_files(self, tmp_path):
        service = _make_service()
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()
        tier = service._model_registry.quality_tier.value

        # Pre-populate persisted .pt + meta via the lazy path (using mock).
        with patch.object(
            service,
            "create_voice_clone_prompt_for_tier",
            new=AsyncMock(return_value=prompt),
        ), _patch_normalize(service):
            await service._ensure_voice_clone_prompt_for_voice(
                None, ref_audio, "hello", tier
            )
        # Drop in-memory cache to simulate a fresh app launch.
        service._voice_clone_prompts.clear()

        # Wire a fake VoiceProfileManager.
        profile = MagicMock()
        profile.voice_type = VoiceType.CLONED
        profile.file_path = ref_audio
        manager = MagicMock()
        manager.get_profiles = MagicMock(return_value={"Sarira-F": profile})
        service.set_voice_profile_manager(manager)

        with _patch_normalize(service):
            hits, total = await service.hydrate_voice_clone_prompt_cache()
        assert total == 1
        assert hits == 1
        cache_key = (str(ref_audio.resolve()), tier)
        assert cache_key in service._voice_clone_prompts


# --------------------------------------------------------------------------- #
# AC #4 / Task 5 — TestPreparingVoiceIndicator
# --------------------------------------------------------------------------- #


class TestPreparingVoiceIndicator:
    """The TTS indicator surfaces the precompute status on miss only."""

    @pytest.mark.asyncio
    async def test_callback_fires_on_miss_entry_and_clears_on_exit(
        self, tmp_path, monkeypatch
    ):
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()

        emissions: List[Optional[str]] = []
        service.set_preparing_voice_callback(emissions.append)

        with patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(return_value="hello"),
        ), patch.object(
            service,
            "_ensure_voice_clone_prompt_for_voice",
            new=AsyncMock(return_value=prompt),
        ), patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ):
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        # Entry: non-empty message; exit: None.
        assert emissions[0] == "Preparing voice for streaming…"
        assert emissions[-1] is None

    @pytest.mark.asyncio
    async def test_callback_silent_on_cache_hit(self, tmp_path, monkeypatch):
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()
        cache_key = (
            str(ref_audio.resolve()),
            service._model_registry.quality_tier.value,
        )
        service._voice_clone_prompts[cache_key] = prompt

        emissions: List[Optional[str]] = []
        service.set_preparing_voice_callback(emissions.append)

        with patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ):
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        # Cache hit: no emission at all.
        assert emissions == []

    @pytest.mark.asyncio
    async def test_callback_clears_on_precompute_failure(
        self, tmp_path, monkeypatch
    ):
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)

        emissions: List[Optional[str]] = []
        service.set_preparing_voice_callback(emissions.append)

        with patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(side_effect=RuntimeError("whisper kaboom")),
        ), patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ):
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        # Entry message arrives, then None even on exception path.
        assert emissions[0] == "Preparing voice for streaming…"
        assert emissions[-1] is None


# --------------------------------------------------------------------------- #
# AC #5 / Task 6 — TestNFR7GracefulDegradation
# --------------------------------------------------------------------------- #


class TestNFR7GracefulDegradation:
    """The dispatch chain MUST still fall back to SENTENCE_STREAM on any
    non-cancel error — Story 16.6's NFR7 contract is preserved."""

    @pytest.mark.asyncio
    async def test_precompute_succeeds_request_carries_voice_clone_prompt(
        self, tmp_path, monkeypatch
    ):
        """Cache miss + successful precompute means TRUE_STREAM dispatch
        sees a populated voice_clone_prompt — the bug at line 2793-2798
        no longer triggers."""
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()

        # Capture the request that reaches _generate_true_stream.
        captured: dict = {}

        async def fake_true_stream(req):
            captured["request"] = req
            return QwenTTSResponse(success=True)

        with patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(return_value="hello"),
        ), patch.object(
            service,
            "_ensure_voice_clone_prompt_for_voice",
            new=AsyncMock(return_value=prompt),
        ), patch.object(
            service, "_generate_true_stream",
            new=AsyncMock(side_effect=fake_true_stream),
        ):
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        # Library contract — list[VoiceClonePromptItem] (see test above).
        assert captured["request"].voice_clone_prompt == [prompt]

    @pytest.mark.asyncio
    async def test_request_voice_clone_prompt_is_a_list_not_bare_item(
        self, tmp_path, monkeypatch
    ):
        """Regression for the bundled-smoke crash (Task 7 evidence run
        2026-05-08): `voice_clone_prompt` MUST reach the dispatch chain
        as a `list[VoiceClonePromptItem]`. The qwen-tts library at
        `qwen_tts/inference/qwen3_tts_model.py:584-586` only converts to
        the model-internal dict-form (`_prompt_items_to_voice_clone_prompt`)
        when the value is a list; a bare item falls into the else-branch
        and is passed straight through to `model.generate(...)` which
        crashes on `voice_clone_prompt['ref_spk_embedding']` (TypeError:
        'VoiceClonePromptItem' object is not subscriptable).

        Mirrors the canonical pattern at qwen_tts_service.py:2254 used by
        `generate_with_embedding`. Both the cache-hit and cache-miss
        branches must wrap.
        """
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()

        # ----- cache-hit branch -----
        cache_key = (
            str(ref_audio.resolve()),
            service._model_registry.quality_tier.value,
        )
        service._voice_clone_prompts[cache_key] = prompt
        with patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ) as mock_hit:
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        hit_request = mock_hit.call_args.args[0]
        assert isinstance(hit_request.voice_clone_prompt, list), (
            "cache hit: voice_clone_prompt must be a list, not a bare item"
        )
        assert hit_request.voice_clone_prompt == [prompt]

        # ----- cache-miss branch -----
        service._voice_clone_prompts.clear()
        ref_audio2 = _make_clone_voice_file(tmp_path, name="OtherVoice")
        prompt2 = _make_synthetic_prompt()
        with patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(return_value="hi"),
        ), patch.object(
            service,
            "_ensure_voice_clone_prompt_for_voice",
            new=AsyncMock(return_value=prompt2),
        ), patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=QwenTTSResponse(success=True)),
        ) as mock_miss:
            await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio2, ref_text="hi", streaming=True
            )
        miss_request = mock_miss.call_args.args[0]
        assert isinstance(miss_request.voice_clone_prompt, list), (
            "cache miss: voice_clone_prompt must be a list, not a bare item"
        )
        assert miss_request.voice_clone_prompt == [prompt2]

    @pytest.mark.asyncio
    async def test_cache_hit_then_oom_falls_back_to_sentence_stream(
        self, tmp_path, monkeypatch
    ):
        """Cache HIT + downstream CUDA-OOM in TRUE_STREAM still falls back
        per Story 16.6 NFR7."""
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        prompt = _make_synthetic_prompt()
        cache_key = (
            str(ref_audio.resolve()),
            service._model_registry.quality_tier.value,
        )
        service._voice_clone_prompts[cache_key] = prompt

        sentence_resp = QwenTTSResponse(
            success=True, used_fallback=True
        )
        with patch.object(
            service, "_generate_true_stream",
            new=AsyncMock(side_effect=RuntimeError("CUDA OOM")),
        ), patch.object(
            service, "_generate_streaming",
            new=AsyncMock(return_value=sentence_resp),
        ), patch.object(
            service, "_generate", new=AsyncMock()
        ):
            response = await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        assert response is sentence_resp

    @pytest.mark.asyncio
    async def test_persistent_whisper_failure_falls_back(
        self, tmp_path, monkeypatch
    ):
        """Cache MISS + Whisper retries exhausted -> precompute raises ->
        request reaches TRUE_STREAM with no prompt -> ValueError caught ->
        fallback to SENTENCE_STREAM."""
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        sentence_resp = QwenTTSResponse(
            success=True, used_fallback=True
        )

        with patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(side_effect=RuntimeError("whisper persistent")),
        ), patch.object(
            service, "_generate_true_stream",
            new=AsyncMock(side_effect=ValueError(
                "TRUE_STREAM voice-clone path requires "
                "request.voice_clone_prompt"
            )),
        ), patch.object(
            service, "_generate_streaming",
            new=AsyncMock(return_value=sentence_resp),
        ), patch.object(
            service, "_generate", new=AsyncMock()
        ):
            response = await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        assert response is sentence_resp

    @pytest.mark.asyncio
    async def test_embedding_compute_failure_falls_back(
        self, tmp_path, monkeypatch
    ):
        """Cache MISS + Whisper succeeds + embedding compute raises ->
        request reaches TRUE_STREAM with no prompt -> fallback fires."""
        service = _make_service()
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        ref_audio = _make_clone_voice_file(tmp_path)
        sentence_resp = QwenTTSResponse(
            success=True, used_fallback=True
        )

        with patch.object(
            service,
            "_ensure_transcription_for_clone_voice",
            new=AsyncMock(return_value="hello"),
        ), patch.object(
            service,
            "_ensure_voice_clone_prompt_for_voice",
            new=AsyncMock(side_effect=RuntimeError("compute kaboom")),
        ), patch.object(
            service, "_generate_true_stream",
            new=AsyncMock(side_effect=ValueError(
                "TRUE_STREAM voice-clone path requires "
                "request.voice_clone_prompt"
            )),
        ), patch.object(
            service, "_generate_streaming",
            new=AsyncMock(return_value=sentence_resp),
        ), patch.object(
            service, "_generate", new=AsyncMock()
        ):
            response = await service.generate_voice_clone(
                text="hi", ref_audio=ref_audio, ref_text="hi", streaming=True
            )
        assert response is sentence_resp
