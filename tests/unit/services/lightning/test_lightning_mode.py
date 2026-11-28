"""Epic 19 — Lightning mode: service wiring, reference codes, settings.

The native engine is replaced by a fake throughout, so these run with no GPU
and no qwen3-tts.cpp binaries. What they pin is MyVoice's side of the
contract: one engine on the GPU at a time, the registry lifecycle, cancel, and
the fall-back when the engine fails.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.services.core.base_service import ServiceStatus
from myvoice.services.lightning import engine as engine_module
from myvoice.services.lightning import ref_codes
from myvoice.services.qwen_tts_service import (
    QwenModelType, QwenTTSResponse, QwenTTSService,
)
from myvoice.services.sessions.session_registry import SessionRegistry


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class FakeEngine:
    """Stands in for ``LightningEngine``: three 0.1 s chunks per line."""

    def __init__(self, fail=None, gate=None):
        self.is_loaded = False
        self.loads = 0
        self.unloads = 0
        self.fail = fail
        self.gate = gate  # threading.Event: block before the 2nd chunk
        self.icl_calls = []
        self.xvector_calls = []

    def load(self, n_threads: int = 4):
        self.is_loaded = True
        self.loads += 1

    def unload(self):
        self.is_loaded = False
        self.unloads += 1

    def synthesize_icl_stream(self, text, ref_audio, ref_text, on_chunk, ref_code_path=None):
        self.icl_calls.append((text, ref_audio, ref_text, ref_code_path))
        if self.fail == "before_audio":
            raise engine_module.LightningError("boom")
        for index in range(3):
            if index == 1 and self.gate is not None:
                self.gate.wait(timeout=5)
            if not on_chunk(np.full(2400, 0.1 * (index + 1), dtype=np.float32)):
                break
            if self.fail == "mid_line":
                raise engine_module.LightningError("boom")
        return np.zeros(0, dtype=np.float32)

    def synthesize_xvector(self, text, ref_audio):
        self.xvector_calls.append((text, ref_audio))
        return np.full(4800, 0.2, dtype=np.float32)


@pytest.fixture
def rig(qapp, monkeypatch, tmp_path):
    registry = SessionRegistry()
    service = QwenTTSService(
        device="cpu", dtype="float32", session_registry=registry,
        app_settings=AppSettings(),
    )
    service.status = ServiceStatus.RUNNING
    service._executor = ThreadPoolExecutor(max_workers=2)
    service._request_semaphore = asyncio.Semaphore(1)
    service._model_registry.unload_all = AsyncMock(return_value=True)
    service._model_registry.ensure_model_loaded = AsyncMock(return_value=(True, None))
    service._save_audio_to_cache = lambda audio, rate: tmp_path / "current.wav"
    monkeypatch.setattr(engine_module, "availability", lambda: (True, ""))
    seen_prompt_caches = []
    service.seen_prompt_caches = seen_prompt_caches

    def _no_codes(ref_audio, cache_dir=None, prompt_cache=None):
        seen_prompt_caches.append(prompt_cache)
        return None

    monkeypatch.setattr(ref_codes, "ensure_ref_code_file", _no_codes)
    fake = FakeEngine()
    service._lightning_engine = fake
    chunks = []
    service.set_audio_chunk_ready_callback(chunks.append)
    yield service, registry, fake, chunks
    service._executor.shutdown(wait=False)


REF = Path("voice.wav")


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestEngineSwap:
    def test_enabling_unloads_the_standard_model_then_loads_lightning(self, rig):
        service, _, fake, _ = rig
        assert _run(service.set_lightning_enabled(True)) == (True, "")
        assert fake.is_loaded and service.lightning_enabled
        service._model_registry.unload_all.assert_awaited()

    def test_disabling_unloads_lightning(self, rig):
        service, _, fake, _ = rig
        _run(service.set_lightning_enabled(True))
        assert _run(service.set_lightning_enabled(False)) == (True, "")
        assert not fake.is_loaded and not service.lightning_enabled

    def test_unavailable_machine_refuses_with_the_reason(self, rig, monkeypatch):
        service, _, fake, _ = rig
        monkeypatch.setattr(engine_module, "availability", lambda: (False, "needs a GPU"))
        assert _run(service.set_lightning_enabled(True)) == (False, "needs a GPU")
        assert not fake.is_loaded and not service.lightning_enabled

    def test_standard_dispatch_releases_a_loaded_lightning_engine(self, rig):
        """A voice Lightning doesn't serve must not load PyTorch beside it."""
        service, _, fake, _ = rig
        _run(service.set_lightning_enabled(True))
        _run(service._release_lightning_for_standard_engine())
        assert not fake.is_loaded
        assert service.lightning_enabled  # the mode stays on for cloned voices

    def test_preload_of_base_model_loads_lightning_not_pytorch(self, rig):
        service, _, fake, _ = rig
        service._lightning_enabled = True
        assert _run(service.preload_model(QwenModelType.BASE)) == (True, None)
        assert fake.is_loaded
        service._model_registry.ensure_model_loaded.assert_not_awaited()

    def test_preload_of_another_model_releases_lightning(self, rig):
        service, _, fake, _ = rig
        _run(service.set_lightning_enabled(True))
        _run(service.preload_model(QwenModelType.CUSTOM_VOICE))
        assert not fake.is_loaded
        service._model_registry.ensure_model_loaded.assert_awaited()

    def test_compile_warmup_is_skipped_in_lightning_mode(self, rig):
        service, _, _, _ = rig
        service._lightning_enabled = True
        service._run_compile_priming = AsyncMock()
        _run(service.warmup_compile_async())
        service._run_compile_priming.assert_not_awaited()


class TestGeneration:
    def test_cloned_voice_streams_through_lightning(self, rig, qapp):
        service, registry, fake, chunks = rig
        states = []
        registry.session_state_changed.connect(lambda sid, state: states.append(state.value))
        service._lightning_enabled = True

        response = _run(service.generate_voice_clone("Cover me!", REF, "reference words"))
        qapp.processEvents()

        assert response.success and response.chunks_generated == 3
        assert len(response.audio_data) == 7200
        assert fake.icl_calls == [("Cover me!", REF, "reference words", None)]
        assert [(c.chunk_index, c.is_final, len(c.audio_data)) for c in chunks] == [
            (0, False, 2400), (1, False, 2400), (2, False, 2400), (3, True, 0),
        ]
        assert states == ["generating", "ready_to_play"]
        service._model_registry.ensure_model_loaded.assert_not_awaited()

    def test_voice_without_transcript_uses_timbre_only_mode(self, rig):
        service, _, fake, chunks = rig
        service._lightning_enabled = True
        response = _run(service.generate_voice_clone("Hi", REF, "", x_vector_only_mode=True))
        assert response.success and fake.xvector_calls == [("Hi", REF)]
        assert not fake.icl_calls
        assert [c.is_final for c in chunks] == [False, True]

    def test_lightning_off_never_touches_the_engine(self, rig):
        service, _, fake, _ = rig
        service._dispatch_by_streaming_mode = AsyncMock(return_value="standard")
        assert _run(service.generate_voice_clone("Hi", REF, "", x_vector_only_mode=True)) == "standard"
        assert not fake.icl_calls and not fake.xvector_calls

    def test_cancel_mid_line_stops_the_engine_and_settles_the_session(self, rig, qapp):
        service, registry, fake, chunks = rig
        fake.gate = threading.Event()
        service._lightning_enabled = True
        states = []
        registry.session_state_changed.connect(lambda sid, state: states.append(state.value))

        async def scenario():
            task = asyncio.ensure_future(
                service.generate_voice_clone("A long line", REF, "reference words")
            )
            while not chunks:
                await asyncio.sleep(0.01)
            assert await service.cancel_generation() is True
            fake.gate.set()
            return await task

        response = _run(scenario())
        qapp.processEvents()

        assert not response.success
        assert response.error_message == "Generation was cancelled"
        assert len(chunks) == 1  # nothing delivered after the cancel
        assert states[-2:] == ["cancelled", "discarded"]
        assert service.lightning_enabled  # a cancel is not an engine failure

    def test_engine_failure_before_audio_hands_the_line_to_the_standard_engine(self, rig, qapp):
        service, _, fake, _ = rig
        fake.fail = "before_audio"
        service._lightning_enabled = True
        notices = []
        service.set_lightning_state_callback(lambda on, message: notices.append(on))
        service._dispatch_by_streaming_mode = AsyncMock(return_value="standard")

        result = _run(service.generate_voice_clone("Hi", REF, "reference words", streaming=False))
        qapp.processEvents()

        assert result == "standard"  # the user still hears the line
        assert notices == [False] and not service.lightning_enabled

    def test_engine_failure_mid_line_does_not_say_the_line_twice(self, rig, qapp):
        service, _, fake, chunks = rig
        fake.fail = "mid_line"
        service._lightning_enabled = True
        service._dispatch_by_streaming_mode = AsyncMock(return_value="standard")

        result = _run(service.generate_voice_clone("Hi", REF, "reference words"))
        qapp.processEvents()

        assert result != "standard" and not result.success
        service._dispatch_by_streaming_mode.assert_not_awaited()
        assert len(chunks) == 1 and not service.lightning_enabled

    def test_empty_text_is_refused_without_unloading_the_engine(self, rig):
        service, _, fake, _ = rig
        _run(service.set_lightning_enabled(True))
        response = _run(service.generate_voice_clone("   ", REF, "reference words"))
        assert not response.success and fake.is_loaded and not fake.icl_calls


class TestRefCodes:
    def _voice(self, tmp_path, frames=5):
        import torch

        wav = tmp_path / "Voice.wav"
        wav.write_bytes(b"RIFF")
        code = torch.arange(frames * 16, dtype=torch.long).reshape(frames, 16)
        torch.save([{"ref_code": code, "ref_spk_embedding": torch.zeros(4)}],
                   str(tmp_path / "Voice.small.pt"))
        return wav

    def test_unwritable_cache_dir_returns_none_instead_of_raising(self, tmp_path):
        wav = self._voice(tmp_path)
        assert ref_codes.ensure_ref_code_file(wav, tmp_path / "missing") is None

    def test_codes_file_matches_the_prompt_cache(self, tmp_path):
        wav = self._voice(tmp_path)
        cache = tmp_path / "cache"
        cache.mkdir()
        path = ref_codes.ensure_ref_code_file(wav, cache)
        lines = path.read_text().splitlines()
        assert len(lines) == 5
        assert lines[0].split() == [str(v) for v in range(16)]
        assert lines[4].split()[-1] == "79"

    def test_second_call_reuses_the_file(self, tmp_path):
        wav = self._voice(tmp_path)
        cache = tmp_path / "cache"
        cache.mkdir()
        first = ref_codes.ensure_ref_code_file(wav, cache)
        stamp = first.stat().st_mtime_ns
        assert ref_codes.ensure_ref_code_file(wav, cache) == first
        assert first.stat().st_mtime_ns == stamp

    def test_replacing_the_clip_produces_a_new_file(self, tmp_path):
        wav = self._voice(tmp_path)
        cache = tmp_path / "cache"
        cache.mkdir()
        first = ref_codes.ensure_ref_code_file(wav, cache)
        wav.write_bytes(b"RIFF-a-different-recording")
        assert ref_codes.ensure_ref_code_file(wav, cache) != first

    def test_voice_without_a_prompt_cache_returns_none(self, tmp_path):
        wav = tmp_path / "New.wav"
        wav.write_bytes(b"RIFF")
        assert ref_codes.ensure_ref_code_file(wav, tmp_path) is None

    def test_corrupt_prompt_cache_returns_none_instead_of_raising(self, tmp_path):
        wav = tmp_path / "Bad.wav"
        wav.write_bytes(b"RIFF")
        (tmp_path / "Bad.small.pt").write_bytes(b"not a torch file")
        assert ref_codes.ensure_ref_code_file(wav, tmp_path) is None


class TestAvailabilityAndSettings:
    def test_missing_files_reported_as_not_installed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MYVOICE_LIGHTNING_DIR", str(tmp_path))
        ok, reason = engine_module.availability()
        assert not ok and reason == "Lightning mode files are not installed."

    def test_setting_round_trips_and_defaults_off(self):
        assert AppSettings().lightning_mode is False
        restored = AppSettings.from_dict(AppSettings(lightning_mode=True).to_dict())
        assert restored.lightning_mode is True
        legacy = AppSettings().to_dict()
        legacy.pop("lightning_mode")
        assert AppSettings.from_dict(legacy).lightning_mode is False


class TestSuppressedRequest:
    """Story 20.2's contract holds on the Lightning path too."""

    def test_suppressed_generation_reaches_no_user_facing_channel(self, rig, qapp):
        from myvoice.services.qwen_tts_service import QwenTTSRequest

        service, registry, fake, chunks = rig
        _run(service._ensure_lightning_loaded())
        saved = []
        service._save_audio_to_cache = lambda audio, rate: saved.append(1)
        service._cancel_requested = True  # a pending user Stop

        request = QwenTTSRequest(
            text="Hello world.", model_type=QwenModelType.BASE, ref_audio=REF,
            ref_text="reference words", suppress_audio_output=True,
        )
        response = _run(service._generate_lightning(request))
        qapp.processEvents()

        assert chunks == [] and saved == []
        assert registry.focal_session_id is None
        assert service._cancel_requested is True
        assert not response.success  # the pending Stop aborted it


class TestStudioEmbeddingVoice:
    """A Voice Design Studio voice (tester's 'AnnaEmotion') is a clone of an
    audio clip kept beside its embedding, so Lightning can speak it."""

    def _voice(self, tmp_path, ref_text="the reference words", tier_folder=True, clip=True):
        import torch

        from myvoice.services.qwen_tts_service import VoiceClonePromptItem

        emotion_dir = tmp_path / "AnnaEmotion" / "neutral"
        pt_dir = emotion_dir / "1.7" if tier_folder else emotion_dir
        pt_dir.mkdir(parents=True)
        torch.save(
            VoiceClonePromptItem(
                ref_code=torch.zeros(4, 16, dtype=torch.long),
                ref_spk_embedding=torch.zeros(2048),  # 1.7B-sized: irrelevant here
                ref_text=ref_text,
            ),
            str(pt_dir / "embedding.pt"),
        )
        if clip:
            (emotion_dir / "source_audio.wav").write_bytes(b"RIFF")
        return pt_dir / "embedding.pt", emotion_dir / "source_audio.wav"

    def test_studio_voice_is_spoken_by_lightning_from_its_source_clip(self, rig, tmp_path):
        service, _, fake, _ = rig
        service._lightning_enabled = True
        embedding, clip = self._voice(tmp_path)

        response = _run(service.generate_with_embedding("sounds good to me", embedding))

        assert response.success
        assert fake.icl_calls == [("sounds good to me", clip, "the reference words", None)]
        assert service.seen_prompt_caches == [embedding]  # codes come from the embedding
        service._model_registry.ensure_model_loaded.assert_not_awaited()

    def test_legacy_layout_without_a_tier_folder(self, rig, tmp_path):
        service, _, fake, _ = rig
        service._lightning_enabled = True
        embedding, clip = self._voice(tmp_path, tier_folder=False)
        assert _run(service.generate_with_embedding("hi", embedding)).success
        assert fake.icl_calls[0][1] == clip

    def test_studio_voice_without_a_transcript_uses_timbre_only_mode(self, rig, tmp_path):
        service, _, fake, _ = rig
        service._lightning_enabled = True
        embedding, clip = self._voice(tmp_path, ref_text=None)
        assert _run(service.generate_with_embedding("hi", embedding)).success
        assert fake.xvector_calls == [("hi", clip)] and not fake.icl_calls

    def test_studio_voice_without_its_clip_goes_to_the_standard_engine(self, rig, tmp_path):
        service, _, fake, _ = rig
        service._lightning_enabled = True
        standard = QwenTTSResponse(success=True)
        service._dispatch_by_streaming_mode = AsyncMock(return_value=standard)
        embedding, _ = self._voice(tmp_path, clip=False)

        assert _run(service.generate_with_embedding("hi", embedding)) is standard
        assert not fake.icl_calls and not fake.xvector_calls
        assert service.lightning_enabled  # not a failure: the mode stays on

    def test_lightning_off_leaves_studio_voices_on_the_standard_engine(self, rig, tmp_path):
        service, _, fake, _ = rig
        standard = QwenTTSResponse(success=True)
        service._dispatch_by_streaming_mode = AsyncMock(return_value=standard)
        embedding, _ = self._voice(tmp_path)
        assert _run(service.generate_with_embedding("hi", embedding)) is standard
        assert not fake.icl_calls

    def test_codes_are_read_from_the_named_embedding_file(self, tmp_path):
        embedding, clip = self._voice(tmp_path)
        cache = tmp_path / "cache"
        cache.mkdir()
        from myvoice.services.lightning.ref_codes import ensure_ref_code_file as real

        path = real(clip, cache, prompt_cache=embedding)
        assert len(path.read_text().splitlines()) == 4


def test_streaming_configuration_is_the_listening_tested_one():
    """Chosen by ear, not by metric: 48 frames of decoder look-back (long
    lines), 10-frame chunks (first audio), exact reference cut (no clipped
    first words). A change here needs a new listening pass."""
    assert engine_module._ENV_DEFAULTS == {
        "QWEN3_TTS_BACKEND": "cuda",
        "QWEN3_TTS_STREAM_CONTEXT_FRAMES": "48",
        "QWEN3_TTS_STREAM_CHUNK_FRAMES": "10",
        "QWEN3_TTS_ECHO_MARGIN_FRAMES": "0",
    }
