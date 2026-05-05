"""
Integration Tests for Optimized Voice Flow

End-to-end tests for the complete optimized voice workflow:
1. Creating optimized voice profile
2. Registering with voice manager
3. Selecting as active voice
4. TTS generation with checkpoint loading
"""

import asyncio
import pytest
import json
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path

# Skip if PyQt6 not available
pytest.importorskip("PyQt6")

from myvoice.models.voice_profile import VoiceProfile, VoiceType
from myvoice.models.service_enums import QwenModelType, ModelState


@pytest.fixture
def temp_voice_dir(tmp_path):
    """Create a temporary voice directory."""
    voice_dir = tmp_path / "voice_files"
    voice_dir.mkdir(parents=True, exist_ok=True)
    return voice_dir


@pytest.fixture
def temp_cache_file(tmp_path):
    """Create a temporary cache file path."""
    cache_dir = tmp_path / "config"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / ".voice_cache.json"


@pytest.fixture
def checkpoint_path(tmp_path):
    """Create a mock checkpoint directory."""
    checkpoint = tmp_path / "checkpoints" / "my_optimized_voice"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text("{}")
    return checkpoint


@pytest.fixture
def optimized_profile(checkpoint_path):
    """Create an optimized voice profile."""
    return VoiceProfile.create_optimized_profile(
        name="my_optimized_voice",
        checkpoint_path=checkpoint_path,
        speaker_name="my_speaker",
        description="Test optimized voice"
    )


class TestOptimizedVoiceCreationFlow:
    """Tests for creating and registering optimized voices."""

    @pytest.mark.asyncio
    async def test_create_and_register_optimized_voice(
        self, temp_voice_dir, temp_cache_file, optimized_profile
    ):
        """Test creating and registering an optimized voice profile."""
        from myvoice.services.voice_profile_service import VoiceProfileManager

        manager = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            auto_scan=False
        )
        await manager.start()

        # Register the optimized profile
        result = await manager.register_profile(optimized_profile)
        assert result is True

        # Verify profile is in manager
        assert optimized_profile.name in manager._profiles

        # Verify profile type
        registered = manager._profiles[optimized_profile.name]
        assert registered.voice_type == VoiceType.OPTIMIZED
        assert registered.checkpoint_path is not None
        assert registered.speaker_name == "my_speaker"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_optimized_voice_persists_in_cache(
        self, temp_voice_dir, temp_cache_file, optimized_profile
    ):
        """Test that optimized voice persists to cache and reloads."""
        from myvoice.services.voice_profile_service import VoiceProfileManager

        # First manager - register profile
        manager1 = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            auto_scan=False
        )
        await manager1.start()
        await manager1.register_profile(optimized_profile)
        await asyncio.sleep(0.1)  # Wait for cache save
        await manager1.stop()

        # Second manager - should load from cache
        manager2 = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            auto_scan=False
        )
        await manager2.start()

        # Profile should be present
        assert optimized_profile.name in manager2._profiles

        # Verify all fields preserved
        loaded = manager2._profiles[optimized_profile.name]
        assert loaded.voice_type == VoiceType.OPTIMIZED
        assert loaded.checkpoint_path == optimized_profile.checkpoint_path
        assert loaded.speaker_name == "my_speaker"
        assert loaded.emotion_capable is True

        await manager2.stop()


class TestOptimizedVoiceSelectionFlow:
    """Tests for selecting optimized voices as active."""

    @pytest.mark.asyncio
    async def test_set_optimized_voice_active(
        self, temp_voice_dir, temp_cache_file, optimized_profile
    ):
        """Test setting optimized voice as active profile."""
        from myvoice.services.voice_profile_service import VoiceProfileManager

        manager = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            auto_scan=False
        )
        await manager.start()
        await manager.register_profile(optimized_profile)

        # Set as active
        result = await manager.set_active_profile(optimized_profile.name)
        assert result is True

        # Verify active profile
        active = manager.get_active_profile()
        assert active is not None
        assert active.name == optimized_profile.name
        assert active.voice_type == VoiceType.OPTIMIZED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_active_optimized_voice_has_all_fields(
        self, temp_voice_dir, temp_cache_file, optimized_profile
    ):
        """Test that active optimized voice has all required fields for TTS."""
        from myvoice.services.voice_profile_service import VoiceProfileManager

        manager = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            auto_scan=False
        )
        await manager.start()
        await manager.register_profile(optimized_profile)
        await manager.set_active_profile(optimized_profile.name)

        active = manager.get_active_profile()

        # All these fields are required for TTS routing in app.py
        assert active.voice_type == VoiceType.OPTIMIZED
        assert active.checkpoint_path is not None
        assert active.speaker_name is not None
        assert active.voice_type.supports_emotion is True

        await manager.stop()


class TestOptimizedVoiceTTSRouting:
    """Tests for TTS routing with optimized voices."""

    def test_voice_type_routing_distinction(self):
        """Test that OPTIMIZED is distinct from other voice types for routing."""
        # This verifies the routing logic can distinguish types
        voice_types = [VoiceType.BUNDLED, VoiceType.DESIGNED, VoiceType.CLONED, VoiceType.OPTIMIZED]

        # All types should be distinct
        assert len(set(voice_types)) == 4

        # Only OPTIMIZED, BUNDLED, and DESIGNED support emotion
        for vt in voice_types:
            if vt == VoiceType.CLONED:
                assert vt.supports_emotion is False
            else:
                assert vt.supports_emotion is True

    @pytest.mark.asyncio
    async def test_tts_service_receives_checkpoint_path(self, checkpoint_path):
        """Test that TTS service receives checkpoint path for optimized voices."""
        from myvoice.services.qwen_tts_service import QwenTTSService, QwenTTSRequest

        # Create a mock TTS service
        with patch('myvoice.services.qwen_tts_service.ModelRegistry'):
            service = QwenTTSService.__new__(QwenTTSService)
            service._running = True
            service.logger = Mock()
            service.EMOTION_PRESETS = {'neutral': None}

            # Track the request
            captured_request = None

            async def mock_generate_streaming(request):
                nonlocal captured_request
                captured_request = request
                return Mock(success=True)

            service._generate_streaming = mock_generate_streaming

            # Call generate_optimized_voice
            from myvoice.services.qwen_tts_service import QwenTTSService as RealService
            service.generate_optimized_voice = RealService.generate_optimized_voice.__get__(
                service, RealService
            )

            await service.generate_optimized_voice(
                text="Hello world",
                checkpoint_path=checkpoint_path,
                speaker_name="test_speaker",
                streaming=True
            )

            # Verify request has checkpoint_path
            assert captured_request is not None
            assert captured_request.checkpoint_path == checkpoint_path
            assert captured_request.speaker == "test_speaker"
            assert captured_request.model_type == QwenModelType.CUSTOM_VOICE


class TestOptimizedVoiceEmotionSupport:
    """Tests for emotion support with optimized voices."""

    def test_optimized_voice_emotion_capable(self, optimized_profile):
        """Test that optimized voice is emotion capable."""
        assert optimized_profile.emotion_capable is True
        assert optimized_profile.voice_type.supports_emotion is True

    @pytest.mark.asyncio
    async def test_emotion_preset_applied_to_optimized_voice(self, checkpoint_path):
        """Test that emotion presets are applied to optimized voice generation."""
        from myvoice.services.qwen_tts_service import QwenTTSService

        with patch('myvoice.services.qwen_tts_service.ModelRegistry'):
            service = QwenTTSService.__new__(QwenTTSService)
            service._running = True
            service.logger = Mock()
            service.EMOTION_PRESETS = {
                'neutral': None,
                'happy': 'Speak with joy and enthusiasm',
                'sad': 'Speak with melancholy and sorrow'
            }

            captured_request = None

            async def mock_generate(request):
                nonlocal captured_request
                captured_request = request
                return Mock(success=True)

            service._generate = mock_generate
            service.generate_optimized_voice = QwenTTSService.generate_optimized_voice.__get__(
                service, QwenTTSService
            )

            # Generate with happy emotion
            await service.generate_optimized_voice(
                text="Hello world",
                checkpoint_path=checkpoint_path,
                speaker_name="test_speaker",
                emotion_preset="happy",
                streaming=False
            )

            # Verify instruct was set
            assert captured_request is not None
            assert captured_request.instruct == "Speak with joy and enthusiasm"


class TestOptimizedVoiceModelLoading:
    """Tests for model loading with optimized voices."""

    @pytest.mark.asyncio
    async def test_checkpoint_path_passed_to_model_registry(self, checkpoint_path):
        """Test that checkpoint path is passed to model registry."""
        from myvoice.services.model_registry import ModelRegistry

        with patch.object(ModelRegistry, '_load_model_sync', return_value=Mock()):
            registry = ModelRegistry(device="cpu", dtype="float32")

            # Track what gets passed to _load_model
            with patch.object(registry, '_load_model', new_callable=AsyncMock) as mock_load:
                mock_load.return_value = (True, None)

                await registry.ensure_model_loaded(
                    QwenModelType.CUSTOM_VOICE,
                    checkpoint_path=str(checkpoint_path)
                )

                # Verify checkpoint_path was passed
                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args.kwargs
                assert call_kwargs.get('checkpoint_path') == str(checkpoint_path)

            registry.shutdown()

    @pytest.mark.asyncio
    async def test_different_checkpoint_triggers_model_reload(self, tmp_path):
        """Test that using a different checkpoint reloads the model."""
        from myvoice.services.model_registry import ModelRegistry

        checkpoint1 = tmp_path / "checkpoint1"
        checkpoint2 = tmp_path / "checkpoint2"
        checkpoint1.mkdir()
        checkpoint2.mkdir()

        with patch.object(ModelRegistry, '_load_model_sync', return_value=Mock()):
            registry = ModelRegistry(device="cpu", dtype="float32")

            # Simulate first checkpoint loaded
            registry._current_model_type = QwenModelType.CUSTOM_VOICE
            registry._current_checkpoint_path = str(checkpoint1)
            registry._models[QwenModelType.CUSTOM_VOICE].state = ModelState.READY
            registry._models[QwenModelType.CUSTOM_VOICE].model_instance = Mock()

            with patch.object(registry, '_unload_model', new_callable=AsyncMock) as mock_unload:
                with patch.object(registry, '_load_model', new_callable=AsyncMock) as mock_load:
                    mock_load.return_value = (True, None)

                    # Request different checkpoint
                    await registry.ensure_model_loaded(
                        QwenModelType.CUSTOM_VOICE,
                        checkpoint_path=str(checkpoint2)
                    )

                    # Should have unloaded old and loaded new
                    mock_unload.assert_called_once()
                    mock_load.assert_called_once()

            registry.shutdown()


class TestOptimizedVoiceValidProfiles:
    """Tests for optimized voices in get_valid_profiles."""

    @pytest.mark.asyncio
    async def test_optimized_voice_in_valid_profiles(
        self, temp_voice_dir, temp_cache_file, optimized_profile
    ):
        """Test that optimized voices appear in get_valid_profiles."""
        from myvoice.services.voice_profile_service import VoiceProfileManager

        manager = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            auto_scan=False
        )
        await manager.start()
        await manager.register_profile(optimized_profile)

        valid_profiles = manager.get_valid_profiles()

        assert optimized_profile.name in valid_profiles
        assert valid_profiles[optimized_profile.name].voice_type == VoiceType.OPTIMIZED

        await manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
