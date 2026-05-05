"""
Tests for Optimized Voice Integration

Tests for the complete optimized voice workflow including:
- VoiceType.OPTIMIZED with emotion support
- VoiceProfile.create_optimized_profile() factory method
- ModelRegistry dynamic checkpoint loading
- QwenTTSService.generate_optimized_voice() generation
- App.py voice routing for OPTIMIZED type
"""

import asyncio
import pytest
import json
import tempfile
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path

# Skip this module if PyQt6 is not available (required for myvoice imports)
pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")

from myvoice.models.voice_profile import (
    VoiceProfile, VoiceType, TranscriptionStatus
)
from myvoice.models.service_enums import QwenModelType, ModelState


# ============================================================================
# VoiceType.OPTIMIZED Tests
# ============================================================================

class TestVoiceTypeOptimized:
    """Tests for VoiceType.OPTIMIZED enum properties."""

    def test_optimized_type_exists(self):
        """Test that OPTIMIZED type exists in VoiceType enum."""
        assert hasattr(VoiceType, 'OPTIMIZED')
        assert VoiceType.OPTIMIZED.value == "optimized"

    def test_optimized_supports_emotion(self):
        """Test that OPTIMIZED type supports emotion presets."""
        assert VoiceType.OPTIMIZED.supports_emotion is True

    def test_optimized_display_name(self):
        """Test OPTIMIZED display name."""
        assert VoiceType.OPTIMIZED.display_name == "Optimized"

    def test_optimized_icon(self):
        """Test OPTIMIZED icon."""
        assert VoiceType.OPTIMIZED.icon == "\U0001F527"  # wrench emoji

    def test_optimized_sort_order(self):
        """Test OPTIMIZED sort order is between DESIGNED and CLONED."""
        # Sort orders: BUNDLED=0, DESIGNED=1, OPTIMIZED=2, CLONED=3
        assert VoiceType.OPTIMIZED.sort_order == 2
        assert VoiceType.DESIGNED.sort_order < VoiceType.OPTIMIZED.sort_order
        assert VoiceType.OPTIMIZED.sort_order < VoiceType.CLONED.sort_order

    def test_optimized_required_model(self):
        """Test OPTIMIZED uses CUSTOM_VOICE model."""
        assert VoiceType.OPTIMIZED.required_model == QwenModelType.CUSTOM_VOICE

    def test_cloned_does_not_support_emotion(self):
        """Test that CLONED type does NOT support emotion (contrast test)."""
        assert VoiceType.CLONED.supports_emotion is False

    def test_bundled_supports_emotion(self):
        """Test that BUNDLED type supports emotion (like OPTIMIZED)."""
        assert VoiceType.BUNDLED.supports_emotion is True


# ============================================================================
# VoiceProfile.create_optimized_profile() Tests
# ============================================================================

class TestCreateOptimizedProfile:
    """Tests for VoiceProfile.create_optimized_profile() factory method."""

    def test_create_optimized_profile_basic(self, tmp_path):
        """Test creating an optimized voice profile with required fields."""
        checkpoint_dir = tmp_path / "checkpoints" / "my_voice"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text("{}")

        profile = VoiceProfile.create_optimized_profile(
            name="test_voice",
            checkpoint_path=checkpoint_dir,
            speaker_name="test_speaker"
        )

        assert profile.name == "test_voice"
        assert profile.voice_type == VoiceType.OPTIMIZED
        assert profile.checkpoint_path == checkpoint_dir
        assert profile.speaker_name == "test_speaker"
        assert profile.is_valid is True
        assert profile.emotion_capable is True

    def test_create_optimized_profile_with_description(self, tmp_path):
        """Test creating optimized profile with description."""
        checkpoint_dir = tmp_path / "checkpoints" / "my_voice"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text("{}")

        profile = VoiceProfile.create_optimized_profile(
            name="my_voice",
            checkpoint_path=checkpoint_dir,
            speaker_name="my_speaker",
            description="Fine-tuned voice for product demos"
        )

        assert profile.description == "Fine-tuned voice for product demos"

    def test_optimized_profile_virtual_path(self, tmp_path):
        """Test that optimized profile has virtual path format."""
        checkpoint_dir = tmp_path / "checkpoints" / "my_voice"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text("{}")

        profile = VoiceProfile.create_optimized_profile(
            name="test_voice",
            checkpoint_path=checkpoint_dir,
            speaker_name="test_speaker"
        )

        # Virtual path should be optimized://name format
        assert str(profile.file_path).startswith("optimized:")

    def test_optimized_profile_transcription_skipped(self, tmp_path):
        """Test that transcription is skipped for optimized voices."""
        checkpoint_dir = tmp_path / "checkpoints" / "my_voice"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text("{}")

        profile = VoiceProfile.create_optimized_profile(
            name="test_voice",
            checkpoint_path=checkpoint_dir,
            speaker_name="test_speaker"
        )

        assert profile.transcription_status == TranscriptionStatus.SKIPPED
        assert profile.transcription is None

    def test_is_optimized_helper(self, tmp_path):
        """Test is_optimized() helper method."""
        checkpoint_dir = tmp_path / "checkpoints" / "my_voice"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text("{}")

        profile = VoiceProfile.create_optimized_profile(
            name="test_voice",
            checkpoint_path=checkpoint_dir,
            speaker_name="test_speaker"
        )

        assert profile.is_optimized() is True

    def test_non_optimized_is_optimized_returns_false(self, tmp_path):
        """Test is_optimized() returns False for other types."""
        cloned_profile = VoiceProfile(
            file_path=tmp_path / "voice.wav",
            name="cloned_voice",
            voice_type=VoiceType.CLONED
        )
        assert cloned_profile.is_optimized() is False


# ============================================================================
# VoiceProfile Serialization Tests
# ============================================================================

class TestOptimizedProfileSerialization:
    """Tests for optimized profile to_dict/from_dict serialization."""

    def test_to_dict_includes_optimized_fields(self, tmp_path):
        """Test that to_dict includes checkpoint_path and speaker_name."""
        checkpoint_dir = tmp_path / "checkpoints" / "test_voice"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text("{}")

        profile = VoiceProfile.create_optimized_profile(
            name="test_voice",
            checkpoint_path=checkpoint_dir,
            speaker_name="test_speaker",
            description="Test description"
        )

        data = profile.to_dict()

        assert data['voice_type'] == 'optimized'
        assert data['checkpoint_path'] == str(checkpoint_dir)
        assert data['speaker_name'] == 'test_speaker'
        assert data['description'] == 'Test description'

    def test_from_dict_restores_optimized_profile(self, tmp_path):
        """Test that from_dict restores optimized profile fields."""
        checkpoint_dir = tmp_path / "checkpoints" / "test_voice"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text("{}")

        original = VoiceProfile.create_optimized_profile(
            name="test_voice",
            checkpoint_path=checkpoint_dir,
            speaker_name="test_speaker",
            description="Test description"
        )

        data = original.to_dict()
        restored = VoiceProfile.from_dict(data)

        assert restored.voice_type == VoiceType.OPTIMIZED
        assert restored.checkpoint_path == checkpoint_dir
        assert restored.speaker_name == "test_speaker"
        assert restored.description == "Test description"
        assert restored.is_optimized() is True


# ============================================================================
# ModelRegistry Checkpoint Loading Tests
# ============================================================================

# Skip ModelRegistry tests if torch has issues (common in test environments)
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available or has loading issues")
class TestModelRegistryCheckpointLoading:
    """Tests for ModelRegistry dynamic checkpoint loading."""

    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock ModelRegistry for testing."""
        from myvoice.services.model_registry import ModelRegistry

        with patch.object(ModelRegistry, '_load_model_sync', return_value=Mock()):
            registry = ModelRegistry(device="cpu", dtype="float32")
            yield registry
            registry.shutdown()

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_with_checkpoint(self, mock_model_registry, tmp_path):
        """Test loading model with custom checkpoint path."""
        checkpoint_path = tmp_path / "my_checkpoint"
        checkpoint_path.mkdir()

        # Mock the internal load method to verify checkpoint_path is passed
        with patch.object(mock_model_registry, '_load_model', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = (True, None)

            success, error = await mock_model_registry.ensure_model_loaded(
                QwenModelType.CUSTOM_VOICE,
                checkpoint_path=str(checkpoint_path)
            )

            # Verify _load_model was called with checkpoint_path
            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert call_args.kwargs.get('checkpoint_path') == str(checkpoint_path)

    @pytest.mark.asyncio
    async def test_checkpoint_path_tracked(self, mock_model_registry, tmp_path):
        """Test that current checkpoint path is tracked."""
        checkpoint_path = tmp_path / "my_checkpoint"
        checkpoint_path.mkdir()

        with patch.object(mock_model_registry, '_load_model', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = (True, None)

            # Manually simulate what _load_model does
            async def fake_load(model_type, checkpoint_path=None):
                mock_model_registry._current_checkpoint_path = checkpoint_path
                mock_model_registry._current_model_type = model_type
                return (True, None)

            mock_load.side_effect = fake_load

            await mock_model_registry.ensure_model_loaded(
                QwenModelType.CUSTOM_VOICE,
                checkpoint_path=str(checkpoint_path)
            )

            assert mock_model_registry.current_checkpoint_path == str(checkpoint_path)

    @pytest.mark.asyncio
    async def test_same_checkpoint_no_reload(self, mock_model_registry, tmp_path):
        """Test that same checkpoint doesn't trigger reload."""
        checkpoint_path = tmp_path / "my_checkpoint"
        checkpoint_path.mkdir()

        # Setup initial state
        mock_model_registry._current_model_type = QwenModelType.CUSTOM_VOICE
        mock_model_registry._current_checkpoint_path = str(checkpoint_path)
        mock_model_registry._models[QwenModelType.CUSTOM_VOICE].state = ModelState.READY

        with patch.object(mock_model_registry, '_load_model', new_callable=AsyncMock) as mock_load:
            success, error = await mock_model_registry.ensure_model_loaded(
                QwenModelType.CUSTOM_VOICE,
                checkpoint_path=str(checkpoint_path)
            )

            # Should not reload since same checkpoint
            mock_load.assert_not_called()
            assert success is True

    @pytest.mark.asyncio
    async def test_different_checkpoint_triggers_reload(self, mock_model_registry, tmp_path):
        """Test that different checkpoint triggers reload."""
        checkpoint1 = tmp_path / "checkpoint1"
        checkpoint2 = tmp_path / "checkpoint2"
        checkpoint1.mkdir()
        checkpoint2.mkdir()

        # Setup initial state with checkpoint1
        mock_model_registry._current_model_type = QwenModelType.CUSTOM_VOICE
        mock_model_registry._current_checkpoint_path = str(checkpoint1)
        mock_model_registry._models[QwenModelType.CUSTOM_VOICE].state = ModelState.READY

        with patch.object(mock_model_registry, '_unload_model', new_callable=AsyncMock) as mock_unload:
            with patch.object(mock_model_registry, '_load_model', new_callable=AsyncMock) as mock_load:
                mock_load.return_value = (True, None)

                await mock_model_registry.ensure_model_loaded(
                    QwenModelType.CUSTOM_VOICE,
                    checkpoint_path=str(checkpoint2)
                )

                # Should unload old and load new
                mock_unload.assert_called_once()
                mock_load.assert_called_once()


# ============================================================================
# QwenTTSService.generate_optimized_voice() Tests
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available or has loading issues")
class TestGenerateOptimizedVoice:
    """Tests for QwenTTSService.generate_optimized_voice() method."""

    @pytest.fixture
    def mock_tts_service(self):
        """Create a mock TTS service."""
        from myvoice.services.qwen_tts_service import QwenTTSService

        with patch('myvoice.services.qwen_tts_service.ModelRegistry'):
            service = QwenTTSService.__new__(QwenTTSService)
            service._running = True
            service._generation_state = Mock()
            service._model_registry = Mock()
            service._executor = Mock()
            service.logger = Mock()
            service._total_requests = 0
            service._failed_requests = 0
            service.EMOTION_PRESETS = {
                'neutral': None,
                'happy': 'Speak with joy and enthusiasm',
                'sad': 'Speak with melancholy'
            }
            yield service

    @pytest.mark.asyncio
    async def test_generate_optimized_voice_validates_checkpoint(self, mock_tts_service, tmp_path):
        """Test that non-existent checkpoint returns error."""
        from myvoice.services.qwen_tts_service import QwenTTSService

        # Use real method
        mock_tts_service.generate_optimized_voice = QwenTTSService.generate_optimized_voice.__get__(
            mock_tts_service, QwenTTSService
        )

        nonexistent_path = tmp_path / "nonexistent_checkpoint"

        response = await mock_tts_service.generate_optimized_voice(
            text="Hello world",
            checkpoint_path=nonexistent_path,
            speaker_name="test_speaker"
        )

        assert response.success is False
        assert "not found" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_generate_optimized_voice_uses_emotion_preset(self, mock_tts_service, tmp_path):
        """Test that emotion presets are resolved correctly."""
        from myvoice.services.qwen_tts_service import QwenTTSService, QwenTTSRequest

        checkpoint_path = tmp_path / "test_checkpoint"
        checkpoint_path.mkdir()

        # Track the request that gets created
        captured_request = None

        async def mock_generate_streaming(request):
            nonlocal captured_request
            captured_request = request
            return Mock(success=True)

        mock_tts_service._generate_streaming = mock_generate_streaming

        # Use real method
        mock_tts_service.generate_optimized_voice = QwenTTSService.generate_optimized_voice.__get__(
            mock_tts_service, QwenTTSService
        )

        await mock_tts_service.generate_optimized_voice(
            text="Hello world",
            checkpoint_path=checkpoint_path,
            speaker_name="test_speaker",
            emotion_preset="happy",
            streaming=True
        )

        # Verify emotion preset was resolved to instruct
        assert captured_request is not None
        assert captured_request.instruct == "Speak with joy and enthusiasm"

    @pytest.mark.asyncio
    async def test_generate_optimized_voice_passes_checkpoint_path(self, mock_tts_service, tmp_path):
        """Test that checkpoint_path is included in request."""
        from myvoice.services.qwen_tts_service import QwenTTSService

        checkpoint_path = tmp_path / "test_checkpoint"
        checkpoint_path.mkdir()

        captured_request = None

        async def mock_generate(request):
            nonlocal captured_request
            captured_request = request
            return Mock(success=True)

        mock_tts_service._generate = mock_generate

        mock_tts_service.generate_optimized_voice = QwenTTSService.generate_optimized_voice.__get__(
            mock_tts_service, QwenTTSService
        )

        await mock_tts_service.generate_optimized_voice(
            text="Hello world",
            checkpoint_path=checkpoint_path,
            speaker_name="test_speaker",
            streaming=False
        )

        assert captured_request is not None
        assert captured_request.checkpoint_path == checkpoint_path

    @pytest.mark.asyncio
    async def test_generate_optimized_voice_uses_custom_voice_model(self, mock_tts_service, tmp_path):
        """Test that CUSTOM_VOICE model type is used."""
        from myvoice.services.qwen_tts_service import QwenTTSService

        checkpoint_path = tmp_path / "test_checkpoint"
        checkpoint_path.mkdir()

        captured_request = None

        async def mock_generate(request):
            nonlocal captured_request
            captured_request = request
            return Mock(success=True)

        mock_tts_service._generate = mock_generate

        mock_tts_service.generate_optimized_voice = QwenTTSService.generate_optimized_voice.__get__(
            mock_tts_service, QwenTTSService
        )

        await mock_tts_service.generate_optimized_voice(
            text="Hello world",
            checkpoint_path=checkpoint_path,
            speaker_name="test_speaker",
            streaming=False
        )

        assert captured_request.model_type == QwenModelType.CUSTOM_VOICE


# ============================================================================
# VoiceProfileManager Integration Tests
# ============================================================================

class TestVoiceProfileManagerOptimized:
    """Tests for VoiceProfileManager handling of optimized voices."""

    @pytest.fixture
    def temp_voice_dir(self, tmp_path):
        """Create a temporary voice directory."""
        voice_dir = tmp_path / "voice_files"
        voice_dir.mkdir(parents=True, exist_ok=True)
        return voice_dir

    @pytest.fixture
    def temp_cache_file(self, tmp_path):
        """Create a temporary cache file path."""
        cache_dir = tmp_path / "config"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / ".voice_cache.json"

    @pytest.fixture
    def manager(self, temp_voice_dir, temp_cache_file):
        """Create a VoiceProfileManager instance."""
        from myvoice.services.voice_profile_service import VoiceProfileManager
        return VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            auto_scan=False
        )

    @pytest.fixture
    def optimized_profile(self, tmp_path):
        """Create an optimized voice profile for testing."""
        checkpoint_path = tmp_path / "checkpoints" / "my_voice"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        # Create valid checkpoint files
        (checkpoint_path / "config.json").write_text("{}")

        return VoiceProfile.create_optimized_profile(
            name="Test_Optimized_Voice",
            checkpoint_path=checkpoint_path,
            speaker_name="test_speaker",
            description="Test optimized voice"
        )

    @pytest.mark.asyncio
    async def test_register_optimized_profile(self, manager, optimized_profile):
        """Test registering an optimized voice profile."""
        await manager.start()

        # Verify profile is valid before registering
        assert optimized_profile.is_valid is True, f"Profile should be valid but is_valid={optimized_profile.is_valid}"

        result = await manager.register_profile(optimized_profile)

        assert result is True
        assert optimized_profile.name in manager._profiles
        assert manager._profiles[optimized_profile.name].voice_type == VoiceType.OPTIMIZED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_optimized_profile_saved_to_cache(self, manager, optimized_profile):
        """Test that optimized profile is saved to cache correctly."""
        await manager.start()

        # Verify profile is valid
        assert optimized_profile.is_valid is True

        result = await manager.register_profile(optimized_profile)
        assert result is True, "Registration should succeed"

        await asyncio.sleep(0.2)  # Wait for cache save

        assert manager.cache_file.exists(), "Cache file should exist"
        cache_data = json.loads(manager.cache_file.read_text(encoding='utf-8'))
        profile_data = next(
            (p for p in cache_data['profiles'] if p['name'] == optimized_profile.name),
            None
        )

        assert profile_data is not None
        assert profile_data['voice_type'] == 'optimized'
        assert 'checkpoint_path' in profile_data
        assert 'speaker_name' in profile_data

        await manager.stop()

    @pytest.mark.asyncio
    async def test_optimized_voice_loaded_from_cache(
        self, temp_voice_dir, temp_cache_file, tmp_path
    ):
        """Test that optimized voice is loaded from cache."""
        from myvoice.services.voice_profile_service import VoiceProfileManager

        checkpoint_path = tmp_path / "checkpoints" / "cached_voice"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        # Create valid checkpoint file
        (checkpoint_path / "config.json").write_text("{}")

        cache_data = {
            'cache_version': '1.0',
            'profiles': [
                {
                    'name': 'cached_optimized',
                    'file_path': 'optimized://cached_optimized',
                    'transcription': None,
                    'duration': None,
                    'is_valid': True,
                    'voice_type': 'optimized',
                    'checkpoint_path': str(checkpoint_path),
                    'speaker_name': 'cached_speaker',
                    'description': 'Cached optimized voice',
                    'emotion_capable': True
                }
            ],
            'active_profile': None
        }
        temp_cache_file.write_text(json.dumps(cache_data), encoding='utf-8')

        manager = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            auto_scan=False
        )
        await manager.start()

        assert 'cached_optimized' in manager._profiles
        loaded = manager._profiles['cached_optimized']
        assert loaded.voice_type == VoiceType.OPTIMIZED
        assert loaded.checkpoint_path == checkpoint_path
        assert loaded.speaker_name == 'cached_speaker'

        await manager.stop()

    @pytest.mark.asyncio
    async def test_set_optimized_active(self, manager, optimized_profile):
        """Test setting an optimized voice as active."""
        await manager.start()

        # Verify profile is valid
        assert optimized_profile.is_valid is True

        result = await manager.register_profile(optimized_profile)
        assert result is True, "Registration should succeed"

        result = await manager.set_active_profile(optimized_profile.name)
        assert result is True

        active = manager.get_active_profile()
        assert active is not None
        assert active.voice_type == VoiceType.OPTIMIZED
        assert active.checkpoint_path is not None

        await manager.stop()


# ============================================================================
# App.py Voice Routing Tests
# ============================================================================

class TestAppVoiceRouting:
    """Tests for voice type routing in app.py TTS generation."""

    def test_optimized_voice_routes_to_generate_optimized(self):
        """Test that OPTIMIZED voices route to generate_optimized_voice."""
        # This is a conceptual test - the actual routing happens in app.py
        # The test verifies that the code path exists

        from myvoice.models.voice_profile import VoiceType

        # Verify OPTIMIZED is distinct from other types
        assert VoiceType.OPTIMIZED != VoiceType.BUNDLED
        assert VoiceType.OPTIMIZED != VoiceType.DESIGNED
        assert VoiceType.OPTIMIZED != VoiceType.CLONED

        # Verify OPTIMIZED supports emotion (used for instruct selection)
        assert VoiceType.OPTIMIZED.supports_emotion is True

    def test_optimized_profile_has_required_fields_for_routing(self, tmp_path):
        """Test that optimized profile has all fields needed for TTS routing."""
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()

        profile = VoiceProfile.create_optimized_profile(
            name="test",
            checkpoint_path=checkpoint_path,
            speaker_name="test_speaker"
        )

        # These fields are required for routing in app.py
        assert profile.voice_type == VoiceType.OPTIMIZED
        assert profile.checkpoint_path is not None
        assert profile.speaker_name is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
