"""
Tests for VoiceProfileManager Service

Tests for voice profile management including:
- Profile registration (register_profile method)
- Cache save/load with voice_type preservation
- Bundled voice re-initialization after full rescan
"""

import asyncio
import pytest
import json
import tempfile
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path

# Skip this module if PyQt6 is not available (required for myvoice imports)
pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")

from myvoice.services.voice_profile_service import VoiceProfileManager
from myvoice.models.voice_profile import VoiceProfile, VoiceType, BUNDLED_SPEAKERS


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
def manager(temp_voice_dir, temp_cache_file):
    """Create a VoiceProfileManager instance with temporary directories."""
    return VoiceProfileManager(
        voice_directory=temp_voice_dir,
        cache_file=temp_cache_file,
        auto_scan=False  # Disable auto scan for tests
    )


@pytest.fixture
def sample_wav_file(temp_voice_dir):
    """Create a sample WAV file for testing."""
    # Create a minimal valid WAV file (44-byte header + some data)
    wav_file = temp_voice_dir / "test_voice.wav"

    # Minimal WAV header for a 1-second mono 16-bit 44100Hz file
    import struct

    sample_rate = 44100
    num_channels = 1
    bits_per_sample = 16
    num_samples = sample_rate  # 1 second of audio
    data_size = num_samples * num_channels * (bits_per_sample // 8)

    # WAV header
    header = b'RIFF'
    header += struct.pack('<I', 36 + data_size)  # File size - 8
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)  # Subchunk1Size
    header += struct.pack('<H', 1)   # AudioFormat (PCM)
    header += struct.pack('<H', num_channels)
    header += struct.pack('<I', sample_rate)
    header += struct.pack('<I', sample_rate * num_channels * (bits_per_sample // 8))  # ByteRate
    header += struct.pack('<H', num_channels * (bits_per_sample // 8))  # BlockAlign
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', data_size)

    # Write header + silence data
    wav_file.write_bytes(header + b'\x00' * data_size)

    return wav_file


@pytest.fixture
def cloned_voice_profile(sample_wav_file):
    """Create a cloned voice profile for testing."""
    return VoiceProfile(
        file_path=sample_wav_file,
        name="Test_Cloned_Voice",
        transcription="Hello, this is a test.",
        duration=1.0,
        is_valid=True,
        voice_type=VoiceType.CLONED,
        emotion_capable=False
    )


@pytest.fixture
def designed_voice_profile(sample_wav_file, temp_voice_dir):
    """Create a designed voice profile for testing."""
    # Copy the sample wav for the designed voice to have its own file
    import shutil
    designed_path = temp_voice_dir / "designed_voice.wav"
    shutil.copy(sample_wav_file, designed_path)

    return VoiceProfile(
        file_path=designed_path,
        name="Test_Designed_Voice",
        transcription=None,
        duration=1.0,  # Match the sample WAV duration
        is_valid=True,
        voice_type=VoiceType.DESIGNED,
        emotion_capable=True  # Designed voices support emotions
    )


class TestVoiceProfileManagerInit:
    """Tests for VoiceProfileManager initialization."""

    def test_init_with_defaults(self, temp_voice_dir, temp_cache_file):
        """Test initialization with default parameters."""
        manager = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file
        )
        assert manager.voice_directory == temp_voice_dir
        assert manager.cache_file == temp_cache_file
        assert manager.max_duration == 10.0
        assert manager.auto_scan is True
        assert len(manager._profiles) == 0

    def test_init_with_custom_max_duration(self, temp_voice_dir, temp_cache_file):
        """Test initialization with custom max duration."""
        manager = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            max_duration=15.0
        )
        assert manager.max_duration == 15.0


class TestRegisterProfile:
    """Tests for the register_profile method."""

    @pytest.mark.asyncio
    async def test_register_valid_cloned_profile(self, manager, cloned_voice_profile):
        """Test registering a valid cloned voice profile."""
        # Start the manager
        await manager.start()

        # Register the profile
        result = await manager.register_profile(cloned_voice_profile)

        assert result is True
        assert cloned_voice_profile.name in manager._profiles
        assert manager._profiles[cloned_voice_profile.name] == cloned_voice_profile

        await manager.stop()

    @pytest.mark.asyncio
    async def test_register_valid_designed_profile(self, manager, designed_voice_profile):
        """Test registering a valid designed voice profile."""
        await manager.start()

        result = await manager.register_profile(designed_voice_profile)

        assert result is True
        assert designed_voice_profile.name in manager._profiles
        assert manager._profiles[designed_voice_profile.name].voice_type == VoiceType.DESIGNED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_register_invalid_profile_rejected(self, manager, temp_voice_dir):
        """Test that invalid profiles are rejected."""
        await manager.start()

        # Create an invalid profile
        invalid_profile = VoiceProfile(
            file_path=temp_voice_dir / "nonexistent.wav",
            name="Invalid_Voice",
            is_valid=False
        )

        result = await manager.register_profile(invalid_profile)

        assert result is False
        assert "Invalid_Voice" not in manager._profiles

        await manager.stop()

    @pytest.mark.asyncio
    async def test_register_profile_saves_to_cache(self, manager, cloned_voice_profile):
        """Test that registering a profile triggers cache save."""
        await manager.start()

        # Register the profile
        await manager.register_profile(cloned_voice_profile)

        # Wait a moment for cache save
        await asyncio.sleep(0.1)

        # Verify cache file exists and contains the profile
        assert manager.cache_file.exists()

        cache_data = json.loads(manager.cache_file.read_text(encoding='utf-8'))
        profile_names = [p['name'] for p in cache_data['profiles']]
        assert cloned_voice_profile.name in profile_names

        await manager.stop()

    @pytest.mark.asyncio
    async def test_register_profile_preserves_voice_type(self, manager, cloned_voice_profile):
        """Test that voice_type is preserved when registering profiles."""
        await manager.start()

        # Register the cloned profile
        await manager.register_profile(cloned_voice_profile)

        # Verify the profile in manager has correct voice_type
        registered = manager._profiles[cloned_voice_profile.name]
        assert registered.voice_type == VoiceType.CLONED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_register_profile_logs_type_info(self, manager, cloned_voice_profile, caplog):
        """Test that registration logs voice type information."""
        import logging
        caplog.set_level(logging.INFO)

        await manager.start()
        await manager.register_profile(cloned_voice_profile)

        # Check log contains voice type info
        assert any("cloned" in record.message.lower() for record in caplog.records)

        await manager.stop()


class TestCacheVoiceTypePersistence:
    """Tests for voice_type persistence in cache."""

    @pytest.mark.asyncio
    async def test_cache_saves_voice_type(self, manager, cloned_voice_profile):
        """Test that voice_type is saved to cache."""
        await manager.start()
        await manager.register_profile(cloned_voice_profile)
        await asyncio.sleep(0.1)  # Wait for cache save

        # Read cache directly
        cache_data = json.loads(manager.cache_file.read_text(encoding='utf-8'))

        # Find the profile in cache
        profile_data = next(
            (p for p in cache_data['profiles'] if p['name'] == cloned_voice_profile.name),
            None
        )

        assert profile_data is not None
        assert profile_data['voice_type'] == VoiceType.CLONED.value

        await manager.stop()

    @pytest.mark.asyncio
    async def test_cache_loads_voice_type(self, temp_voice_dir, temp_cache_file, sample_wav_file):
        """Test that voice_type is loaded from cache."""
        # Create cache with voice_type
        cache_data = {
            'cache_version': '1.0',
            'profiles': [
                {
                    'name': 'test_voice',
                    'file_path': str(sample_wav_file),
                    'transcription': 'Hello',
                    'duration': 1.0,
                    'is_valid': True,
                    'voice_type': 'cloned',  # This should be loaded
                    'file_mtime': sample_wav_file.stat().st_mtime,
                    'file_size': sample_wav_file.stat().st_size
                }
            ],
            'active_profile': None
        }
        temp_cache_file.write_text(json.dumps(cache_data), encoding='utf-8')

        # Create new manager that will load from cache
        manager = VoiceProfileManager(
            voice_directory=temp_voice_dir,
            cache_file=temp_cache_file,
            auto_scan=False
        )
        await manager.start()

        # Verify profile was loaded with correct voice_type
        assert 'test_voice' in manager._profiles
        loaded_profile = manager._profiles['test_voice']
        assert loaded_profile.voice_type == VoiceType.CLONED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_cache_defaults_to_cloned_without_voice_type(
        self, temp_voice_dir, temp_cache_file, sample_wav_file
    ):
        """Test that missing voice_type defaults to CLONED."""
        # Create cache without voice_type field
        cache_data = {
            'cache_version': '1.0',
            'profiles': [
                {
                    'name': 'legacy_voice',
                    'file_path': str(sample_wav_file),
                    'transcription': 'Hello',
                    'duration': 1.0,
                    'is_valid': True,
                    # Note: no voice_type field
                    'file_mtime': sample_wav_file.stat().st_mtime,
                    'file_size': sample_wav_file.stat().st_size
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

        # Verify profile defaults to CLONED
        assert 'legacy_voice' in manager._profiles
        assert manager._profiles['legacy_voice'].voice_type == VoiceType.CLONED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_designed_voice_type_persisted(self, manager, designed_voice_profile):
        """Test that DESIGNED voice type is persisted correctly."""
        await manager.start()
        await manager.register_profile(designed_voice_profile)
        await asyncio.sleep(0.1)

        # Read cache
        cache_data = json.loads(manager.cache_file.read_text(encoding='utf-8'))
        profile_data = next(
            (p for p in cache_data['profiles'] if p['name'] == designed_voice_profile.name),
            None
        )

        assert profile_data is not None
        assert profile_data['voice_type'] == 'designed'

        await manager.stop()


class TestBundledVoiceReinitialization:
    """Tests for bundled voice re-initialization after full rescan."""

    @pytest.mark.asyncio
    async def test_bundled_voices_initialized_on_start(self, manager):
        """Test that bundled voices are initialized when manager starts."""
        await manager.start()

        # All bundled speakers should be present
        for speaker in BUNDLED_SPEAKERS:
            assert speaker in manager._profiles, f"Bundled speaker {speaker} should be initialized"
            profile = manager._profiles[speaker]
            assert profile.voice_type == VoiceType.BUNDLED
            assert profile.emotion_capable is True
            assert profile.is_valid is True

        await manager.stop()

    @pytest.mark.asyncio
    async def test_bundled_voices_preserved_after_incremental_scan(self, manager):
        """Test that bundled voices are preserved during incremental scans."""
        await manager.start()

        # Get initial bundled count
        initial_bundled = manager.get_bundled_profiles()
        assert len(initial_bundled) == len(BUNDLED_SPEAKERS)

        # Perform incremental scan
        await manager.scan_voice_directory(incremental=True)

        # Bundled voices should still be present
        bundled_after = manager.get_bundled_profiles()
        assert len(bundled_after) == len(BUNDLED_SPEAKERS)

        await manager.stop()

    @pytest.mark.asyncio
    async def test_bundled_voices_reinitialized_after_full_rescan(self, manager):
        """Test that bundled voices are re-initialized after full rescan (incremental=False)."""
        await manager.start()

        # Verify bundled voices exist
        initial_bundled = manager.get_bundled_profiles()
        assert len(initial_bundled) == len(BUNDLED_SPEAKERS)

        # Perform full rescan (this clears _profiles)
        result = await manager.scan_voice_directory(incremental=False)

        # Bundled voices should be re-initialized
        assert result['success'] is True
        bundled_after = manager.get_bundled_profiles()
        assert len(bundled_after) == len(BUNDLED_SPEAKERS)

        # Verify each bundled profile is valid
        for speaker in BUNDLED_SPEAKERS:
            assert speaker in manager._profiles
            assert manager._profiles[speaker].voice_type == VoiceType.BUNDLED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_force_rescan_preserves_bundled_voices(self, manager):
        """Test that force_rescan() also re-initializes bundled voices."""
        await manager.start()

        # Perform force rescan
        result = await manager.force_rescan()

        assert result['success'] is True

        # Bundled voices should be present
        bundled = manager.get_bundled_profiles()
        assert len(bundled) == len(BUNDLED_SPEAKERS)

        await manager.stop()

    @pytest.mark.asyncio
    async def test_bundled_voices_not_duplicated(self, manager):
        """Test that bundled voices are not duplicated on multiple initializations."""
        await manager.start()

        # Count bundled voices
        count1 = len(manager.get_bundled_profiles())

        # Manually call _initialize_bundled_voices again
        manager._initialize_bundled_voices()

        # Should still be the same count
        count2 = len(manager.get_bundled_profiles())
        assert count1 == count2

        await manager.stop()

    @pytest.mark.asyncio
    async def test_bundled_voices_skipped_in_cache_load(
        self, temp_voice_dir, temp_cache_file
    ):
        """Test that bundled voice entries in cache are skipped during load."""
        # Create cache with a bundled voice entry (using bundled:// path)
        cache_data = {
            'cache_version': '1.0',
            'profiles': [
                {
                    'name': 'Vivian',
                    'file_path': 'bundled://Vivian',  # Bundled path format
                    'transcription': None,
                    'duration': None,
                    'is_valid': True,
                    'voice_type': 'bundled'
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

        # Vivian should exist (from _initialize_bundled_voices, not from cache)
        assert 'Vivian' in manager._profiles
        # It should be a proper bundled profile
        assert manager._profiles['Vivian'].voice_type == VoiceType.BUNDLED

        await manager.stop()


class TestGetBundledProfiles:
    """Tests for get_bundled_profiles method."""

    @pytest.mark.asyncio
    async def test_returns_only_bundled(self, manager, cloned_voice_profile):
        """Test that get_bundled_profiles returns only bundled voices."""
        await manager.start()

        # Add a cloned profile
        await manager.register_profile(cloned_voice_profile)

        # Get bundled profiles
        bundled = manager.get_bundled_profiles()

        # Should only contain bundled voices
        assert len(bundled) == len(BUNDLED_SPEAKERS)
        for name, profile in bundled.items():
            assert profile.voice_type == VoiceType.BUNDLED

        # Cloned profile should not be in bundled
        assert cloned_voice_profile.name not in bundled

        await manager.stop()


class TestCacheEmotionCapable:
    """Tests for emotion_capable field persistence."""

    @pytest.mark.asyncio
    async def test_emotion_capable_saved_to_cache(self, manager, designed_voice_profile):
        """Test that emotion_capable is saved to cache."""
        await manager.start()
        await manager.register_profile(designed_voice_profile)
        await asyncio.sleep(0.1)

        cache_data = json.loads(manager.cache_file.read_text(encoding='utf-8'))
        profile_data = next(
            (p for p in cache_data['profiles'] if p['name'] == designed_voice_profile.name),
            None
        )

        assert profile_data is not None
        assert profile_data['emotion_capable'] is True

        await manager.stop()

    @pytest.mark.asyncio
    async def test_cloned_voice_not_emotion_capable(self, manager, cloned_voice_profile):
        """Test that cloned voices have emotion_capable=False."""
        await manager.start()
        await manager.register_profile(cloned_voice_profile)

        registered = manager._profiles[cloned_voice_profile.name]
        assert registered.emotion_capable is False

        await manager.stop()


class TestActiveProfileWithVoiceType:
    """Tests for active profile with different voice types."""

    @pytest.mark.asyncio
    async def test_set_bundled_active(self, manager):
        """Test setting a bundled voice as active."""
        await manager.start()

        # Set a bundled voice as active
        result = await manager.set_active_profile("Vivian")
        assert result is True

        active = manager.get_active_profile()
        assert active is not None
        assert active.name == "Vivian"
        assert active.voice_type == VoiceType.BUNDLED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_set_cloned_active(self, manager, cloned_voice_profile):
        """Test setting a cloned voice as active."""
        await manager.start()
        await manager.register_profile(cloned_voice_profile)

        result = await manager.set_active_profile(cloned_voice_profile.name)
        assert result is True

        active = manager.get_active_profile()
        assert active.voice_type == VoiceType.CLONED

        await manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
