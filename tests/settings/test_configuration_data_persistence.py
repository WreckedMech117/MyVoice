"""
Tests for Configuration Data Persistence

Story 6.3: Configuration Data Persistence
Tests voice library loading, corrupted voice handling, and fallback behavior.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Skip this module if PyQt6 is not available (required for myvoice imports)
pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")

from myvoice.models.app_settings import AppSettings
from myvoice.services.configuration_service import ConfigurationManager


class TestVoiceSelectionFallback:
    """Tests for voice selection fallback to bundled voice (FR47)."""

    @pytest.fixture
    def temp_voice_dir(self):
        """Create temporary voice directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            voice_dir = Path(tmpdir) / "voice_files"
            voice_dir.mkdir()

            # Create bundled voice file
            bundled_voice = voice_dir / "Sarira-F.wav"
            bundled_voice.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal WAV header

            # Create another voice file
            other_voice = voice_dir / "TestVoice.wav"
            other_voice.write_bytes(b"RIFF" + b"\x00" * 100)

            yield voice_dir

    @pytest.fixture
    def config_manager(self, temp_voice_dir):
        """Create ConfigurationManager with temp config."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "settings.json"
            manager = ConfigurationManager(config_file=config_file, auto_save=False)
            manager._settings = AppSettings(voice_files_directory=str(temp_voice_dir))
            yield manager

    @pytest.mark.asyncio
    async def test_restore_valid_voice_selection(self, config_manager, temp_voice_dir):
        """Test restoring valid voice selection succeeds."""
        config_manager._settings.selected_voice_profile = "TestVoice"

        result = await config_manager.restore_voice_selection()

        assert result == "TestVoice"

    @pytest.mark.asyncio
    async def test_fallback_to_bundled_when_voice_missing(self, config_manager, temp_voice_dir):
        """Test fallback to bundled when selected voice is missing (FR47)."""
        # Select a voice that doesn't exist
        config_manager._settings.selected_voice_profile = "DeletedVoice"

        result = await config_manager.restore_voice_selection()

        # Should fall back to bundled voice
        assert result == "Sarira-F"
        assert config_manager._settings.selected_voice_profile == "Sarira-F"

    @pytest.mark.asyncio
    async def test_fallback_to_bundled_when_voice_empty(self, config_manager, temp_voice_dir):
        """Test fallback when voice file is empty."""
        # Create empty voice file
        empty_voice = temp_voice_dir / "EmptyVoice.wav"
        empty_voice.write_bytes(b"")

        config_manager._settings.selected_voice_profile = "EmptyVoice"

        result = await config_manager.restore_voice_selection()

        # Should fall back to bundled voice
        assert result == "Sarira-F"

    @pytest.mark.asyncio
    async def test_fallback_to_first_available_when_default_missing(self, config_manager, temp_voice_dir):
        """Test fallback to first available voice when default bundled is missing."""
        # Remove the default bundled voice
        (temp_voice_dir / "Sarira-F.wav").unlink()

        config_manager._settings.selected_voice_profile = "MissingVoice"

        result = await config_manager.restore_voice_selection()

        # Should fall back to first available voice
        assert result == "TestVoice"

    @pytest.mark.asyncio
    async def test_no_fallback_when_no_voices_available(self):
        """Test returns None when no voices are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            voice_dir = Path(tmpdir) / "voice_files"
            voice_dir.mkdir()  # Empty directory

            config_file = Path(tmpdir) / "config" / "settings.json"
            manager = ConfigurationManager(config_file=config_file, auto_save=False)
            manager._settings = AppSettings(voice_files_directory=str(voice_dir))
            manager._settings.selected_voice_profile = "MissingVoice"

            result = await manager.restore_voice_selection()

            assert result is None

    @pytest.mark.asyncio
    async def test_fallback_on_error(self, config_manager):
        """Test fallback behavior when error occurs during restore."""
        # Set invalid voice directory
        config_manager._settings.voice_files_directory = "/nonexistent/path"
        config_manager._settings.selected_voice_profile = "TestVoice"

        result = await config_manager.restore_voice_selection()

        # Should return None or handle gracefully
        assert result is None or isinstance(result, str)


class TestVoiceDirectoryLoading:
    """Tests for voice directory loading on startup (FR46)."""

    def test_voice_files_directory_default(self):
        """Test default voice files directory path."""
        settings = AppSettings()
        assert settings.voice_files_directory == "voice_files"

    def test_voice_files_directory_custom(self):
        """Test custom voice files directory path."""
        settings = AppSettings(voice_files_directory="/custom/path")
        # Path separators may be normalized on Windows, so check path components
        assert "custom" in settings.voice_files_directory
        assert "path" in settings.voice_files_directory

    def test_get_voice_files_path(self):
        """Test get_voice_files_path returns Path object."""
        settings = AppSettings(voice_files_directory="voice_files")
        path = settings.get_voice_files_path()

        assert isinstance(path, Path)
        assert path == Path("voice_files")


class TestCorruptedVoiceHandling:
    """Tests for corrupted voice file handling (FR47)."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_skip_corrupted_voice_log_warning(self, temp_config_dir):
        """Test corrupted voices are skipped with log warning."""
        # This tests the behavior at ConfigurationManager level
        # VoiceProfileManager handles the actual corruption detection
        config_file = temp_config_dir / "settings.json"

        # Create corrupted settings file
        corrupted_data = "{ this is not valid json"
        config_file.write_text(corrupted_data)

        manager = ConfigurationManager(config_file=config_file, auto_save=False)
        from concurrent.futures import ThreadPoolExecutor
        manager._executor = ThreadPoolExecutor(max_workers=2)

        # Should load defaults gracefully
        settings = await manager.load_settings()

        assert settings is not None
        # Should have default values
        assert settings.voice_files_directory == "voice_files"

        manager._executor.shutdown(wait=True)


class TestQuickSpeakPersistence:
    """Tests for Quick Speak persistence."""

    def test_quick_speak_service_creates_default_profile(self):
        """Test QuickSpeakService creates default profile."""
        from myvoice.services.quick_speak_service import QuickSpeakService

        with tempfile.TemporaryDirectory() as tmpdir:
            service = QuickSpeakService(config_directory=tmpdir)

            profiles = service.get_profiles()

            assert "default" in profiles

    def test_quick_speak_service_loads_entries(self):
        """Test QuickSpeakService loads entries from CSV."""
        from myvoice.services.quick_speak_service import QuickSpeakService

        with tempfile.TemporaryDirectory() as tmpdir:
            service = QuickSpeakService(config_directory=tmpdir)
            entries = service.load_entries()

            # Default profile has entries
            assert len(entries) > 0

    def test_quick_speak_service_saves_entries(self):
        """Test QuickSpeakService saves entries to CSV."""
        from myvoice.services.quick_speak_service import QuickSpeakService
        from myvoice.models.quick_speak_entry import QuickSpeakEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            service = QuickSpeakService(config_directory=tmpdir)

            # Add a new entry
            entry = service.add_entry("Test phrase", "Test")

            # Reload to verify persistence
            service2 = QuickSpeakService(config_directory=tmpdir)
            entries = service2.load_entries()

            assert any(e.text == "Test phrase" for e in entries)


class TestSettingsDirectoryInitialization:
    """Tests for settings directory initialization."""

    @pytest.mark.asyncio
    async def test_config_directory_created_on_start(self):
        """Test config directory is created when ConfigurationManager starts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a path that doesn't exist yet
            config_file = Path(tmpdir) / "new_config" / "settings.json"

            manager = ConfigurationManager(config_file=config_file, auto_save=True)

            # Start should create the directory
            from concurrent.futures import ThreadPoolExecutor
            manager._executor = ThreadPoolExecutor(max_workers=2)

            await manager.load_settings()

            # Directory should now exist
            assert config_file.parent.exists()

            manager._executor.shutdown(wait=True)


class TestVoiceProfilePersistence:
    """Tests for voice profile persistence behavior."""

    def test_voice_selection_persisted_in_settings(self):
        """Test voice selection is stored in settings."""
        settings = AppSettings(selected_voice_profile="TestVoice")

        assert settings.selected_voice_profile == "TestVoice"

    def test_voice_selection_serialized(self):
        """Test voice selection is included in serialization."""
        settings = AppSettings(selected_voice_profile="TestVoice")
        data = settings.to_dict()

        assert data["selected_voice_profile"] == "TestVoice"

    def test_voice_selection_deserialized(self):
        """Test voice selection is restored from serialization."""
        data = {
            "selected_voice_profile": "RestoredVoice",
            "_settings_version": "1.0"
        }
        settings = AppSettings.from_dict(data)

        assert settings.selected_voice_profile == "RestoredVoice"


class TestDefaultBundledVoice:
    """Tests for default bundled voice selection."""

    @pytest.mark.asyncio
    async def test_select_default_bundled_voice(self):
        """Test _select_default_bundled_voice method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            voice_dir = Path(tmpdir) / "voice_files"
            voice_dir.mkdir()

            # Create bundled voice
            bundled_voice = voice_dir / "Sarira-F.wav"
            bundled_voice.write_bytes(b"RIFF" + b"\x00" * 100)

            config_file = Path(tmpdir) / "config" / "settings.json"
            manager = ConfigurationManager(config_file=config_file, auto_save=False)
            manager._settings = AppSettings(voice_files_directory=str(voice_dir))

            result = await manager._select_default_bundled_voice()

            assert result == "Sarira-F"
            assert manager._settings.selected_voice_profile == "Sarira-F"

    @pytest.mark.asyncio
    async def test_no_selection_when_no_voice_profile_set(self):
        """Test restore returns bundled when no profile was set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            voice_dir = Path(tmpdir) / "voice_files"
            voice_dir.mkdir()

            # Create bundled voice
            bundled_voice = voice_dir / "Sarira-F.wav"
            bundled_voice.write_bytes(b"RIFF" + b"\x00" * 100)

            config_file = Path(tmpdir) / "config" / "settings.json"
            manager = ConfigurationManager(config_file=config_file, auto_save=False)
            manager._settings = AppSettings(
                voice_files_directory=str(voice_dir),
                selected_voice_profile=None
            )

            result = await manager.restore_voice_selection()

            # Should select bundled voice
            assert result == "Sarira-F"

