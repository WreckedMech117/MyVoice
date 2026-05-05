"""
Tests for User Settings Persistence

Story 6.2: User Settings Persistence
Tests that user settings are properly persisted to disk and recovered on application restart.
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


class TestAppSettingsWindowTransparency:
    """Tests for window_transparency field in AppSettings."""

    def test_default_transparency_is_fully_opaque(self):
        """Test default window transparency is 1.0 (fully opaque)."""
        settings = AppSettings()
        assert settings.window_transparency == 1.0

    def test_transparency_can_be_set(self):
        """Test window transparency can be set to valid values."""
        settings = AppSettings(window_transparency=0.5)
        assert settings.window_transparency == 0.5

    def test_transparency_validation_rejects_negative(self):
        """Test validation fails for negative transparency."""
        settings = AppSettings(window_transparency=-0.1)
        result = settings.validate()

        # Should have validation issue
        transparency_issues = [i for i in result.issues if i.field == "window_transparency"]
        assert len(transparency_issues) > 0
        assert transparency_issues[0].code == "INVALID_TRANSPARENCY_RANGE"

    def test_transparency_validation_rejects_over_one(self):
        """Test validation fails for transparency > 1.0."""
        settings = AppSettings(window_transparency=1.5)
        result = settings.validate()

        transparency_issues = [i for i in result.issues if i.field == "window_transparency"]
        assert len(transparency_issues) > 0
        assert transparency_issues[0].code == "INVALID_TRANSPARENCY_RANGE"

    def test_transparency_zero_is_valid(self):
        """Test transparency of 0.0 (fully transparent) is valid."""
        settings = AppSettings(window_transparency=0.0)
        result = settings.validate()

        transparency_issues = [i for i in result.issues if i.field == "window_transparency"]
        assert len(transparency_issues) == 0

    def test_transparency_one_is_valid(self):
        """Test transparency of 1.0 (fully opaque) is valid."""
        settings = AppSettings(window_transparency=1.0)
        result = settings.validate()

        transparency_issues = [i for i in result.issues if i.field == "window_transparency"]
        assert len(transparency_issues) == 0


class TestAppSettingsSerialization:
    """Tests for AppSettings serialization with new fields."""

    def test_to_dict_includes_window_transparency(self):
        """Test to_dict includes window_transparency."""
        settings = AppSettings(window_transparency=0.8)
        data = settings.to_dict()

        assert "window_transparency" in data
        assert data["window_transparency"] == 0.8

    def test_from_dict_restores_window_transparency(self):
        """Test from_dict restores window_transparency."""
        data = {
            "window_transparency": 0.7,
            "_settings_version": "1.0"
        }
        settings = AppSettings.from_dict(data)

        assert settings.window_transparency == 0.7

    def test_from_dict_uses_default_for_missing_transparency(self):
        """Test from_dict uses default when window_transparency is missing."""
        data = {"_settings_version": "1.0"}
        settings = AppSettings.from_dict(data)

        assert settings.window_transparency == 1.0

    def test_to_dict_includes_window_geometry(self):
        """Test to_dict includes window_geometry."""
        settings = AppSettings(window_geometry={"x": 100, "y": 200, "width": 800, "height": 600})
        data = settings.to_dict()

        assert "window_geometry" in data
        assert data["window_geometry"]["x"] == 100
        assert data["window_geometry"]["y"] == 200
        assert data["window_geometry"]["width"] == 800
        assert data["window_geometry"]["height"] == 600

    def test_to_dict_includes_all_persisted_fields(self):
        """Test to_dict includes all fields required by FR41."""
        settings = AppSettings(
            monitor_device_id="device_1",
            virtual_microphone_device_id="vmic_1",
            selected_voice_profile="TestVoice",
            window_geometry={"x": 0, "y": 0, "width": 400, "height": 300},
            always_on_top=True,
            window_transparency=0.9
        )
        data = settings.to_dict()

        # Verify all FR41 required fields are present
        assert "monitor_device_id" in data
        assert "virtual_microphone_device_id" in data
        assert "selected_voice_profile" in data
        assert "window_geometry" in data
        assert "always_on_top" in data
        assert "window_transparency" in data


class TestAppSettingsResetDefaults:
    """Tests for AppSettings reset_to_defaults with window_transparency."""

    def test_reset_to_defaults_resets_transparency(self):
        """Test reset_to_defaults restores window_transparency to 1.0."""
        settings = AppSettings(window_transparency=0.5)
        assert settings.window_transparency == 0.5

        settings.reset_to_defaults()
        assert settings.window_transparency == 1.0

    def test_reset_to_defaults_clears_window_geometry(self):
        """Test reset_to_defaults clears window_geometry."""
        settings = AppSettings(window_geometry={"x": 100, "y": 100, "width": 800, "height": 600})
        assert settings.window_geometry is not None

        settings.reset_to_defaults()
        assert settings.window_geometry is None


class TestConfigurationManagerWindowSettings:
    """Tests for ConfigurationManager window settings methods."""

    @pytest.fixture
    def config_manager(self):
        """Create ConfigurationManager with temp file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "settings.json"
            manager = ConfigurationManager(config_file=config_file, auto_save=False)
            manager._settings = AppSettings()
            yield manager

    @pytest.mark.asyncio
    async def test_update_window_geometry(self, config_manager):
        """Test update_window_geometry stores geometry."""
        result = await config_manager.update_window_geometry(100, 200, 800, 600)

        assert result is True
        assert config_manager._settings.window_geometry == {
            "x": 100,
            "y": 200,
            "width": 800,
            "height": 600
        }

    @pytest.mark.asyncio
    async def test_update_window_transparency(self, config_manager):
        """Test update_window_transparency stores transparency."""
        result = await config_manager.update_window_transparency(0.75)

        assert result is True
        assert config_manager._settings.window_transparency == 0.75

    @pytest.mark.asyncio
    async def test_update_window_transparency_rejects_invalid(self, config_manager):
        """Test update_window_transparency rejects invalid values."""
        result = await config_manager.update_window_transparency(1.5)

        assert result is False
        assert config_manager._settings.window_transparency == 1.0  # Unchanged

    @pytest.mark.asyncio
    async def test_update_always_on_top(self, config_manager):
        """Test update_always_on_top stores setting."""
        result = await config_manager.update_always_on_top(False)

        assert result is True
        assert config_manager._settings.always_on_top is False

    def test_get_window_geometry(self, config_manager):
        """Test get_window_geometry retrieves stored geometry."""
        config_manager._settings.window_geometry = {"x": 50, "y": 75, "width": 640, "height": 480}

        geometry = config_manager.get_window_geometry()

        assert geometry == {"x": 50, "y": 75, "width": 640, "height": 480}

    def test_get_window_geometry_returns_none_when_not_set(self, config_manager):
        """Test get_window_geometry returns None when not set."""
        config_manager._settings.window_geometry = None

        geometry = config_manager.get_window_geometry()

        assert geometry is None

    def test_get_window_transparency(self, config_manager):
        """Test get_window_transparency retrieves stored value."""
        config_manager._settings.window_transparency = 0.8

        transparency = config_manager.get_window_transparency()

        assert transparency == 0.8

    def test_get_always_on_top(self, config_manager):
        """Test get_always_on_top retrieves stored value."""
        config_manager._settings.always_on_top = False

        always_on_top = config_manager.get_always_on_top()

        assert always_on_top is False


class TestSettingsPersistenceToFile:
    """Tests for settings persistence to JSON file."""

    @pytest.mark.asyncio
    async def test_save_and_load_preserves_all_settings(self):
        """Test save and load preserves all FR41 settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "settings.json"

            # Create and save settings
            manager = ConfigurationManager(config_file=config_file, auto_save=True)
            manager._settings = AppSettings(
                monitor_device_id="monitor_1",
                monitor_device_name="Speakers",
                virtual_microphone_device_id="vmic_1",
                virtual_microphone_device_name="VB-Cable",
                selected_voice_profile="MyVoice",
                window_geometry={"x": 100, "y": 100, "width": 800, "height": 600},
                always_on_top=False,
                window_transparency=0.85
            )

            # Initialize executor for save
            from concurrent.futures import ThreadPoolExecutor
            manager._executor = ThreadPoolExecutor(max_workers=2)

            await manager.save_settings()

            # Verify file was created
            assert config_file.exists()

            # Read raw JSON to verify structure
            with open(config_file, 'r') as f:
                data = json.load(f)

            assert data["monitor_device_id"] == "monitor_1"
            assert data["virtual_microphone_device_id"] == "vmic_1"
            assert data["selected_voice_profile"] == "MyVoice"
            assert data["window_geometry"]["x"] == 100
            assert data["always_on_top"] is False
            assert data["window_transparency"] == 0.85

            manager._executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_load_settings_from_file(self):
        """Test loading settings from existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "settings.json"

            # Create settings file manually
            settings_data = {
                "monitor_device_id": "device_2",
                "virtual_microphone_device_id": "vmic_2",
                "selected_voice_profile": "LoadedVoice",
                "window_geometry": {"x": 200, "y": 150, "width": 1024, "height": 768},
                "always_on_top": True,
                "window_transparency": 0.9,
                "_settings_version": "1.0"
            }

            with open(config_file, 'w') as f:
                json.dump(settings_data, f)

            # Load settings
            manager = ConfigurationManager(config_file=config_file, auto_save=False)
            from concurrent.futures import ThreadPoolExecutor
            manager._executor = ThreadPoolExecutor(max_workers=2)

            await manager.load_settings()

            # Verify all settings were loaded
            assert manager._settings.monitor_device_id == "device_2"
            assert manager._settings.virtual_microphone_device_id == "vmic_2"
            assert manager._settings.selected_voice_profile == "LoadedVoice"
            assert manager._settings.window_geometry["x"] == 200
            assert manager._settings.always_on_top is True
            assert manager._settings.window_transparency == 0.9

            manager._executor.shutdown(wait=True)


class TestCorruptedSettingsRecovery:
    """Tests for graceful recovery from corrupted settings (FR41)."""

    @pytest.mark.asyncio
    async def test_load_corrupted_file_uses_defaults(self):
        """Test loading corrupted file falls back to defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "settings.json"

            # Create corrupted file
            with open(config_file, 'w') as f:
                f.write("{ this is not valid json }")

            manager = ConfigurationManager(config_file=config_file, auto_save=False)
            from concurrent.futures import ThreadPoolExecutor
            manager._executor = ThreadPoolExecutor(max_workers=2)

            settings = await manager.load_settings()

            # Should load default settings
            assert settings is not None
            assert settings.window_transparency == 1.0  # Default
            assert settings.always_on_top is True  # Default

            manager._executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_missing_file_creates_defaults(self):
        """Test missing file creates default settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "nonexistent" / "settings.json"

            manager = ConfigurationManager(config_file=config_file, auto_save=False)
            from concurrent.futures import ThreadPoolExecutor
            manager._executor = ThreadPoolExecutor(max_workers=2)

            settings = await manager.load_settings()

            # Should create default settings
            assert settings is not None
            assert settings.window_transparency == 1.0
            assert settings.always_on_top is True

            manager._executor.shutdown(wait=True)


class TestEmotionNotPersisted:
    """Tests verifying emotion is NOT persisted (defaults to Neutral on launch)."""

    def test_app_settings_has_no_emotion_field(self):
        """Test AppSettings does not persist emotion selection."""
        settings = AppSettings()
        data = settings.to_dict()

        # Verify no emotion_preset or similar field
        assert "emotion_preset" not in data
        assert "selected_emotion" not in data
        assert "current_emotion" not in data

    def test_custom_emotion_text_is_persisted(self):
        """Test custom_emotion_text is persisted (but not active emotion selection)."""
        # Note: custom_emotion_text is for custom preset text, not current selection
        settings = AppSettings(custom_emotion_text="Custom emotion instruct")
        data = settings.to_dict()

        # This is the custom text input, NOT the current emotion selection
        assert "custom_emotion_text" in data


class TestAutoFixTransparency:
    """Tests for auto-fixing invalid transparency values."""

    @pytest.fixture
    def config_manager(self):
        """Create ConfigurationManager for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "settings.json"
            manager = ConfigurationManager(config_file=config_file, auto_save=False)
            yield manager

    @pytest.mark.asyncio
    async def test_auto_fix_invalid_transparency(self, config_manager):
        """Test auto-fix resets invalid transparency to 1.0."""
        # Set invalid transparency
        config_manager._settings = AppSettings()
        config_manager._settings.window_transparency = 2.0  # Invalid

        # Apply auto fixes
        success, report = await config_manager.apply_automatic_fixes()

        # Should fix the transparency
        assert config_manager._settings.window_transparency == 1.0
        assert any("window_transparency" in str(fix) for fix in report.get("fixes_applied", []))

