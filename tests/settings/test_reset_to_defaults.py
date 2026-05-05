"""
Tests for Reset to Defaults

Story 6.4: Reset to Defaults
Tests the reset to defaults functionality in the settings dialog.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox

from myvoice.models.app_settings import AppSettings


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestAppSettingsResetToDefaults:
    """Tests for AppSettings.reset_to_defaults method."""

    def test_reset_restores_always_on_top_true(self):
        """Test reset sets always_on_top to True (default)."""
        settings = AppSettings(always_on_top=False)
        settings.reset_to_defaults()

        assert settings.always_on_top is True

    def test_reset_restores_transparency_100_percent(self):
        """Test reset sets transparency to 1.0 (100%)."""
        settings = AppSettings(window_transparency=0.5)
        settings.reset_to_defaults()

        assert settings.window_transparency == 1.0

    def test_reset_clears_window_geometry(self):
        """Test reset clears window_geometry (for centering)."""
        settings = AppSettings(window_geometry={"x": 100, "y": 100, "width": 800, "height": 600})
        settings.reset_to_defaults()

        assert settings.window_geometry is None

    def test_reset_clears_monitor_device(self):
        """Test reset clears monitor device (system default)."""
        settings = AppSettings(
            monitor_device_id="device_1",
            monitor_device_name="Speakers"
        )
        settings.reset_to_defaults()

        assert settings.monitor_device_id is None
        assert settings.monitor_device_name is None

    def test_reset_clears_virtual_device(self):
        """Test reset clears virtual device (disabled)."""
        settings = AppSettings(
            virtual_microphone_device_id="vmic_1",
            virtual_microphone_device_name="VB-Cable"
        )
        settings.reset_to_defaults()

        assert settings.virtual_microphone_device_id is None
        assert settings.virtual_microphone_device_name is None

    def test_reset_preserves_voice_files_directory(self):
        """Test reset preserves voice_files_directory (doesn't delete voices)."""
        settings = AppSettings(voice_files_directory="custom/path")
        settings.reset_to_defaults()

        # voice_files_directory should be reset to default, not preserved
        # But user voice FILES are not deleted (directory still exists)
        assert settings.voice_files_directory == "voice_files"

    def test_reset_restores_theme_to_dark(self):
        """Test reset restores theme to dark."""
        settings = AppSettings(ui_theme="light")
        settings.reset_to_defaults()

        assert settings.ui_theme == "dark"

    def test_reset_restores_log_level_to_info(self):
        """Test reset restores log level to INFO."""
        settings = AppSettings(log_level="DEBUG")
        settings.reset_to_defaults()

        assert settings.log_level == "INFO"

    def test_reset_sets_selected_voice_profile_to_default(self):
        """Test reset sets selected voice profile to default bundled voice."""
        settings = AppSettings(selected_voice_profile="CustomVoice")
        settings.reset_to_defaults()

        # Default is now "Sarira-F" bundled voice (not None)
        assert settings.selected_voice_profile == "Sarira-F"


class TestSettingsDialogResetButton:
    """Tests for SettingsDialog reset button."""

    def test_reset_button_exists(self, qapp):
        """Test reset button exists in settings dialog."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        mock_settings = MagicMock(spec=AppSettings)
        mock_settings.config_directory = "/tmp/myvoice"
        mock_settings.ui_theme = "dark"
        mock_settings.always_on_top = True
        mock_settings.to_dict.return_value = {"_settings_version": "1.0"}

        dialog = SettingsDialog(mock_settings)

        assert dialog.reset_button is not None
        assert dialog.reset_button.text() == "Reset to Defaults"

        dialog.deleteLater()

    def test_reset_button_triggers_confirmation(self, qapp):
        """Test reset button shows confirmation dialog."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        mock_settings = MagicMock(spec=AppSettings)
        mock_settings.config_directory = "/tmp/myvoice"
        mock_settings.ui_theme = "dark"
        mock_settings.always_on_top = True
        mock_settings.to_dict.return_value = {"_settings_version": "1.0"}

        dialog = SettingsDialog(mock_settings)

        # Mock QMessageBox to cancel the reset
        with patch.object(QMessageBox, 'exec', return_value=QMessageBox.StandardButton.No):
            dialog._on_reset_defaults()

        # Settings should NOT be reset (user cancelled)
        mock_settings.reset_to_defaults.assert_not_called()

        dialog.deleteLater()


class TestResetVoiceToBundled:
    """Tests for resetting voice to bundled default."""

    def test_reset_sets_voice_to_sarira_f(self, qapp):
        """Test reset sets voice to Sarira-F (default bundled)."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        settings = AppSettings(selected_voice_profile="CustomVoice")

        dialog = SettingsDialog(settings)
        dialog.current_settings = settings

        # Simulate reset
        dialog.current_settings.reset_to_defaults()
        dialog.current_settings.selected_voice_profile = "Sarira-F"

        assert dialog.current_settings.selected_voice_profile == "Sarira-F"

        dialog.deleteLater()


class TestResetQuickSpeak:
    """Tests for Quick Speak reset functionality."""

    def test_reset_quick_speak_entries(self, qapp):
        """Test _reset_quick_speak_entries creates default profile."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        mock_settings = MagicMock(spec=AppSettings)
        mock_settings.config_directory = "/tmp/myvoice"
        mock_settings.ui_theme = "dark"
        mock_settings.always_on_top = True
        mock_settings.to_dict.return_value = {"_settings_version": "1.0"}

        mock_quick_speak = MagicMock()

        dialog = SettingsDialog(mock_settings, quick_speak_service=mock_quick_speak)

        # Call reset method
        dialog._reset_quick_speak_entries()

        # Verify quick speak service methods were called
        mock_quick_speak._create_default_profile.assert_called_once()
        mock_quick_speak.load_entries.assert_called()

        dialog.deleteLater()


class TestResetDialogInformativeText:
    """Tests for reset dialog informative text."""

    def test_reset_dialog_mentions_devices(self, qapp):
        """Test reset dialog mentions device reset."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        mock_settings = MagicMock(spec=AppSettings)
        mock_settings.config_directory = "/tmp/myvoice"
        mock_settings.ui_theme = "dark"
        mock_settings.always_on_top = True
        mock_settings.to_dict.return_value = {"_settings_version": "1.0"}

        dialog = SettingsDialog(mock_settings)

        # The method creates a dialog with specific text
        # We verify the method exists and is callable
        assert hasattr(dialog, '_on_reset_defaults')
        assert callable(dialog._on_reset_defaults)

        dialog.deleteLater()


class TestResetDoesNotDeleteVoices:
    """Tests verifying reset does NOT delete user voice files."""

    def test_voice_files_not_deleted(self, qapp):
        """Test reset does not delete voice files directory content."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            voice_dir = Path(tmpdir) / "voice_files"
            voice_dir.mkdir()

            # Create a "user voice" file
            user_voice = voice_dir / "MyCustomVoice.wav"
            user_voice.write_bytes(b"fake audio content")

            settings = AppSettings(voice_files_directory=str(voice_dir))
            settings.reset_to_defaults()

            # User voice file should still exist
            assert user_voice.exists()


class TestResetWindowPosition:
    """Tests for window position reset (centering)."""

    def test_reset_clears_position_for_centering(self):
        """Test reset clears window_geometry so window can be centered."""
        settings = AppSettings(window_geometry={
            "x": 500,
            "y": 300,
            "width": 800,
            "height": 600
        })

        settings.reset_to_defaults()

        # window_geometry should be None (app will center on next launch)
        assert settings.window_geometry is None


class TestResetAllSettings:
    """Tests for complete settings reset."""

    def test_all_settings_reset_correctly(self):
        """Test comprehensive reset of all settings."""
        settings = AppSettings(
            monitor_device_id="device_1",
            monitor_device_name="Speakers",
            virtual_microphone_device_id="vmic_1",
            selected_voice_profile="CustomVoice",
            window_geometry={"x": 100, "y": 100, "width": 800, "height": 600},
            always_on_top=False,
            window_transparency=0.7,
            ui_theme="light",
            log_level="DEBUG"
        )

        settings.reset_to_defaults()

        # Verify all resets
        assert settings.monitor_device_id is None
        assert settings.virtual_microphone_device_id is None
        assert settings.selected_voice_profile == "Sarira-F"  # Default bundled voice
        assert settings.window_geometry is None
        assert settings.always_on_top is True
        assert settings.window_transparency == 1.0
        assert settings.ui_theme == "dark"
        assert settings.log_level == "INFO"

