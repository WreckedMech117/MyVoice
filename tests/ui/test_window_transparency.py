"""
Tests for Window Transparency

Story 7.3: Window Transparency
Tests transparency slider in Settings → Interface, real-time preview,
minimum 20% enforcement, and persistence.

FR41: User can adjust window transparency (20%-100%)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QSlider
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from myvoice.models.app_settings import AppSettings
from myvoice.models.validation import ValidationStatus


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_settings():
    """Create mock AppSettings with default transparency."""
    settings = MagicMock(spec=AppSettings)
    settings.config_directory = "/tmp/myvoice"
    settings.ui_theme = "dark"
    settings.always_on_top = True
    settings.minimize_to_tray = True
    settings.tray_notification_shown = False
    settings.log_level = "INFO"
    settings.window_transparency = 1.0  # 100% opaque by default
    settings.tts_service_url = "http://localhost:9880"
    settings.tts_service_timeout = 30
    settings.max_voice_duration = 10.0
    settings.auto_refresh_interval = 30
    settings.voice_files_directory = "/tmp/voices"
    settings.custom_emotion_text = None
    settings.custom_emotion_presets = ["Rising Frustration", "Growing Excitement"]
    settings.to_dict.return_value = {
        "config_directory": "/tmp/myvoice",
        "ui_theme": "dark",
        "always_on_top": True,
        "window_transparency": 1.0,
        "log_level": "INFO"
    }
    return settings


class TestAppSettingsTransparencyValidation:
    """Test AppSettings transparency validation."""

    def test_transparency_default_value(self):
        """AC1: Transparency defaults to 100% (fully opaque)."""
        settings = AppSettings()
        assert settings.window_transparency == 1.0

    def test_transparency_valid_range(self):
        """AC2: Transparency accepts values between 0.0 and 1.0."""
        settings = AppSettings(window_transparency=0.5)
        validation = settings.validate()
        # Should have no errors for the transparency field
        transparency_errors = [
            issue for issue in validation.issues
            if issue.field == "window_transparency"
        ]
        assert len(transparency_errors) == 0

    def test_transparency_minimum_warning(self):
        """AC3: Values below 20% generate a warning."""
        settings = AppSettings(window_transparency=0.1)  # 10%
        validation = settings.validate()
        # Should have a warning for low transparency
        transparency_warnings = [
            warning for warning in validation.warnings
            if warning.field == "window_transparency" and "below 20%" in warning.message
        ]
        assert len(transparency_warnings) == 1

    def test_transparency_20_percent_no_warning(self):
        """AC4: Exactly 20% does not generate a warning."""
        settings = AppSettings(window_transparency=0.2)  # 20%
        validation = settings.validate()
        # Should not have warnings for transparency at 20%
        transparency_warnings = [
            warning for warning in validation.warnings
            if warning.field == "window_transparency"
        ]
        assert len(transparency_warnings) == 0

    def test_transparency_invalid_below_zero(self):
        """AC5: Values below 0.0 are invalid."""
        settings = AppSettings(window_transparency=-0.1)
        validation = settings.validate()
        transparency_errors = [
            issue for issue in validation.issues
            if issue.field == "window_transparency"
        ]
        assert len(transparency_errors) == 1
        assert not validation.is_valid or any(
            issue.field == "window_transparency" for issue in validation.issues
        )

    def test_transparency_invalid_above_one(self):
        """AC6: Values above 1.0 are invalid."""
        settings = AppSettings(window_transparency=1.5)
        validation = settings.validate()
        transparency_errors = [
            issue for issue in validation.issues
            if issue.field == "window_transparency"
        ]
        assert len(transparency_errors) == 1

    def test_transparency_in_to_dict(self):
        """AC7: Transparency is included in serialization."""
        settings = AppSettings(window_transparency=0.75)
        data = settings.to_dict()
        assert "window_transparency" in data
        assert data["window_transparency"] == 0.75

    def test_transparency_from_dict(self):
        """AC8: Transparency is restored from serialization."""
        data = {"window_transparency": 0.6}
        settings = AppSettings.from_dict(data)
        assert settings.window_transparency == 0.6


class TestMainWindowTransparency:
    """Test MainWindow transparency methods."""

    def test_main_window_has_set_transparency_method(self, qapp):
        """AC9: MainWindow has set_window_transparency method."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        assert hasattr(window, 'set_window_transparency')
        window.deleteLater()

    def test_set_window_transparency_100_percent(self, qapp):
        """AC10: Window can be set to 100% opacity."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        window.set_window_transparency(1.0)
        assert window.windowOpacity() == 1.0
        window.deleteLater()

    def test_set_window_transparency_50_percent(self, qapp):
        """AC11: Window can be set to 50% opacity."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        window.set_window_transparency(0.5)
        # Use approximate comparison due to Qt's internal precision
        assert abs(window.windowOpacity() - 0.5) < 0.01
        window.deleteLater()

    def test_set_window_transparency_20_percent(self, qapp):
        """AC12: Window can be set to minimum 20% opacity."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        window.set_window_transparency(0.2)
        assert window.windowOpacity() == 0.2
        window.deleteLater()

    def test_set_window_transparency_clamps_below_20(self, qapp):
        """AC13: Values below 20% are clamped to 20%."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        window.set_window_transparency(0.1)  # 10%
        assert window.windowOpacity() == 0.2  # Clamped to 20%
        window.deleteLater()

    def test_set_window_transparency_clamps_above_100(self, qapp):
        """AC14: Values above 100% are clamped to 100%."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        window.set_window_transparency(1.5)  # 150%
        assert window.windowOpacity() == 1.0  # Clamped to 100%
        window.deleteLater()

    def test_main_window_has_transparency_preview_handler(self, qapp):
        """AC15: MainWindow has transparency preview handler."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        assert hasattr(window, '_on_transparency_preview')
        window.deleteLater()


class TestSettingsDialogTransparencySlider:
    """Test SettingsDialog transparency slider."""

    def test_settings_dialog_has_transparency_slider(self, qapp):
        """AC16: Transparency slider exists in settings dialog."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings()
        dialog = SettingsDialog(settings)
        assert hasattr(dialog, 'transparency_slider')
        assert isinstance(dialog.transparency_slider, QSlider)
        dialog.deleteLater()

    def test_transparency_slider_minimum_is_20(self, qapp):
        """AC17: Slider minimum is 20%."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings()
        dialog = SettingsDialog(settings)
        assert dialog.transparency_slider.minimum() == 20
        dialog.deleteLater()

    def test_transparency_slider_maximum_is_100(self, qapp):
        """AC18: Slider maximum is 100%."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings()
        dialog = SettingsDialog(settings)
        assert dialog.transparency_slider.maximum() == 100
        dialog.deleteLater()

    def test_transparency_slider_default_value(self, qapp):
        """AC19: Slider defaults to 100% for default settings."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings()  # window_transparency defaults to 1.0
        dialog = SettingsDialog(settings)
        assert dialog.transparency_slider.value() == 100
        dialog.deleteLater()

    def test_transparency_value_label_exists(self, qapp):
        """AC20: Value label shows current percentage."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings()
        dialog = SettingsDialog(settings)
        assert hasattr(dialog, 'transparency_value_label')
        assert dialog.transparency_value_label.text() == "100%"
        dialog.deleteLater()

    def test_transparency_slider_updates_label(self, qapp):
        """AC21: Moving slider updates the label."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings()
        dialog = SettingsDialog(settings)
        dialog.transparency_slider.setValue(50)
        assert dialog.transparency_value_label.text() == "50%"
        dialog.deleteLater()

    def test_transparency_slider_emits_preview_signal(self, qapp):
        """AC22: Slider change emits preview signal with opacity value."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings()
        dialog = SettingsDialog(settings)

        received_values = []
        dialog.transparency_preview_requested.connect(
            lambda v: received_values.append(v)
        )

        dialog.transparency_slider.setValue(75)

        assert len(received_values) >= 1
        # Should emit 0.75 (75% as opacity)
        assert 0.75 in received_values
        dialog.deleteLater()

    def test_transparency_slider_cannot_go_below_20(self, qapp):
        """AC23: Cannot set slider below 20."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings()
        dialog = SettingsDialog(settings)
        dialog.transparency_slider.setValue(10)  # Try to set to 10
        # Slider enforces minimum
        assert dialog.transparency_slider.value() >= 20
        dialog.deleteLater()

    def test_load_settings_sets_slider_value(self, qapp):
        """AC24: Loading settings populates slider correctly."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings(window_transparency=0.65)
        dialog = SettingsDialog(settings)
        # Loading happens in constructor, check current value
        assert dialog.transparency_slider.value() == 65
        dialog.deleteLater()

    def test_save_settings_captures_slider_value(self, qapp):
        """AC25: Saving settings captures slider value."""
        from myvoice.ui.components.settings_dialog import SettingsDialog
        settings = AppSettings()
        dialog = SettingsDialog(settings)
        dialog.transparency_slider.setValue(80)
        dialog._save_current_settings()
        assert dialog.current_settings.window_transparency == 0.8
        dialog.deleteLater()


class TestTransparencyPersistence:
    """Test transparency setting persistence."""

    def test_transparency_survives_round_trip(self):
        """AC26: Transparency value survives serialization round-trip."""
        original = AppSettings(window_transparency=0.45)
        data = original.to_dict()
        restored = AppSettings.from_dict(data)
        assert restored.window_transparency == 0.45

    def test_transparency_default_when_missing(self):
        """AC27: Missing transparency defaults to 1.0."""
        data = {"ui_theme": "dark"}  # No window_transparency
        settings = AppSettings.from_dict(data)
        assert settings.window_transparency == 1.0


class TestTransparencyEdgeCases:
    """Test edge cases for transparency."""

    def test_transparency_type_validation_integer(self):
        """AC28: Integer values are accepted."""
        settings = AppSettings(window_transparency=1)  # int, not float
        validation = settings.validate()
        transparency_errors = [
            issue for issue in validation.issues
            if issue.field == "window_transparency" and issue.code == "INVALID_TYPE"
        ]
        assert len(transparency_errors) == 0

    def test_transparency_boundary_values(self):
        """AC29: Boundary values (0.2, 1.0) are valid without warnings."""
        # Test 0.2 (20%)
        settings_min = AppSettings(window_transparency=0.2)
        validation_min = settings_min.validate()
        min_warnings = [
            w for w in validation_min.warnings
            if w.field == "window_transparency"
        ]
        assert len(min_warnings) == 0

        # Test 1.0 (100%)
        settings_max = AppSettings(window_transparency=1.0)
        validation_max = settings_max.validate()
        max_warnings = [
            w for w in validation_max.warnings
            if w.field == "window_transparency"
        ]
        assert len(max_warnings) == 0
