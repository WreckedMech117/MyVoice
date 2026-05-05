"""
Tests for Settings Panel Access

Story 6.1: Settings Panel Access
Tests the settings button, dialog access, and keyboard shortcuts.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from myvoice.models.app_settings import AppSettings


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_settings():
    """Create mock AppSettings."""
    settings = MagicMock(spec=AppSettings)
    settings.config_directory = "/tmp/myvoice"
    settings.ui_theme = "dark"
    settings.always_on_top = True
    settings.to_dict.return_value = {
        "config_directory": "/tmp/myvoice",
        "ui_theme": "dark",
        "always_on_top": True
    }
    return settings


class TestSettingsDialogInit:
    """Tests for SettingsDialog initialization."""

    def test_dialog_creation(self, qapp, mock_settings):
        """Test settings dialog creates successfully."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        assert dialog is not None
        assert dialog.windowTitle() == "MyVoice Settings"

        dialog.deleteLater()

    def test_dialog_is_modal(self, qapp, mock_settings):
        """Test settings dialog is modal."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        assert dialog.isModal()

        dialog.deleteLater()

    def test_dialog_has_tabs(self, qapp, mock_settings):
        """Test dialog has tabbed interface."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        assert dialog.tab_widget is not None
        # Should have multiple tabs: Audio, Voices, TTS, Interface, Quick Speak
        assert dialog.tab_widget.count() >= 5

        dialog.deleteLater()


class TestSettingsDialogTabs:
    """Tests for settings dialog tabs."""

    def test_audio_tab_exists(self, qapp, mock_settings):
        """Test Audio tab exists."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        tab_names = [dialog.tab_widget.tabText(i) for i in range(dialog.tab_widget.count())]
        assert "Audio" in tab_names

        dialog.deleteLater()

    def test_voices_tab_exists(self, qapp, mock_settings):
        """Test Voices tab exists (Story 4.2: Voice Design Creation Workflow)."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        tab_names = [dialog.tab_widget.tabText(i) for i in range(dialog.tab_widget.count())]
        assert "Voices" in tab_names

        dialog.deleteLater()

    def test_tts_tab_exists(self, qapp, mock_settings):
        """Test TTS tab exists."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        tab_names = [dialog.tab_widget.tabText(i) for i in range(dialog.tab_widget.count())]
        assert "TTS" in tab_names

        dialog.deleteLater()

    def test_interface_tab_exists(self, qapp, mock_settings):
        """Test Interface tab exists."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        tab_names = [dialog.tab_widget.tabText(i) for i in range(dialog.tab_widget.count())]
        assert "Interface" in tab_names

        dialog.deleteLater()

    def test_quick_speak_tab_exists(self, qapp, mock_settings):
        """Test Quick Speak tab exists."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        tab_names = [dialog.tab_widget.tabText(i) for i in range(dialog.tab_widget.count())]
        assert "Quick Speak" in tab_names

        dialog.deleteLater()


class TestSettingsDialogButtons:
    """Tests for settings dialog buttons."""

    def test_ok_button_exists(self, qapp, mock_settings):
        """Test OK button exists."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        assert dialog.ok_button is not None
        assert dialog.ok_button.text() == "OK"

        dialog.deleteLater()

    def test_cancel_button_exists(self, qapp, mock_settings):
        """Test Cancel button exists."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        assert dialog.cancel_button is not None
        assert dialog.cancel_button.text() == "Cancel"

        dialog.deleteLater()

    def test_reset_button_exists(self, qapp, mock_settings):
        """Test Reset to Defaults button exists."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        assert dialog.reset_button is not None
        assert "Reset" in dialog.reset_button.text()

        dialog.deleteLater()


class TestSettingsDialogSignals:
    """Tests for settings dialog signals."""

    def test_settings_changed_signal_exists(self, qapp, mock_settings):
        """Test settings_changed signal exists."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        assert hasattr(dialog, 'settings_changed')

        dialog.deleteLater()

    def test_device_refresh_requested_signal_exists(self, qapp, mock_settings):
        """Test device_refresh_requested signal exists."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        assert hasattr(dialog, 'device_refresh_requested')

        dialog.deleteLater()


class TestSettingsDialogNavigation:
    """Tests for settings dialog navigation methods."""

    def test_navigate_to_voices(self, qapp, mock_settings):
        """Test navigation to Voices tab (Story 4.2)."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        dialog.navigate_to_voices()

        # Voices tab should be selected (index 1 - after Audio)
        assert dialog.tab_widget.currentIndex() == 1

        dialog.deleteLater()

    def test_navigate_to_quick_speak(self, qapp, mock_settings):
        """Test navigation to Quick Speak tab."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        dialog.navigate_to_quick_speak()

        # Quick Speak tab should be selected (index 4 - after Audio, Voices, TTS, Interface)
        assert dialog.tab_widget.currentIndex() == 4

        dialog.deleteLater()

    def test_navigate_to_custom_emotion(self, qapp, mock_settings):
        """Test navigation to Custom Emotion section."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        dialog.navigate_to_custom_emotion()

        # TTS tab should be selected (index 2 - after Audio, Voices)
        assert dialog.tab_widget.currentIndex() == 2

        dialog.deleteLater()


class TestMainWindowSettingsButton:
    """Tests for main window settings button."""

    def test_settings_button_exists(self, qapp):
        """Test settings button exists in main window."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert window.settings_button is not None

        window.deleteLater()

    def test_settings_button_tooltip_has_shortcut(self, qapp):
        """Test settings button tooltip mentions Ctrl+S."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        tooltip = window.settings_button.toolTip()
        assert "Ctrl+S" in tooltip

        window.deleteLater()


class TestSettingsKeyboardShortcut:
    """Tests for Ctrl+S settings keyboard shortcut."""

    def test_ctrl_s_shortcut_in_key_handler(self, qapp):
        """Test Ctrl+S is handled in keyPressEvent."""
        from myvoice.ui.main_window import MainWindow
        from PyQt6.QtGui import QKeyEvent

        window = MainWindow()

        # Mock the settings click handler
        window._on_settings_clicked = MagicMock()

        # Simulate Ctrl+S key press
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_S,
            Qt.KeyboardModifier.ControlModifier
        )
        window.keyPressEvent(event)

        # Verify settings handler was called
        window._on_settings_clicked.assert_called_once()

        window.deleteLater()
