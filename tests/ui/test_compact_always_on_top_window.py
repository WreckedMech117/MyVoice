"""
Tests for Compact Always-On-Top Window

Story 7.1: Compact Always-On-Top Window
Tests window dimensions, frameless design, custom title bar,
always-on-top behavior, and settings persistence.

FR36: Application displays a compact, always-on-top window
FR37: User can toggle always-on-top behavior
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QSize, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QMouseEvent

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
    """Create mock AppSettings with always_on_top enabled."""
    settings = MagicMock(spec=AppSettings)
    settings.config_directory = "/tmp/myvoice"
    settings.ui_theme = "dark"
    settings.always_on_top = True
    settings.log_level = "INFO"
    settings.window_transparency = 1.0
    settings.to_dict.return_value = {
        "config_directory": "/tmp/myvoice",
        "ui_theme": "dark",
        "always_on_top": True,
        "log_level": "INFO",
        "window_transparency": 1.0
    }
    return settings


@pytest.fixture
def mock_settings_top_disabled():
    """Create mock AppSettings with always_on_top disabled."""
    settings = MagicMock(spec=AppSettings)
    settings.config_directory = "/tmp/myvoice"
    settings.ui_theme = "dark"
    settings.always_on_top = False
    settings.log_level = "INFO"
    settings.window_transparency = 1.0
    settings.to_dict.return_value = {
        "config_directory": "/tmp/myvoice",
        "ui_theme": "dark",
        "always_on_top": False,
        "log_level": "INFO",
        "window_transparency": 1.0
    }
    return settings


class TestWindowDimensions:
    """Tests for window size requirements (400×188px)."""

    def test_window_default_size(self, qapp):
        """Test window has correct default size (400×220px)."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        size = window.size()

        assert size.width() == 400
        assert size.height() == 220  # Updated from 188

        window.deleteLater()

    def test_window_minimum_size(self, qapp):
        """Test window has correct minimum size (320×200px)."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        min_size = window.minimumSize()

        assert min_size.width() == 320
        assert min_size.height() == 200  # Updated from 168

        window.deleteLater()

    def test_window_maximum_size(self, qapp):
        """Test window has correct maximum size (600×300px)."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        max_size = window.maximumSize()

        assert max_size.width() == 600
        assert max_size.height() == 300  # Updated from 268

        window.deleteLater()


class TestFramelessWindow:
    """Tests for frameless window with custom title bar."""

    def test_window_is_frameless(self, qapp):
        """Test window has FramelessWindowHint flag."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        flags = window.windowFlags()

        assert flags & Qt.WindowType.FramelessWindowHint

        window.deleteLater()

    def test_window_has_custom_title_bar(self, qapp):
        """Test window has CustomTitleBar component."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert hasattr(window, 'title_bar')
        assert window.title_bar is not None

        window.deleteLater()


class TestCustomTitleBar:
    """Tests for CustomTitleBar component."""

    def test_title_bar_height(self, qapp):
        """Test title bar has correct fixed height (28px)."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        title_bar = CustomTitleBar(None)

        assert title_bar.height() == 28

        title_bar.deleteLater()

    def test_title_bar_has_title(self, qapp):
        """Test title bar displays 'MyVoice' title."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        title_bar = CustomTitleBar(None)

        assert title_bar.title_label is not None
        assert title_bar.title_label.text() == "MyVoice"

        title_bar.deleteLater()

    def test_title_bar_has_minimize_button(self, qapp):
        """Test title bar has minimize button."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        title_bar = CustomTitleBar(None)

        assert title_bar.minimize_button is not None
        assert title_bar.minimize_button.text() == "−"
        assert title_bar.minimize_button.toolTip() == "Minimize"

        title_bar.deleteLater()

    def test_title_bar_has_maximize_button(self, qapp):
        """Test title bar has maximize/restore button."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        title_bar = CustomTitleBar(None)

        assert title_bar.maximize_button is not None
        assert title_bar.maximize_button.text() == "□"
        assert title_bar.maximize_button.toolTip() == "Maximize"

        title_bar.deleteLater()

    def test_title_bar_has_close_button(self, qapp):
        """Test title bar has close button."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        title_bar = CustomTitleBar(None)

        assert title_bar.close_button is not None
        assert title_bar.close_button.text() == "×"
        assert title_bar.close_button.toolTip() == "Close"

        title_bar.deleteLater()

    def test_title_bar_button_sizes(self, qapp):
        """Test title bar buttons have correct size (32×24px)."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        title_bar = CustomTitleBar(None)

        for button in [title_bar.minimize_button, title_bar.maximize_button, title_bar.close_button]:
            assert button.width() == 32
            assert button.height() == 24

        title_bar.deleteLater()


class TestTitleBarDragging:
    """Tests for draggable title bar functionality."""

    def test_title_bar_tracks_drag_state(self, qapp):
        """Test title bar has drag tracking attributes."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        title_bar = CustomTitleBar(None)

        assert hasattr(title_bar, '_is_dragging')
        assert hasattr(title_bar, '_drag_position')
        assert title_bar._is_dragging == False

        title_bar.deleteLater()

    def test_title_bar_mouse_event_methods(self, qapp):
        """Test title bar has mouse event handlers for dragging."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        title_bar = CustomTitleBar(None)

        assert hasattr(title_bar, 'mousePressEvent')
        assert hasattr(title_bar, 'mouseMoveEvent')
        assert hasattr(title_bar, 'mouseReleaseEvent')
        assert hasattr(title_bar, 'mouseDoubleClickEvent')

        title_bar.deleteLater()


class TestAlwaysOnTopDefault:
    """Tests for always-on-top default behavior."""

    def test_always_on_top_default_value_is_true(self):
        """Test AppSettings.always_on_top defaults to True."""
        settings = AppSettings()

        assert settings.always_on_top == True

    def test_window_always_on_top_flag_with_settings(self, qapp, mock_settings):
        """Test window has WindowStaysOnTopHint when always_on_top is True."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        window.set_app_settings(mock_settings)

        flags = window.windowFlags()
        assert flags & Qt.WindowType.WindowStaysOnTopHint

        window.deleteLater()


class TestAlwaysOnTopToggle:
    """Tests for always-on-top toggle functionality."""

    def test_set_always_on_top_true(self, qapp):
        """Test set_always_on_top(True) adds WindowStaysOnTopHint."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        window.set_always_on_top(True)

        flags = window.windowFlags()
        assert flags & Qt.WindowType.WindowStaysOnTopHint

        window.deleteLater()

    def test_set_always_on_top_false(self, qapp):
        """Test set_always_on_top(False) removes WindowStaysOnTopHint."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        window.set_always_on_top(False)

        flags = window.windowFlags()
        assert not (flags & Qt.WindowType.WindowStaysOnTopHint)

        window.deleteLater()

    def test_toggle_always_on_top_preserves_frameless(self, qapp):
        """Test toggling always-on-top preserves frameless window hint."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        # Toggle off
        window.set_always_on_top(False)
        flags = window.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint

        # Toggle on
        window.set_always_on_top(True)
        flags = window.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint

        window.deleteLater()


class TestSettingsInterfaceTab:
    """Tests for Always-on-Top toggle in Settings → Interface."""

    def test_interface_tab_has_always_on_top_checkbox(self, qapp, mock_settings):
        """Test Interface tab has always-on-top checkbox."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        assert hasattr(dialog, 'always_on_top_checkbox')
        assert dialog.always_on_top_checkbox is not None

        dialog.deleteLater()

    def test_always_on_top_checkbox_label(self, qapp, mock_settings):
        """Test always-on-top checkbox has correct label."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        text = dialog.always_on_top_checkbox.text()
        assert "always on top" in text.lower()

        dialog.deleteLater()

    def test_always_on_top_checkbox_loads_current_value(self, qapp, mock_settings):
        """Test checkbox loads current always_on_top setting."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        mock_settings.always_on_top = True
        dialog = SettingsDialog(mock_settings)

        assert dialog.always_on_top_checkbox.isChecked() == True

        dialog.deleteLater()

    def test_always_on_top_checkbox_loads_false_value(self, qapp, mock_settings_top_disabled):
        """Test checkbox loads when always_on_top is False."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings_top_disabled)

        assert dialog.always_on_top_checkbox.isChecked() == False

        dialog.deleteLater()

    def test_interface_tab_in_appearance_group(self, qapp, mock_settings):
        """Test always-on-top is in Appearance group."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        # Navigate to Interface tab (index 2)
        tab_names = [dialog.tab_widget.tabText(i) for i in range(dialog.tab_widget.count())]
        assert "Interface" in tab_names

        dialog.deleteLater()


class TestSettingsPersistence:
    """Tests for always-on-top setting persistence."""

    def test_app_settings_has_always_on_top_field(self):
        """Test AppSettings model has always_on_top field."""
        from myvoice.models.app_settings import AppSettings

        settings = AppSettings()

        assert hasattr(settings, 'always_on_top')
        assert isinstance(settings.always_on_top, bool)

    def test_app_settings_to_dict_includes_always_on_top(self):
        """Test always_on_top is included in to_dict() output."""
        from myvoice.models.app_settings import AppSettings

        settings = AppSettings(always_on_top=True)
        settings_dict = settings.to_dict()

        assert 'always_on_top' in settings_dict
        assert settings_dict['always_on_top'] == True

    def test_app_settings_from_dict_loads_always_on_top(self):
        """Test always_on_top is loaded from dictionary."""
        from myvoice.models.app_settings import AppSettings

        data = {'always_on_top': False}
        settings = AppSettings.from_dict(data)

        assert settings.always_on_top == False

    def test_settings_dialog_saves_always_on_top_to_settings(self, qapp, mock_settings):
        """Test settings dialog saves checkbox state to AppSettings."""
        from myvoice.ui.components.settings_dialog import SettingsDialog

        dialog = SettingsDialog(mock_settings)

        # Toggle checkbox
        dialog.always_on_top_checkbox.setChecked(False)

        # Call save method
        dialog._save_current_settings()

        # Verify setting was updated (on dialog's internal settings object)
        assert dialog.current_settings.always_on_top == False

        dialog.deleteLater()


class TestWindowTitle:
    """Tests for window title."""

    def test_window_title_is_myvoice(self, qapp):
        """Test window title is 'MyVoice'."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert window.windowTitle() == "MyVoice"

        window.deleteLater()


class TestTitleBarMaximizeRestore:
    """Tests for maximize/restore button behavior."""

    def test_maximize_button_toggles_state(self, qapp):
        """Test maximize button changes icon on toggle."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        title_bar = CustomTitleBar(None)

        # Initial state
        assert title_bar._is_maximized == False
        assert title_bar.maximize_button.text() == "□"

        # Update state
        title_bar.update_maximize_state(True)
        assert title_bar._is_maximized == True
        assert title_bar.maximize_button.text() == "❐"
        assert title_bar.maximize_button.toolTip() == "Restore"

        # Toggle back
        title_bar.update_maximize_state(False)
        assert title_bar._is_maximized == False
        assert title_bar.maximize_button.text() == "□"
        assert title_bar.maximize_button.toolTip() == "Maximize"

        title_bar.deleteLater()


class TestSettingsChangedSignal:
    """Tests for settings change signal integration."""

    def test_main_window_has_settings_changed_handler(self, qapp):
        """Test MainWindow has _on_settings_changed method."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert hasattr(window, '_on_settings_changed')
        assert callable(window._on_settings_changed)

        window.deleteLater()

    def test_settings_changed_applies_always_on_top(self, qapp, mock_settings):
        """Test _on_settings_changed applies always_on_top setting."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        window.set_always_on_top = MagicMock()

        mock_settings.always_on_top = False
        window._on_settings_changed(mock_settings)

        window.set_always_on_top.assert_called_with(False)

        window.deleteLater()


class TestWindowIcon:
    """Tests for window icon."""

    def test_title_bar_has_icon(self, qapp):
        """Test title bar displays app icon."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        # Window should have an icon set
        assert not window.windowIcon().isNull()

        window.deleteLater()
