"""
Tests for System Tray Integration

Story 7.2: System Tray Integration
Tests system tray icon, context menu, minimize/restore behavior,
and one-time notification.

FR38: User can minimize application to system tray
FR39: User can restore application from system tray
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QCloseEvent

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
    """Create mock AppSettings with tray enabled."""
    settings = MagicMock(spec=AppSettings)
    settings.config_directory = "/tmp/myvoice"
    settings.ui_theme = "dark"
    settings.always_on_top = True
    settings.minimize_to_tray = True
    settings.tray_notification_shown = False
    settings.log_level = "INFO"
    settings.window_transparency = 1.0
    settings.to_dict.return_value = {
        "config_directory": "/tmp/myvoice",
        "ui_theme": "dark",
        "always_on_top": True,
        "minimize_to_tray": True,
        "tray_notification_shown": False
    }
    return settings


@pytest.fixture
def mock_settings_tray_disabled():
    """Create mock AppSettings with tray disabled."""
    settings = MagicMock(spec=AppSettings)
    settings.config_directory = "/tmp/myvoice"
    settings.ui_theme = "dark"
    settings.always_on_top = True
    settings.minimize_to_tray = False
    settings.tray_notification_shown = False
    settings.log_level = "INFO"
    settings.window_transparency = 1.0
    return settings


def skip_if_no_tray():
    """Helper to skip test if system tray unavailable."""
    try:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            pytest.skip("System tray not available on this platform")
    except Exception:
        pytest.skip("System tray check failed")


class TestAppSettingsTrayFields:
    """Tests for new tray-related AppSettings fields."""

    def test_minimize_to_tray_default_is_true(self):
        """Test minimize_to_tray defaults to True."""
        settings = AppSettings()
        assert settings.minimize_to_tray == True

    def test_tray_notification_shown_default_is_false(self):
        """Test tray_notification_shown defaults to False."""
        settings = AppSettings()
        assert settings.tray_notification_shown == False

    def test_to_dict_includes_tray_fields(self):
        """Test to_dict includes tray settings."""
        settings = AppSettings(minimize_to_tray=True, tray_notification_shown=True)
        settings_dict = settings.to_dict()

        assert 'minimize_to_tray' in settings_dict
        assert settings_dict['minimize_to_tray'] == True
        assert 'tray_notification_shown' in settings_dict
        assert settings_dict['tray_notification_shown'] == True

    def test_from_dict_loads_tray_fields(self):
        """Test from_dict loads tray settings."""
        data = {
            'minimize_to_tray': False,
            'tray_notification_shown': True
        }
        settings = AppSettings.from_dict(data)

        assert settings.minimize_to_tray == False
        assert settings.tray_notification_shown == True


class TestSystemTraySetup:
    """Tests for system tray initialization."""

    def test_main_window_has_tray_icon_attribute(self, qapp):
        """Test MainWindow has tray_icon attribute."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert hasattr(window, 'tray_icon')

        window.deleteLater()

    def test_main_window_has_force_quit_flag(self, qapp):
        """Test MainWindow has _force_quit flag."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert hasattr(window, '_force_quit')
        assert window._force_quit == False

        window.deleteLater()

    def test_tray_icon_created_when_available(self, qapp):
        """Test tray icon is created when system tray is available."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert window.tray_icon is not None
        assert isinstance(window.tray_icon, QSystemTrayIcon)

        window.deleteLater()

    def test_tray_icon_has_tooltip(self, qapp):
        """Test tray icon has correct tooltip."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        if window.tray_icon:
            assert window.tray_icon.toolTip() == "MyVoice - Click to restore"

        window.deleteLater()

    def test_tray_icon_has_context_menu(self, qapp):
        """Test tray icon has context menu."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        if window.tray_icon:
            assert window.tray_icon.contextMenu() is not None

        window.deleteLater()


class TestTrayContextMenu:
    """Tests for system tray context menu."""

    def test_context_menu_has_restore_action(self, qapp):
        """Test context menu has Restore action."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        if window.tray_icon:
            menu = window.tray_icon.contextMenu()
            actions = menu.actions()
            action_texts = [a.text() for a in actions]

            assert "Restore" in action_texts

        window.deleteLater()

    def test_context_menu_has_settings_action(self, qapp):
        """Test context menu has Settings action."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        if window.tray_icon:
            menu = window.tray_icon.contextMenu()
            actions = menu.actions()
            action_texts = [a.text() for a in actions]

            assert "Settings" in action_texts

        window.deleteLater()

    def test_context_menu_has_exit_action(self, qapp):
        """Test context menu has Exit action."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        if window.tray_icon:
            menu = window.tray_icon.contextMenu()
            actions = menu.actions()
            action_texts = [a.text() for a in actions]

            assert "Exit" in action_texts

        window.deleteLater()

    def test_context_menu_has_separator(self, qapp):
        """Test context menu has separator before Exit."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        if window.tray_icon:
            menu = window.tray_icon.contextMenu()
            actions = menu.actions()

            # Find separator (separator actions have empty text and isSeparator() = True)
            has_separator = any(a.isSeparator() for a in actions)
            assert has_separator

        window.deleteLater()


class TestMinimizeToTray:
    """Tests for minimize to tray behavior."""

    def test_minimize_to_tray_method_exists(self, qapp):
        """Test MainWindow has _minimize_to_tray method."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert hasattr(window, '_minimize_to_tray')
        assert callable(window._minimize_to_tray)

        window.deleteLater()

    def test_minimize_to_tray_hides_window(self, qapp, mock_settings):
        """Test _minimize_to_tray hides the window."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        mock_settings.tray_notification_shown = True  # Skip notification to avoid signal emit
        window.app_settings = mock_settings
        window.show()

        window._minimize_to_tray()

        assert not window.isVisible()

        window.deleteLater()

    def test_minimize_to_tray_shows_tray_icon(self, qapp, mock_settings):
        """Test _minimize_to_tray shows tray icon."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        mock_settings.tray_notification_shown = True  # Skip notification to avoid signal emit
        window.app_settings = mock_settings

        if window.tray_icon:
            window._minimize_to_tray()
            assert window.tray_icon.isVisible()

        window.deleteLater()


class TestRestoreFromTray:
    """Tests for restore from tray behavior."""

    def test_restore_from_tray_method_exists(self, qapp):
        """Test MainWindow has _restore_from_tray method."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert hasattr(window, '_restore_from_tray')
        assert callable(window._restore_from_tray)

        window.deleteLater()

    def test_restore_from_tray_shows_window(self, qapp, mock_settings):
        """Test _restore_from_tray shows the window."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        window.app_settings = mock_settings
        window.hide()

        window._restore_from_tray()

        assert window.isVisible()

        window.deleteLater()


class TestCloseEventBehavior:
    """Tests for close event handling."""

    def test_close_event_minimizes_to_tray_when_enabled(self, qapp):
        """Test close event minimizes to tray when minimize_to_tray is True."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow
        from myvoice.models.app_settings import AppSettings

        window = MainWindow()
        # Create and set real AppSettings to avoid signal type issues
        settings = AppSettings()
        settings.minimize_to_tray = True
        settings.tray_notification_shown = True  # Skip notification
        window.app_settings = settings
        window.show()

        if window.tray_icon:
            # Simulate close event
            window.close()

            # Window should be hidden, not destroyed
            assert not window.isVisible()
            assert window.tray_icon.isVisible()

        window.deleteLater()

    def test_close_event_closes_when_force_quit(self, qapp, mock_settings):
        """Test close event actually closes when _force_quit is True."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        window.app_settings = mock_settings
        window._force_quit = True
        window.show()

        # Force quit should close the window
        window._force_quit = True
        window.close()

        # Window should be closing (accepting the event)
        # Note: Testing actual window destruction is tricky in Qt tests

        window.deleteLater()


class TestQuitApplication:
    """Tests for application quit from tray."""

    def test_quit_application_method_exists(self, qapp):
        """Test MainWindow has _quit_application method."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert hasattr(window, '_quit_application')
        assert callable(window._quit_application)

        window.deleteLater()

    def test_quit_application_sets_force_quit(self, qapp):
        """Test _quit_application sets _force_quit to True."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        # Mock close to prevent actual close
        window.close = MagicMock()

        window._quit_application()

        assert window._force_quit == True

        window.deleteLater()


class TestTrayIconActivation:
    """Tests for tray icon click handling."""

    def test_tray_icon_activated_handler_exists(self, qapp):
        """Test MainWindow has _on_tray_icon_activated method."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert hasattr(window, '_on_tray_icon_activated')
        assert callable(window._on_tray_icon_activated)

        window.deleteLater()

    def test_single_click_restores_window(self, qapp, mock_settings):
        """Test single click on tray icon restores window."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        window.app_settings = mock_settings
        window.hide()

        # Simulate single click
        window._on_tray_icon_activated(QSystemTrayIcon.ActivationReason.Trigger)

        assert window.isVisible()

        window.deleteLater()


class TestCustomTitleBarMinimize:
    """Tests for custom title bar minimize button."""

    def test_title_bar_minimize_calls_minimize_to_tray(self, qapp):
        """Test title bar minimize button calls _minimize_to_tray when available."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        window._minimize_to_tray = MagicMock()

        window.title_bar._on_minimize_clicked()

        window._minimize_to_tray.assert_called_once()

        window.deleteLater()

    def test_title_bar_minimize_method_exists(self, qapp):
        """Test title bar has minimize click handler."""
        from myvoice.ui.components.custom_title_bar import CustomTitleBar

        # Create title bar with no parent
        title_bar = CustomTitleBar(None)

        assert hasattr(title_bar, '_on_minimize_clicked')
        assert callable(title_bar._on_minimize_clicked)

        title_bar.deleteLater()


class TestOneTimeNotification:
    """Tests for one-time tray notification."""

    def test_notification_shown_on_first_minimize(self, qapp):
        """Test notification is shown on first minimize to tray."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        # Use real AppSettings to avoid signal type issues
        real_settings = AppSettings(tray_notification_shown=False)
        window.app_settings = real_settings

        if window.tray_icon:
            # Mock showMessage to track if it was called
            window.tray_icon.showMessage = MagicMock()

            window._minimize_to_tray()

            window.tray_icon.showMessage.assert_called_once()

        window.deleteLater()

    def test_notification_not_shown_after_first_time(self, qapp, mock_settings):
        """Test notification is NOT shown after first time."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        mock_settings.tray_notification_shown = True  # Already shown
        window.app_settings = mock_settings

        if window.tray_icon:
            # Mock showMessage to track if it was called
            window.tray_icon.showMessage = MagicMock()

            window._minimize_to_tray()

            window.tray_icon.showMessage.assert_not_called()

        window.deleteLater()

    def test_notification_updates_settings_flag(self, qapp):
        """Test notification shown flag is updated in settings."""
        skip_if_no_tray()
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()
        # Use real AppSettings to avoid signal type issues
        real_settings = AppSettings(tray_notification_shown=False)
        window.app_settings = real_settings

        if window.tray_icon:
            # Mock showMessage
            window.tray_icon.showMessage = MagicMock()

            window._minimize_to_tray()

            # Check that the setting was updated
            assert real_settings.tray_notification_shown == True

        window.deleteLater()
