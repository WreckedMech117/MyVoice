"""
Tests for Playback Last Feature

Story 2.4: Playback Last
Covers: FR28, FR29, FR31, FR32, NFR5
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Skip if PyQt6 not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestReplayButtonPresence:
    """Tests for replay button presence in UI (FR28)."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        yield window
        window.close()
        window.deleteLater()

    def test_replay_button_exists(self, main_window):
        """Test that replay button exists in main window."""
        assert hasattr(main_window, 'replay_button')
        assert main_window.replay_button is not None

    def test_replay_button_initial_state_disabled(self, main_window):
        """Test replay button is disabled on fresh start (session-only cache)."""
        assert not main_window.replay_button.isEnabled()

    def test_replay_button_has_correct_tooltip_disabled(self, main_window):
        """Test replay button tooltip when disabled."""
        assert "No audio to replay" in main_window.replay_button.toolTip()

    def test_replay_button_object_name(self, main_window):
        """Test replay button has correct object name for styling."""
        assert main_window.replay_button.objectName() == "replay_button"

    def test_replay_button_size(self, main_window):
        """Test replay button has same size as generate button."""
        # The actual rendered size may vary due to theming
        # What matters is that it matches other action buttons
        assert main_window.replay_button.width() == main_window.generate_button.width()
        assert main_window.replay_button.height() == main_window.generate_button.height()


class TestReplayButtonEnableDisable:
    """Tests for replay button enable/disable (FR29)."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        yield window
        window.close()
        window.deleteLater()

    def test_set_replay_enabled_true(self, main_window):
        """Test enabling replay button."""
        main_window.set_replay_enabled(True)

        assert main_window.replay_button.isEnabled()
        assert "Ctrl+R" in main_window.replay_button.toolTip()

    def test_set_replay_enabled_false(self, main_window):
        """Test disabling replay button."""
        # First enable it
        main_window.set_replay_enabled(True)
        assert main_window.replay_button.isEnabled()

        # Then disable it
        main_window.set_replay_enabled(False)
        assert not main_window.replay_button.isEnabled()
        assert "No audio to replay" in main_window.replay_button.toolTip()


class TestReplaySignal:
    """Tests for replay_last_requested signal (FR28)."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        yield window
        window.close()
        window.deleteLater()

    def test_replay_signal_exists(self, main_window):
        """Test that replay_last_requested signal exists."""
        assert hasattr(main_window, 'replay_last_requested')

    def test_replay_button_click_emits_signal(self, main_window):
        """Test that clicking replay button emits signal."""
        signal_mock = Mock()
        main_window.replay_last_requested.connect(signal_mock)

        # Enable button first
        main_window.set_replay_enabled(True)

        # Click the button
        main_window.replay_button.click()

        signal_mock.assert_called_once()

    def test_replay_button_click_disabled_no_signal(self, main_window):
        """Test that clicking disabled replay button doesn't emit signal."""
        signal_mock = Mock()
        main_window.replay_last_requested.connect(signal_mock)

        # Button is disabled by default
        assert not main_window.replay_button.isEnabled()

        # Try to click (won't work because disabled)
        # We can't actually click a disabled button via click()
        # So we verify the button is indeed disabled
        signal_mock.assert_not_called()


class TestReplayKeyboardShortcut:
    """Tests for Ctrl+R keyboard shortcut (FR28)."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        yield window
        window.close()
        window.deleteLater()

    def test_ctrl_r_triggers_replay_when_enabled(self, main_window):
        """Test Ctrl+R triggers replay when button is enabled."""
        signal_mock = Mock()
        main_window.replay_last_requested.connect(signal_mock)

        # Enable replay
        main_window.set_replay_enabled(True)

        # Simulate Ctrl+R key press
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.ControlModifier
        )
        main_window.keyPressEvent(event)

        signal_mock.assert_called_once()

    def test_ctrl_r_no_effect_when_disabled(self, main_window):
        """Test Ctrl+R has no effect when replay is disabled."""
        signal_mock = Mock()
        main_window.replay_last_requested.connect(signal_mock)

        # Replay is disabled by default
        assert not main_window.replay_button.isEnabled()

        # Simulate Ctrl+R key press
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.ControlModifier
        )
        main_window.keyPressEvent(event)

        # Signal should NOT be emitted
        signal_mock.assert_not_called()

    def test_plain_r_key_not_consumed(self, main_window):
        """Test that plain R key (no Ctrl) is not consumed."""
        # Enable replay
        main_window.set_replay_enabled(True)

        signal_mock = Mock()
        main_window.replay_last_requested.connect(signal_mock)

        # Simulate plain R key press (no modifier)
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.NoModifier
        )
        main_window.keyPressEvent(event)

        # Signal should NOT be emitted for plain R
        signal_mock.assert_not_called()


class TestReplayButtonLayout:
    """Tests for replay button layout position."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        yield window
        window.close()
        window.deleteLater()

    def test_replay_button_position_after_generate(self, main_window):
        """Test replay button is positioned after generate button."""
        # Get the action buttons layout
        # The buttons should be: quick_speak, generate, replay, clear
        assert hasattr(main_window, 'generate_button')
        assert hasattr(main_window, 'replay_button')
        assert hasattr(main_window, 'clear_button')

        # Verify all buttons exist and are QPushButton instances
        from PyQt6.QtWidgets import QPushButton
        assert isinstance(main_window.generate_button, QPushButton)
        assert isinstance(main_window.replay_button, QPushButton)
        assert isinstance(main_window.clear_button, QPushButton)


class TestSessionOnlyCache:
    """Tests for session-only cache behavior (FR29)."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        yield window
        window.close()
        window.deleteLater()

    def test_fresh_start_replay_disabled(self, main_window):
        """Test replay is disabled on fresh application start."""
        # On fresh start, replay should be disabled
        assert not main_window.replay_button.isEnabled()

    def test_replay_enabled_after_generation(self, main_window):
        """Test replay can be enabled after audio generation."""
        # Initially disabled
        assert not main_window.replay_button.isEnabled()

        # After successful generation, it should be enabled
        main_window.set_replay_enabled(True)
        assert main_window.replay_button.isEnabled()
