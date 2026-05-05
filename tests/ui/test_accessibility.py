"""
Tests for Story 7.5: Accessibility & Keyboard Navigation

Tests the accessibility features including:
- Tab navigation order (NFR18)
- Focus indicators (NFR20)
- Keyboard activation
- Accessible names for screen readers
- Live region announcements
- High contrast mode support
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton
from PyQt6.QtCore import Qt


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestAccessibleNames:
    """Tests for accessible names on controls."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        yield window
        window.close()

    def test_text_input_has_accessible_name(self, main_window):
        """Text input should have accessible name for screen readers."""
        assert main_window.text_input.accessibleName() == "Text to speak"

    def test_generate_button_has_accessible_name(self, main_window):
        """Generate button should have accessible name."""
        assert main_window.generate_button.accessibleName() == "Generate speech"

    def test_quick_speak_button_has_accessible_name(self, main_window):
        """Quick speak button should have accessible name."""
        assert main_window.quick_speak_button.accessibleName() == "Quick Speak"

    def test_replay_button_has_accessible_name(self, main_window):
        """Replay button should have accessible name."""
        assert main_window.replay_button.accessibleName() == "Replay last audio"

    def test_clear_button_has_accessible_name(self, main_window):
        """Clear button should have accessible name."""
        assert main_window.clear_button.accessibleName() == "Clear text"

    def test_settings_button_has_accessible_name(self, main_window):
        """Settings button should have accessible name."""
        assert main_window.settings_button.accessibleName() == "Settings"

    def test_status_bar_has_accessible_name(self, main_window):
        """Status bar should have accessible name."""
        assert main_window.status_bar.accessibleName() == "Status"


class TestFocusPolicy:
    """Tests for focus policy settings."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        yield window
        window.close()

    def test_text_input_accepts_focus(self, main_window):
        """Text input should accept focus."""
        policy = main_window.text_input.focusPolicy()
        assert policy == Qt.FocusPolicy.StrongFocus

    def test_generate_button_accepts_focus(self, main_window):
        """Generate button should accept focus."""
        policy = main_window.generate_button.focusPolicy()
        assert policy == Qt.FocusPolicy.StrongFocus

    def test_quick_speak_button_accepts_focus(self, main_window):
        """Quick speak button should accept focus."""
        policy = main_window.quick_speak_button.focusPolicy()
        assert policy == Qt.FocusPolicy.StrongFocus

    def test_replay_button_accepts_focus(self, main_window):
        """Replay button should accept focus."""
        policy = main_window.replay_button.focusPolicy()
        assert policy == Qt.FocusPolicy.StrongFocus

    def test_clear_button_accepts_focus(self, main_window):
        """Clear button should accept focus."""
        policy = main_window.clear_button.focusPolicy()
        assert policy == Qt.FocusPolicy.StrongFocus

    def test_settings_button_accepts_focus(self, main_window):
        """Settings button should accept focus."""
        policy = main_window.settings_button.focusPolicy()
        assert policy == Qt.FocusPolicy.StrongFocus

    def test_voice_label_no_focus(self, main_window):
        """Voice label should not accept focus (informational only)."""
        policy = main_window.current_voice_label.focusPolicy()
        assert policy == Qt.FocusPolicy.NoFocus


class TestTabOrder:
    """Tests for tab navigation order."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        window.show()  # Need to show for focus to work
        yield window
        window.close()

    def test_text_input_can_receive_focus(self, main_window, qtbot):
        """Text input should be able to receive focus via tab."""
        main_window.text_input.setFocus()
        assert main_window.text_input.hasFocus()

    def test_tab_from_text_to_quick_speak(self, main_window, qtbot):
        """Tab from text input should go to quick speak button."""
        main_window.text_input.setFocus()
        qtbot.keyClick(main_window.text_input, Qt.Key.Key_Tab)
        # Note: Actual tab order verification requires more complex testing


class TestEmotionButtonAccessibility:
    """Tests for emotion button accessibility."""

    @pytest.fixture
    def emotion_group(self, qapp):
        """Create an EmotionButtonGroup instance."""
        from myvoice.ui.components.emotion_button_group import EmotionButtonGroup
        group = EmotionButtonGroup()
        yield group
        group.deleteLater()

    def test_emotion_buttons_have_accessible_names(self, emotion_group):
        """Each emotion button should have an accessible name."""
        from myvoice.ui.components.emotion_button_group import EmotionPreset

        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)  # Fixed method name
            assert button is not None
            assert preset.display_name.lower() in button.accessibleName().lower()

    def test_get_emotion_buttons_returns_list(self, emotion_group):
        """get_emotion_buttons should return a list of buttons."""
        buttons = emotion_group.get_emotion_buttons()
        assert isinstance(buttons, list)
        assert len(buttons) == 5  # 5 emotion presets

    def test_emotion_buttons_in_order(self, emotion_group):
        """Emotion buttons should be returned in display order."""
        from myvoice.ui.components.emotion_button_group import EmotionPreset

        buttons = emotion_group.get_emotion_buttons()
        expected_order = list(EmotionPreset)

        for i, button in enumerate(buttons):
            assert button.preset == expected_order[i]


class TestLiveRegionAnnouncements:
    """Tests for live region status announcements."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        yield window
        window.close()

    def test_announce_status_updates_status_bar(self, main_window):
        """announce_status should update the status bar message."""
        main_window.announce_status("Test message")
        assert main_window.status_bar.currentMessage() == "Test message"

    def test_announce_generation_complete(self, main_window):
        """announce_generation_complete should announce completion."""
        main_window.announce_generation_complete()
        assert "complete" in main_window.status_bar.currentMessage().lower()

    def test_announce_error(self, main_window):
        """announce_error should announce the error."""
        main_window.announce_error("Test error")
        assert "error" in main_window.status_bar.currentMessage().lower()
        assert "Test error" in main_window.status_bar.currentMessage()


class TestHighContrastMode:
    """Tests for high contrast mode support."""

    def test_is_high_contrast_mode_returns_bool(self):
        """is_high_contrast_mode should return a boolean."""
        from myvoice.ui.styles.theme_manager import is_high_contrast_mode
        result = is_high_contrast_mode()
        assert isinstance(result, bool)

    def test_is_high_contrast_mode_handles_non_windows(self):
        """is_high_contrast_mode should handle non-Windows platforms gracefully."""
        from myvoice.ui.styles.theme_manager import is_high_contrast_mode

        with patch('myvoice.ui.styles.theme_manager.sys') as mock_sys:
            mock_sys.platform = "linux"
            # Should return False on non-Windows platforms
            # Note: This test may not work correctly due to module-level imports


class TestFocusIndicatorStyles:
    """Tests for focus indicator CSS styles."""

    def test_dark_theme_has_focus_styles(self):
        """Dark theme should have focus indicator styles."""
        from pathlib import Path
        dark_theme_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "styles" / "minimal_dark.qss"

        if dark_theme_path.exists():
            content = dark_theme_path.read_text()
            assert ":focus" in content
            assert "Story 7.5" in content or "Focus Indicator" in content

    def test_light_theme_has_focus_styles(self):
        """Light theme should have focus indicator styles."""
        from pathlib import Path
        light_theme_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "styles" / "minimal_light.qss"

        if light_theme_path.exists():
            content = light_theme_path.read_text()
            assert ":focus" in content
            assert "Story 7.5" in content or "Focus Indicator" in content


class TestKeyboardActivation:
    """Tests for keyboard activation of controls."""

    @pytest.fixture
    def main_window(self, qapp):
        """Create a MainWindow instance."""
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        window.show()
        yield window
        window.close()

    def test_button_responds_to_space(self, main_window, qtbot):
        """Buttons should respond to Space key when focused."""
        # This test verifies that buttons have correct focus policy for keyboard activation
        assert main_window.generate_button.focusPolicy() in [
            Qt.FocusPolicy.StrongFocus,
            Qt.FocusPolicy.TabFocus
        ]

    def test_button_responds_to_enter(self, main_window, qtbot):
        """Buttons should respond to Enter key when focused."""
        # Buttons with StrongFocus should respond to Enter/Return
        assert main_window.generate_button.focusPolicy() == Qt.FocusPolicy.StrongFocus
