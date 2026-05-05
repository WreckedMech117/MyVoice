"""
Tests for Emotion Keyboard Shortcuts

Story 3.5: Emotion Keyboard Shortcuts
Covers: FR6 (5 emotion presets selection via F1-F5)

Acceptance Criteria:
- AC1: F1 → Neutral (😐)
- AC2: F2 → Happy (😄)
- AC3: F3 → Sad (😢)
- AC4: F4 → Angry (😠)
- AC5: F5 → Flirtatious (😏)
- AC6: F1-F5 in Settings dialog → No effect (window-scoped)
- AC7: F1-F5 while text input focused → Still works (window-scoped)
- AC8: F1-F5 with emotion disabled (cloned voice) → No effect
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtCore import QEvent

from myvoice.ui.components.emotion_button_group import EmotionPreset


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def main_window(qapp):
    """Create a MainWindow instance for testing."""
    from myvoice.ui.main_window import MainWindow
    window = MainWindow()
    yield window
    window.close()
    window.deleteLater()


def create_key_event(key: Qt.Key, event_type: QEvent.Type = QEvent.Type.KeyPress) -> QKeyEvent:
    """Create a QKeyEvent for testing."""
    return QKeyEvent(
        event_type,
        key,
        Qt.KeyboardModifier.NoModifier,
        ""
    )


class TestEmotionKeyboardShortcutsBasic:
    """Basic keyboard shortcut tests - AC1-AC5."""

    def test_f1_selects_neutral(self, main_window):
        """AC1: F1 key selects Neutral emotion."""
        # Set to non-neutral first
        main_window.set_emotion_by_index(2)  # Sad
        assert main_window.get_emotion_preset() != EmotionPreset.NEUTRAL

        # Press F1
        event = create_key_event(Qt.Key.Key_F1)
        main_window.keyPressEvent(event)

        assert main_window.get_emotion_preset() == EmotionPreset.NEUTRAL

    def test_f2_selects_happy(self, main_window):
        """AC2: F2 key selects Happy emotion."""
        event = create_key_event(Qt.Key.Key_F2)
        main_window.keyPressEvent(event)

        assert main_window.get_emotion_preset() == EmotionPreset.HAPPY

    def test_f3_selects_sad(self, main_window):
        """AC3: F3 key selects Sad emotion."""
        event = create_key_event(Qt.Key.Key_F3)
        main_window.keyPressEvent(event)

        assert main_window.get_emotion_preset() == EmotionPreset.SAD

    def test_f4_selects_angry(self, main_window):
        """AC4: F4 key selects Angry emotion."""
        event = create_key_event(Qt.Key.Key_F4)
        main_window.keyPressEvent(event)

        assert main_window.get_emotion_preset() == EmotionPreset.ANGRY

    def test_f5_selects_flirtatious(self, main_window):
        """AC5: F5 key selects Flirtatious emotion."""
        event = create_key_event(Qt.Key.Key_F5)
        main_window.keyPressEvent(event)

        assert main_window.get_emotion_preset() == EmotionPreset.FLIRTATIOUS


class TestEmotionKeyboardShortcutsButtonState:
    """Test button visual state updates with shortcuts."""

    def test_f1_updates_button_checked_state(self, main_window):
        """F1 shortcut updates button checked state to Neutral."""
        # Set to Happy first
        main_window.set_emotion_by_index(1)

        event = create_key_event(Qt.Key.Key_F1)
        main_window.keyPressEvent(event)

        # Verify button is checked
        neutral_button = main_window.emotion_button_group.get_preset_button(EmotionPreset.NEUTRAL)
        assert neutral_button.isChecked()

        # Verify other buttons are not checked
        happy_button = main_window.emotion_button_group.get_preset_button(EmotionPreset.HAPPY)
        assert not happy_button.isChecked()

    def test_shortcut_updates_correct_button(self, main_window):
        """Each shortcut updates the corresponding button's checked state."""
        shortcut_map = {
            Qt.Key.Key_F1: EmotionPreset.NEUTRAL,
            Qt.Key.Key_F2: EmotionPreset.HAPPY,
            Qt.Key.Key_F3: EmotionPreset.SAD,
            Qt.Key.Key_F4: EmotionPreset.ANGRY,
            Qt.Key.Key_F5: EmotionPreset.FLIRTATIOUS,
        }

        for key, expected_preset in shortcut_map.items():
            event = create_key_event(key)
            main_window.keyPressEvent(event)

            # Verify correct button is checked
            for preset in EmotionPreset:
                button = main_window.emotion_button_group.get_preset_button(preset)
                if preset == expected_preset:
                    assert button.isChecked(), f"Expected {preset.display_name} to be checked after {key}"
                else:
                    assert not button.isChecked(), f"Expected {preset.display_name} to NOT be checked after {key}"


class TestEmotionKeyboardShortcutsDisabled:
    """Test shortcuts with emotion disabled (cloned voice) - AC8."""

    def test_f1_f5_no_effect_when_emotion_disabled(self, main_window):
        """AC8: F1-F5 have no effect when emotion control is disabled."""
        # Set initial emotion to Neutral
        main_window.set_emotion_by_index(0)
        assert main_window.get_emotion_preset() == EmotionPreset.NEUTRAL

        # Disable emotion control (simulates cloned voice)
        main_window.set_emotion_enabled(False)

        # Try all shortcuts - should have no effect
        for key in [Qt.Key.Key_F1, Qt.Key.Key_F2, Qt.Key.Key_F3, Qt.Key.Key_F4, Qt.Key.Key_F5]:
            event = create_key_event(key)
            main_window.keyPressEvent(event)

        # Emotion should still be Neutral (unchanged)
        assert main_window.get_emotion_preset() == EmotionPreset.NEUTRAL

    def test_shortcuts_work_after_reenabling_emotion(self, main_window):
        """Shortcuts work again after re-enabling emotion control."""
        # Disable and re-enable
        main_window.set_emotion_enabled(False)
        main_window.set_emotion_enabled(True)

        # F2 should now work
        event = create_key_event(Qt.Key.Key_F2)
        main_window.keyPressEvent(event)

        assert main_window.get_emotion_preset() == EmotionPreset.HAPPY


class TestEmotionKeyboardShortcutsSignals:
    """Test signal emission with keyboard shortcuts."""

    def test_shortcut_emits_emotion_changed_signal(self, main_window):
        """Keyboard shortcut emits emotion_changed signal."""
        signal_mock = Mock()
        main_window.emotion_button_group.emotion_changed.connect(signal_mock)

        # Use F3 to select Sad
        event = create_key_event(Qt.Key.Key_F3)
        main_window.keyPressEvent(event)

        signal_mock.assert_called_once_with(EmotionPreset.SAD)

    def test_no_signal_when_disabled(self, main_window):
        """No signal emitted when emotion is disabled."""
        signal_mock = Mock()
        main_window.emotion_button_group.emotion_changed.connect(signal_mock)

        main_window.set_emotion_enabled(False)

        event = create_key_event(Qt.Key.Key_F2)
        main_window.keyPressEvent(event)

        signal_mock.assert_not_called()


class TestEmotionKeyboardShortcutsWindowScope:
    """Test window-scoped behavior - AC7."""

    def test_shortcut_works_with_text_input_focus(self, main_window):
        """AC7: Shortcuts work even when text input is focused."""
        # Focus text input
        main_window.text_input.setFocus()

        # The keyPressEvent is window-level, so it should still work
        # (In real usage, the text input would receive the event first,
        # but since F-keys are not typically consumed by text edits,
        # they propagate to the window)
        event = create_key_event(Qt.Key.Key_F4)
        main_window.keyPressEvent(event)

        assert main_window.get_emotion_preset() == EmotionPreset.ANGRY


class TestEmotionKeyboardShortcutsInstruct:
    """Test instruct parameter updates with shortcuts."""

    def test_shortcut_updates_instruct_string(self, main_window):
        """Shortcut updates the TTS instruct parameter."""
        event = create_key_event(Qt.Key.Key_F5)
        main_window.keyPressEvent(event)

        instruct = main_window.get_emotion_instruct()
        assert "teas" in instruct.lower() or "flirt" in instruct.lower()

    def test_all_shortcuts_set_correct_instruct(self, main_window):
        """Each shortcut sets the correct instruct string with appropriate keywords."""
        instruct_keywords = {
            Qt.Key.Key_F1: "calm",
            Qt.Key.Key_F2: "joy",
            Qt.Key.Key_F3: "sorrow",
            Qt.Key.Key_F4: "rage",
            Qt.Key.Key_F5: "teas",  # teasing
        }

        for key, keyword in instruct_keywords.items():
            event = create_key_event(key)
            main_window.keyPressEvent(event)

            instruct = main_window.get_emotion_instruct()
            assert keyword in instruct.lower(), f"Key {key} instruct missing '{keyword}'"


class TestEmotionKeyboardShortcutsOtherKeys:
    """Test that other keys are not affected."""

    def test_esc_still_clears_text(self, main_window):
        """ESC key still clears text input."""
        main_window.text_input.setPlainText("Test text")
        assert main_window.text_input.toPlainText() == "Test text"

        event = create_key_event(Qt.Key.Key_Escape)
        main_window.keyPressEvent(event)

        assert main_window.text_input.toPlainText() == ""

    def test_other_function_keys_not_consumed(self, main_window):
        """F6+ keys are not consumed by emotion shortcuts."""
        # Set to Neutral
        main_window.set_emotion_by_index(0)

        # F6 should not change emotion
        event = create_key_event(Qt.Key.Key_F6)
        main_window.keyPressEvent(event)

        # Should still be Neutral
        assert main_window.get_emotion_preset() == EmotionPreset.NEUTRAL

    def test_regular_keys_not_affected(self, main_window):
        """Regular keys like A, B, C are not affected."""
        # Set to Happy
        main_window.set_emotion_by_index(1)

        # Press A key
        event = create_key_event(Qt.Key.Key_A)
        main_window.keyPressEvent(event)

        # Should still be Happy
        assert main_window.get_emotion_preset() == EmotionPreset.HAPPY


class TestEmotionKeyboardShortcutsKeyMapping:
    """Test the F-key to emotion index mapping."""

    def test_correct_key_to_index_mapping(self, main_window):
        """Verify F1-F5 map to indices 0-4 correctly."""
        key_index_map = [
            (Qt.Key.Key_F1, 0, EmotionPreset.NEUTRAL),
            (Qt.Key.Key_F2, 1, EmotionPreset.HAPPY),
            (Qt.Key.Key_F3, 2, EmotionPreset.SAD),
            (Qt.Key.Key_F4, 3, EmotionPreset.ANGRY),
            (Qt.Key.Key_F5, 4, EmotionPreset.FLIRTATIOUS),
        ]

        for key, expected_index, expected_preset in key_index_map:
            # Reset to neutral
            main_window.set_emotion_by_index(0)

            event = create_key_event(key)
            main_window.keyPressEvent(event)

            actual_preset = main_window.get_emotion_preset()
            assert actual_preset == expected_preset, (
                f"Key {key} should map to index {expected_index} ({expected_preset.display_name}), "
                f"but got {actual_preset.display_name}"
            )
