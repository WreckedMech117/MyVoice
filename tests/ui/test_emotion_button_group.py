"""
Tests for EmotionButtonGroup Component

Story 3.1: Emotion Button Group UI
Covers: FR6, FR10
"""

import pytest
from unittest.mock import Mock, MagicMock
import sys

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from myvoice.ui.components.emotion_button_group import (
    EmotionButtonGroup,
    EmotionButton,
    EmotionPreset,
)


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def emotion_group(qapp):
    """Create an EmotionButtonGroup instance."""
    group = EmotionButtonGroup()
    yield group
    group.deleteLater()


class TestEmotionPreset:
    """Tests for EmotionPreset enum."""

    def test_preset_count(self):
        """Test that there are exactly 5 presets."""
        presets = list(EmotionPreset)
        assert len(presets) == 5

    def test_preset_order(self):
        """Test preset order matches spec: Neutral, Happy, Sad, Angry, Flirtatious."""
        presets = list(EmotionPreset)
        assert presets[0] == EmotionPreset.NEUTRAL
        assert presets[1] == EmotionPreset.HAPPY
        assert presets[2] == EmotionPreset.SAD
        assert presets[3] == EmotionPreset.ANGRY
        assert presets[4] == EmotionPreset.FLIRTATIOUS

    def test_preset_emojis(self):
        """Test preset emojis match spec: 😐 😄 😢 😠 😏."""
        assert EmotionPreset.NEUTRAL.emoji == "😐"
        assert EmotionPreset.HAPPY.emoji == "😄"
        assert EmotionPreset.SAD.emoji == "😢"
        assert EmotionPreset.ANGRY.emoji == "😠"
        assert EmotionPreset.FLIRTATIOUS.emoji == "😏"

    def test_preset_shortcuts(self):
        """Test preset shortcuts are F1-F5."""
        assert EmotionPreset.NEUTRAL.shortcut == "F1"
        assert EmotionPreset.HAPPY.shortcut == "F2"
        assert EmotionPreset.SAD.shortcut == "F3"
        assert EmotionPreset.ANGRY.shortcut == "F4"
        assert EmotionPreset.FLIRTATIOUS.shortcut == "F5"

    def test_preset_instruct_strings(self):
        """Test preset instruct strings contain appropriate emotional descriptors."""
        assert "calm" in EmotionPreset.NEUTRAL.instruct.lower()
        assert "joy" in EmotionPreset.HAPPY.instruct.lower() or "happy" in EmotionPreset.HAPPY.instruct.lower()
        assert "sorrow" in EmotionPreset.SAD.instruct.lower() or "grief" in EmotionPreset.SAD.instruct.lower()
        assert "rage" in EmotionPreset.ANGRY.instruct.lower() or "angry" in EmotionPreset.ANGRY.instruct.lower()
        assert "flirt" in EmotionPreset.FLIRTATIOUS.instruct.lower() or "teas" in EmotionPreset.FLIRTATIOUS.instruct.lower()

    def test_from_id(self):
        """Test lookup by ID string."""
        assert EmotionPreset.from_id("neutral") == EmotionPreset.NEUTRAL
        assert EmotionPreset.from_id("happy") == EmotionPreset.HAPPY
        assert EmotionPreset.from_id("invalid") is None

    def test_from_index(self):
        """Test lookup by index."""
        assert EmotionPreset.from_index(0) == EmotionPreset.NEUTRAL
        assert EmotionPreset.from_index(1) == EmotionPreset.HAPPY
        assert EmotionPreset.from_index(4) == EmotionPreset.FLIRTATIOUS
        assert EmotionPreset.from_index(-1) is None
        assert EmotionPreset.from_index(5) is None


class TestEmotionButton:
    """Tests for individual EmotionButton widget."""

    def test_button_creation(self, qapp):
        """Test button creates with correct properties."""
        button = EmotionButton(EmotionPreset.HAPPY)

        assert button.text() == "😄"
        assert button.isCheckable()
        assert "emotion_button_happy" in button.objectName()

        button.deleteLater()

    def test_button_size(self, qapp):
        """Test button is 28x28px per UX spec."""
        button = EmotionButton(EmotionPreset.NEUTRAL)

        assert button.width() == 28
        assert button.height() == 28

        button.deleteLater()

    def test_button_tooltip(self, qapp):
        """Test tooltip shows emotion name and shortcut."""
        button = EmotionButton(EmotionPreset.SAD)

        assert "Sad" in button.toolTip()
        assert "F3" in button.toolTip()

        button.deleteLater()


class TestEmotionButtonGroup:
    """Tests for EmotionButtonGroup widget."""

    def test_initial_state_neutral(self, emotion_group):
        """Test neutral is selected by default (FR9)."""
        assert emotion_group.get_current_preset() == EmotionPreset.NEUTRAL

    def test_five_buttons_present(self, emotion_group):
        """Test all 5 emotion buttons are present."""
        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)
            assert button is not None
            assert button.preset == preset

    def test_custom_button_present(self, emotion_group):
        """Test Custom... button is present."""
        assert emotion_group.custom_button is not None
        assert "Custom" in emotion_group.custom_button.text()

    def test_radio_behavior(self, emotion_group):
        """Test only one button can be selected at a time."""
        # Select Happy
        emotion_group.set_preset(EmotionPreset.HAPPY)

        # Verify only Happy is checked
        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)
            if preset == EmotionPreset.HAPPY:
                assert button.isChecked()
            else:
                assert not button.isChecked()

    def test_set_preset(self, emotion_group):
        """Test programmatic preset selection."""
        emotion_group.set_preset(EmotionPreset.ANGRY)
        assert emotion_group.get_current_preset() == EmotionPreset.ANGRY

    def test_set_preset_by_id(self, emotion_group):
        """Test preset selection by ID string."""
        result = emotion_group.set_preset_by_id("sad")
        assert result is True
        assert emotion_group.get_current_preset() == EmotionPreset.SAD

        result = emotion_group.set_preset_by_id("invalid")
        assert result is False

    def test_set_preset_by_index(self, emotion_group):
        """Test preset selection by index."""
        result = emotion_group.set_preset_by_index(4)  # Flirtatious
        assert result is True
        assert emotion_group.get_current_preset() == EmotionPreset.FLIRTATIOUS

        result = emotion_group.set_preset_by_index(99)
        assert result is False

    def test_get_current_instruct(self, emotion_group):
        """Test instruct string retrieval returns expanded emotion prompt."""
        emotion_group.set_preset(EmotionPreset.HAPPY)
        instruct = emotion_group.get_current_instruct()
        assert "joy" in instruct.lower() or "happy" in instruct.lower()

    def test_emotion_changed_signal(self, emotion_group):
        """Test emotion_changed signal is emitted."""
        signal_mock = Mock()
        emotion_group.emotion_changed.connect(signal_mock)

        emotion_group.set_preset(EmotionPreset.SAD)

        signal_mock.assert_called_once_with(EmotionPreset.SAD)

    def test_custom_emotion_requested_signal(self, emotion_group):
        """Test custom_emotion_requested signal is emitted."""
        signal_mock = Mock()
        emotion_group.custom_emotion_requested.connect(signal_mock)

        emotion_group.custom_button.click()

        signal_mock.assert_called_once()

    def test_enable_disable(self, emotion_group):
        """Test enable/disable functionality (FR10)."""
        # Initially enabled
        assert emotion_group.is_emotion_enabled()

        # Disable
        emotion_group.set_emotion_enabled(False)
        assert not emotion_group.is_emotion_enabled()

        # Check all buttons are disabled
        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)
            assert not button.isEnabled()

        # Check tooltip shows cloned voice message
        button = emotion_group.get_preset_button(EmotionPreset.NEUTRAL)
        assert "cloned voices" in button.toolTip().lower()

        # Re-enable
        emotion_group.set_emotion_enabled(True)
        assert emotion_group.is_emotion_enabled()

        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)
            assert button.isEnabled()

    def test_reset_to_default(self, emotion_group):
        """Test reset to default emotion."""
        emotion_group.set_preset(EmotionPreset.ANGRY)
        assert emotion_group.get_current_preset() == EmotionPreset.ANGRY

        emotion_group.reset_to_default()
        assert emotion_group.get_current_preset() == EmotionPreset.NEUTRAL

    def test_no_signal_on_same_selection(self, emotion_group):
        """Test no signal emitted when selecting already-selected preset."""
        emotion_group.set_preset(EmotionPreset.HAPPY)

        signal_mock = Mock()
        emotion_group.emotion_changed.connect(signal_mock)

        # Select same preset again
        emotion_group.set_preset(EmotionPreset.HAPPY)

        # Signal should not be emitted (no change)
        signal_mock.assert_not_called()


class TestEmotionButtonGroupAccessibility:
    """Accessibility tests for EmotionButtonGroup."""

    def test_buttons_have_accessible_names(self, emotion_group):
        """Test buttons have accessible names for screen readers."""
        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)
            accessible_name = button.accessibleName()
            assert accessible_name
            assert preset.display_name.lower() in accessible_name.lower()

    def test_buttons_have_accessible_descriptions(self, emotion_group):
        """Test buttons have accessible descriptions."""
        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)
            accessible_desc = button.accessibleDescription()
            assert accessible_desc
            assert preset.shortcut in accessible_desc
