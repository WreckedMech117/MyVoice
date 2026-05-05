"""
Tests for EmotionButtonGroup Emotion Variants Support

QA5: Voice Design Studio Refinements - Emotion Variants
Tests for per-emotion button enable/disable based on EMBEDDING voice available_emotions.

Features tested:
- update_available_emotions() method
- Per-button enable/disable
- Tooltips for unavailable emotions
- Auto-selection of first available emotion
"""

import pytest
from unittest.mock import Mock

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from myvoice.ui.components.emotion_button_group import (
    EmotionButtonGroup,
    EmotionPreset,
)


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
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


class TestUpdateAvailableEmotions:
    """Tests for update_available_emotions() method."""

    def test_all_emotions_available(self, emotion_group):
        """Test all buttons enabled when all emotions available."""
        all_emotions = ["neutral", "happy", "sad", "angry", "flirtatious"]

        emotion_group.update_available_emotions(all_emotions)

        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)
            assert button.isEnabled(), f"{preset.id} should be enabled"

    def test_partial_emotions_available(self, emotion_group):
        """Test only some buttons enabled with partial emotions."""
        available = ["neutral", "happy"]

        emotion_group.update_available_emotions(available)

        # Neutral and Happy should be enabled
        assert emotion_group.get_preset_button(EmotionPreset.NEUTRAL).isEnabled()
        assert emotion_group.get_preset_button(EmotionPreset.HAPPY).isEnabled()

        # Others should be disabled
        assert not emotion_group.get_preset_button(EmotionPreset.SAD).isEnabled()
        assert not emotion_group.get_preset_button(EmotionPreset.ANGRY).isEnabled()
        assert not emotion_group.get_preset_button(EmotionPreset.FLIRTATIOUS).isEnabled()

    def test_only_neutral_available(self, emotion_group):
        """Test only neutral button enabled when only neutral available."""
        available = ["neutral"]

        emotion_group.update_available_emotions(available)

        # Only neutral should be enabled
        assert emotion_group.get_preset_button(EmotionPreset.NEUTRAL).isEnabled()
        assert not emotion_group.get_preset_button(EmotionPreset.HAPPY).isEnabled()
        assert not emotion_group.get_preset_button(EmotionPreset.SAD).isEnabled()
        assert not emotion_group.get_preset_button(EmotionPreset.ANGRY).isEnabled()
        assert not emotion_group.get_preset_button(EmotionPreset.FLIRTATIOUS).isEnabled()

    def test_unavailable_emotion_tooltip(self, emotion_group):
        """Test disabled buttons have explanatory tooltip."""
        available = ["neutral"]
        voice_name = "TestVoice"

        emotion_group.update_available_emotions(available, voice_name)

        # Disabled button should have explanatory tooltip
        happy_button = emotion_group.get_preset_button(EmotionPreset.HAPPY)
        tooltip = happy_button.toolTip()

        assert "Happy" in tooltip
        assert "TestVoice" in tooltip
        assert "not" in tooltip.lower() or "was not" in tooltip.lower()

    def test_available_emotion_tooltip_normal(self, emotion_group):
        """Test enabled buttons have normal tooltip."""
        available = ["neutral", "happy"]

        emotion_group.update_available_emotions(available)

        # Enabled button should have normal tooltip
        happy_button = emotion_group.get_preset_button(EmotionPreset.HAPPY)
        tooltip = happy_button.toolTip()

        assert "Happy" in tooltip
        assert "F2" in tooltip  # Shortcut should be in tooltip

    def test_auto_select_first_available(self, emotion_group):
        """Test auto-selects first available emotion when current is unavailable."""
        # Set to Happy first
        emotion_group.set_preset(EmotionPreset.HAPPY)
        assert emotion_group.get_current_preset() == EmotionPreset.HAPPY

        # Update with only sad and angry available
        available = ["sad", "angry"]

        emotion_group.update_available_emotions(available)

        # Should auto-select sad (first available in VALID_EMOTIONS order)
        assert emotion_group.get_current_preset() == EmotionPreset.SAD

    def test_keeps_current_if_available(self, emotion_group):
        """Test keeps current selection if it's available."""
        # Set to Happy first
        emotion_group.set_preset(EmotionPreset.HAPPY)

        # Update with Happy still available
        available = ["neutral", "happy", "sad"]

        emotion_group.update_available_emotions(available)

        # Should keep Happy selected
        assert emotion_group.get_current_preset() == EmotionPreset.HAPPY

    def test_custom_button_enabled(self, emotion_group):
        """Test Custom... button is enabled after update."""
        available = ["neutral"]

        emotion_group.update_available_emotions(available)

        assert emotion_group.custom_button.isEnabled()

    def test_empty_available_emotions(self, emotion_group):
        """Test handling of empty available emotions list."""
        # This shouldn't normally happen but handle gracefully
        available = []

        emotion_group.update_available_emotions(available)

        # All emotion buttons should be disabled
        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)
            assert not button.isEnabled()

    def test_voice_name_in_tooltip_without_name(self, emotion_group):
        """Test tooltip without voice name uses generic message."""
        available = ["neutral"]

        emotion_group.update_available_emotions(available)  # No voice_name

        # Disabled button should have generic tooltip
        happy_button = emotion_group.get_preset_button(EmotionPreset.HAPPY)
        tooltip = happy_button.toolTip()

        assert "not available" in tooltip.lower()

    def test_signal_emitted_on_auto_select(self, emotion_group):
        """Test emotion_changed signal is emitted when auto-selecting."""
        # Set to Angry
        emotion_group.set_preset(EmotionPreset.ANGRY)

        signal_mock = Mock()
        emotion_group.emotion_changed.connect(signal_mock)

        # Update with only neutral available
        available = ["neutral"]
        emotion_group.update_available_emotions(available)

        # Signal should have been emitted with new selection
        signal_mock.assert_called_once_with(EmotionPreset.NEUTRAL)


class TestEmotionVariantsIntegration:
    """Integration tests for emotion variants with EmotionButtonGroup."""

    def test_switch_between_voice_types(self, emotion_group):
        """Test switching from full to partial emotions."""
        # First: All emotions available (like bundled voice)
        all_emotions = ["neutral", "happy", "sad", "angry", "flirtatious"]
        emotion_group.update_available_emotions(all_emotions)

        for preset in EmotionPreset:
            assert emotion_group.get_preset_button(preset).isEnabled()

        # Second: Only some emotions (like EMBEDDING voice)
        partial_emotions = ["neutral", "happy"]
        emotion_group.update_available_emotions(partial_emotions)

        assert emotion_group.get_preset_button(EmotionPreset.NEUTRAL).isEnabled()
        assert emotion_group.get_preset_button(EmotionPreset.HAPPY).isEnabled()
        assert not emotion_group.get_preset_button(EmotionPreset.SAD).isEnabled()

    def test_re_enable_all_emotions(self, emotion_group):
        """Test can re-enable all emotions after partial."""
        # Start with partial
        emotion_group.update_available_emotions(["neutral"])

        # Then enable all
        all_emotions = ["neutral", "happy", "sad", "angry", "flirtatious"]
        emotion_group.update_available_emotions(all_emotions)

        for preset in EmotionPreset:
            assert emotion_group.get_preset_button(preset).isEnabled()

    def test_selection_preserved_across_updates(self, emotion_group):
        """Test that selection is preserved if emotion remains available."""
        # Select Happy
        emotion_group.set_preset(EmotionPreset.HAPPY)

        # Update with various sets that include Happy
        for emotions in [
            ["neutral", "happy"],
            ["happy", "sad", "angry"],
            ["neutral", "happy", "sad", "angry", "flirtatious"]
        ]:
            emotion_group.update_available_emotions(emotions)
            assert emotion_group.get_current_preset() == EmotionPreset.HAPPY

    def test_get_current_instruct_after_update(self, emotion_group):
        """Test get_current_instruct works after update_available_emotions."""
        emotion_group.update_available_emotions(["neutral", "happy"])
        emotion_group.set_preset(EmotionPreset.HAPPY)

        instruct = emotion_group.get_current_instruct()

        assert instruct is not None
        assert len(instruct) > 0
        # Happy instruct should contain happiness-related words
        assert "joy" in instruct.lower() or "happy" in instruct.lower()
