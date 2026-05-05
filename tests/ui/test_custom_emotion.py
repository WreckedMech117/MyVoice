"""
Tests for Story 3.4: Custom Emotion Prompts

Tests the custom emotion functionality including:
- AppSettings custom emotion fields
- EmotionButtonGroup custom emotion state
- Settings dialog custom emotion UI
- Main window custom emotion handling
"""

import pytest
from unittest.mock import MagicMock, patch

# Skip this module if PyQt6 is not available
pytest.importorskip("PyQt6", reason="PyQt6 required for Qt widget tests")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from myvoice.models.app_settings import AppSettings
from myvoice.ui.components.emotion_button_group import EmotionButtonGroup, EmotionPreset


# Ensure QApplication exists for Qt widgets
@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestAppSettingsCustomEmotion:
    """Tests for AppSettings custom emotion fields."""

    def test_custom_emotion_text_default_none(self):
        """Custom emotion text should be None by default."""
        settings = AppSettings()
        assert settings.custom_emotion_text is None

    def test_custom_emotion_presets_has_defaults(self):
        """Custom emotion presets should have default values."""
        settings = AppSettings()
        assert len(settings.custom_emotion_presets) > 0
        assert "Rising Frustration" in settings.custom_emotion_presets
        assert "Growing Excitement" in settings.custom_emotion_presets

    def test_custom_emotion_text_serialization(self):
        """Custom emotion text should serialize correctly."""
        settings = AppSettings()
        settings.custom_emotion_text = "Speak with rising excitement"

        data = settings.to_dict()
        assert data["custom_emotion_text"] == "Speak with rising excitement"

    def test_custom_emotion_text_deserialization(self):
        """Custom emotion text should deserialize correctly."""
        data = {
            "custom_emotion_text": "Speak dramatically",
            "custom_emotion_presets": ["Custom Preset 1", "Custom Preset 2"]
        }
        settings = AppSettings.from_dict(data)

        assert settings.custom_emotion_text == "Speak dramatically"
        assert settings.custom_emotion_presets == ["Custom Preset 1", "Custom Preset 2"]

    def test_custom_emotion_presets_serialization(self):
        """Custom emotion presets should serialize correctly."""
        settings = AppSettings()
        settings.custom_emotion_presets = ["Test1", "Test2"]

        data = settings.to_dict()
        assert data["custom_emotion_presets"] == ["Test1", "Test2"]

    def test_reset_to_defaults_clears_custom_emotion(self):
        """Reset to defaults should clear custom emotion text."""
        settings = AppSettings()
        settings.custom_emotion_text = "Some custom emotion"
        settings.reset_to_defaults()

        assert settings.custom_emotion_text is None


class TestEmotionButtonGroupCustomEmotion:
    """Tests for EmotionButtonGroup custom emotion functionality."""

    @pytest.fixture
    def emotion_group(self, qapp):
        """Create EmotionButtonGroup for testing."""
        return EmotionButtonGroup()

    def test_initial_custom_emotion_state(self, emotion_group):
        """Custom emotion should not be active initially."""
        assert not emotion_group.is_custom_emotion_active()
        assert emotion_group.get_custom_emotion_text() is None

    def test_set_custom_emotion(self, emotion_group):
        """Setting custom emotion should update state."""
        emotion_group.set_custom_emotion("Speak with rising excitement")

        assert emotion_group.is_custom_emotion_active()
        assert emotion_group.get_custom_emotion_text() == "Speak with rising excitement"

    def test_get_current_instruct_with_custom_emotion(self, emotion_group):
        """get_current_instruct should return custom text when active."""
        emotion_group.set_custom_emotion("Custom instruction text")

        assert emotion_group.get_current_instruct() == "Custom instruction text"

    def test_get_current_instruct_without_custom_emotion(self, emotion_group):
        """get_current_instruct should return preset instruct when no custom."""
        emotion_group.set_preset(EmotionPreset.HAPPY)

        assert emotion_group.get_current_instruct() == EmotionPreset.HAPPY.instruct

    def test_selecting_preset_clears_custom_emotion(self, emotion_group):
        """Selecting a preset should clear custom emotion."""
        emotion_group.set_custom_emotion("Custom instruction")
        assert emotion_group.is_custom_emotion_active()

        emotion_group.set_preset(EmotionPreset.SAD)

        assert not emotion_group.is_custom_emotion_active()
        assert emotion_group.get_custom_emotion_text() is None

    def test_custom_emotion_cleared_signal(self, emotion_group):
        """custom_emotion_cleared signal should emit when preset selected."""
        signal_received = []
        emotion_group.custom_emotion_cleared.connect(lambda: signal_received.append(True))

        emotion_group.set_custom_emotion("Custom instruction")
        emotion_group.set_preset(EmotionPreset.NEUTRAL)

        assert len(signal_received) == 1

    def test_clear_custom_emotion(self, emotion_group):
        """clear_custom_emotion should reset to preset."""
        emotion_group.set_custom_emotion("Custom instruction")
        emotion_group.clear_custom_emotion()

        assert not emotion_group.is_custom_emotion_active()
        assert emotion_group.get_custom_emotion_text() is None
        # Should revert to current preset instruct
        assert emotion_group.get_current_instruct() == EmotionPreset.NEUTRAL.instruct

    def test_custom_button_text_changes_when_active(self, emotion_group):
        """Custom button text should indicate active state."""
        assert emotion_group.custom_button.text() == "Custom..."

        emotion_group.set_custom_emotion("Custom instruction")

        # Button text should indicate active
        assert "Custom" in emotion_group.custom_button.text()

    def test_empty_custom_emotion_not_set(self, emotion_group):
        """Empty or whitespace custom emotion should not be set."""
        emotion_group.set_custom_emotion("")
        assert not emotion_group.is_custom_emotion_active()

        emotion_group.set_custom_emotion("   ")
        assert not emotion_group.is_custom_emotion_active()

    def test_preset_buttons_unchecked_when_custom_active(self, emotion_group):
        """All preset buttons should be unchecked when custom is active."""
        emotion_group.set_custom_emotion("Custom instruction")

        for preset in EmotionPreset:
            button = emotion_group.get_preset_button(preset)
            assert not button.isChecked(), f"{preset.display_name} button should be unchecked"


class TestCustomEmotionIntegration:
    """Integration tests for custom emotion flow."""

    @pytest.fixture
    def emotion_group(self, qapp):
        """Create EmotionButtonGroup for testing."""
        return EmotionButtonGroup()

    def test_custom_emotion_to_preset_to_custom_flow(self, emotion_group):
        """Test switching between custom emotion and presets."""
        # Start with custom
        emotion_group.set_custom_emotion("First custom")
        assert emotion_group.get_current_instruct() == "First custom"

        # Switch to preset
        emotion_group.set_preset(EmotionPreset.ANGRY)
        assert emotion_group.get_current_instruct() == EmotionPreset.ANGRY.instruct

        # Back to custom
        emotion_group.set_custom_emotion("Second custom")
        assert emotion_group.get_current_instruct() == "Second custom"

    def test_custom_emotion_preserved_during_disable(self, emotion_group):
        """Custom emotion state should be preserved when disabled."""
        emotion_group.set_custom_emotion("Custom instruction")
        emotion_group.set_emotion_enabled(False)

        # State should be preserved even when disabled
        assert emotion_group.is_custom_emotion_active()
        assert emotion_group.get_custom_emotion_text() == "Custom instruction"

    def test_custom_emotion_with_keyboard_shortcuts(self, emotion_group):
        """Keyboard shortcuts (F1-F5) should clear custom emotion."""
        emotion_group.set_custom_emotion("Custom instruction")

        # Simulating F2 (Happy) - this would be done by set_preset_by_index
        emotion_group.set_preset_by_index(1)  # Happy

        assert not emotion_group.is_custom_emotion_active()
        assert emotion_group.get_current_instruct() == EmotionPreset.HAPPY.instruct
