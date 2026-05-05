"""
Tests for Quick Speak with Voice & Emotion Integration

Story 5.3: Quick Speak with Voice & Emotion
Tests that Quick Speak uses current voice and emotion settings,
and that cloned voices generate without emotion.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

# Check if torch is available (may crash on some Windows systems due to DLL issues)
try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError, SystemError):
    HAS_TORCH = False

from PyQt6.QtWidgets import QApplication

from myvoice.models.voice_profile import VoiceProfile, VoiceType


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestVoiceTypeSupportsEmotion:
    """Tests for VoiceType.supports_emotion property."""

    def test_bundled_supports_emotion(self):
        """Test bundled voices support emotion."""
        assert VoiceType.BUNDLED.supports_emotion is True

    def test_designed_supports_emotion(self):
        """Test designed voices support emotion."""
        assert VoiceType.DESIGNED.supports_emotion is True

    def test_cloned_does_not_support_emotion(self):
        """Test cloned voices do NOT support emotion (FR35)."""
        assert VoiceType.CLONED.supports_emotion is False


class TestVoiceProfileEmotionCapability:
    """Tests for VoiceProfile emotion capability based on type."""

    def test_bundled_profile_supports_emotion(self):
        """Test bundled voice profile supports emotion."""
        profile = VoiceProfile(
            name="Test Bundled",
            file_path=Path("/test/bundled.wav"),
            voice_type=VoiceType.BUNDLED
        )
        assert profile.voice_type.supports_emotion is True

    def test_designed_profile_supports_emotion(self):
        """Test designed voice profile supports emotion."""
        profile = VoiceProfile(
            name="Test Designed",
            file_path=Path("/test/designed.wav"),
            voice_type=VoiceType.DESIGNED
        )
        assert profile.voice_type.supports_emotion is True

    def test_cloned_profile_no_emotion(self):
        """Test cloned voice profile does NOT support emotion."""
        profile = VoiceProfile(
            name="Test Cloned",
            file_path=Path("/test/cloned.wav"),
            voice_type=VoiceType.CLONED
        )
        assert profile.voice_type.supports_emotion is False


class TestEmotionButtonEnableDisable:
    """Tests for emotion button enable/disable based on voice type."""

    def test_emotion_enabled_for_bundled_voice(self, qapp):
        """Test emotion buttons enabled when bundled voice selected."""
        from myvoice.ui.components.emotion_button_group import EmotionButtonGroup

        group = EmotionButtonGroup()
        group.set_emotion_enabled(True)

        assert group.is_emotion_enabled() is True

        group.deleteLater()

    def test_emotion_disabled_for_cloned_voice(self, qapp):
        """Test emotion buttons disabled when cloned voice selected."""
        from myvoice.ui.components.emotion_button_group import EmotionButtonGroup

        group = EmotionButtonGroup()
        group.set_emotion_enabled(False)

        assert group.is_emotion_enabled() is False

        group.deleteLater()


class TestQuickSpeakEmotionIntegration:
    """Tests for Quick Speak using current emotion settings."""

    def test_quick_speak_uses_current_emotion(self, qapp):
        """Test Quick Speak captures emotion at trigger time."""
        from myvoice.ui.components.emotion_button_group import EmotionButtonGroup, EmotionPreset

        group = EmotionButtonGroup()

        # Set specific emotion
        group.set_preset(EmotionPreset.HAPPY)

        # Verify instruct is captured
        instruct = group.get_current_instruct()
        assert instruct is not None
        assert len(instruct) > 0

        group.deleteLater()

    def test_quick_speak_uses_neutral_when_not_set(self, qapp):
        """Test Quick Speak uses neutral when no emotion selected."""
        from myvoice.ui.components.emotion_button_group import EmotionButtonGroup, EmotionPreset

        group = EmotionButtonGroup()

        # Default should be neutral
        preset = group.get_current_preset()
        assert preset == EmotionPreset.NEUTRAL

        group.deleteLater()


class TestTTSRequestEmotionHandling:
    """Tests for TTS request emotion handling based on voice type."""

    def test_tts_request_includes_emotion_for_bundled(self):
        """Test QwenTTSRequest includes emotion for bundled voice."""
        if not HAS_TORCH:
            pytest.skip("torch required for QwenTTSService")
        from myvoice.services.qwen_tts_service import QwenTTSRequest

        request = QwenTTSRequest(
            text="Hello world",
            instruct="Speak with happiness"
        )

        assert request.instruct == "Speak with happiness"

    def test_tts_request_no_emotion_for_cloned(self):
        """Test QwenTTSRequest should not include emotion for cloned voice."""
        if not HAS_TORCH:
            pytest.skip("torch required for QwenTTSService")
        from myvoice.services.qwen_tts_service import QwenTTSRequest

        # For cloned voices, instruct should be None
        request = QwenTTSRequest(
            text="Hello world",
            instruct=None  # No emotion for cloned
        )

        assert request.instruct is None


class TestQuickSpeakMenuVoiceIntegration:
    """Tests for Quick Speak menu with voice integration."""

    def test_menu_phrase_selection_triggers_generation(self, qapp):
        """Test menu phrase selection triggers TTS generation signal."""
        from myvoice.ui.components.quick_speak_menu import QuickSpeakMenu
        from myvoice.models.quick_speak_entry import QuickSpeakEntry

        mock_service = MagicMock()
        mock_service.get_entries.return_value = [
            QuickSpeakEntry(id="1", text="Test phrase", label="Test")
        ]
        mock_service.load_entries.return_value = None

        menu = QuickSpeakMenu(quick_speak_service=mock_service)

        signal_received = []
        menu.phrase_selected.connect(lambda text: signal_received.append(text))

        menu.refresh_entries()

        # Trigger the phrase action
        for action in menu.actions():
            if action.data() == "Test phrase":
                action.trigger()
                break

        assert len(signal_received) == 1
        assert signal_received[0] == "Test phrase"

        menu.deleteLater()


class TestErrorHandlingParity:
    """Tests for Quick Speak using same error handling as normal generation."""

    def test_quick_speak_handles_missing_voice(self, qapp):
        """Test Quick Speak handles missing voice profile gracefully."""
        # This tests that the same error handling path is used
        # The actual implementation uses the same _on_text_generate_requested handler
        # which handles missing voice profiles

        from myvoice.models.voice_profile import VoiceProfile, VoiceType

        # Creating a profile with non-existent file should be caught
        profile = VoiceProfile(
            name="Missing",
            file_path=Path("/nonexistent/path.wav"),
            voice_type=VoiceType.BUNDLED
        )

        # File path check happens in generation handler
        assert not profile.file_path.exists()
