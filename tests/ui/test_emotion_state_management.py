"""
Tests for Emotion State Management

Story 3.3: Emotion State Management
Covers: FR9 (persist emotion selection), FR8a (disable for cloned voices)
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# Skip this module if PyQt6 is not available (required for myvoice imports)
pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")


class TestVoiceProfileEmotionCapable:
    """Tests for VoiceProfile.emotion_capable attribute (FR8a)."""

    def test_voice_profile_has_emotion_capable_attribute(self, tmp_path):
        """Test VoiceProfile has emotion_capable attribute."""
        from myvoice.models.voice_profile import VoiceProfile

        # Create a minimal WAV file
        voice_file = tmp_path / "test_voice.wav"
        # Minimal valid WAV header (44 bytes) + some data
        wav_header = (
            b'RIFF' +
            (44 + 1000).to_bytes(4, 'little') +
            b'WAVE' +
            b'fmt ' +
            (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') +  # PCM
            (1).to_bytes(2, 'little') +  # Mono
            (16000).to_bytes(4, 'little') +  # Sample rate
            (32000).to_bytes(4, 'little') +  # Byte rate
            (2).to_bytes(2, 'little') +  # Block align
            (16).to_bytes(2, 'little') +  # Bits per sample
            b'data' +
            (1000).to_bytes(4, 'little') +
            b'\x00' * 1000
        )
        voice_file.write_bytes(wav_header)

        profile = VoiceProfile(file_path=voice_file, name="Test Voice")

        assert hasattr(profile, 'emotion_capable')

    def test_cloned_voice_emotion_capable_false_by_default(self, tmp_path):
        """Test cloned voice profiles default to emotion_capable=False."""
        from myvoice.models.voice_profile import VoiceProfile

        voice_file = tmp_path / "cloned_voice.wav"
        wav_header = (
            b'RIFF' +
            (44 + 1000).to_bytes(4, 'little') +
            b'WAVE' +
            b'fmt ' +
            (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') +
            (1).to_bytes(2, 'little') +
            (16000).to_bytes(4, 'little') +
            (32000).to_bytes(4, 'little') +
            (2).to_bytes(2, 'little') +
            (16).to_bytes(2, 'little') +
            b'data' +
            (1000).to_bytes(4, 'little') +
            b'\x00' * 1000
        )
        voice_file.write_bytes(wav_header)

        profile = VoiceProfile(file_path=voice_file, name="Cloned Voice")

        # Cloned voices (VoiceProfile) should default to emotion_capable=False
        assert profile.emotion_capable is False

    def test_voice_profile_emotion_capable_serialization(self, tmp_path):
        """Test emotion_capable is serialized in to_dict."""
        from myvoice.models.voice_profile import VoiceProfile

        voice_file = tmp_path / "voice.wav"
        wav_header = (
            b'RIFF' +
            (44 + 1000).to_bytes(4, 'little') +
            b'WAVE' +
            b'fmt ' +
            (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') +
            (1).to_bytes(2, 'little') +
            (16000).to_bytes(4, 'little') +
            (32000).to_bytes(4, 'little') +
            (2).to_bytes(2, 'little') +
            (16).to_bytes(2, 'little') +
            b'data' +
            (1000).to_bytes(4, 'little') +
            b'\x00' * 1000
        )
        voice_file.write_bytes(wav_header)

        # Story 4.1: emotion_capable is now auto-synced from voice_type
        # To get emotion_capable=True, use BUNDLED or DESIGNED voice type
        from myvoice.models.voice_profile import VoiceType
        profile = VoiceProfile(file_path=voice_file, name="Test", voice_type=VoiceType.BUNDLED)
        data = profile.to_dict()

        assert 'emotion_capable' in data
        assert data['emotion_capable'] is True

    def test_voice_profile_emotion_capable_deserialization(self, tmp_path):
        """Test emotion_capable is deserialized from from_dict."""
        from myvoice.models.voice_profile import VoiceProfile

        voice_file = tmp_path / "voice.wav"
        wav_header = (
            b'RIFF' +
            (44 + 1000).to_bytes(4, 'little') +
            b'WAVE' +
            b'fmt ' +
            (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') +
            (1).to_bytes(2, 'little') +
            (16000).to_bytes(4, 'little') +
            (32000).to_bytes(4, 'little') +
            (2).to_bytes(2, 'little') +
            (16).to_bytes(2, 'little') +
            b'data' +
            (1000).to_bytes(4, 'little') +
            b'\x00' * 1000
        )
        voice_file.write_bytes(wav_header)

        # Story 4.1: emotion_capable is now auto-synced from voice_type
        # To get emotion_capable=True, use BUNDLED or DESIGNED voice type
        data = {
            'file_path': str(voice_file),
            'name': 'Test Voice',
            'voice_type': 'bundled',  # BUNDLED voices have emotion_capable=True
        }

        profile = VoiceProfile.from_dict(data)

        assert profile.emotion_capable is True

    def test_voice_profile_emotion_capable_default_on_old_data(self, tmp_path):
        """Test emotion_capable defaults to False when missing from old data."""
        from myvoice.models.voice_profile import VoiceProfile

        voice_file = tmp_path / "voice.wav"
        wav_header = (
            b'RIFF' +
            (44 + 1000).to_bytes(4, 'little') +
            b'WAVE' +
            b'fmt ' +
            (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') +
            (1).to_bytes(2, 'little') +
            (16000).to_bytes(4, 'little') +
            (32000).to_bytes(4, 'little') +
            (2).to_bytes(2, 'little') +
            (16).to_bytes(2, 'little') +
            b'data' +
            (1000).to_bytes(4, 'little') +
            b'\x00' * 1000
        )
        voice_file.write_bytes(wav_header)

        # Old data without emotion_capable field
        data = {
            'file_path': str(voice_file),
            'name': 'Old Voice',
        }

        profile = VoiceProfile.from_dict(data)

        # Should default to False for backwards compatibility
        assert profile.emotion_capable is False


class TestUIStateEmotionFields:
    """Tests for UIState emotion fields (FR9)."""

    def test_ui_state_has_current_emotion(self):
        """Test UIState has current_emotion field."""
        from myvoice.models.ui_state import UIState

        state = UIState()

        assert hasattr(state, 'current_emotion')
        assert state.current_emotion == "neutral"

    def test_ui_state_has_emotion_enabled(self):
        """Test UIState has emotion_enabled field."""
        from myvoice.models.ui_state import UIState

        state = UIState()

        assert hasattr(state, 'emotion_enabled')
        assert state.emotion_enabled is True

    def test_ui_state_emotion_serialization(self):
        """Test emotion fields are serialized in to_dict."""
        from myvoice.models.ui_state import UIState

        state = UIState()
        state.current_emotion = "happy"
        state.emotion_enabled = False

        data = state.to_dict()

        assert 'current_emotion' in data
        assert data['current_emotion'] == "happy"
        assert 'emotion_enabled' in data
        assert data['emotion_enabled'] is False


class TestQwenModelTypeSupportsEmotion:
    """Tests for QwenModelType.supports_emotion property."""

    def test_custom_voice_supports_emotion(self):
        """Test CustomVoice model supports emotion."""
        from myvoice.models.service_enums import QwenModelType

        assert QwenModelType.CUSTOM_VOICE.supports_emotion is True

    def test_voice_design_supports_emotion(self):
        """Test VoiceDesign model supports emotion."""
        from myvoice.models.service_enums import QwenModelType

        assert QwenModelType.VOICE_DESIGN.supports_emotion is True

    def test_base_model_no_emotion_support(self):
        """Test Base model (voice cloning) does NOT support emotion."""
        from myvoice.models.service_enums import QwenModelType

        assert QwenModelType.BASE.supports_emotion is False


class TestEmotionButtonGroupEnableDisable:
    """Tests for EmotionButtonGroup enable/disable functionality."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_emotion_group_disabled_for_cloned_voice(self, qapp):
        """Test emotion buttons are disabled for cloned voices."""
        from myvoice.ui.components.emotion_button_group import (
            EmotionButtonGroup, EmotionPreset
        )

        group = EmotionButtonGroup()

        # Simulate cloned voice (emotion_capable=False)
        group.set_emotion_enabled(False)

        # All buttons should be disabled
        for preset in EmotionPreset:
            button = group.get_preset_button(preset)
            assert not button.isEnabled(), f"Button {preset.name} should be disabled"

        group.deleteLater()

    def test_emotion_group_tooltip_when_disabled(self, qapp):
        """Test tooltip shows cloned voice message when disabled."""
        from myvoice.ui.components.emotion_button_group import (
            EmotionButtonGroup, EmotionPreset
        )

        group = EmotionButtonGroup()
        group.set_emotion_enabled(False)

        button = group.get_preset_button(EmotionPreset.NEUTRAL)
        tooltip = button.toolTip().lower()

        assert "cloned" in tooltip or "not available" in tooltip

        group.deleteLater()

    def test_emotion_group_re_enabled_for_custom_voice(self, qapp):
        """Test emotion buttons are re-enabled when switching to CustomVoice."""
        from myvoice.ui.components.emotion_button_group import (
            EmotionButtonGroup, EmotionPreset
        )

        group = EmotionButtonGroup()

        # First disable (cloned voice)
        group.set_emotion_enabled(False)

        # Then re-enable (CustomVoice)
        group.set_emotion_enabled(True)

        # All buttons should be enabled
        for preset in EmotionPreset:
            button = group.get_preset_button(preset)
            assert button.isEnabled(), f"Button {preset.name} should be enabled"

        group.deleteLater()

    def test_emotion_selection_persists_while_disabled(self, qapp):
        """Test emotion selection is preserved when disabled."""
        from myvoice.ui.components.emotion_button_group import (
            EmotionButtonGroup, EmotionPreset
        )

        group = EmotionButtonGroup()

        # Select happy emotion
        group.set_preset(EmotionPreset.HAPPY)
        assert group.get_current_preset() == EmotionPreset.HAPPY

        # Disable (cloned voice)
        group.set_emotion_enabled(False)

        # Selection should still be happy
        assert group.get_current_preset() == EmotionPreset.HAPPY

        # Re-enable
        group.set_emotion_enabled(True)

        # Selection should still be happy
        assert group.get_current_preset() == EmotionPreset.HAPPY

        group.deleteLater()


class TestMainWindowEmotionIntegration:
    """Tests for MainWindow emotion enable/disable integration."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_main_window_set_emotion_enabled(self, qapp):
        """Test MainWindow.set_emotion_enabled method exists."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        assert hasattr(window, 'set_emotion_enabled')

        # Should not raise
        window.set_emotion_enabled(False)
        window.set_emotion_enabled(True)

        window.close()
        window.deleteLater()

    def test_main_window_emotion_disabled_affects_buttons(self, qapp):
        """Test MainWindow emotion disable affects EmotionButtonGroup."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        # Disable emotion
        window.set_emotion_enabled(False)

        # Check if emotion button group reflects disabled state
        assert not window.emotion_button_group.is_emotion_enabled()

        window.close()
        window.deleteLater()


class TestEmotionStateManagementIntegration:
    """Integration tests for emotion state management flow."""

    def test_cloned_voice_disables_emotion_flow(self, tmp_path):
        """Test flow: cloned voice selected -> emotions disabled."""
        from myvoice.models.voice_profile import VoiceProfile

        # Create a cloned voice profile
        voice_file = tmp_path / "cloned.wav"
        wav_header = (
            b'RIFF' +
            (44 + 1000).to_bytes(4, 'little') +
            b'WAVE' +
            b'fmt ' +
            (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') +
            (1).to_bytes(2, 'little') +
            (16000).to_bytes(4, 'little') +
            (32000).to_bytes(4, 'little') +
            (2).to_bytes(2, 'little') +
            (16).to_bytes(2, 'little') +
            b'data' +
            (1000).to_bytes(4, 'little') +
            b'\x00' * 1000
        )
        voice_file.write_bytes(wav_header)

        profile = VoiceProfile(file_path=voice_file, name="Cloned")

        # Cloned voice should have emotion_capable=False
        assert profile.emotion_capable is False

        # When this voice is selected, emotion controls should be disabled
        # This is tested in the app.py _on_voice_profile_set callback

    def test_emotion_preset_values_complete(self):
        """Test all emotion presets have valid values."""
        pytest.importorskip("PyQt6")

        from myvoice.ui.components.emotion_button_group import EmotionPreset

        expected_presets = ["neutral", "happy", "sad", "angry", "flirtatious"]

        for preset_id in expected_presets:
            preset = EmotionPreset.from_id(preset_id)
            assert preset is not None, f"Preset {preset_id} not found"
            assert preset.id == preset_id
            assert preset.emoji is not None
            assert preset.display_name is not None
            assert preset.shortcut is not None
            assert preset.instruct is not None
