"""
Tests for Emotion-TTS Integration

Story 3.2: Emotion-TTS Integration
Covers: FR8 - Emotion control passes to TTS service via instruct parameter
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

# Check if torch is available (may crash on some Windows systems due to DLL issues)
try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError, SystemError):
    HAS_TORCH = False


class TestQwenTTSRequestInstruct:
    """Tests for QwenTTSRequest instruct parameter (Story 3.2).

    Note: These tests require torch which may have DLL loading issues on some Windows systems.
    Tests will be skipped if torch cannot be imported.
    """

    def test_qwen_tts_request_accepts_instruct(self):
        """Test QwenTTSRequest can be created with instruct parameter."""
        pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")
        if not HAS_TORCH:
            pytest.skip("torch required for QwenTTSService")
        from myvoice.services.qwen_tts_service import QwenTTSRequest

        request = QwenTTSRequest(
            text="Hello world",
            instruct="Speak with bright, joyful enthusiasm"
        )

        assert "joyful" in request.instruct.lower()

    def test_qwen_tts_request_instruct_defaults_to_none(self):
        """Test QwenTTSRequest instruct defaults to None for neutral emotion."""
        pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")
        if not HAS_TORCH:
            pytest.skip("torch required for QwenTTSService")
        from myvoice.services.qwen_tts_service import QwenTTSRequest

        request = QwenTTSRequest(
            text="Hello world"
        )

        assert request.instruct is None

    def test_qwen_tts_request_with_all_emotion_instructs(self):
        """Test QwenTTSRequest accepts all emotion preset instructs."""
        pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")
        if not HAS_TORCH:
            pytest.skip("torch required for QwenTTSService")
        from myvoice.services.qwen_tts_service import QwenTTSRequest
        from myvoice.ui.components.emotion_button_group import EmotionPreset

        # Use actual instruct strings from EmotionPreset
        for preset in EmotionPreset:
            request = QwenTTSRequest(
                text="Test text",
                instruct=preset.instruct
            )
            assert request.instruct == preset.instruct
            assert len(request.instruct) > 0


class TestEmotionPresetInstruct:
    """Tests for EmotionPreset instruct values."""

    def test_emotion_preset_instruct_values(self):
        """Test all EmotionPreset values have appropriate instruct strings."""
        pytest.importorskip("PyQt6")

        from myvoice.ui.components.emotion_button_group import EmotionPreset

        # Check that each instruct contains relevant emotional keywords
        expected_keywords = {
            EmotionPreset.NEUTRAL: "calm",
            EmotionPreset.HAPPY: "joy",
            EmotionPreset.SAD: "sorrow",
            EmotionPreset.ANGRY: "rage",
            EmotionPreset.FLIRTATIOUS: "teas",  # teasing
        }

        for preset, keyword in expected_keywords.items():
            assert keyword in preset.instruct.lower(), f"Preset {preset.name} missing keyword '{keyword}'"
            assert len(preset.instruct) > 20, f"Preset {preset.name} instruct too short"

    def test_emotion_preset_instructs_match_qwen_service(self):
        """Test EmotionPreset instructs align with QwenTTSService presets."""
        pytest.importorskip("PyQt6")
        if not HAS_TORCH:
            pytest.skip("torch required for QwenTTSService")

        from myvoice.ui.components.emotion_button_group import EmotionPreset
        from myvoice.services.qwen_tts_service import QwenTTSService

        # Map our preset IDs to Qwen service preset keys
        preset_mapping = {
            "neutral": EmotionPreset.NEUTRAL,
            "happy": EmotionPreset.HAPPY,
            "sad": EmotionPreset.SAD,
            "angry": EmotionPreset.ANGRY,
            "flirtatious": EmotionPreset.FLIRTATIOUS,
        }

        for qwen_key, ui_preset in preset_mapping.items():
            qwen_instruct = QwenTTSService.EMOTION_PRESETS.get(qwen_key)

            # Neutral has None in QwenTTSService, expanded instruct in UI
            if qwen_key == "neutral":
                assert qwen_instruct is None
                assert "calm" in ui_preset.instruct.lower()
            else:
                # Other emotions: UI instruct should contain the same meaning
                # QwenTTSService uses more verbose instructions
                assert ui_preset.instruct is not None
                # Both should have the same emotional intent
                assert qwen_key in ui_preset.instruct.lower()


class TestEmotionButtonGroupInstruct:
    """Tests for EmotionButtonGroup instruct retrieval."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_get_current_instruct_neutral(self, qapp):
        """Test get_current_instruct returns correct value for neutral."""
        from myvoice.ui.components.emotion_button_group import EmotionButtonGroup

        group = EmotionButtonGroup()
        # Default is neutral
        instruct = group.get_current_instruct()
        assert "calm" in instruct.lower()

        group.deleteLater()

    def test_get_current_instruct_happy(self, qapp):
        """Test get_current_instruct returns correct value for happy."""
        from myvoice.ui.components.emotion_button_group import (
            EmotionButtonGroup, EmotionPreset
        )

        group = EmotionButtonGroup()
        group.set_preset(EmotionPreset.HAPPY)

        instruct = group.get_current_instruct()
        assert "joy" in instruct.lower()

        group.deleteLater()

    def test_get_current_instruct_all_emotions(self, qapp):
        """Test get_current_instruct for all emotion presets."""
        from myvoice.ui.components.emotion_button_group import (
            EmotionButtonGroup, EmotionPreset
        )

        group = EmotionButtonGroup()

        for preset in EmotionPreset:
            group.set_preset(preset)
            instruct = group.get_current_instruct()
            assert instruct == preset.instruct
            assert instruct is not None
            assert len(instruct) > 0

        group.deleteLater()


class TestMainWindowEmotionInstruct:
    """Tests for MainWindow emotion instruct retrieval."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_main_window_get_emotion_instruct(self, qapp):
        """Test MainWindow.get_emotion_instruct returns correct value."""
        from myvoice.ui.main_window import MainWindow

        window = MainWindow()

        # Default is neutral
        instruct = window.get_emotion_instruct()
        assert "calm" in instruct.lower()

        window.close()
        window.deleteLater()

    def test_main_window_get_emotion_instruct_after_change(self, qapp):
        """Test MainWindow.get_emotion_instruct updates after emotion change."""
        from myvoice.ui.main_window import MainWindow
        from myvoice.ui.components.emotion_button_group import EmotionPreset

        window = MainWindow()

        # Change to happy
        window.set_emotion_preset(EmotionPreset.HAPPY)
        instruct = window.get_emotion_instruct()
        assert "joy" in instruct.lower()

        # Change to sad
        window.set_emotion_preset(EmotionPreset.SAD)
        instruct = window.get_emotion_instruct()
        assert "sorrow" in instruct.lower() or "grief" in instruct.lower()

        window.close()
        window.deleteLater()


class TestVoiceCloneEmotionDisabled:
    """Tests for emotion disabled with cloned voices (FR8)."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_emotion_disabled_for_cloned_voice(self, qapp):
        """Test emotion buttons are disabled for cloned voices."""
        from myvoice.ui.components.emotion_button_group import (
            EmotionButtonGroup, EmotionPreset
        )

        group = EmotionButtonGroup()

        # Disable emotion (simulating cloned voice)
        group.set_emotion_enabled(False)

        # All preset buttons should be disabled
        for preset in EmotionPreset:
            button = group.get_preset_button(preset)
            assert not button.isEnabled()

        group.deleteLater()

    def test_emotion_instruct_still_returns_when_disabled(self, qapp):
        """Test get_current_instruct still works when emotion is disabled."""
        from myvoice.ui.components.emotion_button_group import EmotionButtonGroup

        group = EmotionButtonGroup()

        # Disable emotion
        group.set_emotion_enabled(False)

        # Should still return current instruct (for any legacy code)
        instruct = group.get_current_instruct()
        assert instruct is not None

        group.deleteLater()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available for QwenTTSService import")
class TestQwenTTSServiceEmotionIntegration:
    """Tests for QwenTTSService emotion handling."""

    def test_qwen_service_emotion_presets_defined(self):
        """Test QwenTTSService has EMOTION_PRESETS defined."""
        pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")
        if not HAS_TORCH:
            pytest.skip("torch required for QwenTTSService")
        from myvoice.services.qwen_tts_service import QwenTTSService

        presets = QwenTTSService.EMOTION_PRESETS

        assert "neutral" in presets
        assert "happy" in presets
        assert "sad" in presets
        assert "angry" in presets
        assert "flirtatious" in presets

    def test_qwen_service_neutral_has_no_instruct(self):
        """Test neutral emotion has None instruct (use default voice)."""
        pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")
        if not HAS_TORCH:
            pytest.skip("torch required for QwenTTSService")
        from myvoice.services.qwen_tts_service import QwenTTSService

        assert QwenTTSService.EMOTION_PRESETS["neutral"] is None

    def test_qwen_service_emotions_have_instructs(self):
        """Test non-neutral emotions have instruct strings."""
        pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")
        if not HAS_TORCH:
            pytest.skip("torch required for QwenTTSService")
        from myvoice.services.qwen_tts_service import QwenTTSService

        for emotion in ["happy", "sad", "angry", "flirtatious"]:
            instruct = QwenTTSService.EMOTION_PRESETS[emotion]
            assert instruct is not None
            assert len(instruct) > 0
            assert emotion in instruct.lower()
