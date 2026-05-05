"""
Tests for ModelLoadingManager Service

Story 4.5: Model Lazy Loading for Voice Types
Covers: NFR11 (Memory Efficiency)
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from myvoice.services.model_loading_manager import (
    ModelLoadingManager,
    ModelLoadingStatus
)
from myvoice.models.voice_profile import VoiceProfile, VoiceType
from myvoice.models.service_enums import QwenModelType, ModelState


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_tts_service():
    """Create a mock TTS service."""
    service = MagicMock()
    service.get_current_model_type.return_value = QwenModelType.CUSTOM_VOICE
    service.is_model_loaded.return_value = True
    service.ensure_model_loaded = AsyncMock(return_value=(True, None))
    return service


@pytest.fixture
def manager(qapp, mock_tts_service):
    """Create a ModelLoadingManager instance."""
    manager = ModelLoadingManager(tts_service=mock_tts_service)
    yield manager


@pytest.fixture
def bundled_profile():
    """Create a bundled voice profile."""
    profile = MagicMock(spec=VoiceProfile)
    profile.name = "Ryan"
    profile.voice_type = VoiceType.BUNDLED
    profile.file_path = Path("/bundled/ryan.wav")
    return profile


@pytest.fixture
def designed_profile():
    """Create a designed voice profile."""
    profile = MagicMock(spec=VoiceProfile)
    profile.name = "Custom Voice"
    profile.voice_type = VoiceType.DESIGNED
    profile.file_path = Path("/designed/custom/voice.wav")
    return profile


@pytest.fixture
def cloned_profile():
    """Create a cloned voice profile."""
    profile = MagicMock(spec=VoiceProfile)
    profile.name = "My Clone"
    profile.voice_type = VoiceType.CLONED
    profile.file_path = Path("/cloned/myclone/voice.wav")
    return profile


class TestModelLoadingManagerInit:
    """Tests for ModelLoadingManager initialization."""

    def test_manager_creation(self, manager):
        """Test manager creates successfully."""
        assert manager is not None
        assert isinstance(manager, ModelLoadingManager)

    def test_set_tts_service(self, qapp):
        """Test setting TTS service."""
        manager = ModelLoadingManager()
        mock_service = MagicMock()

        manager.set_tts_service(mock_service)

        assert manager._tts_service == mock_service


class TestVoiceTypeRequiredModel:
    """Tests for VoiceType.required_model property."""

    def test_bundled_requires_custom_voice(self):
        """Test bundled voice requires CustomVoice model."""
        assert VoiceType.BUNDLED.required_model == QwenModelType.CUSTOM_VOICE

    def test_designed_requires_voice_design(self):
        """Test designed voice requires VoiceDesign model."""
        assert VoiceType.DESIGNED.required_model == QwenModelType.VOICE_DESIGN

    def test_cloned_requires_base(self):
        """Test cloned voice requires Base model."""
        assert VoiceType.CLONED.required_model == QwenModelType.BASE


class TestModelSwitchDetection:
    """Tests for model switch detection."""

    def test_no_switch_same_model_type(self, manager, bundled_profile):
        """Test no switch needed for same model type."""
        # CustomVoice is already loaded
        assert not manager.requires_model_switch(bundled_profile)

    def test_switch_needed_different_model_type(self, manager, designed_profile):
        """Test switch needed for different model type."""
        # CustomVoice loaded, but designed requires VoiceDesign
        assert manager.requires_model_switch(designed_profile)

    def test_switch_needed_for_cloned(self, manager, cloned_profile):
        """Test switch needed for cloned voice."""
        # CustomVoice loaded, but cloned requires Base
        assert manager.requires_model_switch(cloned_profile)

    def test_switch_needed_when_no_model_loaded(self, qapp, bundled_profile):
        """Test switch needed when no model is loaded."""
        manager = ModelLoadingManager()
        mock_service = MagicMock()
        mock_service.get_current_model_type.return_value = None
        manager.set_tts_service(mock_service)

        assert manager.requires_model_switch(bundled_profile)


class TestInstantSwitchDetection:
    """Tests for instant voice switch detection."""

    def test_instant_switch_same_model(self, manager, bundled_profile):
        """Test instant switch for same model type."""
        assert manager.will_voice_switch_be_instant(bundled_profile)

    def test_not_instant_switch_different_model(self, manager, designed_profile):
        """Test not instant for different model type."""
        assert not manager.will_voice_switch_be_instant(designed_profile)


class TestLoadingMessage:
    """Tests for loading message generation."""

    def test_no_message_instant_switch(self, manager, bundled_profile):
        """Test no message for instant switch."""
        message = manager.get_loading_message_for_voice(bundled_profile)
        assert message is None

    def test_message_for_model_switch(self, manager, designed_profile):
        """Test message generated for model switch."""
        message = manager.get_loading_message_for_voice(designed_profile)
        assert message is not None
        assert "VoiceDesign" in message or "Loading" in message


class TestModelLoadingStatus:
    """Tests for ModelLoadingStatus dataclass."""

    def test_status_creation(self):
        """Test status object creation."""
        status = ModelLoadingStatus(
            is_loading=True,
            model_type=QwenModelType.VOICE_DESIGN,
            progress_percent=50.0,
            message="Loading...",
            is_ready=False
        )
        assert status.is_loading
        assert status.model_type == QwenModelType.VOICE_DESIGN
        assert status.progress_percent == 50.0

    def test_status_with_error(self):
        """Test status with error."""
        status = ModelLoadingStatus(
            is_loading=False,
            model_type=None,
            progress_percent=0,
            message="Failed",
            is_ready=False,
            error="Model not found"
        )
        assert status.error == "Model not found"


class TestLoadingStatusRetrieval:
    """Tests for loading status retrieval."""

    def test_get_loading_status(self, manager):
        """Test get_loading_status returns valid status."""
        status = manager.get_loading_status()
        assert isinstance(status, ModelLoadingStatus)

    def test_is_loading_method(self, manager):
        """Test is_loading method."""
        assert not manager.is_loading()


class TestModelLoadingManagerSignals:
    """Tests for manager signals."""

    def test_loading_started_signal_exists(self, manager):
        """Test loading_started signal exists."""
        assert hasattr(manager, 'loading_started')

    def test_loading_progress_signal_exists(self, manager):
        """Test loading_progress signal exists."""
        assert hasattr(manager, 'loading_progress')

    def test_loading_complete_signal_exists(self, manager):
        """Test loading_complete signal exists."""
        assert hasattr(manager, 'loading_complete')

    def test_model_ready_signal_exists(self, manager):
        """Test model_ready signal exists."""
        assert hasattr(manager, 'model_ready')
