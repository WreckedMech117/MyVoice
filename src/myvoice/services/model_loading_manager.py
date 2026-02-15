"""
Model Loading Manager

This module provides a high-level manager for coordinating TTS model loading
with UI feedback. It bridges the ModelRegistry with the UI layer to provide
user-friendly loading feedback.

Story 4.5: Model Lazy Loading for Voice Types
- Lazy loading: Models loaded only when needed
- Single model at a time (memory constraint ~3.4GB)
- Same-type voice switch is instant (no model reload)
- Loading feedback: "Loading voice model..."
"""

import logging
from typing import Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal

from myvoice.models.service_enums import ModelState, QwenModelType
from myvoice.models.voice_profile import VoiceProfile, VoiceType

if TYPE_CHECKING:
    from myvoice.services.qwen_tts_service import QwenTTSService


@dataclass
class ModelLoadingStatus:
    """Status information for model loading."""
    is_loading: bool
    model_type: Optional[QwenModelType]
    progress_percent: float
    message: str
    is_ready: bool
    error: Optional[str] = None


class ModelLoadingManager(QObject):
    """
    Manager for coordinating TTS model loading with UI feedback.

    This class provides a high-level interface for:
    - Checking if a voice requires model loading
    - Triggering model loading with progress feedback
    - Detecting same-type voice switches (instant, no reload)
    - Emitting signals for UI updates

    Story 4.5 Implementation:
    - At startup, only CustomVoice model is loaded
    - VoiceDesign and Base models are lazy-loaded on demand
    - Only one model is loaded at a time (~3.4GB memory constraint)
    - Same-type voice switch is instant (no model reload needed)

    Signals:
        loading_started: Emitted when model loading begins (model_type, message)
        loading_progress: Emitted during loading (percent, message)
        loading_complete: Emitted when loading finishes (success, error)
        model_ready: Emitted when a model is ready for use (model_type)
    """

    # Signals for UI integration
    loading_started = pyqtSignal(str, str)  # model_type_name, message
    loading_progress = pyqtSignal(float, str)  # percent, message
    loading_complete = pyqtSignal(bool, str)  # success, error_or_message
    model_ready = pyqtSignal(str)  # model_type_name

    def __init__(
        self,
        tts_service: Optional['QwenTTSService'] = None,
        parent: Optional[QObject] = None
    ):
        """
        Initialize the Model Loading Manager.

        Args:
            tts_service: QwenTTSService instance for model operations
            parent: Parent QObject
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._tts_service = tts_service
        self._current_loading_model: Optional[QwenModelType] = None
        self._is_loading = False
        self._last_voice_type: Optional[VoiceType] = None

        self.logger.debug("ModelLoadingManager initialized")

    def set_tts_service(self, tts_service: 'QwenTTSService'):
        """
        Set the TTS service.

        Args:
            tts_service: QwenTTSService instance
        """
        self._tts_service = tts_service
        self.logger.debug("TTS service set")

    def get_current_model_type(self) -> Optional[QwenModelType]:
        """Get the currently loaded model type."""
        if self._tts_service:
            return self._tts_service.get_current_model_type()
        return None

    def is_model_ready(self, model_type: QwenModelType) -> bool:
        """
        Check if a specific model is loaded and ready.

        Args:
            model_type: Model type to check

        Returns:
            bool: True if model is ready
        """
        if self._tts_service:
            return self._tts_service.is_model_loaded(model_type)
        return False

    def is_loading(self) -> bool:
        """Check if a model is currently being loaded."""
        return self._is_loading

    def requires_model_switch(self, voice_profile: VoiceProfile) -> bool:
        """
        Check if selecting this voice requires a model switch.

        Story 4.5: Same-type voice switch should be instant.
        Only returns True if the voice type requires a different model
        than what's currently loaded.

        Args:
            voice_profile: Voice profile to check

        Returns:
            bool: True if model switch is required
        """
        required_model = voice_profile.voice_type.required_model
        current_model = self.get_current_model_type()

        if current_model is None:
            # No model loaded, switch required
            return True

        # Switch required if models are different
        return required_model != current_model

    def requires_model_switch_for_type(self, voice_type: VoiceType) -> bool:
        """
        Check if a voice type requires a model switch.

        Args:
            voice_type: Voice type to check

        Returns:
            bool: True if model switch is required
        """
        required_model = voice_type.required_model
        current_model = self.get_current_model_type()

        if current_model is None:
            return True

        return required_model != current_model

    async def ensure_model_for_voice(
        self,
        voice_profile: VoiceProfile,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Ensure the correct model is loaded for a voice profile.

        If the voice requires a different model than what's currently loaded,
        this method will unload the current model and load the required one.
        If the same model type is already loaded, this returns immediately.

        Args:
            voice_profile: Voice profile requiring model
            progress_callback: Optional callback for progress updates

        Returns:
            tuple[bool, Optional[str]]: (success, error_message)
        """
        required_model = voice_profile.voice_type.required_model

        # Check if model switch is needed
        if not self.requires_model_switch(voice_profile):
            self.logger.debug(
                f"Model {required_model.display_name} already loaded, "
                "voice switch is instant"
            )
            return True, None

        return await self.load_model(required_model, progress_callback)

    async def load_model(
        self,
        model_type: QwenModelType,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Load a specific model type.

        If a different model is currently loaded, it will be unloaded first.
        Emits signals for UI feedback during loading.

        Args:
            model_type: Model type to load
            progress_callback: Optional callback for progress updates

        Returns:
            tuple[bool, Optional[str]]: (success, error_message)
        """
        if not self._tts_service:
            return False, "TTS service not available"

        if self._is_loading:
            return False, "Another model is currently loading"

        try:
            self._is_loading = True
            self._current_loading_model = model_type

            # Emit loading started
            message = f"Loading {model_type.display_name} model..."
            self.loading_started.emit(model_type.display_name, message)
            self.logger.info(message)

            if progress_callback:
                progress_callback(0, message)

            # Perform the model load
            success, error = await self._tts_service.ensure_model_loaded(model_type)

            if success:
                complete_message = f"{model_type.display_name} model ready"
                self.loading_complete.emit(True, complete_message)
                self.model_ready.emit(model_type.display_name)
                self.logger.info(complete_message)

                if progress_callback:
                    progress_callback(100, complete_message)
            else:
                error_message = error or "Unknown error"
                self.loading_complete.emit(False, error_message)
                self.logger.error(f"Model loading failed: {error_message}")

                if progress_callback:
                    progress_callback(0, f"Failed: {error_message}")

            return success, error

        except Exception as e:
            error_message = str(e)
            self.loading_complete.emit(False, error_message)
            self.logger.exception(f"Error loading model: {e}")
            return False, error_message

        finally:
            self._is_loading = False
            self._current_loading_model = None

    def get_loading_status(self) -> ModelLoadingStatus:
        """
        Get current model loading status.

        Returns:
            ModelLoadingStatus: Current status information
        """
        current_model = self.get_current_model_type()

        return ModelLoadingStatus(
            is_loading=self._is_loading,
            model_type=self._current_loading_model if self._is_loading else current_model,
            progress_percent=0 if self._is_loading else 100,
            message="Loading..." if self._is_loading else "Ready",
            is_ready=not self._is_loading and current_model is not None,
            error=None
        )

    def get_model_for_voice_type(self, voice_type: VoiceType) -> QwenModelType:
        """
        Get the model type required for a voice type.

        Args:
            voice_type: Voice type

        Returns:
            QwenModelType: Required model type
        """
        return voice_type.required_model

    def will_voice_switch_be_instant(self, voice_profile: VoiceProfile) -> bool:
        """
        Check if switching to this voice will be instant (no model reload).

        Story 4.5: Same-type voice switch should be instant.

        Args:
            voice_profile: Voice profile to switch to

        Returns:
            bool: True if switch will be instant
        """
        return not self.requires_model_switch(voice_profile)

    def get_loading_message_for_voice(self, voice_profile: VoiceProfile) -> Optional[str]:
        """
        Get loading message if switching to this voice requires model loading.

        Args:
            voice_profile: Voice profile to switch to

        Returns:
            Optional[str]: Loading message if needed, None if instant switch
        """
        if self.will_voice_switch_be_instant(voice_profile):
            return None

        required_model = voice_profile.voice_type.required_model
        return f"Loading {required_model.display_name} model..."
