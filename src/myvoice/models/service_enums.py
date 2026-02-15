"""
Service Status Enumerations

This module contains service status enums to avoid circular imports.
"""

from enum import Enum


class ServiceStatus(Enum):
    """Service status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ModelState(Enum):
    """
    Model lifecycle state enumeration for Qwen3-TTS models.

    State transitions:
        UNLOADED → LOADING → READY → UNLOADING → UNLOADED
        LOADING → ERROR (on failure)
        READY → ERROR (on runtime failure)
    """
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    ERROR = "error"


class QwenModelType(Enum):
    """
    Qwen3-TTS model types.

    CustomVoice: Bundled voices with emotion control (📦)
    VoiceDesign: Create voices from text descriptions (🎭)
    Base: Voice cloning from audio samples (🎤)
    """
    CUSTOM_VOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    VOICE_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    @property
    def supports_emotion(self) -> bool:
        """Check if this model type supports emotion control."""
        return self in (QwenModelType.CUSTOM_VOICE, QwenModelType.VOICE_DESIGN)

    @property
    def display_name(self) -> str:
        """Get human-readable model name."""
        names = {
            QwenModelType.CUSTOM_VOICE: "CustomVoice",
            QwenModelType.VOICE_DESIGN: "VoiceDesign",
            QwenModelType.BASE: "Base (Clone)"
        }
        return names.get(self, self.value)