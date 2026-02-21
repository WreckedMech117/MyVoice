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


class ModelQualityTier(Enum):
    """
    Model quality tier for Qwen3-TTS models.

    SMALL: 0.6B parameter models (~1.2 GB) - Lower VRAM, faster inference
    QUALITY: 1.7B parameter models (~3.4 GB) - Higher quality output
    """
    SMALL = "small"
    QUALITY = "quality"

    @property
    def display_name(self) -> str:
        """Get human-readable tier name."""
        return {
            ModelQualityTier.SMALL: "Small (0.6B)",
            ModelQualityTier.QUALITY: "Quality (1.7B)"
        }.get(self, self.value)

    @property
    def description(self) -> str:
        """Get tier description."""
        return {
            ModelQualityTier.SMALL: "Lower VRAM usage (~1.2 GB), faster inference",
            ModelQualityTier.QUALITY: "Higher quality output (~3.4 GB VRAM)"
        }.get(self, "")

    @property
    def model_size(self) -> str:
        """Get the model size identifier used in HuggingFace model names."""
        return {
            ModelQualityTier.SMALL: "0.6B",
            ModelQualityTier.QUALITY: "1.7B"
        }.get(self, "1.7B")


class QwenModelType(Enum):
    """
    Qwen3-TTS model types.

    CustomVoice: Bundled voices with emotion control (📦)
    VoiceDesign: Create voices from text descriptions (🎭) - Only available in Quality tier
    Base: Voice cloning from audio samples (🎤)

    Note: Model IDs are templates with {size} placeholder replaced at runtime
    based on ModelQualityTier setting.
    """
    # Quality (1.7B) model IDs - default
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

    @property
    def available_in_small_tier(self) -> bool:
        """Check if this model type is available in the Small (0.6B) tier."""
        # VoiceDesign is only available in Quality (1.7B) tier
        return self != QwenModelType.VOICE_DESIGN

    def supports_instruct_in_tier(self, tier: 'ModelQualityTier') -> bool:
        """
        Check if this model type supports instruction control in the given tier.

        The 0.6B models do NOT support instruct parameter for emotion/style control.
        Only the 1.7B CustomVoice and VoiceDesign models support it.

        Args:
            tier: The quality tier to check

        Returns:
            bool: True if instruct parameter is supported
        """
        if tier == ModelQualityTier.SMALL:
            # 0.6B models don't support instruction control
            return False
        # 1.7B CustomVoice and VoiceDesign support instruct
        return self in (QwenModelType.CUSTOM_VOICE, QwenModelType.VOICE_DESIGN)

    def get_model_id(self, tier: 'ModelQualityTier') -> str:
        """
        Get the HuggingFace model ID for a specific quality tier.

        Args:
            tier: The quality tier (SMALL or QUALITY)

        Returns:
            str: The full HuggingFace model ID

        Raises:
            ValueError: If model is not available in the requested tier
        """
        if tier == ModelQualityTier.SMALL and not self.available_in_small_tier:
            raise ValueError(f"{self.display_name} is not available in Small tier")

        # Replace size in model ID
        size = tier.model_size
        return self.value.replace("1.7B", size)