"""
TTS Request Models

This module contains data models for Text-to-Speech requests and responses.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from myvoice.models.emotion_profile import EmotionParameters, DEFAULT_EMOTION_PROFILE


@dataclass
class TTSRequest:
    """
    Data model for TTS generation requests.

    Attributes:
        text: Text to convert to speech
        voice_file_path: Path to the voice sample file
        voice_text: Transcript of the voice sample (optional)
        response_format: Audio format for response (default: wav)
        text_lang: Language of the text (default: en)
        prompt_lang: Language of the voice sample transcript (default: en)
        speed_factor: Speech speed multiplier (default: 1.0)
        emotion_parameters: Emotion parameters for voice synthesis (optional)
    """
    text: str
    voice_file_path: Path
    voice_text: Optional[str] = None
    response_format: str = "wav"
    text_lang: str = "en"
    prompt_lang: str = "en"
    speed_factor: float = 1.0
    emotion_parameters: Optional[EmotionParameters] = None

    def __post_init__(self):
        """Validate request parameters."""
        if not self.text.strip():
            raise ValueError("Text cannot be empty")

        if not self.voice_file_path.exists():
            raise FileNotFoundError(f"Voice file not found: {self.voice_file_path}")

        if not self.voice_file_path.suffix.lower() == '.wav':
            raise ValueError("Voice file must be in WAV format")

        # Validate emotion parameters if provided
        if self.emotion_parameters is not None:
            validation_result = self.emotion_parameters.validate()
            if not validation_result.is_valid:
                error_messages = [issue.message for issue in validation_result.issues]
                raise ValueError(f"Invalid emotion parameters: {'; '.join(error_messages)}")

        # Validate speed_factor
        if not isinstance(self.speed_factor, (int, float)) or self.speed_factor <= 0:
            raise ValueError("Speed factor must be a positive number")

    def get_emotion_parameters(self) -> EmotionParameters:
        """
        Get emotion parameters for this request.

        Returns:
            EmotionParameters: Emotion parameters (defaults to neutral if not set)
        """
        if self.emotion_parameters is not None:
            return self.emotion_parameters

        # Return neutral emotion as default
        from myvoice.models.emotion_profile import EmotionIntensity
        return DEFAULT_EMOTION_PROFILE.get_emotion_parameters(EmotionIntensity.NEUTRAL)

    def set_emotion_by_slider_position(self, position: int):
        """
        Set emotion parameters based on slider position.

        Args:
            position: Slider position (0-6)
        """
        self.emotion_parameters = DEFAULT_EMOTION_PROFILE.get_emotion_by_slider_position(position)

    def get_gpt_sovits_parameters(self) -> dict:
        """
        Get GPT-SoVITS API parameters including emotion settings.

        Returns:
            dict: Parameters for GPT-SoVITS API
        """
        emotion_params = self.get_emotion_parameters()

        return {
            "temperature": emotion_params.temperature,
            "top_p": emotion_params.top_p,
            "repetition_penalty": emotion_params.repetition_penalty,
            "speed_factor": self.speed_factor
        }

    def has_custom_emotion(self) -> bool:
        """
        Check if custom emotion parameters are set.

        Returns:
            bool: True if custom emotion parameters are set
        """
        return self.emotion_parameters is not None


@dataclass
class TTSResponse:
    """
    Data model for TTS generation responses.

    Attributes:
        audio_data: Binary audio data
        content_type: MIME type of the audio data
        success: Whether the request was successful
        error_message: Error message if request failed
    """
    audio_data: Optional[bytes] = None
    content_type: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None