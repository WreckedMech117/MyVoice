"""
Emotion Profile Models

This module contains data models for emotional voice synthesis configuration.
It defines emotion intensity levels and their corresponding GPT-SoVITS parameters.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from myvoice.models.validation import ValidationResult, ValidationIssue, ValidationStatus


logger = logging.getLogger(__name__)


class EmotionIntensity(Enum):
    """
    Enumeration of emotion intensity levels for voice synthesis.

    These levels map to specific GPT-SoVITS parameters to control
    emotional expression in generated speech.
    """
    SUBDUED = 0
    CALM = 1
    GENTLE = 2
    NEUTRAL = 3
    EXPRESSIVE = 4
    DYNAMIC = 5
    INTENSE = 6


@dataclass
class EmotionParameters:
    """
    GPT-SoVITS parameters for a specific emotion intensity level.

    Attributes:
        temperature: Controls randomness in generation (0.1-2.0)
        top_p: Nucleus sampling parameter (0.1-1.0)
        repetition_penalty: Controls repetition in speech (1.0-2.0)
        name: Human-readable name for this emotion level
        description: Brief description of the emotional characteristics
    """
    temperature: float
    top_p: float
    repetition_penalty: float
    name: str
    description: str

    def __post_init__(self):
        """Validate emotion parameters after creation."""
        validation_result = self.validate()
        if not validation_result.is_valid:
            logger.warning(f"Emotion parameters validation issues: {len(validation_result.issues)} issues")
            for issue in validation_result.issues:
                logger.warning(f"  - {issue.field}: {issue.message}")

    def validate(self) -> ValidationResult:
        """
        Validate emotion parameters for GPT-SoVITS compatibility.

        Returns:
            ValidationResult: Detailed validation result
        """
        issues = []
        warnings = []

        try:
            # Validate temperature
            if not isinstance(self.temperature, (int, float)):
                issues.append(ValidationIssue(
                    field="temperature",
                    message="Temperature must be a number",
                    code="INVALID_TYPE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.temperature < 0.1 or self.temperature > 2.0:
                issues.append(ValidationIssue(
                    field="temperature",
                    message="Temperature must be between 0.1 and 2.0",
                    code="INVALID_RANGE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.temperature > 1.8:
                warnings.append(ValidationIssue(
                    field="temperature",
                    message="Very high temperature (>1.8) may produce unpredictable results",
                    code="HIGH_TEMPERATURE",
                    severity=ValidationStatus.WARNING
                ))

            # Validate top_p
            if not isinstance(self.top_p, (int, float)):
                issues.append(ValidationIssue(
                    field="top_p",
                    message="Top_p must be a number",
                    code="INVALID_TYPE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.top_p < 0.1 or self.top_p > 1.0:
                issues.append(ValidationIssue(
                    field="top_p",
                    message="Top_p must be between 0.1 and 1.0",
                    code="INVALID_RANGE",
                    severity=ValidationStatus.INVALID
                ))

            # Validate repetition_penalty
            if not isinstance(self.repetition_penalty, (int, float)):
                issues.append(ValidationIssue(
                    field="repetition_penalty",
                    message="Repetition penalty must be a number",
                    code="INVALID_TYPE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.repetition_penalty < 1.0 or self.repetition_penalty > 2.0:
                issues.append(ValidationIssue(
                    field="repetition_penalty",
                    message="Repetition penalty must be between 1.0 and 2.0",
                    code="INVALID_RANGE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.repetition_penalty > 1.8:
                warnings.append(ValidationIssue(
                    field="repetition_penalty",
                    message="High repetition penalty (>1.8) may cause overly varied speech",
                    code="HIGH_REPETITION_PENALTY",
                    severity=ValidationStatus.WARNING
                ))

            # Validate name
            if not isinstance(self.name, str) or not self.name.strip():
                issues.append(ValidationIssue(
                    field="name",
                    message="Name must be a non-empty string",
                    code="INVALID_NAME",
                    severity=ValidationStatus.INVALID
                ))

            # Validate description
            if not isinstance(self.description, str):
                issues.append(ValidationIssue(
                    field="description",
                    message="Description must be a string",
                    code="INVALID_DESCRIPTION",
                    severity=ValidationStatus.INVALID
                ))

            # Determine overall status
            if issues:
                status = ValidationStatus.INVALID
                is_valid = False
            elif warnings:
                status = ValidationStatus.WARNING
                is_valid = True
            else:
                status = ValidationStatus.VALID
                is_valid = True

            return ValidationResult(
                is_valid=is_valid,
                status=status,
                issues=issues,
                warnings=warnings,
                summary=f"Emotion parameters validation: {len(issues)} issues, {len(warnings)} warnings"
            )

        except Exception as e:
            logger.exception(f"Error during emotion parameters validation: {e}")
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                issues=[ValidationIssue(
                    field="general",
                    message=f"Validation error: {str(e)}",
                    code="VALIDATION_ERROR",
                    severity=ValidationStatus.INVALID
                )],
                warnings=[],
                summary="Emotion parameters validation failed due to internal error"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert emotion parameters to dictionary for serialization.

        Returns:
            Dict[str, Any]: Parameters as dictionary
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "name": self.name,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionParameters':
        """
        Create emotion parameters from dictionary.

        Args:
            data: Dictionary containing parameter data

        Returns:
            EmotionParameters: Parameters instance
        """
        return cls(
            temperature=data["temperature"],
            top_p=data["top_p"],
            repetition_penalty=data["repetition_penalty"],
            name=data["name"],
            description=data["description"]
        )


class EmotionProfile:
    """
    Complete emotion profile defining all intensity levels for voice synthesis.

    This class provides the mapping between UI slider positions and GPT-SoVITS
    parameters for emotional voice synthesis control.
    """

    # Default emotion parameters for each intensity level
    DEFAULT_EMOTIONS = {
        EmotionIntensity.SUBDUED: EmotionParameters(
            temperature=0.3,
            top_p=0.7,
            repetition_penalty=1.2,
            name="Subdued",
            description="Very calm and controlled expression"
        ),
        EmotionIntensity.CALM: EmotionParameters(
            temperature=0.5,
            top_p=0.8,
            repetition_penalty=1.25,
            name="Calm",
            description="Relaxed and peaceful tone"
        ),
        EmotionIntensity.GENTLE: EmotionParameters(
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            name="Gentle",
            description="Soft and mild expression"
        ),
        EmotionIntensity.NEUTRAL: EmotionParameters(
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.35,
            name="Neutral",
            description="Balanced and natural tone"
        ),
        EmotionIntensity.EXPRESSIVE: EmotionParameters(
            temperature=1.2,
            top_p=0.9,
            repetition_penalty=1.4,
            name="Expressive",
            description="More animated and engaging"
        ),
        EmotionIntensity.DYNAMIC: EmotionParameters(
            temperature=1.4,
            top_p=0.8,
            repetition_penalty=1.45,
            name="Dynamic",
            description="Energetic and varied expression"
        ),
        EmotionIntensity.INTENSE: EmotionParameters(
            temperature=1.6,
            top_p=0.7,
            repetition_penalty=1.5,
            name="Intense",
            description="Highly expressive and dramatic"
        )
    }

    def __init__(self, custom_emotions: Optional[Dict[EmotionIntensity, EmotionParameters]] = None):
        """
        Initialize emotion profile with default or custom parameters.

        Args:
            custom_emotions: Custom emotion parameters (optional)
        """
        self.emotions = custom_emotions or self.DEFAULT_EMOTIONS.copy()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate all emotion parameters
        self._validate_all_emotions()

    def _validate_all_emotions(self):
        """Validate all emotion parameters in the profile."""
        for intensity, params in self.emotions.items():
            validation_result = params.validate()
            if not validation_result.is_valid:
                self.logger.error(f"Invalid emotion parameters for {intensity.name}: {validation_result.summary}")

    def get_emotion_parameters(self, intensity: EmotionIntensity) -> EmotionParameters:
        """
        Get emotion parameters for a specific intensity level.

        Args:
            intensity: Emotion intensity level

        Returns:
            EmotionParameters: Parameters for the intensity level

        Raises:
            ValueError: If intensity level is not defined
        """
        if intensity not in self.emotions:
            raise ValueError(f"Emotion intensity {intensity} not defined in profile")

        return self.emotions[intensity]

    def get_emotion_by_slider_position(self, position: int) -> EmotionParameters:
        """
        Get emotion parameters based on slider position (0-6).

        Args:
            position: Slider position (0=Subdued, 3=Neutral, 6=Intense)

        Returns:
            EmotionParameters: Parameters for the slider position

        Raises:
            ValueError: If position is out of range
        """
        if not 0 <= position <= 6:
            raise ValueError(f"Slider position must be between 0 and 6, got {position}")

        intensity = EmotionIntensity(position)
        return self.get_emotion_parameters(intensity)

    def get_slider_position_for_intensity(self, intensity: EmotionIntensity) -> int:
        """
        Get slider position for a specific emotion intensity.

        Args:
            intensity: Emotion intensity level

        Returns:
            int: Slider position (0-6)
        """
        return intensity.value

    def get_all_emotion_names(self) -> List[str]:
        """
        Get list of all emotion names in slider order.

        Returns:
            List[str]: Emotion names from position 0 to 6
        """
        return [self.emotions[EmotionIntensity(i)].name for i in range(7)]

    def get_emotion_name_by_position(self, position: int) -> str:
        """
        Get emotion name for a specific slider position.

        Args:
            position: Slider position (0-6)

        Returns:
            str: Emotion name
        """
        return self.get_emotion_by_slider_position(position).name

    def get_neutral_position(self) -> int:
        """
        Get the slider position for neutral emotion.

        Returns:
            int: Neutral emotion position (always 3)
        """
        return EmotionIntensity.NEUTRAL.value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert emotion profile to dictionary for serialization.

        Returns:
            Dict[str, Any]: Profile as dictionary
        """
        return {
            "emotions": {
                intensity.name: params.to_dict()
                for intensity, params in self.emotions.items()
            },
            "version": "1.0"
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionProfile':
        """
        Create emotion profile from dictionary.

        Args:
            data: Dictionary containing profile data

        Returns:
            EmotionProfile: Profile instance
        """
        emotions = {}
        for intensity_name, params_data in data["emotions"].items():
            intensity = EmotionIntensity[intensity_name]
            emotions[intensity] = EmotionParameters.from_dict(params_data)

        return cls(custom_emotions=emotions)

    def validate(self) -> ValidationResult:
        """
        Validate the complete emotion profile.

        Returns:
            ValidationResult: Detailed validation result
        """
        issues = []
        warnings = []

        try:
            # Check that all required intensities are present
            required_intensities = set(EmotionIntensity)
            present_intensities = set(self.emotions.keys())

            missing_intensities = required_intensities - present_intensities
            if missing_intensities:
                for intensity in missing_intensities:
                    issues.append(ValidationIssue(
                        field="emotions",
                        message=f"Missing emotion intensity: {intensity.name}",
                        code="MISSING_INTENSITY",
                        severity=ValidationStatus.INVALID
                    ))

            # Validate each emotion parameter set
            for intensity, params in self.emotions.items():
                param_validation = params.validate()
                if not param_validation.is_valid:
                    issues.extend(param_validation.issues)
                warnings.extend(param_validation.warnings)

            # Check for reasonable parameter progression
            if len(self.emotions) == 7:  # Only if we have all emotions
                temperatures = [self.emotions[EmotionIntensity(i)].temperature for i in range(7)]
                if not all(temperatures[i] <= temperatures[i+1] for i in range(6)):
                    warnings.append(ValidationIssue(
                        field="temperature_progression",
                        message="Temperature values should generally increase with intensity",
                        code="IRREGULAR_PROGRESSION",
                        severity=ValidationStatus.WARNING
                    ))

            # Determine overall status
            if issues:
                status = ValidationStatus.INVALID
                is_valid = False
            elif warnings:
                status = ValidationStatus.WARNING
                is_valid = True
            else:
                status = ValidationStatus.VALID
                is_valid = True

            return ValidationResult(
                is_valid=is_valid,
                status=status,
                issues=issues,
                warnings=warnings,
                summary=f"Emotion profile validation: {len(issues)} issues, {len(warnings)} warnings"
            )

        except Exception as e:
            logger.exception(f"Error during emotion profile validation: {e}")
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                issues=[ValidationIssue(
                    field="general",
                    message=f"Validation error: {str(e)}",
                    code="VALIDATION_ERROR",
                    severity=ValidationStatus.INVALID
                )],
                warnings=[],
                summary="Emotion profile validation failed due to internal error"
            )

    def __str__(self) -> str:
        """String representation of emotion profile."""
        return f"EmotionProfile(emotions={len(self.emotions)})"

    def __repr__(self) -> str:
        """Developer representation of emotion profile."""
        emotion_names = [params.name for params in self.emotions.values()]
        return f"EmotionProfile(emotions={emotion_names})"


# Create default global emotion profile instance
DEFAULT_EMOTION_PROFILE = EmotionProfile()