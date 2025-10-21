"""
VoiceProfile Models

This module contains the VoiceProfile data model for managing voice cloning
samples with validation for audio format and duration constraints.
"""

import wave
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from enum import Enum
from datetime import datetime

from myvoice.models.validation import ValidationResult, ValidationIssue, ValidationStatus


logger = logging.getLogger(__name__)


class TranscriptionStatus(Enum):
    """Status of transcription for a voice profile."""
    NOT_STARTED = "not_started"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # For files that shouldn't be transcribed


@dataclass
class VoiceProfile:
    """
    Data model for voice cloning profiles.

    A VoiceProfile represents a voice sample file used for TTS generation with
    voice cloning. It includes validation for audio format, duration constraints,
    transcription status tracking, and metadata about the voice sample.

    Attributes:
        file_path: Path to the voice sample WAV file
        name: Human-readable name for the voice profile
        transcription: Text transcription of the voice sample (optional)
        duration: Duration of the audio in seconds
        is_valid: Whether the voice profile passes validation
        transcription_status: Current status of transcription processing
        transcription_error: Error message if transcription failed
        transcription_confidence: Confidence score of the transcription (0.0-1.0)
        last_transcription_attempt: Timestamp of last transcription attempt
        transcription_model: Name of the model used for transcription
        created_at: Timestamp when profile was created
        updated_at: Timestamp when profile was last updated
    """
    file_path: Path
    name: str
    transcription: Optional[str] = None
    duration: Optional[float] = None
    is_valid: bool = False
    transcription_status: TranscriptionStatus = TranscriptionStatus.NOT_STARTED
    transcription_error: Optional[str] = None
    transcription_confidence: Optional[float] = None
    last_transcription_attempt: Optional[datetime] = None
    transcription_model: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize and validate the voice profile."""
        # Convert string path to Path object if needed
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

        # Auto-generate name from filename if not provided
        if not self.name:
            self.name = self.file_path.stem

        # Validate the profile
        validation_result = self.validate()
        self.is_valid = validation_result.is_valid

        # Extract duration if file is valid
        if self.is_valid and self.duration is None:
            try:
                self.duration = self._extract_duration()
            except Exception as e:
                logger.warning(f"Failed to extract duration for {self.file_path}: {e}")
                self.duration = 0.0

        # Auto-detect transcription if not provided
        if self.transcription is None:
            transcription_file = self.file_path.with_suffix('.txt')
            if transcription_file.exists():
                try:
                    self.transcription = transcription_file.read_text(encoding='utf-8').strip()
                    self.transcription_status = TranscriptionStatus.COMPLETED
                    logger.debug(f"Auto-detected transcription for {self.file_path}")
                except Exception as e:
                    logger.warning(f"Failed to read transcription file {transcription_file}: {e}")

        # Set transcription status based on transcription presence
        if self.transcription and self.transcription_status == TranscriptionStatus.NOT_STARTED:
            self.transcription_status = TranscriptionStatus.COMPLETED

    def validate(self) -> ValidationResult:
        """
        Validate the voice profile for format and constraints.

        Returns:
            ValidationResult: Detailed validation result with issues and warnings
        """
        issues = []
        warnings = []

        try:
            # File existence validation
            if not self.file_path.exists():
                issues.append(ValidationIssue(
                    field="file_path",
                    message=f"Voice file not found: {self.file_path}",
                    code="FILE_NOT_FOUND",
                    severity=ValidationStatus.INVALID
                ))
                # Can't do further validation if file doesn't exist
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=issues,
                    warnings=warnings
                )

            # File format validation
            if self.file_path.suffix.lower() != '.wav':
                issues.append(ValidationIssue(
                    field="file_path",
                    message="Voice file must be in WAV format",
                    code="INVALID_FORMAT",
                    severity=ValidationStatus.INVALID
                ))

            # File size validation
            file_size = self.file_path.stat().st_size
            if file_size == 0:
                issues.append(ValidationIssue(
                    field="file_path",
                    message="Voice file is empty",
                    code="EMPTY_FILE",
                    severity=ValidationStatus.INVALID
                ))
            elif file_size > 25 * 1024 * 1024:  # 25MB limit
                issues.append(ValidationIssue(
                    field="file_path",
                    message="Voice file is too large (maximum 25MB)",
                    code="FILE_TOO_LARGE",
                    severity=ValidationStatus.INVALID
                ))
            elif file_size < 1024:  # Very small files are suspicious
                warnings.append(ValidationIssue(
                    field="file_path",
                    message="Voice file is very small, quality may be poor",
                    code="SMALL_FILE",
                    severity=ValidationStatus.WARNING
                ))

            # WAV format validation (header check)
            if self.file_path.suffix.lower() == '.wav':
                try:
                    with wave.open(str(self.file_path), 'rb') as wav_file:
                        # Basic WAV validation
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        framerate = wav_file.getframerate()
                        frames = wav_file.getnframes()

                        # Duration validation
                        duration = frames / framerate if framerate > 0 else 0

                        # Duration constraints based on TTS best practices
                        if duration < 3.0:
                            warnings.append(ValidationIssue(
                                field="duration",
                                message=f"Audio duration is short ({duration:.1f}s), may affect voice quality",
                                code="SHORT_DURATION",
                                severity=ValidationStatus.WARNING
                            ))
                        elif duration > 30.0:
                            warnings.append(ValidationIssue(
                                field="duration",
                                message=f"Audio duration is long ({duration:.1f}s), consider trimming for better performance",
                                code="LONG_DURATION",
                                severity=ValidationStatus.WARNING
                            ))

                        # Sample rate validation
                        if framerate < 16000:
                            warnings.append(ValidationIssue(
                                field="sample_rate",
                                message=f"Low sample rate ({framerate}Hz), may affect voice quality",
                                code="LOW_SAMPLE_RATE",
                                severity=ValidationStatus.WARNING
                            ))
                        elif framerate > 48000:
                            warnings.append(ValidationIssue(
                                field="sample_rate",
                                message=f"Very high sample rate ({framerate}Hz), may be unnecessary",
                                code="HIGH_SAMPLE_RATE",
                                severity=ValidationStatus.WARNING
                            ))

                        # Channel validation
                        if channels > 1:
                            warnings.append(ValidationIssue(
                                field="channels",
                                message="Multi-channel audio detected, mono is recommended for voice cloning",
                                code="MULTI_CHANNEL",
                                severity=ValidationStatus.WARNING
                            ))

                        # Bit depth validation
                        if sample_width < 2:  # Less than 16-bit
                            warnings.append(ValidationIssue(
                                field="bit_depth",
                                message="Low bit depth detected, 16-bit or higher recommended",
                                code="LOW_BIT_DEPTH",
                                severity=ValidationStatus.WARNING
                            ))

                except wave.Error as e:
                    issues.append(ValidationIssue(
                        field="file_path",
                        message=f"Invalid WAV file format: {str(e)}",
                        code="INVALID_WAV_FORMAT",
                        severity=ValidationStatus.INVALID
                    ))
                except Exception as e:
                    issues.append(ValidationIssue(
                        field="file_path",
                        message=f"Error reading WAV file: {str(e)}",
                        code="WAV_READ_ERROR",
                        severity=ValidationStatus.INVALID
                    ))

            # Name validation
            if not self.name or not self.name.strip():
                warnings.append(ValidationIssue(
                    field="name",
                    message="Voice profile name is empty",
                    code="EMPTY_NAME",
                    severity=ValidationStatus.WARNING
                ))

            # Transcription validation
            if self.transcription and len(self.transcription) > 500:
                warnings.append(ValidationIssue(
                    field="transcription",
                    message="Transcription is very long, consider shortening",
                    code="LONG_TRANSCRIPTION",
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
                warnings=warnings
            )

        except Exception as e:
            logger.exception(f"Error during voice profile validation: {e}")
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
                summary="Validation failed due to internal error"
            )

    def _extract_duration(self) -> float:
        """
        Extract duration from the WAV file.

        Returns:
            float: Duration in seconds

        Raises:
            Exception: If duration cannot be extracted
        """
        try:
            with wave.open(str(self.file_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                framerate = wav_file.getframerate()
                return frames / framerate if framerate > 0 else 0.0
        except Exception as e:
            logger.error(f"Failed to extract duration from {self.file_path}: {e}")
            raise

    def get_audio_info(self) -> dict:
        """
        Get detailed audio information from the WAV file.

        Returns:
            dict: Audio information including channels, sample rate, bit depth, duration
        """
        if not self.file_path.exists() or self.file_path.suffix.lower() != '.wav':
            return {}

        try:
            with wave.open(str(self.file_path), 'rb') as wav_file:
                return {
                    'channels': wav_file.getnchannels(),
                    'sample_width': wav_file.getsampwidth(),
                    'framerate': wav_file.getframerate(),
                    'frames': wav_file.getnframes(),
                    'duration': wav_file.getnframes() / wav_file.getframerate(),
                    'bit_depth': wav_file.getsampwidth() * 8,
                    'file_size': self.file_path.stat().st_size
                }
        except Exception as e:
            logger.error(f"Failed to get audio info from {self.file_path}: {e}")
            return {}

    def __str__(self) -> str:
        """String representation of the voice profile."""
        duration_str = f"{self.duration:.1f}s" if self.duration else "unknown"
        status = "VALID" if self.is_valid else "INVALID"
        return f"[{status}] {self.name} ({duration_str}) - {self.file_path.name}"

    def __repr__(self) -> str:
        """Developer representation of the voice profile."""
        return (f"VoiceProfile(name='{self.name}', file_path='{self.file_path}', "
                f"duration={self.duration}, is_valid={self.is_valid})")

    @classmethod
    def create_profile_from_file(cls, file_path: Path, name: Optional[str] = None,
                                 transcription: Optional[str] = None) -> 'VoiceProfile':
        """
        Factory method to create a VoiceProfile from a file path.

        This method automatically validates the file and extracts metadata
        during profile creation.

        Args:
            file_path: Path to the voice sample WAV file
            name: Optional custom name for the profile (defaults to filename)
            transcription: Optional transcription of the voice sample

        Returns:
            VoiceProfile: Validated voice profile instance

        Examples:
            # Create from file with auto-generated name
            profile = VoiceProfile.create_profile_from_file(Path("voice.wav"))

            # Create with custom name and transcription
            profile = VoiceProfile.create_profile_from_file(
                Path("samples/john.wav"),
                name="John's Voice",
                transcription="Hello, this is John speaking."
            )
        """
        # Convert string to Path if needed
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Generate name from filename if not provided
        if name is None:
            name = file_path.stem

        logger.debug(f"Creating voice profile from file: {file_path}")

        # Create and return the profile (validation happens in __post_init__)
        profile = cls(
            file_path=file_path,
            name=name,
            transcription=transcription
        )

        logger.info(f"Created voice profile: {profile}")
        return profile

    def update_transcription_status(self, status: TranscriptionStatus,
                                   error_message: Optional[str] = None,
                                   confidence: Optional[float] = None,
                                   model_name: Optional[str] = None) -> None:
        """
        Update the transcription status and related metadata.

        Args:
            status: New transcription status
            error_message: Error message if status is FAILED
            confidence: Transcription confidence score (0.0-1.0)
            model_name: Name of the transcription model used
        """
        self.transcription_status = status
        self.last_transcription_attempt = datetime.now()
        self.updated_at = datetime.now()

        if status == TranscriptionStatus.FAILED:
            self.transcription_error = error_message
            self.transcription = None
            self.transcription_confidence = None
        elif status == TranscriptionStatus.COMPLETED:
            self.transcription_error = None
            if confidence is not None:
                self.transcription_confidence = confidence
            if model_name is not None:
                self.transcription_model = model_name
        elif status == TranscriptionStatus.PROCESSING:
            self.transcription_error = None
        elif status == TranscriptionStatus.QUEUED:
            self.transcription_error = None

        logger.debug(f"Updated transcription status for {self.name}: {status.value}")

    def set_transcription_result(self, transcription_text: str,
                                confidence: float,
                                model_name: str) -> None:
        """
        Set the transcription result and mark as completed.

        Args:
            transcription_text: The transcribed text
            confidence: Confidence score (0.0-1.0)
            model_name: Name of the transcription model used
        """
        self.transcription = transcription_text
        self.transcription_confidence = confidence
        self.transcription_model = model_name
        self.update_transcription_status(TranscriptionStatus.COMPLETED)

    def mark_transcription_failed(self, error_message: str) -> None:
        """
        Mark transcription as failed with error message.

        Args:
            error_message: Description of the transcription failure
        """
        self.update_transcription_status(TranscriptionStatus.FAILED, error_message)

    def needs_transcription(self) -> bool:
        """
        Check if this voice profile needs transcription.

        Returns:
            bool: True if transcription is needed and should be attempted
        """
        return (
            self.transcription_status in [
                TranscriptionStatus.NOT_STARTED,
                TranscriptionStatus.FAILED
            ] and
            self.is_valid and
            self.transcription_status != TranscriptionStatus.SKIPPED
        )

    def can_queue_transcription(self) -> bool:
        """
        Check if this voice profile can be queued for transcription.

        Returns:
            bool: True if can be queued (not already processing or completed)
        """
        return self.transcription_status not in [
            TranscriptionStatus.PROCESSING,
            TranscriptionStatus.QUEUED,
            TranscriptionStatus.COMPLETED,
            TranscriptionStatus.SKIPPED
        ]

    @property
    def transcription_progress_message(self) -> str:
        """Get a human-readable message about transcription progress."""
        status_messages = {
            TranscriptionStatus.NOT_STARTED: "Transcription not started",
            TranscriptionStatus.QUEUED: "Queued for transcription",
            TranscriptionStatus.PROCESSING: "Transcribing audio...",
            TranscriptionStatus.COMPLETED: f"Transcription completed ({self.transcription_confidence:.1%} confidence)" if self.transcription_confidence else "Transcription completed",
            TranscriptionStatus.FAILED: f"Transcription failed: {self.transcription_error}" if self.transcription_error else "Transcription failed",
            TranscriptionStatus.SKIPPED: "Transcription skipped"
        }
        return status_messages.get(self.transcription_status, "Unknown status")

    def to_dict(self) -> dict:
        """
        Convert voice profile to dictionary format for serialization.

        Returns:
            dict: Dictionary representation of the voice profile
        """
        return {
            'file_path': str(self.file_path),
            'name': self.name,
            'transcription': self.transcription,
            'duration': self.duration,
            'is_valid': self.is_valid,
            'transcription_status': self.transcription_status.value,
            'transcription_error': self.transcription_error,
            'transcription_confidence': self.transcription_confidence,
            'last_transcription_attempt': self.last_transcription_attempt.isoformat() if self.last_transcription_attempt else None,
            'transcription_model': self.transcription_model,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VoiceProfile':
        """
        Create VoiceProfile from dictionary data.

        Args:
            data: Dictionary containing voice profile data

        Returns:
            VoiceProfile: Reconstructed voice profile instance
        """
        # Parse datetime fields
        created_at = datetime.now()
        updated_at = datetime.now()
        last_transcription_attempt = None

        if data.get('created_at'):
            try:
                created_at = datetime.fromisoformat(data['created_at'])
            except ValueError:
                pass

        if data.get('updated_at'):
            try:
                updated_at = datetime.fromisoformat(data['updated_at'])
            except ValueError:
                pass

        if data.get('last_transcription_attempt'):
            try:
                last_transcription_attempt = datetime.fromisoformat(data['last_transcription_attempt'])
            except ValueError:
                pass

        # Parse transcription status
        transcription_status = TranscriptionStatus.NOT_STARTED
        if data.get('transcription_status'):
            try:
                transcription_status = TranscriptionStatus(data['transcription_status'])
            except ValueError:
                pass

        return cls(
            file_path=Path(data['file_path']),
            name=data['name'],
            transcription=data.get('transcription'),
            duration=data.get('duration'),
            is_valid=data.get('is_valid', False),
            transcription_status=transcription_status,
            transcription_error=data.get('transcription_error'),
            transcription_confidence=data.get('transcription_confidence'),
            last_transcription_attempt=last_transcription_attempt,
            transcription_model=data.get('transcription_model'),
            created_at=created_at,
            updated_at=updated_at
        )