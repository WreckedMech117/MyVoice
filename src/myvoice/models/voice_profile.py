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


class VoiceType(Enum):
    """
    Voice type enumeration for categorizing voices in the voice library.

    Story 4.1: Voice Library & Selection (FR11)
    Story 1.6: Emotion Control for Saved Voice

    Types:
        BUNDLED: Pre-installed voices using CustomVoice model (📦)
        DESIGNED: User-created voices from text descriptions using VoiceDesign model (🎭)
        CLONED: User-created voices from audio samples using Base model (🎤)
        OPTIMIZED: Fine-tuned voices from training data, supports emotion control (🔧)
        EMBEDDING: Voice embeddings from Voice Design Studio, full emotion support (🧬)
    """
    BUNDLED = "bundled"
    DESIGNED = "designed"
    CLONED = "cloned"
    OPTIMIZED = "optimized"
    EMBEDDING = "embedding"

    @property
    def icon(self) -> str:
        """Get the display icon for this voice type."""
        icons = {
            VoiceType.BUNDLED: "📦",
            VoiceType.DESIGNED: "🎭",
            VoiceType.CLONED: "🎤",
            VoiceType.OPTIMIZED: "🔧",
            VoiceType.EMBEDDING: "🧬"
        }
        return icons.get(self, "🎤")

    @property
    def display_name(self) -> str:
        """Get human-readable name for this voice type."""
        names = {
            VoiceType.BUNDLED: "Bundled",
            VoiceType.DESIGNED: "Designed",
            VoiceType.CLONED: "Cloned",
            VoiceType.OPTIMIZED: "Optimized",
            VoiceType.EMBEDDING: "Embedding"
        }
        return names.get(self, "Unknown")

    @property
    def supports_emotion(self) -> bool:
        """Check if this voice type supports emotion control (FR8a)."""
        # BUNDLED/DESIGNED/OPTIMIZED: Support emotion via instruct parameter
        # EMBEDDING: Supports emotion via multiple embeddings (one per emotion)
        # CLONED: No emotion support (single Base model embedding)
        return self in (VoiceType.BUNDLED, VoiceType.DESIGNED, VoiceType.OPTIMIZED, VoiceType.EMBEDDING)

    @property
    def sort_order(self) -> int:
        """Get sort order for grouping (bundled first)."""
        order = {
            VoiceType.BUNDLED: 0,
            VoiceType.EMBEDDING: 1,  # Embedding voices after bundled
            VoiceType.DESIGNED: 2,
            VoiceType.OPTIMIZED: 3,
            VoiceType.CLONED: 4
        }
        return order.get(self, 99)

    @property
    def required_model(self) -> 'QwenModelType':
        """
        Get the QwenModelType required for this voice type.

        Story 4.5: Model Lazy Loading
        Story 1.6: Emotion Control for Saved Voice
        - BUNDLED voices use CustomVoice model
        - DESIGNED voices use VoiceDesign model
        - CLONED voices use Base model
        - OPTIMIZED voices use fine-tuned checkpoint (loaded as CUSTOM_VOICE variant)
        - EMBEDDING voices use Base model with voice_clone_prompt
          (emotion via multiple embeddings, not instruct parameter)

        Returns:
            QwenModelType: The model type required for TTS generation
        """
        from myvoice.models.service_enums import QwenModelType
        model_map = {
            VoiceType.BUNDLED: QwenModelType.CUSTOM_VOICE,
            VoiceType.DESIGNED: QwenModelType.VOICE_DESIGN,
            VoiceType.CLONED: QwenModelType.BASE,
            VoiceType.OPTIMIZED: QwenModelType.CUSTOM_VOICE,  # Fine-tuned uses custom_voice generation
            VoiceType.EMBEDDING: QwenModelType.BASE  # Embeddings use Base model with voice_clone_prompt
        }
        return model_map.get(self, QwenModelType.CUSTOM_VOICE)


# Valid emotion names for Emotion Variants feature
# Emotion Variants: EMBEDDING voices can have multiple embeddings, one per emotion
VALID_EMOTIONS = ["neutral", "happy", "sad", "angry", "flirtatious"]


# Bundled speaker metadata for CustomVoice model timbres
# Task: Integrate 9 Template Timbres into Voice Library
BUNDLED_SPEAKERS = {
    "Vivian": {
        "description": "Bright, slightly edgy young female",
        "language": "Chinese",
        "gender": "female",
    },
    "Serena": {
        "description": "Warm, gentle young female",
        "language": "Chinese",
        "gender": "female",
    },
    "Uncle_Fu": {
        "description": "Seasoned male, low and mellow voice",
        "language": "Chinese",
        "gender": "male",
    },
    "Dylan": {
        "description": "Youthful Beijing male",
        "language": "Chinese",
        "gender": "male",
    },
    "Eric": {
        "description": "Lively Chengdu male",
        "language": "Chinese",
        "gender": "male",
    },
    "Ryan": {
        "description": "Dynamic male with strong rhythm",
        "language": "English",
        "gender": "male",
    },
    "Aiden": {
        "description": "Sunny American male",
        "language": "English",
        "gender": "male",
    },
    "Ono_Anna": {
        "description": "Playful Japanese female",
        "language": "Japanese",
        "gender": "female",
    },
    "Sohee": {
        "description": "Warm Korean female",
        "language": "Korean",
        "gender": "female",
    },
}


@dataclass
class VoiceProfile:
    """
    Data model for voice cloning profiles.

    A VoiceProfile represents a voice sample file used for TTS generation with
    voice cloning. It includes validation for audio format, duration constraints,
    transcription status tracking, and metadata about the voice sample.

    Attributes:
        file_path: Path to the voice sample WAV file (or virtual path for bundled/optimized)
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
        emotion_capable: Whether this voice supports emotion control (Story 3.3: FR8a)
                        False for cloned voices (Base model), True for CustomVoice/VoiceDesign/Optimized
        voice_type: Voice type for library categorization (Story 4.1: FR11)
                   BUNDLED (📦), DESIGNED (🎭), CLONED (🎤), or OPTIMIZED (🔧)
        checkpoint_path: Path to fine-tuned model checkpoint (OPTIMIZED voices only)
        speaker_name: Speaker name used during fine-tuning (OPTIMIZED voices only)
        description: Voice description text (DESIGNED/OPTIMIZED voices)
        available_emotions: List of available emotion embeddings for EMBEDDING voices
                           (Emotion Variants feature). Valid values: neutral, happy, sad,
                           angry, flirtatious. Default: ["neutral"]
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
    emotion_capable: bool = False  # Story 3.3: FR8a - cloned voices don't support emotion
    voice_type: VoiceType = VoiceType.CLONED  # Story 4.1: Voice type for library categorization
    # OPTIMIZED voice metadata
    checkpoint_path: Optional[Path] = None  # Path to fine-tuned checkpoint directory
    speaker_name: Optional[str] = None  # Speaker name registered during fine-tuning
    description: Optional[str] = None  # Voice description (for DESIGNED/OPTIMIZED)
    # Emotion Variants: Available emotions for EMBEDDING voices
    available_emotions: List[str] = field(default_factory=lambda: ["neutral"])

    def __post_init__(self):
        """Initialize and validate the voice profile."""
        # Convert string path to Path object if needed
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

        # Convert checkpoint_path to Path if needed
        if self.checkpoint_path and isinstance(self.checkpoint_path, str):
            self.checkpoint_path = Path(self.checkpoint_path)

        # Auto-generate name from filename if not provided
        if not self.name:
            self.name = self.file_path.stem

        # Validate available_emotions (Emotion Variants feature)
        if self.available_emotions:
            invalid_emotions = [e for e in self.available_emotions if e not in VALID_EMOTIONS]
            if invalid_emotions:
                logger.warning(
                    f"Invalid emotions {invalid_emotions} for {self.name}. "
                    f"Valid: {VALID_EMOTIONS}"
                )
                # Filter to only valid emotions
                self.available_emotions = [e for e in self.available_emotions if e in VALID_EMOTIONS]
            # Ensure at least neutral is present
            if not self.available_emotions:
                self.available_emotions = ["neutral"]

        # Check if this is a bundled voice (virtual path: bundled://SpeakerName)
        # Bundled voices are pre-trained timbres embedded in the CustomVoice model
        # Note: On Windows, Path("bundled://X") becomes "bundled:\X", so check both formats
        file_path_str = str(self.file_path)
        is_bundled_path = (
            file_path_str.startswith("bundled://") or
            file_path_str.startswith("bundled:\\") or
            file_path_str.startswith("bundled:")
        )

        # Check if this is an optimized voice (virtual path: optimized://VoiceName)
        # Optimized voices are fine-tuned from user training data
        # Note: On Windows, Path("optimized://X") becomes "optimized:\X", so check both formats
        is_optimized_path = (
            file_path_str.startswith("optimized://") or
            file_path_str.startswith("optimized:\\") or
            file_path_str.startswith("optimized:")
        )

        # Check if this is an embedding voice (virtual path: embedding://VoiceName)
        # Embedding voices use acoustic embeddings from Voice Design Studio (Story 1.6)
        # Note: On Windows, Path("embedding://X") becomes "embedding:\X", so check both formats
        is_embedding_path = (
            file_path_str.startswith("embedding://") or
            file_path_str.startswith("embedding:\\") or
            file_path_str.startswith("embedding:")
        )

        if is_bundled_path:
            # Bundled voices don't need file validation
            # They use pre-trained timbres embedded in the CustomVoice model
            self.is_valid = True
            self.voice_type = VoiceType.BUNDLED
            self.emotion_capable = True
            self.transcription_status = TranscriptionStatus.SKIPPED
            logger.debug(f"Bundled voice profile initialized: {self.name}")
            return

        if is_optimized_path:
            # Optimized voices use fine-tuned checkpoints, no audio file needed
            # Validation depends on checkpoint_path existence
            if self.checkpoint_path and self.checkpoint_path.exists():
                self.is_valid = True
            else:
                self.is_valid = False
                logger.warning(f"Optimized voice checkpoint not found: {self.checkpoint_path}")
            self.voice_type = VoiceType.OPTIMIZED
            self.emotion_capable = True  # Optimized voices support emotion presets
            self.transcription_status = TranscriptionStatus.SKIPPED
            logger.debug(f"Optimized voice profile initialized: {self.name}")
            return

        if is_embedding_path:
            # Embedding voices use acoustic embeddings (.pt files), no audio file needed
            # Validation depends on checkpoint_path (embedding file) existence
            # Story 1.6: Emotion Control for Saved Voice
            if self.checkpoint_path and self.checkpoint_path.exists():
                self.is_valid = True
            else:
                self.is_valid = False
                logger.warning(f"Embedding file not found: {self.checkpoint_path}")
            self.voice_type = VoiceType.EMBEDDING
            self.emotion_capable = True  # Embedding voices support emotion presets
            self.transcription_status = TranscriptionStatus.SKIPPED
            logger.debug(f"Embedding voice profile initialized: {self.name}")
            return

        # Validate the profile (for file-based voices: CLONED and DESIGNED)
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

        # Story 4.1: Auto-sync emotion_capable with voice_type
        # Bundled, Designed, and Optimized voices support emotion; Cloned voices don't
        self.emotion_capable = self.voice_type.supports_emotion

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

    @classmethod
    def create_bundled_profile(cls, speaker_name: str) -> 'VoiceProfile':
        """
        Factory method to create a VoiceProfile for a bundled CustomVoice speaker.

        Task: Integrate 9 Template Timbres into Voice Library

        Bundled profiles don't have audio files - they use pre-trained timbres
        embedded in the Qwen3-TTS CustomVoice model. These profiles are created
        programmatically and marked as BUNDLED type with emotion support.

        Args:
            speaker_name: Name of the bundled speaker (e.g., "Ryan", "Vivian")

        Returns:
            VoiceProfile: Bundled voice profile instance

        Raises:
            ValueError: If speaker_name is not a valid bundled speaker

        Examples:
            profile = VoiceProfile.create_bundled_profile("Ryan")
            profile = VoiceProfile.create_bundled_profile("Vivian")
        """
        if speaker_name not in BUNDLED_SPEAKERS:
            valid_speakers = ", ".join(BUNDLED_SPEAKERS.keys())
            raise ValueError(
                f"Unknown bundled speaker: {speaker_name}. "
                f"Valid speakers: {valid_speakers}"
            )

        speaker_info = BUNDLED_SPEAKERS[speaker_name]

        logger.debug(f"Creating bundled voice profile: {speaker_name}")

        # Create profile with bundled-specific attributes
        # file_path is set to a placeholder since bundled voices don't have files
        profile = cls(
            file_path=Path(f"bundled://{speaker_name}"),  # Virtual path marker
            name=speaker_name,
            transcription=None,  # Bundled voices don't need transcription
            duration=None,  # No audio file, no duration
            is_valid=True,  # Bundled voices are always valid
            transcription_status=TranscriptionStatus.SKIPPED,  # Not applicable
            emotion_capable=True,  # AC: emotion_capable=True
            voice_type=VoiceType.BUNDLED,  # AC: voice_type=VoiceType.BUNDLED
        )

        # Override post_init validation (bundled profiles are always valid)
        profile.is_valid = True

        logger.info(f"Created bundled voice profile: {speaker_name} ({speaker_info['description']})")
        return profile

    @classmethod
    def create_optimized_profile(
        cls,
        name: str,
        checkpoint_path: Path,
        speaker_name: str,
        description: Optional[str] = None
    ) -> 'VoiceProfile':
        """
        Factory method to create a VoiceProfile for a fine-tuned optimized voice.

        Optimized voices are created by fine-tuning the Qwen3-TTS Base model on
        user-provided training data. Unlike cloned voices, optimized voices
        support emotional presets because they use the generate_custom_voice()
        method with the fine-tuned checkpoint.

        Args:
            name: Display name for the voice profile
            checkpoint_path: Path to the fine-tuned model checkpoint directory
            speaker_name: Speaker name registered during fine-tuning (used in generation)
            description: Optional description of the voice characteristics

        Returns:
            VoiceProfile: Optimized voice profile instance

        Raises:
            ValueError: If checkpoint_path doesn't exist or is invalid

        Examples:
            profile = VoiceProfile.create_optimized_profile(
                name="My Custom Voice",
                checkpoint_path=Path("models/my_voice_checkpoint"),
                speaker_name="speaker_1",
                description="Warm, friendly voice optimized from training samples"
            )
        """
        # Convert string to Path if needed
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)

        # Validate checkpoint path exists
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

        # Check for expected checkpoint files (config.json or pytorch_model.bin)
        has_valid_checkpoint = (
            (checkpoint_path / "config.json").exists() or
            (checkpoint_path / "pytorch_model.bin").exists() or
            (checkpoint_path / "model.safetensors").exists()
        )
        if not has_valid_checkpoint:
            logger.warning(
                f"Checkpoint directory exists but may not contain valid model files: {checkpoint_path}"
            )

        logger.debug(f"Creating optimized voice profile: {name} from {checkpoint_path}")

        # Create profile with optimized-specific attributes
        profile = cls(
            file_path=Path(f"optimized://{name}"),  # Virtual path marker
            name=name,
            transcription=None,
            duration=None,
            is_valid=True,
            transcription_status=TranscriptionStatus.SKIPPED,
            emotion_capable=True,  # Optimized voices support emotion presets
            voice_type=VoiceType.OPTIMIZED,
            checkpoint_path=checkpoint_path,
            speaker_name=speaker_name,
            description=description
        )

        logger.info(f"Created optimized voice profile: {name} (speaker: {speaker_name})")
        return profile

    @classmethod
    def create_embedding_profile(
        cls,
        name: str,
        embedding_path: Path,
        description: Optional[str] = None,
        preview_audio_path: Optional[Path] = None,
        available_emotions: Optional[List[str]] = None,
        transcription: Optional[str] = None
    ) -> 'VoiceProfile':
        """
        Factory method to create a VoiceProfile for an acoustic embedding voice.

        Story 1.6: Emotion Control for Saved Voice
        Emotion Variants: Multiple embeddings per voice enable emotion support

        Embedding voices are created by the Voice Design Studio. They contain
        acoustic embeddings (.pt files) that capture voice characteristics.
        Emotion support is achieved via multiple embeddings (one per emotion),
        not via the instruct parameter (uses Base model, not VoiceDesign).

        Args:
            name: Display name for the voice profile
            embedding_path: Path to the embedding .pt file (or base directory for multi-emotion)
            description: Voice description text
            preview_audio_path: Optional path to preview audio file
            available_emotions: List of emotions this voice supports (Emotion Variants).
                               Valid values: neutral, happy, sad, angry, flirtatious.
                               Default: ["neutral"]
            transcription: Reference text used for embedding (ref_text for TTS generation)

        Returns:
            VoiceProfile: Embedding voice profile instance

        Raises:
            ValueError: If embedding_path doesn't exist

        Examples:
            # Single-emotion (legacy) voice
            profile = VoiceProfile.create_embedding_profile(
                name="My Voice",
                embedding_path=Path("voice_files/embeddings/MyVoice/embedding.pt"),
                description="Warm, friendly voice with slight accent"
            )

            # Multi-emotion voice (Emotion Variants)
            profile = VoiceProfile.create_embedding_profile(
                name="Expressive Voice",
                embedding_path=Path("voice_files/embeddings/ExpressiveVoice"),
                description="Voice with full emotion range",
                available_emotions=["neutral", "happy", "sad", "angry", "flirtatious"]
            )
        """
        # Convert string to Path if needed
        if isinstance(embedding_path, str):
            embedding_path = Path(embedding_path)

        # Validate embedding path exists
        if not embedding_path.exists():
            raise ValueError(f"Embedding file does not exist: {embedding_path}")

        # Get duration from preview audio if available
        duration = None
        if preview_audio_path and preview_audio_path.exists():
            try:
                import wave
                with wave.open(str(preview_audio_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    framerate = wav_file.getframerate()
                    duration = frames / framerate if framerate > 0 else None
            except Exception as e:
                logger.debug(f"Could not extract duration from preview audio: {e}")

        logger.debug(f"Creating embedding voice profile: {name} from {embedding_path}")

        # Default to neutral-only if no emotions specified
        if available_emotions is None:
            available_emotions = ["neutral"]

        # Create profile with embedding-specific attributes
        # Use embedding:// virtual path marker for consistency
        # Emotion Variants: EMBEDDING voices support emotion via multiple embeddings
        profile = cls(
            file_path=Path(f"embedding://{name}"),  # Virtual path marker
            name=name,
            transcription=transcription,  # ref_text for TTS generation
            duration=duration,
            is_valid=True,
            transcription_status=TranscriptionStatus.COMPLETED if transcription else TranscriptionStatus.SKIPPED,
            emotion_capable=True,  # Supports emotion via multiple embeddings (not instruct)
            voice_type=VoiceType.EMBEDDING,
            checkpoint_path=embedding_path,  # Store embedding path in checkpoint_path field
            speaker_name=None,  # Not used for embeddings
            description=description,
            available_emotions=available_emotions  # Emotion Variants
        )

        logger.info(f"Created embedding voice profile: {name} (emotions: {available_emotions})")
        return profile

    def is_bundled(self) -> bool:
        """
        Check if this is a bundled voice profile.

        Returns:
            bool: True if this is a bundled voice (no audio file required)
        """
        return self.voice_type == VoiceType.BUNDLED

    def is_optimized(self) -> bool:
        """
        Check if this is an optimized (fine-tuned) voice profile.

        Returns:
            bool: True if this is an optimized voice with a fine-tuned checkpoint
        """
        return self.voice_type == VoiceType.OPTIMIZED

    def is_embedding(self) -> bool:
        """
        Check if this is an embedding voice profile.

        Story 1.6: Emotion Control for Saved Voice

        Returns:
            bool: True if this is an embedding voice from Voice Design Studio
        """
        return self.voice_type == VoiceType.EMBEDDING

    def get_embedding_path(self) -> Optional[Path]:
        """
        Get the embedding file path for embedding voices.

        Returns:
            Path to embedding .pt file, or None if not an embedding voice
        """
        if self.voice_type == VoiceType.EMBEDDING:
            return self.checkpoint_path  # Embedding path stored in checkpoint_path field
        return None

    def has_emotion(self, emotion: str) -> bool:
        """
        Check if this voice has a specific emotion available.

        Emotion Variants: EMBEDDING voices can have multiple emotions.
        Other voice types either support all emotions (via instruct) or none.

        Args:
            emotion: Emotion name to check (e.g., "neutral", "happy")

        Returns:
            bool: True if the emotion is available for this voice
        """
        if self.voice_type == VoiceType.EMBEDDING:
            return emotion in self.available_emotions
        elif self.voice_type in (VoiceType.BUNDLED, VoiceType.DESIGNED, VoiceType.OPTIMIZED):
            # These voice types support all emotions via instruct parameter
            return emotion in VALID_EMOTIONS
        else:
            # CLONED voices don't support emotion control
            return emotion == "neutral"

    def get_available_emotions(self) -> List[str]:
        """
        Get list of available emotions for this voice.

        Emotion Variants: Returns the available_emotions list for EMBEDDING voices,
        all valid emotions for instruct-capable voices, or just neutral for others.

        Returns:
            List[str]: List of available emotion names
        """
        if self.voice_type == VoiceType.EMBEDDING:
            return self.available_emotions.copy()
        elif self.voice_type in (VoiceType.BUNDLED, VoiceType.DESIGNED, VoiceType.OPTIMIZED):
            return VALID_EMOTIONS.copy()
        else:
            return ["neutral"]

    # =========================================================================
    # Emotion Variants: Path helpers for emotion-specific embeddings
    # =========================================================================

    def get_embedding_base_dir(self) -> Optional[Path]:
        """
        Get the base directory for this voice's embeddings.

        Emotion Variants: For multi-emotion voices, this is the parent directory
        containing emotion subfolders. For legacy voices, this is the directory
        containing embedding.pt directly.

        Structure:
            voice_files/embeddings/{voice_name}/  <- this directory
                neutral/embedding.pt
                happy/embedding.pt
                ...
                preview.wav
                metadata.json

        Returns:
            Path to embedding base directory, or None if not an embedding voice
        """
        if self.voice_type != VoiceType.EMBEDDING or not self.checkpoint_path:
            return None

        # checkpoint_path could be:
        # 1. Legacy: voice_files/embeddings/{name}/embedding.pt -> parent is base
        # 2. New: voice_files/embeddings/{name} -> this is base
        # 3. New: voice_files/embeddings/{name}/neutral/embedding.pt -> parent.parent is base
        if self.checkpoint_path.is_file():
            # It's a file path (legacy or emotion-specific)
            parent = self.checkpoint_path.parent
            if parent.name in VALID_EMOTIONS:
                # It's emotion-specific: {base}/{emotion}/embedding.pt
                return parent.parent
            else:
                # It's legacy: {base}/embedding.pt
                return parent
        else:
            # It's a directory path (new structure)
            return self.checkpoint_path

    def get_emotion_embedding_path(self, emotion: str) -> Optional[Path]:
        """
        Get the path to a specific emotion's embedding file.

        Emotion Variants: Returns the path to {base_dir}/{emotion}/embedding.pt

        Args:
            emotion: Emotion name (e.g., "neutral", "happy")

        Returns:
            Path to the emotion's embedding.pt file, or None if not available
        """
        if emotion not in VALID_EMOTIONS:
            logger.warning(f"Invalid emotion: {emotion}")
            return None

        base_dir = self.get_embedding_base_dir()
        if not base_dir:
            return None

        # Check for emotion subfolder structure first
        emotion_path = base_dir / emotion / "embedding.pt"
        if emotion_path.exists():
            return emotion_path

        # Legacy fallback: single embedding.pt at root (treated as neutral)
        if emotion == "neutral":
            legacy_path = base_dir / "embedding.pt"
            if legacy_path.exists():
                return legacy_path

        return None

    def get_emotion_source_audio_path(self, emotion: str) -> Optional[Path]:
        """
        Get the path to a specific emotion's source audio file.

        Emotion Variants: Returns the path to {base_dir}/{emotion}/source_audio.wav

        Args:
            emotion: Emotion name (e.g., "neutral", "happy")

        Returns:
            Path to the emotion's source audio, or None if not available
        """
        if emotion not in VALID_EMOTIONS:
            return None

        base_dir = self.get_embedding_base_dir()
        if not base_dir:
            return None

        source_path = base_dir / emotion / "source_audio.wav"
        return source_path if source_path.exists() else None

    def get_preview_audio_path(self) -> Optional[Path]:
        """
        Get the path to this voice's preview audio file.

        Returns:
            Path to preview.wav, or None if not available
        """
        base_dir = self.get_embedding_base_dir()
        if not base_dir:
            return None

        preview_path = base_dir / "preview.wav"
        return preview_path if preview_path.exists() else None

    def get_metadata_path(self) -> Optional[Path]:
        """
        Get the path to this voice's metadata.json file.

        Returns:
            Path to metadata.json, or None if not an embedding voice
        """
        base_dir = self.get_embedding_base_dir()
        if not base_dir:
            return None

        return base_dir / "metadata.json"

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

    def to_embedding_metadata(self) -> dict:
        """
        Generate metadata.json content for embedding voice storage.

        Emotion Variants: Creates the metadata structure for saving embedding voices
        with emotion subfolders.

        Schema (version 2.0):
        {
            "version": "2.0",
            "name": "Voice Name",
            "description": "Voice description text",
            "available_emotions": ["neutral", "happy", "sad"],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }

        Returns:
            dict: Metadata structure for JSON serialization
        """
        return {
            "version": "2.0",  # Emotion Variants schema version
            "name": self.name,
            "description": self.description,
            "available_emotions": self.available_emotions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @staticmethod
    def detect_available_emotions(voice_dir: Path) -> List[str]:
        """
        Detect available emotions from a voice directory's folder structure.

        Emotion Variants: Scans for emotion subfolders containing embedding.pt files.
        Falls back to ["neutral"] for legacy single-embedding structure.

        Directory structures supported:
        1. New (v2.0): emotion subfolders with embedding.pt
           voice_dir/neutral/embedding.pt
           voice_dir/happy/embedding.pt
           ...

        2. Legacy (v1.0): single embedding.pt at root
           voice_dir/embedding.pt

        Args:
            voice_dir: Path to the voice's embedding directory

        Returns:
            List[str]: List of detected emotion names, sorted in VALID_EMOTIONS order
        """
        if not voice_dir.exists() or not voice_dir.is_dir():
            return ["neutral"]

        detected = []

        # Check for emotion subfolders
        for emotion in VALID_EMOTIONS:
            emotion_dir = voice_dir / emotion
            embedding_file = emotion_dir / "embedding.pt"
            if embedding_file.exists():
                detected.append(emotion)

        # If emotion subfolders found, return them
        if detected:
            # Sort in VALID_EMOTIONS order for consistency
            return [e for e in VALID_EMOTIONS if e in detected]

        # Legacy fallback: check for root embedding.pt
        root_embedding = voice_dir / "embedding.pt"
        if root_embedding.exists():
            return ["neutral"]

        # No embeddings found
        return []

    @staticmethod
    def parse_embedding_metadata(metadata_path: Path) -> dict:
        """
        Parse metadata.json from an embedding voice directory.

        Handles both legacy (v1.0) and new (v2.0) metadata formats.

        Args:
            metadata_path: Path to metadata.json file

        Returns:
            dict: Parsed metadata with normalized fields:
                - version: "1.0" or "2.0"
                - name: Voice name
                - description: Voice description or None
                - available_emotions: List of emotions (default ["neutral"] for v1.0)
        """
        import json

        if not metadata_path.exists():
            return {
                "version": "1.0",
                "name": metadata_path.parent.name,
                "description": None,
                "available_emotions": ["neutral"]
            }

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Normalize to v2.0 schema
            version = data.get("version", "1.0")
            name = data.get("name", metadata_path.parent.name)
            description = data.get("description")

            # Handle available_emotions
            if "available_emotions" in data:
                available_emotions = data["available_emotions"]
            else:
                # Legacy v1.0: Single embedding = neutral only
                available_emotions = ["neutral"]

            # Validate emotions
            available_emotions = [e for e in available_emotions if e in VALID_EMOTIONS]
            if not available_emotions:
                available_emotions = ["neutral"]

            return {
                "version": version,
                "name": name,
                "description": description,
                "available_emotions": available_emotions
            }

        except Exception as e:
            logger.warning(f"Error parsing metadata from {metadata_path}: {e}")
            return {
                "version": "1.0",
                "name": metadata_path.parent.name,
                "description": None,
                "available_emotions": ["neutral"]
            }

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
            'updated_at': self.updated_at.isoformat(),
            'emotion_capable': self.emotion_capable,  # Story 3.3: FR8a
            'voice_type': self.voice_type.value,  # Story 4.1: FR11
            # OPTIMIZED voice metadata
            'checkpoint_path': str(self.checkpoint_path) if self.checkpoint_path else None,
            'speaker_name': self.speaker_name,
            'description': self.description,
            # Emotion Variants
            'available_emotions': self.available_emotions,
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

        # Parse voice type (Story 4.1: FR11)
        voice_type = VoiceType.CLONED  # Default to CLONED for legacy profiles
        if data.get('voice_type'):
            try:
                voice_type = VoiceType(data['voice_type'])
            except ValueError:
                pass

        # Parse checkpoint_path for OPTIMIZED voices
        checkpoint_path = None
        if data.get('checkpoint_path'):
            checkpoint_path = Path(data['checkpoint_path'])

        # Parse available_emotions (Emotion Variants)
        available_emotions = data.get('available_emotions', ["neutral"])
        if not isinstance(available_emotions, list):
            available_emotions = ["neutral"]

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
            updated_at=updated_at,
            emotion_capable=data.get('emotion_capable', False),  # Story 3.3: FR8a
            voice_type=voice_type,  # Story 4.1: FR11
            # OPTIMIZED voice metadata
            checkpoint_path=checkpoint_path,
            speaker_name=data.get('speaker_name'),
            description=data.get('description'),
            # Emotion Variants
            available_emotions=available_emotions,
        )