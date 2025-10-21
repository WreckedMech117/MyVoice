"""
TranscriptionResult Data Models

This module contains comprehensive data models for audio transcription results,
including validation methods and factory methods for creating results from Whisper output.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pathlib import Path

from myvoice.models.validation import ValidationResult, ValidationIssue, ValidationStatus, UserFriendlyMessage
from myvoice.models.error import MyVoiceError, ErrorSeverity


logger = logging.getLogger(__name__)


class TranscriptionStatus(Enum):
    """Status of transcription operation."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    PROCESSING = "processing"


class LanguageConfidence(Enum):
    """Language detection confidence levels."""
    HIGH = "high"      # > 0.9
    MEDIUM = "medium"  # 0.7 - 0.9
    LOW = "low"        # 0.5 - 0.7
    VERY_LOW = "very_low"  # < 0.5


@dataclass
class TranscriptionSegment:
    """
    A segment of transcribed audio with timing information.

    Attributes:
        id: Segment identifier
        start: Start time in seconds
        end: End time in seconds
        text: Transcribed text for this segment
        tokens: Token IDs used by the model
        temperature: Temperature used for this segment
        avg_logprob: Average log probability
        compression_ratio: Compression ratio
        no_speech_prob: Probability that this segment contains no speech
    """
    id: int
    start: float
    end: float
    text: str
    tokens: Optional[List[int]] = None
    temperature: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None

    def __post_init__(self):
        """Validate segment data."""
        if self.start < 0:
            raise ValueError("Start time cannot be negative")
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")
        if not self.text.strip():
            raise ValueError("Segment text cannot be empty")

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end - self.start

    @property
    def confidence_score(self) -> float:
        """
        Calculate confidence score based on available metrics.

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if self.avg_logprob is not None and self.no_speech_prob is not None:
            # Convert log probability to probability and account for no_speech_prob
            prob_score = min(1.0, max(0.0, 1.0 + self.avg_logprob))
            speech_confidence = 1.0 - self.no_speech_prob
            return (prob_score + speech_confidence) / 2.0
        elif self.avg_logprob is not None:
            return min(1.0, max(0.0, 1.0 + self.avg_logprob))
        elif self.no_speech_prob is not None:
            return 1.0 - self.no_speech_prob
        else:
            return 0.5  # Default when no confidence metrics available


@dataclass
class WordTimestamp:
    """
    Word-level timestamp information.

    Attributes:
        word: The transcribed word
        start: Start time in seconds
        end: End time in seconds
        confidence: Confidence score for this word (0.0 - 1.0)
    """
    word: str
    start: float
    end: float
    confidence: Optional[float] = None

    def __post_init__(self):
        """Validate word timestamp data."""
        if self.start < 0:
            raise ValueError("Start time cannot be negative")
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")
        if not self.word.strip():
            raise ValueError("Word cannot be empty")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def duration(self) -> float:
        """Duration of this word in seconds."""
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """
    Comprehensive result of audio transcription with validation and factory methods.

    This model represents the complete output from a transcription operation,
    including text, confidence metrics, timing information, and status.

    Attributes:
        text: Full transcribed text
        confidence: Overall confidence score (0.0 - 1.0)
        duration: Total audio duration in seconds
        language: Detected/specified language code
        status: Transcription operation status
        error_message: Error description if transcription failed
        segments: List of transcription segments with timing
        word_timestamps: Word-level timestamps (optional)
        model_name: Name/size of the model used
        processing_time: Time taken to process (seconds)
        created_at: Timestamp when result was created
        source_file: Path to the source audio file
        metadata: Additional metadata from the transcription
    """
    text: str
    confidence: float
    duration: float
    language: str
    status: TranscriptionStatus = TranscriptionStatus.SUCCESS
    error_message: Optional[str] = None
    segments: List[TranscriptionSegment] = field(default_factory=list)
    word_timestamps: Optional[List[WordTimestamp]] = None
    model_name: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize transcription result data."""
        # Validate confidence score
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

        # Validate duration
        if self.duration < 0:
            raise ValueError("Duration cannot be negative")

        # Validate language code (basic check)
        if not self.language or len(self.language) < 2:
            raise ValueError("Language code must be at least 2 characters")

        # Ensure error message is present for failed status
        if self.status == TranscriptionStatus.FAILED and not self.error_message:
            raise ValueError("Error message is required for failed status")

        # Convert string path to Path object if needed
        if isinstance(self.source_file, str):
            self.source_file = Path(self.source_file)

    @property
    def is_successful(self) -> bool:
        """Whether the transcription was successful."""
        return self.status == TranscriptionStatus.SUCCESS

    @property
    def has_segments(self) -> bool:
        """Whether the result includes segment information."""
        return len(self.segments) > 0

    @property
    def has_word_timestamps(self) -> bool:
        """Whether the result includes word-level timestamps."""
        return self.word_timestamps is not None and len(self.word_timestamps) > 0

    @property
    def language_confidence(self) -> LanguageConfidence:
        """Language detection confidence level."""
        if self.confidence > 0.9:
            return LanguageConfidence.HIGH
        elif self.confidence > 0.7:
            return LanguageConfidence.MEDIUM
        elif self.confidence > 0.5:
            return LanguageConfidence.LOW
        else:
            return LanguageConfidence.VERY_LOW

    @property
    def word_count(self) -> int:
        """Number of words in the transcribed text."""
        return len(self.text.split()) if self.text else 0

    @property
    def average_segment_confidence(self) -> float:
        """Average confidence score across all segments."""
        if not self.segments:
            return self.confidence

        segment_scores = [seg.confidence_score for seg in self.segments]
        return sum(segment_scores) / len(segment_scores)

    def validate(self) -> ValidationResult:
        """
        Validate the transcription result for completeness and consistency.

        Returns:
            ValidationResult indicating any issues found
        """
        issues = []

        try:
            # Check text quality
            if not self.text or not self.text.strip():
                issues.append(ValidationIssue(
                    field="text",
                    message="Transcribed text is empty",
                    code="EMPTY_TEXT",
                    severity=ValidationStatus.INVALID
                ))

            # Check confidence threshold
            if self.confidence < 0.3:
                issues.append(ValidationIssue(
                    field="confidence",
                    message=f"Low confidence score: {self.confidence:.2f}",
                    code="LOW_CONFIDENCE",
                    severity=ValidationStatus.WARNING
                ))

            # Check segment consistency
            if self.segments:
                total_segment_duration = sum(seg.duration for seg in self.segments)
                if abs(total_segment_duration - self.duration) > 1.0:  # 1 second tolerance
                    issues.append(ValidationIssue(
                        field="segments",
                        message="Segment durations don't match total duration",
                        code="DURATION_MISMATCH",
                        severity=ValidationStatus.WARNING
                    ))

            # Check for overlapping segments
            for i, segment in enumerate(self.segments[:-1]):
                next_segment = self.segments[i + 1]
                if segment.end > next_segment.start:
                    issues.append(ValidationIssue(
                        field="segments",
                        message=f"Overlapping segments at {segment.end:.2f}s",
                        code="OVERLAPPING_SEGMENTS",
                        severity=ValidationStatus.WARNING
                    ))

            # Check word timestamp consistency
            if self.word_timestamps:
                for word in self.word_timestamps:
                    if word.start >= word.end:
                        issues.append(ValidationIssue(
                            field="word_timestamps",
                            message=f"Invalid word timing: {word.word}",
                            code="INVALID_WORD_TIMING",
                            severity=ValidationStatus.INVALID
                        ))

            if issues:
                errors = [issue for issue in issues if issue.severity == ValidationStatus.INVALID]
                warnings = [issue for issue in issues if issue.severity == ValidationStatus.WARNING]

                return ValidationResult(
                    is_valid=len(errors) == 0,
                    status=ValidationStatus.INVALID if errors else ValidationStatus.WARNING,
                    issues=errors,
                    warnings=warnings
                )

            return ValidationResult(
                is_valid=True,
                status=ValidationStatus.VALID,
                issues=[],
                warnings=[]
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                issues=[ValidationIssue(
                    field="validation",
                    message=f"Validation error: {e}",
                    code="VALIDATION_ERROR",
                    severity=ValidationStatus.INVALID
                )],
                warnings=[]
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert transcription result to dictionary format.

        Returns:
            Dictionary representation of the transcription result
        """
        return {
            "text": self.text,
            "confidence": self.confidence,
            "duration": self.duration,
            "language": self.language,
            "status": self.status.value,
            "error_message": self.error_message,
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "confidence": seg.confidence_score,
                    "tokens": seg.tokens,
                    "temperature": seg.temperature,
                    "avg_logprob": seg.avg_logprob,
                    "compression_ratio": seg.compression_ratio,
                    "no_speech_prob": seg.no_speech_prob,
                }
                for seg in self.segments
            ],
            "word_timestamps": [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "confidence": word.confidence,
                }
                for word in self.word_timestamps
            ] if self.word_timestamps else None,
            "model_name": self.model_name,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat(),
            "source_file": str(self.source_file) if self.source_file else None,
            "metadata": self.metadata,
            "word_count": self.word_count,
            "language_confidence": self.language_confidence.value,
        }

    @classmethod
    def from_whisper_output(
        cls,
        whisper_result: Dict[str, Any],
        source_file: Optional[Path] = None,
        model_name: Optional[str] = None,
        processing_time: Optional[float] = None
    ) -> "TranscriptionResult":
        """
        Factory method to create TranscriptionResult from Whisper model output.

        Args:
            whisper_result: Raw output from Whisper model
            source_file: Path to the source audio file
            model_name: Name/size of the model used
            processing_time: Time taken to process

        Returns:
            TranscriptionResult instance

        Raises:
            MyVoiceError: If whisper_result format is invalid
        """
        try:
            # Extract basic information
            text = whisper_result.get("text", "").strip()
            language = whisper_result.get("language", "unknown")

            # Calculate overall confidence from segments
            segments_data = whisper_result.get("segments", [])
            if segments_data:
                confidences = []
                for seg_data in segments_data:
                    # Calculate confidence from available metrics
                    avg_logprob = seg_data.get("avg_logprob", 0.0)
                    no_speech_prob = seg_data.get("no_speech_prob", 0.0)

                    if avg_logprob is not None and no_speech_prob is not None:
                        prob_score = min(1.0, max(0.0, 1.0 + avg_logprob))
                        speech_confidence = 1.0 - no_speech_prob
                        seg_confidence = (prob_score + speech_confidence) / 2.0
                    else:
                        seg_confidence = 0.5

                    confidences.append(seg_confidence)

                overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            else:
                overall_confidence = 0.5

            # Parse segments
            segments = []
            for seg_data in segments_data:
                segment = TranscriptionSegment(
                    id=seg_data.get("id", 0),
                    start=seg_data.get("start", 0.0),
                    end=seg_data.get("end", 0.0),
                    text=seg_data.get("text", ""),
                    tokens=seg_data.get("tokens"),
                    temperature=seg_data.get("temperature"),
                    avg_logprob=seg_data.get("avg_logprob"),
                    compression_ratio=seg_data.get("compression_ratio"),
                    no_speech_prob=seg_data.get("no_speech_prob"),
                )
                segments.append(segment)

            # Parse word timestamps if available
            word_timestamps = None
            words_data = whisper_result.get("words", [])
            if words_data:
                word_timestamps = []
                for word_data in words_data:
                    word = WordTimestamp(
                        word=word_data.get("word", ""),
                        start=word_data.get("start", 0.0),
                        end=word_data.get("end", 0.0),
                        confidence=word_data.get("confidence"),
                    )
                    word_timestamps.append(word)

            # Calculate duration from segments or use provided duration
            if segments:
                duration = max(seg.end for seg in segments)
            else:
                duration = whisper_result.get("duration", 0.0)

            # Create the result
            return cls(
                text=text,
                confidence=overall_confidence,
                duration=duration,
                language=language,
                status=TranscriptionStatus.SUCCESS,
                segments=segments,
                word_timestamps=word_timestamps,
                model_name=model_name,
                processing_time=processing_time,
                source_file=source_file,
                metadata={
                    "whisper_version": whisper_result.get("whisper_version"),
                    "task": whisper_result.get("task", "transcribe"),
                    "original_result": whisper_result,
                }
            )

        except Exception as e:
            logger.error(f"Failed to create TranscriptionResult from Whisper output: {e}")
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="TRANSCRIPTION_RESULT_CREATION_FAILED",
                user_message="Failed to process transcription result",
                technical_details=str(e),
                suggested_action="Check Whisper output format and try again"
            )

    @classmethod
    def create_error_result(
        cls,
        error_message: str,
        source_file: Optional[Path] = None,
        language: str = "unknown",
        error_code: Optional[str] = None
    ) -> "TranscriptionResult":
        """
        Factory method to create a failed TranscriptionResult.

        Args:
            error_message: Description of the error
            source_file: Path to the source audio file
            language: Language code (if known)
            error_code: Optional error code

        Returns:
            TranscriptionResult with failed status
        """
        return cls(
            text="",
            confidence=0.0,
            duration=0.0,
            language=language,
            status=TranscriptionStatus.FAILED,
            error_message=error_message,
            source_file=source_file,
            metadata={"error_code": error_code} if error_code else {}
        )