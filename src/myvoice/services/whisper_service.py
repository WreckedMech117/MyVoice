"""
WhisperService Implementation

This module implements the Whisper-based transcription service with async operations,
validation, and comprehensive error handling for voice sample transcription.
"""

import asyncio
import logging
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None
    torch = None

from myvoice.services.core.base_service import BaseService
from myvoice.models.service_enums import ServiceStatus
from myvoice.models.validation import (
    ValidationResult, ValidationIssue, ValidationStatus, UserFriendlyMessage
)
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.utils.validation import require_service_health


class WhisperModelSize(Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class TranscriptionRequest:
    """Request for audio transcription."""
    audio_file_path: Path
    language: Optional[str] = None
    model_size: WhisperModelSize = WhisperModelSize.BASE
    temperature: float = 0.0
    word_timestamps: bool = False


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    text: str
    language: str
    segments: List[Dict[str, Any]]
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None


@dataclass
class ModelLoadProgress:
    """Progress information for model loading."""
    stage: str  # "validating", "downloading", "loading", "ready"
    progress_percent: float  # 0-100
    message: str
    estimated_size_mb: Optional[float] = None
    downloaded_size_mb: Optional[float] = None


@dataclass
class ModelInfo:
    """Information about a Whisper model."""
    name: str
    size: WhisperModelSize
    estimated_size_mb: float
    requires_download: bool
    device_compatible: bool


class WhisperService(BaseService):
    """
    Whisper-based transcription service with async operations.

    This service provides transcription capabilities for voice samples using
    OpenAI's Whisper model. It includes audio format validation, async processing
    to prevent UI blocking, and comprehensive error handling.

    Features:
    - Async transcription with non-blocking operations
    - Automatic model download on first use with progress feedback
    - Audio format validation (.wav, .mp3, .m4a, .flac)
    - Batch transcription support
    - Multiple Whisper model sizes with automatic validation
    - CPU-only inference configuration for compatibility
    - Voice file validation for GPT-SoVITS compatibility
    - Resource management and cleanup
    """

    # Model size estimates in MB (approximate)
    MODEL_SIZES = {
        WhisperModelSize.TINY: 39.0,
        WhisperModelSize.BASE: 74.0,
        WhisperModelSize.SMALL: 244.0,
        WhisperModelSize.MEDIUM: 769.0,
        WhisperModelSize.LARGE: 1550.0,
    }

    def __init__(
        self,
        model_size: WhisperModelSize = WhisperModelSize.BASE,
        device: Optional[str] = None,
        max_concurrent_transcriptions: int = 2,
        max_file_size_mb: int = 100,
        supported_formats: Optional[List[str]] = None,
        force_cpu: bool = False,
        progress_callback: Optional[Callable[[ModelLoadProgress], None]] = None
    ):
        """
        Initialize the WhisperService.

        Args:
            model_size: Whisper model size to use
            device: Device to run on ("cpu", "cuda", or None for auto)
            max_concurrent_transcriptions: Maximum concurrent transcription tasks
            max_file_size_mb: Maximum file size in MB
            supported_formats: List of supported audio formats
            force_cpu: Force CPU-only inference for compatibility
            progress_callback: Callback for model loading progress updates
        """
        super().__init__("WhisperService")

        self.model_size = model_size
        self.force_cpu = force_cpu
        self.max_concurrent_transcriptions = max_concurrent_transcriptions
        self.max_file_size_mb = max_file_size_mb
        self.supported_formats = supported_formats or ['.wav', '.mp3', '.m4a', '.flac']
        self.progress_callback = progress_callback

        # Determine device with CPU fallback and force_cpu option
        if force_cpu:
            self.device = "cpu"
        elif device:
            self.device = device
        else:
            # Auto-detect with fallback to CPU
            try:
                self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        self._model: Optional[Any] = None  # whisper.Whisper when available
        self._executor: Optional[ThreadPoolExecutor] = None
        self._transcription_semaphore: Optional[asyncio.Semaphore] = None
        self._model_info: Optional[ModelInfo] = None

    def _notify_progress(self, stage: str, progress: float, message: str,
                        estimated_mb: Optional[float] = None, downloaded_mb: Optional[float] = None) -> None:
        """Notify progress callback if available."""
        if self.progress_callback:
            try:
                progress_info = ModelLoadProgress(
                    stage=stage,
                    progress_percent=progress,
                    message=message,
                    estimated_size_mb=estimated_mb,
                    downloaded_size_mb=downloaded_mb
                )
                self.progress_callback(progress_info)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def _get_model_cache_path(self) -> Optional[Path]:
        """Get the path where Whisper models are cached or bundled."""
        try:
            # First, check for bundled models in the application directory
            if getattr(sys, 'frozen', False):
                # Running in PyInstaller bundle
                app_dir = Path(sys._MEIPASS)
                bundled_model_dir = app_dir / "whisper_models"
                if bundled_model_dir.exists():
                    self.logger.info(f"Using bundled Whisper models from: {bundled_model_dir}")
                    return bundled_model_dir
            else:
                # Running from source - check for bundled models in project root
                project_root = Path(__file__).parent.parent.parent.parent
                bundled_model_dir = project_root / "whisper_models"
                if bundled_model_dir.exists():
                    self.logger.info(f"Using bundled Whisper models from: {bundled_model_dir}")
                    return bundled_model_dir

            # Fallback: use local whisper_models directory
            self.logger.warning("Bundled Whisper models not found, using local directory")
            cache_path = Path("whisper_models")
            cache_path.mkdir(parents=True, exist_ok=True)
            return cache_path

        except Exception as e:
            self.logger.warning(f"Could not determine model directory: {e}")
            return None

    def _check_model_availability(self) -> ModelInfo:
        """Check if the model is available locally or needs downloading."""
        estimated_size = self.MODEL_SIZES.get(self.model_size, 100.0)

        # Check if model is already cached
        requires_download = True
        cache_path = self._get_model_cache_path()
        if cache_path and cache_path.exists():
            # Look for model file (Whisper saves as .pt files)
            model_file = cache_path / f"{self.model_size.value}.pt"
            if model_file.exists():
                requires_download = False

        # Check device compatibility
        device_compatible = True
        if self.device == "cuda":
            try:
                device_compatible = torch and torch.cuda.is_available()
            except Exception:
                device_compatible = False

        return ModelInfo(
            name=self.model_size.value,
            size=self.model_size,
            estimated_size_mb=estimated_size,
            requires_download=requires_download,
            device_compatible=device_compatible
        )

    def _validate_system_resources(self, model_info: ModelInfo) -> None:
        """Validate system has enough resources for the model."""
        # Check available disk space if downloading is required
        if model_info.requires_download:
            cache_path = self._get_model_cache_path()
            if cache_path:
                try:
                    import shutil
                    free_space_bytes = shutil.disk_usage(cache_path.parent).free
                    free_space_mb = free_space_bytes / (1024 * 1024)

                    # Need at least 2x model size for download + extraction
                    required_space_mb = model_info.estimated_size_mb * 2

                    if free_space_mb < required_space_mb:
                        raise MyVoiceError(
                            severity=ErrorSeverity.CRITICAL,
                            code="INSUFFICIENT_DISK_SPACE",
                            user_message=f"Not enough disk space to download {model_info.name} model",
                            technical_details=f"Need {required_space_mb:.1f}MB, available: {free_space_mb:.1f}MB",
                            suggested_action="Free up disk space or use a smaller model"
                        )
                except ImportError:
                    pass  # shutil not available, skip disk space check
                except Exception as e:
                    self.logger.warning(f"Could not check disk space: {e}")

        # Warn about device compatibility
        if not model_info.device_compatible and self.device == "cuda":
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

    async def start_service(self) -> None:
        """Start the Whisper service and load the model."""
        if not WHISPER_AVAILABLE:
            raise MyVoiceError(
                severity=ErrorSeverity.CRITICAL,
                code="WHISPER_NOT_AVAILABLE",
                user_message="Whisper transcription library is not available",
                technical_details="openai-whisper and torch packages are not installed",
                suggested_action="Install the required packages: pip install openai-whisper torch"
            )

        try:
            self.status = ServiceStatus.STARTING
            self.logger.info(f"Starting Whisper service with {self.model_size.value} model on {self.device}")

            # Stage 1: Validate model and system resources
            self._notify_progress("validating", 10.0, "Validating model and system resources...")
            self._model_info = self._check_model_availability()
            self._validate_system_resources(self._model_info)

            # Stage 2: Initialize executor and semaphore
            self._notify_progress("initializing", 20.0, "Initializing transcription service...")

            # Create thread pool executor for CPU-bound transcription tasks
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent_transcriptions,
                thread_name_prefix="whisper-transcription"
            )

            # Create semaphore to limit concurrent transcriptions
            self._transcription_semaphore = asyncio.Semaphore(self.max_concurrent_transcriptions)

            # Stage 3: Load Whisper model with progress tracking
            if self._model_info.requires_download:
                self._notify_progress(
                    "downloading",
                    30.0,
                    f"Downloading {self.model_size.value} model ({self._model_info.estimated_size_mb:.1f}MB)...",
                    estimated_mb=self._model_info.estimated_size_mb
                )
            else:
                self._notify_progress("loading", 60.0, f"Loading {self.model_size.value} model from cache...")

            # Load Whisper model in background thread to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                self._executor,
                self._load_whisper_model_with_progress
            )

            # Stage 4: Final validation and completion
            self._notify_progress("ready", 100.0, "Whisper service ready for transcription")

            self.status = ServiceStatus.RUNNING
            self.logger.info(f"Whisper service started successfully with {self.model_size.value} model on {self.device}")

        except MyVoiceError:
            # Re-raise MyVoiceError as-is
            self.status = ServiceStatus.ERROR
            raise
        except Exception as e:
            self.status = ServiceStatus.ERROR
            error = MyVoiceError(
                severity=ErrorSeverity.CRITICAL,
                code="WHISPER_SERVICE_START_FAILED",
                user_message="Failed to start transcription service",
                technical_details=str(e),
                suggested_action="Check if system has enough memory and GPU drivers are installed"
            )
            self.logger.error(f"Failed to start Whisper service: {e}")
            raise error

    def _load_whisper_model_with_progress(self) -> Any:
        """Load the Whisper model with progress tracking and comprehensive error handling (runs in thread pool)."""
        import time
        import gc
        import tempfile

        try:
            # Configure Whisper cache directory to ensure it's writable
            # This fixes 'NoneType' object has no attribute 'write' errors on Windows
            cache_dir = self._get_model_cache_path()
            if cache_dir:
                os.environ["WHISPER_CACHE_DIR"] = str(cache_dir)
                os.environ["XDG_CACHE_HOME"] = str(cache_dir.parent)
                self.logger.info(f"Using Whisper cache directory: {cache_dir}")
            else:
                self.logger.warning("Could not determine cache directory, using Whisper defaults")

            # Progress update during loading
            progress_stage = "downloading" if self._model_info and self._model_info.requires_download else "loading"
            progress_start = 30.0 if progress_stage == "downloading" else 60.0

            # Simulate progress updates during loading
            # Note: Whisper doesn't provide direct download progress, so we estimate
            try:
                # First progress update
                progress = progress_start + 20
                asyncio.get_event_loop().call_soon_threadsafe(
                    lambda: self._notify_progress(progress_stage, progress, f"Loading {self.model_size.value} model...")
                )
            except Exception:
                pass  # Progress notification failed, continue with loading

            # Attempt to load model with comprehensive error handling and fallbacks
            model = None
            last_error = None
            attempted_fallbacks = []

            # Define fallback strategies in order of preference
            fallback_strategies = [
                (self.device, self.model_size),  # Original configuration
            ]

            # Add device fallbacks
            if self.device != "cpu":
                fallback_strategies.append(("cpu", self.model_size))  # CPU fallback

            # Add model size fallbacks for larger models
            if self.model_size != WhisperModelSize.BASE:
                fallback_strategies.append(("cpu", WhisperModelSize.BASE))  # Smaller model on CPU
            if self.model_size not in [WhisperModelSize.TINY, WhisperModelSize.BASE]:
                fallback_strategies.append(("cpu", WhisperModelSize.TINY))  # Smallest model as last resort

            for attempt, (device, model_size) in enumerate(fallback_strategies):
                try:
                    strategy_name = f"{model_size.value} on {device}"

                    if attempt == 0:
                        self.logger.info(f"Attempting to load {strategy_name} (primary)")
                    else:
                        self.logger.info(f"Attempting fallback: {strategy_name}")
                        attempted_fallbacks.append(strategy_name)

                        # Update progress for fallback attempts
                        try:
                            asyncio.get_event_loop().call_soon_threadsafe(
                                lambda: self._notify_progress("loading", progress_start + 30 + (attempt * 10),
                                                            f"Trying fallback: {strategy_name}...")
                            )
                        except Exception:
                            pass

                    # Clear any previous model from memory before loading new one
                    if model is not None:
                        del model
                        gc.collect()
                        if torch and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Set a timeout for model loading to prevent hanging
                    start_time = time.time()

                    # Load model with explicit download_root to ensure cache directory is used
                    if cache_dir:
                        model = whisper.load_model(model_size.value, device=device, download_root=str(cache_dir))
                    else:
                        model = whisper.load_model(model_size.value, device=device)

                    load_time = time.time() - start_time

                    # Verify model loaded successfully
                    if model is None:
                        raise RuntimeError("Model loaded but returned None")

                    # Update instance variables if fallback was used
                    if device != self.device:
                        self.logger.info(f"Device changed from {self.device} to {device} due to fallback")
                        self.device = device
                    if model_size != self.model_size:
                        self.logger.info(f"Model size changed from {self.model_size.value} to {model_size.value} due to fallback")
                        self.model_size = model_size

                    self.logger.info(f"Successfully loaded {strategy_name} in {load_time:.2f}s")
                    break

                except Exception as e:
                    last_error = e
                    error_msg = str(e)

                    # Categorize and log specific error types
                    if "CUDA" in error_msg or "cuda" in error_msg.lower():
                        self.logger.warning(f"CUDA error loading {strategy_name}: {error_msg}")
                    elif "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
                        self.logger.warning(f"Memory error loading {strategy_name}: {error_msg}")
                    elif "download" in error_msg.lower() or "connection" in error_msg.lower():
                        self.logger.warning(f"Network/download error loading {strategy_name}: {error_msg}")
                    elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                        self.logger.warning(f"Permission error loading {strategy_name}: {error_msg}")
                    elif "NoneType" in error_msg and "write" in error_msg:
                        self.logger.warning(f"Cache directory error loading {strategy_name}: {error_msg} (cache: {cache_dir})")
                    else:
                        self.logger.warning(f"Unknown error loading {strategy_name}: {error_msg}")

                    # Clean up after failed attempt
                    gc.collect()
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # If this isn't the last strategy, continue to next fallback
                    if attempt < len(fallback_strategies) - 1:
                        continue

            if model is None:
                # Generate comprehensive error message based on attempted fallbacks
                fallback_info = f" (tried fallbacks: {', '.join(attempted_fallbacks)})" if attempted_fallbacks else ""

                # Provide specific suggestions based on the last error
                if last_error:
                    error_str = str(last_error).lower()
                    if "cuda" in error_str or "gpu" in error_str:
                        suggested_action = "Install proper CUDA drivers, check GPU compatibility, or use CPU-only mode (force_cpu=True)"
                    elif "memory" in error_str or "out of memory" in error_str:
                        suggested_action = "Free up system memory, close other applications, or use a smaller model (tiny/base)"
                    elif "download" in error_str or "connection" in error_str:
                        suggested_action = "Check internet connection, verify firewall settings, or manually download model files"
                    elif "permission" in error_str or "access" in error_str:
                        suggested_action = "Check file permissions for model cache directory, run as administrator if needed"
                    else:
                        suggested_action = "Try using CPU mode (force_cpu=True), use a smaller model, or check system requirements"
                else:
                    suggested_action = "Try using CPU mode (force_cpu=True), use a smaller model, or check system requirements"

                raise MyVoiceError(
                    severity=ErrorSeverity.CRITICAL,
                    code="WHISPER_MODEL_LOAD_FAILED",
                    user_message=f"Failed to load any Whisper transcription model{fallback_info}",
                    technical_details=f"Last error: {last_error}. Attempted strategies: {[f'{s[1].value} on {s[0]}' for s in fallback_strategies]}",
                    suggested_action=suggested_action
                )

            # Final progress update
            try:
                asyncio.get_event_loop().call_soon_threadsafe(
                    lambda: self._notify_progress("loading", 90.0, "Model loaded successfully")
                )
            except Exception:
                pass

            return model

        except MyVoiceError:
            raise
        except Exception as e:
            # Handle unexpected errors during model loading
            raise MyVoiceError(
                severity=ErrorSeverity.CRITICAL,
                code="WHISPER_MODEL_LOAD_FAILED",
                user_message=f"Unexpected error loading {self.model_size.value} transcription model",
                technical_details=str(e),
                suggested_action="Try using CPU mode (force_cpu=True), use a smaller model, or check system requirements"
            )

    async def stop_service(self) -> None:
        """Stop the Whisper service and cleanup resources."""
        try:
            self.status = ServiceStatus.STOPPING
            self.logger.info("Stopping Whisper service")

            # Shutdown thread pool executor
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None

            # Clear model from memory
            self._model = None

            # Force garbage collection to free GPU memory
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.status = ServiceStatus.STOPPED
            self.logger.info("Whisper service stopped successfully")

        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.logger.error(f"Error stopping Whisper service: {e}")
            raise

    def validate_audio_format(self, file_path: Path) -> ValidationResult:
        """
        Validate audio file format and properties with comprehensive checks.

        Args:
            file_path: Path to the audio file

        Returns:
            ValidationResult with validation status and details
        """
        import os

        issues = []
        warnings = []

        try:
            # Check if file exists
            if not file_path.exists():
                issues.append(ValidationIssue(
                    field="file_path",
                    message=f"Audio file not found: {file_path}",
                    code="FILE_NOT_FOUND",
                    severity=ValidationStatus.INVALID
                ))
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=issues,
                    warnings=[],
                    summary="Audio file not found"
                )

            # Check if it's actually a file (not a directory)
            if not file_path.is_file():
                issues.append(ValidationIssue(
                    field="file_path",
                    message=f"Path is not a file: {file_path}",
                    code="NOT_A_FILE",
                    severity=ValidationStatus.INVALID
                ))
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=issues,
                    warnings=[],
                    summary="Path is not a valid file"
                )

            # Check file permissions
            if not os.access(file_path, os.R_OK):
                issues.append(ValidationIssue(field="file_access", message=f"No read permission for file: {file_path}", code="FILE_ACCESS", severity=ValidationStatus.INVALID
                ))

            # Check file extension
            file_extension = file_path.suffix.lower()
            if not file_extension:
                issues.append(ValidationIssue(field="file_format", message="File has no extension", code="FILE_FORMAT", severity=ValidationStatus.INVALID
                ))
            elif file_extension not in self.supported_formats:
                # Provide specific guidance for common unsupported formats
                if file_extension in ['.mp4', '.avi', '.mkv', '.mov']:
                    suggestion = "This appears to be a video file. Extract audio first or use audio-only formats."
                elif file_extension in ['.txt', '.doc', '.pdf']:
                    suggestion = "This appears to be a text/document file. Audio files are required for transcription."
                elif file_extension in ['.aac', '.ogg', '.wma']:
                    suggestion = "This format may work but is not officially supported. Convert to WAV, MP3, or FLAC for best results."
                else:
                    suggestion = f"Use supported formats: {', '.join(self.supported_formats)}"

                issues.append(ValidationIssue(field="file_format", message=f"Unsupported audio format: {file_extension}. {suggestion}", code="FILE_FORMAT", severity=ValidationStatus.INVALID
                ))

            # Check file size
            try:
                file_stats = file_path.stat()
                file_size_bytes = file_stats.st_size
                file_size_mb = file_size_bytes / (1024 * 1024)

                if file_size_bytes == 0:
                    issues.append(ValidationIssue(field="file_size", message="Audio file is empty (0 bytes)", code="FILE_SIZE", severity=ValidationStatus.INVALID
                    ))
                elif file_size_mb > self.max_file_size_mb:
                    issues.append(ValidationIssue(field="file_size", message=f"File too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)", code="FILE_SIZE", severity=ValidationStatus.INVALID
                    ))
                elif file_size_mb > (self.max_file_size_mb * 0.8):  # Warn when approaching limit
                    warnings.append(ValidationIssue(field="file_size", message=f"Large file: {file_size_mb:.1f}MB (close to {self.max_file_size_mb}MB limit)", code="FILE_SIZE", severity=ValidationStatus.WARNING
                    ))

                # Warn about very small files
                if file_size_mb < 0.01:  # Less than 10KB
                    warnings.append(ValidationIssue(field="file_size", message=f"Very small file: {file_size_mb*1024:.1f}KB. May not contain enough audio for transcription.", code="FILE_SIZE", severity=ValidationStatus.WARNING
                    ))

            except OSError as e:
                issues.append(ValidationIssue(field="file_access", message=f"Cannot access file information: {e}", code="FILE_ACCESS", severity=ValidationStatus.INVALID
                ))

            # Check if file is readable and examine header
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(1024)  # Read first 1KB

                    if len(header) == 0:
                        issues.append(ValidationIssue(field="file_content", message="File appears to be empty", code="FILE_CONTENT", severity=ValidationStatus.INVALID
                        ))
                    else:
                        # Basic format validation based on file headers
                        if file_extension == '.wav':
                            if not header.startswith(b'RIFF') or b'WAVE' not in header[:12]:
                                warnings.append(ValidationIssue(field="file_format", message="File has .wav extension but may not be a valid WAV file", code="FILE_FORMAT", severity=ValidationStatus.WARNING
                                ))
                        elif file_extension == '.mp3':
                            if not (header.startswith(b'ID3') or header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3')):
                                warnings.append(ValidationIssue(field="file_format", message="File has .mp3 extension but may not be a valid MP3 file", code="FILE_FORMAT", severity=ValidationStatus.WARNING
                                ))
                        elif file_extension == '.flac':
                            if not header.startswith(b'fLaC'):
                                warnings.append(ValidationIssue(field="file_format", message="File has .flac extension but may not be a valid FLAC file", code="FILE_FORMAT", severity=ValidationStatus.WARNING
                                ))

            except PermissionError as e:
                issues.append(ValidationIssue(field="file_access", message=f"Permission denied reading audio file: {e}", code="FILE_ACCESS", severity=ValidationStatus.INVALID
                ))
            except OSError as e:
                issues.append(ValidationIssue(field="file_access", message=f"Cannot read audio file: {e}", code="FILE_ACCESS", severity=ValidationStatus.INVALID
                ))

            # Determine overall validation status
            if issues:
                return ValidationResult(is_valid=False, status=ValidationStatus.INVALID, issues=issues + warnings, warnings=[], summary="Audio file validation failed")
            elif warnings:
                return ValidationResult(is_valid=True, status=ValidationStatus.VALID, issues=warnings, warnings=[], summary="Audio file is valid with warnings")
            else:
                return ValidationResult(is_valid=True, status=ValidationStatus.VALID, issues=[], warnings=[], summary="Audio file is valid")

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                issues=[ValidationIssue(field="validation", message=f"Validation error: {e}", code="VALIDATION", severity=ValidationStatus.INVALID)],
                warnings=[],
                summary="Failed to validate audio file"
            )

    @require_service_health
    async def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        word_timestamps: bool = False,
        temperature: float = 0.0,
        timeout_seconds: Optional[float] = None
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file with comprehensive error handling and timeout support.

        Args:
            file_path: Path to the audio file
            language: Language code (e.g., 'en', 'es') or None for auto-detection
            word_timestamps: Whether to include word-level timestamps
            temperature: Temperature for decoding (0.0 for deterministic)
            timeout_seconds: Maximum time to wait for transcription (None for no timeout)

        Returns:
            TranscriptionResult with text and metadata

        Raises:
            MyVoiceError: If transcription fails
        """
        file_path = Path(file_path)

        # Enhanced validation with more detailed error handling
        validation_result = self.validate_audio_format(file_path)
        if validation_result.status != ValidationStatus.VALID:
            # Extract specific error types for better user messaging
            error_types = [issue.field for issue in validation_result.issues]

            if "file_path" in error_types:
                suggested_action = "Check that the file exists and the path is correct"
            elif "file_format" in error_types:
                suggested_action = f"Use a supported audio format: {', '.join(self.supported_formats)}"
            elif "file_size" in error_types:
                suggested_action = f"Use an audio file smaller than {self.max_file_size_mb}MB or increase the size limit"
            elif "file_access" in error_types:
                suggested_action = "Check file permissions and ensure the file is not locked by another application"
            else:
                suggested_action = "Please select a valid audio file"

            raise MyVoiceError(
                severity=ValidationStatus.INVALID,
                code="INVALID_AUDIO_FILE",
                user_message="Invalid audio file",
                technical_details=str(validation_result.issues),
                suggested_action=suggested_action
            )

        # Calculate timeout based on file size if not provided
        if timeout_seconds is None:
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                # Estimate: ~5 seconds per MB for large files, minimum 30 seconds
                timeout_seconds = max(30.0, file_size_mb * 5.0)
                # Cap at 10 minutes for very large files
                timeout_seconds = min(timeout_seconds, 600.0)
            except Exception:
                timeout_seconds = 120.0  # Default 2 minutes

        # Perform transcription in background thread with timeout
        async with self._transcription_semaphore:
            try:
                self.logger.info(f"Starting transcription of {file_path} (timeout: {timeout_seconds}s)")

                loop = asyncio.get_event_loop()

                # Use asyncio.wait_for to implement timeout
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        self._transcribe_sync,
                        str(file_path),
                        language,
                        word_timestamps,
                        temperature
                    ),
                    timeout=timeout_seconds
                )

                self.logger.info(f"Transcription completed for {file_path}")
                return result

            except asyncio.TimeoutError:
                error = MyVoiceError(
                    severity=ValidationStatus.INVALID,
                    code="TRANSCRIPTION_TIMEOUT",
                    user_message=f"Transcription timed out after {timeout_seconds:.0f} seconds",
                    technical_details=f"File: {file_path}, Size: {file_path.stat().st_size / (1024 * 1024):.1f}MB",
                    suggested_action="Try using a smaller audio file, increase timeout, or use a smaller Whisper model"
                )
                self.logger.error(f"Transcription timeout for {file_path} after {timeout_seconds}s")
                raise error

            except Exception as e:
                # Enhanced error categorization for better user guidance
                error_str = str(e).lower()

                if "cuda" in error_str or "gpu" in error_str:
                    code = "TRANSCRIPTION_GPU_ERROR"
                    user_message = "GPU error during transcription"
                    suggested_action = "Try using CPU mode (force_cpu=True) or check GPU drivers"
                elif "memory" in error_str or "out of memory" in error_str:
                    code = "TRANSCRIPTION_MEMORY_ERROR"
                    user_message = "Not enough memory for transcription"
                    suggested_action = "Close other applications, use a smaller model, or process smaller audio files"
                elif "corrupt" in error_str or "invalid" in error_str or "format" in error_str:
                    code = "TRANSCRIPTION_AUDIO_ERROR"
                    user_message = "Audio file appears to be corrupted or in an unsupported format"
                    suggested_action = "Try converting the audio file to WAV format or use a different audio file"
                elif "permission" in error_str or "access" in error_str:
                    code = "TRANSCRIPTION_ACCESS_ERROR"
                    user_message = "Cannot access audio file for transcription"
                    suggested_action = "Check file permissions and ensure the file is not locked by another application"
                else:
                    code = "TRANSCRIPTION_FAILED"
                    user_message = "Failed to transcribe audio file"
                    suggested_action = "Check if the audio file is valid, try a different file, or check system resources"

                error = MyVoiceError(
                    severity=ValidationStatus.INVALID,
                    code=code,
                    user_message=user_message,
                    technical_details=str(e),
                    suggested_action=suggested_action
                )
                self.logger.error(f"Transcription failed for {file_path}: {e}")
                raise error

    def _transcribe_sync(
        self,
        file_path: str,
        language: Optional[str],
        word_timestamps: bool,
        temperature: float
    ) -> TranscriptionResult:
        """Synchronous transcription method with enhanced error handling (runs in thread pool)."""
        import time
        import os

        if not self._model:
            raise RuntimeError("Whisper model not loaded")

        start_time = time.time()
        file_path_obj = Path(file_path)

        try:
            # Additional pre-transcription checks
            if not file_path_obj.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"Cannot read audio file: {file_path}")

            # Log file details for debugging
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            self.logger.debug(f"Transcribing {file_path} ({file_size_mb:.1f}MB)")

            # Attempt transcription with enhanced error handling
            try:
                result = self._model.transcribe(
                    file_path,
                    language=language,
                    word_timestamps=word_timestamps,
                    temperature=temperature
                )
            except Exception as e:
                error_str = str(e).lower()

                # Handle specific Whisper/FFmpeg errors
                if "ffmpeg" in error_str:
                    raise RuntimeError(f"Audio processing error: {e}. FFmpeg may not be installed or audio format may be unsupported.")
                elif "cuda" in error_str:
                    raise RuntimeError(f"CUDA error during transcription: {e}")
                elif "memory" in error_str or "out of memory" in error_str:
                    raise RuntimeError(f"Insufficient memory for transcription: {e}")
                elif "invalid" in error_str or "corrupt" in error_str:
                    raise RuntimeError(f"Audio file appears to be corrupted: {e}")
                else:
                    raise RuntimeError(f"Transcription failed: {e}")

            # Validate transcription result
            if not result or not isinstance(result, dict):
                raise RuntimeError("Whisper returned invalid result")

            if "text" not in result:
                raise RuntimeError("Whisper result missing text field")

            transcription_text = result.get("text", "").strip()

            # Check for common Whisper failure patterns
            if not transcription_text:
                self.logger.warning(f"Empty transcription result for {file_path}")
            elif len(transcription_text) < 3:
                self.logger.warning(f"Very short transcription result for {file_path}: '{transcription_text}'")

            # Calculate confidence score if available
            confidence = None
            if "segments" in result and result["segments"]:
                # Calculate average confidence from segments if available
                segment_confidences = []
                for segment in result["segments"]:
                    if "confidence" in segment and segment["confidence"] is not None:
                        segment_confidences.append(segment["confidence"])
                    # Alternative: use avg_logprob to estimate confidence
                    elif "avg_logprob" in segment and segment["avg_logprob"] is not None:
                        # Convert log probability to rough confidence estimate
                        logprob = segment["avg_logprob"]
                        estimated_confidence = max(0.0, min(1.0, (logprob + 1.0) / 1.0))
                        segment_confidences.append(estimated_confidence)

                if segment_confidences:
                    confidence = sum(segment_confidences) / len(segment_confidences)

            processing_time = time.time() - start_time
            self.logger.debug(f"Transcription completed in {processing_time:.2f}s")

            return TranscriptionResult(
                text=transcription_text,
                language=result.get("language", "unknown"),
                segments=result.get("segments", []),
                word_timestamps=result.get("words") if word_timestamps else None,
                confidence=confidence
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Transcription failed after {processing_time:.2f}s: {e}")
            raise

    @require_service_health
    async def batch_transcribe(
        self,
        file_paths: List[Union[str, Path]],
        language: Optional[str] = None,
        word_timestamps: bool = False
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files in batch.

        Args:
            file_paths: List of paths to audio files
            language: Language code or None for auto-detection
            word_timestamps: Whether to include word-level timestamps

        Returns:
            List of TranscriptionResult objects

        Raises:
            MyVoiceError: If any transcription fails
        """
        if not file_paths:
            return []

        self.logger.info(f"Starting batch transcription of {len(file_paths)} files")

        # Create transcription tasks
        tasks = []
        for file_path in file_paths:
            task = self.transcribe_file(
                file_path=file_path,
                language=language,
                word_timestamps=word_timestamps
            )
            tasks.append(task)

        try:
            # Execute all transcriptions concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions
            transcription_results = []
            errors = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"File {file_paths[i]}: {result}")
                else:
                    transcription_results.append(result)

            if errors:
                raise MyVoiceError(
                    severity=ValidationStatus.INVALID,
                    code="BATCH_TRANSCRIPTION_FAILED",
                    user_message=f"Failed to transcribe {len(errors)} files",
                    technical_details="; ".join(errors),
                    suggested_action="Check failed files and try again"
                )

            self.logger.info(f"Batch transcription completed successfully")
            return transcription_results

        except Exception as e:
            if isinstance(e, MyVoiceError):
                raise

            error = MyVoiceError(
                severity=ValidationStatus.INVALID,
                code="BATCH_TRANSCRIPTION_ERROR",
                user_message="Batch transcription failed",
                technical_details=str(e),
                suggested_action="Try transcribing files individually"
            )
            self.logger.error(f"Batch transcription error: {e}")
            raise error

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded Whisper model.

        Returns:
            Dictionary with model information
        """
        base_info = {
            "model_size": self.model_size.value,
            "device": self.device,
            "force_cpu": self.force_cpu,
            "supported_formats": self.supported_formats,
            "max_file_size_mb": self.max_file_size_mb,
            "max_concurrent_transcriptions": self.max_concurrent_transcriptions,
            "estimated_size_mb": self.MODEL_SIZES.get(self.model_size, 0.0)
        }

        if not self._model:
            base_info["status"] = "not_loaded"
            return base_info

        base_info["status"] = "loaded"

        if self._model_info:
            base_info.update({
                "requires_download": self._model_info.requires_download,
                "device_compatible": self._model_info.device_compatible
            })

        return base_info

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get information about all available Whisper models.

        Returns:
            List of dictionaries with model information
        """
        models = []
        for model_size in WhisperModelSize:
            model_info = {
                "name": model_size.value,
                "size": model_size.value,
                "estimated_size_mb": self.MODEL_SIZES.get(model_size, 0.0),
                "is_current": model_size == self.model_size
            }

            # Check if model is cached locally
            cache_path = self._get_model_cache_path()
            if cache_path and cache_path.exists():
                model_file = cache_path / f"{model_size.value}.pt"
                model_info["is_cached"] = model_file.exists()
            else:
                model_info["is_cached"] = False

            models.append(model_info)

        return models

    def set_progress_callback(self, callback: Optional[Callable[[ModelLoadProgress], None]]) -> None:
        """
        Set or update the progress callback for model loading.

        Args:
            callback: Function to call with progress updates, or None to disable
        """
        self.progress_callback = callback

    async def validate_model_compatibility(self, model_size: Optional[WhisperModelSize] = None) -> Dict[str, Any]:
        """
        Validate if a model size is compatible with the current system.

        Args:
            model_size: Model size to validate, or None to validate current model

        Returns:
            Dictionary with validation results
        """
        check_model = model_size or self.model_size
        original_model = self.model_size

        try:
            # Temporarily set model size for validation
            self.model_size = check_model
            model_info = self._check_model_availability()

            validation_results = {
                "model_size": check_model.value,
                "estimated_size_mb": model_info.estimated_size_mb,
                "requires_download": model_info.requires_download,
                "device_compatible": model_info.device_compatible,
                "issues": []
            }

            # Check system resources without raising exceptions
            try:
                self._validate_system_resources(model_info)
                validation_results["disk_space_sufficient"] = True
            except MyVoiceError as e:
                validation_results["disk_space_sufficient"] = False
                validation_results["issues"].append({
                    "type": "disk_space",
                    "message": e.user_message,
                    "details": e.technical_details
                })

            # Check device compatibility
            if not model_info.device_compatible and self.device == "cuda":
                validation_results["issues"].append({
                    "type": "device_compatibility",
                    "message": "CUDA not available, will use CPU",
                    "details": "Model will run on CPU instead of GPU"
                })

            validation_results["is_compatible"] = len(validation_results["issues"]) == 0

            return validation_results

        finally:
            # Restore original model size
            self.model_size = original_model

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the service.

        Returns:
            Dictionary with health status information
        """
        try:
            health_info = {
                "status": self.status.value,
                "model_loaded": self._model is not None,
                "device": self.device,
                "force_cpu": self.force_cpu,
                "model_size": self.model_size.value,
                "estimated_size_mb": self.MODEL_SIZES.get(self.model_size, 0.0)
            }

            # Add model information if available
            if self._model_info:
                health_info.update({
                    "model_cached": not self._model_info.requires_download,
                    "device_compatible": self._model_info.device_compatible
                })

            # Check CUDA availability if relevant
            if torch:
                health_info["cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    health_info["cuda_device_count"] = torch.cuda.device_count()

            # Check cache directory
            cache_path = self._get_model_cache_path()
            if cache_path:
                health_info["cache_directory"] = str(cache_path)
                health_info["cache_directory_exists"] = cache_path.exists()

            # Basic transcription capability test
            if self._model and self.status == ServiceStatus.RUNNING:
                # For now, just mark as ready for transcription
                # In the future, could test with a small audio sample
                health_info["transcription_ready"] = True
            else:
                health_info["transcription_ready"] = False

            return health_info

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_loaded": False,
                "transcription_ready": False
            }

    # BaseService abstract method implementations
    async def start(self) -> bool:
        """Start the service (BaseService interface)."""
        try:
            await self.start_service()
            return self.status == ServiceStatus.RUNNING
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the service (BaseService interface)."""
        try:
            await self.stop_service()
            return self.status == ServiceStatus.STOPPED
        except Exception as e:
            self.logger.error(f"Failed to stop service: {e}")
            return False

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """
        Perform a health check (BaseService interface).

        Returns:
            tuple: (is_healthy, error_if_any)
        """
        try:
            health_info = await self.get_health_info()
            is_healthy = (
                health_info.get("status") == "running" and
                health_info.get("model_loaded", False) and
                health_info.get("transcription_ready", False)
            )
            return is_healthy, None

        except Exception as e:
            error = MyVoiceError(
                severity=ValidationStatus.INVALID,
                code="HEALTH_CHECK_FAILED",
                user_message="Health check failed",
                technical_details=str(e),
                suggested_action="Check service status and logs"
            )
            return False, error

    async def get_health_info(self) -> Dict[str, Any]:
        """
        Get detailed health information (separate from BaseService interface).

        Returns:
            Dictionary with detailed health status information
        """
        try:
            health_info = {
                "status": self.status.value,
                "model_loaded": self._model is not None,
                "device": self.device,
                "force_cpu": self.force_cpu,
                "model_size": self.model_size.value,
                "estimated_size_mb": self.MODEL_SIZES.get(self.model_size, 0.0)
            }

            # Add model information if available
            if self._model_info:
                health_info.update({
                    "model_cached": not self._model_info.requires_download,
                    "device_compatible": self._model_info.device_compatible
                })

            # Check CUDA availability if relevant
            if torch:
                health_info["cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    health_info["cuda_device_count"] = torch.cuda.device_count()

            # Check cache directory
            cache_path = self._get_model_cache_path()
            if cache_path:
                health_info["cache_directory"] = str(cache_path)
                health_info["cache_directory_exists"] = cache_path.exists()

            # Basic transcription capability test
            if self._model and self.status == ServiceStatus.RUNNING:
                # For now, just mark as ready for transcription
                # In the future, could test with a small audio sample
                health_info["transcription_ready"] = True
            else:
                health_info["transcription_ready"] = False

            return health_info

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_loaded": False,
                "transcription_ready": False
            }