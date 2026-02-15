"""
VoiceProfileManager Service

This module implements the VoiceProfileManager service for managing voice cloning
profiles, directory scanning, validation, transcription, and profile activation.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from concurrent.futures import ThreadPoolExecutor

from myvoice.models.voice_profile import VoiceProfile, VoiceType, BUNDLED_SPEAKERS
from myvoice.models.transcription_result import TranscriptionResult
from myvoice.models.validation import ValidationResult, ValidationIssue, ValidationStatus, UserFriendlyMessage
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.service_enums import QwenModelType

# Import base service after other models to avoid circular imports
from myvoice.services.core.base_service import BaseService, ServiceStatus


logger = logging.getLogger(__name__)


class VoiceProfileManager(BaseService):
    """
    Service for managing voice profiles with directory scanning, validation, and transcription.

    This service provides centralized management of voice cloning profiles including:
    - Directory scanning for voice files
    - Voice file validation
    - Profile creation and management
    - Transcription services (placeholder for future Whisper integration)
    - Active profile management

    Features:
    - Async directory scanning with progress tracking
    - Comprehensive validation with duration constraints (≤10 seconds)
    - Profile caching and persistence
    - Thread-safe operations
    - Integration with existing error handling patterns
    """

    def __init__(
        self,
        voice_directory: Optional[Path] = None,
        cache_file: Optional[Path] = None,
        max_duration: float = 300.0,  # 5 minutes - Qwen3-TTS handles longer files
        max_workers: int = 4,
        auto_scan: bool = True,
        transcription_queue_service: Optional[Any] = None,
        enable_auto_transcription: bool = True
    ):
        self._cache_save_task: Optional[asyncio.Task] = None
        """
        Initialize the VoiceProfileManager.

        Args:
            voice_directory: Directory to scan for voice files (default: voice_files/)
            cache_file: File to cache profile data (default: config/voice_cache.json)
            max_duration: Maximum allowed duration in seconds (default: 10.0)
            max_workers: Maximum thread pool workers for file operations
            auto_scan: Whether to automatically scan directory on startup
            transcription_queue_service: TranscriptionQueueService instance for background transcription
            enable_auto_transcription: Whether to automatically queue transcriptions for new profiles
        """
        super().__init__("VoiceProfileManager")

        # Configuration
        self.voice_directory = voice_directory or Path("voice_files")
        self.cache_file = cache_file or Path("config/.voice_cache.json")  # Use hidden file as per task
        self.max_duration = max_duration
        self.max_workers = max_workers
        self.auto_scan = auto_scan
        self.transcription_queue_service = transcription_queue_service
        self.enable_auto_transcription = enable_auto_transcription

        # Service state
        self._profiles: Dict[str, VoiceProfile] = {}
        self._active_profile: Optional[VoiceProfile] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._scan_lock = asyncio.Lock()

        # Metrics
        self._total_scanned = 0
        self._valid_profiles = 0
        self._last_scan_time: Optional[float] = None

        self.logger.debug(f"VoiceProfileManager initialized with directory: {self.voice_directory}")
        self.logger.debug(f"Max duration: {self.max_duration}s, Cache: {self.cache_file}")

    async def start(self) -> bool:
        """
        Start the VoiceProfileManager service.

        Returns:
            bool: True if service started successfully, False otherwise
        """
        try:
            await self._update_status(ServiceStatus.STARTING)
            self.logger.info("Starting VoiceProfileManager service")

            # Initialize thread executor
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="VoiceProfile"
            )

            # Create directories with comprehensive error handling
            await self._ensure_directories_exist()

            # Load cached profiles
            await self._load_profile_cache()

            # Initialize bundled voices (9 pre-trained timbres)
            self._initialize_bundled_voices()

            # Scan embeddings directory for saved embedding voices (Story 1.6)
            self._scan_embeddings_directory()

            # Auto-scan if enabled
            if self.auto_scan:
                self.logger.info("Performing initial directory scan")
                await self.scan_voice_directory()

            await self._update_status(ServiceStatus.RUNNING)
            self.logger.info("VoiceProfileManager service started successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to start VoiceProfileManager service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def stop(self) -> bool:
        """
        Stop the VoiceProfileManager service.

        Returns:
            bool: True if service stopped successfully, False otherwise
        """
        try:
            await self._update_status(ServiceStatus.STOPPING)
            self.logger.info("Stopping VoiceProfileManager service")

            # Wait for any pending cache save task to complete
            if self._cache_save_task and not self._cache_save_task.done():
                self.logger.debug("Waiting for pending cache save task to complete")
                try:
                    await asyncio.wait_for(self._cache_save_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Cache save task timed out, cancelling")
                    self._cache_save_task.cancel()
                    try:
                        await self._cache_save_task
                    except asyncio.CancelledError:
                        pass

            # Save profile cache
            await self._save_profile_cache()

            # Cleanup resources - QA Round 2 Item #8: Non-blocking shutdown
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

            await self._update_status(ServiceStatus.STOPPED)
            self.logger.info("VoiceProfileManager service stopped successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error stopping VoiceProfileManager service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """
        Check VoiceProfileManager service health.

        Returns:
            tuple[bool, Optional[MyVoiceError]]: (is_healthy, error_if_any)
        """
        try:
            # Check service status
            if not self.is_running():
                return False, MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="SERVICE_NOT_RUNNING",
                    user_message="VoiceProfileManager service is not running",
                    suggested_action="Start the VoiceProfileManager service"
                )

            # Check voice directory accessibility
            if not self.voice_directory.exists():
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="VOICE_DIRECTORY_MISSING",
                    user_message=f"Voice directory not found: {self.voice_directory}",
                    suggested_action="Create the voice directory or scan a different location"
                )

            if not self.voice_directory.is_dir():
                return False, MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="VOICE_DIRECTORY_INVALID",
                    user_message=f"Voice directory path is not a directory: {self.voice_directory}",
                    suggested_action="Check the voice directory path configuration"
                )

            # Check cache file accessibility
            if self.cache_file.exists() and not self.cache_file.is_file():
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="CACHE_FILE_INVALID",
                    user_message=f"Cache file path is not a file: {self.cache_file}",
                    suggested_action="Remove the invalid cache path"
                )

            return True, None

        except Exception as e:
            self.logger.exception(f"Health check failed: {e}")
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="HEALTH_CHECK_FAILED",
                user_message="Failed to check VoiceProfileManager service health",
                technical_details=str(e)
            )

    async def scan_voice_directory(self, directory: Optional[Path] = None, incremental: bool = True) -> Dict[str, Any]:
        """
        Scan directory for voice files and create profiles.

        Args:
            directory: Directory to scan (defaults to configured voice_directory)
            incremental: If True, only scan new/modified files (default: True)

        Returns:
            Dict[str, Any]: Scan results with statistics and found profiles
        """
        scan_dir = directory or self.voice_directory

        async with self._scan_lock:
            try:
                self.logger.info(f"Scanning voice directory: {scan_dir} (incremental={incremental})")
                import time
                start_time = time.time()

                # Initialize metrics
                self._total_scanned = 0
                valid_count = 0
                invalid_count = 0
                found_profiles = []
                cached_count = 0

                # Find WAV files
                if not scan_dir.exists():
                    self.logger.warning(f"Directory does not exist: {scan_dir}")
                    return {
                        "success": False,
                        "error": f"Directory not found: {scan_dir}",
                        "total_scanned": 0,
                        "valid_profiles": 0,
                        "invalid_profiles": 0,
                        "profiles": []
                    }

                # Get all WAV files, excluding internal directories
                # - design_sessions/: temporary session files from Voice Design Studio
                # - embeddings/: source audio for embedding voices (handled separately)
                excluded_dirs = {'design_sessions', 'embeddings'}
                all_wav_files = [
                    f for f in scan_dir.rglob("*.wav")
                    if not any(excluded in f.relative_to(scan_dir).parts for excluded in excluded_dirs)
                ]
                self._total_scanned = len(all_wav_files)
                self.logger.debug(f"Found {self._total_scanned} WAV files (excluding internal directories)")

                # If incremental scanning, filter out files that are already cached and unchanged
                files_to_process = []
                if incremental:
                    for wav_file in all_wav_files:
                        # Check if file is in cache and unchanged
                        profile_name = wav_file.stem
                        if profile_name in self._profiles:
                            cached_profile = self._profiles[profile_name]
                            try:
                                # Check if file has been modified
                                current_mtime = wav_file.stat().st_mtime
                                current_size = wav_file.stat().st_size

                                # If we have cached mtime data, compare it
                                if hasattr(cached_profile, '_cached_mtime'):
                                    if (abs(current_mtime - cached_profile._cached_mtime) <= 1.0 and
                                        current_size == getattr(cached_profile, '_cached_size', 0)):
                                        # File unchanged, use cached version
                                        cached_count += 1
                                        if cached_profile.is_valid:
                                            found_profiles.append({
                                                "name": cached_profile.name,
                                                "file_path": str(cached_profile.file_path),
                                                "duration": cached_profile.duration,
                                                "is_valid": cached_profile.is_valid
                                            })
                                            valid_count += 1
                                        else:
                                            invalid_count += 1
                                        continue
                            except (OSError, AttributeError):
                                # File access error or missing cache metadata, need to re-process
                                pass

                        # File needs processing
                        files_to_process.append(wav_file)
                else:
                    # Full scan - process all files
                    files_to_process = all_wav_files
                    self._profiles.clear()  # Clear existing profiles for full rescan

                self.logger.info(f"Found {len(all_wav_files)} WAV files, processing {len(files_to_process)} (cached: {cached_count})")

                # Process files in parallel using thread pool
                if files_to_process:
                    loop = asyncio.get_running_loop()
                    tasks = []

                    for wav_file in files_to_process:
                        task = loop.run_in_executor(
                            self._executor,
                            self._process_voice_file_sync,
                            wav_file
                        )
                        tasks.append(task)

                    # Wait for all processing to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Error processing {files_to_process[i]}: {result}")
                            invalid_count += 1
                        elif isinstance(result, VoiceProfile):
                            # Add cache metadata
                            try:
                                file_stat = result.file_path.stat()
                                result._cached_mtime = file_stat.st_mtime
                                result._cached_size = file_stat.st_size
                            except OSError:
                                pass

                            if result.is_valid:
                                self._profiles[result.name] = result
                                found_profiles.append({
                                    "name": result.name,
                                    "file_path": str(result.file_path),
                                    "duration": result.duration,
                                    "is_valid": result.is_valid
                                })
                                valid_count += 1
                            else:
                                invalid_count += 1
                                self.logger.debug(f"Invalid profile: {result.name}")

                # Update metrics
                self._valid_profiles = len([p for p in self._profiles.values() if p.is_valid])
                self._last_scan_time = time.time()
                scan_duration = self._last_scan_time - start_time

                # Re-initialize bundled voices (they were cleared during full rescan)
                # This ensures the 9 timbre voices are always available
                if not incremental:
                    bundled_count = self._initialize_bundled_voices()
                    self.logger.info(f"Re-initialized {bundled_count} bundled voices after full rescan")
                    valid_count += bundled_count

                # Scan embeddings directory for saved embedding voices (Story 1.6)
                # This discovers voices created in Voice Design Studio
                embedding_count = self._scan_embeddings_directory()
                if embedding_count > 0:
                    self.logger.info(f"Found {embedding_count} embedding voices")
                    valid_count += embedding_count

                # Save updated cache
                await self._save_profile_cache()

                scan_results = {
                    "success": True,
                    "total_scanned": self._total_scanned,
                    "files_processed": len(files_to_process),
                    "cached_files": cached_count,
                    "valid_profiles": valid_count,
                    "invalid_profiles": invalid_count,
                    "profiles": found_profiles,
                    "scan_duration": scan_duration,
                    "scan_directory": str(scan_dir),
                    "incremental": incremental
                }

                self.logger.info(f"Directory scan completed: {valid_count}/{self._total_scanned} valid profiles "
                               f"({len(files_to_process)} processed, {cached_count} cached) in {scan_duration:.2f}s")
                return scan_results

            except Exception as e:
                self.logger.exception(f"Error during directory scan: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "total_scanned": self._total_scanned,
                    "valid_profiles": 0,
                    "invalid_profiles": 0,
                    "profiles": []
                }

    def _process_voice_file_sync(self, file_path: Path) -> VoiceProfile:
        """
        Synchronous voice file processing for thread pool execution.

        Args:
            file_path: Path to the voice file

        Returns:
            VoiceProfile: Created voice profile
        """
        try:
            profile = VoiceProfile.create_profile_from_file(file_path)

            # Apply duration constraint (5 minute max for practical purposes)
            if profile.duration and profile.duration > self.max_duration:
                # Mark as invalid due to duration
                profile.is_valid = False
                self.logger.warning(
                    f"Voice file '{profile.name}' rejected: duration {profile.duration:.1f}s exceeds limit of {self.max_duration:.0f}s."
                )

            return profile

        except Exception as e:
            self.logger.error(f"Error processing voice file {file_path}: {e}")
            # Return invalid profile
            return VoiceProfile(
                file_path=file_path,
                name=file_path.stem,
                is_valid=False
            )

    async def validate_voice_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a voice file for format and duration constraints.

        Args:
            file_path: Path to the voice file to validate

        Returns:
            ValidationResult: Detailed validation result
        """
        try:
            if isinstance(file_path, str):
                file_path = Path(file_path)

            self.logger.debug(f"Validating voice file: {file_path}")

            # Create temporary profile for validation
            loop = asyncio.get_running_loop()
            profile = await loop.run_in_executor(
                self._executor,
                VoiceProfile.create_profile_from_file,
                file_path
            )

            # Get base validation result
            validation = profile.validate()

            # Add duration constraint validation
            if profile.duration and profile.duration > self.max_duration:
                validation.issues.append(ValidationIssue(
                    field="duration",
                    message=f"Voice file duration ({profile.duration:.1f}s) exceeds maximum allowed duration ({self.max_duration}s)",
                    code="DURATION_TOO_LONG",
                    severity=ValidationStatus.INVALID
                ))
                validation.is_valid = False
                validation.status = ValidationStatus.INVALID

            self.logger.debug(f"Validation result for {file_path}: {validation.status.value}")
            return validation

        except FileNotFoundError:
            self.logger.error(f"Voice file not found: {file_path}")
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                issues=[ValidationIssue(
                    field="file_path",
                    message=f"Voice file not found: {file_path}",
                    code="FILE_NOT_FOUND",
                    severity=ValidationStatus.INVALID,
                    user_message=UserFriendlyMessage(
                        title="File Not Found",
                        message=f"The voice file could not be found: {file_path.name}",
                        suggestion="Please check that the file exists and is accessible."
                    )
                )],
                warnings=[],
                summary="Voice file not found"
            )
        except PermissionError:
            self.logger.error(f"Permission denied accessing voice file: {file_path}")
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                issues=[ValidationIssue(
                    field="file_path",
                    message=f"Permission denied accessing voice file: {file_path}",
                    code="PERMISSION_DENIED",
                    severity=ValidationStatus.INVALID,
                    user_message=UserFriendlyMessage(
                        title="Permission Denied",
                        message=f"Cannot access voice file: {file_path.name}",
                        suggestion="Please check file permissions or try running as administrator."
                    )
                )],
                warnings=[],
                summary="Permission denied accessing voice file"
            )
        except Exception as e:
            self.logger.exception(f"Error validating voice file {file_path}: {e}")

            # Provide specific error messages for common issues
            error_message = str(e).lower()
            if "format" in error_message or "codec" in error_message or "audio" in error_message:
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=[ValidationIssue(
                        field="format",
                        message=f"Unsupported audio format: {str(e)}",
                        code="UNSUPPORTED_FORMAT",
                        severity=ValidationStatus.INVALID,
                        user_message=UserFriendlyMessage(
                            title="Unsupported Audio Format",
                            message=f"The audio file format is not supported: {file_path.name}",
                            suggestion="Please convert the file to WAV, MP3, or FLAC format."
                        )
                    )],
                    warnings=[],
                    summary="Unsupported audio format"
                )
            elif "corrupt" in error_message or "invalid" in error_message:
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=[ValidationIssue(
                        field="integrity",
                        message=f"Corrupted audio file: {str(e)}",
                        code="CORRUPTED_FILE",
                        severity=ValidationStatus.INVALID,
                        user_message=UserFriendlyMessage(
                            title="Corrupted Audio File",
                            message=f"The audio file appears to be corrupted: {file_path.name}",
                            suggestion="Please try re-recording the voice sample or use a different file."
                        )
                    )],
                    warnings=[],
                    summary="Corrupted audio file"
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=[ValidationIssue(
                        field="general",
                        message=f"Validation error: {str(e)}",
                        code="VALIDATION_ERROR",
                        severity=ValidationStatus.INVALID,
                        user_message=UserFriendlyMessage(
                            title="Voice File Validation Failed",
                            message=f"Unable to process voice file: {file_path.name}",
                            suggestion="Please ensure the file is a valid audio file and try again."
                        )
                    )],
                    warnings=[],
                    summary="Validation failed due to internal error"
                )

    async def create_profile_from_file(self, file_path: Union[str, Path],
                                     name: Optional[str] = None,
                                     transcription: Optional[str] = None) -> VoiceProfile:
        """
        Create a voice profile from a file with validation.

        Args:
            file_path: Path to the voice file
            name: Optional custom name for the profile
            transcription: Optional transcription text

        Returns:
            VoiceProfile: Created and validated voice profile
        """
        try:
            if isinstance(file_path, str):
                file_path = Path(file_path)

            self.logger.info(f"Creating voice profile from file: {file_path}")

            # Create profile in thread pool
            loop = asyncio.get_running_loop()
            profile = await loop.run_in_executor(
                self._executor,
                VoiceProfile.create_profile_from_file,
                file_path,
                name,
                transcription
            )

            # Apply duration constraint
            if profile.duration and profile.duration > self.max_duration:
                profile.is_valid = False
                self.logger.warning(f"Profile {profile.name} exceeds max duration: {profile.duration:.1f}s")

            # Add to managed profiles if valid
            if profile.is_valid:
                self._profiles[profile.name] = profile
                await self._save_profile_cache()
                self.logger.info(f"Added valid profile: {profile.name}")

                # Queue for automatic transcription if enabled and needed
                if self.enable_auto_transcription and profile.needs_transcription():
                    await self._queue_transcription_if_available(profile)

            else:
                self.logger.warning(f"Created invalid profile: {profile.name}")

            return profile

        except Exception as e:
            self.logger.exception(f"Error creating profile from file {file_path}: {e}")
            # Return invalid profile
            return VoiceProfile(
                file_path=file_path,
                name=name or file_path.stem,
                transcription=transcription,
                is_valid=False
            )

    async def get_transcription(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Get transcription for a voice file.

        Note: This is a placeholder implementation. In the future, this will
        integrate with Whisper or another transcription service.

        Args:
            file_path: Path to the voice file

        Returns:
            Optional[str]: Transcription text if available, None otherwise
        """
        try:
            if isinstance(file_path, str):
                file_path = Path(file_path)

            self.logger.debug(f"Getting transcription for: {file_path}")

            # TODO: Implement Whisper integration
            # For now, check if a matching .txt file exists
            txt_file = file_path.with_suffix('.txt')
            if txt_file.exists():
                transcription = txt_file.read_text(encoding='utf-8').strip()
                self.logger.debug(f"Found transcription file: {txt_file}")
                return transcription

            # Placeholder: Return None for now
            # Future implementation will use Whisper to generate transcription
            self.logger.debug(f"No transcription available for {file_path}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting transcription for {file_path}: {e}")
            return None

    def get_transcription_file_path(self, voice_file_path: Union[str, Path]) -> Path:
        """
        Get the path where transcription should be stored for a voice file.

        Args:
            voice_file_path: Path to the voice file

        Returns:
            Path: Path to the transcription JSON file
        """
        if isinstance(voice_file_path, str):
            voice_file_path = Path(voice_file_path)

        # Store transcription files alongside voice files with .transcription.json extension
        return voice_file_path.with_suffix('.transcription.json')

    async def save_transcription(
        self,
        voice_file_path: Union[str, Path],
        transcription_result: TranscriptionResult
    ) -> None:
        """
        Save transcription result to file with atomic writes.

        Args:
            voice_file_path: Path to the voice file
            transcription_result: TranscriptionResult to save

        Raises:
            MyVoiceError: If save operation fails
        """
        try:
            transcription_path = self.get_transcription_file_path(voice_file_path)

            # Convert transcription result to dictionary
            transcription_data = transcription_result.to_dict()

            # Ensure parent directory exists
            transcription_path.parent.mkdir(parents=True, exist_ok=True)

            # Use atomic write: write to temp file then rename
            temp_path = transcription_path.with_suffix('.tmp')

            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription_data, f, indent=2, ensure_ascii=False)

                # Atomic rename (works on Windows and Unix)
                temp_path.replace(transcription_path)

                self.logger.info(f"Saved transcription to {transcription_path}")

            except Exception as e:
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                raise e

        except Exception as e:
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="TRANSCRIPTION_SAVE_FAILED",
                user_message="Failed to save transcription file",
                technical_details=str(e),
                suggested_action="Check file permissions and disk space"
            )

    async def load_transcription(self, voice_file_path: Union[str, Path]) -> Optional[TranscriptionResult]:
        """
        Load transcription result from file.

        Args:
            voice_file_path: Path to the voice file

        Returns:
            Optional[TranscriptionResult]: Loaded transcription or None if not found

        Raises:
            MyVoiceError: If load operation fails with file corruption
        """
        from myvoice.models.transcription_result import TranscriptionSegment, WordTimestamp, TranscriptionStatus
        from datetime import datetime

        try:
            transcription_path = self.get_transcription_file_path(voice_file_path)

            if not transcription_path.exists():
                return None

            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)

            # Validate required fields are present
            required_fields = ['text', 'confidence', 'duration', 'language']
            for field in required_fields:
                if field not in transcription_data:
                    raise ValueError(f"Missing required field: {field}")

            # Reconstruct TranscriptionResult from saved data
            # Convert segments back to TranscriptionSegment objects if present
            segments = []
            if transcription_data.get('segments'):
                for seg_data in transcription_data['segments']:
                    segment = TranscriptionSegment(
                        id=seg_data['id'],
                        start=seg_data['start'],
                        end=seg_data['end'],
                        text=seg_data['text'],
                        tokens=seg_data.get('tokens'),
                        temperature=seg_data.get('temperature'),
                        avg_logprob=seg_data.get('avg_logprob'),
                        compression_ratio=seg_data.get('compression_ratio'),
                        no_speech_prob=seg_data.get('no_speech_prob')
                    )
                    segments.append(segment)

            # Convert word timestamps back to WordTimestamp objects if present
            word_timestamps = None
            if transcription_data.get('word_timestamps'):
                word_timestamps = []
                for word_data in transcription_data['word_timestamps']:
                    word = WordTimestamp(
                        word=word_data['word'],
                        start=word_data['start'],
                        end=word_data['end'],
                        confidence=word_data.get('confidence')
                    )
                    word_timestamps.append(word)

            # Convert status back to enum
            status = TranscriptionStatus(transcription_data.get('status', 'success'))

            # Parse created_at if present
            created_at = datetime.now()
            if transcription_data.get('created_at'):
                try:
                    created_at = datetime.fromisoformat(transcription_data['created_at'])
                except ValueError:
                    pass  # Use current time if parsing fails

            # Parse source_file if present
            source_file = None
            if transcription_data.get('source_file'):
                source_file = Path(transcription_data['source_file'])

            transcription_result = TranscriptionResult(
                text=transcription_data['text'],
                confidence=transcription_data['confidence'],
                duration=transcription_data['duration'],
                language=transcription_data['language'],
                status=status,
                error_message=transcription_data.get('error_message'),
                segments=segments,
                word_timestamps=word_timestamps,
                model_name=transcription_data.get('model_name'),
                processing_time=transcription_data.get('processing_time'),
                created_at=created_at,
                source_file=source_file,
                metadata=transcription_data.get('metadata', {})
            )

            self.logger.debug(f"Loaded transcription from {transcription_path}")
            return transcription_result

        except FileNotFoundError:
            return None
        except json.JSONDecodeError as e:
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="TRANSCRIPTION_FILE_CORRUPT",
                user_message="Transcription file is corrupted",
                technical_details=f"JSON decode error: {e}",
                suggested_action="Delete the corrupted transcription file and regenerate"
            )
        except Exception as e:
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="TRANSCRIPTION_LOAD_FAILED",
                user_message="Failed to load transcription file",
                technical_details=str(e),
                suggested_action="Check file permissions and try again"
            )

    def validate_transcription_file(self, transcription_path: Path) -> ValidationResult:
        """
        Validate a transcription file for completeness and integrity.

        Args:
            transcription_path: Path to the transcription file

        Returns:
            ValidationResult: Validation status and any issues found
        """
        issues = []

        try:
            # Check if file exists
            if not transcription_path.exists():
                issues.append(ValidationIssue(
                    field="file_path",
                    message=f"Transcription file not found: {transcription_path}",
                    code="FILE_NOT_FOUND",
                    severity=ValidationStatus.INVALID
                ))
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=issues,
                    warnings=[]
                )

            # Check if file is readable
            try:
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    transcription_data = json.load(f)
            except json.JSONDecodeError as e:
                issues.append(ValidationIssue(
                    field="file_format",
                    message=f"Invalid JSON format: {e}",
                    code="INVALID_JSON",
                    severity=ValidationStatus.INVALID
                ))
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=issues,
                    warnings=[]
                )
            except Exception as e:
                issues.append(ValidationIssue(
                    field="file_access",
                    message=f"Cannot read file: {e}",
                    code="FILE_ACCESS_ERROR",
                    severity=ValidationStatus.INVALID
                ))
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=issues,
                    warnings=[]
                )

            # Validate required fields
            required_fields = ['text', 'confidence', 'duration', 'language']
            for field in required_fields:
                if field not in transcription_data:
                    issues.append(ValidationIssue(
                        field=field,
                        message=f"Missing required field: {field}",
                        code="MISSING_FIELD",
                        severity=ValidationStatus.INVALID
                    ))

            # Validate field types and values
            if 'confidence' in transcription_data:
                confidence = transcription_data['confidence']
                if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                    issues.append(ValidationIssue(
                        field="confidence",
                        message=f"Invalid confidence value: {confidence}",
                        code="INVALID_CONFIDENCE",
                        severity=ValidationStatus.INVALID
                    ))

            if 'duration' in transcription_data:
                duration = transcription_data['duration']
                if not isinstance(duration, (int, float)) or duration < 0:
                    issues.append(ValidationIssue(
                        field="duration",
                        message=f"Invalid duration value: {duration}",
                        code="INVALID_DURATION",
                        severity=ValidationStatus.INVALID
                    ))

            if 'text' in transcription_data:
                text = transcription_data['text']
                if not isinstance(text, str):
                    issues.append(ValidationIssue(
                        field="text",
                        message="Text field must be a string",
                        code="INVALID_TEXT_TYPE",
                        severity=ValidationStatus.INVALID
                    ))

            # Determine overall validation status
            if issues:
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.INVALID,
                    issues=issues,
                    warnings=[]
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

    async def has_transcription(self, voice_file_path: Union[str, Path]) -> bool:
        """
        Check if a transcription file exists for the given voice file.

        Args:
            voice_file_path: Path to the voice file

        Returns:
            bool: True if transcription file exists and is valid, False otherwise
        """
        try:
            transcription_path = self.get_transcription_file_path(voice_file_path)

            if not transcription_path.exists():
                return False

            # Validate the transcription file
            validation_result = self.validate_transcription_file(transcription_path)
            return validation_result.is_valid

        except Exception as e:
            self.logger.error(f"Error checking transcription for {voice_file_path}: {e}")
            return False

    async def get_missing_transcriptions(self, voice_files: List[Path]) -> List[Path]:
        """
        Get list of voice files that are missing transcriptions.

        Args:
            voice_files: List of voice file paths to check

        Returns:
            List[Path]: Voice files missing valid transcriptions
        """
        missing_transcriptions = []

        for voice_file in voice_files:
            try:
                if not await self.has_transcription(voice_file):
                    missing_transcriptions.append(voice_file)
            except Exception as e:
                self.logger.error(f"Error checking transcription for {voice_file}: {e}")
                # Assume missing if we can't check
                missing_transcriptions.append(voice_file)

        return missing_transcriptions

    async def _queue_transcription_if_available(self, profile: VoiceProfile) -> bool:
        """
        Queue a profile for transcription if transcription service is available.

        Args:
            profile: VoiceProfile to queue for transcription

        Returns:
            bool: True if successfully queued, False otherwise
        """
        if not self.transcription_queue_service:
            self.logger.debug(f"No transcription queue service available for {profile.name}")
            return False

        try:
            # Import here to avoid circular imports
            from myvoice.services.transcription_queue_service import QueuePriority

            # Queue with normal priority for auto-discovered files
            success = await self.transcription_queue_service.queue_transcription(
                voice_profile=profile,
                priority=QueuePriority.NORMAL
            )

            if success:
                self.logger.info(f"Queued transcription for new profile: {profile.name}")
            else:
                self.logger.debug(f"Profile {profile.name} was not queued for transcription")

            return success

        except Exception as e:
            self.logger.error(f"Failed to queue transcription for {profile.name}: {e}")
            return False

    async def queue_transcription_for_profile(self, profile_name: str, priority: str = "normal") -> bool:
        """
        Manually queue a specific profile for transcription.

        Args:
            profile_name: Name of the profile to transcribe
            priority: Priority level ("low", "normal", "high", "urgent")

        Returns:
            bool: True if successfully queued, False otherwise

        Raises:
            MyVoiceError: If profile not found or transcription service unavailable
        """
        if not self.transcription_queue_service:
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="TRANSCRIPTION_SERVICE_UNAVAILABLE",
                user_message="Transcription service is not available",
                technical_details="TranscriptionQueueService is not initialized",
                suggested_action="Ensure transcription service is properly configured"
            )

        profile = self._profiles.get(profile_name)
        if not profile:
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="VOICE_PROFILE_NOT_FOUND",
                user_message=f"Voice profile '{profile_name}' not found",
                technical_details=f"No profile found with name: {profile_name}",
                suggested_action="Check available profiles and try again"
            )

        if not profile.can_queue_transcription():
            self.logger.info(f"Profile {profile_name} cannot be queued: {profile.transcription_status}")
            return False

        try:
            # Import here to avoid circular imports
            from myvoice.services.transcription_queue_service import QueuePriority

            # Map string priority to enum
            priority_map = {
                "low": QueuePriority.LOW,
                "normal": QueuePriority.NORMAL,
                "high": QueuePriority.HIGH,
                "urgent": QueuePriority.URGENT
            }

            queue_priority = priority_map.get(priority.lower(), QueuePriority.NORMAL)

            success = await self.transcription_queue_service.queue_transcription(
                voice_profile=profile,
                priority=queue_priority
            )

            if success:
                self.logger.info(f"Manually queued transcription for {profile_name} with priority {priority}")

            return success

        except Exception as e:
            error_msg = f"Failed to queue transcription for {profile_name}: {e}"
            self.logger.error(error_msg)
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="TRANSCRIPTION_QUEUE_FAILED",
                user_message="Failed to queue transcription",
                technical_details=error_msg,
                suggested_action="Check transcription service status and try again"
            )

    async def queue_missing_transcriptions(self, priority: str = "normal") -> int:
        """
        Queue all profiles missing transcriptions for background processing.

        Args:
            priority: Priority level for the transcriptions

        Returns:
            int: Number of profiles queued for transcription

        Raises:
            MyVoiceError: If transcription service is unavailable
        """
        if not self.transcription_queue_service:
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="TRANSCRIPTION_SERVICE_UNAVAILABLE",
                user_message="Transcription service is not available",
                technical_details="TranscriptionQueueService is not initialized",
                suggested_action="Ensure transcription service is properly configured"
            )

        try:
            # Find profiles that need transcription
            profiles_needing_transcription = [
                profile for profile in self._profiles.values()
                if profile.needs_transcription()
            ]

            if not profiles_needing_transcription:
                self.logger.info("No profiles need transcription")
                return 0

            # Import here to avoid circular imports
            from myvoice.services.transcription_queue_service import QueuePriority

            # Map string priority to enum
            priority_map = {
                "low": QueuePriority.LOW,
                "normal": QueuePriority.NORMAL,
                "high": QueuePriority.HIGH,
                "urgent": QueuePriority.URGENT
            }

            queue_priority = priority_map.get(priority.lower(), QueuePriority.NORMAL)

            # Queue each profile
            queued_count = 0
            for profile in profiles_needing_transcription:
                try:
                    success = await self.transcription_queue_service.queue_transcription(
                        voice_profile=profile,
                        priority=queue_priority
                    )
                    if success:
                        queued_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to queue transcription for {profile.name}: {e}")

            self.logger.info(f"Queued {queued_count} profiles for transcription with priority {priority}")
            return queued_count

        except Exception as e:
            error_msg = f"Failed to queue missing transcriptions: {e}"
            self.logger.error(error_msg)
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="BATCH_TRANSCRIPTION_QUEUE_FAILED",
                user_message="Failed to queue missing transcriptions",
                technical_details=error_msg,
                suggested_action="Check transcription service status and try again"
            )

    def get_transcription_stats(self) -> Dict[str, Any]:
        """
        Get statistics about transcription status across all profiles.

        Returns:
            dict: Statistics including counts by status and queue information
        """
        from myvoice.models.voice_profile import TranscriptionStatus

        stats = {
            'total_profiles': len(self._profiles),
            'by_status': {},
            'queue_service_available': self.transcription_queue_service is not None
        }

        # Count profiles by transcription status
        for status in TranscriptionStatus:
            count = sum(1 for p in self._profiles.values() if p.transcription_status == status)
            stats['by_status'][status.value] = count

        # Add queue service stats if available
        if self.transcription_queue_service:
            try:
                queue_stats = self.transcription_queue_service.get_queue_stats()
                stats['queue_stats'] = queue_stats
            except Exception as e:
                self.logger.error(f"Failed to get queue stats: {e}")
                stats['queue_stats'] = {'error': str(e)}

        return stats

    async def refresh_transcription_status(self) -> int:
        """
        Refresh transcription status for all profiles by checking saved transcription files.

        This method checks for transcription files that may have been created externally
        and updates the profile status accordingly.

        Returns:
            int: Number of profiles whose status was updated
        """
        from myvoice.models.voice_profile import TranscriptionStatus

        updated_count = 0

        for profile in self._profiles.values():
            try:
                # Check if transcription file exists
                has_transcription = await self.has_transcription(profile.file_path)

                if has_transcription and profile.transcription_status == TranscriptionStatus.NOT_STARTED:
                    # Load the transcription and update profile
                    transcription_result = await self.load_transcription(profile.file_path)
                    if transcription_result:
                        profile.set_transcription_result(
                            transcription_text=transcription_result.text,
                            confidence=transcription_result.confidence,
                            model_name=transcription_result.model_name or "unknown"
                        )
                        updated_count += 1
                        self.logger.info(f"Updated transcription status for {profile.name}")

            except Exception as e:
                self.logger.error(f"Error refreshing transcription status for {profile.name}: {e}")

        if updated_count > 0:
            await self._save_profile_cache()
            self.logger.info(f"Refreshed transcription status for {updated_count} profiles")

        return updated_count

    async def set_active_profile(self, profile_name: Optional[str]) -> bool:
        """
        Set the active voice profile.

        Args:
            profile_name: Name of the profile to activate, or None to clear

        Returns:
            bool: True if profile was set successfully, False otherwise
        """
        try:
            if profile_name is None:
                self._active_profile = None
                self.logger.info("Cleared active voice profile")
                return True

            if profile_name not in self._profiles:
                self.logger.error(f"Profile not found: {profile_name}")
                return False

            profile = self._profiles[profile_name]
            if not profile.is_valid:
                self.logger.error(f"Cannot activate invalid profile: {profile_name}")
                return False

            self._active_profile = profile
            self.logger.info(f"Set active voice profile: {profile_name}")

            # Cancel any existing cache save task (don't await - just cancel and move on)
            if self._cache_save_task and not self._cache_save_task.done():
                self.logger.debug("Cancelling existing cache save task")
                self._cache_save_task.cancel()

            # Save cache in background - don't wait for it to complete
            # This prevents blocking the voice switch and avoids conflicts with config saves
            try:
                # Check if event loop is running before creating task
                try:
                    loop = asyncio.get_running_loop()
                    self._cache_save_task = asyncio.create_task(self._save_profile_cache())
                except RuntimeError:
                    # No running event loop - save synchronously without blocking
                    self.logger.debug("No running event loop, skipping background cache save")
            except Exception as e:
                self.logger.error(f"Error starting profile cache save task: {e}")

            return True

        except Exception as e:
            self.logger.exception(f"Error setting active profile {profile_name}: {e}")
            return False

    def get_active_profile(self) -> Optional[VoiceProfile]:
        """Get the currently active voice profile."""
        return self._active_profile

    def get_active_profile_model_type(self) -> Optional[QwenModelType]:
        """
        Get the QwenModelType required for the currently active voice profile.

        This is used to determine which model should be loaded at startup
        and when switching between voice groups.

        Returns:
            QwenModelType: The model type required for the active profile,
                          or None if no active profile is set.

        Example:
            - BUNDLED voice -> QwenModelType.CUSTOM_VOICE
            - DESIGNED voice -> QwenModelType.VOICE_DESIGN
            - CLONED voice -> QwenModelType.BASE
        """
        if self._active_profile is None:
            return None

        if self._active_profile.voice_type is None:
            # Default to CUSTOM_VOICE for unknown types
            self.logger.warning(
                f"Active profile '{self._active_profile.name}' has no voice_type, "
                f"defaulting to CUSTOM_VOICE"
            )
            return QwenModelType.CUSTOM_VOICE

        return self._active_profile.voice_type.required_model

    def get_profiles(self) -> Dict[str, VoiceProfile]:
        """Get all managed voice profiles."""
        return self._profiles.copy()

    def get_valid_profiles(self) -> Dict[str, VoiceProfile]:
        """Get only valid voice profiles."""
        return {name: profile for name, profile in self._profiles.items() if profile.is_valid}

    def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get service performance metrics.

        Returns:
            Dict[str, Any]: Service metrics and statistics
        """
        metrics = self.get_status_info()
        metrics.update({
            "total_profiles": len(self._profiles),
            "valid_profiles": len(self.get_valid_profiles()),
            "invalid_profiles": len(self._profiles) - len(self.get_valid_profiles()),
            "active_profile": self._active_profile.name if self._active_profile else None,
            "voice_directory": str(self.voice_directory),
            "max_duration": self.max_duration,
            "last_scan_time": self._last_scan_time,
            "total_scanned": self._total_scanned
        })
        return metrics

    def _initialize_bundled_voices(self) -> int:
        """
        Initialize bundled voice profiles for the 9 pre-trained CustomVoice timbres.

        Task: Integrate 9 Template Timbres into Voice Library

        Creates VoiceProfile entries for all 9 bundled speakers:
        - Vivian, Serena, Uncle_Fu, Dylan, Eric (Chinese)
        - Ryan, Aiden (English)
        - Ono_Anna (Japanese)
        - Sohee (Korean)

        These profiles:
        - Have voice_type=VoiceType.BUNDLED
        - Have emotion_capable=True (support emotion presets)
        - Don't require audio files (use embedded model timbres)
        - Are always valid and ready to use

        Returns:
            int: Number of bundled profiles initialized
        """
        initialized_count = 0

        for speaker_name in BUNDLED_SPEAKERS:
            # Skip if already exists (from cache or previous init)
            if speaker_name in self._profiles:
                self.logger.debug(f"Bundled voice already exists: {speaker_name}")
                continue

            try:
                # Create bundled profile using factory method
                profile = VoiceProfile.create_bundled_profile(speaker_name)
                self._profiles[speaker_name] = profile
                initialized_count += 1
                self.logger.debug(f"Initialized bundled voice: {speaker_name}")

            except Exception as e:
                self.logger.error(f"Failed to initialize bundled voice {speaker_name}: {e}")

        if initialized_count > 0:
            self.logger.info(f"Initialized {initialized_count} bundled voice profiles")

        return initialized_count

    def get_bundled_profiles(self) -> Dict[str, VoiceProfile]:
        """
        Get only bundled voice profiles.

        Returns:
            Dict[str, VoiceProfile]: Dictionary of bundled profiles by name
        """
        from myvoice.models.voice_profile import VoiceType
        return {
            name: profile for name, profile in self._profiles.items()
            if profile.voice_type == VoiceType.BUNDLED
        }

    def _scan_embeddings_directory(self) -> int:
        """
        Scan voice_files/embeddings/ directory for saved embedding voices.

        Story 1.6: Emotion Control for Saved Voice
        Emotion Variants: Support for multi-emotion embedding voices

        Scans the embeddings directory for voice folders. Supports two structures:

        1. Emotion Variants (v2.0): Emotion subfolders with embedding.pt
           embeddings/{voice_name}/
               neutral/embedding.pt
               happy/embedding.pt
               ...
               preview.wav
               metadata.json (with available_emotions)

        2. Legacy (v1.0): Single embedding.pt at root
           embeddings/{voice_name}/
               embedding.pt
               preview.wav
               metadata.json

        Returns:
            int: Number of embedding profiles loaded
        """
        embeddings_dir = self.voice_directory / "embeddings"

        if not embeddings_dir.exists():
            self.logger.debug(f"Embeddings directory does not exist: {embeddings_dir}")
            return 0

        loaded_count = 0

        # Iterate through subdirectories (each voice has its own folder)
        for voice_dir in embeddings_dir.iterdir():
            if not voice_dir.is_dir():
                continue

            voice_name = voice_dir.name

            # QA8: Check if cached profile has stale path - update if needed
            if voice_name in self._profiles:
                existing_profile = self._profiles[voice_name]
                if existing_profile.voice_type == VoiceType.EMBEDDING:
                    # Verify cached checkpoint_path matches disk location
                    if existing_profile.checkpoint_path and existing_profile.checkpoint_path != voice_dir:
                        self.logger.info(
                            f"Embedding voice '{voice_name}' path mismatch: "
                            f"cached={existing_profile.checkpoint_path}, disk={voice_dir}. Updating."
                        )
                        # Path changed - need to rescan this voice
                        del self._profiles[voice_name]
                    else:
                        self.logger.debug(f"Embedding voice already exists: {voice_name}")
                        continue
                else:
                    self.logger.debug(f"Voice already exists (non-embedding): {voice_name}")
                    continue

            try:
                # Emotion Variants: Detect available emotions from folder structure
                available_emotions = VoiceProfile.detect_available_emotions(voice_dir)

                # Skip if no embeddings found at all
                if not available_emotions:
                    self.logger.debug(f"No embeddings found in {voice_dir}, skipping")
                    continue

                # Determine embedding path (for compatibility with create_embedding_profile)
                # Use the base directory for v2.0, or specific file for legacy
                if len(available_emotions) > 1 or (
                    len(available_emotions) == 1 and
                    (voice_dir / available_emotions[0] / "embedding.pt").exists()
                ):
                    # v2.0 structure: use directory path
                    embedding_path = voice_dir
                else:
                    # Legacy structure: use root embedding.pt
                    embedding_path = voice_dir / "embedding.pt"

                # Load and parse metadata
                metadata_file = voice_dir / "metadata.json"
                description = None
                transcription = None  # ref_text for TTS generation

                if metadata_file.exists():
                    # Use the helper method from VoiceProfile for consistent parsing
                    parsed_metadata = VoiceProfile.parse_embedding_metadata(metadata_file)
                    description = parsed_metadata.get('description')
                    transcription = parsed_metadata.get('transcription')

                    # Use metadata name if present
                    if parsed_metadata.get('name'):
                        voice_name = parsed_metadata['name']

                    # Prefer metadata's available_emotions if present (v2.0)
                    if parsed_metadata.get('available_emotions'):
                        metadata_emotions = parsed_metadata['available_emotions']
                        # Verify they actually exist on disk
                        available_emotions = [
                            e for e in metadata_emotions
                            if e in available_emotions or (
                                e == "neutral" and (voice_dir / "embedding.pt").exists()
                            )
                        ]
                        if not available_emotions:
                            available_emotions = ["neutral"]

                # Check for preview audio
                preview_audio = voice_dir / "preview.wav"
                preview_path = preview_audio if preview_audio.exists() else None

                # Create embedding profile using factory method
                profile = VoiceProfile.create_embedding_profile(
                    name=voice_name,
                    embedding_path=embedding_path,
                    description=description,
                    preview_audio_path=preview_path,
                    available_emotions=available_emotions,
                    transcription=transcription
                )

                self._profiles[voice_name] = profile
                loaded_count += 1

                # Log with emotion info
                if len(available_emotions) > 1:
                    self.logger.debug(
                        f"Loaded embedding voice: {voice_name} "
                        f"(emotions: {', '.join(available_emotions)})"
                    )
                else:
                    self.logger.debug(f"Loaded embedding voice: {voice_name} (neutral only)")

            except Exception as e:
                self.logger.error(f"Failed to load embedding voice from {voice_dir}: {e}")

        if loaded_count > 0:
            self.logger.info(f"Loaded {loaded_count} embedding voice profiles from {embeddings_dir}")

        return loaded_count

    def get_embedding_profiles(self) -> Dict[str, VoiceProfile]:
        """
        Get only embedding voice profiles.

        Story 1.6: Emotion Control for Saved Voice

        Returns:
            Dict[str, VoiceProfile]: Dictionary of embedding profiles by name
        """
        from myvoice.models.voice_profile import VoiceType
        return {
            name: profile for name, profile in self._profiles.items()
            if profile.voice_type == VoiceType.EMBEDDING
        }

    async def register_profile(self, profile: VoiceProfile) -> bool:
        """
        Register an externally created voice profile with the manager.

        This method is used to add profiles created outside of the normal
        scanning process, such as cloned or designed voices created through
        dialogs. The profile is added to the managed profiles dictionary
        and the cache is saved.

        Args:
            profile: VoiceProfile to register

        Returns:
            bool: True if profile was registered successfully
        """
        try:
            if not profile.is_valid:
                self.logger.warning(f"Cannot register invalid profile: {profile.name}")
                return False

            # Add to managed profiles
            self._profiles[profile.name] = profile
            self.logger.info(f"Registered profile: {profile.name} (type={profile.voice_type.value})")

            # Save cache in background
            await self._save_profile_cache()

            return True

        except Exception as e:
            self.logger.exception(f"Error registering profile {profile.name}: {e}")
            return False

    async def _load_profile_cache(self):
        """Load profiles from cache file with invalidation checking."""
        try:
            if not self.cache_file.exists():
                self.logger.debug("No cache file found, starting with empty profiles")
                return

            cache_data = json.loads(self.cache_file.read_text(encoding='utf-8'))
            cache_version = cache_data.get('cache_version', '0.0')

            # Check cache version compatibility
            if cache_version != '1.0':
                self.logger.info(f"Cache version {cache_version} incompatible, clearing cache")
                return

            loaded_count = 0
            invalidated_count = 0

            # Load profiles from cache with invalidation checking
            for profile_data in cache_data.get('profiles', []):
                try:
                    file_path = Path(profile_data['file_path'])
                    file_path_str = str(file_path)

                    # Bundled voices have virtual paths (bundled://SpeakerName or bundled:\SpeakerName on Windows)
                    # Skip these as they'll be re-initialized by _initialize_bundled_voices()
                    is_bundled_path = (
                        file_path_str.startswith("bundled://") or
                        file_path_str.startswith("bundled:\\") or
                        file_path_str.startswith("bundled:")
                    )
                    if is_bundled_path:
                        self.logger.debug(f"Skipping bundled voice in cache: {profile_data['name']}")
                        continue

                    # Optimized voices have virtual paths (optimized://VoiceName or optimized:\VoiceName on Windows)
                    # These are fine-tuned from checkpoints, no audio file needed
                    is_optimized_path = (
                        file_path_str.startswith("optimized://") or
                        file_path_str.startswith("optimized:\\") or
                        file_path_str.startswith("optimized:")
                    )

                    # For optimized voices, validate checkpoint instead of file
                    if is_optimized_path:
                        checkpoint_path = profile_data.get('checkpoint_path')
                        if checkpoint_path and Path(checkpoint_path).exists():
                            # Load optimized voice from cache
                            profile = VoiceProfile(
                                file_path=file_path,
                                name=profile_data['name'],
                                transcription=None,
                                duration=None,
                                is_valid=True,
                                voice_type=VoiceType.OPTIMIZED,
                                checkpoint_path=Path(checkpoint_path),
                                speaker_name=profile_data.get('speaker_name'),
                                description=profile_data.get('description'),
                            )
                            self._profiles[profile.name] = profile
                            loaded_count += 1
                            self.logger.debug(f"Loaded optimized voice from cache: {profile.name}")
                        else:
                            self.logger.debug(f"Optimized voice checkpoint not found: {checkpoint_path}")
                            invalidated_count += 1
                        continue

                    # Embedding voices have virtual paths (embedding://VoiceName or embedding:\VoiceName on Windows)
                    # Story 1.6: Emotion Control for Saved Voice
                    is_embedding_path = (
                        file_path_str.startswith("embedding://") or
                        file_path_str.startswith("embedding:\\") or
                        file_path_str.startswith("embedding:")
                    )

                    # For embedding voices, validate embedding file exists
                    if is_embedding_path:
                        checkpoint_path = profile_data.get('checkpoint_path')
                        if checkpoint_path and Path(checkpoint_path).exists():
                            # QA8: Load available_emotions from cache
                            available_emotions = profile_data.get('available_emotions', ["neutral"])
                            if not available_emotions:
                                available_emotions = ["neutral"]

                            # Load embedding voice from cache
                            profile = VoiceProfile(
                                file_path=file_path,
                                name=profile_data['name'],
                                transcription=None,
                                duration=profile_data.get('duration'),
                                is_valid=True,
                                voice_type=VoiceType.EMBEDDING,
                                checkpoint_path=Path(checkpoint_path),
                                speaker_name=None,
                                description=profile_data.get('description'),
                                available_emotions=available_emotions,
                            )
                            self._profiles[profile.name] = profile
                            loaded_count += 1
                            self.logger.debug(f"Loaded embedding voice from cache: {profile.name} (emotions: {available_emotions})")
                        else:
                            self.logger.debug(f"Embedding file not found: {checkpoint_path}")
                            invalidated_count += 1
                        continue

                    # Check if file exists (for non-virtual voices only)
                    if not file_path.exists():
                        self.logger.debug(f"Cached profile file not found: {file_path}")
                        invalidated_count += 1
                        continue

                    # Cache invalidation: check file modification time and size
                    cached_mtime = profile_data.get('file_mtime')
                    cached_size = profile_data.get('file_size')

                    if cached_mtime is not None and cached_size is not None:
                        current_mtime = file_path.stat().st_mtime
                        current_size = file_path.stat().st_size

                        # If file has been modified or size changed, invalidate cache entry
                        if abs(current_mtime - cached_mtime) > 1.0 or current_size != cached_size:
                            self.logger.debug(f"Cache invalidated for {file_path}: file modified")
                            invalidated_count += 1
                            continue

                    # Recreate profile from cached data
                    # Restore voice_type if present in cache, otherwise default to CLONED
                    voice_type = VoiceType.CLONED
                    if profile_data.get('voice_type'):
                        try:
                            voice_type = VoiceType(profile_data['voice_type'])
                        except ValueError:
                            pass  # Keep default

                    profile = VoiceProfile(
                        file_path=file_path,
                        name=profile_data['name'],
                        transcription=profile_data.get('transcription'),
                        duration=profile_data.get('duration'),
                        is_valid=profile_data.get('is_valid', False),
                        voice_type=voice_type,
                    )
                    self._profiles[profile.name] = profile
                    loaded_count += 1

                except Exception as e:
                    self.logger.error(f"Error loading cached profile: {e}")
                    invalidated_count += 1

            # Load active profile
            active_name = cache_data.get('active_profile')
            if active_name and active_name in self._profiles:
                self._active_profile = self._profiles[active_name]

            self.logger.info(f"Loaded {loaded_count} profiles from cache, invalidated {invalidated_count}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Cache file contains invalid JSON: {e}")
            await self._recover_corrupted_cache("invalid_json")
        except (PermissionError, OSError) as e:
            self.logger.warning(f"Cannot access cache file: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading profile cache: {e}")
            await self._recover_corrupted_cache("unexpected_error")

    async def _recover_corrupted_cache(self, reason: str):
        """
        Recover from corrupted cache by backing up and starting fresh.

        Args:
            reason: Reason for cache corruption
        """
        try:
            self.logger.warning(f"Recovering from corrupted cache (reason: {reason})")

            # Create backup of corrupted cache
            if self.cache_file.exists():
                backup_file = self.cache_file.with_suffix(f'.backup.{int(__import__("time").time())}')
                try:
                    import shutil
                    shutil.copy2(self.cache_file, backup_file)
                    self.logger.info(f"Backed up corrupted cache to: {backup_file}")
                except Exception as e:
                    self.logger.warning(f"Cannot backup corrupted cache: {e}")

                # Remove corrupted cache
                try:
                    self.cache_file.unlink()
                    self.logger.info("Removed corrupted cache file")
                except Exception as e:
                    self.logger.warning(f"Cannot remove corrupted cache: {e}")

            # Clear in-memory profiles and start fresh
            self._profiles.clear()
            self._active_profile = None
            self.logger.info("Starting with fresh cache after corruption recovery")

        except Exception as e:
            self.logger.error(f"Error during cache recovery: {e}")

    def _save_profile_cache_sync(self, profile_data: List[Dict[str, Any]], active_profile_name: Optional[str], last_scan_time: Optional[float]) -> bool:
        """
        Synchronous cache saving for thread pool execution.

        This method does ALL file I/O operations,
        to avoid any blocking operations in the async caller.

        Args:
            profile_data: List of serialized profile dictionaries with file_path strings
            active_profile_name: Name of active profile or None
            last_scan_time: Last scan timestamp

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            import time
            from pathlib import Path

            # Build cache data with file stats
            enriched_profiles = []
            for profile_dict in profile_data:
                file_path = Path(profile_dict['file_path'])
                enriched_profile = profile_dict.copy()

                # Add file stats - this is the ONLY place we do file I/O
                try:
                    if file_path.exists():
                        stat_info = file_path.stat()
                        enriched_profile['file_mtime'] = stat_info.st_mtime
                        enriched_profile['file_size'] = stat_info.st_size
                    else:
                        enriched_profile['file_mtime'] = None
                        enriched_profile['file_size'] = None
                except (OSError, PermissionError):
                    enriched_profile['file_mtime'] = None
                    enriched_profile['file_size'] = None

                enriched_profile['cached_at'] = time.time()
                enriched_profiles.append(enriched_profile)

            cache_data = {
                'profiles': enriched_profiles,
                'active_profile': active_profile_name,
                'last_updated': last_scan_time,
                'cache_version': '1.0'
            }

            # Create directory if it doesn't exist
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Write atomically using temporary file
            temp_file = self.cache_file.with_suffix('.tmp')
            temp_file.write_text(json.dumps(cache_data, indent=2), encoding='utf-8')
            temp_file.replace(self.cache_file)

            self.logger.debug(f"Saved {len(enriched_profiles)} profiles to cache")
            return True

        except Exception as e:
            self.logger.error(f"Error saving profile cache: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def _save_profile_cache(self):
        """Save profiles to cache file using thread pool to avoid blocking."""
        try:
            self.logger.debug(f"Starting profile cache save, {len(self._profiles)} profiles")

            # Take a snapshot of the profiles dictionary to avoid
            # "dictionary changed size during iteration" errors
            profiles_snapshot = list(self._profiles.values())
            self.logger.debug(f"Created snapshot of {len(profiles_snapshot)} profiles")

            # Serialize ALL data to simple types BEFORE passing to executor
            # This ensures no VoiceProfile objects or Path objects cross thread boundaries
            profile_data = []
            for i, profile in enumerate(profiles_snapshot):
                try:
                    data = {
                        'name': profile.name,
                        'file_path': str(profile.file_path),  # Convert Path to string
                        'transcription': profile.transcription,
                        'duration': profile.duration,
                        'is_valid': profile.is_valid,
                        'voice_type': profile.voice_type.value if profile.voice_type else None,
                        'emotion_capable': profile.emotion_capable,
                        # OPTIMIZED voice metadata
                        'checkpoint_path': str(profile.checkpoint_path) if profile.checkpoint_path else None,
                        'speaker_name': profile.speaker_name,
                        'description': profile.description,
                        # QA8: Include available_emotions for EMBEDDING voices
                        'available_emotions': profile.available_emotions if hasattr(profile, 'available_emotions') else ["neutral"],
                    }
                    profile_data.append(data)
                    self.logger.debug(f"Serialized profile {i+1}/{len(profiles_snapshot)}: {profile.name}")
                except Exception as e:
                    self.logger.error(f"Error serializing profile {i}: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")

            active_profile_name = self._active_profile.name if self._active_profile else None
            self.logger.debug(f"Active profile: {active_profile_name}")

            # Save using thread pool to avoid blocking the event loop
            # Only pass simple serialized data (no objects, no Path instances)
            loop = asyncio.get_running_loop()
            self.logger.debug("Submitting cache save to executor")
            await loop.run_in_executor(
                self._executor,
                self._save_profile_cache_sync,
                profile_data,
                active_profile_name,
                self._last_scan_time
            )
            self.logger.debug("Profile cache save completed successfully")

        except Exception as e:
            self.logger.error(f"Error in async save profile cache: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    async def invalidate_cache(self, profile_name: Optional[str] = None) -> bool:
        """
        Invalidate cache for specific profile or entire cache.

        Args:
            profile_name: Name of specific profile to invalidate, or None for all

        Returns:
            bool: True if invalidation was successful
        """
        try:
            if profile_name is None:
                # Invalidate entire cache
                self._profiles.clear()
                self._active_profile = None
                if self.cache_file.exists():
                    self.cache_file.unlink()
                self.logger.info("Entire profile cache invalidated")
                return True
            else:
                # Invalidate specific profile
                if profile_name in self._profiles:
                    del self._profiles[profile_name]
                    if self._active_profile and self._active_profile.name == profile_name:
                        self._active_profile = None
                    await self._save_profile_cache()
                    self.logger.info(f"Profile cache invalidated for: {profile_name}")
                    return True
                else:
                    self.logger.warning(f"Profile not found for invalidation: {profile_name}")
                    return False

        except Exception as e:
            self.logger.error(f"Error invalidating cache: {e}")
            return False

    async def _ensure_directories_exist(self) -> None:
        """
        Ensure required directories exist with comprehensive error handling.

        Raises:
            MyVoiceError: If directories cannot be created due to permission issues
        """
        try:
            # Create voice files directory
            self.voice_directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Voice directory ensured: {self.voice_directory}")

            # Verify we can write to the voice directory
            test_file = self.voice_directory / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                self.logger.debug("Voice directory write permission verified")
            except (PermissionError, OSError) as e:
                raise MyVoiceError(
                    message=f"Voice directory is not writable: {self.voice_directory}",
                    details={"directory": str(self.voice_directory), "error": str(e)},
                    severity=ErrorSeverity.HIGH,
                    user_message=UserFriendlyMessage(
                        title="Directory Permission Error",
                        message=f"Cannot write to voice files directory: {self.voice_directory}",
                        suggestion="Please check directory permissions or select a different directory."
                    )
                )

        except PermissionError as e:
            self.logger.error(f"Permission denied creating voice directory: {e}")
            raise MyVoiceError(
                message=f"Permission denied creating voice directory: {self.voice_directory}",
                details={"directory": str(self.voice_directory), "error": str(e)},
                severity=ErrorSeverity.HIGH,
                user_message=UserFriendlyMessage(
                    title="Directory Creation Failed",
                    message=f"Cannot create voice files directory: {self.voice_directory}",
                    suggestion="Please check folder permissions or run as administrator."
                )
            )
        except OSError as e:
            self.logger.error(f"OS error creating voice directory: {e}")
            raise MyVoiceError(
                message=f"System error creating voice directory: {self.voice_directory}",
                details={"directory": str(self.voice_directory), "error": str(e)},
                severity=ErrorSeverity.HIGH,
                user_message=UserFriendlyMessage(
                    title="Directory Creation Failed",
                    message=f"System error creating directory: {self.voice_directory}",
                    suggestion="Please check disk space and system permissions."
                )
            )

        try:
            # Create cache directory
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Cache directory ensured: {self.cache_file.parent}")

            # Verify we can write to the cache directory
            test_file = self.cache_file.parent / ".cache_write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                self.logger.debug("Cache directory write permission verified")
            except (PermissionError, OSError) as e:
                self.logger.warning(f"Cache directory not writable, caching disabled: {e}")
                # This is a warning, not a fatal error - continue without caching

        except (PermissionError, OSError) as e:
            self.logger.warning(f"Cannot create cache directory, caching disabled: {e}")
            # Cache directory creation failure is not fatal - continue without caching

    async def force_rescan(self, directory: Optional[Path] = None) -> Dict[str, Any]:
        """
        Force a complete rescan of the voice directory, ignoring cache.

        Args:
            directory: Directory to scan (defaults to configured voice_directory)

        Returns:
            Dict[str, Any]: Scan results with statistics
        """
        self.logger.info("Forcing complete rescan (ignoring cache)")
        return await self.scan_voice_directory(directory, incremental=False)