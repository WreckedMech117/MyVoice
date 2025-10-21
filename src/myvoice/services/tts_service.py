"""
TTS Service Core Implementation

This module implements the core Text-to-Speech service with async operations,
validation, and comprehensive error handling.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor

from myvoice.services.core.base_service import BaseService, ServiceStatus
from myvoice.services.integrations.gpt_sovits_client import GPTSoVITSClient
from myvoice.models.tts_request import TTSRequest, TTSResponse
from myvoice.models.voice_profile import VoiceProfile, TranscriptionStatus
from myvoice.models.validation import (
    ValidationResult, ValidationIssue, ValidationStatus, UserFriendlyMessage
)
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.ui_state import ServiceHealthStatus
from myvoice.models.retry_config import RetryConfig, RetryConfigs, RetryAttempt
from myvoice.models.audio_playback_task import AudioPlaybackTask


class TTSService(BaseService):
    """
    Core Text-to-Speech service with async operations.

    This service provides the main interface for TTS generation using the
    GPT-SoVITS backend. It includes comprehensive validation, error handling,
    and async operation support for desktop application integration.

    Features:
    - Async TTS generation with non-blocking operations
    - Request validation with detailed feedback
    - Service health monitoring
    - User-friendly error messages
    - Resource management and cleanup
    """

    def __init__(
        self,
        audio_coordinator: Optional['AudioCoordinator'] = None,
        audio_manager: Optional['AudioManager'] = None,
        gpt_sovits_host: str = "127.0.0.1",
        gpt_sovits_port: int = 9880,
        request_timeout: float = 30.0,
        max_concurrent_requests: int = 3,
        health_check_interval: float = 30.0,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize the TTS service with dual-service audio coordination.

        Args:
            audio_coordinator: AudioCoordinator for dual-service audio routing (preferred)
            audio_manager: AudioManager instance (deprecated, for backward compatibility)
            gpt_sovits_host: GPT-SoVITS service host
            gpt_sovits_port: GPT-SoVITS service port
            request_timeout: Request timeout in seconds
            max_concurrent_requests: Maximum concurrent TTS requests
            health_check_interval: Health check interval in seconds
            retry_config: Retry configuration for failed requests
        """
        super().__init__("TTSService")

        # Audio integration - prefer AudioCoordinator over AudioManager
        self.audio_coordinator = audio_coordinator
        self.audio_manager = audio_manager  # Keep for backward compatibility

        # Determine which audio system to use
        if self.audio_coordinator:
            self.logger.info("TTSService using AudioCoordinator (dual-service architecture)")
            self._audio_mode = "coordinator"
        elif self.audio_manager:
            self.logger.info("TTSService using AudioManager (legacy mode)")
            self._audio_mode = "legacy"
        else:
            self.logger.warning("TTSService initialized without audio integration")
            self._audio_mode = "none"

        # Service configuration
        self.gpt_sovits_host = gpt_sovits_host
        self.gpt_sovits_port = gpt_sovits_port
        self.request_timeout = request_timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.health_check_interval = health_check_interval
        self.retry_config = retry_config or RetryConfigs.STANDARD

        # Service components
        self._client: Optional[GPTSoVITSClient] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._request_semaphore: Optional[asyncio.Semaphore] = None

        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_status_callback: Optional[Callable[[ServiceHealthStatus, Optional[str]], None]] = None
        self._last_health_status = ServiceHealthStatus.UNKNOWN

        # Retry feedback
        self._retry_callback: Optional[Callable[[RetryAttempt], None]] = None

        # Metrics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0

        # Virtual microphone configuration
        self._preferred_virtual_device_id: Optional[str] = None

        self.logger.debug(f"TTSService initialized with host={gpt_sovits_host}:{gpt_sovits_port}")
        self.logger.debug(f"Retry config: {self.retry_config.max_attempts} attempts")

    async def start(self) -> bool:
        """
        Start the TTS service.

        Returns:
            bool: True if service started successfully, False otherwise
        """
        try:
            await self._update_status(ServiceStatus.STARTING)
            self.logger.info("Starting TTS service")

            # Initialize GPT-SoVITS client with retry configuration
            self._client = GPTSoVITSClient(
                host=self.gpt_sovits_host,
                port=self.gpt_sovits_port,
                timeout=self.request_timeout,
                retry_config=self.retry_config,
                retry_callback=self._retry_callback
            )

            # Initialize thread executor for blocking operations
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent_requests,
                thread_name_prefix="TTS"
            )

            # Initialize request semaphore for concurrency control
            self._request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            # Verify service connectivity (but don't fail if backend is unavailable)
            is_healthy, error = await self._check_backend_health()
            if not is_healthy:
                self.logger.warning(f"Backend not available during initialization: {error.user_message if error else 'Unknown error'}")
                self.logger.info("TTS service will continue starting and attempt to reconnect via health monitoring")
                # Don't fail - allow service to start and recover later
            else:
                self.logger.info("Backend health check passed during initialization")

            # Start periodic health monitoring (will attempt reconnection)
            await self._start_health_monitoring()

            # Trigger initial health status callback immediately
            if self._health_status_callback:
                initial_health_status = ServiceHealthStatus.HEALTHY if is_healthy else ServiceHealthStatus.ERROR
                error_message = error.user_message if error else None
                self._last_health_status = initial_health_status
                self.logger.info(f"Triggering initial health status callback: {initial_health_status.value}")
                try:
                    self._health_status_callback(initial_health_status, error_message)
                    self.logger.info("Initial health status callback completed successfully")
                except Exception as e:
                    self.logger.error(f"Error in health status callback: {e}", exc_info=True)

            await self._update_status(ServiceStatus.RUNNING)
            self.logger.info("TTS service started successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to start TTS service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def stop(self) -> bool:
        """
        Stop the TTS service.

        Returns:
            bool: True if service stopped successfully, False otherwise
        """
        try:
            await self._update_status(ServiceStatus.STOPPING)
            self.logger.info("Stopping TTS service")

            # Stop health monitoring
            await self._stop_health_monitoring()

            # Cleanup resources
            if self._client:
                self._client.close()
                self._client = None

            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None

            self._request_semaphore = None

            await self._update_status(ServiceStatus.STOPPED)
            self.logger.info("TTS service stopped successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error stopping TTS service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """
        Check TTS service health.

        Returns:
            tuple[bool, Optional[MyVoiceError]]: (is_healthy, error_if_any)
        """
        try:
            # Check service status
            if not self.is_running():
                return False, MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="SERVICE_NOT_RUNNING",
                    user_message="TTS service is not running",
                    suggested_action="Start the TTS service"
                )

            # Check backend connectivity
            return await self._check_backend_health()

        except Exception as e:
            self.logger.exception(f"Health check failed: {e}")
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="HEALTH_CHECK_FAILED",
                user_message="Failed to check TTS service health",
                technical_details=str(e)
            )

    async def generate_speech_from_profile(
        self,
        text: str,
        voice_profile: VoiceProfile,
        response_format: str = "wav",
        fallback_text: Optional[str] = None
    ) -> TTSResponse:
        """
        Generate speech using a voice profile with automatic transcription integration.

        This method automatically uses the voice profile's transcription for better
        voice cloning quality, with fallback handling for missing transcriptions.

        Args:
            text: Text to convert to speech
            voice_profile: Voice profile containing audio file and transcription
            response_format: Audio format for response (default: wav)
            fallback_text: Fallback transcription text if profile transcription is unavailable

        Returns:
            TTSResponse: Response containing audio data or error information
        """
        try:
            # Get transcription from voice profile with fallback handling
            voice_text = self._get_voice_transcription(voice_profile, fallback_text)

            # Create TTS request with transcription
            request = TTSRequest(
                text=text,
                voice_file_path=voice_profile.file_path,
                voice_text=voice_text,
                response_format=response_format
            )

            self.logger.info(f"Generating speech with voice profile: {voice_profile.name}")
            if voice_text:
                self.logger.debug(f"Using transcription: {voice_text[:50]}...")
            else:
                self.logger.warning(f"No transcription available for voice profile: {voice_profile.name}")

            # Generate speech using the standard method
            return await self.generate_speech(request)

        except Exception as e:
            self.logger.exception(f"Error in profile-based speech generation: {e}")
            self._failed_requests += 1
            user_message = self.handle_service_error(e, "generate_speech_from_profile")
            return TTSResponse(
                success=False,
                error_message=user_message.message
            )

    async def generate_speech(self, request: TTSRequest) -> TTSResponse:
        """
        Generate speech from text using voice cloning.

        This is the main async interface for TTS generation. It validates
        the request, manages concurrency, and provides comprehensive error handling.

        Args:
            request: TTS request containing text and voice information

        Returns:
            TTSResponse: Response containing audio data or error information
        """
        self._total_requests += 1

        try:
            self.logger.info(f"Starting TTS generation for text: {request.text[:50]}...")

            # Validate request
            validation_result = self.validate_request(request)
            if not validation_result.is_valid:
                self._failed_requests += 1
                return TTSResponse(
                    success=False,
                    error_message=f"Validation failed: {validation_result.summary}"
                )

            # Check backend health before processing (not just service status)
            # This allows recovery even if service was in ERROR state initially
            is_healthy, health_error = await self._check_backend_health()
            if not is_healthy:
                self._failed_requests += 1
                self.logger.warning(f"Backend health check failed during TTS request: {health_error.user_message if health_error else 'Unknown error'}")
                return TTSResponse(
                    success=False,
                    error_message=health_error.user_message if health_error else "GPT-SoVITS backend is not available"
                )
            else:
                # Backend is healthy - ensure service status is RUNNING
                if self.status != ServiceStatus.RUNNING:
                    self.logger.info("Backend healthy, updating service status to RUNNING")
                    await self._update_status(ServiceStatus.RUNNING)

            # Use semaphore to control concurrency
            async with self._request_semaphore:
                # Execute TTS generation in thread pool to avoid blocking
                response = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._client.generate_speech,
                    request
                )

                if response.success:
                    self._successful_requests += 1
                    self.logger.info(f"TTS generation completed successfully, {len(response.audio_data)} bytes")

                    # Audio playback is handled by the app layer to respect user monitoring preferences
                    # await self._handle_successful_tts_generation(response, request)
                else:
                    self._failed_requests += 1
                    self.logger.error(f"TTS generation failed: {response.error_message}")

                return response

        except asyncio.CancelledError:
            self.logger.info("TTS generation was cancelled")
            self._failed_requests += 1
            return TTSResponse(
                success=False,
                error_message="Operation was cancelled"
            )

        except Exception as e:
            self.logger.exception(f"Error during TTS generation: {e}")
            self._failed_requests += 1

            # Convert to user-friendly message
            user_message = self.handle_service_error(e, "generate_speech")
            return TTSResponse(
                success=False,
                error_message=user_message.message
            )

    def validate_request(self, request: TTSRequest) -> ValidationResult:
        """
        Validate TTS request parameters.

        Args:
            request: TTS request to validate

        Returns:
            ValidationResult: Detailed validation result
        """
        issues = []
        warnings = []

        try:
            # Text validation
            if not request.text or not request.text.strip():
                issues.append(ValidationIssue(
                    field="text",
                    message="Text cannot be empty",
                    code="EMPTY_TEXT",
                    severity=ValidationStatus.INVALID
                ))

            # Check text length
            text_length = len(request.text.strip())
            if text_length > 1000:  # Reasonable limit for TTS
                warnings.append(ValidationIssue(
                    field="text",
                    message=f"Text is quite long ({text_length} characters), generation may take longer",
                    code="LONG_TEXT",
                    severity=ValidationStatus.WARNING
                ))

            # Voice file validation
            if not request.voice_file_path.exists():
                issues.append(ValidationIssue(
                    field="voice_file_path",
                    message=f"Voice file not found: {request.voice_file_path}",
                    code="FILE_NOT_FOUND",
                    severity=ValidationStatus.INVALID
                ))
            else:
                # Check file format
                if request.voice_file_path.suffix.lower() != '.wav':
                    issues.append(ValidationIssue(
                        field="voice_file_path",
                        message="Voice file must be in WAV format",
                        code="INVALID_FORMAT",
                        severity=ValidationStatus.INVALID
                    ))

                # Check file size
                file_size = request.voice_file_path.stat().st_size
                if file_size == 0:
                    issues.append(ValidationIssue(
                        field="voice_file_path",
                        message="Voice file is empty",
                        code="EMPTY_FILE",
                        severity=ValidationStatus.INVALID
                    ))
                elif file_size > 25 * 1024 * 1024:  # 25MB limit
                    issues.append(ValidationIssue(
                        field="voice_file_path",
                        message="Voice file is too large (maximum 25MB)",
                        code="FILE_TOO_LARGE",
                        severity=ValidationStatus.INVALID
                    ))
                elif file_size < 1024:  # Very small files are suspicious
                    warnings.append(ValidationIssue(
                        field="voice_file_path",
                        message="Voice file is very small, quality may be poor",
                        code="SMALL_FILE",
                        severity=ValidationStatus.WARNING
                    ))

            # Response format validation
            valid_formats = ["wav", "mp3", "flac"]
            if request.response_format not in valid_formats:
                warnings.append(ValidationIssue(
                    field="response_format",
                    message=f"Response format '{request.response_format}' may not be supported",
                    code="UNSUPPORTED_FORMAT",
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
            self.logger.exception(f"Error during request validation: {e}")
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

    def handle_service_errors(self, error: Exception) -> UserFriendlyMessage:
        """
        Handle service-specific errors and convert to user-friendly messages.

        Args:
            error: The exception that occurred

        Returns:
            UserFriendlyMessage: User-friendly error message
        """
        # Use the base service error handler with TTS-specific context
        return self.handle_service_error(error, "TTS generation")

    async def _check_backend_health(self) -> tuple[bool, Optional[MyVoiceError]]:
        """
        Check GPT-SoVITS backend health.

        Returns:
            tuple[bool, Optional[MyVoiceError]]: (is_healthy, error_if_any)
        """
        if not self._client:
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="CLIENT_NOT_INITIALIZED",
                user_message="TTS client is not initialized"
            )

        try:
            # Run health check in executor to avoid blocking
            is_healthy, error = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._client.health_check
            )
            return is_healthy, error

        except Exception as e:
            self.logger.exception(f"Backend health check failed: {e}")
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="BACKEND_HEALTH_CHECK_FAILED",
                user_message="Failed to check backend service health",
                technical_details=str(e)
            )

    def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get service performance metrics.

        Returns:
            Dict[str, Any]: Service metrics and statistics
        """
        metrics = self.get_status_info()
        metrics.update({
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / self._total_requests * 100
                if self._total_requests > 0 else 0
            ),
            "max_concurrent_requests": self.max_concurrent_requests,
            "backend_host": f"{self.gpt_sovits_host}:{self.gpt_sovits_port}",
            "request_timeout": self.request_timeout,
            "health_check_interval": self.health_check_interval,
            "last_health_status": self._last_health_status.value
        })
        return metrics

    def set_health_status_callback(self, callback: Callable[[ServiceHealthStatus, Optional[str]], None]):
        """
        Set callback for health status updates.

        Args:
            callback: Function to call when health status changes
                     Parameters: (health_status, error_message)
        """
        self._health_status_callback = callback
        self.logger.debug("Health status callback registered")

    def set_retry_callback(self, callback: Callable[[RetryAttempt], None]):
        """
        Set callback for retry attempts to provide user feedback.

        Args:
            callback: Function to call on each retry attempt
                     Parameters: (retry_attempt)
        """
        self._retry_callback = callback

        # If client is already initialized, update its callback
        if self._client:
            self._client.set_retry_callback(callback)

        self.logger.debug("Retry callback registered")

    def update_retry_config(self, config: RetryConfig):
        """
        Update the retry configuration.

        Args:
            config: New retry configuration
        """
        self.retry_config = config

        # If client is already initialized, update its configuration
        if self._client:
            self._client.update_retry_config(config)

        self.logger.debug(f"Retry configuration updated: {config.max_attempts} attempts")

    def get_retry_config(self) -> RetryConfig:
        """Get the current retry configuration."""
        return self.retry_config

    async def check_service_health(self) -> bool:
        """
        Public async method to check service health as required by the task.

        Returns:
            bool: True if service is healthy, False otherwise
        """
        is_healthy, error = await self.health_check()
        return is_healthy

    async def _start_health_monitoring(self):
        """Start the periodic health check task."""
        if self._health_check_task and not self._health_check_task.done():
            self.logger.warning("Health monitoring task already running")
            return

        self.logger.info(f"Starting health monitoring with {self.health_check_interval}s interval")
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())

    async def _stop_health_monitoring(self):
        """Stop the periodic health check task."""
        if self._health_check_task and not self._health_check_task.done():
            self.logger.info("Stopping health monitoring")
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

    async def _health_monitoring_loop(self):
        """Main health monitoring loop that runs periodically."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Perform health check
                is_healthy, error = await self.health_check()

                # Determine health status
                if is_healthy:
                    new_status = ServiceHealthStatus.HEALTHY
                    error_message = None

                    # RECOVERY LOGIC: If service was in ERROR state and is now healthy,
                    # update the service status to RUNNING to allow TTS generation
                    if self.status == ServiceStatus.ERROR:
                        self.logger.info("Backend connection recovered! Updating service status to RUNNING")
                        await self._update_status(ServiceStatus.RUNNING)

                elif error and error.severity == ErrorSeverity.WARNING:
                    new_status = ServiceHealthStatus.WARNING
                    error_message = error.user_message
                else:
                    new_status = ServiceHealthStatus.ERROR
                    error_message = error.user_message if error else "Service health check failed"

                    # If backend is unavailable but service was running, mark as ERROR
                    # (but keep it running so monitoring can recover it later)
                    if self.status == ServiceStatus.RUNNING:
                        self.logger.warning("Backend connection lost, but service will continue monitoring for recovery")

                # Update status if changed
                if new_status != self._last_health_status:
                    self.logger.info(f"Service health status changed: {self._last_health_status.value} -> {new_status.value}")
                    self._last_health_status = new_status

                    # Notify callback if registered
                    if self._health_status_callback:
                        try:
                            self._health_status_callback(new_status, error_message)
                        except Exception as e:
                            self.logger.exception(f"Error in health status callback: {e}")

            except asyncio.CancelledError:
                self.logger.debug("Health monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.exception(f"Error in health monitoring loop: {e}")
                # Continue monitoring despite errors
                await asyncio.sleep(5)  # Short delay before retrying

    async def _handle_successful_tts_generation(self, response: TTSResponse, request: TTSRequest) -> None:
        """
        Handle successful TTS generation by creating dual-routing AudioPlaybackTask for both monitor and virtual microphone.

        Args:
            response: Successful TTS response with audio data
            request: Original TTS request for context

        Raises:
            Exception: Audio playback errors are logged but don't fail the TTS operation
        """
        try:
            if not response.audio_data:
                self.logger.warning("No audio data in TTS response")
                return

            self.logger.debug(f"Starting audio routing for TTS: {request.text[:50]}...")

            if self._audio_mode == "coordinator":
                # Use new dual-service architecture
                await self._execute_coordinated_audio_routing(response, request)
            elif self._audio_mode == "legacy":
                # Fall back to legacy AudioManager (will be deprecated)
                await self._execute_legacy_audio_routing(response, request)
            else:
                self.logger.warning("No audio integration available for TTS playback")

        except Exception as e:
            # Audio playback failures should not fail the TTS operation
            # Log the error but don't raise it
            self.logger.error(f"Failed to execute audio playback after TTS generation: {e}")
            self.logger.exception("Audio playback error details:")

    async def _execute_coordinated_audio_routing(self, response: TTSResponse, request: TTSRequest) -> None:
        """
        Execute audio routing using AudioCoordinator (dual-service architecture).

        Args:
            response: TTS response with audio data
            request: Original TTS request for context
        """
        try:
            if not self.audio_coordinator:
                self.logger.error("AudioCoordinator not available")
                return

            # Check if dual-stream routing is desired
            coordinator_status = await self.audio_coordinator.get_coordinator_status()

            monitor_available = (coordinator_status.get("monitor_service", {}).get("initialized", False))
            virtual_available = (coordinator_status.get("virtual_service", {}).get("initialized", False))

            self.logger.info(f"Audio services status - Monitor: {'✓' if monitor_available else '✗'}, Virtual: {'✓' if virtual_available else '✗'}")

            if monitor_available and virtual_available:
                # Execute dual-stream routing
                self.logger.info("Executing dual-stream audio routing via AudioCoordinator")
                await self._execute_dual_service_playback(response, request)
            elif monitor_available:
                # Monitor-only fallback
                self.logger.info("Executing monitor-only audio routing")
                await self._execute_monitor_only_coordinated(response, request)
            elif virtual_available:
                # Virtual-only fallback
                self.logger.info("Executing virtual-only audio routing")
                await self._execute_virtual_only_coordinated(response, request)
            else:
                self.logger.error("No audio services available for playback")

        except Exception as e:
            self.logger.error(f"Coordinated audio routing failed: {e}")
            raise

    async def _execute_monitor_only_playback(self, response: TTSResponse, request: TTSRequest) -> None:
        """
        Execute monitor-only audio playback (fallback method).

        Args:
            response: TTS response with audio data
            request: Original TTS request for context
        """
        try:
            self.logger.debug("Executing monitor-only audio playback")

            # Create and execute monitor audio playback
            playback_task = await self.audio_manager.play_monitor_audio(
                audio_data=response.audio_data,
                volume_level=1.0  # Default volume
            )

            if playback_task:
                self.logger.info(f"Monitor audio playback started successfully: {playback_task.playback_id}")
            else:
                self.logger.warning("Monitor audio playback returned no task")

        except Exception as e:
            self.logger.error(f"Failed to execute monitor-only playback: {e}")
            raise

    async def _execute_virtual_microphone_only_playback(self, response: TTSResponse, request: TTSRequest, virtual_device) -> None:
        """
        Execute virtual microphone-only audio playback.

        Args:
            response: TTS response with audio data
            request: Original TTS request for context
            virtual_device: Virtual microphone device to use
        """
        try:
            self.logger.debug(f"Executing virtual microphone-only playback to {virtual_device.name}")

            # Create and execute virtual microphone audio playback
            playback_task = await self.audio_manager.play_virtual_microphone(
                audio_data=response.audio_data,
                device_id=virtual_device.device_id,
                volume_level=1.0
            )

            if playback_task:
                self.logger.info(f"Virtual microphone playback started successfully: {playback_task.playback_id}")
            else:
                self.logger.warning("Virtual microphone playback returned no task")

        except Exception as e:
            self.logger.error(f"Failed to execute virtual microphone-only playback: {e}")
            raise

    def set_audio_manager(self, audio_manager: 'AudioManager') -> None:
        """
        Set or update the AudioManager instance for monitor audio playback.

        Args:
            audio_manager: AudioManager instance to use for playback
        """
        self.audio_manager = audio_manager
        self.logger.info("AudioManager integration enabled for TTS service")

    def get_audio_manager(self) -> Optional['AudioManager']:
        """
        Get the current AudioManager instance.

        Returns:
            Optional[AudioManager]: Current AudioManager or None if not set
        """
        return self.audio_manager

    async def enable_dual_stream_routing(self) -> bool:
        """
        Enable dual-stream routing for TTS output to both monitor and virtual microphone.

        Returns:
            bool: True if dual-stream routing was enabled successfully, False otherwise
        """
        try:
            if not self.audio_manager:
                self.logger.error("Cannot enable dual-stream routing: AudioManager not available")
                return False

            # Check if virtual devices are available
            virtual_devices = await self.audio_manager.get_virtual_input_devices_async()
            if not virtual_devices:
                self.logger.warning("No virtual microphone devices found for dual-stream routing")
                return False

            # Enable dual-stream routing in AudioManager
            self.audio_manager.enable_dual_stream_routing()
            self.logger.info(f"Dual-stream routing enabled with {len(virtual_devices)} virtual device(s) available")
            return True

        except Exception as e:
            self.logger.error(f"Failed to enable dual-stream routing: {e}")
            return False

    async def disable_dual_stream_routing(self) -> bool:
        """
        Disable dual-stream routing and fall back to monitor-only output.

        Returns:
            bool: True if dual-stream routing was disabled successfully, False otherwise
        """
        try:
            if not self.audio_manager:
                self.logger.error("Cannot disable dual-stream routing: AudioManager not available")
                return False

            # Disable dual-stream routing in AudioManager
            if hasattr(self.audio_manager, 'dual_stream_enabled'):
                self.audio_manager.dual_stream_enabled = False

            self.logger.info("Dual-stream routing disabled, TTS will use monitor-only output")
            return True

        except Exception as e:
            self.logger.error(f"Failed to disable dual-stream routing: {e}")
            return False

    async def get_virtual_microphone_status(self) -> Dict[str, Any]:
        """
        Get the current status of virtual microphone configuration.

        Returns:
            Dict[str, Any]: Virtual microphone status information
        """
        try:
            status = {
                "audio_manager_available": self.audio_manager is not None,
                "dual_stream_enabled": False,
                "virtual_devices_available": 0,
                "virtual_devices": [],
                "error": None
            }

            if not self.audio_manager:
                status["error"] = "AudioManager not available"
                return status

            # Check dual-stream status
            status["dual_stream_enabled"] = (
                hasattr(self.audio_manager, 'dual_stream_enabled') and
                self.audio_manager.dual_stream_enabled
            )

            # Get virtual devices
            virtual_devices = await self.audio_manager.get_virtual_input_devices_async()
            status["virtual_devices_available"] = len(virtual_devices)
            status["virtual_devices"] = [
                {
                    "name": device.name,
                    "device_id": device.device_id,
                    "is_default": device.is_default
                }
                for device in virtual_devices
            ]

            return status

        except Exception as e:
            self.logger.error(f"Failed to get virtual microphone status: {e}")
            return {
                "audio_manager_available": self.audio_manager is not None,
                "dual_stream_enabled": False,
                "virtual_devices_available": 0,
                "virtual_devices": [],
                "error": str(e)
            }

    async def set_preferred_virtual_device(self, device_id: Optional[str]) -> bool:
        """
        Set the preferred virtual microphone device for dual routing.

        Args:
            device_id: Device ID of the preferred virtual microphone, or None to use default

        Returns:
            bool: True if the preferred device was set successfully, False otherwise
        """
        try:
            if not self.audio_manager:
                self.logger.error("Cannot set preferred virtual device: AudioManager not available")
                return False

            # Validate the device ID if provided
            if device_id:
                virtual_devices = await self.audio_manager.get_virtual_input_devices_async()
                device_exists = any(device.device_id == device_id for device in virtual_devices)

                if not device_exists:
                    self.logger.warning(f"Virtual device ID not found: {device_id}. This may indicate the device was disconnected or device IDs have changed.")
                    # Clear the invalid device ID to prevent repeated errors
                    self._preferred_virtual_device_id = None
                    return False

            # Store the preferred device ID (could be extended to AudioManager if needed)
            self._preferred_virtual_device_id = device_id

            if device_id:
                self.logger.info(f"Preferred virtual microphone device set to: {device_id}")
            else:
                self.logger.info("Preferred virtual microphone device reset to default")

            return True

        except Exception as e:
            self.logger.error(f"Failed to set preferred virtual device: {e}")
            return False

    async def validate_virtual_microphone_setup(self) -> tuple[bool, Optional[str]]:
        """
        Validate the virtual microphone setup for dual routing.

        Returns:
            tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            if not self.audio_manager:
                return False, "AudioManager not available"

            # Check if virtual devices exist
            virtual_devices = await self.audio_manager.get_virtual_input_devices_async()
            if not virtual_devices:
                return False, "No virtual microphone devices found"

            # Check if preferred device is valid (if set)
            if self._preferred_virtual_device_id:
                device_exists = any(
                    device.device_id == self._preferred_virtual_device_id
                    for device in virtual_devices
                )
                if not device_exists:
                    return False, f"Preferred virtual device '{self._preferred_virtual_device_id}' not found"

            # Check if monitor devices exist
            output_devices = await self.audio_manager.get_output_devices()
            if not output_devices:
                return False, "No monitor/output devices found"

            # Validate dual-stream capability
            if hasattr(self.audio_manager, 'dual_stream_enabled'):
                if self.audio_manager.dual_stream_enabled:
                    self.logger.info(f"Virtual microphone setup validated: {len(virtual_devices)} virtual device(s), {len(output_devices)} output device(s)")
                    return True, None
                else:
                    return False, "Dual-stream routing is disabled"

            return True, None

        except Exception as e:
            error_msg = f"Failed to validate virtual microphone setup: {e}"
            self.logger.error(error_msg)
            return False, error_msg

    def _get_voice_transcription(
        self,
        voice_profile: VoiceProfile,
        fallback_text: Optional[str] = None
    ) -> Optional[str]:
        """
        Get transcription text from voice profile with fallback handling.

        Args:
            voice_profile: Voice profile to get transcription from
            fallback_text: Fallback text if transcription is unavailable

        Returns:
            Optional[str]: Transcription text or None if unavailable
        """
        try:
            # Check if voice profile has a valid transcription
            if self._is_transcription_valid(voice_profile):
                self.logger.debug(f"Using voice profile transcription for {voice_profile.name}")
                return voice_profile.transcription

            # Check transcription status for specific handling
            if voice_profile.transcription_status == TranscriptionStatus.NOT_STARTED:
                self.logger.info(f"Transcription not started for {voice_profile.name}, using fallback")
            elif voice_profile.transcription_status == TranscriptionStatus.FAILED:
                self.logger.warning(f"Transcription failed for {voice_profile.name}, using fallback")
            elif voice_profile.transcription_status == TranscriptionStatus.PROCESSING:
                self.logger.warning(f"Transcription still processing for {voice_profile.name}, using fallback")
            else:
                self.logger.warning(f"Transcription status unknown for {voice_profile.name}: {voice_profile.transcription_status}")

            # Use fallback text if provided
            if fallback_text and fallback_text.strip():
                self.logger.info(f"Using fallback transcription for {voice_profile.name}")
                return fallback_text.strip()

            # No transcription available
            self.logger.warning(f"No transcription or fallback available for {voice_profile.name}")
            return None

        except Exception as e:
            self.logger.exception(f"Error getting voice transcription: {e}")
            # Return fallback text if available, otherwise None
            return fallback_text.strip() if fallback_text and fallback_text.strip() else None

    def _is_transcription_valid(self, voice_profile: VoiceProfile) -> bool:
        """
        Check if a voice profile has a valid transcription.

        Args:
            voice_profile: Voice profile to check

        Returns:
            bool: True if transcription is valid and usable
        """
        try:
            # Check if transcription status is completed
            if voice_profile.transcription_status != TranscriptionStatus.COMPLETED:
                return False

            # Check if transcription text exists and is not empty
            if not voice_profile.transcription or not voice_profile.transcription.strip():
                return False

            # Check transcription length (very short transcriptions may be corrupted)
            transcription_text = voice_profile.transcription.strip()
            if len(transcription_text) < 3:  # Minimum reasonable length
                self.logger.warning(f"Transcription too short for {voice_profile.name}: '{transcription_text}'")
                return False

            # Check for common corruption indicators
            if self._is_transcription_corrupted(transcription_text):
                self.logger.warning(f"Transcription appears corrupted for {voice_profile.name}")
                return False

            return True

        except Exception as e:
            self.logger.exception(f"Error validating transcription for {voice_profile.name}: {e}")
            return False

    def _is_transcription_corrupted(self, transcription_text: str) -> bool:
        """
        Check if transcription text appears to be corrupted.

        Args:
            transcription_text: Transcription text to check

        Returns:
            bool: True if transcription appears corrupted
        """
        try:
            # Check for excessive repetition (common Whisper failure mode)
            words = transcription_text.lower().split()
            if len(words) > 3:
                # Check if more than 50% of words are the same
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1

                max_count = max(word_counts.values())
                if max_count > len(words) * 0.5:
                    return True

            # Check for known corruption patterns
            corruption_patterns = [
                "thank you for watching",  # Common Whisper hallucination
                "thank you for listening",
                "music playing",
                "[music]",
                "[silence]",
                "please subscribe",
                "like and subscribe"
            ]

            text_lower = transcription_text.lower()
            for pattern in corruption_patterns:
                if pattern in text_lower:
                    return True

            # Check for excessive special characters
            special_char_count = sum(1 for c in transcription_text if not c.isalnum() and not c.isspace())
            if special_char_count > len(transcription_text) * 0.3:  # More than 30% special characters
                return True

            return False

        except Exception as e:
            self.logger.exception(f"Error checking transcription corruption: {e}")
            return False  # Assume not corrupted if we can't check

    def validate_transcription_quality(
        self,
        voice_profile: VoiceProfile,
        min_confidence: float = 0.7,
        min_length: int = 5
    ) -> tuple[bool, Optional[str]]:
        """
        Validate transcription quality for TTS usage.

        Args:
            voice_profile: Voice profile to validate
            min_confidence: Minimum confidence score (0.0-1.0)
            min_length: Minimum transcription length

        Returns:
            tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            if not self._is_transcription_valid(voice_profile):
                return False, "Transcription is not available or invalid"

            transcription = voice_profile.transcription.strip()

            # Check minimum length
            if len(transcription) < min_length:
                return False, f"Transcription too short ({len(transcription)} chars, minimum {min_length})"

            # Check confidence if available
            if (voice_profile.transcription_confidence is not None and
                voice_profile.transcription_confidence < min_confidence):
                return False, f"Transcription confidence too low ({voice_profile.transcription_confidence:.2f}, minimum {min_confidence:.2f})"

            # Check for corruption
            if self._is_transcription_corrupted(transcription):
                return False, "Transcription appears to be corrupted"

            # All checks passed
            return True, None

        except Exception as e:
            error_msg = f"Error validating transcription quality: {e}"
            self.logger.exception(error_msg)
            return False, error_msg

    # New methods for AudioCoordinator integration

    async def _execute_dual_service_playback(self, response: TTSResponse, request: TTSRequest) -> None:
        """
        Execute dual-service playback using both MonitorAudioService and VirtualMicrophoneService.

        Args:
            response: TTS response with audio data
            request: Original TTS request for context
        """
        try:
            # Use AudioCoordinator for proper dual-service routing
            dual_result = await self.audio_coordinator.play_dual_stream(
                audio_data=response.audio_data,
                monitor_device=None,  # Use service defaults
                virtual_device=None,  # Use service defaults
                volume=1.0
            )

            if dual_result.success:
                if dual_result.both_successful:
                    self.logger.info(f"Dual-stream playback successful: coordination={dual_result.coordination_id}")
                elif dual_result.any_successful:
                    self.logger.info(f"Partial dual-stream success: coordination={dual_result.coordination_id}")
                else:
                    self.logger.warning(f"Dual-stream playback failed: {dual_result.error_message}")
            else:
                self.logger.error(f"Dual-stream coordination failed: {dual_result.error_message}")

            # Store coordination info for potential cancellation
            if not hasattr(self, '_active_coordinations'):
                self._active_coordinations = {}
            self._active_coordinations[dual_result.coordination_id] = dual_result

        except Exception as e:
            self.logger.error(f"Dual-service playback execution failed: {e}")
            raise

    async def _execute_monitor_only_coordinated(self, response: TTSResponse, request: TTSRequest) -> None:
        """
        Execute monitor-only playback using AudioCoordinator.

        Args:
            response: TTS response with audio data
            request: Original TTS request for context
        """
        try:
            monitor_task = await self.audio_coordinator.play_monitor_only(
                audio_data=response.audio_data,
                device=None,  # Use service default
                volume=1.0
            )

            if monitor_task:
                self.logger.info(f"Monitor-only playback started: {monitor_task.playback_id}")
            else:
                self.logger.error("Monitor-only playback failed to start")

        except Exception as e:
            self.logger.error(f"Monitor-only coordinated playback failed: {e}")
            raise

    async def _execute_virtual_only_coordinated(self, response: TTSResponse, request: TTSRequest) -> None:
        """
        Execute virtual-only playback using AudioCoordinator.

        Args:
            response: TTS response with audio data
            request: Original TTS request for context
        """
        try:
            virtual_task = await self.audio_coordinator.play_virtual_only(
                audio_data=response.audio_data,
                device=None,  # Use service default
                volume=1.0
            )

            if virtual_task:
                self.logger.info(f"Virtual-only playback started: {virtual_task.playback_id}")
            else:
                self.logger.error("Virtual-only playback failed to start")

        except Exception as e:
            self.logger.error(f"Virtual-only coordinated playback failed: {e}")
            raise

    async def _execute_legacy_audio_routing(self, response: TTSResponse, request: TTSRequest) -> None:
        """
        Execute audio routing using legacy AudioManager (for backward compatibility).

        Args:
            response: TTS response with audio data
            request: Original TTS request for context
        """
        try:
            self.logger.warning("Using legacy AudioManager - consider upgrading to AudioCoordinator")

            if not self.audio_manager:
                self.logger.error("AudioManager not available for legacy routing")
                return

            # Use legacy monitor-only approach (avoid play_synchronized_dual_stream)
            monitor_task = await self.audio_manager.play_monitor_audio(
                audio_data=response.audio_data,
                device=None,  # Use default device
                volume=1.0
            )

            if monitor_task:
                self.logger.info(f"Legacy monitor playback started: {monitor_task.playback_id}")
            else:
                self.logger.error("Legacy monitor playback failed")

        except Exception as e:
            self.logger.error(f"Legacy audio routing failed: {e}")
            raise

    def get_audio_coordinator(self):
        """
        Get audio coordinator instance.

        Returns:
            AudioCoordinator instance or None
        """
        return self.audio_coordinator