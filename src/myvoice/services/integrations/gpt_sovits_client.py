"""
GPT-SoVITS HTTP Client Integration

This module provides HTTP client functionality for communicating with the GPT-SoVITS
text-to-speech service. It handles multipart/form-data requests, binary audio responses,
and comprehensive error handling.
"""

import logging
import requests
from pathlib import Path
from typing import Optional, Tuple, Callable
from requests.exceptions import (
    ConnectionError, Timeout, RequestException, HTTPError
)

from myvoice.models.tts_request import TTSRequest, TTSResponse
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.retry_config import RetryConfig, RetryConfigs, RetryAttempt
from myvoice.utils.retry_handler import RetryHandler


class GPTSoVITSClient:
    """
    HTTP client for GPT-SoVITS text-to-speech service.

    This client handles communication with a GPT-SoVITS service running on localhost,
    providing text-to-speech generation with voice cloning capabilities.

    Attributes:
        base_url: Base URL for the GPT-SoVITS service
        timeout: Request timeout in seconds
        logger: Logger instance for this client
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9880,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        retry_callback: Optional[Callable[[RetryAttempt], None]] = None
    ):
        """
        Initialize the GPT-SoVITS client.

        Args:
            host: GPT-SoVITS service host (default: 127.0.0.1)
            port: GPT-SoVITS service port (default: 9880)
            timeout: Request timeout in seconds (default: 30.0)
            retry_config: Retry configuration (default: STANDARD)
            retry_callback: Optional callback for retry attempts
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)

        # Retry configuration
        self.retry_config = retry_config or RetryConfigs.STANDARD
        self.retry_config.request_timeout = timeout  # Sync with client timeout

        # Set up retry callback for user feedback
        if retry_callback:
            self.retry_config.on_retry_callback = retry_callback

        # Initialize retry handler
        self.retry_handler = RetryHandler(self.retry_config, self.logger)

        # Session for connection pooling and performance
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MyVoice/1.0 GPT-SoVITS Client"
        })

        self.logger.debug(f"GPT-SoVITS client initialized for {self.base_url} with retry config")
        self.logger.debug(f"Retry config: {self.retry_config.max_attempts} attempts, "
                         f"{self.retry_config.initial_delay}s initial delay")

    def generate_speech(self, request: TTSRequest) -> TTSResponse:
        """
        Generate speech from text using voice cloning with retry logic.

        Args:
            request: TTS request containing text and voice information

        Returns:
            TTSResponse: Response containing audio data or error information
        """
        try:
            self.logger.info(f"Generating speech for text: {request.text[:50]}...")

            # Validate request (no retry for validation errors)
            self._validate_request(request)

            # Execute TTS generation with retry logic
            def _generate_with_retry():
                # Prepare JSON payload for API v2
                payload = self._prepare_request_data(request)

                # Make HTTP request (this is the part that can fail and be retried)
                response = self._make_request(payload)

                # Process response
                return self._process_response(response)

            # Execute with retry logic
            return self.retry_handler.execute(_generate_with_retry)

        except MyVoiceError:
            # MyVoiceError from retry handler - convert to TTSResponse
            raise
        except Exception as e:
            self.logger.exception(f"Error during speech generation: {e}")
            return TTSResponse(
                success=False,
                error_message=str(e)
            )

    def health_check(self) -> Tuple[bool, Optional[MyVoiceError]]:
        """
        Check if the GPT-SoVITS service is available and responding with retry logic.

        Returns:
            Tuple[bool, Optional[MyVoiceError]]: (is_healthy, error_if_any)
        """
        try:
            self.logger.debug("Performing health check on GPT-SoVITS service")

            # Use a lighter retry config for health checks
            health_check_config = RetryConfig(
                max_attempts=2,  # Only 2 attempts for health checks
                initial_delay=0.5,
                max_delay=5.0,
                exponential_base=2.0,
                jitter=False
            )
            health_check_handler = RetryHandler(health_check_config, self.logger)

            def _health_check_with_retry():
                # Try to connect to the base URL
                response = self.session.get(
                    self.base_url,
                    timeout=5.0  # Shorter timeout for health checks
                )

                # Accept any HTTP response as healthy (200, 404, etc.)
                # What matters is that the service is responding
                if 200 <= response.status_code < 500:
                    self.logger.debug(f"GPT-SoVITS service is healthy (status: {response.status_code})")
                    return True, None
                else:
                    error = MyVoiceError(
                        severity=ErrorSeverity.WARNING,
                        code="SERVICE_UNHEALTHY",
                        user_message="GPT-SoVITS service returned server error",
                        technical_details=f"Status code: {response.status_code}",
                        suggested_action="Check GPT-SoVITS service logs for server errors"
                    )
                    return False, error

            # Execute with limited retry logic
            return health_check_handler.execute(_health_check_with_retry)

        except MyVoiceError as e:
            # Already a structured error
            return False, e

        except ConnectionError:
            error = MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="SERVICE_UNAVAILABLE",
                user_message="Cannot connect to GPT-SoVITS service",
                technical_details="Connection refused",
                suggested_action="Start the GPT-SoVITS service and ensure it's running on the correct port"
            )
            return False, error

        except Timeout:
            error = MyVoiceError(
                severity=ErrorSeverity.WARNING,
                code="SERVICE_TIMEOUT",
                user_message="GPT-SoVITS service is not responding",
                technical_details="Health check timeout",
                suggested_action="Check if the service is overloaded or restart it"
            )
            return False, error

        except Exception as e:
            error = MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="HEALTH_CHECK_FAILED",
                user_message="Failed to check GPT-SoVITS service status",
                technical_details=str(e),
                suggested_action="Check network connectivity and service configuration"
            )
            return False, error

    def _validate_request(self, request: TTSRequest) -> None:
        """
        Validate TTS request parameters.

        Args:
            request: TTS request to validate

        Raises:
            ValueError: If request parameters are invalid
        """
        if not request.text.strip():
            raise ValueError("Text cannot be empty")

        if not request.voice_file_path.exists():
            raise FileNotFoundError(f"Voice file not found: {request.voice_file_path}")

        if request.voice_file_path.stat().st_size == 0:
            raise ValueError("Voice file is empty")

        # Check file size (25MB limit based on research)
        max_size = 25 * 1024 * 1024  # 25MB in bytes
        if request.voice_file_path.stat().st_size > max_size:
            raise ValueError(f"Voice file too large. Maximum size: {max_size / (1024*1024):.1f}MB")

        self.logger.debug("Request validation passed")

    def _prepare_request_data(self, request: TTSRequest) -> dict:
        """
        Prepare JSON payload for GPT-SoVITS API v2.

        Args:
            request: TTS request

        Returns:
            dict: JSON payload for the request
        """
        import re

        # Clean text to handle emojis and special characters more carefully
        # First, ensure text is properly encoded as UTF-8
        try:
            # Encode and decode to ensure clean UTF-8
            text_bytes = request.text.encode('utf-8', errors='ignore')
            text_utf8 = text_bytes.decode('utf-8')

            # Remove problematic characters that cause encoding issues
            # Keep most Unicode characters but remove specific problematic ones
            import unicodedata
            cleaned_text = ''.join(
                char for char in text_utf8
                if unicodedata.category(char) not in ['Cc', 'Cf', 'Cs', 'Co', 'Cn']  # Remove control chars
                and ord(char) != 0xac  # Remove the specific problematic character from your error
            )

            # If we removed too much, try a more lenient approach
            if not cleaned_text.strip():
                # Only remove emojis and extreme control characters
                cleaned_text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u200d\u2640-\u2642\u2695\u2696\u2708\u2709\u270A-\u270D\u270F]', '', text_utf8)

            # Final fallback if all text was removed
            if not cleaned_text.strip():
                cleaned_text = "Hello there!"

        except Exception as encoding_error:
            self.logger.warning(f"Text encoding issue: {encoding_error}, using fallback")
            cleaned_text = "Hello there!"

        # Get emotion parameters from request (defaults to neutral if not set)
        gpt_sovits_params = request.get_gpt_sovits_parameters()

        # Log emotion parameters being used
        emotion_params = request.get_emotion_parameters()
        self.logger.debug(f"Using emotion parameters: {emotion_params.name} "
                         f"(temp={emotion_params.temperature}, top_p={emotion_params.top_p}, "
                         f"rep_penalty={emotion_params.repetition_penalty})")

        # Prepare JSON payload for API v2
        payload = {
            "text": cleaned_text,
            "text_lang": request.text_lang,  # Use configured language
            "ref_audio_path": str(request.voice_file_path.absolute()),  # Use absolute path
            "prompt_text": request.voice_text or "",  # Use voice_text as prompt_text
            "prompt_lang": request.prompt_lang,  # Use configured prompt language
            "text_split_method": "cut5",  # Required for API v2
            "batch_size": 1,              # Required for API v2
            "streaming_mode": False,      # Required for API v2
            "speed_factor": gpt_sovits_params["speed_factor"],  # Use emotion-aware speed
            "top_k": 5,                   # API v2 default
            "top_p": gpt_sovits_params["top_p"],              # Use emotion-aware top_p
            "temperature": gpt_sovits_params["temperature"],   # Use emotion-aware temperature
            "repetition_penalty": gpt_sovits_params["repetition_penalty"],  # Use emotion-aware repetition penalty
            "sample_steps": 32,           # API v2 default
            "super_sampling": False       # API v2 default
        }

        self.logger.debug(f"Prepared API v2 JSON payload with {len(payload)} fields")
        return payload

    def _make_request(self, payload: dict) -> requests.Response:
        """
        Make HTTP POST request to GPT-SoVITS service using JSON payload.

        Args:
            payload: JSON payload for the request

        Returns:
            requests.Response: HTTP response

        Raises:
            ConnectionError: If service is unavailable
            Timeout: If request times out
            HTTPError: For HTTP error status codes
        """
        url = f"{self.base_url}/tts"

        try:
            self.logger.debug(f"Making JSON request to {url}")
            # Log payload without sensitive data for debugging
            safe_payload = payload.copy()
            if 'ref_audio_path' in safe_payload:
                safe_payload['ref_audio_path'] = '...' + safe_payload['ref_audio_path'][-30:]  # Show only last 30 chars
            self.logger.debug(f"Payload: {safe_payload}")

            # Set proper headers for JSON
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'audio/wav, application/json'
            }

            response = self.session.post(
                url,
                json=payload,  # Use json parameter for proper encoding
                headers=headers,
                timeout=self.timeout
            )

            # Check for HTTP errors
            response.raise_for_status()

            self.logger.debug(f"Request successful, status: {response.status_code}")
            return response

        except ConnectionError as e:
            self.logger.error(f"Connection error: {e}")
            raise ConnectionError(
                "Cannot connect to GPT-SoVITS service. "
                "Please ensure the service is running and accessible."
            )

        except Timeout as e:
            self.logger.error(f"Request timeout: {e}")
            raise Timeout(
                f"Request timed out after {self.timeout} seconds. "
                "The service may be overloaded or processing may be taking too long."
            )

        except HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
            # Try to get more detailed error information
            error_details = "Unknown error"
            if e.response:
                try:
                    if 'application/json' in e.response.headers.get('content-type', ''):
                        error_json = e.response.json()
                        error_details = error_json.get('detail', str(error_json))
                    else:
                        error_details = e.response.text[:200]  # First 200 chars
                except Exception:
                    error_details = f"HTTP {e.response.status_code}"

            if e.response and hasattr(e.response, 'status_code'):
                if e.response.status_code == 400:
                    raise HTTPError(
                        f"Bad request: {error_details}. "
                        "Check that the reference audio file path is accessible by the server "
                        "and the text is properly formatted."
                    )
                elif e.response.status_code == 422:
                    raise HTTPError(
                        f"Validation error: {error_details}. "
                        "Check the payload format and required fields."
                    )
                elif e.response.status_code == 500:
                    raise HTTPError(
                        f"GPT-SoVITS service encountered an internal error: {error_details}. "
                        "Check the service logs for details."
                    )
                else:
                    raise HTTPError(f"HTTP {e.response.status_code}: {error_details}")
            else:
                raise HTTPError(f"HTTP error: {str(e)}")

    def _process_response(self, response: requests.Response) -> TTSResponse:
        """
        Process HTTP response from GPT-SoVITS service with proper encoding handling.

        Args:
            response: HTTP response object

        Returns:
            TTSResponse: Processed response with audio data
        """
        try:
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            self.logger.debug(f"Response content type: {content_type}")

            # Handle different response types
            if 'application/json' in content_type:
                # Error response in JSON format
                try:
                    error_data = response.json()
                    error_message = error_data.get('detail', str(error_data))
                    self.logger.error(f"GPT-SoVITS returned error: {error_message}")
                    return TTSResponse(
                        success=False,
                        error_message=f"Service error: {error_message}"
                    )
                except Exception as json_error:
                    self.logger.error(f"Failed to parse JSON error response: {json_error}")
                    return TTSResponse(
                        success=False,
                        error_message="Service returned an error in unexpected format"
                    )

            # Expect audio response
            if 'audio' not in content_type and 'application/octet-stream' not in content_type:
                self.logger.warning(f"Unexpected content type: {content_type}")

            # Get audio data
            audio_data = response.content

            if not audio_data:
                raise ValueError("Response contains no audio data")

            # Validate audio data (basic WAV header check)
            if len(audio_data) < 44:  # WAV header is 44 bytes
                raise ValueError("Audio data too small to be valid WAV file")

            # Check for WAV file signature
            if not audio_data.startswith(b'RIFF') or b'WAVE' not in audio_data[:12]:
                self.logger.warning("Audio data does not appear to be WAV format")
                # But continue anyway - some services might return other formats

            self.logger.info(f"Successfully received {len(audio_data)} bytes of audio data")

            return TTSResponse(
                audio_data=audio_data,
                content_type=content_type,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            return TTSResponse(
                success=False,
                error_message=f"Failed to process audio response: {str(e)}"
            )

    def set_retry_callback(self, callback: Callable[[RetryAttempt], None]):
        """
        Set callback for retry attempts to provide user feedback.

        Args:
            callback: Function to call on each retry attempt
        """
        self.retry_config.on_retry_callback = callback
        self.logger.debug("Retry callback updated")

    def get_retry_config(self) -> RetryConfig:
        """Get the current retry configuration."""
        return self.retry_config

    def update_retry_config(self, config: RetryConfig):
        """
        Update the retry configuration.

        Args:
            config: New retry configuration
        """
        self.retry_config = config
        self.retry_config.request_timeout = self.timeout  # Keep timeout in sync
        self.retry_handler = RetryHandler(self.retry_config, self.logger)
        self.logger.debug(f"Retry configuration updated: {config.max_attempts} attempts")

    def close(self) -> None:
        """Close the HTTP session and clean up resources."""
        if self.session:
            self.session.close()
            self.logger.debug("GPT-SoVITS client session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()