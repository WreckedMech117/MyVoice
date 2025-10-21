"""
Validation utilities for MyVoice services.
"""

import functools
import struct
from typing import Callable, Any

from myvoice.services.core.base_service import ServiceStatus
from myvoice.models.error import MyVoiceError, ErrorSeverity


def require_service_health(func: Callable) -> Callable:
    """
    Decorator to ensure service is running before executing method.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function that checks service health

    Raises:
        MyVoiceError: If service is not running
    """
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs) -> Any:
        if not hasattr(self, 'status') or self.status != ServiceStatus.RUNNING:
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="SERVICE_NOT_RUNNING",
                user_message=f"{getattr(self, 'service_name', 'Service')} is not running",
                technical_details=f"Service status: {getattr(self, 'status', 'unknown')}",
                suggested_action="Start the service before using it"
            )

        return await func(self, *args, **kwargs)

    return wrapper


def validate_wav_data(audio_data: bytes) -> bool:
    """
    Validate that audio data is a valid WAV file.

    Args:
        audio_data: Audio data to validate

    Returns:
        bool: True if data appears to be valid WAV format
    """
    try:
        if len(audio_data) < 44:  # Minimum WAV header size
            return False

        # Check RIFF header
        if audio_data[:4] != b'RIFF':
            return False

        # Check WAVE format
        if audio_data[8:12] != b'WAVE':
            return False

        # Check fmt chunk
        if audio_data[12:16] != b'fmt ':
            return False

        # Basic validation passed
        return True

    except Exception:
        return False