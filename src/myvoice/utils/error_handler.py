"""
Error Handler Utility

Story 7.6: Centralized error handling with user-friendly messages.
Implements NFR6 (stability) and NFR10 (user-friendly error messages).

Provides:
- ErrorMessageFormatter: Formats errors with "[What happened] + [What to do]" pattern
- Global exception handler to prevent crashes
- Error logging infrastructure
"""

import logging
import sys
import traceback
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from enum import Enum

from myvoice.models.error import MyVoiceError, ErrorSeverity


class ErrorCode(Enum):
    """Standard error codes for consistent error handling."""

    # TTS Errors
    TTS_GENERATION_FAILED = "TTS_GENERATION_FAILED"
    TTS_MODEL_LOAD_FAILED = "TTS_MODEL_LOAD_FAILED"
    TTS_MODEL_MEMORY_ERROR = "TTS_MODEL_MEMORY_ERROR"
    TTS_STREAMING_FAILED = "TTS_STREAMING_FAILED"

    # Audio Errors
    AUDIO_DEVICE_UNAVAILABLE = "AUDIO_DEVICE_UNAVAILABLE"
    AUDIO_PLAYBACK_FAILED = "AUDIO_PLAYBACK_FAILED"
    AUDIO_FORMAT_INVALID = "AUDIO_FORMAT_INVALID"

    # Voice Errors
    VOICE_FILE_CORRUPTED = "VOICE_FILE_CORRUPTED"
    VOICE_NOT_FOUND = "VOICE_NOT_FOUND"
    VOICE_CLONE_FAILED = "VOICE_CLONE_FAILED"

    # Settings Errors
    SETTINGS_LOAD_FAILED = "SETTINGS_LOAD_FAILED"
    SETTINGS_SAVE_FAILED = "SETTINGS_SAVE_FAILED"

    # General Errors
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    NETWORK_ERROR = "NETWORK_ERROR"


# Story 7.6: User-friendly error message templates
# Format: "what" (what happened) + "action" (what to do)
ERROR_MESSAGE_TEMPLATES: Dict[str, Dict[str, str]] = {
    # TTS Errors
    ErrorCode.TTS_GENERATION_FAILED.value: {
        "what": "Speech generation failed",
        "action": "Please try again.",
        "severity": ErrorSeverity.ERROR
    },
    ErrorCode.TTS_MODEL_LOAD_FAILED.value: {
        "what": "Could not load the voice model",
        "action": "Try restarting MyVoice. If the problem continues, reinstall the application.",
        "severity": ErrorSeverity.CRITICAL
    },
    ErrorCode.TTS_MODEL_MEMORY_ERROR.value: {
        "what": "Not enough memory to load voice model",
        "action": "Try closing other applications and restart MyVoice.",
        "severity": ErrorSeverity.ERROR
    },
    ErrorCode.TTS_STREAMING_FAILED.value: {
        "what": "Audio streaming interrupted",
        "action": "Please try again. The speech will be generated in batch mode.",
        "severity": ErrorSeverity.WARNING
    },

    # Audio Errors
    ErrorCode.AUDIO_DEVICE_UNAVAILABLE.value: {
        "what": "Audio device '{device}' is not available",
        "action": "Please select a different device in Settings.",
        "severity": ErrorSeverity.ERROR
    },
    ErrorCode.AUDIO_PLAYBACK_FAILED.value: {
        "what": "Could not play audio",
        "action": "Check your audio settings and try again.",
        "severity": ErrorSeverity.ERROR
    },
    ErrorCode.AUDIO_FORMAT_INVALID.value: {
        "what": "The audio format is not supported",
        "action": "Try a different audio file (WAV, MP3, or M4A).",
        "severity": ErrorSeverity.ERROR
    },

    # Voice Errors
    ErrorCode.VOICE_FILE_CORRUPTED.value: {
        "what": "Could not load voice '{voice}'",
        "action": "The voice file may be corrupted. Try recreating the voice.",
        "severity": ErrorSeverity.ERROR
    },
    ErrorCode.VOICE_NOT_FOUND.value: {
        "what": "Voice '{voice}' was not found",
        "action": "The voice may have been deleted. Please select a different voice.",
        "severity": ErrorSeverity.WARNING
    },
    ErrorCode.VOICE_CLONE_FAILED.value: {
        "what": "Voice cloning failed",
        "action": "Try using a clearer audio sample with less background noise.",
        "severity": ErrorSeverity.ERROR
    },

    # Settings Errors
    ErrorCode.SETTINGS_LOAD_FAILED.value: {
        "what": "Could not load your settings",
        "action": "Default settings will be used. Your preferences may need to be reconfigured.",
        "severity": ErrorSeverity.WARNING
    },
    ErrorCode.SETTINGS_SAVE_FAILED.value: {
        "what": "Could not save your settings",
        "action": "Check that MyVoice has permission to write to the settings folder.",
        "severity": ErrorSeverity.ERROR
    },

    # General Errors
    ErrorCode.UNEXPECTED_ERROR.value: {
        "what": "An unexpected error occurred",
        "action": "Please try again. If the problem continues, restart MyVoice.",
        "severity": ErrorSeverity.ERROR
    },
    ErrorCode.PERMISSION_DENIED.value: {
        "what": "Permission denied",
        "action": "Check that MyVoice has permission to access the required files.",
        "severity": ErrorSeverity.ERROR
    },
    ErrorCode.FILE_NOT_FOUND.value: {
        "what": "The requested file was not found",
        "action": "The file may have been moved or deleted.",
        "severity": ErrorSeverity.WARNING
    },
    ErrorCode.NETWORK_ERROR.value: {
        "what": "Network connection failed",
        "action": "Check your internet connection and try again.",
        "severity": ErrorSeverity.ERROR
    },
}


class ErrorMessageFormatter:
    """
    Formats error messages with user-friendly patterns.

    Story 7.6: All errors follow the pattern "[What happened] + [What to do]"
    to help users understand and resolve issues without technical knowledge.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._templates = ERROR_MESSAGE_TEMPLATES.copy()

    def format_error(
        self,
        error_code: str,
        **kwargs
    ) -> str:
        """
        Format an error message using the standard template.

        Args:
            error_code: Error code from ErrorCode enum
            **kwargs: Variables to substitute in the message (e.g., device="Speakers")

        Returns:
            Formatted user-friendly error message
        """
        template = self._templates.get(error_code)
        if not template:
            # Fallback to generic error
            template = self._templates[ErrorCode.UNEXPECTED_ERROR.value]
            self.logger.warning(f"Unknown error code: {error_code}")

        what = template["what"]
        action = template["action"]

        # Substitute variables in the message
        try:
            what = what.format(**kwargs)
            action = action.format(**kwargs)
        except KeyError as e:
            self.logger.warning(f"Missing variable in error message: {e}")

        # Combine into final message: "[What happened]. [What to do]"
        return f"{what}. {action}"

    def format_myvoice_error(self, error: MyVoiceError) -> str:
        """
        Format a MyVoiceError into a user-friendly message.

        Args:
            error: MyVoiceError instance

        Returns:
            Formatted user-friendly error message
        """
        # If error has suggested_action, use it
        if error.suggested_action:
            return f"{error.user_message}. {error.suggested_action}"

        # Otherwise, just return the user message
        return error.user_message

    def get_severity(self, error_code: str) -> ErrorSeverity:
        """
        Get the severity level for an error code.

        Args:
            error_code: Error code from ErrorCode enum

        Returns:
            ErrorSeverity level
        """
        template = self._templates.get(error_code)
        if template and "severity" in template:
            return template["severity"]
        return ErrorSeverity.ERROR

    def create_error(
        self,
        error_code: str,
        technical_details: Optional[str] = None,
        **kwargs
    ) -> MyVoiceError:
        """
        Create a complete MyVoiceError from an error code.

        Args:
            error_code: Error code from ErrorCode enum
            technical_details: Optional technical details for logging
            **kwargs: Variables to substitute in the message

        Returns:
            MyVoiceError with formatted message
        """
        template = self._templates.get(error_code, self._templates[ErrorCode.UNEXPECTED_ERROR.value])

        what = template["what"]
        action = template["action"]

        # Substitute variables
        try:
            what = what.format(**kwargs)
            action = action.format(**kwargs)
        except KeyError:
            pass

        severity = template.get("severity", ErrorSeverity.ERROR)

        return MyVoiceError(
            severity=severity,
            code=error_code,
            user_message=what,
            suggested_action=action,
            technical_details=technical_details
        )


class GlobalExceptionHandler:
    """
    Global exception handler to prevent application crashes.

    Story 7.6 (NFR6): Application never crashes on errors.
    Catches unhandled exceptions and shows user-friendly messages.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._error_formatter = ErrorMessageFormatter()
        self._error_callback: Optional[Callable[[str, str, str], None]] = None
        self._critical_callback: Optional[Callable[[str, str, str], None]] = None
        self._original_excepthook = None

    def install(self):
        """Install the global exception handler."""
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._handle_exception
        self.logger.info("Global exception handler installed")

    def uninstall(self):
        """Uninstall the global exception handler and restore original."""
        if self._original_excepthook:
            sys.excepthook = self._original_excepthook
            self._original_excepthook = None
            self.logger.info("Global exception handler uninstalled")

    def set_error_callback(self, callback: Callable[[str, str, str], None]):
        """
        Set callback for displaying non-critical errors.

        Args:
            callback: Function that receives (title, message, details)
        """
        self._error_callback = callback

    def set_critical_callback(self, callback: Callable[[str, str, str], None]):
        """
        Set callback for displaying critical errors.

        Args:
            callback: Function that receives (title, message, details)
        """
        self._critical_callback = callback

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """
        Handle uncaught exceptions.

        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        # Don't intercept KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            if self._original_excepthook:
                self._original_excepthook(exc_type, exc_value, exc_traceback)
            return

        # Log the full error with traceback
        error_time = datetime.now().isoformat()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = "".join(tb_lines)

        self.logger.error(
            f"Uncaught exception at {error_time}:\n"
            f"Type: {exc_type.__name__}\n"
            f"Value: {exc_value}\n"
            f"Traceback:\n{tb_text}"
        )

        # Prepare user-friendly message
        if isinstance(exc_value, MyVoiceError):
            title = "Error"
            message = self._error_formatter.format_myvoice_error(exc_value)
            details = exc_value.technical_details or str(exc_value)
            is_critical = exc_value.severity == ErrorSeverity.CRITICAL
        else:
            title = "Unexpected Error"
            message = self._error_formatter.format_error(ErrorCode.UNEXPECTED_ERROR.value)
            details = f"{exc_type.__name__}: {exc_value}\n\n{tb_text}"
            is_critical = issubclass(exc_type, (MemoryError, SystemError))

        # Show appropriate dialog
        try:
            if is_critical and self._critical_callback:
                self._critical_callback(title, message, details)
            elif self._error_callback:
                self._error_callback(title, message, details)
            else:
                # Fallback: print to stderr
                print(f"\n[{title}] {message}\n", file=sys.stderr)
        except Exception as dialog_error:
            self.logger.error(f"Failed to show error dialog: {dialog_error}")
            print(f"\n[{title}] {message}\n", file=sys.stderr)


# Global instances
_error_formatter: Optional[ErrorMessageFormatter] = None
_exception_handler: Optional[GlobalExceptionHandler] = None


def get_error_formatter() -> ErrorMessageFormatter:
    """Get the global ErrorMessageFormatter instance."""
    global _error_formatter
    if _error_formatter is None:
        _error_formatter = ErrorMessageFormatter()
    return _error_formatter


def get_exception_handler() -> GlobalExceptionHandler:
    """Get the global GlobalExceptionHandler instance."""
    global _exception_handler
    if _exception_handler is None:
        _exception_handler = GlobalExceptionHandler()
    return _exception_handler


def format_error(error_code: str, **kwargs) -> str:
    """Convenience function to format an error message."""
    return get_error_formatter().format_error(error_code, **kwargs)


def create_error(error_code: str, technical_details: Optional[str] = None, **kwargs) -> MyVoiceError:
    """Convenience function to create a MyVoiceError."""
    return get_error_formatter().create_error(error_code, technical_details, **kwargs)
