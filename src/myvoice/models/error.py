"""
MyVoice Error Models

This module contains standardized error response formats for the MyVoice application.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MyVoiceError(Exception):
    """Standardized error response format that can be raised as an exception"""
    severity: ErrorSeverity
    code: str
    user_message: str
    technical_details: Optional[str] = None
    suggested_action: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()

        # Initialize the Exception base class with the user message
        super().__init__(self.user_message)

    @classmethod
    def from_exception(cls, exception: Exception, user_message: str = None) -> 'MyVoiceError':
        """
        Create a MyVoiceError from an exception.

        Args:
            exception: The exception to convert
            user_message: Optional user-friendly message

        Returns:
            MyVoiceError: Standardized error object
        """
        # Default user message if not provided
        if user_message is None:
            user_message = "An unexpected error occurred"

        # Determine severity based on exception type
        from requests.exceptions import ConnectionError, Timeout, HTTPError

        severity = ErrorSeverity.ERROR
        suggested_action = "Try again later or contact support if the problem persists"

        if isinstance(exception, ConnectionError):
            severity = ErrorSeverity.ERROR
            suggested_action = "Check your internet connection and try again"
        elif isinstance(exception, Timeout):
            severity = ErrorSeverity.WARNING
            suggested_action = "Try again with a smaller request or check your connection"
        elif isinstance(exception, HTTPError):
            severity = ErrorSeverity.ERROR
            suggested_action = "Check the service status and try again"
        elif isinstance(exception, (ValueError, TypeError)):
            severity = ErrorSeverity.ERROR
            suggested_action = "Check your input parameters and try again"

        return cls(
            severity=severity,
            code=exception.__class__.__name__.upper(),
            user_message=user_message,
            technical_details=str(exception),
            suggested_action=suggested_action
        )