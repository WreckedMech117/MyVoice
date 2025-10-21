"""
Retry Configuration Models

This module contains models for configuring retry behavior including
exponential backoff, retry attempts, and error classification.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Union
from enum import Enum
import time
import random
import logging
from requests.exceptions import ConnectionError, Timeout, HTTPError

from myvoice.models.error import MyVoiceError, ErrorSeverity


class RetryableErrorType(Enum):
    """Types of errors that can be retried."""
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT = "rate_limit"
    TEMPORARY_SERVER_ERROR = "temporary_server_error"
    UNKNOWN = "unknown"


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    error: Exception
    delay_seconds: float
    timestamp: float = field(default_factory=time.time)

    @property
    def error_type(self) -> RetryableErrorType:
        """Classify the error type for retry decisions."""
        if isinstance(self.error, ConnectionError):
            return RetryableErrorType.NETWORK_ERROR
        elif isinstance(self.error, Timeout):
            return RetryableErrorType.TIMEOUT
        elif isinstance(self.error, HTTPError):
            if hasattr(self.error, 'response') and self.error.response:
                status_code = self.error.response.status_code
                if status_code == 503:
                    return RetryableErrorType.SERVICE_UNAVAILABLE
                elif status_code == 429:
                    return RetryableErrorType.RATE_LIMIT
                elif 500 <= status_code <= 599:
                    return RetryableErrorType.TEMPORARY_SERVER_ERROR
        return RetryableErrorType.UNKNOWN


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.

    This class defines retry parameters including maximum attempts,
    backoff strategy, and error classification rules.
    """

    # Basic retry parameters
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True

    # Timeout configuration
    request_timeout: float = 30.0
    total_timeout: float = 300.0  # Maximum total time across all retries

    # Error classification
    retryable_errors: Tuple[type, ...] = (ConnectionError, Timeout)
    retryable_status_codes: Tuple[int, ...] = (502, 503, 504, 429)
    permanent_error_codes: Tuple[int, ...] = (400, 401, 403, 404)

    # Callbacks
    on_retry_callback: Optional[Callable[[RetryAttempt], None]] = None
    on_failure_callback: Optional[Callable[[List[RetryAttempt]], None]] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay <= 0:
            raise ValueError("initial_delay must be positive")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if self.exponential_base <= 1:
            raise ValueError("exponential_base must be > 1")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.total_timeout < self.request_timeout:
            raise ValueError("total_timeout must be >= request_timeout")

    def calculate_delay(self, attempt_number: int) -> float:
        """
        Calculate delay for a specific retry attempt.

        Args:
            attempt_number: The retry attempt number (1-based)

        Returns:
            float: Delay in seconds before the retry
        """
        if attempt_number <= 0:
            return 0.0

        # Calculate exponential backoff
        delay = self.initial_delay * (self.exponential_base ** (attempt_number - 1))

        # Apply maximum delay limit
        delay = min(delay, self.max_delay)

        # Add jitter to avoid thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1  # ±10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        # Ensure delay is non-negative
        return max(0.0, delay)

    def is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: The exception that occurred

        Returns:
            bool: True if the error should be retried
        """
        # Check if error type is retryable
        if isinstance(error, self.retryable_errors):
            return True

        # Check HTTP status codes for HTTPError
        if isinstance(error, HTTPError) and hasattr(error, 'response') and error.response:
            status_code = error.response.status_code

            # Permanent errors should not be retried
            if status_code in self.permanent_error_codes:
                return False

            # Retryable status codes
            if status_code in self.retryable_status_codes:
                return True

            # Other 5xx errors are generally retryable
            if 500 <= status_code <= 599:
                return True

        return False

    def get_user_friendly_message(self, attempt: RetryAttempt, total_attempts: int) -> str:
        """
        Get user-friendly message for retry attempt.

        Args:
            attempt: The retry attempt information
            total_attempts: Total number of attempts made

        Returns:
            str: User-friendly message describing the retry
        """
        error_type_messages = {
            RetryableErrorType.NETWORK_ERROR: "connection issue",
            RetryableErrorType.TIMEOUT: "timeout",
            RetryableErrorType.SERVICE_UNAVAILABLE: "service temporarily unavailable",
            RetryableErrorType.RATE_LIMIT: "rate limit exceeded",
            RetryableErrorType.TEMPORARY_SERVER_ERROR: "temporary server error",
            RetryableErrorType.UNKNOWN: "temporary issue"
        }

        error_desc = error_type_messages.get(attempt.error_type, "temporary issue")

        if attempt.attempt_number < self.max_attempts:
            return (f"Retrying due to {error_desc} "
                   f"(attempt {attempt.attempt_number}/{self.max_attempts}, "
                   f"waiting {attempt.delay_seconds:.1f}s)")
        else:
            return f"Failed after {total_attempts} attempts due to {error_desc}"

    def create_retry_error(self, attempts: List[RetryAttempt]) -> MyVoiceError:
        """
        Create a MyVoiceError from failed retry attempts.

        Args:
            attempts: List of all retry attempts

        Returns:
            MyVoiceError: Comprehensive error information
        """
        if not attempts:
            return MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="RETRY_FAILED",
                user_message="Operation failed",
                suggested_action="Try again later"
            )

        last_attempt = attempts[-1]
        error_types = {attempt.error_type for attempt in attempts}

        # Determine primary error type
        if RetryableErrorType.NETWORK_ERROR in error_types:
            primary_error = "Network connectivity issues"
            suggested_action = "Check your internet connection and try again"
        elif RetryableErrorType.SERVICE_UNAVAILABLE in error_types:
            primary_error = "Service temporarily unavailable"
            suggested_action = "The service may be overloaded. Try again in a few minutes"
        elif RetryableErrorType.TIMEOUT in error_types:
            primary_error = "Request timeout"
            suggested_action = "Try with a smaller request or check your connection"
        elif RetryableErrorType.RATE_LIMIT in error_types:
            primary_error = "Rate limit exceeded"
            suggested_action = "Wait a moment before trying again"
        else:
            primary_error = "Service error"
            suggested_action = "Try again later or contact support"

        return MyVoiceError(
            severity=ErrorSeverity.ERROR,
            code="RETRY_EXHAUSTED",
            user_message=f"{primary_error} (failed after {len(attempts)} attempts)",
            technical_details=f"Last error: {str(last_attempt.error)}",
            suggested_action=suggested_action
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            "max_attempts": self.max_attempts,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "jitter": self.jitter,
            "request_timeout": self.request_timeout,
            "total_timeout": self.total_timeout,
            "retryable_status_codes": list(self.retryable_status_codes),
            "permanent_error_codes": list(self.permanent_error_codes)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RetryConfig':
        """Create RetryConfig from dictionary."""
        # Convert tuples back from lists
        if 'retryable_status_codes' in data:
            data['retryable_status_codes'] = tuple(data['retryable_status_codes'])
        if 'permanent_error_codes' in data:
            data['permanent_error_codes'] = tuple(data['permanent_error_codes'])

        return cls(**data)


# Predefined configurations for common scenarios
class RetryConfigs:
    """Predefined retry configurations for common use cases."""

    # Conservative retry for critical operations
    CONSERVATIVE = RetryConfig(
        max_attempts=2,
        initial_delay=2.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True
    )

    # Standard retry for most operations
    STANDARD = RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True
    )

    # Aggressive retry for non-critical operations
    AGGRESSIVE = RetryConfig(
        max_attempts=5,
        initial_delay=0.5,
        max_delay=120.0,
        exponential_base=1.5,
        jitter=True
    )

    # Network-focused retry for connection issues
    NETWORK_RESILIENT = RetryConfig(
        max_attempts=4,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
        retryable_errors=(ConnectionError, Timeout, OSError),
        retryable_status_codes=(502, 503, 504, 429, 408)
    )

    # Fast retry for interactive operations
    INTERACTIVE = RetryConfig(
        max_attempts=2,
        initial_delay=0.5,
        max_delay=5.0,
        exponential_base=2.0,
        jitter=False,
        request_timeout=10.0,
        total_timeout=30.0
    )


def create_retry_callback(logger: logging.Logger) -> Callable[[RetryAttempt], None]:
    """
    Create a retry callback function for logging.

    Args:
        logger: Logger instance to use

    Returns:
        Callable: Callback function for retry attempts
    """
    def retry_callback(attempt: RetryAttempt):
        logger.warning(
            f"Retry attempt {attempt.attempt_number} after {attempt.error_type.value}: "
            f"{str(attempt.error)}, waiting {attempt.delay_seconds:.1f}s"
        )

    return retry_callback


def create_failure_callback(logger: logging.Logger) -> Callable[[List[RetryAttempt]], None]:
    """
    Create a failure callback function for logging.

    Args:
        logger: Logger instance to use

    Returns:
        Callable: Callback function for retry failure
    """
    def failure_callback(attempts: List[RetryAttempt]):
        error_summary = ", ".join([
            f"attempt {i+1}: {attempt.error_type.value}"
            for i, attempt in enumerate(attempts)
        ])
        logger.error(f"All retry attempts failed: {error_summary}")

    return failure_callback