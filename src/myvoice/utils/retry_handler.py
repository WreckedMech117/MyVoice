"""
Retry Handler Utilities

This module provides retry decorators and handlers for implementing
exponential backoff retry logic with comprehensive error handling.
"""

import time
import asyncio
import logging
from typing import Callable, Any, List, Optional, TypeVar, Union
from functools import wraps

from myvoice.models.retry_config import RetryConfig, RetryAttempt, RetryConfigs
from myvoice.models.error import MyVoiceError

T = TypeVar('T')


class RetryHandler:
    """
    Handler for retry logic with exponential backoff.

    This class implements retry logic that can be used with both
    synchronous and asynchronous functions.
    """

    def __init__(self, config: RetryConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the retry handler.

        Args:
            config: Retry configuration
            logger: Optional logger for retry events
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            T: Function result

        Raises:
            MyVoiceError: If all retry attempts fail
        """
        attempts: List[RetryAttempt] = []
        start_time = time.time()

        for attempt_number in range(1, self.config.max_attempts + 1):
            try:
                # Check total timeout
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.config.total_timeout:
                    self.logger.error(f"Total timeout ({self.config.total_timeout}s) exceeded")
                    break

                # Execute the function
                self.logger.debug(f"Executing function, attempt {attempt_number}/{self.config.max_attempts}")
                result = func(*args, **kwargs)

                # Success - log if there were previous attempts
                if attempts:
                    self.logger.info(f"Operation succeeded on attempt {attempt_number} after {len(attempts)} failures")

                return result

            except Exception as e:
                self.logger.debug(f"Attempt {attempt_number} failed: {str(e)}")

                # Check if error is retryable
                if not self.config.is_retryable_error(e):
                    self.logger.error(f"Non-retryable error: {str(e)}")
                    raise MyVoiceError.from_exception(e, "Operation failed with non-retryable error")

                # Calculate delay for next attempt
                delay = self.config.calculate_delay(attempt_number)

                # Create retry attempt record
                attempt = RetryAttempt(
                    attempt_number=attempt_number,
                    error=e,
                    delay_seconds=delay
                )
                attempts.append(attempt)

                # Call retry callback if configured
                if self.config.on_retry_callback:
                    try:
                        self.config.on_retry_callback(attempt)
                    except Exception as callback_error:
                        self.logger.warning(f"Retry callback failed: {callback_error}")

                # Check if this was the last attempt
                if attempt_number >= self.config.max_attempts:
                    break

                # Check if delay would exceed total timeout
                if time.time() - start_time + delay >= self.config.total_timeout:
                    self.logger.warning("Next retry would exceed total timeout, aborting")
                    break

                # Wait before retry
                if delay > 0:
                    self.logger.info(self.config.get_user_friendly_message(attempt, len(attempts)))
                    time.sleep(delay)

        # All attempts failed
        self.logger.error(f"All {len(attempts)} retry attempts failed")

        # Call failure callback if configured
        if self.config.on_failure_callback:
            try:
                self.config.on_failure_callback(attempts)
            except Exception as callback_error:
                self.logger.warning(f"Failure callback failed: {callback_error}")

        # Create comprehensive error
        retry_error = self.config.create_retry_error(attempts)
        raise retry_error

    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute an async function with retry logic.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            T: Function result

        Raises:
            MyVoiceError: If all retry attempts fail
        """
        attempts: List[RetryAttempt] = []
        start_time = time.time()

        for attempt_number in range(1, self.config.max_attempts + 1):
            try:
                # Check total timeout
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.config.total_timeout:
                    self.logger.error(f"Total timeout ({self.config.total_timeout}s) exceeded")
                    break

                # Execute the async function
                self.logger.debug(f"Executing async function, attempt {attempt_number}/{self.config.max_attempts}")
                result = await func(*args, **kwargs)

                # Success - log if there were previous attempts
                if attempts:
                    self.logger.info(f"Async operation succeeded on attempt {attempt_number} after {len(attempts)} failures")

                return result

            except Exception as e:
                self.logger.debug(f"Async attempt {attempt_number} failed: {str(e)}")

                # Check if error is retryable
                if not self.config.is_retryable_error(e):
                    self.logger.error(f"Non-retryable async error: {str(e)}")
                    raise MyVoiceError.from_exception(e, "Async operation failed with non-retryable error")

                # Calculate delay for next attempt
                delay = self.config.calculate_delay(attempt_number)

                # Create retry attempt record
                attempt = RetryAttempt(
                    attempt_number=attempt_number,
                    error=e,
                    delay_seconds=delay
                )
                attempts.append(attempt)

                # Call retry callback if configured
                if self.config.on_retry_callback:
                    try:
                        self.config.on_retry_callback(attempt)
                    except Exception as callback_error:
                        self.logger.warning(f"Async retry callback failed: {callback_error}")

                # Check if this was the last attempt
                if attempt_number >= self.config.max_attempts:
                    break

                # Check if delay would exceed total timeout
                if time.time() - start_time + delay >= self.config.total_timeout:
                    self.logger.warning("Next async retry would exceed total timeout, aborting")
                    break

                # Wait before retry (async)
                if delay > 0:
                    self.logger.info(self.config.get_user_friendly_message(attempt, len(attempts)))
                    await asyncio.sleep(delay)

        # All attempts failed
        self.logger.error(f"All {len(attempts)} async retry attempts failed")

        # Call failure callback if configured
        if self.config.on_failure_callback:
            try:
                self.config.on_failure_callback(attempts)
            except Exception as callback_error:
                self.logger.warning(f"Async failure callback failed: {callback_error}")

        # Create comprehensive error
        retry_error = self.config.create_retry_error(attempts)
        raise retry_error


def with_retry(config: Optional[RetryConfig] = None, logger: Optional[logging.Logger] = None):
    """
    Decorator for adding retry logic to synchronous functions.

    Args:
        config: Retry configuration (defaults to STANDARD)
        logger: Optional logger for retry events

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry(RetryConfigs.AGGRESSIVE)
        def unreliable_function():
            # Function that might fail
            pass
    """
    if config is None:
        config = RetryConfigs.STANDARD

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = RetryHandler(config, logger)
            return handler.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def with_async_retry(config: Optional[RetryConfig] = None, logger: Optional[logging.Logger] = None):
    """
    Decorator for adding retry logic to asynchronous functions.

    Args:
        config: Retry configuration (defaults to STANDARD)
        logger: Optional logger for retry events

    Returns:
        Decorated async function with retry logic

    Example:
        @with_async_retry(RetryConfigs.NETWORK_RESILIENT)
        async def unreliable_async_function():
            # Async function that might fail
            pass
    """
    if config is None:
        config = RetryConfigs.STANDARD

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            handler = RetryHandler(config, logger)
            return await handler.execute_async(func, *args, **kwargs)
        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retry operations with custom handling.

    This provides a way to execute operations with retry logic
    and access attempt information.
    """

    def __init__(self, config: RetryConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize retry context.

        Args:
            config: Retry configuration
            logger: Optional logger for retry events
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.handler = RetryHandler(config, logger)
        self.attempts: List[RetryAttempt] = []

    def __enter__(self):
        """Enter retry context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit retry context."""
        # Context manager doesn't handle exceptions automatically
        # Use execute() or execute_async() methods instead
        pass

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        return self.handler.execute(func, *args, **kwargs)

    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with retry logic."""
        return await self.handler.execute_async(func, *args, **kwargs)

    def get_attempts(self) -> List[RetryAttempt]:
        """Get list of retry attempts made."""
        return self.attempts.copy()


# Helper functions for common retry patterns
def retry_on_network_error(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Execute function with network-resilient retry logic.

    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        T: Function result
    """
    handler = RetryHandler(RetryConfigs.NETWORK_RESILIENT)
    return handler.execute(func, *args, **kwargs)


async def async_retry_on_network_error(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Execute async function with network-resilient retry logic.

    Args:
        func: Async function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        T: Function result
    """
    handler = RetryHandler(RetryConfigs.NETWORK_RESILIENT)
    return await handler.execute_async(func, *args, **kwargs)


def create_retry_handler(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    logger: Optional[logging.Logger] = None
) -> RetryHandler:
    """
    Create a retry handler with custom configuration.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        logger: Optional logger for retry events

    Returns:
        RetryHandler: Configured retry handler
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter
    )
    return RetryHandler(config, logger)