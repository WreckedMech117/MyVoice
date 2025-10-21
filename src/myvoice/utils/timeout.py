"""
Timeout Utility Module

Provides timeout protection for operations that may hang indefinitely,
particularly useful for PyAudio operations that can deadlock when Windows
Audio Service (audiodg.exe) has issues.
"""

import threading
from functools import wraps
from typing import Any, Callable, TypeVar


class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout limit."""
    pass


T = TypeVar('T')


def timeout_decorator(seconds: float = 5.0):
    """
    Decorator to add timeout protection to function calls.

    This decorator uses threading to implement timeouts since signal.alarm()
    is not available on Windows. The decorated function will raise TimeoutError
    if it doesn't complete within the specified time.

    Args:
        seconds: Timeout duration in seconds

    Returns:
        Decorated function that raises TimeoutError on timeout

    Example:
        >>> @timeout_decorator(seconds=5)
        ... def slow_operation():
        ...     time.sleep(10)  # This will timeout after 5 seconds
        ...
        >>> try:
        ...     slow_operation()
        ... except TimeoutError as e:
        ...     print(f"Operation timed out: {e}")

    Note:
        - Uses daemon threads to prevent blocking on exit
        - Cannot be used with non-thread-safe operations
        - The timed-out function will continue running in background
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Container for result or exception
            result = [TimeoutError(f"Operation timed out after {seconds}s")]

            def target():
                """Target function to run in thread."""
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            # Start function in daemon thread
            thread = threading.Thread(target=target, daemon=True)
            thread.start()

            # Wait for completion with timeout
            thread.join(timeout=seconds)

            # Check if thread is still running (timed out)
            if thread.is_alive():
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

            # Check if result is an exception and raise it
            if isinstance(result[0], Exception):
                raise result[0]

            return result[0]

        return wrapper
    return decorator


class AudioTimeoutConfig:
    """Timeout configuration for PyAudio operations."""

    # Device enumeration can be slow with many devices or problematic drivers
    DEVICE_ENUM_TIMEOUT = 10.0  # seconds

    # Stream opening should be fast, but can hang with audiodg.exe issues
    STREAM_OPEN_TIMEOUT = 5.0

    # PyAudio initialization includes device enumeration internally
    PYAUDIO_INIT_TIMEOUT = 15.0

    # Stream write operations should complete quickly
    STREAM_WRITE_TIMEOUT = 2.0

    # Device info queries can hang on problematic devices
    DEVICE_INFO_TIMEOUT = 3.0


def run_with_timeout(func: Callable[..., T], timeout: float, *args, **kwargs) -> T:
    """
    Run a function with timeout protection (non-decorator version).

    Args:
        func: Function to run with timeout
        timeout: Timeout in seconds
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Return value from func

    Raises:
        TimeoutError: If function doesn't complete within timeout

    Example:
        >>> result = run_with_timeout(slow_function, 5.0, arg1, arg2, key=value)
    """
    result = [TimeoutError(f"Operation timed out after {timeout}s")]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            result[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"{func.__name__} timed out after {timeout}s")

    if isinstance(result[0], Exception):
        raise result[0]

    return result[0]
